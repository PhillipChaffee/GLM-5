"""DFlash2 zero-training acceptance smoke on Modal: three GLM-5.3-Flash serving arms.

Arms on one 4xB200 node (map ticket #12): the released DFLASH drafter
(incoai/GLM-5.3-Flash-DFlash2, block 8), native MTP (EAGLE 5/1/6), and
spec-off. Each arm boots SGLang pinned to SGLANG_IMAGE_REF, runs its
measurement windows (temp-0 capture, tau suite, concurrency ladder), and
commits one result json to the volume after every window so a mid-run crash
keeps partial data. Arms run sequentially: the Modal plan's GPU concurrency
cap silently queues parallel multi-GPU calls.
"""

from __future__ import annotations

import http.client
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

VOL = "/vol"
VOLUME_NAME = "glm-serve-weights"
DATA_DIR = "/work/data"
LOCAL_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = LOCAL_DIR / "results"
TARGET = "zai-org/GLM-5.3-Flash"
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"
SGLANG_IMAGE_REF = "lmsysorg/sglang:v0.5.21"
SGLANG_TAG_SHA = "e00930c5489053f26d86b179cee0d087f846acbb"
DFLASH_CAPTURE_PR = 36708
GPU_SPEC = "B200+:4"  # B300 fallback at B200 rates (Modal community rec, 2026-10-06)
TP_SIZE = 4
PORT = 30000
BASE_URL = f"http://127.0.0.1:{PORT}"
METRIC_VERIFY_CALLS = "sglang:spec_verify_calls_total"
METRIC_GEN_TOKENS = "sglang:generation_tokens_total"
TEMP = 1.0
TEMP0_MAX_TOKENS = 512
SUITE_MAX_TOKENS = 1024
LADDER_MAX_TOKENS = 256
REQUEST_TIMEOUT_S = 900.0
HEALTH_DEADLINE_S = 45 * 60
POLL_S = 5.0
HTTP_OK = 200
MAX_LISTED_FILES = 40  # inspect lists at most this many files per directory
DFLASH_EXPECTED_BLOCK = 8
MTP_STEPS = 5
MTP_TOPK = 1
MTP_DRAFT_TOKENS = 6
INFO_KEYS = (
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_dflash_block_size",
    "speculative_num_draft_tokens",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_draft_attention_backend",
    "speculative_dsa_topk_backend",
    "attention_backend",
    "dsa_prefill_backend",
    "dsa_decode_backend",
    "kv_cache_dtype",
    "moe_runner_backend",
    "tp_size",
    "mem_fraction_static",
    "max_running_requests",
    "chunked_prefill_size",
)
ARM_TEMP0: dict[str, bool] = {"dflash": True, "mtp": False, "plain": True}
ARM_TASKS: dict[str, tuple[str, ...]] = {
    "dflash": ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench", "alpaca"),
    "mtp": ("mt-bench", "alpaca"),
    "plain": (),
}
ARM_RUNGS: dict[str, tuple[int, ...]] = {
    "dflash": (1, 8, 16, 32, 64),
    "mtp": (1, 16),
    "plain": (1, 16),
}

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

util_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "hf_xet")
    .add_local_dir(LOCAL_DIR / "data", DATA_DIR)
)
serve_image = modal.Image.from_registry(SGLANG_IMAGE_REF).add_local_dir(
    LOCAL_DIR / "data", DATA_DIR
)

app = modal.App("dflash2-smoke")


def _log(message: str) -> None:
    """Print a line immediately (the print-free lint gate owns this shape)."""
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def _server_log_path(tag: str) -> Path:
    """Return the volume path for one arm's server boot log."""
    return Path(VOL) / "results" / f"boot_{tag}.log"


def server_command(arm: str) -> list[str]:
    """Build the SGLang launch command for one arm."""
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        f"{VOL}/weights/target",
        "--tp-size",
        str(TP_SIZE),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--enable-metrics",
        "--moe-runner-backend",
        "flashinfer_trtllm",
    ]
    if arm == "dflash":
        cmd += [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            f"{VOL}/weights/drafter",
            "--speculative-draft-attention-backend",
            "fa4",
        ]
    elif arm == "mtp":
        cmd += [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            str(MTP_STEPS),
            "--speculative-eagle-topk",
            str(MTP_TOPK),
            "--speculative-num-draft-tokens",
            str(MTP_DRAFT_TOKENS),
        ]
    elif arm != "plain":
        msg = f"unknown arm: {arm}"
        raise RuntimeError(msg)
    return cmd


def _request(method: str, path: str, body: bytes | None, timeout: float) -> tuple[int, str]:
    """Send one HTTP request to the local server and return (status, body)."""
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


def wait_health(server: subprocess.Popen[str]) -> float:
    """Poll /health until the server answers; return the boot wall time."""
    start = time.time()
    while True:
        if server.poll() is not None:
            msg = f"server exited during startup after {time.time() - start:.0f}s"
            raise RuntimeError(msg)
        try:
            status, _ = _request("GET", "/health", None, 5.0)
            if status == HTTP_OK:
                break
        except OSError as exc:
            if time.time() - start > HEALTH_DEADLINE_S:
                msg = f"server not healthy within {HEALTH_DEADLINE_S}s"
                raise RuntimeError(msg) from exc
            time.sleep(POLL_S)
    return time.time() - start


def _sum_metric(snapshot: str, name: str) -> float:
    """Sum one prometheus counter across every labeled series in a snapshot."""
    pattern = re.compile(
        r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
        re.MULTILINE,
    )
    total = 0.0
    for match in pattern.finditer(snapshot):
        value = match.group(1)
        if value not in {"NaN", "+Inf", "-Inf"}:
            total += float(value)
    return total


def counter_snapshot() -> dict[str, float]:
    """Read /metrics and return the counters this run aggregates."""
    _, body = _request("GET", "/metrics", None, 30.0)
    return {
        METRIC_VERIFY_CALLS: _sum_metric(body, METRIC_VERIFY_CALLS),
        METRIC_GEN_TOKENS: _sum_metric(body, METRIC_GEN_TOKENS),
    }


def boot_report(arm: str) -> dict[str, Any]:
    """Fetch resolved server args and fail loudly on a wrong arm config."""
    status, body = _request("GET", "/get_server_info", None, 30.0)
    if status != HTTP_OK:
        msg = f"get_server_info unavailable (http {status})"
        raise RuntimeError(msg)
    info = json.loads(body)
    args: dict[str, Any] = info.get("server_args", info)
    report = {key: args.get(key) for key in INFO_KEYS}
    algorithm = report["speculative_algorithm"]
    if arm == "dflash":
        draft_tokens = report["speculative_num_draft_tokens"]
        if algorithm != "DFLASH" or draft_tokens != DFLASH_EXPECTED_BLOCK:
            msg = f"dflash boot mismatch: algorithm={algorithm} block={draft_tokens}"
            raise RuntimeError(msg)
    elif arm == "mtp":
        steps = report["speculative_num_steps"]
        topk = report["speculative_eagle_topk"]
        draft_tokens = report["speculative_num_draft_tokens"]
        if algorithm != "EAGLE" or (steps, topk, draft_tokens) != (
            MTP_STEPS,
            MTP_TOPK,
            MTP_DRAFT_TOKENS,
        ):
            msg = f"mtp boot mismatch: alg={algorithm} {steps}/{topk}/{draft_tokens}"
            raise RuntimeError(msg)
    elif algorithm not in {None, "NONE"}:
        msg = f"plain arm booted with spec decoding: {algorithm}"
        raise RuntimeError(msg)
    return report


def completion_body(
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    spec_details: bool = False,
    logprobs: int | None = None,
) -> dict[str, Any]:
    """Build one /v1/completions request body for a rendered prompt."""
    body: dict[str, Any] = {
        "model": TARGET,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    if spec_details:
        body["return_spec_tokens_details"] = True
    if logprobs is not None:
        body["logprobs"] = logprobs
    return body


def one_request(prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    """Run one completion and flatten the response into a result row."""
    started = time.time()
    status, body = _request(
        "POST",
        "/v1/completions",
        json.dumps(completion_body(prompt, max_tokens, temperature, spec_details=True)).encode(),
        REQUEST_TIMEOUT_S,
    )
    if status != HTTP_OK:
        msg = f"completion failed (http {status}): {body[:300]}"
        raise RuntimeError(msg)
    response: dict[str, Any] = json.loads(body)
    usage = response.get("usage", {})
    details = usage.get("spec_tokens_details") or {}
    choice = (response.get("choices") or [{}])[0]
    return {
        "latency": time.time() - started,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "spec_accept_length": details.get("spec_accept_length"),
        "text": choice.get("text"),
    }


def run_batch(
    prompts: list[str], concurrency: int, max_tokens: int, temperature: float
) -> tuple[list[dict[str, Any]], float]:
    """Run prompts at fixed concurrency; return rows plus the window wall time."""
    started = time.time()
    if concurrency <= 1:
        rows = [one_request(prompt, max_tokens, temperature) for prompt in prompts]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(
                pool.map(
                    lambda prompt: one_request(prompt, max_tokens, temperature),
                    prompts,
                )
            )
    return rows, time.time() - started


def _throughput_fields(rows: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    """Compute tokens, aggregate tok/s, per-user tok/s, and median latency."""
    tokens = sum(int(row["completion_tokens"] or 0) for row in rows)
    latencies = sorted(float(row["latency"]) for row in rows)
    mid = len(latencies) // 2
    per_user = [
        int(row["completion_tokens"] or 0) / float(row["latency"])
        for row in rows
        if float(row["latency"]) > 0
    ]
    return {
        "tokens": tokens,
        "aggregate_tok_s": tokens / wall if wall > 0 else None,
        "per_user_tok_s_mean": sum(per_user) / len(per_user) if per_user else None,
        "p50_latency_s": latencies[mid] if latencies else 0.0,
    }


def _accept_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute tau from per-request acceptance; empty when spec is off."""
    accepts = [
        (int(row["completion_tokens"] or 0), float(row["spec_accept_length"] or 0)) for row in rows
    ]
    accepts = [(tokens_row, acc) for tokens_row, acc in accepts if acc > 0]
    if not accepts:
        return {}
    weighted_rounds = sum(tokens_row / acc for tokens_row, acc in accepts)
    return {
        "tau_req_mean": sum(acc for _, acc in accepts) / len(accepts),
        "tau_token_weighted": (
            sum(tokens_row for tokens_row, _ in accepts) / weighted_rounds
            if weighted_rounds > 0
            else None
        ),
    }


def summarize(rows: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    """Compute the per-window summary fields every measurement reports."""
    summary: dict[str, Any] = {
        "n": len(rows),
        "wall_s": wall,
        **_throughput_fields(rows, wall),
        **_accept_fields(rows),
    }
    return summary


def flush_cache() -> None:
    """Flush the radix cache so every ladder rung pays cold prefill."""
    _request("POST", "/flush_cache", b"{}", 120.0)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one suite jsonl from the baked-in data directory."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_temp0() -> dict[str, Any]:
    """Capture temp-0 outputs on the fixed check set for the losslessness diff."""
    rows = [
        (row["id"], row["prompt"], int(row["max_tokens"]))
        for row in _load_jsonl(Path(DATA_DIR) / "temp0_prompts.jsonl")
    ]
    requests: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = [
            (row_id, pool.submit(one_request, prompt, max_tokens, 0.0))
            for row_id, prompt, max_tokens in rows
        ]
        for row_id, future in futures:
            requests.append({"id": row_id, **future.result()})
    wall = time.time() - started
    _log(f"temp0 done: {len(requests)} prompts in {wall:.1f}s")
    return {"window": "temp0", "n": len(requests), "wall_s": wall, "requests": requests}


def run_suite_task(task: str) -> dict[str, Any]:
    """Measure tau on one task at protocol settings (c1, temp 1.0, 1024 tokens)."""
    prompts = [row["prompt"] for row in _load_jsonl(Path(DATA_DIR) / f"suite_{task}.jsonl")]
    before = counter_snapshot()
    rows, wall = run_batch(prompts, 1, SUITE_MAX_TOKENS, TEMP)
    after = counter_snapshot()
    row: dict[str, Any] = {
        "window": f"suite:{task}",
        "concurrency": 1,
        "temperature": TEMP,
    }
    row.update(summarize(rows, wall))
    rounds = after[METRIC_VERIFY_CALLS] - before[METRIC_VERIFY_CALLS]
    tokens = after[METRIC_GEN_TOKENS] - before[METRIC_GEN_TOKENS]
    row["rounds_counter_delta"] = rounds
    if rounds > 0:
        row["tau_counter"] = tokens / rounds
    _log(f"suite {task}: n={row['n']} tau={row.get('tau_req_mean')} wall={wall:.1f}s")
    return row


def run_ladder_rung(rung: int) -> dict[str, Any]:
    """Measure one ladder rung on the fixed alpaca workload at temp 0."""
    prompts = [row["prompt"] for row in _load_jsonl(Path(DATA_DIR) / "ladder_alpaca.jsonl")]
    flush_cache()
    before = counter_snapshot()
    rows, wall = run_batch(prompts, rung, LADDER_MAX_TOKENS, 0.0)
    after = counter_snapshot()
    row: dict[str, Any] = {
        "window": f"ladder:c{rung}",
        "concurrency": rung,
        "temperature": 0.0,
    }
    row.update(summarize(rows, wall))
    rounds = after[METRIC_VERIFY_CALLS] - before[METRIC_VERIFY_CALLS]
    tokens = after[METRIC_GEN_TOKENS] - before[METRIC_GEN_TOKENS]
    row["rounds_counter_delta"] = rounds
    if rounds > 0:
        row["tau_counter"] = tokens / rounds
    _log(
        f"ladder c{rung}: agg={row['aggregate_tok_s']:.1f} tok/s "
        f"per-user={row['per_user_tok_s_mean']:.1f} tau={row.get('tau_counter')}"
    )
    return row


def save_result(tag: str, payload: dict[str, Any]) -> None:
    """Write the arm's result json to the volume and commit it."""
    results = Path(VOL) / "results"
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{tag}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    volume.commit()
    _log(f"committed {path}")


def stop_server(server: subprocess.Popen[str]) -> None:
    """Terminate the server subprocess cleanly, killing it if it hangs."""
    server.terminate()
    try:
        server.wait(timeout=180)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=60)


def gpu_names() -> list[str]:
    """Report the GPU model names actually scheduled for this container."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def measure(
    arm: str,
    tag: str,
    *,
    tasks: str = "",
    rungs: str = "",
    temp0: bool | None = None,
) -> None:
    """Boot one arm, run its windows, and commit results after each one."""
    Path(VOL, "results").mkdir(parents=True, exist_ok=True)
    log_handle = _server_log_path(tag).open("w")
    command = server_command(arm)
    _log(f"booting arm={arm}: {' '.join(command)}")
    started = time.time()
    server = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    try:
        boot_wall = wait_health(server)
        _log(f"healthy after {boot_wall:.0f}s")
        payload: dict[str, Any] = {
            "arm": arm,
            "tag": tag,
            "sglang_image": SGLANG_IMAGE_REF,
            "sglang_tag_sha": SGLANG_TAG_SHA,
            "dflash_capture_pr": DFLASH_CAPTURE_PR,
            "gpu": GPU_SPEC,
            "gpu_names": gpu_names(),
            "tp_size": TP_SIZE,
            "boot_wall_s": boot_wall,
            "boot_info": boot_report(arm),
        }
        save_result(tag, payload)
        with_temp0 = ARM_TEMP0[arm] if temp0 is None else temp0
        if with_temp0:
            payload["temp0"] = run_temp0()
            save_result(tag, payload)
        task_names = tuple(item for item in tasks.split(",") if item) if tasks else ARM_TASKS[arm]
        payload["suite"] = {"rows": [run_suite_task(task) for task in task_names]}
        save_result(tag, payload)
        rung_list = (
            tuple(int(item) for item in rungs.split(",") if item) if rungs else ARM_RUNGS[arm]
        )
        payload["ladder"] = {"rows": [run_ladder_rung(rung) for rung in rung_list]}
        payload["total_wall_s"] = time.time() - started
        save_result(tag, payload)
        _log(f"arm {arm} complete in {(time.time() - started) / 60:.1f} min")
    finally:
        stop_server(server)
        log_handle.close()


def probe_logprobs(arm: str, tag: str, ids: str) -> None:
    """Rerun selected temp-0 prompts with top-5 logprobs on one arm."""
    wanted = {item for item in ids.split(",") if item}
    rows = [
        row for row in _load_jsonl(Path(DATA_DIR) / "temp0_prompts.jsonl") if row["id"] in wanted
    ]
    if not rows:
        msg = f"no temp0 prompts matched ids: {ids}"
        raise RuntimeError(msg)
    log_handle = _server_log_path(tag).open("w")
    command = server_command(arm)
    _log(f"probe arm={arm}: {' '.join(command)}")
    server = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    try:
        wait_health(server)
        out: dict[str, Any] = {
            "arm": arm,
            "tag": tag,
            "boot_info": boot_report(arm),
            "requests": [],
        }
        for row in rows:
            status, body = _request(
                "POST",
                "/v1/completions",
                json.dumps(
                    completion_body(row["prompt"], int(row["max_tokens"]), 0.0, logprobs=5)
                ).encode(),
                REQUEST_TIMEOUT_S,
            )
            if status != HTTP_OK:
                msg = f"probe completion failed (http {status}): {body[:300]}"
                raise RuntimeError(msg)
            response: dict[str, Any] = json.loads(body)
            choice = (response.get("choices") or [{}])[0]
            logprobs_out = choice.get("logprobs") or {}
            out["requests"].append(
                {
                    "id": row["id"],
                    "text": choice.get("text"),
                    "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
                    "tokens": logprobs_out.get("tokens"),
                    "top_logprobs": logprobs_out.get("top_logprobs"),
                }
            )
            _log(f"probe {row['id']}: {out['requests'][-1]['completion_tokens']} tokens")
        save_result(tag, out)
    finally:
        stop_server(server)
        log_handle.close()


@app.function(
    image=serve_image,
    gpu=GPU_SPEC,
    volumes={VOL: volume},
    secrets=[hf_secret],
    timeout=6 * 3600,
    env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
)
def serve_and_measure(
    arm: str,
    tag: str,
    *,
    tasks: str = "",
    rungs: str = "",
    temp0: bool | None = None,
) -> None:
    """Run one arm's measurement on the 4xB200 node."""
    measure(arm, tag, tasks=tasks, rungs=rungs, temp0=temp0)


@app.function(
    image=serve_image,
    gpu=GPU_SPEC,
    volumes={VOL: volume},
    secrets=[hf_secret],
    timeout=2 * 3600,
    env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
)
def serve_and_probe(arm: str, tag: str, ids: str) -> None:
    """Run one arm's logprobs probe on the 4xB200 node."""
    probe_logprobs(arm, tag, ids)


@app.function(image=util_image, volumes={VOL: volume}, secrets=[hf_secret], timeout=4 * 3600)
def stage_weights() -> None:
    """Pre-stage target FP8 and drafter weights to the shared volume once."""
    from huggingface_hub import snapshot_download

    volume.reload()
    target_dir = Path(VOL) / "weights" / "target"
    drafter_dir = Path(VOL) / "weights" / "drafter"
    if (target_dir / "config.json").exists() and (drafter_dir / "model.safetensors").exists():
        _log("weights already staged; skipping")
        return
    for repo, out_dir in ((TARGET, target_dir), (DRAFTER, drafter_dir)):
        _log(f"downloading {repo} -> {out_dir}")
        snapshot_download(repo, local_dir=out_dir)
        total = sum(item.stat().st_size for item in out_dir.rglob("*") if item.is_file())
        _log(f"staged {repo}: {total / 1e9:.1f} GB")
    volume.commit()


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def inspect() -> None:
    """Print the volume layout so dependent stages can be verified first."""
    volume.reload()
    for root in ("weights", "results"):
        path = Path(VOL) / root
        if not path.exists():
            _log(f"{path}: missing")
            continue
        entries = sorted(item for item in path.rglob("*") if item.is_file())
        total = sum(item.stat().st_size for item in entries)
        _log(f"{path}: {total / 1e9:.2f} GB across {len(entries)} files")
        for item in entries[:MAX_LISTED_FILES]:
            _log(f"  {item.relative_to(VOL)} {item.stat().st_size / 1e9:.3f} GB")


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def fetch_result(name: str) -> bytes:
    """Read one result json from the volume."""
    volume.reload()
    return (Path(VOL) / "results" / f"{name}.json").read_bytes()


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def peek_result(name: str) -> bytes | None:
    """Read one result json if present, else None (progress-safe polling)."""
    volume.reload()
    path = Path(VOL) / "results" / f"{name}.json"
    if not path.exists():
        return None
    return path.read_bytes()


@app.local_entrypoint()
def main(
    stage: str = "stage-weights",
    *,
    arm: str = "dflash",
    tag: str = "",
    tasks: str = "",
    rungs: str = "",
    temp0: bool | None = None,
    fetch_dir: str = "",
    call_id: str = "",
    ids: str = "",
) -> None:
    """Dispatch one stage; spawn-measure detaches the GPU call from this CLI."""
    dispatch: dict[str, tuple[Any, dict[str, Any]]] = {
        "stage-weights": (stage_weights.remote, {}),
        "inspect": (inspect.remote, {}),
        "spawn-measure": (
            serve_and_measure.spawn,
            {
                "arm": arm,
                "tag": tag or arm,
                "tasks": tasks,
                "rungs": rungs,
                "temp0": temp0,
            },
        ),
        "measure": (
            serve_and_measure.remote,
            {
                "arm": arm,
                "tag": tag or arm,
                "tasks": tasks,
                "rungs": rungs,
                "temp0": temp0,
            },
        ),
        "wait": (None, {"call_id": call_id}),
        "poll": (None, {"call_id": call_id}),
        "fetch": (fetch_result.remote, {"name": tag or arm}),
        "peek": (peek_result.remote, {"name": tag or arm}),
        "probe": (serve_and_probe.remote, {"arm": arm, "tag": tag or f"probe-{arm}", "ids": ids}),
    }
    if stage in {"wait", "poll"}:
        call = modal.FunctionCall.from_id(call_id)
        call.wait()
        _log(f"call {call_id}: finished")
        return
    fn, kwargs = dispatch[stage]
    result = fn(**kwargs)
    if stage in {"fetch", "peek"}:
        if result is None:
            _log(f"{kwargs['name']}: no result yet (still booting or queued)")
            return
        out_dir = Path(fetch_dir) if fetch_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{kwargs['name']}.json"
        out_path.write_bytes(result)
        present = sorted(json.loads(result.decode()).keys()) if stage == "peek" else []
        _log(f"fetched -> {out_path}" + (f" (windows: {present})" if present else ""))
    elif stage == "spawn-measure":
        _log(f"spawned arm measure; call_id={result.object_id}")
