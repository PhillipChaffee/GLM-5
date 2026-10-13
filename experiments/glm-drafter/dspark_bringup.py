"""Bring the DeepSpec GLM-5.3-Flash drafter up in sglang DSPARK on Modal (#47).

Prepares a serve copy of the step_20 smoke checkpoint (config edits per the
compat report A.3), boots SGLang TP4 with the DSPARK worker and the two
in-place sglang patches (post-final-norm capture on glm5_next; the
Glm5DsparkModel drafter class), sends a small prompt set, and records the
resolved spec config, acceptance counters, and sample outputs. This is the
eval-path de-risk the pre-train review (#46) needs; tau ranking and the
capture-parity smoke belong to #52, so the tau here is a liveness signal on
an undertrained checkpoint, never a quality read.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import py_compile
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

_LIB_CANDIDATES = (
    Path(__file__).resolve().parent.parent,  # local: experiments/ holds lib/
    Path(__file__).resolve().parent,  # Modal runtime: lib/ is mounted beside the app
)
_EXPERIMENTS_DIR = next(
    candidate for candidate in _LIB_CANDIDATES if (candidate / "lib" / "modal_pipeline.py").exists()
)
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from lib.modal_pipeline import fetch_result_bytes
from lib.modal_pipeline import log as _log

VOL = "/vol"
RUN_VOLUME_NAME = "glm-drafter-run"
RUN_VOL = "/vol-run"
SERVE_VOLUME_NAME = "glm-serve-weights"
CKPT_REL = "home/checkpoints/deepspec/glm_drafter_block7_glm53flash/step_20"
DRAFTER_SERVE_DIR = f"{VOL}/weights/drafter-glm-step20"
TARGET_DIR = f"{VOL}/weights/target"
PATCH_DIR = "/work/patch"
DATA_DIR = "/work/data"
LOCAL_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = LOCAL_DIR / "results"
SGLANG_IMAGE_REF = "lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48"
GPU_SPEC = "B200+:4"  # B300 fallback at B200 rates (modal-runs rule; #12 precedent)
TP_SIZE = 4
PORT = 30000
METRIC_VERIFY_CALLS = "sglang:spec_verify_calls_total"
METRIC_GEN_TOKENS = "sglang:generation_tokens_total"
REQUEST_TIMEOUT_S = 900.0
HEALTH_DEADLINE_S = 45 * 60
POLL_S = 5.0
HTTP_OK = 200
GREEDY_PROMPTS = 8
SAMPLED_PROMPTS = 4
SMOKE_MAX_TOKENS = 256
# Contract keys the serve checkpoint must carry before sglang ever sees it.
EXPECT_KEYS: dict[str, Any] = {
    "target_layer_ids": [1, 12, 23, 34, 42, 43, 44],
    "block_size": 7,
    "mask_token_id": 154821,
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "num_hidden_layers": 5,
    "num_target_layers": 45,
}
INFO_KEYS = (
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_dspark_block_size",
    "speculative_num_draft_tokens",
    "speculative_draft_attention_backend",
    "attention_backend",
    "kv_cache_dtype",
    "moe_runner_backend",
    "tp_size",
    "max_running_requests",
)

# The glm5_next capture patch (issue #47, per #33's resolution): exact-string
# replacements against the installed file, whose snippets pin the image's
# source to the analyzed serving SHA (961bcf48). Pairs must each match once.
_INIT_PAIR = (
    """        self.layers_to_capture = []
        self.dflash_capture = False""",
    """        self.layers_to_capture = []
        self.dflash_capture = False
        self.capture_final_norm_output = False""",
)
_SETTER_PAIR = (
    """        self.capture_aux_hidden_states = True
        self.model.dflash_capture = True
        # Capturing before layer k + 1 gives the completed output of layer k.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]""",
    """        self.capture_aux_hidden_states = True
        self.model.dflash_capture = True
        # Capturing before layer k + 1 gives the completed output of layer k.
        # The final layer has no such capture point (there is no layer N), so
        # a requested id N-1 cannot fire here. DeepSpec GLM-port drafters
        # train that id on the post-final-norm state (the LM-head input), so
        # capture it from final_norm's return value instead; it appends after
        # all loop captures, matching the checkpoint's feature order because
        # the final id must be the maximum of target_layer_ids. Mirrors
        # sglang PR #32334 (gpt-oss final-boundary capture) but delivers the
        # norm's OUTPUT, per the DeepSpec GLM training convention (the
        # Kimi-K3 seam delivers the norm's INPUT and would silently mismatch
        # those checkpoints).
        num_layers = self.config.num_hidden_layers
        self.model.capture_final_norm_output = (num_layers - 1) in layer_ids
        if self.model.capture_final_norm_output and max(layer_ids) != num_layers - 1:
            raise ValueError(
                "DFlash/DSPARK capture: the final-layer id must be the maximum "
                "of target_layer_ids for the post-final-norm capture to land "
                "in feature order; got "
                f"{layer_ids!r} with num_hidden_layers={num_layers}."
            )
        self.model.layers_to_capture = [
            val + 1 for val in layer_ids if val != num_layers - 1
        ]""",
)
_FORWARD_PAIR = (
    """            if not forward_batch.forward_mode.is_idle():
                hidden_states = residual_batch.final_norm(
                    hidden_states, forward_batch, self.norm
                )""",
    """            if not forward_batch.forward_mode.is_idle():
                hidden_states = residual_batch.final_norm(
                    hidden_states, forward_batch, self.norm
                )
                if self.capture_final_norm_output:
                    aux_hidden_states.capture(hidden_states)""",
)
GLM5_NEXT_PATCH = (_INIT_PAIR, _SETTER_PAIR, _FORWARD_PAIR)

# The serve config keeps only the keys the draft path reads (compat report
# A.2/A.3): the checkpoint config is a deepcopy of the GLM-5.3-Flash text
# config, and its inherited hybrid fields (linear_attn_config, mhc/hc_*,
# index_* DSA keys, MoE fields, num_nextn_predict_layers) make sglang treat
# the dense 5-layer drafter as a hybrid target — the serving failure that
# the clean-config edit fixes. Mirrors the shipped DFlash2 config's shape.
SERVE_CONFIG_KEEP: dict[str, Any] = {
    "model_type": "qwen3",  # DFlash2 precedent
    "architectures": ["Glm5DsparkModel"],
    "head_dim": 64,  # inherited 0 hard-crashes sglang's attention
    "use_cache": False,
    "dtype": "bfloat16",
    "sample_from_anchor": True,
    "target_final_hidden_normed": True,  # id 44 = post-final-norm (#33)
    "rope_theta": 1000000.0,  # never applied (NoPE class); feeds the parent's rope build
    # max_position_embeddings must match the target's context: sglang derives
    # the server context from the draft config too and hard-errors when the
    # draft's value is smaller.
}
SERVE_CONFIG_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "hidden_act",
    "attention_bias",
    "attention_dropout",
    "vocab_size",
    "tie_word_embeddings",
    "block_size",
    "num_target_layers",
    "target_layer_ids",
    "mask_token_id",
    "markov_rank",
    "markov_head_type",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "layer_types",
    "mlp_layer_types",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "initializer_range",
    "num_anchors",
    "max_position_embeddings",
)

run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True)
serve_volume = modal.Volume.from_name(SERVE_VOLUME_NAME, create_if_missing=True)

util_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "hf_xet")
    .add_local_dir(_EXPERIMENTS_DIR / "lib", "/root/lib")
)
serve_image = (
    modal.Image.from_registry(SGLANG_IMAGE_REF)
    .add_local_dir(_EXPERIMENTS_DIR / "lib", "/root/lib")
    .add_local_dir(LOCAL_DIR / "sglang-patch", PATCH_DIR)
    .add_local_dir(Path(__file__).resolve().parent.parent / "dflash2-smoke" / "data", DATA_DIR)
)

app = modal.App("glm-dspark-bringup")


def _request(method: str, path: str, body: bytes | None, timeout: float) -> tuple[int, str]:
    """Send one HTTP request to the local server and return (status, body)."""
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", errors="replace")
    finally:
        conn.close()


def wait_health(server: subprocess.Popen[Any]) -> float:
    """Poll /health until the server answers; return the boot wall time."""
    started = time.time()
    while time.time() - started < HEALTH_DEADLINE_S:
        if server.poll() is not None:
            msg = f"server exited rc={server.returncode} before health"
            raise RuntimeError(msg)
        try:
            status, _ = _request("GET", "/health", None, 5.0)
            if status == HTTP_OK:
                return time.time() - started
        except OSError:
            pass
        time.sleep(POLL_S)
    msg = f"server not healthy within {HEALTH_DEADLINE_S}s"
    raise RuntimeError(msg)


_METRIC_LINE = re.compile(r"^(sglang:[a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")


def counter_snapshot() -> dict[str, float]:
    """Read /metrics and return label-stripped sglang metrics, summed.

    Counters carry per-label sets (one line per label combination); the tau
    deltas need the totals, so same-name lines sum. Gauges land here too;
    only _total deltas are read downstream.
    """
    _, body = _request("GET", "/metrics", None, 30.0)
    snapshot: dict[str, float] = {}
    for line in body.splitlines():
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        snapshot[match.group(1)] = snapshot.get(match.group(1), 0.0) + value
    return snapshot


def spec_metric_lines() -> list[str]:
    """Return the raw /metrics lines mentioning spec or accept (with labels)."""
    _, body = _request("GET", "/metrics", None, 30.0)
    return [
        line
        for line in body.splitlines()
        if ("spec" in line or "accept" in line) and not line.startswith("#")
    ]


def get_server_info() -> dict[str, Any]:
    """Fetch /get_server_info and keep the keys this bring-up asserts on."""
    status, body = _request("GET", "/get_server_info", None, 30.0)
    if status != HTTP_OK:
        msg = f"/get_server_info returned {status}"
        raise RuntimeError(msg)
    info = json.loads(body)
    return {key: info.get(key) for key in INFO_KEYS}


def completion(prompt: str, temperature: float) -> dict[str, Any]:
    """Run one /v1/completions request and flatten it into a result row."""
    body = json.dumps(
        {
            "model": "target",
            "prompt": prompt,
            "max_tokens": SMOKE_MAX_TOKENS,
            "temperature": temperature,
        }
    ).encode()
    started = time.time()
    status, resp = _request("POST", "/v1/completions", body, REQUEST_TIMEOUT_S)
    if status != HTTP_OK:
        return {"status": status, "error": resp[:500], "wall_s": time.time() - started}
    try:
        payload = json.loads(resp)
        choice = payload["choices"][0]
        return {
            "status": HTTP_OK,
            "text": choice["text"][:300],
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
            "wall_s": time.time() - started,
        }
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return {"status": status, "error": f"parse: {exc}", "wall_s": time.time() - started}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one prompt jsonl from the baked-in data directory."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_wave(
    tag: str, prompts: list[str], temperature: float, before: dict[str, float]
) -> dict[str, Any]:
    """Run one prompt wave and derive counter tau from the metric deltas."""
    rows: list[dict[str, Any]]
    started = time.time()
    with ThreadPoolExecutor(max_workers=min(4, len(prompts))) as pool:
        futures = [pool.submit(completion, prompt, temperature) for prompt in prompts]
        rows = [future.result() for future in futures]
    wall = time.time() - started
    after = counter_snapshot()
    rounds = after.get(METRIC_VERIFY_CALLS, 0.0) - before.get(METRIC_VERIFY_CALLS, 0.0)
    tokens = after.get(METRIC_GEN_TOKENS, 0.0) - before.get(METRIC_GEN_TOKENS, 0.0)
    wave: dict[str, Any] = {
        "window": tag,
        "temperature": temperature,
        "n": len(rows),
        "wall_s": wall,
        "ok": sum(1 for row in rows if row.get("status") == HTTP_OK),
        "counter_tau": (tokens / rounds) if rounds > 0 else None,
        "verify_calls_delta": rounds,
        "generation_tokens_delta": tokens,
        "requests": rows,
    }
    _log(f"wave {tag}: ok={wave['ok']}/{len(rows)} counter_tau={wave['counter_tau']}")
    return wave


def install_patch() -> dict[str, str]:
    """Patch the installed sglang files and verify each edit landed.

    The glm5_next capture patch is applied as exact-string replacements so
    no upstream mirror file rides in this repo: the expected snippets pin
    the image's source to the analyzed serving SHA, and every replacement
    must fire exactly once or the boot fails loudly.
    """
    import sglang  # resolves only inside the serving image

    models = Path(sglang.__file__).parent / "srt" / "models"
    target = models / "glm5_next.py"
    source = target.read_text(encoding="utf-8")
    for original, replacement in GLM5_NEXT_PATCH:
        hits = source.count(original)
        if hits != 1:
            msg = f"patch snippet matched {hits} times (want 1): {original[:60]!r}"
            raise RuntimeError(msg)
        source = source.replace(original, replacement)
    target.write_text(source, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)
    if "capture_final_norm_output" not in source:
        msg = "patched glm5_next.py lacks the capture flag"
        raise RuntimeError(msg)
    draft = models / "glm5_dspark.py"
    shutil.copy(Path(PATCH_DIR) / draft.name, draft)
    py_compile.compile(str(draft), doraise=True)
    return {
        "glm5_next.py": hashlib.sha256(source.encode()).hexdigest()[:16],
        "glm5_dspark.py": hashlib.sha256(draft.read_bytes()).hexdigest()[:16],
    }


def boot_report(server: subprocess.Popen[Any], cmd: list[str]) -> dict[str, Any]:
    """Build the boot section of the result: info, command, health wall."""
    boot_wall = wait_health(server)
    _log(f"healthy after {boot_wall:.0f}s")
    return {
        "command": cmd,
        "health_wall_s": boot_wall,
        "server_info": get_server_info(),
    }


def stop_server(server: subprocess.Popen[Any]) -> None:
    """Terminate the server subprocess cleanly, killing it if it hangs."""
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=30)


def server_command() -> list[str]:
    """Build the SGLang launch command for the DSPARK bring-up arm."""
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET_DIR,
        "--tp-size",
        str(TP_SIZE),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--enable-metrics",
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        DRAFTER_SERVE_DIR,
        "--speculative-dspark-block-size",
        "7",
        "--speculative-draft-attention-backend",
        # fa4: the DFlash2 precedent on this target; trtllm_mha's fused path
        # breaks the target's DSA pool (see #47 findings)
        "fa4",
    ]


@app.function(
    image=util_image,
    volumes={RUN_VOL: run_volume, VOL: serve_volume},
    timeout=60 * 30,
)
def prepare_drafter() -> None:
    """Copy step_20 to a serve dir on the weights volume with edited config."""
    src = Path(RUN_VOL) / CKPT_REL
    dst = Path(DRAFTER_SERVE_DIR)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    config = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    for key, expected in EXPECT_KEYS.items():
        if config.get(key) != expected:
            msg = f"checkpoint {key}={config.get(key)!r} != expected {expected!r}"
            raise RuntimeError(msg)
    with (dst / "model.safetensors").open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    gate_shape = header["layers.0.mlp.gate_proj.weight"]["shape"]
    serve_config = dict(SERVE_CONFIG_KEEP)
    for key in SERVE_CONFIG_KEYS:
        if key not in config:
            msg = f"checkpoint config lacks {key!r}; the serve whitelist cannot be built"
            raise RuntimeError(msg)
        serve_config[key] = config[key]
    serve_config["intermediate_size"] = gate_shape[0]
    (dst / "config.json").write_text(json.dumps(serve_config, indent=2) + "\n", encoding="utf-8")
    report = {
        "source": CKPT_REL,
        "serve_dir": DRAFTER_SERVE_DIR,
        "gate_proj_shape": gate_shape,
        "dropped_keys": sorted(set(config) - set(serve_config)),
    }
    report_path = Path(VOL) / "results" / "dspark_prep.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    serve_volume.commit()  # explicit: volume writes persist only on commit
    _log(f"prepared {dst} from {src}; gate_proj={gate_shape}")


@app.function(
    image=serve_image,
    volumes={VOL: serve_volume},
    gpu=GPU_SPEC,
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("hf-token")],
)
def serve_smoke() -> None:
    """Boot sglang DSPARK with the patched files, smoke it, tear down."""
    fingerprints = install_patch()
    cmd = server_command()
    log_path = Path(VOL) / "results" / "bringup_boot.log"
    log_handle = log_path.open("w")
    server = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    try:
        report: dict[str, Any] = {"patch": fingerprints, **boot_report(server, cmd)}
        info = report["server_info"]
        if info.get("speculative_algorithm") != "DSPARK":
            msg = f"speculative_algorithm={info.get('speculative_algorithm')!r} != DSPARK"
            raise RuntimeError(msg)
        temp0 = load_jsonl(Path(DATA_DIR) / "temp0_prompts.jsonl")[:GREEDY_PROMPTS]
        alpaca = load_jsonl(Path(DATA_DIR) / "suite_alpaca.jsonl")[:SAMPLED_PROMPTS]
        report["waves"] = [
            run_wave("temp0", [row["prompt"] for row in temp0], 0.0, counter_snapshot()),
            run_wave(
                "alpaca-temp1",
                [row["prompt"] for row in alpaca],
                1.0,
                counter_snapshot(),
            ),
        ]
        report["spec_metric_names"] = sorted(
            name for name in counter_snapshot() if "spec" in name or "accept" in name
        )
        report["spec_metrics_raw"] = spec_metric_lines()
        result_path = Path(VOL) / "results" / "dspark_bringup.json"
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        serve_volume.commit()
        _log(f"result written to {result_path}")
    finally:
        stop_server(server)
        log_handle.close()
        serve_volume.commit()


@app.local_entrypoint()
def main(stage: str = "serve-smoke", name: str = "dspark_bringup") -> None:
    """Dispatch one stage; fetch reads a result json back to the repo tree."""
    if stage == "prepare-drafter":
        prepare_drafter.remote()
    elif stage == "serve-smoke":
        serve_smoke.remote()
    elif stage == "fetch":
        out = RESULTS_DIR / f"{name}.json"
        out.write_bytes(fetch_result_bytes(SERVE_VOLUME_NAME, name))
        _log(f"fetched {out}")
    else:
        known = "prepare-drafter, serve-smoke, fetch"
        msg = f"unknown stage {stage!r}; known: {known}"
        raise ValueError(msg)


@app.function(image=util_image, volumes={VOL: serve_volume})
def peek() -> dict[str, Any]:
    """Summarize the bring-up artifacts on the volume without pulling them."""
    results = Path(VOL) / "results"
    summary: dict[str, Any] = {
        "results": sorted(p.name for p in results.glob("dspark_*")) if results.exists() else []
    }
    prep = results / "dspark_prep.json"
    if prep.exists():
        summary["prep"] = json.loads(prep.read_text(encoding="utf-8"))
    bring = results / "dspark_bringup.json"
    if bring.exists():
        payload = json.loads(bring.read_text(encoding="utf-8"))
        summary["bringup"] = {
            "server_info": payload.get("server_info"),
            "waves": [
                {
                    "window": wave["window"],
                    "ok": wave["ok"],
                    "counter_tau": wave["counter_tau"],
                }
                for wave in payload.get("waves", [])
            ],
        }
    return summary
