"""GLM-5.3-Flash drafter pipeline on Modal.

Stages: probe, split, regen, cache, train, evaluate, inspect, peek, fetch,
push-hf, restore-hf. Mirrors the dspark_proxy-run pipeline stages against the
DeepSpec repo pinned at DEEPSPEC_SHA; the worktree overlay (deepspec_glm) is
copied over the clone inside every deepspec image so the GLM trainer,
evaluator, parser template, config, and single-process cache builder are
active. The volume carries data, caches, checkpoints, and results.
"""

from __future__ import annotations

import importlib.util
import sys
import time
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

from lib.modal_pipeline import (
    DEEPSPEC_DIR,
    DEEPSPEC_REPO,
    DEEPSPEC_SHA,
    DEFAULT_TASKS,
    HOME,
    MAX_LISTED_FILES,
    VOL,
    VOLUME_ROOTS,
    build_cache_cmd,
    build_eval_cmd,
    build_push_jobs,
    build_train_cmd,
    fetch_result_bytes,
    inspect_volume,
    push_jobs_to_hf,
    restore_from_hf,
    serve_and_run_client,
)
from lib.modal_pipeline import (
    log as _log,
)
from lib.modal_pipeline import (
    run as _run,
)

VOLUME_NAME = "glm-drafter-run"
TARGET = "zai-org/GLM-5.3-Flash"
DFLASH2_DRAFT = "incoai/GLM-5.3-Flash-DFlash2"
LOCAL_DIR = Path(__file__).parent.resolve()
OVERLAY_DIR = LOCAL_DIR / "deepspec_glm"
TRAIN_GPU = "B200:4"
EVAL_GPU = "B200:4"
HF_ASSET_REPO = "phillipchaffee/glm-drafter-run"
RESTORE_PATTERNS = ("cache/", "checkpoints/")

PROXY_SCRIPTS_DIR = LOCAL_DIR.parent / "proxy-run" / "scripts"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


_deepspec_layers = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("huggingface_hub")
    .env({"HF_HUB_CACHE": "/hf"})
    .add_local_dir(OVERLAY_DIR, "/overlay", copy=True)
    .add_local_dir(LOCAL_DIR / "scripts", "/work/scripts", copy=True)
    # the GLM data scripts run with cwd=/deepspec via deepspec-relative paths
    .add_local_dir(LOCAL_DIR / "scripts", "/deepspec/scripts", copy=True)
    .add_local_dir(PROXY_SCRIPTS_DIR, "/proxy/scripts", copy=True)
    .add_local_dir(LOCAL_DIR.parent / "lib", "/root/lib", copy=True)
    .run_commands(
        f"git init {DEEPSPEC_DIR}",
        f"git -C {DEEPSPEC_DIR} remote add origin {DEEPSPEC_REPO}",
        f"git -C {DEEPSPEC_DIR} fetch --depth 1 origin {DEEPSPEC_SHA}",
        f"git -C {DEEPSPEC_DIR} checkout FETCH_HEAD",
        "pip install -r /deepspec/requirements.txt",
        # GLM-5.3-Flash (glm5_next) needs newer transformers than the
        # upstream requirements pin: the hub config declares
        # transformers_version 5.16.0, and glm5_next lands by 5.19.
        "pip install 'transformers==5.19.0'",
        # device_map="auto" (single-process cache/eval) requires accelerate;
        # transformers 5.19 does not pull it as a hard dependency.
        "pip install 'accelerate>=1.11.0'",
        # the fp8 checkpoint's finegrained kernels load through the kernels
        # package on transformers 5.19 (0.17.x is the compatible range)
        "pip install 'kernels==0.17.0'",
        "cp -r /overlay/. /deepspec/",
    )
)

deepspec_image = _deepspec_layers.env({"PYTHONPATH": DEEPSPEC_DIR})

regen_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("sglang[all]", "openai", "datasets", "tqdm")
    .run_commands(
        f"git init {DEEPSPEC_DIR}",
        f"git -C {DEEPSPEC_DIR} remote add origin {DEEPSPEC_REPO}",
        f"git -C {DEEPSPEC_DIR} fetch --depth 1 origin {DEEPSPEC_SHA}",
        f"git -C {DEEPSPEC_DIR} checkout FETCH_HEAD",
    )
    .add_local_dir(LOCAL_DIR / "scripts", "/work/scripts")
    .add_local_dir(LOCAL_DIR.parent / "lib", "/root/lib")
    .add_local_dir(PROXY_SCRIPTS_DIR, "/proxy/scripts")
)

util_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets", "tqdm", "huggingface_hub", "hf_xet")
    .add_local_dir(LOCAL_DIR / "scripts", "/work/scripts")
    .add_local_dir(LOCAL_DIR.parent / "lib", "/root/lib")
    .add_local_dir(PROXY_SCRIPTS_DIR, "/proxy/scripts")
)

hf_secret = modal.Secret.from_name("hf-token")

app = modal.App("glm-drafter-run")


def _import_train_path_modules() -> None:
    """Import every deepspec module the GPU train/eval paths execute, loudly."""
    # The GPU paths are unexercisable locally (no torch outside the image), so
    # a broken import in this chain only surfaces on GPU: PR #50 rewrote the
    # glm5_trainer import and it died on rank 0 at train time, post image build.
    # deepspec.trainer pulls base/dspark/eagle3/glm5 trainers via __init__;
    # glm5_eval pulls the evaluator chain (base_evaluator + evaluator).
    train_path_modules = (
        "deepspec.trainer",
        "deepspec.trainer.glm5_trainer",
        "deepspec.modeling.dspark.glm5",
        "deepspec.eval.dspark.glm5_eval",
    )
    for module_name in train_path_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            msg = f"probe: import failed for {module_name}: {exc}"
            raise RuntimeError(msg) from exc
        _log(f"probe: {module_name} imports OK")


def _exec_entry_script() -> None:
    """Execute-load the stage entry scripts like the stages invoke them."""
    # Loaded as files (train.py's --config loader and build_eval_cmd both hand
    # python a path), guarded main()s after.
    entry_scripts = (
        "/deepspec/config/dspark/glm5_flash.py",
        "/deepspec/scripts/timed_eval_glm.py",
        "/deepspec/scripts/data/generate_train_data_glm.py",
        "/deepspec/scripts/data/prepare_target_cache_glm.py",
    )
    for script_name in entry_scripts:
        spec = importlib.util.spec_from_file_location(Path(script_name).stem, script_name)
        if spec is None or spec.loader is None:
            msg = f"probe: cannot load {script_name}"
            raise RuntimeError(msg)
        try:
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        except Exception as exc:
            msg = f"probe: import failed for {script_name}: {exc}"
            raise RuntimeError(msg) from exc
        _log(f"probe: {Path(script_name).name} imports OK")


@app.function(
    image=deepspec_image,
    volumes={VOL: volume},
    timeout=1800,
    secrets=[hf_secret],
)
def probe() -> None:
    """CPU-cheap compatibility checks before any GPU spend.

    Verifies the image transformers version against glm5_next availability,
    loads the gated target config/tokenizer, renders the chat template with
    the regen kwargs, reads the DFlash2 baseline config, and compares the
    trainer's expected checkpoint keys against the real safetensors index.
    It then imports every overlay module the GPU stages execute (trainer chain,
    drafter package, evaluator chain, config module) plus the three data/eval
    entry scripts, so a broken import dies here at image time instead of after
    a GPU acquisition cycle.
    """
    import transformers

    _log(f"probe: transformers {transformers.__version__}")
    from transformers.models.glm5_next import modeling_glm5_next

    class_count = (
        len(modeling_glm5_next.__all__) if hasattr(modeling_glm5_next, "__all__") else "n/a"
    )
    _log(f"probe: glm5_next available, classes={class_count}")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(TARGET, trust_remote_code=False)
    _log(f"probe: target model_type={config.model_type}")
    text_config = getattr(config, "text_config", config)
    _log(
        f"probe: text model_type={getattr(text_config, 'model_type', 'n/a')} "
        f"layers={getattr(text_config, 'num_hidden_layers', 'n/a')} "
        f"hidden={getattr(text_config, 'hidden_size', 'n/a')}"
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
        clear_thinking=True,
    )
    _log(f"probe: chat template render ok, {len(rendered)} chars")

    import json

    from huggingface_hub import hf_hub_download

    dflash2_config = json.loads(
        Path(hf_hub_download(DFLASH2_DRAFT, "config.json")).read_text(encoding="utf-8")
    )
    _log(f"probe: DFlash2 draft config model_type={dflash2_config.get('model_type')}")

    index_path = hf_hub_download(TARGET, "model.safetensors.index.json")
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    keys = list(index.get("weight_map", {}))
    _log(f"probe: safetensors index has {len(keys)} entries")

    from deepspec.trainer.glm5_trainer import resolve_target_io_tensors

    io_specs = resolve_target_io_tensors.__doc__ or ""
    expected_markers = ("embed", "lm_head")
    for marker in expected_markers:
        _log(f"probe: trainer io marker '{marker}' documented={marker in io_specs}")
    interesting = sorted(
        {k for k in keys if "embed" in k or "lm_head" in k or "hyper" in k.lower()}
    )
    for key in interesting[:40]:
        _log(f"probe: index key {key}")
    _import_train_path_modules()
    _exec_entry_script()
    _log("probe: OK")


@app.function(image=util_image, volumes={VOL: volume}, timeout=3600, cpu=8, memory=32768)
def split(sample_size: int = 20000, seed: int = 42) -> None:
    """Write the shuffled training split onto the volume."""
    _run(
        [
            "python",
            "/proxy/scripts/make_split.py",
            "--sample-size",
            str(sample_size),
            "--seed",
            str(seed),
            "--output",
            f"{VOL}/data/perfectblend_train.jsonl",
        ]
    )
    volume.commit()


@app.function(
    image=regen_image,
    gpu=TRAIN_GPU,
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "HF_HUB_CACHE": f"{HOME}/.cache/huggingface"},
    secrets=[hf_secret],
)
def regen(sample_size: int = 700, concurrency: int = 64, max_tokens: int = 4096) -> None:
    """Serve the target with sglang (TP4) and regenerate training data through it."""
    elapsed = serve_and_run_client(
        server_cmd=[
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            TARGET,
            "--tp-size",
            "4",
            "--port",
            "30000",
            "--context-length",
            "12288",
            "--mem-fraction-static",
            "0.85",
        ],
        health_url="http://127.0.0.1:30000/health",
        health_deadline_s=60 * 60,
        server_name="sglang",
        client_cmd=[
            "python",
            "/work/scripts/data/generate_train_data_glm.py",
            "--model",
            TARGET,
            "--server-address",
            "127.0.0.1:30000",
            "--concurrency",
            str(concurrency),
            "--max-tokens",
            str(max_tokens),
            "--resume",
            "--num-samples",
            str(sample_size),
            "--input-file-path",
            f"{VOL}/data/perfectblend_train.jsonl",
            "--output-file-path",
            f"{VOL}/data/regen.jsonl",
        ],
    )
    volume.commit()
    _log(f"regen done in {elapsed / 60:.1f} min")


@app.function(
    image=deepspec_image,
    gpu=TRAIN_GPU,
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "HF_HUB_CACHE": f"{HOME}/.cache/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    secrets=[hf_secret],
)
def cache(local_batch_size: int = 2) -> None:
    """Build the target hidden-state cache in a single process with device_map=auto."""
    _run(
        build_cache_cmd(
            config="config/dspark/glm5_flash.py",
            cache_script="scripts/data/prepare_target_cache_glm.py",
            train_data_path=f"{VOL}/data/regen.jsonl",
            output_dir=f"{VOL}/cache",
            local_batch_size=local_batch_size,
        ),
        cwd=DEEPSPEC_DIR,
    )
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu=TRAIN_GPU,
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_CACHE": f"{HOME}/.cache/huggingface",
    },
    secrets=[hf_secret],
)
def train(
    exp_name: str = "glm_drafter_block7_glm53flash",
    epochs: int = 10,
    local_batch_size: int = 4,
    head_type: str = "vanilla",
    max_steps: int = 0,
    *,
    torch_compile: bool | None = None,
) -> None:
    """Train the block-7 GLM drafter with DeepSpec on the cached states."""
    _run(
        build_train_cmd(
            config="config/dspark/glm5_flash.py",
            exp_name=exp_name,
            epochs=epochs,
            local_batch_size=local_batch_size,
            head_type=head_type,
            cache_path="/vol/cache",
            max_steps=max_steps,
            torch_compile=torch_compile,
        ),
        cwd=DEEPSPEC_DIR,
    )
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu=TRAIN_GPU,
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "HF_HUB_CACHE": f"{HOME}/.cache/huggingface",
    },
    secrets=[hf_secret],
)
def train_smoke(
    exp_name: str = "glm_drafter_block7_glm53flash_smoke",
    epochs: int = 2,
    local_batch_size: int = 4,
    head_type: str = "vanilla",
    max_steps: int = 0,
) -> None:
    """Run a short local training pass to smoke-test the pipeline."""
    train.local(
        exp_name=exp_name,
        epochs=epochs,
        local_batch_size=local_batch_size,
        head_type=head_type,
        max_steps=max_steps,
    )


SWEEP_ARMS: tuple[tuple[int, bool], ...] = (
    (4, False),
    (8, False),
    (16, False),
    (4, True),
    (8, True),
    (16, True),
)


@app.function(
    image=deepspec_image,
    gpu=TRAIN_GPU,
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "HF_HUB_CACHE": f"{HOME}/.cache/huggingface",
    },
    secrets=[hf_secret],
)
def train_sweep(max_steps: int = 20) -> None:
    """Measure larger micro-batches and torch_compile engagement at smoke scale (issue 36)."""
    for micro, compile_on in SWEEP_ARMS:
        _log(f"[sweep-arm] micro={micro} compile={compile_on} max_steps={max_steps}")
        try:
            train.local(
                exp_name=f"glm_drafter_swp_m{micro}_c{int(compile_on)}",
                local_batch_size=micro,
                max_steps=max_steps,
                torch_compile=compile_on,
            )
        except Exception as exc:  # noqa: BLE001 - a lost arm is data, not a run-killer
            _log(f"[sweep-arm] FAILED micro={micro} compile={compile_on}: {exc}")


@app.function(
    image=deepspec_image,
    gpu=EVAL_GPU,
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "HF_HUB_CACHE": f"{HOME}/.cache/huggingface",
    },
    secrets=[hf_secret],
)
def evaluate(
    draft: str,
    tag: str,
    tasks: str = DEFAULT_TASKS,
    *,
    max_new_tokens: int = 1024,
    confidence_threshold: float = 0.0,
    baseline: bool = False,
) -> None:
    """Run the timed evaluator for one drafter and commit the result json."""
    Path(f"{VOL}/results").mkdir(parents=True, exist_ok=True)
    _run(
        build_eval_cmd(
            script="/work/scripts/timed_eval_glm.py",
            target=TARGET,
            draft=draft,
            tasks=tasks,
            max_new_tokens=max_new_tokens,
            confidence_threshold=confidence_threshold,
            out_path=f"{VOL}/results/{tag}.json",
            baseline=baseline,
        ),
        cwd=DEEPSPEC_DIR,
    )
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu=EVAL_GPU,
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "HF_HUB_CACHE": f"{HOME}/.cache/huggingface",
    },
    secrets=[hf_secret],
)
def evaluate_smoke(
    draft: str,
    tag: str,
    tasks: str = "gsm8k:12",
    *,
    max_new_tokens: int = 256,
    confidence_threshold: float = 0.0,
    baseline: bool = False,
) -> None:
    """Run the evaluator on a tiny task subset to smoke-test a drafter."""
    evaluate.local(
        draft=draft,
        tag=tag,
        tasks=tasks,
        max_new_tokens=max_new_tokens,
        confidence_threshold=confidence_threshold,
        baseline=baseline,
    )


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def inspect() -> None:
    """Print a size summary of every directory on the volume."""
    volume.reload()
    inspect_volume(VOL, VOLUME_ROOTS, MAX_LISTED_FILES)


def _peek_regen() -> None:
    """Report regen output line counts and last-modified times."""
    for label, path in (
        ("regen", Path(VOL) / "data/regen.jsonl"),
        ("regen_errors", Path(VOL) / "data/regen_error.jsonl"),
    ):
        if not path.exists():
            _log(f"{label}: missing")
            continue
        with path.open(encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
        _log(f"{label}: {lines} lines, last modified {mtime}")


def _peek_checkpoints() -> None:
    """Report checkpoint files on the volume."""
    ckpt_root = Path(VOL) / "home/checkpoints"
    if not ckpt_root.exists():
        _log("checkpoints: missing")
        return
    for ckpt in sorted(ckpt_root.rglob("*")):
        if ckpt.is_file():
            _log(f"checkpoint: {ckpt.relative_to(VOL)} {ckpt.stat().st_size / 1e9:.2f} GB")


def _peek_results() -> None:
    """Report result json files on the volume."""
    results_dir = Path(VOL) / "results"
    if not results_dir.exists():
        _log("results: missing")
        return
    for item in sorted(results_dir.glob("*.json")):
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.stat().st_mtime))
        _log(f"result: {item.name} {item.stat().st_size} bytes, modified {mtime}")


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def peek() -> None:
    """Poll progress of detached runs: regen line count, checkpoints, results."""
    volume.reload()
    _peek_regen()
    _peek_checkpoints()
    _peek_results()


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def fetch_result(name: str) -> bytes:
    """Read one result json from the volume."""
    return fetch_result_bytes(VOL, name)


@app.function(
    image=util_image,
    volumes={VOL: volume},
    timeout=4 * 3600,
    cpu=8,
    memory=32768,
    secrets=[hf_secret],
)
def restore_hf(repo_id: str = HF_ASSET_REPO) -> None:
    """Restore cache shards and checkpoints from the HF dataset repo."""
    restore_from_hf(VOL, repo_id, RESTORE_PATTERNS, volume.commit)


@app.function(
    image=util_image,
    volumes={VOL: volume},
    timeout=12 * 3600,
    secrets=[hf_secret],
    cpu=8,
)
def push_to_hf(repo_id: str = "", exp_name: str = "") -> None:
    """Upload the cache, regen data, and final checkpoint to an HF dataset repo."""
    push_jobs_to_hf(repo_id, build_push_jobs(VOL, exp_name), "glm-drafter-run")
    volume.commit()


def _build_dispatch(
    *,
    sample_size: int,
    seed: int,
    concurrency: int,
    exp_name: str,
    epochs: int,
    cache_batch_size: int,
    local_batch_size: int,
    head_type: str,
    max_steps: int,
    draft: str,
    tag: str,
    tasks: str,
    max_new_tokens: int,
    confidence_threshold: float,
    baseline: bool,
    repo_id: str,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Map every stage name to its function and flag-derived parameters."""
    return {
        "probe": (probe, {}),
        "split": (split, {"sample_size": sample_size, "seed": seed}),
        "regen": (regen, {"sample_size": sample_size, "concurrency": concurrency}),
        "cache": (cache, {"local_batch_size": cache_batch_size}),
        "train": (
            train,
            {
                "exp_name": exp_name,
                "epochs": epochs,
                "local_batch_size": local_batch_size,
                "head_type": head_type,
                "max_steps": max_steps,
            },
        ),
        "train-smoke": (
            train_smoke,
            {
                "exp_name": exp_name,
                "epochs": epochs,
                "local_batch_size": local_batch_size,
                "head_type": head_type,
                "max_steps": max_steps,
            },
        ),
        "train-sweep": (train_sweep, {"max_steps": max_steps}),
        "eval": (
            evaluate,
            {
                "draft": draft,
                "tag": tag,
                "tasks": tasks,
                "max_new_tokens": max_new_tokens,
                "confidence_threshold": confidence_threshold,
                "baseline": baseline,
            },
        ),
        "eval-smoke": (
            evaluate_smoke,
            {
                "draft": draft,
                "tag": tag,
                "tasks": tasks,
                "max_new_tokens": max_new_tokens,
                "confidence_threshold": confidence_threshold,
                "baseline": baseline,
            },
        ),
        "inspect": (inspect, {}),
        "peek": (peek, {}),
        "fetch": (fetch_result, {"name": tag}),
        "restore-hf": (restore_hf, {"repo_id": repo_id} if repo_id else {}),
        "push-hf": (push_to_hf, {"repo_id": repo_id, "exp_name": exp_name}),
    }


@app.local_entrypoint()
def main(
    stage: str,
    *,
    sample_size: int = 700,
    seed: int = 42,
    concurrency: int = 64,
    exp_name: str = "glm_drafter_block7_glm53flash",
    epochs: int = 10,
    cache_batch_size: int = 2,
    local_batch_size: int = 4,
    head_type: str = "vanilla",
    max_steps: int = 0,
    draft: str = DFLASH2_DRAFT,
    tag: str = "eval",
    tasks: str = DEFAULT_TASKS,
    max_new_tokens: int = 1024,
    confidence_threshold: float = 0.0,
    baseline: bool = False,
    repo_id: str = "",
    foreground: bool = False,
) -> None:
    """Dispatch one pipeline stage; every flag maps to a stage parameter."""
    dispatch = _build_dispatch(
        sample_size=sample_size,
        seed=seed,
        concurrency=concurrency,
        exp_name=exp_name,
        epochs=epochs,
        cache_batch_size=cache_batch_size,
        local_batch_size=local_batch_size,
        head_type=head_type,
        max_steps=max_steps,
        draft=draft,
        tag=tag,
        tasks=tasks,
        max_new_tokens=max_new_tokens,
        confidence_threshold=confidence_threshold,
        baseline=baseline,
        repo_id=repo_id,
    )
    fn, kwargs = dispatch[stage]
    if foreground:
        sys.stdout.write(f"{fn.local(**kwargs)}\n")
    else:
        call = fn.spawn(**kwargs)
        sys.stdout.write(f"spawned stage={stage} call_id={call.object_id}\n")
