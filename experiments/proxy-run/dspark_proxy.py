"""DSpark proxy-run pipeline on Modal: split, regen, cache, train, evaluate.

Stages run against the DeepSpec repo pinned at DEEPSPEC_SHA inside Modal
images; the volume carries data, caches, checkpoints, and results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import modal

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
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

VOLUME_NAME = "dspark-proxy-run"
TARGET = "Qwen/Qwen3-4B"
RELEASED_DRAFT = "deepseek-ai/dspark_qwen3_4b_block7"
LOCAL_DIR = Path(__file__).parent.resolve()

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _bake_weights() -> None:
    """Pre-download the target and released-drafter weights into the image."""
    from huggingface_hub import snapshot_download

    snapshot_download(TARGET)
    snapshot_download(RELEASED_DRAFT)


_deepspec_layers = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("huggingface_hub")
    .env({"HF_HUB_CACHE": "/hf"})
    .run_function(_bake_weights, timeout=3600)
    .run_commands(
        f"git init {DEEPSPEC_DIR}",
        f"git -C {DEEPSPEC_DIR} remote add origin {DEEPSPEC_REPO}",
        f"git -C {DEEPSPEC_DIR} fetch --depth 1 origin {DEEPSPEC_SHA}",
        f"git -C {DEEPSPEC_DIR} checkout FETCH_HEAD",
        "pip install -r /deepspec/requirements.txt",
    )
)

deepspec_image = _deepspec_layers.add_local_dir(LOCAL_DIR / "scripts", "/work/scripts")

regen_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("vllm", "openai", "datasets", "tqdm")
    .run_commands(
        f"git init {DEEPSPEC_DIR}",
        f"git -C {DEEPSPEC_DIR} remote add origin {DEEPSPEC_REPO}",
        f"git -C {DEEPSPEC_DIR} fetch --depth 1 origin {DEEPSPEC_SHA}",
        f"git -C {DEEPSPEC_DIR} checkout FETCH_HEAD",
    )
    .add_local_dir(LOCAL_DIR / "scripts", "/work/scripts")
)

util_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets", "tqdm", "huggingface_hub", "hf_xet")
    .add_local_dir(LOCAL_DIR / "scripts", "/work/scripts")
)

hf_secret = modal.Secret.from_name("hf-token")

app = modal.App("dspark-proxy-run")


@app.function(image=util_image, volumes={VOL: volume}, timeout=3600, cpu=8, memory=32768)
def split(sample_size: int = 20000, seed: int = 42) -> None:
    """Write the shuffled training split onto the volume."""
    _run(
        [
            "python",
            "/work/scripts/make_split.py",
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
    gpu="H100",
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "HF_HUB_CACHE": f"{HOME}/.cache/huggingface"},
)
def regen(sample_size: int = 20000, concurrency: int = 64, max_tokens: int = 4096) -> None:
    """Serve vLLM on the target and regenerate DeepSpec training data through it."""
    elapsed = serve_and_run_client(
        server_cmd=[
            "vllm",
            "serve",
            TARGET,
            "--port",
            "30000",
            "--max-model-len",
            "12288",
        ],
        health_url="http://127.0.0.1:30000/health",
        health_deadline_s=30 * 60,
        server_name="vllm",
        client_cmd=[
            "python",
            f"{DEEPSPEC_DIR}/scripts/data/generate_train_data.py",
            "--model",
            TARGET,
            "--server-address",
            "127.0.0.1:30000",
            "--concurrency",
            str(concurrency),
            "--temperature",
            "0.7",
            "--top-p",
            "0.8",
            "--top-k",
            "20",
            "--max-tokens",
            str(max_tokens),
            "--disable-thinking",
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
    gpu="H100",
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "PYTHONPATH": DEEPSPEC_DIR},
)
def cache(local_batch_size: int = 16) -> None:
    """Build the target hidden-state cache from the regenerated data."""
    _run(
        build_cache_cmd(
            config="config/dspark/dspark_qwen3_4b.py",
            cache_script="scripts/data/prepare_target_cache.py",
            train_data_path=f"{VOL}/data/regen.jsonl",
            output_dir=f"{VOL}/cache",
            local_batch_size=local_batch_size,
        ),
        cwd=DEEPSPEC_DIR,
    )
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu="H100:4",
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)
def train(
    exp_name: str = "dspark_block7_qwen3_4b",
    epochs: int = 10,
    local_batch_size: int = 4,
    head_type: str = "vanilla",
    max_steps: int = 0,
) -> None:
    """Train the block-7 DSpark drafter with DeepSpec on the cached states.

    head_type selects the sequential head (vanilla/gated/rnn) via the
    config-only override model.markov_head_type. max_steps truncates training
    after that many optimizer steps (0 = no override; smoke runs use it).
    """
    _run(
        build_train_cmd(
            config="config/dspark/dspark_qwen3_4b.py",
            exp_name=exp_name,
            epochs=epochs,
            local_batch_size=local_batch_size,
            head_type=head_type,
            cache_path="/vol/cache",
            max_steps=max_steps,
        ),
        cwd=DEEPSPEC_DIR,
    )
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "PYTHONPATH": DEEPSPEC_DIR},
)
def train_smoke(
    exp_name: str = "dspark_block7_qwen3_4b_smoke",
    epochs: int = 2,
    local_batch_size: int = 2,
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


@app.function(
    image=deepspec_image,
    gpu="H100:8",
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "PYTHONPATH": DEEPSPEC_DIR},
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
            script="/work/scripts/timed_eval.py",
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
    gpu="H100",
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "PYTHONPATH": DEEPSPEC_DIR},
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


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def fetch_result(name: str) -> bytes:
    """Read one result json from the volume."""
    return fetch_result_bytes(VOL, name)


HF_ASSET_REPO = "phillipchaffee/dspark-proxy-run"
RESTORE_PATTERNS = ("cache/", "checkpoints/")


@app.function(
    image=util_image,
    volumes={VOL: volume},
    timeout=4 * 3600,
    cpu=8,
    memory=32768,
    secrets=[hf_secret],
)
def restore_hf(repo_id: str = HF_ASSET_REPO) -> None:
    """Restore cache shards and the baseline checkpoint from the HF dataset repo."""
    restore_from_hf(VOL, repo_id, RESTORE_PATTERNS, volume.commit)


@app.function(
    image=util_image,
    volumes={VOL: volume},
    timeout=12 * 3600,
    secrets=[hf_secret],
    cpu=8,
)
def push_to_hf(repo_id: str = "") -> None:
    """Upload the cache, regen data, and full-run checkpoint to a public HF dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    repo = repo_id or f"{user}/dspark-proxy-run"
    api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)
    cache_dir = Path(VOL) / "cache"
    data_dir = Path(VOL) / "data"
    jobs: list[tuple[Path, str]] = [
        (item, f"cache/{item.name}") for item in sorted(cache_dir.iterdir()) if item.is_file()
    ]
    jobs.extend(
        (item, f"data/{item.name}") for item in sorted(data_dir.iterdir()) if item.is_file()
    )
    ckpt_dir = Path(VOL) / "home/checkpoints/deepspec/dspark_block7_qwen3_4b_full/step_380"
    jobs.extend(
        (ckpt_dir / name, f"checkpoints/dspark_block7_qwen3_4b_full_step_380/{name}")
        for name in ("model.safetensors", "config.json", "train_config.py")
    )
    push_jobs_to_hf(repo, jobs, "dspark-proxy-run")
    volume.commit()


@app.local_entrypoint()
def main(
    stage: str,
    *,
    sample_size: int = 20000,
    seed: int = 42,
    concurrency: int = 64,
    exp_name: str = "dspark_block7_qwen3_4b",
    epochs: int = 10,
    local_batch_size: int = 4,
    head_type: str = "vanilla",
    max_steps: int = 0,
    draft: str = RELEASED_DRAFT,
    tag: str = "eval",
    tasks: str = DEFAULT_TASKS,
    max_new_tokens: int = 1024,
    confidence_threshold: float = 0.0,
    baseline: bool = False,
    repo_id: str = "",
    foreground: bool = False,
) -> None:
    """Dispatch one pipeline stage; every flag maps to a stage parameter."""
    dispatch: dict[str, tuple[Any, dict[str, Any]]] = {
        "split": (split, {"sample_size": sample_size, "seed": seed}),
        "regen": (regen, {"sample_size": sample_size, "concurrency": concurrency}),
        "cache": (cache, {}),
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
        "restore-hf": (restore_hf, {"repo_id": repo_id} if repo_id else {}),
        "push-hf": (push_to_hf, {"repo_id": repo_id}),
    }
    fn, kwargs = dispatch[stage]
    if foreground:
        sys.stdout.write(f"{fn.local(**kwargs)}\n")
    else:
        call = fn.spawn(**kwargs)
        sys.stdout.write(f"spawned stage={stage} call_id={call.object_id}\n")
