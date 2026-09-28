"""DSpark proxy-run pipeline on Modal: split, regen, cache, train, evaluate.

Stages run against the DeepSpec repo pinned at DEEPSPEC_SHA inside Modal
images; the volume carries data, caches, checkpoints, and results.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import modal

DEEPSPEC_REPO = "https://github.com/deepseek-ai/DeepSpec"
DEEPSPEC_SHA = "005e03b81cec38b7da6399833d609ee89a2587f2"
DEEPSPEC_DIR = "/deepspec"
VOL = "/vol"
HOME = "/vol/home"
VOLUME_NAME = "dspark-proxy-run"
TARGET = "Qwen/Qwen3-4B"
RELEASED_DRAFT = "deepseek-ai/dspark_qwen3_4b_block7"
LOCAL_DIR = Path(__file__).parent.resolve()
DEFAULT_TASKS = "gsm8k:300,math500:300,humaneval:164,mbpp:256,mt-bench:80,alpaca:300"
MAX_LISTED_FILES = 60  # inspect lists at most this many files per directory

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


def _log(message: str) -> None:
    """Print a line immediately (the print-free lint gate owns this shape)."""
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def _run(cmd: list[str], cwd: str | None = None) -> None:
    """Run a command, failing loudly on a nonzero exit."""
    _log("+ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        msg = f"command failed rc={exc.returncode}: {' '.join(cmd)}"
        raise RuntimeError(msg) from exc


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
    input_path = f"{VOL}/data/perfectblend_train.jsonl"
    output_path = f"{VOL}/data/regen.jsonl"
    t0 = time.time()
    server = subprocess.Popen(
        ["vllm", "serve", TARGET, "--port", "30000", "--max-model-len", "12288"],
    )
    deadline = time.time() + 30 * 60
    while True:
        if server.poll() is not None:
            msg = "vllm server exited during startup"
            raise RuntimeError(msg)
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=5)
            break
        except OSError as exc:
            if time.time() > deadline:
                server.terminate()
                msg = "vllm server did not become healthy in 30 minutes"
                raise RuntimeError(msg) from exc
            time.sleep(10)
    _log(f"vllm healthy after {time.time() - t0:.0f}s")
    try:
        _run(
            [
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
                input_path,
                "--output-file-path",
                output_path,
            ]
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
    volume.commit()
    _log(f"regen done in {(time.time() - t0) / 60:.1f} min")


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
        [
            "python",
            "scripts/data/prepare_target_cache.py",
            "--config",
            "config/dspark/dspark_qwen3_4b.py",
            "--train-data-path",
            f"{VOL}/data/regen.jsonl",
            "--output-dir",
            f"{VOL}/cache",
            "--local-batch-size",
            str(local_batch_size),
        ],
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
    exp_name: str = "dspark_block7_qwen3_4b", epochs: int = 10, local_batch_size: int = 4
) -> None:
    """Train the block-7 DSpark drafter with DeepSpec on the cached states."""
    _run(
        [
            "python",
            "train.py",
            "--config",
            "config/dspark/dspark_qwen3_4b.py",
            "--opts",
            f"exp_name={exp_name}",
            "--opts",
            f"train.num_train_epochs={epochs}",
            "--opts",
            f"train.local_batch_size={local_batch_size}",
            "--opts",
            "data.target_cache_path=/vol/cache",
        ],
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
) -> None:
    """Run a short local training pass to smoke-test the pipeline."""
    train.local(
        exp_name=exp_name,
        epochs=epochs,
        local_batch_size=local_batch_size,
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
    cmd = [
        "python",
        "/work/scripts/timed_eval.py",
        "--target",
        TARGET,
        "--draft",
        draft,
        "--tasks",
        tasks,
        "--max-new-tokens",
        str(max_new_tokens),
        "--confidence-threshold",
        str(confidence_threshold),
        "--out",
        f"{VOL}/results/{tag}.json",
    ]
    if baseline:
        cmd.append("--baseline")
    _run(cmd, cwd=DEEPSPEC_DIR)
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
    for root in ("data", "cache", "results", "home/checkpoints", "home/tensorboard"):
        path = Path(VOL) / root
        if not path.exists():
            _log(f"{path}: missing")
            continue
        entries = []
        total = 0
        for item in sorted(path.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                total += size
                entries.append(f"  {item.relative_to(VOL)} {size}")
        _log(f"{path}: {total / 1e9:.2f} GB across {len(entries)} files")
        for line in entries[:MAX_LISTED_FILES]:
            _log(line)
        if len(entries) > MAX_LISTED_FILES:
            _log(f"  ... {len(entries) - MAX_LISTED_FILES} more files")


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def fetch_result(name: str) -> bytes:
    """Read one result json from the volume."""
    return (Path(VOL) / "results" / f"{name}.json").read_bytes()


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
    jobs: list[tuple[Path, str]] = []
    cache_dir = Path(VOL) / "cache"
    for item in sorted(cache_dir.iterdir()):
        if item.is_file():
            jobs.append((item, f"cache/{item.name}"))
    data_dir = Path(VOL) / "data"
    for item in sorted(data_dir.iterdir()):
        if item.is_file():
            jobs.append((item, f"data/{item.name}"))
    ckpt_dir = Path(VOL) / "home/checkpoints/deepspec/dspark_block7_qwen3_4b_full/step_380"
    for name in ("model.safetensors", "config.json", "train_config.py"):
        jobs.append((ckpt_dir / name, f"checkpoints/dspark_block7_qwen3_4b_full_step_380/{name}"))
    t0 = time.time()
    for idx, (src, dst) in enumerate(jobs, 1):
        size = src.stat().st_size
        _log(f"[{idx}/{len(jobs)}] {dst} {size / 1e9:.1f} GB uploading")
        f0 = time.time()
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo, repo_type="dataset")
        _log(f"[{idx}/{len(jobs)}] {dst} done in {(time.time() - f0) / 60:.1f} min")
    _log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min -> https://huggingface.co/datasets/{repo}")
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
            {"exp_name": exp_name, "epochs": epochs, "local_batch_size": local_batch_size},
        ),
        "train-smoke": (
            train_smoke,
            {"exp_name": exp_name, "epochs": epochs, "local_batch_size": local_batch_size},
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
        "push-hf": (push_to_hf, {"repo_id": repo_id}),
    }
    fn, kwargs = dispatch[stage]
    if foreground:
        sys.stdout.write(f"{fn.local(**kwargs)}\n")
    else:
        call = fn.spawn(**kwargs)
        sys.stdout.write(f"spawned stage={stage} call_id={call.object_id}\n")
