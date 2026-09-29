from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

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

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _bake_weights():
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
    .pip_install("datasets", "tqdm")
    .add_local_dir(LOCAL_DIR / "scripts", "/work/scripts")
)

app = modal.App("dspark-proxy-run")


def _run(cmd, cwd=None):
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {' '.join(cmd)}")


@app.function(image=util_image, volumes={VOL: volume}, timeout=3600, cpu=8, memory=32768)
def split(sample_size: int = 20000, seed: int = 42):
    _run([
        "python", "/work/scripts/make_split.py",
        "--sample-size", str(sample_size),
        "--seed", str(seed),
        "--output", f"{VOL}/data/perfectblend_train.jsonl",
    ])
    volume.commit()


@app.function(
    image=regen_image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "HF_HUB_CACHE": f"{HOME}/.cache/huggingface"},
)
def regen(sample_size: int = 20000, concurrency: int = 64, max_tokens: int = 4096):
    input_path = f"{VOL}/data/perfectblend_train.jsonl"
    output_path = f"{VOL}/data/regen.jsonl"
    t0 = time.time()
    server = subprocess.Popen(
        ["vllm", "serve", TARGET, "--port", "30000", "--max-model-len", "12288"],
    )
    deadline = time.time() + 30 * 60
    while True:
        if server.poll() is not None:
            raise RuntimeError("vllm server exited during startup")
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=5)
            break
        except Exception:
            if time.time() > deadline:
                server.terminate()
                raise RuntimeError("vllm server did not become healthy in 30 minutes")
            time.sleep(10)
    print(f"vllm healthy after {time.time() - t0:.0f}s", flush=True)
    try:
        _run([
            "python", f"{DEEPSPEC_DIR}/scripts/data/generate_train_data.py",
            "--model", TARGET,
            "--server-address", "127.0.0.1:30000",
            "--concurrency", str(concurrency),
            "--temperature", "0.7", "--top-p", "0.8", "--top-k", "20",
            "--max-tokens", str(max_tokens),
            "--disable-thinking", "--resume",
            "--num-samples", str(sample_size),
            "--input-file-path", input_path,
            "--output-file-path", output_path,
        ])
    finally:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
    volume.commit()
    print(f"regen done in {(time.time() - t0) / 60:.1f} min", flush=True)


@app.function(
    image=deepspec_image,
    gpu="H100",
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "PYTHONPATH": DEEPSPEC_DIR},
)
def cache(local_batch_size: int = 16):
    _run([
        "python", "scripts/data/prepare_target_cache.py",
        "--config", "config/dspark/dspark_qwen3_4b.py",
        "--train-data-path", f"{VOL}/data/regen.jsonl",
        "--output-dir", f"{VOL}/cache",
        "--local-batch-size", str(local_batch_size),
    ], cwd=DEEPSPEC_DIR)
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu="H100:4",
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "PYTHONPATH": DEEPSPEC_DIR, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)
def train(exp_name: str = "dspark_block7_qwen3_4b", epochs: int = 10, local_batch_size: int = 4):
    _run([
        "python", "train.py",
        "--config", "config/dspark/dspark_qwen3_4b.py",
        "--opts", f"exp_name={exp_name}",
        "--opts", f"train.num_train_epochs={epochs}",
        "--opts", f"train.local_batch_size={local_batch_size}",
        "--opts", "data.target_cache_path=/vol/cache",
    ], cwd=DEEPSPEC_DIR)
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu="H100",
    timeout=24 * 3600,
    volumes={VOL: volume},
    env={"HOME": HOME, "PYTHONPATH": DEEPSPEC_DIR},
)
def train_smoke(exp_name: str = "dspark_block7_qwen3_4b_smoke", epochs: int = 2, local_batch_size: int = 2):
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
    max_new_tokens: int = 1024,
    confidence_threshold: float = 0.0,
    baseline: bool = False,
):
    os.makedirs(f"{VOL}/results", exist_ok=True)
    cmd = [
        "python", "/work/scripts/timed_eval.py",
        "--target", TARGET,
        "--draft", draft,
        "--tasks", tasks,
        "--max-new-tokens", str(max_new_tokens),
        "--confidence-threshold", str(confidence_threshold),
        "--out", f"{VOL}/results/{tag}.json",
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
    max_new_tokens: int = 256,
    confidence_threshold: float = 0.0,
    baseline: bool = False,
):
    evaluate.local(
        draft=draft, tag=tag, tasks=tasks, max_new_tokens=max_new_tokens,
        confidence_threshold=confidence_threshold, baseline=baseline,
    )


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def inspect():
    volume.reload()
    for root in ("data", "cache", "results", "home/checkpoints", "home/tensorboard"):
        path = Path(VOL) / root
        if not path.exists():
            print(f"{path}: missing", flush=True)
            continue
        entries = []
        total = 0
        for item in sorted(path.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                total += size
                entries.append(f"  {item.relative_to(VOL)} {size}")
        print(f"{path}: {total / 1e9:.2f} GB across {len(entries)} files", flush=True)
        for line in entries[:60]:
            print(line, flush=True)
        if len(entries) > 60:
            print(f"  ... {len(entries) - 60} more files")


@app.function(image=util_image, volumes={VOL: volume}, timeout=600)
def fetch_result(name: str) -> bytes:
    return (Path(VOL) / "results" / f"{name}.json").read_bytes()


@app.local_entrypoint()
def main(
    stage: str,
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
    foreground: bool = False,
):
    dispatch = {
        "split": (split, dict(sample_size=sample_size, seed=seed)),
        "regen": (regen, dict(sample_size=sample_size, concurrency=concurrency)),
        "cache": (cache, dict()),
        "train": (train, dict(exp_name=exp_name, epochs=epochs, local_batch_size=local_batch_size)),
        "train-smoke": (train_smoke, dict(exp_name=exp_name, epochs=epochs, local_batch_size=local_batch_size)),
        "eval": (evaluate, dict(
            draft=draft, tag=tag, tasks=tasks, max_new_tokens=max_new_tokens,
            confidence_threshold=confidence_threshold, baseline=baseline,
        )),
        "eval-smoke": (evaluate_smoke, dict(
            draft=draft, tag=tag, tasks=tasks, max_new_tokens=max_new_tokens,
            confidence_threshold=confidence_threshold, baseline=baseline,
        )),
        "inspect": (inspect, dict()),
    }
    fn, kwargs = dispatch[stage]
    if foreground:
        print(fn.local(**kwargs))
    else:
        call = fn.spawn(**kwargs)
        print(f"spawned stage={stage} call_id={call.object_id}")