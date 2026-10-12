"""Shared Modal pipeline plumbing for the experiment apps.

Extracted from the two experiment Modal apps so the volume plumbing
(logging, process runner, volume inspection, HF restore/push) lives in one
place; each app keeps its own stage decorators, constants, and dispatch.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def log(message: str) -> None:
    """Print a line immediately (the print-free lint gate owns this shape)."""
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def run(cmd: list[str], cwd: str | None = None) -> None:
    """Run a command, failing loudly on a nonzero exit."""
    log("+ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        msg = f"command failed rc={exc.returncode}: {' '.join(cmd)}"
        raise RuntimeError(msg) from exc


def inspect_volume(
    vol: str,
    roots: Sequence[str],
    max_listed_files: int,
) -> None:
    """Print a size summary of the requested directories on the volume."""
    for root in roots:
        path = Path(vol) / root
        if not path.exists():
            log(f"{path}: missing")
            continue
        entries = []
        total = 0
        for item in sorted(path.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                total += size
                entries.append(f"  {item.relative_to(vol)} {size}")
        log(f"{path}: {total / 1e9:.2f} GB across {len(entries)} files")
        for line in entries[:max_listed_files]:
            log(line)
        if len(entries) > max_listed_files:
            log(f"  ... {len(entries) - max_listed_files} more files")


def fetch_result_bytes(vol: str, name: str) -> bytes:
    """Read one result json from the volume."""
    return (Path(vol) / "results" / f"{name}.json").read_bytes()


def restore_from_hf(
    vol: str,
    repo_id: str,
    patterns: Sequence[str],
    commit: Callable[[], None],
) -> None:
    """Restore named patterns from an HF dataset repo onto the volume.

    Downloads with a small worker pool (3) and commits after each file: a
    crashed run resumes where it stopped, and the gentle concurrency avoids
    the parallel-xet contention that killed the first attempt (14 streams,
    container heartbeat deadlines). Commits run on the main thread only.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfApi, hf_hub_download

    t0 = time.time()
    api = HfApi()
    files = [
        name
        for name in api.list_repo_files(repo_id, repo_type="dataset")
        if name.startswith(tuple(patterns))
    ]

    def fetch(name: str) -> str:
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name, local_dir=vol)
        return name

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, name) for name in files]
        for idx, future in enumerate(as_completed(futures), 1):
            name = future.result()
            commit()
            log(f"[{idx}/{len(files)}] {name} done in {(time.time() - t0) / 60:.1f} min")
    log(f"restore from {repo_id} done in {(time.time() - t0) / 60:.1f} min")


DEEPSPEC_REPO = "https://github.com/deepseek-ai/DeepSpec"
DEEPSPEC_SHA = "005e03b81cec38b7da6399833d609ee89a2587f2"
DEEPSPEC_DIR = "/deepspec"
VOL = "/vol"
HOME = "/vol/home"
VOLUME_ROOTS = ("data", "cache", "results", "home/checkpoints", "home/tensorboard")
DEFAULT_TASKS = "gsm8k:300,math500:300,humaneval:164,mbpp:256,mt-bench:80,alpaca:300"
MAX_LISTED_FILES = 60  # inspect lists at most this many files per directory


def serve_and_run_client(
    *,
    server_cmd: list[str],
    health_url: str,
    health_deadline_s: float,
    server_name: str,
    client_cmd: list[str],
) -> float:
    """Boot an OpenAI-compatible server, wait for health, run the client, stop it."""
    import urllib.request

    # The health URL is repo-authored (http://127.0.0.1:<port>/health for
    # the in-container server); no user input reaches it.
    t0 = time.time()
    server = subprocess.Popen(server_cmd)
    while True:
        if server.poll() is not None:
            msg = f"{server_name} server exited during startup"
            raise RuntimeError(msg)
        try:
            urllib.request.urlopen(health_url, timeout=5)  # noqa: S310
            break
        except OSError as exc:
            if time.time() > t0 + health_deadline_s:
                server.terminate()
                wait_min = health_deadline_s / 60
                msg = f"{server_name} server did not become healthy in {wait_min:.0f} minutes"
                raise RuntimeError(msg) from exc
            time.sleep(10)
    log(f"{server_name} healthy after {time.time() - t0:.0f}s")
    try:
        run(client_cmd)
    finally:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
    return time.time() - t0


def build_cache_cmd(
    *,
    config: str,
    train_data_path: str,
    output_dir: str,
    local_batch_size: int,
    cache_script: str,
) -> list[str]:
    """Build the target-cache command shared by the pipeline apps."""
    return [
        "python",
        cache_script,
        "--config",
        config,
        "--train-data-path",
        train_data_path,
        "--output-dir",
        output_dir,
        "--local-batch-size",
        str(local_batch_size),
    ]


def build_train_cmd(
    *,
    config: str,
    exp_name: str,
    epochs: int,
    local_batch_size: int,
    head_type: str,
    cache_path: str,
    max_steps: int,
    train_script: str = "train.py",
) -> list[str]:
    """Build the drafter-training command shared by the pipeline apps."""
    cmd = [
        "python",
        train_script,
        "--config",
        config,
        "--opts",
        f"exp_name={exp_name}",
        "--opts",
        f"train.num_train_epochs={epochs}",
        "--opts",
        f"train.local_batch_size={local_batch_size}",
        "--opts",
        f"model.markov_head_type={head_type}",
        "--opts",
        f"data.target_cache_path={cache_path}",
    ]
    if max_steps:
        cmd.extend(["--opts", f"train.max_train_steps={max_steps}"])
    return cmd


def build_eval_cmd(
    *,
    script: str,
    target: str,
    draft: str,
    tasks: str,
    max_new_tokens: int,
    confidence_threshold: float,
    out_path: str,
    baseline: bool,
) -> list[str]:
    """Build the timed-eval command shared by the pipeline apps."""
    cmd = [
        "python",
        script,
        "--target",
        target,
        "--draft",
        draft,
        "--tasks",
        tasks,
        "--max-new-tokens",
        str(max_new_tokens),
        "--confidence-threshold",
        str(confidence_threshold),
        "--out",
        out_path,
    ]
    if baseline:
        cmd.append("--baseline")
    return cmd


def push_jobs_to_hf(
    repo_id: str,
    jobs: Sequence[tuple[Path, str]],
    default_repo_suffix: str,
) -> None:
    """Upload a fixed list of (source, destination) files to an HF dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo = repo_id or f"{api.whoami()['name']}/{default_repo_suffix}"
    api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)
    t0 = time.time()
    for idx, (src, dst) in enumerate(jobs, 1):
        size = src.stat().st_size
        log(f"[{idx}/{len(jobs)}] {dst} {size / 1e9:.1f} GB uploading")
        f0 = time.time()
        api.upload_file(
            path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo, repo_type="dataset"
        )
        log(f"[{idx}/{len(jobs)}] {dst} done in {(time.time() - f0) / 60:.1f} min")
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min -> https://huggingface.co/datasets/{repo}")
