"""Shared Modal pipeline plumbing for the experiment apps.

Extracted from the two experiment Modal apps so the volume plumbing
(logging, process runner, volume inspection, HF restore/push) lives in one
place; each app keeps its own stage decorators, constants, and dispatch.
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
import threading
import time
import zipfile
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
    cache_token_budget: int | None = None,
    sample_start: int | None = None,
    sample_end: int | None = None,
) -> list[str]:
    """Build the target-cache command shared by the pipeline apps.

    ``cache_token_budget`` turns the capture pass's fixed-count batching
    into serving-style admission control (length-sorted, whole-conversation
    packs bounded by the token budget with ``local_batch_size`` as the
    count cap); unset leaves the proven batch-2 floor untouched.
    ``sample_start``/``sample_end`` bound the build to rows ``[start, end)``
    of the training jsonl (the resumable slice build); both must be given
    together or both omitted.
    """
    if (sample_start is None) != (sample_end is None):
        msg = "sample_start and sample_end must be given together"
        raise ValueError(msg)
    command = [
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
    if cache_token_budget is not None:
        command += [
            "--cache-token-budget",
            str(cache_token_budget),
        ]
    if sample_start is not None:
        command += [
            "--sample-start",
            str(sample_start),
            "--sample-end",
            str(sample_end),
        ]
    return command


def count_jsonl_lines(path: str) -> int:
    """Count lines of a jsonl file (the slice plan's row total)."""
    with Path(path).open("rb") as handle:
        return sum(1 for _ in handle)


def plan_cache_slices(total_rows: int, slice_rows: int) -> list[tuple[int, int]]:
    """Split ``total_rows`` into contiguous ``[start, end)`` slice ranges."""
    if total_rows < 0:
        msg = f"total_rows must be non-negative, got {total_rows}"
        raise ValueError(msg)
    if slice_rows <= 0:
        msg = f"slice_rows must be positive, got {slice_rows}"
        raise ValueError(msg)
    return [
        (start, min(start + slice_rows, total_rows)) for start in range(0, total_rows, slice_rows)
    ]


def slice_cache_complete(cache_dir: str) -> bool:
    """A slice dir is complete iff its canonical manifest loads and validates.

    Runs only where the deepspec package is importable (the deepspec image
    and the experiment test env); elsewhere the ImportError propagates —
    the caller must not treat an uncheckable slice as complete.
    """
    from deepspec.data.target_cache_dataset import load_target_cache_manifest

    try:
        load_target_cache_manifest(cache_dir)
    except Exception:  # noqa: BLE001 - any invalid state means "not complete"
        return False
    return True


SLICE_PLAN_FILE_NAME = "slice_plan.json"


def write_slice_plan(cache_root: str, total_rows: int, slice_rows: int) -> list[tuple[int, int]]:
    """Write the cache root's slice plan atomically and return it.

    The plan freezes the row set the merged cache will cover: a later run
    with a different ``slice_rows`` or a grown ``total_rows`` fails loudly
    instead of silently overlapping already-built slices.
    """
    slices = plan_cache_slices(total_rows, slice_rows)
    plan = {
        "total_rows": total_rows,
        "slice_rows": slice_rows,
        "slices": [list(bounds) for bounds in slices],
    }
    plan_path = Path(cache_root) / SLICE_PLAN_FILE_NAME
    if plan_path.exists():
        existing: dict[str, object] = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            msg = (
                f"existing slice plan at {plan_path} covers total_rows="
                f"{existing.get('total_rows')} with slice_rows={existing.get('slice_rows')}; "
                f"refusing to re-plan with total_rows={total_rows}, slice_rows={slice_rows} — "
                f"pass --slice-rows {existing.get('slice_rows')} or delete the cache root"
            )
            raise RuntimeError(msg)
        return slices
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{plan_path}.tmp")
    tmp_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    tmp_path.replace(plan_path)
    return slices
>>>>>>> origin/main


def build_train_cmd(
    *,
    config: str,
    exp_name: str,
    epochs: int,
    local_batch_size: int,
    head_type: str,
    cache_path: str,
    max_steps: int,
    torch_compile: bool | None = None,
    checkpointing_steps: int | None = None,
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
    if torch_compile is not None:
        cmd.extend(["--opts", f"train.torch_compile={torch_compile}"])
    if checkpointing_steps is not None:
        cmd.extend(["--opts", f"logging.checkpointing_steps={checkpointing_steps}"])
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


def _cache_shard_jobs(vol_path: Path) -> list[tuple[Path, str]]:
    """Cache-shard upload jobs from a volume; the caller decides to include them."""
    return [
        (item, f"cache/{item.name}")
        for item in sorted((vol_path / "cache").iterdir())
        if item.is_file()
    ]


def build_push_jobs(
    vol: str, exp_name: str = "", *, include_cache: bool = False
) -> list[tuple[Path, str]]:
    """Build the volume upload list: regen data, final checkpoint, opt-in cache.

    The cache arm is opt-in: the #22 decision names corpus text + checkpoints
    as this effort's only durable artifacts, while the plan-relevant cache is
    a hold-then-delete derived artifact (its upload cost and the ~$2-4k
    rebuild lever are scoped in the push-hf ticket #70); ``include_cache=
    True`` exists for pipelines whose cache *is* durable (the proxy run kept
    its 611 GB cache for the A/B).
    The checkpoint arm resolves ``home/checkpoints/deepspec/<exp_name>`` and
    uploads only the files of its last ``step_*`` directory; an unset or
    unmatched ``exp_name`` yields no checkpoint jobs and logs why, so a
    renamed experiment can never silently upload nothing.
    """
    vol_path = Path(vol)
    jobs: list[tuple[Path, str]] = _cache_shard_jobs(vol_path) if include_cache else []
    jobs.extend(
        (item, f"data/{item.name}")
        for item in sorted((vol_path / "data").iterdir())
        if item.is_file()
    )
    ckpt_root = vol_path / "home/checkpoints/deepspec"
    ckpt_dir = ckpt_root / exp_name if exp_name else None
    if ckpt_dir is None:
        log("push jobs: no exp_name given; checkpoint arm skipped")
        return jobs
    if not ckpt_dir.is_dir():
        available = sorted(item.name for item in ckpt_root.iterdir()) if ckpt_root.is_dir() else []
        log(f"push jobs: no checkpoint dir at {ckpt_dir}; available: {available or 'none'}")
        return jobs
    step_dirs = sorted(ckpt_dir.glob("step_*"))
    if not step_dirs:
        log(f"push jobs: no step_* directories under {ckpt_dir}; checkpoint arm skipped")
        return jobs
    final_step = step_dirs[-1]
    jobs.extend(
        (final_step / name, f"checkpoints/{exp_name}/{name}")
        for name in ("model.safetensors", "config.json", "train_config.py")
        if (final_step / name).exists()
    )
    return jobs


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


STEP_DIR_PATTERN = re.compile(r"^step_(\d+)$")
STEP_LATEST_NAME = "step_latest"


def os_symlink(target: Path, link: Path) -> None:
    """Create ``link -> target``, replacing a stale non-link path if present."""
    if link.exists() and not link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def checkpoint_dir_is_intact(step_dir: Path) -> bool:
    """Structurally validate a step dir's checkpoint files.

    A kill mid-save can commit a truncated ``.pt`` (zip without its trailing
    central directory — the drill-D load failure) or a truncated
    ``model.safetensors``; both must be rejected before a resume points at
    the dir. Checks only layout, not content: the central directory read for
    zip files, the 8-byte header length for safetensors.
    """
    for path in sorted(step_dir.iterdir()):
        if path.name.endswith(".pt") and path.is_file():
            try:
                with zipfile.ZipFile(path) as archive:
                    archive.namelist()
            except (OSError, zipfile.BadZipFile):
                return False
        elif path.name == "model.safetensors" and path.is_file():
            try:
                with path.open("rb") as handle:
                    (header_len,) = struct.unpack("<Q", handle.read(8))
            except (OSError, struct.error):
                return False
            if 8 + header_len >= path.stat().st_size:
                return False
    return True


def heal_latest_checkpoint(checkpoint_root: Path) -> str | None:
    """Point a missing ``step_latest`` at the newest intact step dir.

    A killed container lost the symlink while its checkpoint files survived
    (measured twice in the #44 drills); recovery would otherwise restart the
    train from scratch. Step dirs left truncated by a mid-save kill are
    skipped newest-first; an existing link or a checkpoint-less root is a
    no-op.
    """
    if not checkpoint_root.is_dir():
        return None
    steps = sorted(
        (int(match.group(1)), item)
        for item in checkpoint_root.iterdir()
        if item.is_dir() and (match := STEP_DIR_PATTERN.match(item.name))
    )
    link = checkpoint_root / STEP_LATEST_NAME
    if link.is_symlink():
        return None
    skipped: list[str] = []
    for _step_number, step_dir in reversed(steps):
        if checkpoint_dir_is_intact(step_dir):
            target = step_dir.resolve()
            os_symlink(target, link)
            note = f"restored {link} -> {target} after an earlier kill left it missing"
            if skipped:
                note += f" (intact check skipped truncated dirs: {skipped})"
            return note
        skipped.append(step_dir.name)
    return None


def _step_state(checkpoint_root: Path) -> tuple[int, str] | None:
    """(newest step number, step_latest target) as seen on disk."""
    if not checkpoint_root.is_dir():
        return None
    steps = [
        int(match.group(1))
        for item in checkpoint_root.iterdir()
        if item.is_dir() and (match := STEP_DIR_PATTERN.match(item.name))
    ]
    try:
        latest = str((checkpoint_root / STEP_LATEST_NAME).readlink())
    except OSError:
        latest = ""
    if not steps and not latest:
        return None
    return max(steps, default=0), latest


def spawn_checkpoint_committer(
    *,
    checkpoint_root: Path,
    commit: Callable[[], None],
    poll_seconds: float = 15.0,
) -> tuple[threading.Thread, threading.Event]:
    """Commit the volume each time the trainer publishes a new checkpoint.

    Background commits carried the checkpoint files after a kill but not the
    step_latest symlink (the #44 drill finding), so the stage commits
    explicitly as DeepSpec's checkpoint state advances. The watcher watches
    the newest ``step_N`` directory and the ``step_latest`` target; nothing
    dirty happens between saves, so commits stay off the training critical
    path. The caller stops via the returned event and joins the thread.
    """
    stop_event = threading.Event()

    def watch() -> None:
        local_seen: tuple[int, str] | None = None
        while not stop_event.wait(poll_seconds):
            state = _step_state(checkpoint_root)
            if state is not None and state != local_seen:
                commit()
                local_seen = state

    watcher = threading.Thread(target=watch, name="train-checkpoint-committer", daemon=True)
    watcher.start()
    return watcher, stop_event
