"""Budget-probe sweep for the target-hidden-state capture pass (issue #57).

Measures the OOM boundary the design doc calls "probe, don't derive": batch
rows padded to the collator max_length (4,096) through the REAL production
path (the 320B target loaded ``device_map=auto`` across B200:4, hooks pinning
7 hidden-state streams, ``logits_to_keep=1``, expandable segments), stepping
batch sizes ascending with a hard stop at the first cuda trouble. The last
size that passed is the safe capacity ceiling; each size maps to an
equivalent token budget of rows x max_length, and admission in the packer
bounds each pack's padded peak by that budget.

Run detached per the house recipe (a dead local client kills a live app):
``nohup uvx modal run --detach experiments/glm-drafter/budget_probe.py
> <log> 2>&1 &`` — poll the log with short sleeps until first life, then
lengthen. The transcript lands on the volume under results/ and the log.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.modal_pipeline import DEEPSPEC_DIR, DEEPSPEC_REPO, DEEPSPEC_SHA, VOL

VOLUME_NAME = "glm-drafter-run"
TARGET = "zai-org/GLM-5.3-Flash"
OVERLAY_DIR = Path(__file__).resolve().parent / "deepspec_glm"
LOCAL_DIR = Path(__file__).resolve().parent

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

PROBED_PASS_BUDGET = 8_192
EXPECTED_GPU_COUNT = 4

_deepspec_layers = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("huggingface_hub")
    .env({"HF_HUB_CACHE": "/hf"})
    .add_local_dir(OVERLAY_DIR, "/overlay", copy=True)
    .add_local_dir(LOCAL_DIR / "scripts", "/work/scripts", copy=True)
    .add_local_dir(LOCAL_DIR / "scripts", "/deepspec/scripts", copy=True)
    .add_local_dir(LOCAL_DIR.parent / "lib", "/root/lib", copy=True)
    .run_commands(
        f"git init {DEEPSPEC_DIR}",
        f"git -C {DEEPSPEC_DIR} remote add origin {DEEPSPEC_REPO}",
        f"git -C {DEEPSPEC_DIR} fetch --depth 1 origin {DEEPSPEC_SHA}",
        f"git -C {DEEPSPEC_DIR} checkout FETCH_HEAD",
        "pip install -r /deepspec/requirements.txt",
        "pip install 'transformers==5.19.0'",
        "pip install 'accelerate>=1.11.0'",
        "pip install 'kernels==0.17.0'",
        "cp -r /overlay/. /deepspec/",
    )
)

deepspec_image = _deepspec_layers.env({"PYTHONPATH": DEEPSPEC_DIR})
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App("glm-budget-probe")


def _hardware_block(device_count: int) -> dict[str, Any]:
    """Record the hardware the run actually got (the boundary is per-card)."""
    import torch

    return {
        "gpu_count": device_count,
        "gpus": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "total_memory_gb": [
            round(torch.cuda.get_device_properties(index).total_memory / 2**30, 1)
            for index in range(device_count)
        ],
    }


def _measure_one_row(
    *,
    run_target_forward_with_hooks: Callable[..., Any],
    target_model: nn.Module,
    target_layer_ids: list[int],
    size: int,
    max_length: int,
    vocab_size: int,
    boundary_message: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Forward one synthetic batch and build its step's transcript record.

    A torch-CUDA OOM is recorded with ``status: "oom"``; any other
    RuntimeError — a shape/hook/kernel bug, not a memory fact — is recorded
    ``status: "error"``, so the boundary mislabeling from the review can
    never claim a bug was an OOM. Either way this boot stops the ladder:
    a failed-cx state can taint later rows.
    """
    import gc

    import torch

    input_ids = torch.randint(0, vocab_size, (size, max_length), device=target_model.device)
    attention_mask = torch.ones(size, max_length, dtype=torch.long, device=target_model.device)
    record: dict[str, Any] = {
        "batch_rows": size,
        "equivalent_budget": size * max_length,
    }
    try:
        start = time.perf_counter()
        result = run_target_forward_with_hooks(
            target_model=target_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_layer_ids=target_layer_ids,
        )
        elapsed = time.perf_counter() - start
        record.update(
            {
                "status": "ok",
                "seconds": round(elapsed, 2),
                "tok_per_s": round(size * max_length / elapsed),
                "padded_tok_per_s": round(size * max_length / elapsed),
                "states_gb": round(
                    result.target_hidden_states.element_size()
                    * result.target_hidden_states.numel()
                    / 2**30,
                    2,
                ),
                "reserved_gb": [
                    round(torch.cuda.memory_reserved(index) / 2**30, 1)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
        del result
    except torch.cuda.OutOfMemoryError as exc:
        boundary_message = f"{type(exc).__name__}: {str(exc)[:200]}"
        record.update({"status": "oom", "error_first_200": boundary_message})
    except RuntimeError as exc:
        boundary_message = f"{type(exc).__name__}: {str(exc)[:200]}"
        record.update({"status": "error", "error_first_200": boundary_message})
    del input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    return record, boundary_message


def sweep(max_rows: int = 48) -> dict[str, Any]:
    """Sweep batch-row capacity at synthetic max_length, ascending, hard stop."""
    import importlib.util

    import torch
    from transformers import AutoModel, AutoTokenizer

    # Load the capture script by path: its own sibling-import of
    # token_budget_batcher resolves at runtime (the pipeline's direct
    # execution gets this for free via sys.path[0]).
    sys.path.insert(0, "/deepspec/scripts/data")
    spec = importlib.util.spec_from_file_location(
        "prepare_target_cache_glm",
        "/deepspec/scripts/data/prepare_target_cache_glm.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - fixed path on the image
        loader_error = "capture script not found at its image path"
        raise RuntimeError(loader_error)
    prepare_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_module)
    run_target_forward_with_hooks = prepare_module.run_target_forward_with_hooks

    max_length = 4_096
    target_layer_ids = [1, 12, 23, 34, 42, 43, 44]
    row_sizes = [size for size in (2, 4, 8, 12, 16, 24, 32, 48) if size <= max_rows]
    transcript: list[dict[str, Any]] = []
    boundary_message = None
    gpu_count = int(torch.cuda.device_count())
    if gpu_count != EXPECTED_GPU_COUNT:
        hardware_error = (
            f"the recorded boundary is a B200:4 quantity; this run got {gpu_count} GPUs"
        )
        raise RuntimeError(hardware_error)
    hardware_block = _hardware_block(gpu_count)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    print(json_line({"stage": "gpu-verified", **hardware_block}), flush=True)
    target_model = AutoModel.from_pretrained(
        TARGET,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    vocab_size = int(AutoTokenizer.from_pretrained(TARGET).vocab_size)
    print(json_line({"stage": "model-loaded", "vocab": vocab_size}), flush=True)
    for size in row_sizes:
        if boundary_message is not None:
            transcript.append(
                {
                    "batch_rows": size,
                    "equivalent_budget": size * max_length,
                    "status": "skipped",
                    "note": "cuda boundary reached at a smaller size",
                }
            )
            print(json_line(transcript[-1]), flush=True)
            continue
        record, boundary_message = _measure_one_row(
            run_target_forward_with_hooks=run_target_forward_with_hooks,
            target_model=target_model,
            target_layer_ids=target_layer_ids,
            size=size,
            max_length=max_length,
            vocab_size=vocab_size,
            boundary_message=boundary_message,
        )
        transcript.append(record)
        print(json_line(record), flush=True)
    first_passing = max(
        (record["equivalent_budget"] for record in transcript if record["status"] == "ok"),
        default=None,
    )
    stopping = [
        record["equivalent_budget"] for record in transcript if record["status"] in ("oom", "error")
    ]
    first_stopping = stopping[0] if stopping else None
    summary = {
        "max_rows": max_rows,
        "started_at": started_at,
        **hardware_block,
        "max_length": max_length,
        "sweep": transcript,
        "first_passing_budget": first_passing,
        "first_stopping_budget": first_stopping,
    }
    out_path = Path(VOL) / "results" / "budget-probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json_line({"transcript": str(out_path)}), flush=True)
    volume.commit()
    return summary


def json_line(payload: dict[str, Any]) -> str:
    """Serialize one transcript record for the streamed Modal log."""
    return json.dumps(payload)


@app.local_entrypoint()
def main(max_rows: int = 48) -> None:
    """Run the probe sweep and print its summary."""
    # typing.cast Any: modal's decorator returns a SyncEntrypoint wrapper
    remote_fn: Any = cast("Any", sweep)
    summary = remote_fn.remote(max_rows=max_rows)
    print("probe summary:")
    print(json.dumps(summary, indent=2))
