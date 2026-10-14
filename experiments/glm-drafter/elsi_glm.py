"""ELSI layer-selection stages for the GLM drafter work (issue 51) on Modal.

Stages: stats (CPU, statistics over the cached 7 streams), capture (GPU, the
all-45-layer + post-norm + predictive-entropy capture), select (CPU, the ELSI
and ELSI-C beam searches over the captured 46 columns), fetch (pull a small
result file locally). Mirrors glm_drafter.py's image and dispatch patterns;
the app name stays independent so the main pipeline app is untouched.
"""

from __future__ import annotations

import sys
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
    HOME,
    VOL,
)
from lib.modal_pipeline import (
    run as _run,
)

VOLUME_NAME = "glm-drafter-run"
TARGET = "zai-org/GLM-5.3-Flash"
LOCAL_DIR = Path(__file__).parent.resolve()
OVERLAY_DIR = LOCAL_DIR / "deepspec_glm"
# Exact B200 preferred but capacity-starved (2026-10-13: queued >30 min with no
# schedule); H200:4 (sm_90) is fully supported by the deepspec torch 2.9.1
# image, holds the 306-GiB FP8 target comfortably, and bills below B200 rates.
CAPTURE_GPU = "H200:4"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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
        # GLM-5.3-Flash (glm5_next) needs newer transformers than the upstream
        # pin (see glm_drafter.py for the same proven chain)
        "pip install 'transformers==5.19.0'",
        "pip install 'accelerate>=1.11.0'",
        "pip install 'kernels==0.17.0'",
        "cp -r /overlay/. /deepspec/",
    )
)

deepspec_image = _deepspec_layers.env({"PYTHONPATH": DEEPSPEC_DIR})

hf_secret = modal.Secret.from_name("hf-token")

app = modal.App("elsi-glm")


@app.function(
    image=deepspec_image,
    volumes={VOL: volume},
    timeout=4 * 3600,
    cpu=8,
    memory=32768,
)
def stats() -> None:
    """Stage 1: entropy/CKA statistics over the cached 7 streams (CPU only)."""
    volume.reload()
    _run(
        [
            "python",
            "scripts/data/elsi_analyze.py",
            "stage1-cache",
            "--cache-dir",
            f"{VOL}/cache",
            "--out",
            f"{VOL}/elsi/stage1.json",
        ],
        cwd=DEEPSPEC_DIR,
    )
    volume.commit()


@app.function(
    image=deepspec_image,
    gpu=CAPTURE_GPU,
    timeout=12 * 3600,
    volumes={VOL: volume},
    env={
        "HOME": HOME,
        "PYTHONPATH": DEEPSPEC_DIR,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_CACHE": f"{HOME}/.cache/huggingface",
    },
    secrets=[hf_secret],
)
def capture(
    num_rows: int = 250,
    max_length: int = 2048,
    rho: float = 0.3,
    local_batch_size: int = 2,
    output_dir: str = "/vol/elsi/capture",
) -> None:
    """Stage 2: the all-45-layer capture pass with predictive-entropy pooling."""
    _run(
        [
            "python",
            "scripts/data/elsi_capture_glm.py",
            "--train-data-path",
            f"{VOL}/data/regen.jsonl",
            "--output-dir",
            output_dir,
            "--target-model",
            TARGET,
            "--num-rows",
            str(num_rows),
            "--max-length",
            str(max_length),
            "--rho",
            str(rho),
            "--local-batch-size",
            str(local_batch_size),
        ],
        cwd=DEEPSPEC_DIR,
    )
    volume.commit()


@app.function(
    image=deepspec_image,
    volumes={VOL: volume},
    timeout=4 * 3600,
    cpu=8,
    memory=32768,
)
def select(*, drop_post_norm: bool = False, out_key: str = "select.json") -> None:
    """Stage 3: statistics and ELSI/ELSI-C beam searches over the capture.

    ``drop_post_norm`` runs the all-raw pool (the post-final-norm column
    excluded) and ``out_key`` names the report file on /vol/elsi, so the
    recorded mixed run's select.json is never clobbered.
    """
    volume.reload()
    flags = [
        "python",
        "scripts/data/elsi_analyze.py",
        "elsi-select",
        "--capture-dir",
        f"{VOL}/elsi/capture",
        "--out",
        f"{VOL}/elsi/{out_key}",
    ]
    if drop_post_norm:
        flags.append("--drop-post-norm")
    _run(flags, cwd=DEEPSPEC_DIR)
    volume.commit()


@app.function(image=deepspec_image, volumes={VOL: volume}, timeout=600)
def fetch(key: str) -> bytes:
    """Read one result file from /vol/elsi for local transport."""
    path = Path(VOL) / "elsi" / key
    if not path.exists():
        msg = f"missing on volume: {path}"
        raise FileNotFoundError(msg)
    return path.read_bytes()


def _build_dispatch(
    *,
    num_rows: int,
    max_length: int,
    rho: float,
    local_batch_size: int,
    output_dir: str,
    key: str,
    drop_post_norm: bool,
    out_key: str,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Map every stage name to its function and flag-derived parameters."""
    return {
        "stats": (stats, {}),
        "capture": (
            capture,
            {
                "num_rows": num_rows,
                "max_length": max_length,
                "rho": rho,
                "local_batch_size": local_batch_size,
                "output_dir": output_dir,
            },
        ),
        "select": (
            select,
            {"drop_post_norm": drop_post_norm, "out_key": out_key},
        ),
        "fetch": (fetch, {"key": key}),
    }


@app.local_entrypoint()
def main(
    stage: str,
    *,
    num_rows: int = 250,
    max_length: int = 2048,
    rho: float = 0.3,
    local_batch_size: int = 2,
    output_dir: str = "/vol/elsi/capture",
    key: str = "select.json",
    drop_post_norm: bool = False,
    out_key: str = "select.json",
    foreground: bool = False,
) -> None:
    """Dispatch one stage; every flag maps to a stage parameter."""
    dispatch = _build_dispatch(
        num_rows=num_rows,
        max_length=max_length,
        rho=rho,
        local_batch_size=local_batch_size,
        output_dir=output_dir,
        key=key,
        drop_post_norm=drop_post_norm,
        out_key=out_key,
    )
    fn, kwargs = dispatch[stage]
    if foreground:
        # blocking remote call: the stage runs on Modal with streaming logs
        result = fn.remote(**kwargs)
        if result is not None:
            sys.stdout.write(f"{result!r}\n")
    else:
        call = fn.spawn(**kwargs)
        sys.stdout.write(f"spawned stage={stage} call_id={call.object_id}\n")
