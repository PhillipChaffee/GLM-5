"""Merge completed target-cache slices into one canonical DeepSpec cache dir.

The sliced build (issue #44) writes each row-range slice with the stock
DeepSpec cache script into ``<cache-root>/slice_<start>/`` — each a complete
canonical cache (shards + samples.idx + manifest.json). This script folds the
slices into ``--output-dir`` (default: the cache root itself) so training
reads a single stock DeepSpec-format cache:

- shard files are ``Path.replace``d (renamed, never copied) into the merged
  namespace ``shard-<global_id:05d>.bin``;
- index records are rewritten (sample ids renumbered densely across slices,
  shard ids remapped) into one ``samples.idx``;
- one merged ``manifest.json`` is written and re-validated with the upstream
  loader before the script reports success.

Every step is idempotent, so a crashed merge re-run converges: a slice whose
shards were already moved is detected (source missing, expected global shard
present) and only its index records are re-merged. The expected slice set
comes from ``slice_plan.json``, written by the cache stage before the first
slice, so a changed row plan can never silently overlap an old one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO

from deepspec.data.target_cache_dataset import (
    INDEX_RECORD_SIZE,
    TARGET_CACHE_VERSION,
    build_target_cache_manifest,
    load_target_cache_manifest,
    pack_index_record,
    unpack_index_record,
    write_target_cache_manifest,
)

PLAN_FILE_NAME = "slice_plan.json"


def read_slice_plan(cache_root: Path) -> dict[str, Any]:
    """Load the slice plan the cache stage wrote before its first slice."""
    plan_path = cache_root / PLAN_FILE_NAME
    if not plan_path.exists():
        msg = (
            f"missing {plan_path}: the sliced cache stage writes the plan "
            "before its first slice; run the cache stage first"
        )
        raise RuntimeError(msg)
    plan: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    for field in ("total_rows", "slice_rows", "slices"):
        if field not in plan:
            msg = f"slice plan at {plan_path} is missing field {field!r}"
            raise RuntimeError(msg)
    return plan


def validate_slice_manifest(*, manifest: dict[str, Any], index_path: Path) -> None:
    """Structural check of one slice manifest (shard files may already be merged)."""
    required = {"version", "num_samples", "shards", "index_record_size", "hidden_size"}
    missing = sorted(required - set(manifest))
    if missing:
        msg = f"slice manifest missing fields {missing}"
        raise RuntimeError(msg)
    if int(manifest["version"]) != TARGET_CACHE_VERSION:
        msg = f"unsupported slice cache version {manifest['version']} != {TARGET_CACHE_VERSION}"
        raise RuntimeError(msg)
    if int(manifest["index_record_size"]) != INDEX_RECORD_SIZE:
        msg = f"slice index_record_size {manifest['index_record_size']} != {INDEX_RECORD_SIZE}"
        raise RuntimeError(msg)
    for expected_shard_id, shard in enumerate(manifest["shards"]):
        if int(shard["shard_id"]) != expected_shard_id:
            msg = f"slice shard ids must be contiguous from 0: expected {expected_shard_id}"
            raise RuntimeError(msg)
    if not index_path.exists():
        msg = f"missing slice index file: {index_path}"
        raise RuntimeError(msg)
    expected_size = int(manifest["num_samples"]) * INDEX_RECORD_SIZE
    if index_path.stat().st_size != expected_size:
        msg = f"slice index size {index_path.stat().st_size} != expected {expected_size}"
        raise RuntimeError(msg)


def _slice_dirs(cache_root: Path, plan: dict[str, Any]) -> list[tuple[Path, list[int]]]:
    """Return (slice_dir, [start, end]) per planned slice, verifying the set."""
    planned = {f"slice_{start:09d}": bounds for start, bounds in plan["slices"]}
    found = {
        item.name
        for item in cache_root.iterdir()
        if item.is_dir() and item.name.startswith("slice_")
    }
    unexpected = sorted(found - set(planned))
    if unexpected:
        msg = (
            f"unexpected slice dirs under {cache_root}: {unexpected}; the slice plan "
            f"expects {[f'slice_{start:09d}' for start, _ in plan['slices']]}"
        )
        raise RuntimeError(msg)
    missing = sorted(set(planned) - found)
    if missing:
        msg = (
            f"planned slices not built yet under {cache_root}: {missing}; "
            "run the cache stage until every slice is complete"
        )
        raise RuntimeError(msg)
    return [(cache_root / name, planned[name]) for name in sorted(planned)]


def _move_slice_shards(
    *, slice_dir: Path, output_dir: Path, manifest: dict[str, Any], shard_base: int
) -> int:
    """Rename a slice's shard files into the merged namespace; return next base.

    Idempotent: a shard whose source is gone must already sit at its expected
    global path, else the state is corrupt and the merge fails loudly.
    """
    for shard in manifest["shards"]:
        global_shard_id = shard_base + int(shard["shard_id"])
        source = slice_dir / shard["file_name"]
        target = output_dir / f"shard-{global_shard_id:05d}.bin"
        if source.exists():
            source.replace(target)
        elif not target.exists():
            msg = (
                f"slice shard {source} is missing and no merged shard exists at "
                f"{target}; the slice is corrupt"
            )
            raise RuntimeError(msg)
    return shard_base + len(manifest["shards"])


def _append_slice_index(
    *,
    slice_dir: Path,
    manifest: dict[str, Any],
    output_handle: BinaryIO,
    sample_base: int,
    shard_base: int,
) -> int:
    """Rewrite one slice's index records into the merged stream; return next base."""
    index_path = slice_dir / "samples.idx"
    local_bytes = index_path.read_bytes()
    if len(local_bytes) % INDEX_RECORD_SIZE != 0:
        msg = f"slice index is not a multiple of the record size: {index_path}"
        raise RuntimeError(msg)
    for local_position, offset in enumerate(range(0, len(local_bytes), INDEX_RECORD_SIZE)):
        record = unpack_index_record(local_bytes, offset)
        if int(record["sample_id"]) != local_position:
            msg = (
                f"slice index is not densely ordered: {index_path} record "
                f"{local_position} carries sample_id {record['sample_id']}"
            )
            raise RuntimeError(msg)
        record["sample_id"] = sample_base + local_position
        record["shard_id"] = shard_base + int(record["shard_id"])
        output_handle.write(pack_index_record(**record))
    return sample_base + int(manifest["num_samples"])


def merge_cache_slices(*, cache_root: Path, output_dir: Path) -> dict[str, int]:
    """Fold every planned slice under ``cache_root`` into one canonical cache."""
    plan = read_slice_plan(cache_root)
    slices = _slice_dirs(cache_root, plan)
    first_manifest: dict[str, Any] = json.loads(
        (slices[0][0] / "manifest.json").read_text(encoding="utf-8")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_tmp_path = output_dir / "samples.idx.tmp"
    shards: list[dict[str, Any]] = []
    sample_base = 0
    shard_base = 0
    with index_tmp_path.open("wb") as output_handle:
        for slice_dir, _bounds in slices:
            manifest: dict[str, Any] = json.loads(
                (slice_dir / "manifest.json").read_text(encoding="utf-8")
            )
            validate_slice_manifest(manifest=manifest, index_path=slice_dir / "samples.idx")
            local_shard_base = shard_base
            shard_base = _move_slice_shards(
                slice_dir=slice_dir,
                output_dir=output_dir,
                manifest=manifest,
                shard_base=local_shard_base,
            )
            sample_base = _append_slice_index(
                slice_dir=slice_dir,
                manifest=manifest,
                output_handle=output_handle,
                sample_base=sample_base,
                shard_base=local_shard_base,
            )
            for shard in manifest["shards"]:
                global_shard_id = local_shard_base + int(shard["shard_id"])
                shards.append(
                    {"shard_id": global_shard_id, "file_name": f"shard-{global_shard_id:05d}.bin"}
                )
        output_handle.flush()
        os.fsync(output_handle.fileno())
    index_tmp_path.replace(output_dir / "samples.idx")

    reserved = {
        "version",
        "num_samples",
        "num_shards",
        "shards",
        "index_record_size",
        "target_layer_ids",
        "hidden_size",
    }
    extra_fields: dict[str, Any] = {
        key: value for key, value in first_manifest.items() if key not in reserved
    }
    extra_fields["merged_from"] = {
        "num_slices": len(slices),
        "slice_rows": int(plan["slice_rows"]),
        "total_rows": int(plan["total_rows"]),
    }
    merged = build_target_cache_manifest(
        num_samples=sample_base,
        shards=shards,
        target_layer_ids=first_manifest["target_layer_ids"],
        hidden_size=first_manifest["hidden_size"],
        extra_fields=extra_fields,
    )
    write_target_cache_manifest(output_dir=str(output_dir), manifest=merged)

    validated = load_target_cache_manifest(str(output_dir))
    total_shard_bytes = sum(
        (output_dir / shard["file_name"]).stat().st_size for shard in validated["shards"]
    )
    return {
        "num_slices": len(slices),
        "num_samples": int(validated["num_samples"]),
        "num_shards": int(validated["num_shards"]),
        "total_shard_bytes": total_shard_bytes,
    }


def main() -> None:
    """CLI entry: merge a sliced cache build into one canonical cache dir."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", required=True, help="Directory holding slice_<start> dirs")
    parser.add_argument(
        "--output-dir", default=None, help="Merged cache dir (default: the cache root itself)"
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.cache_root)
    summary = merge_cache_slices(cache_root=Path(args.cache_root), output_dir=output_dir)
    sys.stdout.write(
        f"Merged {summary['num_slices']} slices into {output_dir}: "
        f"{summary['num_samples']} samples across {summary['num_shards']} shards "
        f"({summary['total_shard_bytes'] / 1e9:.1f} GB).\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    if Path(".git").exists():
        from deepspec.utils import get_git_sha

        sys.stdout.write(f"git: {get_git_sha()}\n")
    main()
