"""Unit tests for the sliced, resumable cache build and its merge (issue #44).

Builds tiny canonical slice dirs with the upstream DeepSpec writer stack
(real shard files, 56-byte index records, manifests), then drives the merge
script over them — including its crash states — and reads the merged cache
back through the upstream CacheDataset.
"""

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    CacheDataset,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    load_target_cache_manifest,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)

TESTS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TESTS_DIR.parent
_LIB = runpy.run_path(str(EXPERIMENT_DIR.parent / "lib" / "modal_pipeline.py"))
plan_cache_slices = _LIB["plan_cache_slices"]
slice_cache_complete = _LIB["slice_cache_complete"]
write_slice_plan = _LIB["write_slice_plan"]


def _load(path: Path) -> ModuleType:
    # a script run as a file resolves sibling imports via its own directory
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None, f"cannot load {path}"
    assert spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge_slices = _load(EXPERIMENT_DIR / "scripts/data/merge_cache_slices.py")
cache_script = _load(EXPERIMENT_DIR / "scripts/data/prepare_target_cache_glm.py")
regen_script = _load(EXPERIMENT_DIR / "scripts/data/generate_train_data_glm.py")

TARGET_LAYERS = [1, 12, 23, 34, 42, 43, 44]
HIDDEN = 8
SHARD_BYTES = 512  # small enough that a few samples roll into several shards


def _build_slice(
    slice_dir: Path,
    *,
    sample_start: int,
    seq_lens: list[int],
    extra_fields: dict[str, object] | None = None,
) -> None:
    """Write one canonical slice dir exactly as the stock cache script leaves it."""
    slice_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = slice_dir / "_tmp" / "rank_0"
    rank_dir.mkdir(parents=True, exist_ok=True)
    writer = AsyncTargetCacheWriter(rank_dir=str(rank_dir), max_shard_bytes=SHARD_BYTES)
    try:
        for seq_len in seq_lens:
            writer.write_sample(
                input_ids=torch.arange(seq_len, dtype=torch.int32),
                attention_mask=torch.ones(seq_len, dtype=torch.uint8),
                loss_mask=torch.ones(seq_len, dtype=torch.uint8),
                target_hidden_states=torch.zeros(seq_len, len(TARGET_LAYERS) * HIDDEN),
                target_last_hidden_states=torch.zeros(seq_len, HIDDEN),
            )
    finally:
        writer.close()
    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=sample_start,
        source_sample_end=sample_start + len(seq_lens),
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), str(rank_dir / "summary.json"))
    stored = load_local_cache_write_summary(str(rank_dir))
    shard_map, shards = build_global_target_cache_shard_map([stored])
    rename_local_target_cache_shards(
        output_dir=str(slice_dir), rank_dir=str(rank_dir), summary=stored, shard_map=shard_map
    )
    finalize_target_cache_index(output_dir=str(slice_dir), summaries=[stored], shard_map=shard_map)
    manifest = build_target_cache_manifest(
        num_samples=int(stored["num_local_samples"]),
        shards=shards,
        target_layer_ids=TARGET_LAYERS,
        hidden_size=HIDDEN,
        extra_fields=extra_fields
        or {"target_model_name_or_path": "zai-org/GLM-5.3-Flash", "max_length": 4096},
    )
    write_target_cache_manifest(output_dir=str(slice_dir), manifest=manifest)
    cleanup_target_cache_tmp_dir(str(slice_dir))
    load_target_cache_manifest(str(slice_dir))  # the writer stack left a valid dir


def _write_plan(cache_root: Path, total_rows: int, slice_rows: int) -> None:
    write_slice_plan(str(cache_root), total_rows=total_rows, slice_rows=slice_rows)


def _read_manifest(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def test_plan_matches_row_ranges() -> None:
    """The lib plan and the plan file agree on the slice layout."""
    assert plan_cache_slices(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_merge_folds_slices_into_one_canonical_cache(tmp_path: Path) -> None:
    """Two slices merge into a cache the upstream loader and reader accept."""
    cache_root = tmp_path / "cache"
    _write_plan(cache_root, total_rows=8, slice_rows=4)
    _build_slice(cache_root / "slice_000000000", sample_start=0, seq_lens=[4, 6, 4, 6])
    _build_slice(cache_root / "slice_000000004", sample_start=4, seq_lens=[8, 4, 6, 4])

    summary = merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)
    assert summary["num_slices"] == 2
    assert summary["num_samples"] == 8

    manifest = load_target_cache_manifest(str(cache_root))
    assert manifest["num_samples"] == 8
    assert manifest["merged_from"] == {"num_slices": 2, "slice_rows": 4, "total_rows": 8}
    assert manifest["target_model_name_or_path"] == "zai-org/GLM-5.3-Flash"

    dataset = CacheDataset(cache_dir=str(cache_root))
    assert len(dataset) == 8
    for index, seq_len in enumerate([4, 6, 4, 6, 8, 4, 6, 4]):
        sample = dataset[index]
        assert sample["input_ids"].shape[0] == seq_len
        assert sample["target_hidden_states"].shape == (seq_len, len(TARGET_LAYERS) * HIDDEN)
    dataset.close()
    for slice_name in ("slice_000000000", "slice_000000004"):
        slice_dir = cache_root / slice_name
        assert (slice_dir / "manifest.json").exists()
        assert (slice_dir / "samples.idx").exists()
        assert not list(slice_dir.glob("shard-*.bin"))


def test_merge_is_idempotent_after_completion(tmp_path: Path) -> None:
    """Re-running the merge over already-merged slices converges."""
    cache_root = tmp_path / "cache"
    _write_plan(cache_root, total_rows=6, slice_rows=3)
    _build_slice(cache_root / "slice_000000000", sample_start=0, seq_lens=[4, 4, 4])
    _build_slice(cache_root / "slice_000000003", sample_start=3, seq_lens=[6, 6, 6])
    first = merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)
    second = merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)
    assert first == second
    assert load_target_cache_manifest(str(cache_root))["num_samples"] == 6


def test_merge_survives_a_crash_mid_rename(tmp_path: Path) -> None:
    """Shards moved for slice 0 only: re-merge detects it and completes."""
    cache_root = tmp_path / "cache"
    _write_plan(cache_root, total_rows=6, slice_rows=3)
    _build_slice(cache_root / "slice_000000000", sample_start=0, seq_lens=[4, 4, 4])
    _build_slice(cache_root / "slice_000000003", sample_start=3, seq_lens=[6, 6, 6])
    for shard in _read_manifest(cache_root / "slice_000000000" / "manifest.json")["shards"]:
        (cache_root / "slice_000000000" / shard["file_name"]).replace(
            cache_root / shard["file_name"]
        )
    summary = merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)
    assert summary["num_samples"] == 6
    load_target_cache_manifest(str(cache_root))


def test_merge_survives_a_crash_before_the_index_write(tmp_path: Path) -> None:
    """Shards fully moved, no samples.idx at the root: re-merge rebuilds it."""
    cache_root = tmp_path / "cache"
    _write_plan(cache_root, total_rows=4, slice_rows=4)
    _build_slice(cache_root / "slice_000000000", sample_start=0, seq_lens=[4, 4, 8, 4])
    merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)
    (cache_root / "samples.idx").unlink()
    (cache_root / "manifest.json").unlink()
    summary = merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)
    assert summary["num_samples"] == 4
    load_target_cache_manifest(str(cache_root))


def test_merge_fails_loudly_on_an_unbuilt_slice(tmp_path: Path) -> None:
    """A planned-but-missing slice dir stops the merge."""
    cache_root = tmp_path / "cache"
    _write_plan(cache_root, total_rows=6, slice_rows=3)
    _build_slice(cache_root / "slice_000000000", sample_start=0, seq_lens=[4, 4, 4])
    with pytest.raises(RuntimeError, match="not built yet"):
        merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)


def test_merge_fails_loudly_on_an_unplanned_slice(tmp_path: Path) -> None:
    """A slice dir outside the plan stops the merge (stale numbering)."""
    cache_root = tmp_path / "cache"
    _write_plan(cache_root, total_rows=4, slice_rows=4)
    _build_slice(cache_root / "slice_000000000", sample_start=0, seq_lens=[4, 4])
    _build_slice(cache_root / "slice_000000002", sample_start=2, seq_lens=[4, 4])
    with pytest.raises(RuntimeError, match="unexpected slice dirs"):
        merge_slices.merge_cache_slices(cache_root=cache_root, output_dir=cache_root)


def test_slice_completeness_gate(tmp_path: Path) -> None:
    """Complete dirs pass; missing manifests, shards, or empty dirs fail."""
    good = tmp_path / "good"
    _build_slice(good, sample_start=0, seq_lens=[4, 4])
    assert slice_cache_complete(str(good)) is True

    empty = tmp_path / "empty"
    empty.mkdir()
    assert slice_cache_complete(str(empty)) is False

    broken = tmp_path / "broken"
    _build_slice(broken, sample_start=0, seq_lens=[4, 4])
    for shard in broken.glob("shard-*.bin"):
        shard.unlink()
    assert slice_cache_complete(str(broken)) is False

    assert slice_cache_complete(str(tmp_path / "absent")) is False


def test_resolve_sample_range_validation() -> None:
    """The cache script's range override validates bounds and pairing."""
    resolve = cache_script.resolve_sample_range
    assert resolve(None, None, 10) == (0, 10)
    assert resolve(2, 5, 10) == (2, 5)
    assert resolve(10, 10, 10) == (10, 10)  # empty range is legal and builds nothing
    with pytest.raises(ValueError, match="together"):
        resolve(2, None, 10)
    for bad in ((5, 2, 10), (0, 11, 10), (-1, 5, 10)):
        with pytest.raises(ValueError, match="invalid sample range"):
            resolve(*bad)


def test_regen_resume_state_is_index_keyed(tmp_path: Path) -> None:
    """Committed input indices are the resume state, from both output streams."""
    output = tmp_path / "regen.jsonl"
    error = tmp_path / "regen_error.jsonl"
    output.write_text(
        '{"id": 1, "input_index": 3}\n{"id": 2, "input_index": 7}\n', encoding="utf-8"
    )
    error.write_text('{"status": "error", "input_index": 5}\n', encoding="utf-8")
    done = regen_script.collect_done_indices(str(output), str(error))
    assert done == {3, 5, 7}


def test_regen_resume_ignores_legacy_and_unparsable_rows(tmp_path: Path) -> None:
    """Rows without input_index (pre-fix runs) contribute nothing to the state."""
    output = tmp_path / "regen.jsonl"
    output.write_text('{"id": 1}\nnot json\n{"input_index": 9}\n', encoding="utf-8")
    done = regen_script.collect_done_indices(str(output), str(tmp_path / "absent.jsonl"))
    assert done == {9}


def test_regen_truncates_a_partial_trailing_line(tmp_path: Path) -> None:
    """A killed write mid-line is dropped; complete files are untouched."""
    partial = tmp_path / "partial.jsonl"
    partial.write_bytes(b'{"input_index": 1}\n{"input_in')
    regen_script.truncate_partial_line(str(partial))
    assert partial.read_bytes() == b'{"input_index": 1}\n'

    clean = tmp_path / "clean.jsonl"
    clean.write_bytes(b'{"input_index": 1}\n')
    regen_script.truncate_partial_line(str(clean))
    assert clean.read_bytes() == b'{"input_index": 1}\n'

    absent = tmp_path / "absent.jsonl"
    regen_script.truncate_partial_line(str(absent))  # no-op, no crash
