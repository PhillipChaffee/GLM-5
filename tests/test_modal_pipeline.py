"""Checks for the shared Modal pipeline helpers."""

import json
import runpy
from pathlib import Path

import pytest

# The lib sits beside the experiment apps, not on the import path (see the
# E402 sys.path shims in the apps); runpy loads it without inventing a
# second module name for a file mypy already tracks as lib.modal_pipeline.
_lib = runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "experiments" / "lib" / "modal_pipeline.py")
)
build_push_jobs = _lib["build_push_jobs"]
plan_cache_slices = _lib["plan_cache_slices"]
count_jsonl_lines = _lib["count_jsonl_lines"]
write_slice_plan = _lib["write_slice_plan"]
build_cache_cmd = _lib["build_cache_cmd"]
build_train_cmd = _lib["build_train_cmd"]


def _seed_volume(tmp_path: Path, exp_name: str) -> None:
    """Lay out a fake volume: cache shards, regen data, and two checkpoint steps."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "shard-0000.tar").write_bytes(b"c")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "corpus.jsonl").write_bytes(b"d")
    for step in ("step_19", "step_20"):
        step_dir = tmp_path / "home/checkpoints/deepspec" / exp_name / step
        step_dir.mkdir(parents=True)
        (step_dir / "optimizer.pt").write_bytes(b"o")
    final = tmp_path / "home/checkpoints/deepspec" / exp_name / "step_20"
    for name in ("model.safetensors", "config.json", "train_config.py"):
        (final / name).write_bytes(b"x")


def test_checkpoint_resolves_for_the_current_experiment_name(tmp_path: Path) -> None:
    """The upload list points at the last step_* under the named experiment."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path), "glm_drafter_block7_glm53flash_full")
    ckpt = {dst: src for src, dst in jobs if dst.startswith("checkpoints/")}
    assert set(ckpt) == {
        "checkpoints/glm_drafter_block7_glm53flash_full/model.safetensors",
        "checkpoints/glm_drafter_block7_glm53flash_full/config.json",
        "checkpoints/glm_drafter_block7_glm53flash_full/train_config.py",
    }
    assert all(src.parent.name == "step_20" for src in ckpt.values())


def test_checkpoint_resolves_for_a_renamed_experiment(tmp_path: Path) -> None:
    """A different experiment name uploads under that name, not the old literal."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_7union")
    jobs = build_push_jobs(str(tmp_path), "glm_drafter_block7_glm53flash_7union")
    assert "checkpoints/glm_drafter_block7_glm53flash_7union/model.safetensors" in {
        dst for _, dst in jobs
    }


def test_cache_and_data_upload_alongside_the_checkpoint(tmp_path: Path) -> None:
    """The cache shards and regen data stay in the upload list next to the checkpoint."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path), "glm_drafter_block7_glm53flash_full")
    assert {(src.name, dst) for src, dst in jobs if not dst.startswith("checkpoints/")} == {
        ("shard-0000.tar", "cache/shard-0000.tar"),
        ("corpus.jsonl", "data/corpus.jsonl"),
    }


def test_unset_experiment_name_skips_the_checkpoint_arm(tmp_path: Path) -> None:
    """No exp_name: cache and data still upload, no checkpoint jobs appear."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path))
    assert all(not dst.startswith("checkpoints/") for _, dst in jobs)
    assert len(jobs) == 2


def test_wrong_experiment_name_skips_the_checkpoint_arm(tmp_path: Path) -> None:
    """A name with no directory on the volume yields no checkpoint jobs."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path), "glm_drafter_block7_glm53flash_fuller")
    assert all(not dst.startswith("checkpoints/") for _, dst in jobs)


def test_slice_plan_covers_every_row_in_order() -> None:
    """Slices are contiguous [start, end) ranges; the last one may be short."""
    assert plan_cache_slices(698, 100) == [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 600),
        (600, 698),
    ]
    assert plan_cache_slices(0, 100) == []
    assert plan_cache_slices(50, 100) == [(0, 50)]


def test_slice_plan_rejects_bad_shapes() -> None:
    """Negative totals and non-positive slice rows fail loudly."""
    with pytest.raises(ValueError, match="slice_rows"):
        plan_cache_slices(100, 0)
    with pytest.raises(ValueError, match="total_rows"):
        plan_cache_slices(-1, 100)


def test_count_jsonl_lines_counts_newlines(tmp_path: Path) -> None:
    """The row count is the line count of the jsonl file."""
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    assert count_jsonl_lines(str(path)) == 3
    path.write_bytes(b'{"a": 1}\n{"a": 2}')
    assert count_jsonl_lines(str(path)) == 2


def test_slice_plan_is_written_once_and_frozen(tmp_path: Path) -> None:
    """A re-plan with the same shape is idempotent; a different one fails loudly."""
    slices = write_slice_plan(str(tmp_path), total_rows=698, slice_rows=100)
    assert slices[0] == (0, 100)
    assert slices[-1] == (600, 698)
    assert json.loads((tmp_path / "slice_plan.json").read_text())["slice_rows"] == 100
    assert write_slice_plan(str(tmp_path), total_rows=698, slice_rows=100) == slices
    with pytest.raises(RuntimeError, match="refusing to re-plan"):
        write_slice_plan(str(tmp_path), total_rows=698, slice_rows=200)
    with pytest.raises(RuntimeError, match="refusing to re-plan"):
        write_slice_plan(str(tmp_path), total_rows=700, slice_rows=100)


def test_cache_cmd_carries_the_slice_range() -> None:
    """Slice bounds land on the command line; half a range is rejected."""
    cmd = build_cache_cmd(
        config="c",
        train_data_path="t",
        output_dir="o",
        local_batch_size=2,
        cache_script="s",
        sample_start=5000,
        sample_end=10000,
    )
    assert cmd[-4:] == ["--sample-start", "5000", "--sample-end", "10000"]
    bare = build_cache_cmd(
        config="c", train_data_path="t", output_dir="o", local_batch_size=2, cache_script="s"
    )
    assert "--sample-start" not in bare
    with pytest.raises(ValueError, match="together"):
        build_cache_cmd(
            config="c",
            train_data_path="t",
            output_dir="o",
            local_batch_size=2,
            cache_script="s",
            sample_start=0,
        )


def test_train_cmd_can_override_the_checkpoint_cadence() -> None:
    """The drill-sized checkpointing override maps to the logging opts path."""
    cmd = build_train_cmd(
        config="c",
        exp_name="e",
        epochs=1,
        local_batch_size=4,
        head_type="vanilla",
        cache_path="/vol/cache",
        max_steps=12,
        checkpointing_steps=5,
    )
    assert "logging.checkpointing_steps=5" in cmd
    bare = build_train_cmd(
        config="c",
        exp_name="e",
        epochs=1,
        local_batch_size=4,
        head_type="vanilla",
        cache_path="/vol/cache",
        max_steps=12,
    )
    assert not any("checkpointing_steps" in part for part in bare)
