"""Checks for the shared Modal pipeline helpers."""

import runpy
from pathlib import Path

# The lib sits beside the experiment apps, not on the import path (see the
# E402 sys.path shims in the apps); runpy loads it without inventing a
# second module name for a file mypy already tracks as lib.modal_pipeline.
build_push_jobs = runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "experiments" / "lib" / "modal_pipeline.py")
)["build_push_jobs"]


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


def test_cache_upload_is_opt_in_and_data_defaults_in(tmp_path: Path) -> None:
    """Default uploads corpus + checkpoint (#22 durable artifacts); no cache."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path), "glm_drafter_block7_glm53flash_full")
    assert {(src.name, dst) for src, dst in jobs if not dst.startswith("checkpoints/")} == {
        ("corpus.jsonl", "data/corpus.jsonl"),
    }


def test_explicit_flag_uploads_the_cache_arm(tmp_path: Path) -> None:
    """include_cache=True restores the legacy behavior for cache-durable pipelines."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path), "glm_drafter_block7_glm53flash_full", include_cache=True)
    assert ("shard-0000.tar", "cache/shard-0000.tar") in {(src.name, dst) for src, dst in jobs}


def test_unset_experiment_name_skips_the_checkpoint_arm(tmp_path: Path) -> None:
    """No exp_name: data still uploads (cache opt-in), no checkpoint jobs appear."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path))
    assert not any(dst.startswith(("checkpoints/", "cache/")) for _, dst in jobs)
    assert len(jobs) == 1


def test_wrong_experiment_name_skips_the_checkpoint_arm(tmp_path: Path) -> None:
    """A name with no directory on the volume yields no checkpoint jobs."""
    _seed_volume(tmp_path, "glm_drafter_block7_glm53flash_full")
    jobs = build_push_jobs(str(tmp_path), "glm_drafter_block7_glm53flash_fuller")
    assert all(not dst.startswith("checkpoints/") for _, dst in jobs)
    assert ("corpus.jsonl", "data/corpus.jsonl") in {(src.name, dst) for src, dst in jobs}
