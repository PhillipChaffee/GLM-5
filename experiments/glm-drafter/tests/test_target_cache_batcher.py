"""Tests for the token-budget packer used by the target-cache capture pass."""

import importlib.util
import runpy
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_BUDGET_BATCHER_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "data" / "token_budget_batcher.py"
)

# The packer sits beside the data scripts, not on the import path; runpy
# loads it without inventing a second module name for one file mypy already
# tracks.
_batcher = runpy.run_path(str(_BUDGET_BATCHER_PATH))
build_packs = _batcher["build_packs"]
padded_token_count = _batcher["padded_token_count"]
sorted_order = _batcher["sorted_order"]

_LENGTHS = [50, 40, 30, 20, 10]


@pytest.fixture
def data_scripts_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the data scripts dir on the path for script-side sibling imports."""
    data_dir = str(Path(__file__).resolve().parent.parent / "scripts" / "data")
    monkeypatch.syspath_prepend(data_dir)


def test_sorted_order_descends_with_index_tiebreak() -> None:
    assert sorted_order([3, 1, 3, 2]) == [0, 2, 3, 1]


def test_packs_recorded_example() -> None:
    packs = build_packs(_LENGTHS, token_budget=60, max_count=2, max_row_tokens=60)
    assert packs == [[0], [1], [2, 3], [4]]


def test_count_cap_closes_packs() -> None:
    packs = build_packs([1, 1, 1, 1], token_budget=100, max_count=2, max_row_tokens=100)
    assert packs == [[0, 1], [2, 3]]


def test_budget_closes_packs_before_the_cap() -> None:
    packs = build_packs([3, 3, 3], token_budget=6, max_count=5, max_row_tokens=3)
    assert packs == [[0, 1], [2]]


def test_budget_hits_exactly_with_tied_lengths() -> None:
    packs = build_packs([4, 3, 4, 3], token_budget=8, max_count=3, max_row_tokens=8)
    assert packs == [[0, 2], [1, 3]]


def test_row_exceeding_declared_max_fails_closed() -> None:
    """Rows beyond the declared max are a caller bug; fail loudly."""
    with pytest.raises(ValueError, match="beyond the declared"):
        build_packs([9, 2, 1], token_budget=10, max_count=4, max_row_tokens=5)


def test_budget_below_max_row_tokens_fails_closed() -> None:
    with pytest.raises(ValueError, match="must cover the collator's max_length"):
        build_packs(
            [7_172, 200],
            token_budget=4_096,
            max_count=2,
            max_row_tokens=4_096 + 1,
        )


def test_non_positive_budget_or_cap_fail() -> None:
    with pytest.raises(ValueError, match="token_budget must be positive"):
        build_packs(_LENGTHS, token_budget=0, max_count=2, max_row_tokens=100)
    with pytest.raises(ValueError, match="max_count must be positive"):
        build_packs(_LENGTHS, token_budget=10, max_count=0, max_row_tokens=100)


def test_packs_are_a_partition_in_sorted_order() -> None:
    for lengths in (_LENGTHS, [1, 2, 3, 4, 5, 6], [100, 0, 0], [7]):
        budget = max(lengths) * 3
        packs = build_packs(
            lengths,
            token_budget=budget,
            max_count=3,
            max_row_tokens=max(lengths),
        )
        flat = [index for pack in packs for index in pack]
        assert sorted(flat) == list(range(len(lengths)))
        assert flat == sorted_order(lengths)
        tokenized = [lengths[index] for index in flat]
        assert tokenized == sorted(tokenized, reverse=True)
        for pack in packs:
            longest = max(lengths[index] for index in pack)
            assert longest * len(pack) <= budget


def test_empty_lengths_build_no_packs() -> None:
    assert build_packs([], token_budget=10, max_count=2, max_row_tokens=10) == []
    assert padded_token_count([], []) == 0


def test_padded_token_count_is_rows_times_pack_maximum() -> None:
    lengths = [4096, 50, 30, 20]
    packs = [[0], [1, 2, 3]]
    assert padded_token_count(lengths, packs) == 4096 + 3 * 50


def test_packs_cover_each_row_exactly_once() -> None:
    lengths = [9, 2, 1, 4096]
    packs = build_packs(lengths, token_budget=4096, max_count=2, max_row_tokens=4096)
    seen = [index for pack in packs for index in pack]
    assert sorted(seen) == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in pack) <= 4096 for pack in packs)
    for pack in packs:
        longest = max(lengths[index] for index in pack)
        assert longest * len(pack) <= 4096


class _StubCollator:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any] | None:
        self.calls.append(features)
        if features and features[0]["too_short"]:
            return None
        length = int(features[0]["length"])
        return {"loss_mask": torch.zeros(1, length, dtype=torch.uint8)}


class _StubSubset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class _StubModel:
    """Duck type returned by the injected fake DataLoader."""


def _load_prepare() -> types.ModuleType:
    """Load the capture script as a real module so tests can rebind globals."""
    spec = importlib.util.spec_from_file_location(
        "prepare_target_cache_under_test",
        str(
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "data"
            / "prepare_target_cache_glm.py"
        ),
    )
    if spec is None or spec.loader is None:
        missing_error = "prepare_target_cache_glm.py not found beside the tests"
        raise RuntimeError(missing_error)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_loader_capture(prepare: types.ModuleType) -> dict[str, Any]:
    """Swap in a recording DataLoader; returns the kwargs dict it filled."""
    captured: dict[str, Any] = {}

    # The module object from _load_prepare is live; assigning the attribute
    # rebinds the name _build_cache_dataloader resolves for its loader.
    def _fake_loader(dataset: object, **kwargs: object) -> _StubModel:
        captured["args"] = (dataset,)
        captured["kwargs"] = kwargs
        return _StubModel()

    vars(prepare)["DataLoader"] = _fake_loader
    return captured


def test_trunc_lengths_count_through_the_collator(
    data_scripts_on_path: None,
) -> None:
    """The counting seam uses the collator itself, min-loss rows reading 0."""
    prepare = _load_prepare()
    stub_collator = _StubCollator()
    subset = _StubSubset(
        [
            {"length": 4, "too_short": False},
            {"length": 7, "too_short": False},
            {"length": 2, "too_short": True},
        ]
    )
    compute = prepare._compute_trunc_lengths
    lengths, filtered = compute(subset, stub_collator)
    assert lengths == [4, 7, 0]
    assert filtered == 1
    assert len(stub_collator.calls) == 3
    packs = build_packs(lengths, token_budget=8, max_count=2, max_row_tokens=8)
    assert packs == [[1], [0, 2]]


def test_progress_through_batch_both_modes(data_scripts_on_path: None) -> None:
    """Fixed lane clamps at the total; budget lane reads pack prefix counts."""
    prepare = _load_prepare()
    progress = prepare._progress_through_batch
    assert progress(1, None, 2, 7) == 4
    assert progress(3, None, 2, 7) == 7
    assert progress(2, [1, 3, 6], 2, 7) == 6
    assert progress(1, [1, 3], 2, 6) == 3


def test_fixed_lane_loader_shape_is_pinned(data_scripts_on_path: None) -> None:
    """Unset budget constructs exactly the pre-#77 proven loader shape."""
    prepare = _load_prepare()
    captured = _stub_loader_capture(prepare)
    cli_args = SimpleNamespace(cache_token_budget=None, local_batch_size=2, num_workers=4)
    config = SimpleNamespace(data=SimpleNamespace(max_length=4096))
    loader, prefix = prepare._build_cache_dataloader(
        local_subset=_StubSubset([{"length": 3, "too_short": False}] * 5),
        train_collator=_StubCollator(),
        cli_args=cli_args,
        config=config,
        local_total_samples=5,
    )
    assert isinstance(loader, _StubModel)
    assert prefix is None
    assert set(captured["kwargs"]) == {
        "batch_size",
        "collate_fn",
        "num_workers",
        "pin_memory",
        "drop_last",
    }
    assert captured["kwargs"]["batch_size"] == 2
    assert captured["kwargs"]["drop_last"] is False


def test_budget_lane_empty_subset_builds_a_zero_pack_loader(
    data_scripts_on_path: None,
) -> None:
    """A legal empty range ([start,end) == 0 rows) survives budget mode."""
    prepare = _load_prepare()
    captured = _stub_loader_capture(prepare)
    cli_args = SimpleNamespace(cache_token_budget=8192, local_batch_size=16, num_workers=4)
    config = SimpleNamespace(data=SimpleNamespace(max_length=4096))
    loader, prefix = prepare._build_cache_dataloader(
        local_subset=_StubSubset([]),
        train_collator=_StubCollator(),
        cli_args=cli_args,
        config=config,
        local_total_samples=0,
    )
    assert isinstance(loader, _StubModel)
    assert prefix == []
    assert captured["kwargs"]["batch_sampler"] == []


def test_budget_lane_all_filtered_rows_pack_to_zero(data_scripts_on_path: None) -> None:
    """A slice whose rows all fail min-loss packs to nothing, not a crash."""
    prepare = _load_prepare()
    captured = _stub_loader_capture(prepare)
    stub_collator = _StubCollator()
    cli_args = SimpleNamespace(cache_token_budget=8192, local_batch_size=16, num_workers=4)
    config = SimpleNamespace(data=SimpleNamespace(max_length=4096))
    loader, prefix = prepare._build_cache_dataloader(
        local_subset=_StubSubset(
            [
                {"length": 2, "too_short": True},
                {"length": 5, "too_short": True},
            ]
        ),
        train_collator=stub_collator,
        cli_args=cli_args,
        config=config,
        local_total_samples=2,
    )
    assert isinstance(loader, _StubModel)
    assert prefix == [2]
    assert captured["kwargs"]["batch_sampler"] == [[0, 1]]
