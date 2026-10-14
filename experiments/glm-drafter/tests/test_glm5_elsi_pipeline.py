"""End-to-end loader tests with fabricated cache/capture data (issue 51).

Exercises the stage-1 accumulation and table build plus the select report
against synthetic rows whose sums/counts/CKA are independently known, so a
misalignment in real data reading surfaces before any GPU or volume spend.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import elsi_analyze
import pytest
import torch
from elsi_analyze import (
    CHUNK_TOKENS,
    POST_NORM_LABEL,
    CacheSource,
    matrix_entropy_nats,
    run_select,
    stage1_accumulate,
    stage1_tables,
)
from elsi_capture_glm import (
    accumulate_batch,
    build_accumulators,
    token_entropies,
    top_fraction_mask,
)

if TYPE_CHECKING:
    from pathlib import Path

HIDDEN = 8
LAYER_IDS = [1, 12, 23, 34, 42, 43, 44]


def _fabricated_rows(counts: list[int]) -> list[dict[str, torch.Tensor]]:
    """Fake cache rows: post-norm column identical to last_hidden (sanity = 0)."""
    rows = []
    for index, length in enumerate(counts):
        hidden = torch.zeros(length, HIDDEN * 7)
        for column in range(7):
            hidden[:, column * HIDDEN : (column + 1) * HIDDEN] = (column + 1) * (
                index + 1
            ) + 0.01 * torch.arange(length, dtype=torch.float32).unsqueeze(1)
        last_column = hidden[:, 6 * HIDDEN :]
        mask = torch.zeros(length, dtype=torch.uint8)
        mask[int(length * 0.4) : length] = 1
        rows.append(
            {
                "input_ids": torch.zeros(length, dtype=torch.int32),
                "loss_mask": mask,
                "target_hidden_states": hidden.to(torch.bfloat16),
                "target_last_hidden_states": last_column.to(torch.bfloat16),
            }
        )
    return rows


class _FakeCacheDataset:
    """Duck-typed loader source returning prepared rows in order."""

    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._rows[index]


def test_stage1_accumulate_and_tables(tmp_path: Path) -> None:
    counts = [70, 33, 130, 256]
    rows = _fabricated_rows(counts)
    source = CacheSource(
        dataset=_FakeCacheDataset(rows),
        num_samples=len(rows),
        layer_ids=LAYER_IDS,
        hidden_size=HIDDEN,
        max_length=2048,
        source_paths=["fake.jsonl"],
    )
    snapshots, segment = stage1_accumulate(source, tmp_path)
    expected_assistant = [length - int(length * 0.4) for length in counts]
    assert segment["tokens_all"] == sum(counts)
    assert segment["tokens_assistant"] == sum(expected_assistant)
    assert segment["assistant_counts"] == expected_assistant
    for view, (sums, view_counts) in snapshots.items():
        assert sums.shape[1] == HIDDEN * 7
        expected_tokens = segment["tokens_all"] if view == "all" else segment["tokens_assistant"]
        assert float(view_counts.sum()) == expected_tokens
        assert sums.shape[0] == (expected_tokens + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    tables = stage1_tables(snapshots, LAYER_IDS, HIDDEN)
    for view in ("all", "assistant"):
        per_view = tables[view]
        assert per_view["token_total"] == (
            segment["tokens_all"] if view == "all" else segment["tokens_assistant"]
        )
        assert len(per_view["views_u"]) == 4
        for u_payload in per_view["views_u"].values():
            matrix = u_payload["cka_matrix"]
            values = matrix["values"]
            size = len(values)
            for index in range(size):
                diagonal_value = values[index][index]
                assert abs(diagonal_value - 1.0) < 1e-2
            labels = u_payload["per_layer_s1_nats"]
            assert len(labels) == 7
            asserted = u_payload["subsets_5_6_7"]
            assert len(asserted) == 21 + 7 + 1
            mapped = asserted[0]["labels"]
            assert len(mapped) == len(asserted[0]["members"])


def test_select_loader_against_fabricated_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Select-loader checks on the fabricated capture at u=32.

    Structural only: the full-fidelity u=256/839 beams run on the Modal
    stage, not in unit time.
    """
    monkeypatch.setattr(elsi_analyze, "SELECT_POOLED_VIEWS", (32,))
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    num_layers = 45
    num_columns = 46
    chunk_texts = 5
    for column in range(num_columns):
        sums = (column + 1) + 0.5 * torch.arange(chunk_texts * HIDDEN, dtype=torch.float32).reshape(
            chunk_texts, HIDDEN
        )
        counts = torch.tensor([CHUNK_TOKENS] * chunk_texts, dtype=torch.float32)
        torch.save({"sums": sums, "counts": counts}, capture_dir / f"col_{column:03d}.pt")
    (capture_dir / "capture_meta.json").write_text(
        json.dumps(
            {
                "num_layers": num_layers,
                "num_columns": num_columns,
                "num_rows": 4,
                "rho": 0.3,
                "max_length": 2048,
                "tokens_total": chunk_texts * CHUNK_TOKENS * num_columns,
            }
        )
        + "\n"
    )
    out = tmp_path / "select.json"
    run_select(str(capture_dir), str(out))
    report = json.loads(out.read_text())
    assert report["meta"]["num_columns"] == num_columns
    constants = report["meta"]["search_constants"]
    assert constants["k_beam"] == 10
    assert constants["k_beam_source"].startswith("paper-stated")
    assert constants["k_expand"] == 10
    assert constants["k_expand_k_pool_source"] != ""
    for view_key, view_payload in report["tables_by_u"].items():
        assert view_key in {"32", "32_rownorm", "256", "256_rownorm"}
        labels = view_payload["cka_matrix"]["labels"]
        assert labels[-1] == POST_NORM_LABEL
        assert len(labels) == num_columns
        assert len(view_payload["standing_sets"]) == 6
        for beam_key in ("elsi_m5_top", "elsi_m7_top", "elsi_c_m5_top", "elsi_c_m7_top"):
            for entry in view_payload[beam_key]:
                assert len(entry["labels"]) in (5, 7)
        reads = view_payload["post_norm_reads"]
        assert set(reads) >= {"cka_raw43_postN", "cka_raw44_postN", "cka_raw43_raw44"}
        chance = view_payload["chance_baseline"]
        assert chance["band"]["min"] <= chance["band"]["max"]
        assert set(chance["columns"]) == {5, 15, 30, 42}


def test_select_grid_assert_rejects_mismatched_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A column whose pooled chunk grid differs from column 0 fails the load."""
    monkeypatch.setattr(elsi_analyze, "SELECT_POOLED_VIEWS", (32,))
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    for column in range(4):
        sums = torch.arange(6 * HIDDEN, dtype=torch.float32).reshape(6, HIDDEN)
        counts = torch.tensor([CHUNK_TOKENS] * 6, dtype=torch.float32)
        if column == 3:  # one dropped chunk: a different pooled grid
            sums = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)
            counts = torch.tensor([CHUNK_TOKENS] * 5, dtype=torch.float32)
        torch.save({"sums": sums, "counts": counts}, capture_dir / f"col_{column:03d}.pt")
    (capture_dir / "capture_meta.json").write_text(
        json.dumps({"num_layers": 3, "num_columns": 4, "rho": 0.3}) + "\n"
    )
    with pytest.raises(ValueError, match="pooled chunk grid"):
        run_select(str(capture_dir), str(tmp_path / "select.json"))


def test_select_all_raw_pool_drops_post_norm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--drop-post-norm: labels keep rawN for all columns, postN arms/reads go."""
    monkeypatch.setattr(elsi_analyze, "SELECT_POOLED_VIEWS", (32,))
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    num_layers, num_columns = 45, 46
    for column in range(num_columns):
        sums = torch.arange(6 * HIDDEN, dtype=torch.float32).reshape(6, HIDDEN) + 7.0 * column
        counts = torch.tensor([CHUNK_TOKENS] * 6, dtype=torch.float32)
        torch.save({"sums": sums, "counts": counts}, capture_dir / f"col_{column:03d}.pt")
    (capture_dir / "capture_meta.json").write_text(
        json.dumps(
            {
                "num_layers": num_layers,
                "num_columns": num_columns,
                "num_rows": 4,
                "rho": 0.3,
                "max_length": 2048,
            }
        )
        + "\n"
    )
    out = tmp_path / "select_allraw.json"
    run_select(str(capture_dir), str(out), drop_post_norm=True)
    report = json.loads(out.read_text())
    assert "post-final-norm column is excluded entirely" in report["meta"]["pool"]
    for view_payload in report["tables_by_u"].values():
        labels = view_payload["cka_matrix"]["labels"]
        assert labels[-1] == "raw44"
        assert len(labels) == 45
        assert "post_norm_reads" not in view_payload
        arm_names = {arm["name"] for arm in view_payload["standing_sets"]}
        assert arm_names == {
            "5-stock [1,12,23,34,42]",
            "6-stock [1,12,23,34,42,43]",
            "base4 [1,12,23,34]",
            "DFlash2 convention [5,14,24,33,42]",
        }


def test_select_identity_pooling_u_equals_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With u == chunk count the 839-style grid leaves each rep row a chunk mean."""
    monkeypatch.setattr(elsi_analyze, "SELECT_POOLED_VIEWS", (64,))
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    chunks = 64
    torch.manual_seed(5)
    for column in range(5):
        sums = torch.randn(chunks, HIDDEN)
        counts = torch.tensor([CHUNK_TOKENS] * (chunks - 1) + [18.0], dtype=torch.float32)
        torch.save({"sums": sums, "counts": counts}, capture_dir / f"col_{column:03d}.pt")
    (capture_dir / "capture_meta.json").write_text(
        json.dumps({"num_layers": 4, "num_columns": 5, "rho": 0.3}) + "\n"
    )
    out = tmp_path / "select.json"
    run_select(str(capture_dir), str(out))
    report = json.loads(out.read_text())
    per_layer = report["tables_by_u"]["64"]["per_layer_s1_nats"]
    for column in range(4):
        bundle = torch.load(capture_dir / f"col_{column:03d}.pt", weights_only=True)
        reps = bundle["sums"] / bundle["counts"].unsqueeze(1)  # exact pooled-grid identity
        expected = matrix_entropy_nats(reps @ reps.T)
        # tol: the loader path goes through an fp32 cumsum/diff roundtrip
        assert per_layer[f"raw{column}"] == pytest.approx(expected, abs=1e-4)


def test_capture_pooling_matches_reference(tmp_path: Path) -> None:
    """accumulate_batch matches a dense per-mask reference incl. short rows.

    Row 1 holds only 10 real tokens padded to the batch width of 25: garbage
    pad positions must never enter the top-rho selection (that is the fix
    proven here), so both rows are sliced to ``row_lengths`` before masking.
    """
    torch.manual_seed(51)
    rows_acc, width, dim = 2, 25, 8
    streams = [torch.randn(rows_acc, width, dim) for _ in range(3)]
    entropies = torch.rand(rows_acc, width)
    row_lengths = [25, 10]
    accumulators = build_accumulators(3, dim)
    running, batch_counts = accumulate_batch(
        accumulators,
        streams,
        entropies,
        row_lengths=row_lengths,
        rho=0.3,
        running=0,
    )
    kept_counts = [
        int(top_fraction_mask(entropies[row, :length], 0.3).sum())
        for row, length in enumerate(row_lengths)
    ]
    assert running == sum(kept_counts)
    assert batch_counts == kept_counts
    reference = _dense_pooled_reference(streams, entropies, row_lengths, rho=0.3)
    for column, acc in accumulators.items():
        sums, counts = acc.snapshot()
        total = running
        chunks = (total + CHUNK_TOKENS - 1) // CHUNK_TOKENS
        assert sums.shape[0] == chunks
        expect_sums, expect_counts = reference[column]
        assert torch.allclose(counts, expect_counts)
        assert torch.allclose(sums, expect_sums, rtol=1e-4, atol=1e-4)


def _dense_pooled_reference(
    streams: list[torch.Tensor],
    entropies: torch.Tensor,
    row_lengths: list[int],
    rho: float,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Independent pooling: piecewise collect then chunk-sum via flat buffers."""
    dim = streams[0].shape[-1]
    total = sum(
        int(top_fraction_mask(entropies[row, :length], rho).sum())
        for row, length in enumerate(row_lengths)
    )
    chunks = (total + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    expected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for column, stream in enumerate(streams):
        kept_flat = []
        for row, length in enumerate(row_lengths):
            mask = top_fraction_mask(entropies[row, :length], rho)
            positions = mask.nonzero().flatten()
            kept = stream[row, :length].index_select(0, positions)
            kept_flat.append(kept)
        pooled = torch.cat(kept_flat, dim=0)
        sums = torch.zeros(chunks, dim, dtype=torch.float32)
        counts = torch.zeros(chunks, dtype=torch.float32)
        for start in range(0, pooled.shape[0], CHUNK_TOKENS):
            chunk_id = start // CHUNK_TOKENS
            block = pooled[start : start + CHUNK_TOKENS]
            sums[chunk_id] = block.sum(0)
            counts[chunk_id] = block.shape[0]
        expected[column] = (sums, counts)
    return expected


def test_token_entropies_uniform_and_peaked() -> None:
    """Softmax of a zero row is uniform (ln V); peaked logits give ~zero."""
    rows, vocab = 3, 40
    logits = torch.zeros(rows * 2 + 5, vocab)
    logits[2, 3] = 60.0  # -> essentially one-hot
    entropy = token_entropies(logits, chunk_rows=4)
    assert entropy.shape == (rows * 2 + 5,)
    assert abs(float(entropy[0]) - math.log(vocab)) < 1e-3
    assert float(entropy[2]) < 1e-4
