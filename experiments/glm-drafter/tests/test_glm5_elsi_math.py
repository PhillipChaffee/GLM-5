"""Known-answer tests for the ELSI analysis math (repo issue 51).

Deterministic synthetic checks for every primitive the ticket's numbers rest
on, plus subset-additivity identities exercised through a direct
(non-streaming) path. Runs under the torch venv; the gate environment skips
this directory (no torch there by design).
"""

from __future__ import annotations

import math

import pytest
import torch
from elsi_analyze import (
    CHUNK_TOKENS,
    StatsView,
    StreamAccumulator,
    chunk_rep_rows,
    cka_of_grams,
    column_grams,
    elsi_c_rank,
    elsi_rank,
    matrix_entropy_nats,
)


def _dense_reference(
    positions: torch.Tensor, tokens: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent non-streaming chunk-sum implementation (boolean masks)."""
    total = int(positions.amax().item()) + 1
    chunks = (total + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    sums = torch.zeros(chunks, dim, dtype=torch.float32)
    counts = torch.zeros(chunks, dtype=torch.float32)
    flat = torch.zeros(total, dim, dtype=torch.float32)
    flat[positions] = tokens
    for chunk_id in range(chunks):
        start = chunk_id * CHUNK_TOKENS
        block = flat[start : start + CHUNK_TOKENS]
        sums[chunk_id] = block.sum(dim=0)
        counts[chunk_id] = block.shape[0]
    return sums, counts


def test_accumulator_matches_dense_reference() -> None:
    """Multi-add sequence (with the same offsets the capture loop produces)."""
    torch.manual_seed(11)
    dim = 64
    lengths = (9, 33, 15, 40, 7)
    acc = StreamAccumulator(dim)
    all_positions = []
    all_tokens = []
    running = 0
    for length in lengths:
        block = torch.randn(length, dim)
        positions = torch.arange(length) + running
        acc.add(positions, block)
        all_positions.append(positions)
        all_tokens.append(block)
        running += length
    positions = torch.cat(all_positions)
    tokens = torch.cat(all_tokens)
    sums, counts = acc.snapshot()
    expect_sums, expect_counts = _dense_reference(positions, tokens, dim)
    assert sums.shape == expect_sums.shape
    assert torch.allclose(counts, expect_counts)
    assert torch.allclose(sums, expect_sums, rtol=1e-4, atol=1e-4)


def test_chunk_rep_rows_matches_direct_token_means() -> None:
    """Pooled rows equal the mean over token ranges implied by chunk edges."""
    torch.manual_seed(7)
    tokens = torch.randn(97, 16)  # 4 chunks of 32-token sums: 32,32,32,1
    sums = torch.stack([tokens[start : start + 32].sum(0) for start in range(0, 97, 32)])
    counts = torch.tensor([32.0, 32.0, 32.0, 1.0])
    reps, row_counts = chunk_rep_rows(sums, counts, 2)
    assert reps.shape == (2, 16)
    assert torch.allclose(row_counts, torch.tensor([64.0, 33.0]))
    assert torch.allclose(reps[0], tokens[0:64].mean(0), rtol=1e-5, atol=1e-5)
    assert torch.allclose(reps[1], tokens[64:97].mean(0), rtol=1e-5, atol=1e-5)


def test_entropy_known_answers() -> None:
    """Rank-1 -> 0; equal spectrum of size n -> ln n; empty -> 0."""
    rank1 = torch.diag(torch.tensor([7.0, 0.0, 0.0]))
    assert matrix_entropy_nats(rank1) == pytest.approx(0.0, abs=1e-6)
    flat = torch.eye(4) * 2.0
    assert matrix_entropy_nats(flat) == pytest.approx(math.log(4.0), rel=1e-5)
    assert matrix_entropy_nats(torch.zeros(3, 3)) == 0.0


def test_cka_identical_and_independent() -> None:
    """Identical rows give CKA 1; independent random columns score small.

    CKA is feature-rotation invariant (both Grams depend only on row inner
    products), so "exactly 0" is not achievable with a precondition of
    independence alone; the load-bearing bounds are the 1.0 identity and the
    small-noise baseline, both pinned by the seed.
    """
    torch.manual_seed(5)
    rows = 24
    reps_a = torch.randn(rows, 16)
    ones = torch.ones(rows)
    grams_a = column_grams(reps_a, ones, 0, rows)
    grams_dup = column_grams(reps_a.clone(), ones.clone(), 1, rows)
    stats = StatsView([grams_a, grams_dup])
    identical = cka_of_grams(grams_a.centered, grams_a.centered.clone())
    assert identical == pytest.approx(1.0, abs=1e-6)
    assert stats.cka([0], [1]) == pytest.approx(1.0, abs=1e-6)
    independent = StatsView(
        [column_grams(torch.randn(rows, 16), ones.clone(), column, rows) for column in (0, 1)]
    )
    value = independent.cka([0], [1])
    assert value != pytest.approx(1.0, abs=0.1)
    assert 0.05 < value < 0.45


def test_subset_additivity_and_duplicate_invariance() -> None:
    """Concat-set Gram == sum of per-column Grams (both centered and raw)."""
    torch.manual_seed(19)
    rows, dim = 24, 8
    reps = [torch.randn(rows, dim) for _ in range(3)]
    stats = StatsView(
        [column_grams(reps[index], torch.ones(rows), index, rows) for index in range(3)]
    )
    concat = torch.cat(reps, dim=1)
    direct_gram = concat @ concat.T
    direct_zero_mean = concat - concat.mean(0, keepdim=True)
    direct_centered = direct_zero_mean @ direct_zero_mean.T
    assert torch.allclose(stats.set_gram([0, 1, 2]), direct_gram, rtol=1e-5, atol=1e-5)
    assert torch.allclose(stats.set_centered([0, 1, 2]), direct_centered, rtol=1e-5, atol=1e-5)
    assert stats.set_entropy([0, 1, 2]) == pytest.approx(matrix_entropy_nats(direct_gram), rel=1e-5)
    assert math.isclose(
        stats.cka([0], [2]), cka_of_grams(stats.set_centered([0]), stats.set_centered([2]))
    )
    direct_a = _center_gram_rows(reps[0])
    direct_c = _center_gram_rows(reps[2])
    assert stats.cka([0], [2]) == pytest.approx(cka_of_grams(direct_a, direct_c), abs=1e-6)
    left_reps = torch.cat([reps[0], reps[1]], dim=1)
    right_reps = torch.cat([reps[1], reps[2]], dim=1)
    assert stats.cka([0, 1], [1, 2]) == pytest.approx(
        cka_of_grams(_center_gram_rows(left_reps), _center_gram_rows(right_reps)),
        abs=1e-6,
    )
    # a duplicate column rescales singular values only: entropy unchanged
    duplicate = column_grams(reps[0], torch.ones(rows), 9, rows)
    extended = StatsView([stats.items[0], duplicate])
    assert extended.set_entropy([0, 9]) == pytest.approx(extended.set_entropy([0]), rel=1e-5)


def _center_gram_rows(reps: torch.Tensor) -> torch.Tensor:
    """Row-centered Gram computed the direct way (no chunk machinery)."""
    zero_mean = reps - reps.mean(dim=0, keepdim=True)
    return zero_mean @ zero_mean.T


def test_elsi_planted_pair_and_determinism() -> None:
    """ELSI picks two rich spectra over a rank-1 filler; repeats deterministically."""
    torch.manual_seed(31)
    reps = [
        torch.randn(32, 16),  # full-rank spectrum
        torch.randn(32, 16),  # independent full-rank spectrum
        torch.randn(32, 16)[:1].expand(32, 16).clamp(-1.0, 1.0),  # rank-1 filler
    ]
    stats = StatsView([column_grams(reps[index], torch.ones(32), index, 32) for index in range(3)])
    first = elsi_rank(stats, [0, 1, 2], 2)
    repeat = elsi_rank(stats, [0, 1, 2], 2)
    assert first == repeat
    assert all(len(members) == 2 for _score, members in first)
    assert first[0][1] == frozenset({0, 1})
    c_ranked = elsi_c_rank(stats, [0, 1, 2], 3)
    c_repeat = elsi_c_rank(stats, [0, 1, 2], 3)
    assert c_ranked == c_repeat
    assert all(len(members) == 3 for _score, members in c_ranked)
