"""Offline layer-set analysis for the GLM-5.3-Flash drafter (map ticket 51).

Implements the ELSI/ELSI-C recipe of arXiv:2610.12939 ("Surfing the Activation
Wave") over GLM target hidden-stream columns: pooled block representatives,
Renyi-1 matrix entropy of concatenated sets (nats, the paper's S1), and linear
CKA with token-axis centering, plus both beam searches. Streaming chunk sums
keep CPU memory bounded: each stream column is scatter-added into global
32-token chunk buckets along one retained-token axis, so a candidate subset's
row Gram is the plain sum of the [U, U] Grams of its columns.

Two CLI entry points drive map ticket #51:

- ``stage1-cache``: statistics over the existing 686-sample 7-stream smoke
  cache (DeepSpec ``CacheDataset``), enumerating every 5/6/7-subset of the
  cached columns. The paper's predictive-entropy token filter is not
  available here (the cache stores no logits), so the token views are "all
  rendered positions" and "assistant-only (loss mask)".
- ``elsi-select``: statistics and both beam searches over the all-45-layer
  capture produced by ``elsi_capture_glm.py`` (46 columns: raw folded streams
  0..44 plus the post-final-norm stream 45), with the paper's per-sample
  top-rho predictive-entropy filter applied at capture time.
  ``--drop-post-norm`` re-runs on the all-raw pool (columns 0..44 only).

Column indexing is shared by the entry points: column ``c < N`` is the raw
(folded) output of decoder layer ``c``; column ``N`` (the 46th when N=45) is
the post-final-norm stream. DeepSpec canonical ``target_layer_ids`` map to
columns id-to-id, with the id ``N-1`` entry meaning the 46th column.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

from itertools import combinations

CHUNK_TOKENS = 32
POST_NORM_LABEL = "postN"
# The post-norm reads reference raw43/raw44/postN by id, so they can only be
# printed for captures with that many raw columns.
POST_NORM_READS_MIN_COLUMNS = 45
SANITY_SAMPLES = 3
# Kbeam = 10 is the paper's stated value (main text, citing supplementary
# fig. S3); Kexpand and Kpool have no stated value in the accessible paper
# text and are harness assumptions (recorded into select-report meta).
BEAM_EXPAND = 10
BEAM_WIDTH = 10
BEAM_POOL = 10
POOLED_VIEWS = (32, 256)
# Select-stage granularities: 839 is the recorded capture's chunk count
# (26,834 retained tokens / 32 = 839 chunks), so the 839 view is the
# pooled-grid identity: one rep row per 32-token chunk sum.
SELECT_POOLED_VIEWS = (32, 256, 839)
BASELINE_COLUMNS = (5, 15, 30, 42)
PROGRESS_EVERY = 64


def log(message: str) -> None:
    """Emit one script line immediately (the print-free lint gate owns this shape)."""
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def round6(value: float) -> float:
    """Round a float for JSON transport."""
    return round(float(value), 6)


class StreamAccumulator:
    """Growable global chunk sums for one stream column under one token view.

    Pooled position ``p`` (0-based along the view's retained-token axis in
    sample order) belongs to chunk ``p // CHUNK_TOKENS``. Chunk usage is a
    contiguous prefix because pooled positions ascend sample by sample.
    """

    def __init__(self, dim: int) -> None:
        """Start with empty zero-filled buffers of width ``dim``; growth doubles."""
        self._dim = int(dim)
        self._sums = torch.zeros((0, self._dim), dtype=torch.float32)
        self._counts = torch.zeros(0, dtype=torch.float32)

    def _ensure(self, capacity: int) -> None:
        if capacity <= self._sums.shape[0]:
            return
        grown = max(capacity, 2 * self._sums.shape[0], 4096)
        sums = torch.zeros((grown, self._dim), dtype=torch.float32)
        sums[: self._sums.shape[0]] = self._sums
        counts = torch.zeros(grown, dtype=torch.float32)
        counts[: self._counts.shape[0]] = self._counts
        self._sums = sums
        self._counts = counts

    def add(self, pooled_positions: torch.Tensor, tokens: torch.Tensor) -> None:
        """Scatter-add an ``[N, dim]`` fp32 token block at pooled positions."""
        if pooled_positions.numel() == 0:
            return
        need = int(pooled_positions.amax().item()) // CHUNK_TOKENS + 1
        self._ensure(need)
        chunk_ids = (pooled_positions // CHUNK_TOKENS).to(torch.int64)
        self._sums.index_add_(0, chunk_ids, tokens)
        ones = torch.ones(chunk_ids.shape[0], dtype=torch.float32)
        self._counts.index_add_(0, chunk_ids, ones)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop to the used prefix: sums ``[used, dim]`` and counts ``[used]``."""
        nonzero = self._counts.gt(0.0).nonzero().flatten()
        if nonzero.numel() == 0:
            msg = "no tokens have been accumulated in this view"
            raise ValueError(msg)
        used_chunks = int(nonzero.amax().item()) + 1
        return self._sums[:used_chunks].clone(), self._counts[:used_chunks].clone()


@dataclass(frozen=True)
class ColumnGrams:
    """Gram metas for one stream column in one pooled view."""

    column: int
    gram: torch.Tensor
    centered: torch.Tensor
    count: float

    @property
    def entropy(self) -> float:
        """Singleton Renyi-1 matrix entropy (nats)."""
        return matrix_entropy_nats(self.gram)


def chunk_rep_rows(
    sums: torch.Tensor, counts: torch.Tensor, u: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool chunk sums into ``u`` rep rows aligned on whole-chunk edges.

    Rep row ``r`` covers chunk ids ``[round(r*n/u), round((r+1)*n/u))`` where
    ``n`` is the chunk count. Returns the ``[u, dim]`` rep rows and the ``[u]``
    per-row token counts; rows covering zero tokens are all-zero (visible in
    the returned counts).
    """
    chunks = sums.shape[0]
    edges = [round(index * chunks / int(u)) for index in range(int(u) + 1)]
    sum_prefix = torch.cat(
        (torch.zeros(1, sums.shape[1], dtype=torch.float32), sums.cumsum(dim=0)), dim=0
    )
    count_prefix = torch.cat((torch.zeros(1, dtype=torch.float32), counts.cumsum(dim=0)), dim=0)
    row_sums = sum_prefix[edges[1:]] - sum_prefix[edges[:-1]]
    row_counts = count_prefix[edges[1:]] - count_prefix[edges[:-1]]
    reps = row_sums / row_counts.clamp_min(1.0).unsqueeze(1)
    return reps, row_counts


def column_grams(
    sums: torch.Tensor,
    counts: torch.Tensor,
    column: int,
    u: int,
    *,
    row_normalized: bool = False,
) -> ColumnGrams:
    """Build one column's Gram metas at pooled granularity ``u``.

    The uncentered Gram is ``X X^T`` (paper's K_X for S1); the centered one is
    ``(X - row_mean)(X - row_mean)^T`` (the paper's H K H with H = I - 11^T/T
    on the token axis). Centered Grams of concatenated sets decompose into
    per-block sums exactly because row centering subtracts each block's own
    per-row mean.

    ``row_normalized=True`` rescales every rep row to unit norm first: raw
    residual streams and the post-final-norm stream then carry comparable
    variance budgets (the final RMS norm inflates the normalized stream's
    spectrum), so mixed sets score directional structure, not scale artifacts.
    """
    reps, _row_counts = chunk_rep_rows(sums, counts, u)
    if row_normalized:
        norms = reps.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        reps = reps / norms
    gram = reps @ reps.T
    zero_mean = reps - reps.mean(dim=0, keepdim=True)
    centered = zero_mean @ zero_mean.T
    return ColumnGrams(column=column, gram=gram, centered=centered, count=float(counts.sum()))


def matrix_entropy_nats(gram: torch.Tensor) -> float:
    """Renyi-1 matrix entropy (nats, paper sec 3.1) of the reps behind a Gram."""
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    total = float(eigen.sum())
    if total <= 0.0:
        return 0.0
    # filter the normalized shares, not the raw eigenvalues: a subnormal
    # eigenvalue under a huge total rounds its share to 0.0, whose log would
    # NaN the sum; a zero share contributes nothing by convention anyway.
    shares = eigen / total
    weights = shares[shares > 0.0]
    return float(-(weights * torch.log(weights)).sum())


def cka_of_grams(centered_a: torch.Tensor, centered_b: torch.Tensor) -> float:
    """Linear CKA between two token-axis-centered row Grams (symmetrized)."""
    left = 0.5 * (centered_a + centered_a.T)
    right = 0.5 * (centered_b + centered_b.T)
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return 0.0
    return float((left * right).sum() / denominator)


class StatsView:
    """All stream columns of one pooled granularity plus subset math.

    Subset scores are additive: a concatenated set's row Gram is the
    elementwise sum of its member columns' Grams (the inner product of two
    concatenated rows decomposes over feature blocks), and centering a
    concatenated set subtracts each block's own row mean, so centered set
    Grams also sum exactly. Every candidate subset therefore scores in
    O(U**2) after one initialization.
    """

    def __init__(self, columns: list[ColumnGrams]) -> None:
        """Index the view's columns by their column id (duplicates replaced)."""
        self.items: dict[int, ColumnGrams] = {item.column: item for item in columns}

    def _stat(self, column: int) -> ColumnGrams:
        try:
            return self.items[column]
        except KeyError as exc:
            msg = f"column {column} not present in this view"
            raise ValueError(msg) from exc

    def __getitem__(self, column: int) -> ColumnGrams:
        """Column metas by id (syntactic sugar for ``_stat``)."""
        return self._stat(column)

    def set_gram(self, members: Iterable[int]) -> torch.Tensor:
        """Additive uncentered row Gram of the concatenated column set."""
        total: torch.Tensor | None = None
        for column in members:
            gram = self._stat(column).gram
            total = gram if total is None else total + gram
        if total is None:
            msg = "set Gram needs at least one column"
            raise ValueError(msg)
        return total

    def set_centered(self, members: Iterable[int]) -> torch.Tensor:
        """Additive row-centered Gram of the concatenated column set."""
        total: torch.Tensor | None = None
        for column in members:
            gram = self._stat(column).centered
            total = gram if total is None else total + gram
        if total is None:
            msg = "set centered Gram needs at least one column"
            raise ValueError(msg)
        return total

    def set_entropy(self, members: Iterable[int]) -> float:
        """S1 in nats of the concatenated set (the ELSI expansion score)."""
        return matrix_entropy_nats(self.set_gram(members))

    def cka(self, left: Iterable[int], right: Iterable[int]) -> float:
        """Linear CKA between two concatenated column sets."""
        left_columns = tuple(left)
        right_columns = tuple(right)
        return cka_of_grams(self.set_centered(left_columns), self.set_centered(right_columns))


def _top_beam(
    pairs: list[tuple[float, frozenset[int]]], width: int
) -> list[tuple[float, frozenset[int]]]:
    """Sort scored sets by entropy descending and keep ``width`` members."""
    return sorted(pairs, key=lambda item: item[0], reverse=True)[:width]


def _seed_beam(stats: StatsView, columns: list[int], width: int) -> list[frozenset[int]]:
    """Paper [S1]/[C1]: top-``width`` singletons by their own matrix entropy."""
    scored = [(stats[column].entropy, frozenset({column})) for column in columns]
    return [members for _score, members in _top_beam(scored, width)]


def elsi_rank(
    stats: StatsView,
    columns: list[int],
    size: int,
    k_expand: int = BEAM_EXPAND,
    k_beam: int = BEAM_WIDTH,
) -> list[tuple[float, frozenset[int]]]:
    """Paper Algorithm 1 (ELSI): entropy-driven beam search up to ``size``."""
    beam = _seed_beam(stats, columns, k_beam)
    for _depth in range(2, size + 1):
        proposals: dict[frozenset[int], float] = {}
        for parent in beam:
            scored = [
                (stats.set_entropy(parent | {column}), parent | {column})
                for column in columns
                if column not in parent
            ]
            for score, child in _top_beam(scored, k_expand):
                proposals[child] = max(proposals.get(child, 0.0), score)
        beam = [
            members
            for _score, members in _top_beam(
                [(score, members) for members, score in proposals.items()], k_beam
            )
        ]
    return _top_beam([(stats.set_entropy(members), members) for members in beam], k_beam)


def _conditional_cka_transitions(
    stats: StatsView, parent: frozenset[int], columns: list[int]
) -> dict[int, float]:
    """Per-candidate conditional-CKA transition score (paper Alg. 2 [C2]).

    Rows live in the conditional matrix M_L(i, j) = CKA(centered set Gram of
    the parents' set union with column i (then j)); a column already in the
    parent contributes a plain parent row (its Gram is not doubled). A
    candidate's score is delta_L(r) = ||M_L(r,:) - M_L(r-1,:)||_2 against the
    previous column id's row, so an exact content duplicate of the adjacent
    lower column coincides with that column's row and scores exactly 0.0
    (their Grams are bit-identical copies), while the harness's earlier
    surrogate scored Frobenius scale differences instead.
    """
    base = stats.set_centered(parent)
    added = {
        column: base if column in parent else base + stats[column].centered for column in columns
    }
    rows = {
        column: torch.tensor([cka_of_grams(added[column], added[other]) for other in columns])
        for column in columns
    }
    scores: dict[int, float] = {}
    for index in range(1, len(columns)):
        previous = columns[index - 1]
        scores[columns[index]] = float((rows[columns[index]] - rows[previous]).pow(2).sum().sqrt())
    return scores


def elsi_c_rank(
    stats: StatsView,
    columns: list[int],
    size: int,
    k_pool: int = BEAM_POOL,
    k_beam: int = BEAM_WIDTH,
) -> list[tuple[float, frozenset[int]]]:
    """Paper Algorithm 2 (ELSI-C): entropy init, conditional-CKA expansion.

    The transition score of candidate ``r`` is the Frobenius norm of the row
    difference of the paper's conditional-CKA matrix over strict set unions
    (``delta_L(r)``), so a candidate duplicating the adjacent lower column's
    content scores ~0 and is not pooled. The lowest column id gets no
    ``delta_L`` row (there is no r-1) and is therefore never pooled.
    Top-``k_pool`` candidates per parent feed the beam; dedup and entropy
    pruning close every depth.
    """
    beam = _seed_beam(stats, columns, k_beam)
    for _depth in range(2, size + 1):
        proposals: dict[frozenset[int], float] = {}
        for parent in beam:
            distances = _conditional_cka_transitions(stats, parent, columns)
            pool = sorted(
                (distances[column], column)
                for column in columns
                if column not in parent and column in distances
            )[-k_pool:]
            for _distance, column in pool:
                child = parent | {column}
                proposals[child] = max(proposals.get(child, 0.0), stats.set_entropy(child))
            del distances
        beam = [
            members
            for _score, members in _top_beam(
                [(score, members) for members, score in proposals.items()], k_beam
            )
        ]
    return _top_beam([(stats.set_entropy(members), members) for members in beam], k_beam)


def column_label(column: int, num_layers: int) -> str:
    """Canonical display label: ``rawN`` for raw streams, ``postN`` for the last."""
    if column < num_layers:
        return f"raw{column}"
    if column == num_layers:
        return POST_NORM_LABEL
    msg = f"column {column} out of range for {num_layers} raw layers"
    raise ValueError(msg)


def labels_of(members: Iterable[int], num_layers: int) -> list[str]:
    """Labels for a column set, order-normalized."""
    return [column_label(column, num_layers) for column in sorted(members)]


def _load_json(path: Path) -> dict[str, Any]:
    """Read one of this module's small report JSONs."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"expected a JSON object at {path}"
        raise TypeError(msg)
    return loaded


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Dump a report JSON, creating the parent directory once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


type Stage1LabelSet = list[str]


def _stage1_labels(layer_ids: list[int]) -> Stage1LabelSet:
    """Labels for cached columns: canonical raw ids, last carries the norm mark."""
    return [str(layer_id) for layer_id in layer_ids[:-1]] + [f"{layer_ids[-1]}*"]


def enumerate_subsets(
    stats: StatsView, columns: list[int], sizes: Iterable[int]
) -> list[dict[str, Any]]:
    """Rank every combination of the given sizes by Concatenated entropy score."""
    scored: list[tuple[float, tuple[int, ...]]] = []
    for size in sizes:
        for combo in combinations(columns, size):
            members = tuple(sorted(combo))
            scored.append((stats.set_entropy(members), members))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"members": list(members), "entropy": round6(score)} for score, members in scored]


def rank_scores(stats: StatsView, entries: list[tuple[str, list[int]]]) -> list[dict[str, Any]]:
    """Scores for named standing candidate sets (labels preserved)."""
    return [
        {"name": name, "columns": sorted(columns), "entropy": round6(stats.set_entropy(columns))}
        for name, columns in entries
    ]


def percentile_of(values: list[int], quantile: float) -> float:
    """Nearest-rank percentile of an integer sample (sorted once)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return float(ordered[rank])


@dataclass
class CacheSource:
    """Open DeepSpec cache plus the manifest fields the reports cite."""

    dataset: Any
    num_samples: int
    layer_ids: list[int]
    hidden_size: int
    max_length: int | None
    source_paths: list[str]


def open_cache(cache_dir: Path) -> CacheSource:
    """Open the smoke cache through DeepSpec's own dataset reader."""
    from deepspec.data.target_cache_dataset import CacheDataset  # vendored in-image

    dataset = CacheDataset(cache_dir=str(cache_dir))
    manifest = dataset.manifest
    extra = manifest.get("extra_fields", {})
    if not isinstance(extra, dict):
        extra = {}
    max_length = extra.get("max_length", manifest.get("max_length"))
    paths = extra.get("source_jsonl_paths", manifest.get("source_jsonl_paths", []))
    if not isinstance(paths, list):
        paths = [paths]
    return CacheSource(
        dataset=dataset,
        num_samples=int(manifest["num_samples"]),
        layer_ids=[int(value) for value in manifest["target_layer_ids"]],
        hidden_size=int(manifest["hidden_size"]),
        max_length=None if max_length is None else int(max_length),
        source_paths=[str(value) for value in paths],
    )


def stage1_accumulate(
    source: CacheSource, out_dir: Path
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Read every cached sample once, filling both token views' chunk sums.

    Two views share the 7-stream column layout: ``all`` keeps every rendered
    position on one pooled token axis; ``assistant`` keeps loss-mask positions
    only (its own axis). Snapshots go to ``stage1_sums.pt`` next to the report
    for re-analysis without re-reading the 33 GB cache.
    """
    dim = source.hidden_size * len(source.layer_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = {name: StreamAccumulator(dim) for name in ("all", "assistant")}
    running = dict.fromkeys(views, 0)
    seq_lens: list[int] = []
    assistant_counts: list[int] = []
    sanity: list[dict[str, Any]] = []
    for index in range(source.num_samples):
        item = source.dataset[index]
        hidden = item["target_hidden_states"].to(torch.float32)
        length = hidden.shape[0]
        seq_lens.append(int(length))
        views["all"].add(torch.arange(length) + running["all"], hidden)
        running["all"] += int(length)
        kept = item["loss_mask"].gt(0)
        nr = int(kept.sum())
        assistant_counts.append(nr)
        if nr > 0:
            views["assistant"].add(torch.arange(nr) + running["assistant"], hidden[kept])
            running["assistant"] += nr
        if index < SANITY_SAMPLES:
            post_norm_column = hidden[:, dim - source.hidden_size :]
            last = item["target_last_hidden_states"].to(torch.float32)
            diff = (post_norm_column - last).abs()
            sanity.append(
                {
                    "sample": index,
                    "max_abs": round6(float(diff.amax())),
                    "mean_abs": round6(float(diff.mean())),
                }
            )
        if (index + 1) % PROGRESS_EVERY == 0:
            log(f"[stage1] {index + 1}/{source.num_samples}")
    snapshots = {name: views[name].snapshot() for name in views}
    bundle = {name: {"sums": sums, "counts": counts} for name, (sums, counts) in snapshots.items()}
    torch.save(bundle, out_dir / "stage1_sums.pt")
    segment_meta = {
        "num_samples": source.num_samples,
        "assistant_counts": assistant_counts,
        "seq_p50": percentile_of(seq_lens, 0.50),
        "seq_p90": percentile_of(seq_lens, 0.90),
        "seq_p99": percentile_of(seq_lens, 0.99),
        "seq_max": max(seq_lens) if seq_lens else 0,
        "tokens_all": running["all"],
        "tokens_assistant": running["assistant"],
        "sanity_post_norm_column": sanity,
    }
    return snapshots, segment_meta


def _with_labels(entries: list[dict[str, Any]], label_map: dict[int, str]) -> list[dict[str, Any]]:
    """Attach canonical labels to enumerated subset entries."""
    return [
        {
            "members": entry["members"],
            "labels": [label_map[column] for column in entry["members"]],
            "entropy": entry["entropy"],
        }
        for entry in entries
    ]


def stage1_tables(
    snapshots: dict[str, tuple[torch.Tensor, torch.Tensor]],
    layer_ids: list[int],
    hidden_size: int,
) -> dict[str, Any]:
    """All stage-1 tables per token view and pooled granularity."""
    labels = _stage1_labels(layer_ids)
    label_map = dict(enumerate(labels))
    n_streams = len(layer_ids)
    standing = [
        ("base4 [1,12,23,34]", [0, 1, 2, 3]),
        ("base4 + 42", [0, 1, 2, 3, 4]),
        ("base4 + 43", [0, 1, 2, 3, 5]),
        ("base4 + 44*", [0, 1, 2, 3, 6]),
        ("6-stock (no 44*)", [0, 1, 2, 3, 4, 5]),
        ("7-union", list(range(n_streams))),
    ]
    tables: dict[str, Any] = {}
    for view, (sums, counts) in snapshots.items():
        per_u: dict[str, Any] = {}
        for variant_key, normalized in (("raw", False), ("rownorm", True)):
            for u in POOLED_VIEWS:
                columns = [
                    column_grams(
                        sums[:, column * hidden_size : (column + 1) * hidden_size],
                        counts,
                        column,
                        u,
                        row_normalized=normalized,
                    )
                    for column in range(n_streams)
                ]
                stats = StatsView(columns)
                entry_key = str(u) if variant_key == "raw" else f"{u}_rownorm"
                per_u[entry_key] = _enumerated_entry(stats, labels, label_map, n_streams, standing)
        tables[view] = {
            "labels": labels,
            "chunk_rows": int(counts.shape[0]),
            "token_total": float(counts.sum().item()),
            "views_u": per_u,
        }
    return tables


def _enumerated_entry(
    stats: StatsView,
    labels: list[str],
    label_map: dict[int, str],
    n_streams: int,
    standing: list[tuple[str, list[int]]],
) -> dict[str, Any]:
    """One pooled-view payload: per-layer entropy, CKA matrix, subsets."""
    return {
        "per_layer_s1_nats": {
            label: round6(stats[column].entropy) for column, label in enumerate(labels)
        },
        "cka_matrix": {
            "labels": labels,
            "values": _cka_matrix_values(stats),
        },
        "pairs": _with_labels(enumerate_subsets(stats, list(range(n_streams)), (2,)), label_map),
        "subsets_5_6_7": _with_labels(
            enumerate_subsets(stats, list(range(n_streams)), (5, 6, 7)),
            label_map,
        ),
        "standing_sets": rank_scores(stats, standing),
    }


def _cka_matrix_values(stats: StatsView) -> list[list[float]]:
    """Full pairwise CKA table over the view's columns, in id order.

    Pairs are computed in float64 so an identical pair reports its exact 1.0
    (fp32 roundoff printed the two-sided band 0.999996-1.000005 on the
    recorded data).
    """
    columns = [stats[column] for column in sorted(stats.items)]
    grams = [column_stat.centered.double() for column_stat in columns]
    return [
        [round6(cka_of_grams(row_gram, column_gram)) for column_gram in grams] for row_gram in grams
    ]


def run_stage1(cache_dir: str, out: str) -> None:
    """Stage-1 CLI body: statistics over the cached 7 streams (CPU only)."""
    out_path = Path(out)
    source = open_cache(Path(cache_dir))
    log(
        f"[stage1] cache {source.num_samples} samples, ids={source.layer_ids}, "
        f"hidden={source.hidden_size}"
    )
    snapshots, segment = stage1_accumulate(source, out_path.parent)
    tables = stage1_tables(snapshots, source.layer_ids, source.hidden_size)
    write_json(
        {
            "meta": {
                "cache_dir": str(Path(cache_dir)),
                "num_samples": source.num_samples,
                "target_layer_ids": source.layer_ids,
                "hidden_size": source.hidden_size,
                "max_length": source.max_length,
                "source_jsonl_paths": source.source_paths,
                "predictive_entropy_note": (
                    "the smoke cache stores no target logits, so the paper's "
                    "rho=0.3 token-entropy filter is unavailable here; token views "
                    "are all-positions and assistant-only (loss mask). Stage 2 "
                    "captures predictive entropies and applies the filter exactly."
                ),
            },
            "token_axis": segment,
            "tables": tables,
        },
        out_path,
    )
    log(f"[stage1] wrote {out_path}")


def _baseline_reps_for(
    sums: torch.Tensor, counts: torch.Tensor, u: int, *, normalized: bool
) -> torch.Tensor:
    """Baseline reps at granularity ``u`` (optionally row-normalized)."""
    reps, _row_counts = chunk_rep_rows(sums, counts, u)
    if normalized:
        norms = reps.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        reps = reps / norms
    return reps.clone()


def _assert_shared_grid(
    sums: torch.Tensor, counts: torch.Tensor, column: int, reference: torch.Tensor | None
) -> torch.Tensor | None:
    """Check one column's pooled grid; counts identity is the first (or kept).

    Raises when sums and counts rows disagree or the column's chunk counts
    differ from the reference column's: the additive set-Gram identity mixes
    Grams from different pooled grids silently otherwise.
    """
    if sums.shape[0] != counts.shape[0]:
        msg = f"column {column}: sums rows {sums.shape[0]} != counts rows {counts.shape[0]}"
        raise ValueError(msg)
    if reference is None:
        return counts
    if not torch.equal(counts, reference):
        msg = (
            f"column {column} pooled chunk grid differs from column 0 "
            "(counts identity must hold for the additive set-Gram identity)"
        )
        raise ValueError(msg)
    return reference


def _load_capture_views(
    capture_path: Path, num_columns: int, variants: list[tuple[str, int, bool]]
) -> tuple[dict[str, StatsView], dict[tuple[str, int], torch.Tensor]]:
    """Load every column's pooled Gram metas plus the per-(table-key) baseline reps."""
    pending: dict[str, list[ColumnGrams]] = {key: [] for key, _, _ in variants}
    baseline_reps: dict[tuple[str, int], torch.Tensor] = {}
    reference_counts: torch.Tensor | None = None
    for column in range(num_columns):
        path = capture_path / f"col_{column:03d}.pt"
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        sums = bundle["sums"]
        counts = bundle["counts"]
        reference_counts = _assert_shared_grid(sums, counts, column, reference_counts)
        for key, u, normalized in variants:
            pending[key].append(column_grams(sums, counts, column, u, row_normalized=normalized))
            if column in BASELINE_COLUMNS:
                baseline_reps[(key, column)] = _baseline_reps_for(
                    sums, counts, u, normalized=normalized
                )
        if (column + 1) % 8 == 0:
            log(f"[select] column {column + 1}/{num_columns}")
        del bundle, sums, counts
    views: dict[str, StatsView] = {key: StatsView(columns) for key, columns in pending.items()}
    return views, baseline_reps


def _select_variants(pooled_views: tuple[int, ...]) -> list[tuple[str, int, bool]]:
    """Interleaved raw/row-normalized variants for every pooled granularity."""
    variants: list[tuple[str, int, bool]] = []
    for u in pooled_views:
        variants.append((f"{u}", u, False))
        variants.append((f"{u}_rownorm", u, True))
    return variants


def run_select(capture_dir: str, out: str, *, drop_post_norm: bool = False) -> None:
    """Stage-3 CLI body: statistics and ELSI/ELSI-C over the captured columns.

    ``drop_post_norm`` builds the all-raw pool (columns 0..num_layers-1, the
    post-final-norm column excluded entirely) for the mixed-canon re-analysis.
    """
    capture_path = Path(capture_dir)
    meta = _load_json(capture_path / "capture_meta.json")
    num_layers = int(meta["num_layers"])
    num_columns = int(meta["num_columns"])
    if drop_post_norm:
        if num_columns != num_layers + 1:
            msg = f"capture holds {num_columns} columns, expected num_layers+1 for the all-raw drop"
            raise ValueError(msg)
        num_columns = num_layers
    views, baseline_reps = _load_capture_views(
        capture_path, num_columns, _select_variants(SELECT_POOLED_VIEWS)
    )
    arms = [
        ("7-union [1,12,23,34,42,43,44*]", [1, 12, 23, 34, 42, 43, num_layers]),
        ("5-stock [1,12,23,34,42]", [1, 12, 23, 34, 42]),
        ("5-spread [1,12,23,34,44*]", [1, 12, 23, 34, num_layers]),
        ("6-stock [1,12,23,34,42,43]", [1, 12, 23, 34, 42, 43]),
        ("base4 [1,12,23,34]", [1, 12, 23, 34]),
        ("DFlash2 convention [5,14,24,33,42]", [5, 14, 24, 33, 42]),
    ]
    arms = [arm for arm in arms if all(column < num_columns for column in arm[1])]
    per_u_tables: dict[str, Any] = {}
    for key, stats in views.items():
        labels = [column_label(column, num_layers) for column in range(num_columns)]
        entry = _select_entry(stats, labels, num_layers, arms)
        baseline = {
            column: baseline_reps[(key, column)]
            for column in BASELINE_COLUMNS
            if (key, column) in baseline_reps
        }
        entry["chance_baseline"] = chance_baseline(baseline)
        per_u_tables[key] = entry
    payload_meta: dict[str, Any] = {
        "capture_dir": str(capture_path),
        "pool": (
            "all-raw: only the raw folded stream columns enter the pool "
            "(the post-final-norm column is excluded entirely)"
            if drop_post_norm
            else "mixed: raw folded streams plus the post-final-norm stream"
        ),
        "search_constants": {
            "k_beam": BEAM_WIDTH,
            "k_expand": BEAM_EXPAND,
            "k_pool": BEAM_POOL,
            "k_beam_source": "paper-stated (main text, citing supplementary fig. S3: Kbeam=10)",
            "k_expand_k_pool_source": (
                "harness assumption; no value for Kexpand/Kpool is stated in "
                "the accessible paper text"
            ),
        },
        **meta,
    }
    write_json({"meta": payload_meta, "tables_by_u": per_u_tables}, Path(out))
    log(f"[select] wrote {Path(out)}")


def _select_entry(
    stats: StatsView,
    labels: list[str],
    num_layers: int,
    arms: list[tuple[str, list[int]]],
) -> dict[str, Any]:
    """One select payload: layers, CKA matrix, arms, beams, post-norm reads.

    The printed CKA matrix is computed in float64 so the diagonal reads
    exactly 1.0 instead of fp32's two-sided ±5e-6 roundoff band.
    """
    num_columns = len(labels)
    values = _cka_matrix_values(stats)
    beam_entries = {
        f"elsi_m{size}_top": [
            {"entropy": round6(score), "labels": labels_of(members, num_layers)}
            for score, members in elsi_rank(stats, list(range(num_columns)), size)[:3]
        ]
        for size in (5, 7)
    }
    beam_c_entries = {
        f"elsi_c_m{size}_top": [
            {"entropy": round6(score), "labels": labels_of(members, num_layers)}
            for score, members in elsi_c_rank(stats, list(range(num_columns)), size)[:3]
        ]
        for size in (5, 7)
    }
    payload: dict[str, Any] = {
        "per_layer_s1_nats": {
            labels[column]: round6(stats[column].entropy) for column in range(num_columns)
        },
        "cka_matrix": {"labels": labels, "values": values},
        "transition_peaks": _transition_peaks(values),
        "standing_sets": rank_scores(stats, arms),
        **beam_entries,
        **beam_c_entries,
    }
    if num_layers < num_columns and num_columns >= POST_NORM_READS_MIN_COLUMNS:
        payload["post_norm_reads"] = {
            "cka_raw43_postN": round6(stats.cka([43], [num_layers])),
            "cka_raw44_postN": round6(stats.cka([44], [num_layers])),
            "cka_raw43_raw44": round6(stats.cka([43], [44])),
            "s1_5stock": round6(stats.set_entropy([1, 12, 23, 34, 42])),
            "s1_5stock_plus_43": round6(stats.set_entropy([1, 12, 23, 34, 42, 43])),
            "s1_5stock_plus_postN": round6(stats.set_entropy([1, 12, 23, 34, 42, num_layers])),
        }
    return payload


def chance_baseline(baseline_reps: dict[int, torch.Tensor]) -> dict[str, Any]:
    """Row-permuted chance CKA band from a few designated columns.

    With thousands of features but few pooled rows, independent columns have a
    nonzero finite-sample CKA; these row-shuffled pairs define the noise band
    a real redundancy read must rise above (paper's validation is adjacent
    CKA around 0.7+ at the top of the depth ladder).
    """
    generator = torch.Generator().manual_seed(1234)
    pair_reports: list[dict[str, Any]] = []
    identifiers = sorted(baseline_reps)
    for left_index, left in enumerate(identifiers):
        for right in identifiers[left_index + 1 :]:
            rows = baseline_reps[right]
            permuted = rows[torch.randperm(rows.shape[0], generator=generator)]
            pair_reports.append(
                {
                    "pair": [left, right],
                    "cka": round6(
                        cka_of_grams(
                            centered_gram_of_rows(baseline_reps[left]),
                            centered_gram_of_rows(permuted),
                        )
                    ),
                }
            )
    values = [pair["cka"] for pair in pair_reports]
    return {
        "columns": identifiers,
        "pairs": pair_reports,
        "band": {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
        if values
        else {"min": 0.0, "mean": 0.0, "max": 0.0},
        "note": (
            "CKA of row-shuffled pairs fixes the finite-sample noise floor; "
            "features greatly outnumber pooled rows, so independent columns do "
            "not score near zero"
        ),
    }


def centered_gram_of_rows(reps: torch.Tensor) -> torch.Tensor:
    """Direct row-centered Gram of a rep matrix (no chunk machinery)."""
    zero_mean = reps - reps.mean(dim=0, keepdim=True)
    return zero_mean @ zero_mean.T


def _transition_peaks(cka_values: list[list[float]]) -> list[dict[str, Any]]:
    """Per-layer CKA transition deltas from the pairwise matrix (paper Eq. 8).

    ``values[i][j]`` is CKA(column i, column j); the transition at boundary
    ``i -> i+1`` is the full-row ``delta_l = ||M_{l+1,:} - M_{l,:}||_2``, self
    and adjacent entries included exactly as Eq. 8 defines it (measured on
    the recorded data: top-5 peak boundaries and order unchanged, large
    boundary deltas shifted ~2-9%, deep-tail deltas under ~2.3%).
    Returns every boundary sorted by delta descending.
    """
    deltas: list[tuple[float, int]] = []
    count = len(cka_values)
    for left in range(count - 1):
        right = left + 1
        squared = [
            (cka_values[right][index] - cka_values[left][index]) ** 2 for index in range(count)
        ]
        deltas.append((float(sum(squared)) ** 0.5, left))
    deltas.sort(key=lambda item: item[0], reverse=True)
    return [
        {"boundary": f"col{left} -> col{left + 1}", "delta": round6(delta)}
        for delta, left in deltas
    ]


def main() -> None:
    """CLI surface: two subcommands over the same analysis core."""
    parser = argparse.ArgumentParser(
        description="ELSI/ELSI-C layer-set analysis for GLM-5.3-Flash drafter work (issue 51)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage1 = subparsers.add_parser(
        "stage1-cache", help="statistics over the cached 7 streams (CPU only)."
    )
    stage1.add_argument("--cache-dir", required=True, help="smoke-cache directory.")
    stage1.add_argument("--out", required=True, help="report JSON path.")
    select = subparsers.add_parser(
        "elsi-select", help="statistics + beam searches over the captured 46 columns."
    )
    select.add_argument("--capture-dir", required=True, help="capture output directory.")
    select.add_argument("--out", required=True, help="report JSON path.")
    select.add_argument(
        "--drop-post-norm",
        action="store_true",
        help="build the all-raw pool: exclude the post-final-norm column entirely.",
    )
    args = parser.parse_args()
    if args.command == "stage1-cache":
        run_stage1(args.cache_dir, args.out)
    else:
        run_select(args.capture_dir, args.out, drop_post_norm=args.drop_post_norm)


if __name__ == "__main__":
    main()
