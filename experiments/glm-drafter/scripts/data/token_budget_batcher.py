"""Token-budget batching for the hidden-state capture pass.

Implements the design recorded in
``docs/research/token-budget-cache-batching.md``: fairseq-style greedy
packing over a length-sorted stream, admitting whole conversations only and
bounded simultaneously by a token budget and a conversation-count cap. The
sort is strictly descending post-truncation token length with source-index
tie-breaks, so every rerun derives a byte-identical order and the resumable
build's slices are ranges of one canonical ordering.
"""

from collections.abc import Sequence

type Lengths = Sequence[int]


def sorted_order(lengths: Lengths) -> list[int]:
    """Source indices sorted by descending token length, ties by source order.

    Descending puts the densest forward pass first: the OOM boundary is
    reached deliberately at the start of a build rather than hours into it,
    matching HF's ``group_by_length`` rationale and the probe protocol.
    """
    return sorted(range(len(lengths)), key=lambda index: (-int(lengths[index]), index))


def build_packs(
    lengths: Lengths,
    *,
    token_budget: int,
    max_count: int,
    max_row_tokens: int | None = None,
) -> list[list[int]]:
    """Greedy token-budget packing of ``lengths`` into whole-sequence packets.

    Each pack is a list of source indices whose truncated token lengths sum
    to at most ``token_budget`` and whose size is at most ``max_count``; a
    row too large for the budget starts and ends its own pack, so every row
    is admitted exactly once (the ``budget >= max_length`` rule pins that
    case to the truncated tail only). ``max_row_tokens``, when given, is the
    collator's ``max_length``: the budget must cover it so the rule can
    never bind, and passing it fails closed instead of relying on callers.
    """
    if token_budget < 1:
        budget_error = f"token_budget must be positive, got {token_budget}"
        raise ValueError(budget_error)
    if max_count < 1:
        cap_error = f"max_count must be positive, got {max_count}"
        raise ValueError(cap_error)
    if max_row_tokens is not None and token_budget < int(max_row_tokens):
        budget_error = (
            "token_budget must cover the collator's max_length "
            f"({token_budget} < {max_row_tokens}); otherwise whole-sequence "
            "admission cannot bound the packed forward pass"
        )
        raise ValueError(budget_error)
    packs: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index in sorted_order(lengths):
        row_tokens = int(lengths[index])
        if current and (current_tokens + row_tokens > token_budget or len(current) >= max_count):
            packs.append(current)
            current = []
            current_tokens = 0
        current.append(index)
        current_tokens += row_tokens
    if current:
        packs.append(current)
    return packs


def padded_token_count(lengths: Lengths, packs: Sequence[Sequence[int]]) -> int:
    """Total padded tokens across packs: rows times each pack's longest row.

    Equal to the model's per-forward activation token count after padding to
    the in-batch longest, so it is the quantity a budget probe is measured
    against and the throughput table's denominator.
    """
    total = 0
    for pack in packs:
        longest = max(int(lengths[index]) for index in pack)
        total += longest * len(pack)
    return total
