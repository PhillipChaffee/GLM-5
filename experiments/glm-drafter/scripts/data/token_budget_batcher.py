"""Token-budget batching for the hidden-state capture pass.

Implements the design recorded in
``docs/research/token-budget-cache-batching.md``: fairseq-style greedy
packing over a length-sorted stream, admitting whole conversations only and
bounded by a token budget and a conversation-count cap. Admission keys on
the pack's **padded peak** (rows times the pack's longest row), so a packed
forward pass never exceeds the budget the OOM-boundary probe measured,
whatever mix of long and short rows leads it. Each resumable slice derives
its own packs from the canonical descending sort over that slice's rows;
write order is free downstream (the trainer reads only the cache index).
"""

from collections.abc import Sequence

type Lengths = Sequence[int]


def sorted_order(lengths: Lengths) -> list[int]:
    """Source indices sorted by descending token length, ties by source order.

    Descending makes the pack's row-order first element its longest member,
    which is what lets admission close on a padded-peak bound directly.
    """
    return sorted(range(len(lengths)), key=lambda index: (-int(lengths[index]), index))


def build_packs(
    lengths: Lengths,
    *,
    token_budget: int,
    max_count: int,
    max_row_tokens: int,
) -> list[list[int]]:
    """Greedy padded-peak packing of ``lengths`` into whole-sequence packets.

    Each pack is a list of source indices whose padded peak — its row
    count times its longest row, which is the forward pass's activation
    token count after padding to the in-batch longest — stays within
    ``token_budget``, and whose size stays within ``max_count``; every row
    is admitted exactly once. ``max_row_tokens`` (the collator's
    ``max_length``) is required: the budget must cover it so one full-length
    row always fits a solo pack declaratively, and rows longer than it fail
    closed instead of silently inflating a pack's padded peak.
    """
    if token_budget < 1:
        budget_error = f"token_budget must be positive, got {token_budget}"
        raise ValueError(budget_error)
    if max_count < 1:
        cap_error = f"max_count must be positive, got {max_count}"
        raise ValueError(cap_error)
    if token_budget < int(max_row_tokens):
        budget_error = (
            "token_budget must cover the collator's max_length "
            f"({token_budget} < {max_row_tokens}); otherwise whole-sequence "
            "admission cannot bound the packed forward pass"
        )
        raise ValueError(budget_error)
    packs: list[list[int]] = []
    current: list[int] = []
    for index in sorted_order(lengths):
        row_tokens = int(lengths[index])
        if row_tokens > int(max_row_tokens):
            cap_error = (
                f"row {index} is {row_tokens} tokens, beyond the declared "
                f"max_row_tokens {max_row_tokens}; lengths must be "
                "post-truncation for the padded-peak bound to hold"
            )
            raise ValueError(cap_error)
        if current and (
            (len(current) + 1) * int(lengths[current[0]]) > token_budget
            or len(current) >= max_count
        ):
            packs.append(current)
            current = []
        current.append(index)
    if current:
        packs.append(current)
    return packs


def padded_token_count(lengths: Lengths, packs: Sequence[Sequence[int]]) -> int:
    """Total padded tokens across packs: rows times each pack's longest row.

    Equal to the model's total forward activation token count after padding
    to the in-batch longest; `build_packs` bounds each pack's share of it
    by ``token_budget``, which is the quantity the OOM-boundary probe is
    measured against (rows x max_length equivalents).
    """
    total = 0
    for pack in packs:
        longest = max(int(lengths[index]) for index in pack)
        total += longest * len(pack)
    return total
