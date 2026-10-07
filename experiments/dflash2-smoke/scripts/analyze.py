"""Analyze the DFlash2 smoke results: losslessness diff, tau tables, ladder shape.

Reads the three arm result jsons (dflash, mtp, plain), diffs the temp-0
completions of the drafter arm against the spec-off reference, and prints the
per-task tau and concurrency-ladder tables with speedups against the spec-off
floor.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def load_arm(results_dir: Path, tag: str) -> Row:
    """Load one arm's result json."""
    path = results_dir / f"{tag}.json"
    with path.open(encoding="utf-8") as handle:
        data: Row = json.load(handle)
    return data


def diff_temp0(dflash: Row, plain: Row) -> Row:
    """Exact string diff of the two arms' temp-0 completions."""
    ref = {row["id"]: row for row in plain["temp0"]["requests"]}
    rows: list[Row] = []
    for row in dflash["temp0"]["requests"]:
        other = ref.get(row["id"])
        if other is None:
            msg = f"temp0 id {row['id']} missing from plain arm"
            raise RuntimeError(msg)
        same = row.get("text") == other.get("text")
        rows.append(
            {
                "id": row["id"],
                "match": same,
                "dflash_tokens": row.get("completion_tokens"),
                "plain_tokens": other.get("completion_tokens"),
            }
        )
    mismatches = [row for row in rows if not row["match"]]
    return {"n": len(rows), "matches": len(rows) - len(mismatches), "mismatches": mismatches}


def suite_table(arm: Row) -> list[Row]:
    """Flatten one arm's suite rows into printable table rows."""
    return [
        {
            "task": row["window"].split(":", 1)[1],
            "tau": row.get("tau_counter"),
            "tok_s": row.get("aggregate_tok_s"),
            "per_user_tok_s": row.get("per_user_tok_s_mean"),
            "n": row["n"],
            "wall_s": row.get("wall_s"),
        }
        for row in arm.get("suite", {}).get("rows", [])
    ]


def ladder_table(arm: Row) -> list[Row]:
    """Flatten one arm's ladder rows into printable table rows."""
    return [
        {
            "rung": row["window"].split(":c", 1)[1],
            "tau": row.get("tau_counter"),
            "agg_tok_s": row.get("aggregate_tok_s"),
            "per_user_tok_s": row.get("per_user_tok_s_mean"),
            "p50_lat_s": row.get("p50_latency_s"),
        }
        for row in arm.get("ladder", {}).get("rows", [])
    ]


def _fmt(value: float | str | None, spec: str = ".2f") -> str:
    """Format one nullable metric cell for the tables."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return format(value, spec)


def print_table(title: str, rows: list[Row], columns: list[tuple[str, str, str]]) -> None:
    """Print one aligned table: columns are (header, key, format spec)."""
    sys.stdout.write(f"\n{title}\n")
    cells = [[_fmt(row.get(key), fmt) for _, key, fmt in columns] for row in rows]
    header = [head for head, _, _ in columns]
    widths = [
        max([len(head)] + [len(line[i]) for line in cells])
        for i, (head, _, _) in enumerate(columns)
    ]
    sys.stdout.write(
        "  ".join(head.ljust(w) for head, w in zip(header, widths, strict=True)) + "\n"
    )
    for line in cells:
        sys.stdout.write(
            "  ".join(text.ljust(w) for text, w in zip(line, widths, strict=True)) + "\n"
        )
    sys.stdout.flush()


def _ranked_top(top: dict[str, float] | None) -> list[tuple[str, float]]:
    """Return one position's token-logprob pairs, best candidate first."""
    return sorted((top or {}).items(), key=lambda item: item[1], reverse=True)


def _first_divergence(spec_tokens: list[str], ref_tokens: list[str]) -> int | None:
    """Index of the first differing token, or the shorter length on a prefix."""
    for index in range(min(len(spec_tokens), len(ref_tokens))):
        if spec_tokens[index] != ref_tokens[index]:
            return index
    if len(spec_tokens) != len(ref_tokens):
        return min(len(spec_tokens), len(ref_tokens))
    return None


def _classify_row(row: Row, other: Row) -> Row:
    """Classify one divergent prompt against the spec-off reference row."""
    spec_tokens = list(row.get("tokens") or [])
    ref_tokens = list(other.get("tokens") or [])
    position = _first_divergence(spec_tokens, ref_tokens)
    entry: Row = {
        "id": row["id"],
        "divergence_position": position,
        "spec_len": len(spec_tokens),
        "ref_len": len(ref_tokens),
    }
    if position is None:
        return entry
    tops = other.get("top_logprobs") or []
    ranked = _ranked_top(tops[position] if position < len(tops) else None)
    if not ranked:
        return entry
    spec_token = spec_tokens[position] if position < len(spec_tokens) else None
    entry.update(
        {
            "ref_top1": ranked[0][0],
            "ref_top2": ranked[1][0],
            "ref_gap_top1_top2": ranked[0][1] - ranked[1][1],
            "spec_token": spec_token,
            "spec_token_rank_in_ref": next(
                (rank for rank, (token, _) in enumerate(ranked) if token == spec_token),
                None,
            ),
            "spec_token_logprob_in_ref": (tops[position] or {}).get(spec_token),
        }
    )
    return entry


def classify_probe(results_dir: Path) -> Row:
    """Classify each divergent prompt: tie-flip vs machinery bug via top-2 gap."""
    dflash = load_arm(results_dir, "probe-dflash")
    plain = load_arm(results_dir, "probe-plain")
    ref = {row["id"]: row for row in plain["requests"]}
    rows = [_classify_row(row, ref[row["id"]]) for row in dflash["requests"]]
    return {"n": len(rows), "rows": rows}


def main() -> None:
    """Run the full analysis over the three arm result files."""
    parser = argparse.ArgumentParser(
        description="Losslessness diff plus tau and ladder tables for the smoke."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    if args.probe:
        probe = classify_probe(args.results_dir)
        sys.stdout.write(f"probe: {probe['n']} divergent prompts\n")
        for row in probe["rows"]:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if args.out:
            with args.out.open("w", encoding="utf-8") as handle:
                json.dump(probe, handle, indent=2, ensure_ascii=False)
            sys.stdout.write(f"wrote {args.out}\n")
            sys.stdout.flush()
        return

    dflash = load_arm(args.results_dir, "dflash")
    mtp = load_arm(args.results_dir, "mtp")
    plain = load_arm(args.results_dir, "plain")

    diff = diff_temp0(dflash, plain)
    report: Row = {
        "dflash_boot": dflash["boot_info"],
        "mtp_boot": mtp["boot_info"],
        "plain_boot": plain["boot_info"],
        "gpu_names": {
            tag: arm.get("gpu_names")
            for tag, arm in (("dflash", dflash), ("mtp", mtp), ("plain", plain))
        },
        "losslessness": diff,
        "suite_dflash": suite_table(dflash),
        "suite_mtp": suite_table(mtp),
        "ladder_dflash": ladder_table(dflash),
        "ladder_mtp": ladder_table(mtp),
        "ladder_plain": ladder_table(plain),
    }

    sys.stdout.write(f"losslessness: {diff['matches']}/{diff['n']} temp-0 completions identical\n")
    for row in diff["mismatches"]:
        sys.stdout.write(f"  MISMATCH {row['id']}: {row}\n")
    sys.stdout.flush()
    print_table(
        "suite (arm A = DFlash2, c1, temp 1.0):",
        report["suite_dflash"],
        [
            ("task", "task", ""),
            ("tau", "tau", ".3f"),
            ("tok/s", "tok_s", ".1f"),
            ("per-user tok/s", "per_user_tok_s", ".1f"),
            ("n", "n", "d"),
        ],
    )
    print_table(
        "suite (arm B = native MTP, c1, temp 1.0):",
        report["suite_mtp"],
        [
            ("task", "task", ""),
            ("tau", "tau", ".3f"),
            ("tok/s", "tok_s", ".1f"),
            ("per-user tok/s", "per_user_tok_s", ".1f"),
            ("n", "n", "d"),
        ],
    )
    for tag, key in (
        ("A (DFlash2)", "ladder_dflash"),
        ("B (MTP)", "ladder_mtp"),
        ("C (off)", "ladder_plain"),
    ):
        print_table(
            f"ladder (arm {tag}, alpaca-256, temp 0):",
            report[key],
            [
                ("c", "rung", ""),
                ("tau", "tau", ".3f"),
                ("agg tok/s", "agg_tok_s", ".1f"),
                ("per-user tok/s", "per_user_tok_s", ".1f"),
                ("p50 lat s", "p50_lat_s", ".2f"),
            ],
        )
    if args.out:
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        sys.stdout.write(f"wrote {args.out}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
