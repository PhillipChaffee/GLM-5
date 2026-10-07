"""Build the DFlash2 smoke's fixed prompt suites from DeepSpec's eval datasets.

Renders every prompt through the GLM-5.3-Flash chat template (reasoning effort
low, clear thinking) so all three serving arms receive byte-identical strings.
Source datasets come from a DeepSpec checkout at DEEPSPEC_SHA (the same commit
the proxy run validated); the tokenizer is the target's, so the rendered
generation prompt ends in the model's thinking-open marker.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Protocol

from transformers import AutoTokenizer

DEEPSPEC_SHA = "005e03b81cec38b7da6399833d609ee89a2587f2"
DEFAULT_TOKENIZER = "zai-org/GLM-5.3-Flash"
TASK_CAPS: dict[str, int] = {
    "gsm8k": 80,
    "math500": 60,
    "humaneval": 60,
    "mbpp": 60,
    "mt-bench": 80,
    "alpaca": 80,
}
LADDER_TASK = "alpaca"
LADDER_SIZE = 256
TEMP0_SOURCE = "temp0_check.jsonl"
TEMP0_OUTPUT = "temp0_prompts.jsonl"
DEFAULT_SEED = 42

Message = dict[str, str]
Row = dict[str, Any]


class ChatTokenizer(Protocol):
    """The tokenizer surface this script needs: chat-template rendering."""

    def apply_chat_template(self, *args: object, **kwargs: object) -> str:
        """Render chat messages into the generation prompt string."""
        ...


def load_rows(path: Path) -> list[Row]:
    """Read one jsonl file into a list of dicts, skipping blank lines."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_prompt(tokenizer: ChatTokenizer, messages: list[Message]) -> str:
    """Render chat messages into the generation-prompt string every arm serves."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
        clear_thinking=True,
    )
    return str(rendered)


def subset_rows(rows: list[Row], offset: int, size: int, seed: int) -> list[Row]:
    """Shuffle rows with a seeded generator and return one window of them."""
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    return pool[offset : offset + size]


def build_task_rows(
    tokenizer: ChatTokenizer,
    task: str,
    *,
    rows: list[Row],
    offset: int,
    size: int,
    seed: int,
) -> list[Row]:
    """Render one task's subsample into suite rows with final prompt strings."""
    out: list[Row] = []
    for index, row in enumerate(subset_rows(rows, offset, size, seed), start=offset):
        turns = row["turns"]
        messages: list[Message] = [{"role": "user", "content": str(turns[0])}]
        out.append(
            {
                "id": f"{task}-{index:04d}",
                "task": task,
                "turns": turns,
                "prompt": render_prompt(tokenizer, messages),
            }
        )
    return out


def write_jsonl(path: Path, rows: list[Row]) -> None:
    """Write rows as jsonl, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.stdout.write(f"wrote {len(rows)} rows -> {path}\n")
    sys.stdout.flush()


def build_suites(tokenizer: ChatTokenizer, deepspec_dir: Path, out_dir: Path, seed: int) -> None:
    """Render the six-task suite and the ladder workload onto out_dir."""
    for task, cap in TASK_CAPS.items():
        source = deepspec_dir / "eval_datasets" / f"{task}.jsonl"
        rows = build_task_rows(
            tokenizer, task, rows=load_rows(source), offset=0, size=cap, seed=seed
        )
        write_jsonl(out_dir / f"suite_{task}.jsonl", rows)
    alpaca = load_rows(deepspec_dir / "eval_datasets" / f"{LADDER_TASK}.jsonl")
    offset = TASK_CAPS[LADDER_TASK]
    ladder = build_task_rows(
        tokenizer,
        LADDER_TASK,
        rows=alpaca,
        offset=offset,
        size=LADDER_SIZE,
        seed=seed,
    )
    write_jsonl(out_dir / f"ladder_{LADDER_TASK}.jsonl", ladder)


def build_temp0(tokenizer: ChatTokenizer, data_dir: Path) -> None:
    """Render the authored temp-0 check set into the served prompt file."""
    rows: list[Row] = [
        {
            "id": row["id"],
            "note": row["note"],
            "max_tokens": row["max_tokens"],
            "messages": row["messages"],
            "prompt": render_prompt(tokenizer, row["messages"]),
        }
        for row in load_rows(data_dir / TEMP0_SOURCE)
    ]
    write_jsonl(data_dir / TEMP0_OUTPUT, rows)


def main() -> None:
    """Render every suite from a DeepSpec checkout at the pinned SHA."""
    parser = argparse.ArgumentParser(
        description=(
            "Render the DFlash2 smoke suites (six-task subset, ladder workload, "
            "temp-0 check set) with the GLM chat template."
        )
    )
    parser.add_argument("--deepspec-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.deepspec_dir.is_dir() or not (args.deepspec_dir / ".git").exists():
        msg = f"missing DeepSpec checkout at {args.deepspec_dir} (expect {DEEPSPEC_SHA})"
        raise RuntimeError(msg)
    tokenizer: ChatTokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    build_suites(tokenizer, args.deepspec_dir, args.data_dir, args.seed)
    build_temp0(tokenizer, args.data_dir)


if __name__ == "__main__":
    main()
