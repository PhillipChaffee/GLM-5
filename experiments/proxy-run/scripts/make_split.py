"""Write a shuffled open-perfectblend subset in DeepSpec train-jsonl format."""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROLE_MAPPING: dict[str, str] = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def valid_conversations(conversations: list[dict[str, str]]) -> bool:
    """A conversation list usable as DeepSpec training data."""
    if not conversations or conversations[0]["role"] != "user":
        return False
    for message in conversations:
        if message["role"] not in {"user", "assistant"}:
            return False
        if not isinstance(message["content"], str) or not message["content"]:
            return False
    return True


def main() -> None:
    """Shuffle the open-perfectblend dataset and write the DeepSpec jsonl."""
    parser = argparse.ArgumentParser(
        description="Write a shuffled open-perfectblend subset in DeepSpec train-jsonl format."
    )
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset("mlabonne/open-perfectblend", split="train")
    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row_number, row in enumerate(dataset, start=1):
            conversations = [
                {"role": ROLE_MAPPING[message["from"]], "content": message["value"]}
                for message in row["conversations"]
                if message["from"] in ROLE_MAPPING
            ]
            if not valid_conversations(conversations):
                skipped += 1
                continue
            handle.write(
                json.dumps(
                    {"id": row_number, "conversations": conversations},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    sys.stdout.write(f"wrote {count} rows (skipped {skipped}) -> {args.output}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
