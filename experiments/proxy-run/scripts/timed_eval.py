"""Timed DSpark evaluation: accepted length plus wall-clock latency/throughput.

Target-only baseline and confidence-threshold pruning included; mirrors
deepspec eval.py's protocol on a task subset.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from deepspec.data.parser import encode_chat_messages
from deepspec.eval.base_evaluator import (
    load_and_process_dataset,
    resolve_stop_token_ids,
)
from deepspec.eval.dspark import Qwen3DSparkEvaluator
from deepspec.eval.dspark.confidence_head import summarize_confidence_row
from deepspec.utils import seed_all

DEFAULT_TASKS = "gsm8k:300,math500:300,humaneval:164,mbpp:256,mt-bench:80,alpaca:300"


def parse_args() -> argparse.Namespace:
    """Parse the eval CLI and split the task list into (name, cap) pairs."""
    parser = argparse.ArgumentParser(
        description=(
            "Timed DSpark evaluation: accepted length plus wall-clock "
            "latency/throughput, target-only baseline, and confidence-threshold "
            "pruning. Mirrors deepspec eval.py's protocol on a task subset."
        )
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    cli = parser.parse_args()
    cli.tasks = [
        (name, int(cap)) for name, cap in (item.split(":") for item in cli.tasks.split(",") if item)
    ]
    return cli


def allreduce_scalar(value: float, op: dist.ReduceOp) -> float:
    """All-reduce one scalar across ranks and return rank 0's view."""
    tensor = torch.tensor([value], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=op)
    return float(tensor[0].item())


def timed_spec_run(
    evaluator: Qwen3DSparkEvaluator, dataset_name: str, max_samples: int | None
) -> dict[str, object]:
    """One speculative-eval pass over a dataset, with timing and confidences."""
    recorder = evaluator.confidence_head_recorder
    if recorder is not None:
        recorder.start()
    seed_all(int(evaluator.args.seed))
    start = time.perf_counter()
    responses = evaluator.run_dataset(dataset_name=dataset_name, max_samples=max_samples)
    wall = time.perf_counter() - start
    tokens = float(sum(int(r.num_output_tokens) for r in responses))
    rounds = float(sum(int(r.verify_count) for r in responses))
    tokens_total = allreduce_scalar(tokens, dist.ReduceOp.SUM)
    rounds_total = allreduce_scalar(rounds, dist.ReduceOp.SUM)
    wall_max = allreduce_scalar(wall, dist.ReduceOp.MAX)
    metric_summary = evaluator.allreduce_response_metrics(responses)
    metrics_row = evaluator.build_metrics_row(
        dataset_name=dataset_name,
        metric_summary=metric_summary,
    )
    confidence_row = (
        recorder.finish(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
        )
        if recorder is not None
        else None
    )
    row = dict(metrics_row)
    world = dist.get_world_size()
    row["world_size"] = world
    row["wall_s"] = wall_max
    row["tok_s_spec_total"] = tokens_total / wall_max if wall_max > 0 else None
    row["tok_s_spec_per_stream"] = tokens_total / (wall_max * world) if wall_max > 0 else None
    row["ms_per_round_per_stream"] = (
        1000.0 * wall_max * world / rounds_total if rounds_total > 0 else None
    )
    if confidence_row is not None:
        row["confidence_summary"] = summarize_confidence_row(confidence_row)
        row["confidence_per_position"] = [
            {
                "position": int(entry["position"]),
                "ece": entry["ece"],
                "auc": entry["auc"],
                "brier": entry["brier"],
                "pred_mean": entry["pred_mean"],
                "target_mean": entry["target_mean"],
            }
            for entry in confidence_row["per_position"]
        ]
    return row


def timed_baseline_run(
    evaluator: Qwen3DSparkEvaluator, dataset_name: str, max_samples: int | None
) -> dict[str, object]:
    """One target-only generation pass over a dataset, with timing."""
    device = evaluator.device
    rank = dist.get_rank()
    world = dist.get_world_size()
    seed_all(int(evaluator.args.seed))
    dataset = load_and_process_dataset(dataset_name)
    if max_samples is not None and len(dataset) > max_samples:
        rng = random.Random(int(evaluator.args.seed))
        dataset = list(dataset)
        rng.shuffle(dataset)
        dataset = dataset[:max_samples]
    stop_token_ids = resolve_stop_token_ids(evaluator.target_model, evaluator.tokenizer)
    tokens = 0.0
    samples = 0
    start = time.perf_counter()
    for idx in range(rank, len(dataset), world):
        seed_all(int(evaluator.args.seed) + idx)
        row = dataset[idx]
        input_ids = encode_chat_messages(
            evaluator.tokenizer,
            [{"role": "user", "content": row["turns"][0]}],
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(device)
        output = evaluator.target_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=int(evaluator.args.max_new_tokens),
            do_sample=True,
            temperature=float(evaluator.args.temperature),
            top_p=1.0,
            top_k=0,
            pad_token_id=evaluator.tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
        )
        tokens += int(output.shape[1] - input_ids.shape[1])
        samples += 1
    wall = time.perf_counter() - start
    tokens_total = allreduce_scalar(tokens, dist.ReduceOp.SUM)
    samples_total = allreduce_scalar(samples, dist.ReduceOp.SUM)
    wall_max = allreduce_scalar(wall, dist.ReduceOp.MAX)
    return {
        "dataset": dataset_name,
        "num_samples": int(samples_total),
        "tokens": int(tokens_total),
        "wall_s": wall_max,
        "world_size": world,
        "tok_s_baseline_total": tokens_total / wall_max if wall_max > 0 else None,
        "tok_s_baseline": tokens_total / (wall_max * world) if wall_max > 0 else None,
    }


def main(local_rank: int, cli: argparse.Namespace) -> None:
    """Run the timed eval across all configured tasks and write rank 0's json."""
    ns = argparse.Namespace(
        target_name_or_path=cli.target,
        draft_name_or_path=cli.draft,
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        confidence_threshold=cli.confidence_threshold,
        tensorboard_dir=None,
        step=None,
        seed=cli.seed,
        tasks=cli.tasks,
    )
    evaluator = Qwen3DSparkEvaluator(local_rank, ns)
    rows = []
    for dataset_name, max_samples in cli.tasks:
        if cli.baseline:
            row = timed_baseline_run(evaluator, dataset_name, max_samples)
            sys.stdout.write(f"[baseline] {row}\n")
        else:
            row = timed_spec_run(evaluator, dataset_name, max_samples)
            sys.stdout.write(f"[spec] {json.dumps(row, default=str)}\n")
        sys.stdout.flush()
        rows.append(row)
    if dist.get_rank() == 0 and cli.out:
        payload = {
            "mode": "baseline" if cli.baseline else "spec",
            "target": cli.target,
            "draft": cli.draft,
            "max_new_tokens": cli.max_new_tokens,
            "temperature": cli.temperature,
            "confidence_threshold": cli.confidence_threshold,
            "seed": cli.seed,
            "tasks": [[name, cap] for name, cap in cli.tasks],
            "rows": rows,
        }
        out_path = Path(cli.out)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        sys.stdout.write(f"wrote {cli.out}\n")
        sys.stdout.flush()
    evaluator.clean_up()


if __name__ == "__main__":
    cli = parse_args()
    mp.spawn(main, args=(cli,), nprocs=torch.cuda.device_count())
