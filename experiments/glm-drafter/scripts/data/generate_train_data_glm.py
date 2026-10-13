"""Regenerate JSONL conversations on-policy through GLM-5.3-Flash servers.

Mirrors the upstream DeepSpec ``scripts/data/generate_train_data.py`` per-turn
regeneration loop, server validation, and resume support, with GLM template
kwargs (``reasoning_effort`` and ``clear_thinking`` via ``chat_template_kwargs``),
DeepSpec sampling defaults, and assistant turns stored inline as
``think-open + reasoning + think-close + answer``.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/think" + ">"


def parse_args():
    """Parse command-line arguments for the GLM regeneration run."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate JSONL conversations through OpenAI-compatible GLM-5.3-Flash sglang servers."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-address", nargs="+", required=True)
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high"],
        default="low",
        help=(
            "GLM reasoning effort; values outside {low, high} silently fall "
            "back to the template default."
        ),
    )
    parser.add_argument(
        "--clear-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strip prior assistant reasoning from the rendered context "
            "server-side; the stored corpus keeps reasoning inline."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    """Reject out-of-range sampling parameters before any server call."""
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    if args.top_p is not None and not 0.0 <= args.top_p <= 1.0:
        raise ValueError("top-p must be between 0.0 and 1.0")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("top-k must be greater than 0")
    if args.min_p is not None and not 0.0 <= args.min_p <= 1.0:
        raise ValueError("min-p must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")


def build_query_kwargs(args, messages, max_tokens=None):
    """Build the chat-completions request for one on-policy turn."""
    query_kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    extra_body = {}
    if args.repetition_penalty is not None:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    extra_body.setdefault("chat_template_kwargs", {})
    extra_body["chat_template_kwargs"]["reasoning_effort"] = args.reasoning_effort
    extra_body["chat_template_kwargs"]["clear_thinking"] = args.clear_thinking
    query_kwargs["extra_body"] = extra_body
    return query_kwargs


def build_inline_assistant(reasoning_content, content):
    """Store one assistant turn inline as think-open + reasoning + close + answer.

    Follows the chat template's own precedence: a separate ``reasoning_content``
    field wins; otherwise the reasoning is recovered from inline think tags the
    server may have left in ``content``.
    """
    if content is None:
        content = ""
    if reasoning_content is None and THINK_CLOSE in content:
        head, _, tail = content.partition(THINK_CLOSE)
        reasoning_content = head.split(THINK_OPEN)[-1]
        content = tail
    if reasoning_content is None:
        reasoning_content = ""
    return THINK_OPEN + reasoning_content + THINK_CLOSE + content


def call_sglang(args, server_address, sample, max_tokens=None):
    """Regenerate one sample turn by turn, conditioning on prior regenerated turns."""
    conversations = sample.get("conversations")
    if not conversations:
        return error_sample(sample, "Missing conversations")
    if conversations[0].get("role") == "assistant":
        return error_sample(sample, "Data starts with an assistant message")

    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
    regenerated = []

    for message in conversations:
        role = message.get("role")
        if role == "system":
            regenerated.append(message)
            continue
        if role == "assistant":
            continue
        if role != "user":
            return error_sample(sample, f"Invalid message role: {role}")

        regenerated.append(message)
        try:
            response = client.chat.completions.create(
                **build_query_kwargs(args, regenerated, max_tokens=max_tokens)
            )
        except Exception as exc:
            return error_sample(sample, str(exc))

        usage = getattr(response, "usage", None)
        turn_tokens = {
            "turn_completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "turn_prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "turn_reasoning_tokens": getattr(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0
            )
            or 0,
        }
        sample.setdefault("_usage", []).append(turn_tokens)

        response_message = response.choices[0].message
        regenerated.append(
            {
                "role": "assistant",
                "content": build_inline_assistant(
                    getattr(response_message, "reasoning_content", None),
                    response_message.content,
                ),
            }
        )

    sample["conversations"] = regenerated
    sample["status"] = "success"
    return sample


def error_sample(sample, message):
    """Mark a sample as failed with the given error message."""
    sample["status"] = "error"
    sample["error"] = message
    return sample


def compute_context_length(conversations):
    """Estimate the word length of a regenerated conversation."""
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    length += len(part["text"].split())
    return length


def count_lines(path, *, required=False):
    """Count the lines in a JSONL file; missing paths raise when required."""
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Input file not found: {path}")
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def truncate_partial_line(path):
    """Drop a trailing write that a killed container may have left mid-line.

    A complete row always ends with a newline; a committed file that does
    not was cut mid-write, and appending after it would glue rows together.
    """
    if not os.path.exists(path):
        return
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
    with open(path, "rb") as handle:
        data = handle.read()
    with open(path, "wb") as handle:
        handle.write(data[: data.rfind(b"\n") + 1])
    print(f"Truncated a partial trailing line in {path}", flush=True)


def collect_done_indices(output_path, error_path):
    """Input row indices already processed, read from committed rows.

    Results are written in completion order, not input order, and a kill
    commits an arbitrary subset of them — so a line count is not a resume
    cursor (the #44 regen drill measured 37 duplicate rows from that
    assumption). Each row carries ``input_index``; the set of committed
    indices is the exact resume state.
    """
    done = set()
    for path in (output_path, error_path):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    print(f"Skipping unparsable line in {path} during resume scan", flush=True)
                    continue
                index = row.get("input_index")
                if isinstance(index, int) and index >= 0:
                    done.add(index)
    return done


def resolve_resume_state(args, error_path):
    """Compute the already-done input indices and the output open mode."""
    if not args.resume:
        return set(), "w"
    truncate_partial_line(args.output_file_path)
    truncate_partial_line(error_path)
    done_indices = collect_done_indices(args.output_file_path, error_path)
    if done_indices:
        print(
            f"Resume mode: {len(done_indices)} input rows already processed "
            f"({count_lines(args.output_file_path)} output lines, "
            f"{count_lines(error_path)} error lines)",
            flush=True,
        )
    return done_indices, "a"


def validate_server(args, server_address, probe):
    """Probe one server with a 1-token request and time the round trip."""
    start_time = time.perf_counter()
    try:
        result = call_sglang(args, server_address, dict(probe), max_tokens=1)
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    elapsed = time.perf_counter() - start_time
    return server_address, result, elapsed


def validate_servers(args):
    """Return the subset of servers that answer the probe successfully."""
    probe = {"conversations": [{"role": "user", "content": "Hello"}]}
    server_results = {}
    server_count = len(args.server_address)

    print(f"Validating {server_count} GLM servers in parallel...", flush=True)
    with ThreadPoolExecutor(max_workers=server_count) as executor:
        future_to_server = {
            executor.submit(validate_server, args, server_address, probe): server_address
            for server_address in args.server_address
        }

        for completed_count, future in enumerate(as_completed(future_to_server), 1):
            server_address = future_to_server[future]
            try:
                _, result, elapsed = future.result()
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
                elapsed = 0.0

            if result.get("status") == "success":
                server_results[server_address] = True
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"ok server {server_address} elapsed={elapsed:.2f}s",
                    flush=True,
                )
            else:
                server_results[server_address] = False
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"skip server {server_address} elapsed={elapsed:.2f}s: "
                    f"{result.get('error')}",
                    flush=True,
                )

    valid_servers = [
        server_address for server_address in args.server_address if server_results[server_address]
    ]
    if not valid_servers:
        raise RuntimeError("No available GLM server")
    print(f"Using servers: {valid_servers}")
    return valid_servers


def write_finished_result(future, output_handle, error_handle, stats):
    """Consume one finished future into the success or error JSONL stream."""
    sample = future.result()
    for turn in sample.pop("_usage", []):
        stats["completion_tokens"] += turn["turn_completion_tokens"]
        stats["prompt_tokens"] += turn["turn_prompt_tokens"]
        stats["reasoning_tokens"] += turn["turn_reasoning_tokens"]
    if sample["status"] == "error":
        error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        stats["errors"] += 1
        return

    context_length = compute_context_length(sample.get("conversations", []))
    stats["context_sum"] += context_length
    stats["context_min"] = (
        context_length
        if stats["context_min"] is None
        else min(stats["context_min"], context_length)
    )
    stats["context_max"] = max(stats["context_max"], context_length)
    stats["success"] += 1
    output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_config(args):
    """Log the run configuration."""
    print("Configuration:")
    print(f"  model: {args.model}")
    print(f"  servers: {args.server_address}")
    print(f"  input: {args.input_file_path}")
    print(f"  output: {args.output_file_path}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  min_p: {args.min_p}")
    print(f"  reasoning_effort: {args.reasoning_effort}")
    print(f"  clear_thinking: {args.clear_thinking}")
    print(f"  resume: {args.resume}")


def wait_for_queue_slot(
    args,
    queues,
    server_address,
    *,
    output_handle,
    error_handle,
    stats,
):
    """Block until the server's queue has a free slot, consuming finished futures."""
    while len(queues[server_address]) >= args.concurrency:
        wrote_result = False
        for future in list(queues[server_address]):
            if future.done():
                write_finished_result(future, output_handle, error_handle, stats)
                queues[server_address].remove(future)
                wrote_result = True
                break
        if not wrote_result:
            time.sleep(0.05)


def report_stats(stats, elapsed):
    """Print the final success/error counts, token accounting, and throughput."""
    print("Processing completed.")
    print(f"  success: {stats['success']}")
    print(f"  errors: {stats['errors']}")
    completion_tokens = stats["completion_tokens"]
    prompt_tokens = stats["prompt_tokens"]
    reasoning_tokens = stats["reasoning_tokens"]
    print(f"  completion_tokens: {completion_tokens}")
    print(f"  prompt_tokens: {prompt_tokens}")
    print(f"  reasoning_tokens: {reasoning_tokens}")
    if completion_tokens and stats["success"] > 0:
        print(f"  completion_tokens_per_row: {completion_tokens / stats['success']:.0f}")
    if elapsed > 0 and completion_tokens > 0:
        print(f"  elapsed_s: {elapsed:.1f}")
        print(f"  completion_tok_s: {completion_tokens / elapsed:.1f}")
        print(f"  rows_per_min: {stats['success'] / (elapsed / 60):.1f}")
    if stats["success"] == 0:
        return
    avg_context = stats["context_sum"] / stats["success"]
    print(f"  context_min: {stats['context_min']}")
    print(f"  context_max: {stats['context_max']}")
    print(f"  context_avg: {avg_context:.2f}")


def main():
    """Run the on-policy regeneration over the input JSONL."""
    args = parse_args()
    validate_args(args)
    print_config(args)

    total_lines = count_lines(args.input_file_path, required=True)
    error_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    done_indices, file_mode = resolve_resume_state(args, error_path)
    done_count = len(done_indices)

    valid_servers = validate_servers(args)
    stats = {
        "success": 0,
        "errors": 0,
        "context_sum": 0,
        "context_min": None,
        "context_max": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "reasoning_tokens": 0,
    }
    start_time = time.perf_counter()
    queues = {server_address: [] for server_address in valid_servers}
    next_server_index = 0
    submitted_count = 0

    with (
        open(args.input_file_path, encoding="utf-8") as input_handle,
        open(args.output_file_path, file_mode, encoding="utf-8") as output_handle,
        open(error_path, file_mode, encoding="utf-8") as error_handle,
        ThreadPoolExecutor(max_workers=args.concurrency * len(valid_servers)) as executor,
    ):
        progress_total = (
            total_lines
            if args.num_samples is None
            else min(total_lines, done_count + args.num_samples)
        )
        progress = tqdm(total=progress_total, initial=done_count, desc="Processing")
        for row_index, line in enumerate(input_handle):
            if args.num_samples is not None and submitted_count >= args.num_samples:
                break

            if row_index in done_indices:
                continue

            sample = json.loads(line)
            sample["input_index"] = row_index
            server_address = valid_servers[next_server_index]
            next_server_index = (next_server_index + 1) % len(valid_servers)

            wait_for_queue_slot(
                args,
                queues,
                server_address,
                output_handle=output_handle,
                error_handle=error_handle,
                stats=stats,
            )
            future = executor.submit(call_sglang, args, server_address, sample)
            queues[server_address].append(future)
            submitted_count += 1
            progress.update(1)

        for server_address in valid_servers:
            for future in queues[server_address]:
                write_finished_result(future, output_handle, error_handle, stats)
        progress.close()

    report_stats(stats, time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
