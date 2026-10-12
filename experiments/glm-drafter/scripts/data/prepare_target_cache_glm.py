"""Prepare the GLM-5.3-Flash target hidden-state cache in a single process.

Mirrors the upstream DeepSpec ``scripts/data/prepare_target_cache.py`` but
loads the 320B FP8 target once with ``device_map="auto"`` across the GPU
group instead of spawning one process per GPU, and folds the mHC stream
outputs (mean over the stream axis) with the final norm applied to the last
decoder layer, so the on-disk format stays canonical while train and eval
agree on the last-layer payload.
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
from deepspec.data import ConversationCollator
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    compute_local_sample_range,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)
from deepspec.modeling.dspark.glm5.hidden_states import (
    fold_streams,
    glm_target_hidden_states_from_hooks,
)
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_diff,
    get_git_sha,
    load_config,
    parse_opts_to_config,
    seed_all,
)
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TargetForwardResult:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


def _get_target_backbone(target_model):
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    return getattr(target_model, "model", target_model)


def _get_target_hidden_size(target_model) -> int:
    text_config = getattr(target_model.config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None):
        return int(text_config.hidden_size)
    return int(target_model.config.hidden_size)


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids,
    align_device: torch.device | None = None,
):
    backbone = _get_target_backbone(target_model)
    layer_modules = backbone.layers
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    final_layer_id = len(layer_modules) - 1
    final_norm = backbone.norm
    captured_hidden_states = {}
    handles = []

    def capture_layer(layer_id: int):
        def hook(_module, _inputs, output):
            captured_hidden_states[layer_id] = _get_hook_tensor(output).detach()

        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(backbone.embed_tokens.register_forward_hook(capture_layer(-1)))
        for layer_id in target_layer_ids:
            if layer_id < 0:
                continue
            handles.append(layer_modules[layer_id].register_forward_hook(capture_layer(layer_id)))

        with torch.no_grad():
            target_output = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                logits_to_keep=1,
            )
            target_last_hidden_states = target_output.last_hidden_state.detach()
    finally:
        for handle in handles:
            handle.remove()

    if align_device is None:
        align_device = final_norm.weight.device
    aligned = {}
    for layer_id in target_layer_ids:
        tensor = fold_streams(captured_hidden_states[layer_id])
        aligned[layer_id] = tensor.to(align_device, non_blocking=True)
    captured_hidden_states.clear()
    target_hidden_states = glm_target_hidden_states_from_hooks(
        captured=aligned,
        target_layer_ids=target_layer_ids,
        final_layer_id=final_layer_id,
        final_norm=final_norm,
    )
    target_hidden_states = target_hidden_states.to(align_device, non_blocking=True)
    target_last_hidden_states = target_last_hidden_states.to(align_device, non_blocking=True)

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    cli_args = parser.parse_args()
    config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))
    return cli_args, config


def _write_manifest(
    *,
    output_dir: str,
    config,
    train_data_paths,
    target_layer_ids,
    hidden_size: int,
    min_loss_tokens: int,
    shards,
    num_samples: int,
):
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(config.model.target_model_name_or_path),
            "source_jsonl_paths": [str(path) for path in train_data_paths],
            "chat_template": str(config.data.chat_template),
            "max_length": int(config.data.max_length),
            "min_loss_tokens": int(min_loss_tokens),
            "project_name": (
                str(config.get("project_name")) if config.get("project_name") is not None else None
            ),
            "exp_name": (
                str(config.get("exp_name")) if config.get("exp_name") is not None else None
            ),
            "git_sha": str(get_git_sha()),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)


def _print_prepare_progress(processed_samples: int, total_samples: int):
    print(
        f"[prepare] {processed_samples}/{total_samples} samples",
        flush=True,
    )


def main():
    cli_args, config = parse_args()
    train_data_paths = list(cli_args.train_data_path)
    target_layer_ids = [int(layer_id) for layer_id in config.model.target_layer_ids]
    min_loss_tokens = int(cli_args.min_loss_tokens)
    seed_all(int(config.seed))
    output_dir = os.path.abspath(cli_args.output_dir)
    print(json.dumps(config, indent=4, cls=CustomJSONEncoder), flush=True)
    print(
        json.dumps(
            {
                "train_data_path": train_data_paths,
                "output_dir": output_dir,
                "target_layer_ids": target_layer_ids,
                "min_loss_tokens": min_loss_tokens,
                "max_shard_bytes": int(cli_args.max_shard_bytes),
                "local_batch_size": int(cli_args.local_batch_size),
                "num_workers": int(cli_args.num_workers),
            },
            indent=4,
        ),
        flush=True,
    )
    prepare_target_cache_output_dir(output_dir)

    rank_dir = os.path.join(output_dir, "_tmp", "rank_0")
    os.makedirs(rank_dir, exist_ok=True)

    dataset = JsonLineDataset(data_paths=train_data_paths)
    local_start, local_end = compute_local_sample_range(
        num_samples=len(dataset),
        rank=0,
        world_size=1,
    )
    local_total_samples = local_end - local_start

    local_subset = Subset(dataset, range(local_start, local_end))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path,
    )
    target_model = AutoModel.from_pretrained(
        config.model.target_model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    target_hidden_size = _get_target_hidden_size(target_model)
    train_collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=config.data.chat_template,
        max_length=config.data.max_length,
        min_loss_tokens=min_loss_tokens,
    )
    dataloader = DataLoader(
        local_subset,
        batch_size=int(cli_args.local_batch_size),
        collate_fn=train_collator,
        num_workers=int(cli_args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),
        max_queue_size=int(cli_args.local_batch_size) * 4,
    )

    processed_local_samples = 0
    last_progress_printed = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                processed_local_samples = min(
                    (batch_idx + 1) * int(cli_args.local_batch_size),
                    local_total_samples,
                )
                should_print_progress = (
                    processed_local_samples - last_progress_printed >= 100
                    or processed_local_samples == local_total_samples
                )
                if batch is None:
                    if should_print_progress:
                        _print_prepare_progress(
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                batch = {
                    key: value.to(target_model.device, non_blocking=True)
                    for key, value in batch.items()
                }
                target_result = run_target_forward_with_hooks(
                    target_model=target_model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_layer_ids=target_layer_ids,
                )
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                for sample_idx_in_batch, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    writer.write_sample(
                        input_ids=batch["input_ids"][sample_idx_in_batch, :seq_len],
                        attention_mask=batch["attention_mask"][sample_idx_in_batch, :seq_len],
                        loss_mask=batch["loss_mask"][sample_idx_in_batch, :seq_len],
                        target_hidden_states=target_result.target_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                        target_last_hidden_states=target_result.target_last_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                    )
                if should_print_progress:
                    _print_prepare_progress(
                        processed_samples=processed_local_samples,
                        total_samples=local_total_samples,
                    )
                    last_progress_printed = processed_local_samples
    finally:
        writer.close()
    del target_model
    torch.cuda.empty_cache()
    dataset.close()
    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))

    stored_summary = load_local_cache_write_summary(rank_dir)
    shard_map, shards = build_global_target_cache_shard_map([stored_summary])
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=stored_summary,
        shard_map=shard_map,
    )

    num_valid_samples = finalize_target_cache_index(
        output_dir=output_dir,
        summaries=[stored_summary],
        shard_map=shard_map,
    )
    _write_manifest(
        output_dir=output_dir,
        config=config,
        train_data_paths=train_data_paths,
        target_layer_ids=target_layer_ids,
        hidden_size=target_hidden_size,
        min_loss_tokens=min_loss_tokens,
        shards=shards,
        num_samples=int(stored_summary["num_local_samples"]),
    )
    cleanup_target_cache_tmp_dir(output_dir)
    print(
        f"Prepared target cache at {output_dir} with "
        f"{num_valid_samples}/{len(dataset)} valid samples.",
        flush=True,
    )


if __name__ == "__main__":
    if os.path.exists(".git"):
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    main()
