"""GLM-5 DSpark trainer: target-free embedding/lm_head initialization."""

from __future__ import annotations

import json
import os

from deepspec.modeling.dspark.glm5 import Glm5DsparkModel, build_draft_config
from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
from transformers import AutoConfig, AutoTokenizer


class _WeightHolder:
    """Minimal weight carrier matching the copy interface of nn modules."""

    def __init__(self, weight) -> None:
        self.weight = weight


def _load_target_tensor(model_path: str, candidate_names: list[str]) -> dict:
    """Read tensors from a sharded safetensors checkpoint without a full load."""
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_file = os.path.join(model_path, "model.safetensors")
    wanted: dict[str, str] = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index["weight_map"]
        for name in candidate_names:
            if name in weight_map:
                wanted[name] = weight_map[name]
    elif os.path.exists(single_file):
        for name in candidate_names:
            wanted[name] = os.path.basename(single_file)
    assert wanted, (
        f"None of {candidate_names} found in checkpoint {model_path}; "
        f"index present: {os.path.exists(index_path)}"
    )
    tensors: dict[str, object] = {}
    for name, shard in wanted.items():
        with safe_open(os.path.join(model_path, shard), framework="pt") as handle:
            tensors[name] = handle.get_tensor(name)
    return tensors


def resolve_target_io_tensors(target_model_or_path):
    """Return (embed_tokens_weight, lm_head_weight) for a target checkpoint.

    The path form avoids loading the full multi-hundred-GB target into CPU
    memory just to read two tensors.
    """
    if isinstance(target_model_or_path, str):
        if not os.path.isdir(target_model_or_path):
            from huggingface_hub import snapshot_download

            target_model_or_path = snapshot_download(target_model_or_path)
        loaded = _load_target_tensor(
            target_model_or_path,
            [
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
                "model.lm_head.weight",
            ],
        )
        embed_key = next(key for key in loaded if key.endswith("embed_tokens.weight"))
        head_key = next(key for key in loaded if "lm_head" in key)
        return loaded[embed_key], loaded[head_key]
    return (
        target_model_or_path.get_input_embeddings().weight,
        target_model_or_path.get_output_embeddings().weight,
    )


class Glm5DsparkTrainer(Qwen3DSparkTrainer):
    """DSpark trainer for GLM-5.3-Flash targets.

    Subclasses Qwen3DSparkTrainer to inherit the generic DSpark train loop
    (run_batch/compute_dspark_loss); the GLM overrides are build_models
    (target embed/lm_head resolved from the safetensors index) and
    _build_draft_model.
    """

    """DSpark trainer for GLM-5.3-Flash targets."""

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Glm5DsparkModel(draft_config)

    def build_models(self):
        model_args = self.args.model
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )
        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)
        embed_weight, head_weight = resolve_target_io_tensors(model_args.target_model_name_or_path)
        embed_weight = embed_weight.to(device=self.device, dtype=self.precision_dtype)
        head_weight = head_weight.to(device=self.device, dtype=self.precision_dtype)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=_WeightHolder(embed_weight),
            lm_head=_WeightHolder(head_weight),
            freeze=True,
        )
        return draft_model, tokenizer
