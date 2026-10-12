"""Config builder for the GLM-5.3-Flash DSpark drafter."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Protocol

from deepspec.modeling.dspark.common import validate_target_layer_ids

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


class DraftModelArgs(Protocol):
    """Structural view of DeepSpec's model args: attribute access + membership.

    The runtime args namespace supports both ``args.x`` and ``"x" in args``
    (upstream's AttrDict), so the builder's defensive membership checks and
    attribute reads are both honest here.
    """

    def __contains__(self, item: object) -> bool:
        """Membership mirrors the runtime args namespace."""
        ...

    num_draft_layers: int
    target_layer_ids: list[int]
    block_size: int
    mask_token_id: int
    num_anchors: int
    confidence_head_alpha: float
    confidence_head_with_markov: bool
    markov_rank: int
    markov_head_type: str


def build_draft_config(
    target_config: PretrainedConfig, model_args: DraftModelArgs
) -> PretrainedConfig:
    """Build the drafter config from the GLM-5.3-Flash text config."""
    text_config = getattr(target_config, "text_config", target_config)
    num_target_layers = int(text_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    if "target_layer_ids" not in model_args:
        msg = "target_layer_ids must be provided."
        raise ValueError(msg)
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )
    confidence_head_alpha = float(model_args.confidence_head_alpha)
    if confidence_head_alpha < 0.0:
        msg = f"confidence_head_alpha must be >= 0, got {confidence_head_alpha}"
        raise ValueError(msg)
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head and "confidence_head_with_markov" not in model_args:
        msg = "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        raise ValueError(msg)
    markov_rank = int(model_args.markov_rank)
    if markov_rank < 0:
        msg = f"markov_rank must be >= 0, got {markov_rank}"
        raise ValueError(msg)
    if markov_rank > 0 and "markov_head_type" not in model_args:
        msg = "markov_head_type must be provided when markov_rank > 0."
        raise ValueError(msg)
    draft_config = copy.deepcopy(text_config)
    draft_config.architectures = ["Glm5DsparkModel"]
    draft_config.model_type = "glm5_dspark"
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    # the deepcopy carries the target's layer-count-coupled lists; the
    # generic config validator requires them to match num_hidden_layers
    draft_config.mlp_layer_types = ["dense"] * num_draft_layers
    # transformers pins the attention implementation through an
    # underscore-named config attribute with no public setter.
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION  # noqa: SLF001
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = ["DraftModelArgs", "build_draft_config"]
