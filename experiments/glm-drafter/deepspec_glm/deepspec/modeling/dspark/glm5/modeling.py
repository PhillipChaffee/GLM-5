"""GLM-5.3-Flash DSpark drafter: dense 5-layer backbone, NoPE attention.

Mirrors ``deepspec.modeling.dspark.qwen3.modeling`` with GLM-5 family
settings: no rotary embeddings (the target's text stack is NoPE), GLM
vocabulary width, and the mean-fold mHC convention for captured target
hidden states (see deepspec_glm.modeling.dspark.glm5.hidden_states).
"""

from __future__ import annotations

from typing import cast

import torch
from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.utils.sampling import sample_tokens
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextRMSNorm

_BASE_LOGITS_NDIM = 2  # sample_draft_token_step contract: [batch, vocab]


class Glm5DsparkConfig(PretrainedConfig):
    """Config for the GLM-5.3-Flash DSpark drafter."""

    model_type = "glm5_dspark"

    def __init__(
        self,
        *,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        block_size: int = 7,
        target_layer_ids: list[int] | None = None,
        mask_token_id: int = 154821,
        num_anchors: int = 512,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        **kwargs: object,
    ) -> None:
        """Initialize the GLM drafter config; HF kwargs pass to PretrainedConfig."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.block_size = block_size
        self.target_layer_ids = target_layer_ids or []
        self.mask_token_id = mask_token_id
        self.num_anchors = num_anchors
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov
        super().__init__(**kwargs)


class Glm5DsparkAttention(nn.Module):
    """Bidirectional-over-context attention without rotary position bias.

    GLM-5.3-Flash's text stack runs NoPE, so the drafter consumes raw
    positions; the context key/values come from the captured target layers.
    """

    def __init__(self, config: Glm5DsparkConfig, layer_idx: int) -> None:
        """Build one NoPE drafter attention layer."""
        super().__init__()
        del layer_idx
        self.config = config
        self.head_dim = int(
            getattr(config, "head_dim", 0) or config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Glm5NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Glm5NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        if isinstance(attention_mask, BlockMask):
            if self.attention_dropout != 0.0:
                msg = "flex_attention path does not apply attention_dropout"
                raise ValueError(msg)
            attn_output = flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                scale=self.scaling,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
            )
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class Glm5DsparkDecoderLayer(nn.Module):
    def __init__(self, config: Glm5DsparkConfig, layer_idx: int) -> None:
        """Build one drafter decoder layer (attention + MLP, pre/post norms)."""
        super().__init__()
        del layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Glm5DsparkAttention(config=config, layer_idx=0)
        self.mlp = Glm5DsparkMLP(config)
        self.input_layernorm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Glm5DsparkMLP(nn.Module):
    def __init__(self, config: Glm5DsparkConfig) -> None:
        """Build the SwiGLU MLP block."""
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Glm5DsparkModel(PreTrainedModel):
    """Dense DSpark drafter for GLM-5.3-Flash targets."""

    config_class = Glm5DsparkConfig
    base_model_prefix = "glm5_dspark"
    _supports_sdpa = True
    _supports_flex_attn = True

    def __init__(self, config: Glm5DsparkConfig) -> None:
        """Validate drafter config fields and build the GLM drafter modules."""
        super().__init__(config)
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            if not hasattr(config, field):
                msg = f"config.{field} must be provided."
                raise ValueError(msg)
        if int(config.markov_rank) > 0 and not hasattr(config, "markov_head_type"):
            msg = "config.markov_head_type must be provided when markov_rank > 0."
            raise ValueError(msg)
        if bool(config.enable_confidence_head) and not hasattr(
            config, "confidence_head_with_markov"
        ):
            msg = (
                "config.confidence_head_with_markov must be provided when "
                "enable_confidence_head is true."
            )
            raise ValueError(msg)
        self.target_layer_ids = list(config.target_layer_ids)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Glm5DsparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        self.num_anchors = int(config.num_anchors)
        self.markov_head = build_markov_head(config)
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", False)
        )
        confidence_requires_markov = (
            self.enable_confidence_head
            and self.confidence_head_with_markov
            and self.markov_head is None
        )
        if confidence_requires_markov:
            msg = "confidence_head_with_markov requires a markov head."
            raise ValueError(msg)
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += int(config.markov_rank)
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ) -> None:
        """Copy the target's embedding/LM-head weights in and optionally freeze."""
        if self.embed_tokens.weight.shape != embed_tokens.weight.shape:
            msg = (
                "embed_tokens shape mismatch: "
                f"{tuple(self.embed_tokens.weight.shape)} vs "
                f"{tuple(embed_tokens.weight.shape)}"
            )
            raise ValueError(msg)
        if self.lm_head.weight.shape != lm_head.weight.shape:
            msg = (
                "lm_head shape mismatch: "
                f"{tuple(self.lm_head.weight.shape)} vs "
                f"{tuple(lm_head.weight.shape)}"
            )
            raise ValueError(msg)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool) -> None:  # noqa: FBT001
        """Freeze or unfreeze the copied embedding and LM-head weights.

        The boolean stays positional: upstream ``ckpt_manager`` resumes with a
        positional call.
        """
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project drafter hidden states to vocabulary logits."""
        return self.lm_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Predict per-position acceptance confidence for the drafted block."""
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            if self.markov_head is None or prev_token_ids is None:
                msg = "markov-conditioned confidence needs head and prev ids."
                raise ValueError(msg)
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a full draft block from base logits (markov-conditioned)."""
        batch_size = base_logits.shape[0]
        proposal_len = base_logits.shape[1]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return cast(
            "tuple[torch.Tensor, torch.Tensor]",
            self.markov_head.sample_block_tokens(
                base_logits,
                first_prev_token_ids=first_prev_token_ids,
                hidden_states=hidden_states,
                temperature=temperature,
            ),
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one draft step from ``[batch, vocab]`` base logits."""
        if base_logits.ndim != _BASE_LOGITS_NDIM:
            msg = (
                "sample_draft_token_step expects base_logits shaped [batch, vocab], "
                f"got {tuple(base_logits.shape)}."
            )
            raise ValueError(msg)
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits

    def _forward_backbone(
        self,
        *,
        noise_embedding: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        is_causal: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run the drafter backbone; HF carry-along args are ignored by design."""
        del position_ids, past_key_values, use_cache, is_causal, kwargs
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None = None,
    ) -> DSparkForwardOutput:
        """Run the DSpark training forward over sampled anchor blocks."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
        )
        output_hidden = self._forward_backbone(
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )
        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )
        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )
        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                    dtype=output_hidden_4d.dtype
                )
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()
        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = ["Glm5DsparkConfig", "Glm5DsparkModel"]
