"""GLM-5.3-Flash DSpark draft model (NoPE, DeepSpec GLM-port checkpoints).

The DeepSpec GLM drafter (experiments/glm-drafter) is a 5-layer dense
bidirectional backbone with NO positional encoding anywhere: training deletes
position_ids and runs flex-attention BlockMasks. The stock DFlashAttention
builds and applies neox RoPE unconditionally, so a thin per-arch subclass
drops RoPE the same way lfm2_dspark.py subclasses for LFM2 quirks. Config and
weight names come from the checkpoint; sglang fuses qkv/gate_up itself and
shares the target's embed/lm_head, so no remaps are needed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as torch_functional
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.models.utils import apply_qk_norm

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Glm5DsparkAttention(DFlashAttention):
    """DFlashAttention without RoPE (NoPE drafter).

    rotary_emb is set to None so the fused ctx-KV bundle builder
    (DSparkDraftMixin._build_fused_kv_write_bundle) bails to the per-layer
    path, whose apply_k_rope override below is the identity.
    """

    def __init__(
        self,
        config: object,
        layer_id: int,
        quant_config: object | None = None,
        prefix: str = "",
    ) -> None:
        """Build the parent attention, then drop its RoPE module."""
        super().__init__(config, layer_id, quant_config, prefix)
        self.rotary_emb = None

    def forward(
        self,
        positions: torch.Tensor,  # noqa: ARG002 — kept for the keyword call contract; NoPE ignores it
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Run qk-norm attention with no RoPE and no fused-table fast path.

        v is made contiguous before the attention call: the KV pool's
        set_kv_buffer views it row-wise, and the split view (which the stock
        path happens to survive on this stack) spans two strides.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        if self.v_scale is not None:
            v = v * self.v_scale
        attn_output = self.attn(q, k, v.contiguous(), forward_batch)
        attn_output = self.apply_attention_output(attn_output, hidden_states)
        output, _ = self.o_proj(attn_output)
        return output

    def apply_k_rope(self, _positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Identity: the drafter trains without any positional encoding."""
        return k


class Glm5DsparkDecoderLayer(DFlashDecoderLayer):
    """Decoder layer wired to the NoPE attention."""

    attention_cls = Glm5DsparkAttention


class Glm5DsparkModel(DSparkDraftModel):
    """The serving entry class for DeepSpec GLM-port drafter checkpoints."""

    decoder_layer_cls = Glm5DsparkDecoderLayer

    def _project_ctx_kv_stacked(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,  # noqa: ARG002 — the stacked path applies no RoPE on a NoPE drafter
        stacked: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stacked ctx-KV projection without RoPE (the NoPE stacked variant).

        Mirrors DSparkDraftMixin's default-on stacked path (the pinned
        nightly ships SGLANG_DSPARK_STACKED_CTX_KV=True) minus its RoPE
        application: the stacked weights and per-layer k-norms are
        arch-agnostic, and the only rope touch in the parent is the
        rotary_emb call this drafter must not make.
        """
        attn0 = self.layers[0].self_attn
        num_layers = len(self.layers)
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_kv_heads = attn0.num_kv_heads
        tokens = ctx_hidden.shape[0]

        kv_all = torch_functional.linear(ctx_hidden, stacked["weight"], stacked["bias"])
        kv_all = kv_all.view(tokens, num_layers, 2, kv_size)
        k32 = (
            kv_all[:, :, 0, :].reshape(tokens, num_layers, num_kv_heads, head_dim).to(torch.float32)
        )
        variance = k32.pow(2).mean(dim=-1, keepdim=True)
        k32 = k32 * torch.rsqrt(variance + stacked["eps"])
        k32 = k32 * stacked["k_norm_weight"].view(1, num_layers, 1, head_dim)
        k_all = k32.to(ctx_hidden.dtype)
        v_all = (
            kv_all[:, :, 1, :]
            .view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        k_all = (
            k_all.view(tokens, num_layers, num_kv_heads, head_dim).permute(1, 0, 2, 3).contiguous()
        )
        return k_all, v_all


EntryClass = [Glm5DsparkModel]
