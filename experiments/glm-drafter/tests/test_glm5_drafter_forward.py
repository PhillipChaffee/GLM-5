"""Tiny-config forward tests for the GLM-5.3-Flash DSpark drafter."""

import random

import pytest
import torch
from deepspec.modeling.dspark.glm5 import Glm5DsparkModel
from deepspec.modeling.dspark.glm5.config import build_draft_config
from deepspec.utils.config import ConfigNode
from transformers import PretrainedConfig

TINY_TEXT_CONFIG = {
    "model_type": "glm5_next_text",
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "num_hidden_layers": 45,
    "hc_mult": 4,
}


def _tiny_model_args() -> ConfigNode:
    return ConfigNode(
        num_draft_layers=2,
        target_layer_ids=[0, 1],
        confidence_head_alpha=0.1,
        confidence_head_with_markov=True,
        markov_rank=16,
        markov_head_type="vanilla",
        block_size=7,
        mask_token_id=3,
        num_anchors=4,
    )


def _tiny_inputs(
    batch_size: int = 2, seq_len: int = 24, num_context_layers: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    random.seed(0)
    input_ids = torch.randint(0, 512, (batch_size, seq_len))
    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    loss_mask[:, 12:] = 1
    target_hidden_states = torch.randn(batch_size, seq_len, num_context_layers * 64)
    target_last_hidden_states = torch.randn(batch_size, seq_len, 64)
    return input_ids, loss_mask, target_hidden_states, target_last_hidden_states


def test_tiny_drafter_forward_shapes_and_finiteness() -> None:
    config = build_draft_config(
        PretrainedConfig(**TINY_TEXT_CONFIG),
        _tiny_model_args(),
    )
    model = Glm5DsparkModel(config).eval()
    input_ids, loss_mask, target_hidden_states, target_last_hidden_states = _tiny_inputs()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden_states,
        )
    assert out.draft_logits.dim() == 4
    assert out.draft_logits.shape[0] == 2
    assert out.draft_logits.shape[2] == 7
    assert out.draft_logits.shape[3] == 512
    assert 0 < out.draft_logits.shape[1] <= 4
    assert torch.isfinite(out.draft_logits).all()
    assert out.confidence_pred is not None
    assert out.confidence_pred.shape[:3] == out.draft_logits.shape[:3]
    assert torch.isfinite(out.confidence_pred).all()


def test_attention_dense_mask_backward_produces_gradients() -> None:
    config = build_draft_config(
        PretrainedConfig(**TINY_TEXT_CONFIG),
        _tiny_model_args(),
    )
    layer = __import__(
        "deepspec.modeling.dspark.glm5.modeling", fromlist=["Glm5DsparkAttention"]
    ).Glm5DsparkAttention(config, layer_idx=0)
    torch.manual_seed(0)
    hidden = torch.randn(2, 8, 64, requires_grad=True)
    target_hidden = torch.randn(2, 10, 64)
    dense_mask = torch.zeros(2, 1, 8, 18)
    out = layer(hidden, target_hidden, attention_mask=dense_mask)
    out.sum().backward()
    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert grads, "expected at least one gradient after backward"
    assert all(torch.isfinite(g).all() for g in grads)
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flex_attention backward requires a GPU; CPU runs the dense path above",
)
def test_full_model_backward_produces_gradients() -> None:
    config = build_draft_config(
        PretrainedConfig(**TINY_TEXT_CONFIG),
        _tiny_model_args(),
    )
    model = Glm5DsparkModel(config)
    input_ids, loss_mask, target_hidden_states, target_last_hidden_states = _tiny_inputs()
    out = model(
        input_ids=input_ids,
        target_hidden_states=target_hidden_states,
        loss_mask=loss_mask,
        target_last_hidden_states=target_last_hidden_states,
    )
    out.draft_logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one gradient after backward"
    assert all(torch.isfinite(g).all() for g in grads)
