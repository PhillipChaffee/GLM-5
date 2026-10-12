"""Target hidden-state extraction for GLM-5.3-Flash.

GLM-5.3 decoder layers emit ``hc_mult`` interleaved streams of shape
``[B, S, N, D]``. The target collapses them with an unweighted mean
(``Glm5NextTextHyperHead``) before the final norm, so both the cache writer
and the evaluator fold captured streams the same way: mean over the stream
axis per intermediate layer, and the model's own ``last_hidden_state``
(fold + final norm) for the last layer.
"""

from __future__ import annotations

import torch


def fold_streams(hidden_states: torch.Tensor) -> torch.Tensor:
    """Collapse an mHC stream tensor ``[B, S, N, D]`` to ``[B, S, D]``."""
    if hidden_states.dim() == 3:
        return hidden_states
    assert hidden_states.dim() == 4, (
        f"Expected [B, S, N, D] hidden-state streams, got {tuple(hidden_states.shape)}"
    )
    return hidden_states.mean(dim=2)


def extract_glm_context_feature(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    layer_ids: list[int],
    final_hidden_state: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the folded per-layer hidden states for ``layer_ids``.

    ``hidden_states`` follows the transformers ``output_hidden_states``
    convention: index ``i`` is the output of decoder layer ``i - 1``, and the
    final entry is the post-final-norm hidden state. Layer ids equal to
    ``len(hidden_states) - 2`` therefore resolve to the final hidden state,
    matching what the cache stores for that layer.
    """
    final_layer_id = len(hidden_states) - 2
    parts = []
    for layer_id in layer_ids:
        if layer_id == final_layer_id:
            parts.append(final_hidden_state)
            continue
        part = hidden_states[layer_id + 1]
        if part.dim() == 4:
            part = part.mean(dim=2)
        parts.append(part)
    return torch.cat(parts, dim=-1)


def normalize_final_stream(target_model, folded_streams: torch.Tensor) -> torch.Tensor:
    """Apply the target's final RMSNorm to a folded last-layer stream."""
    language_model = target_model.language_model
    return language_model.norm(folded_streams)


def glm_target_hidden_states_from_hooks(
    captured: dict[int, torch.Tensor],
    target_layer_ids: list[int],
    final_layer_id: int,
    final_norm,
) -> torch.Tensor:
    """Build the cache payload for one batch from per-layer hook captures.

    The last decoder layer's raw stream output is folded and re-normalized so
    it matches the evaluator-side ``output_hidden_states`` entry for the same
    layer (which is the post-norm ``last_hidden_state``).
    """
    parts = []
    for layer_id in target_layer_ids:
        tensor = captured[layer_id]
        if tensor.dim() == 4:
            tensor = tensor.mean(dim=2)
        if layer_id == final_layer_id:
            tensor = final_norm(tensor)
        parts.append(tensor)
    return torch.cat(parts, dim=-1)


def assert_glm_hidden_shapes(
    hidden_states: tuple[torch.Tensor, ...],
    num_layers: int,
) -> None:
    """Fail fast when the captured hidden-state tuple has an unexpected shape."""
    assert len(hidden_states) == num_layers + 1, (
        f"Expected {num_layers + 1} hidden-state entries, got {len(hidden_states)}"
    )
    stream_entry = hidden_states[1]
    if stream_entry.dim() != 4:
        raise AssertionError(
            "GLM-5.3-Flash hidden states must carry the mHC stream dimension; "
            f"got ndim={stream_entry.dim()} shape={tuple(stream_entry.shape)}"
        )


__all__ = [
    "assert_glm_hidden_shapes",
    "extract_glm_context_feature",
    "fold_streams",
    "glm_target_hidden_states_from_hooks",
    "normalize_final_stream",
]
