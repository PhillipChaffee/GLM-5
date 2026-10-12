"""Hook-vs-output_hidden_states equivalence for the GLM mHC fold+norm path.

The cache writer folds captured raw layer streams and applies the final norm
to the last layer; the evaluator reads the ``output_hidden_states`` tuple.
Both paths must produce identical context features.
"""

from collections.abc import Callable

import torch
from deepspec.modeling.dspark.glm5.hidden_states import (
    extract_glm_context_feature,
    glm_target_hidden_states_from_hooks,
)
from torch import nn


class MockMhcLayer(nn.Module):
    def __init__(self, dim: int, num_streams: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.stream_offset = nn.Parameter(torch.randn(num_streams, dim) * 0.1)
        self.num_streams = num_streams

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.mean(dim=2)
        base = self.proj(hidden_states)
        streams = base.unsqueeze(2) + self.stream_offset
        return streams, torch.arange(self.num_streams)


class MockMhcTarget(nn.Module):
    def __init__(self, dim: int = 32, num_streams: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embed_tokens = nn.Embedding(64, dim)
        self.layers = nn.ModuleList(
            MockMhcLayer(dim, num_streams, seed=10 + i) for i in range(num_layers)
        )
        self.norm = nn.RMSNorm(dim)

    def fold(self, streams: torch.Tensor) -> torch.Tensor:
        if streams.dim() == 4:
            return streams.mean(dim=2)
        return streams

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_states = []
        current = self.embed_tokens(input_ids)
        hidden_states.append(current)
        for layer in self.layers:
            streams, _ = layer(current)
            current = streams
            hidden_states.append(streams)
        last_hidden_state = self.norm(self.fold(current))
        hidden_states[-1] = last_hidden_state
        return {
            "last_hidden_state": last_hidden_state,
            "hidden_states": tuple(hidden_states),
        }


def test_hook_path_matches_output_hidden_states_path() -> None:
    target = MockMhcTarget().eval()
    input_ids = torch.randint(0, 64, (2, 12))
    layer_ids = [0, 1]
    final_layer_id = len(target.layers) - 1

    captured = {}

    def capture(layer_id: int) -> Callable[..., None]:
        def hook(
            _module: object,
            _inputs: object,
            output: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            captured[layer_id] = output[0].detach()

        return hook

    with torch.no_grad():
        handles = [layer.register_forward_hook(capture(i)) for i, layer in enumerate(target.layers)]
        try:
            output = target(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        hook_features = glm_target_hidden_states_from_hooks(
            captured=captured,
            target_layer_ids=layer_ids,
            final_layer_id=final_layer_id,
            final_norm=target.norm,
        )
        hidden_states_features = extract_glm_context_feature(
            output["hidden_states"],
            layer_ids,
            output["last_hidden_state"],
        )

    assert torch.allclose(hook_features, hidden_states_features, atol=1e-6)


def test_last_hidden_state_equals_normed_fold_of_final_streams() -> None:
    target = MockMhcTarget().eval()
    input_ids = torch.randint(0, 64, (1, 8))

    captured = {}

    def hook(
        _module: object,
        _inputs: object,
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        captured["final"] = output[0].detach()

    with torch.no_grad():
        handle = target.layers[-1].register_forward_hook(hook)
        try:
            output = target(input_ids)
        finally:
            handle.remove()
        recomputed = target.norm(captured["final"].mean(dim=2))

    assert torch.allclose(output["last_hidden_state"], recomputed, atol=1e-6)
