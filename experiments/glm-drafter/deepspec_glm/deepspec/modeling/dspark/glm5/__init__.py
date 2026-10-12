"""GLM-5 DSpark drafter model, config, and hidden-state extraction."""

from deepspec.modeling.dspark.glm5.config import build_draft_config
from deepspec.modeling.dspark.glm5.hidden_states import (
    assert_glm_hidden_shapes,
    extract_glm_context_feature,
    fold_streams,
    glm_target_hidden_states_from_hooks,
    normalize_final_stream,
)
from deepspec.modeling.dspark.glm5.modeling import Glm5DsparkConfig, Glm5DsparkModel

__all__ = [
    "Glm5DsparkConfig",
    "Glm5DsparkModel",
    "assert_glm_hidden_shapes",
    "build_draft_config",
    "extract_glm_context_feature",
    "fold_streams",
    "glm_target_hidden_states_from_hooks",
    "normalize_final_stream",
]
