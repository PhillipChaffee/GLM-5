"""DSpark drafter training config for the GLM-5.3-Flash target.

Block size 7, dense 5-layer drafter, vanilla Markov head rank 256 with the
confidence head kept, and the 7-layer context union {1, 12, 23, 34, 42, 43, 44}
of the 45-layer GLM-5.3-Flash text stack.
"""

from pathlib import Path

from deepspec.trainer import Glm5DsparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, GLM_5_3_FLASH

project_name = "deepspec"
exp_name = "glm_drafter_block7_glm53flash"
seed = 42

model = {
    "target_model_name_or_path": GLM_5_3_FLASH,
    "block_size": 7,
    "num_draft_layers": 5,
    "target_layer_ids": [1, 12, 23, 34, 42, 43, 44],
    "mask_token_id": 154821,
    "num_anchors": 512,
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "confidence_head_alpha": 1.0,
    "confidence_head_with_markov": True,
    "loss_decay_gamma": 4.0,
    "ce_loss_alpha": 0.1,
    "l1_loss_alpha": 0.9,
}

train = {
    "trainer_cls": Glm5DsparkTrainer,
    "lr": 6.0e-4,
    "warmup_ratio": 0.04,
    "weight_decay": 0.0,
    "precision": "bf16",
    "local_batch_size": 4,
    "global_batch_size": 512,
    "num_train_epochs": 10,
    "max_train_steps": None,
    "max_grad_norm": 1.0,
    "sharding_strategy": "no_shard",
    "torch_compile": True,
}

logging = {
    "logging_steps": 10,
    "checkpointing_steps": 3000,
}

data = {
    "target_cache_path": None,
    "chat_template": "glm",
    "max_length": 4096,
    "num_workers": 4,
}


def finalize_cfg(cfg: dict[str, object]) -> dict[str, object]:
    """Point the logging config at the run's checkpoint/tensorboard directories."""
    raw_logging = cfg["logging"]
    if not isinstance(raw_logging, dict):
        msg = "config logging section must be a dict."
        raise TypeError(msg)
    logging_cfg = dict(raw_logging)
    project_name_str = str(cfg["project_name"])
    exp_name_str = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = str(Path(BASE_CKPT_DIR) / project_name_str / exp_name_str)
    logging_cfg["tensorboard_dir"] = str(Path(BASE_TB_DIR) / project_name_str / exp_name_str)
    cfg["logging"] = logging_cfg
    return cfg
