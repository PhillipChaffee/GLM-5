"""GLM-5.3-Flash DSpark evaluator.

Differences from the Qwen3 evaluator:
- the target loads via ``Glm5NextForConditionalGeneration`` (VL wrapper) with
  sharded device placement, single process;
- prompt rendering uses ``reasoning_effort``/``clear_thinking`` chat-template
  kwargs (the #12-bar serving config);
- context extraction folds mHC streams and resolves the final layer to the
  model's own ``last_hidden_state``.
"""

from __future__ import annotations

import functools
import random
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Protocol

import torch
import torch.distributed as dist
from deepspec.eval.base_evaluator import (
    DEFAULT_DATASET_ROOT,
    VerificationResult,
    load_and_process_dataset,
    resolve_stop_token_ids,
    seed_all,
)
from deepspec.eval.dspark.evaluator import (
    Qwen3DSparkEvaluator,
)
from deepspec.modeling.dspark.glm5 import (
    Glm5DsparkConfig,
    Glm5DsparkModel,
    extract_glm_context_feature,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DynamicCache,
    Glm5NextForConditionalGeneration,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

GLM_TARGET = "zai-org/GLM-5.3-Flash"


class _EvaluatorArgs(Protocol):
    """Structural view of the evaluator args the GLM evaluator reads."""

    tasks: str
    seed: int
    target_name_or_path: str
    draft_name_or_path: str


@functools.cache
def _register_dspark_config() -> None:
    """Register the GLM drafter config class with transformers (run once)."""
    AutoConfig.register("glm5_dspark", Glm5DsparkConfig)


class _NoOpDraftCache:
    """Cache stub for the GLM drafter.

    The drafter re-attends to full context each proposal and never writes
    incremental key/value states; draft_ops only requires get_seq_length()
    (always 0) and a no-op crop().
    """

    def get_seq_length(self, _layer_idx: int | None = None) -> int:
        """Report zero cached draft length; the drafter re-attends to full context."""
        return 0

    def crop(self, max_length: int) -> None:
        del max_length


class Glm5DsparkEvaluator(Qwen3DSparkEvaluator):
    """Evaluator over a GLM-5.3-Flash target with a DSpark drafter."""

    draft_model_cls = Glm5DsparkModel
    chat_template_kwargs: ClassVar[dict[str, object]] = {
        "reasoning_effort": "low",
        "clear_thinking": True,
    }

    def __init__(self, local_rank: int, args: _EvaluatorArgs) -> None:
        """Initialize the single-process, world-1 GLM evaluator."""
        del local_rank
        self.args = args
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29700",
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=60),
            device_id=device,
        )
        self.device = device
        self.global_rank = 0
        self.world_size = 1
        self.tasks = args.tasks
        self.target_model, self.draft_model, self.tokenizer = self.build_models()
        self.metrics_rows: list[dict[str, object]] = []
        self.confidence_head_recorder = self._build_confidence_head_recorder()

    def build_models(
        self,
    ) -> tuple[Glm5NextForConditionalGeneration, Glm5DsparkModel, PreTrainedTokenizerBase]:
        """Load the GLM target (sharded, single process) and the drafter."""
        _register_dspark_config()
        target_model = Glm5NextForConditionalGeneration.from_pretrained(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
            device_map="auto",
        ).eval()
        draft_model = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device)
        draft_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        return target_model, draft_model, tokenizer

    def new_target_cache(self) -> DynamicCache:
        """Build the target cache for GLM's hybrid linear-attention layers.

        A bare DynamicCache would hold only KV layers; the GLM text stack
        needs per-layer entries for its linear-attention (recurrent) layers.
        """
        return DynamicCache(config=self.target_model.config.text_config)

    def _extract(self, output: MoeCausalLMOutputWithPast) -> torch.Tensor:
        return extract_glm_context_feature(
            output.hidden_states,
            list(self.draft_model.target_layer_ids),
            output.hidden_states[-1],
        )

    def _init_context(
        self,
        *,
        initial_output: MoeCausalLMOutputWithPast,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            past_key_values_draft=_NoOpDraftCache(),
            target_hidden_states=self._extract(initial_output),
        )

    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        context.target_hidden_states = self._extract(verification.target_output)[
            :,
            : verification.accepted_draft_tokens + 1,
            :,
        ]

    def run_dataset(
        self,
        *,
        dataset_name: str,
        max_samples: int | None,
    ) -> list[SimpleNamespace]:
        """Render the dataset prompts and generate one draft-verified response each."""
        seed_all(int(self.args.seed))
        dataset = load_and_process_dataset(
            dataset_name,
            dataset_root=DEFAULT_DATASET_ROOT,
        )
        if max_samples is not None and len(dataset) > max_samples:
            rng = random.Random(int(self.args.seed))
            dataset = list(dataset)
            rng.shuffle(dataset)
            dataset = dataset[:max_samples]
        stop_token_ids = resolve_stop_token_ids(self.target_model, self.tokenizer)
        responses = []
        for idx in range(len(dataset)):
            seed_all(int(self.args.seed) + idx)
            instance = dataset[idx]
            messages = [{"role": "user", "content": instance["turns"][0]}]
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            input_ids = torch.tensor([token_ids], device=self.device)
            responses.append(
                self.generate_one_sample(
                    input_ids=input_ids,
                    stop_token_ids=stop_token_ids,
                )
            )
        return responses


__all__ = ["GLM_TARGET", "Glm5DsparkEvaluator"]
