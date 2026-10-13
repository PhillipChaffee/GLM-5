"""Capture all GLM-5.3-Flash layer streams plus predictive entropies (issue 51).

Stage-2 of the layer-selection analysis: run the target over a slice of the
regenerated smoke corpus, capture the raw (mHC-folded) output of every decoder
layer, the post-final-norm stream, and the per-token predictive entropies from
the full logits, then accumulate 32-token chunk sums of the per-sample
top-rho (paper: rho=0.3, "Surfing the Activation Wave") highest-entropy
positions into per-column pooling files. Streams go CPU-side per layer hook so
GPU memory stays at model+batch scale, never 45-layer scale.

The deepspec package and the HF token must be present (the deepspec Modal
image mounts the overlay at PYTHONPATH and carries the hf-token secret).
Deltas against the cache-builder mirror (scripts/data/prepare_target_cache_glm.py)
are deliberate: multi-layer capture, full-sequence logits, predictive-entropy
pooling; forward-loop idioms that mirror the cache builder are kept so the
two capture paths stay diff-able.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.hooks import RemovableHandle

os.environ.setdefault("USE_TORCH", "true")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepspec.data import ConversationCollator
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.modeling.dspark.glm5.hidden_states import fold_streams
from elsi_analyze import StreamAccumulator, log

torch.set_float32_matmul_precision("high")


def _get_hook_tensor(output: object) -> torch.Tensor:
    """First tensor of a target-module output (same helper as the cache builder)."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    msg = f"Unsupported hook output type: {type(output)!r}"
    raise TypeError(msg)


def token_entropies(logits: torch.Tensor, chunk_rows: int = 1024) -> torch.Tensor:
    """Per-row predictive entropies of flat ``[B*T, V]`` logits, fp32, chunked.

    Logit rows are soft-capped at -1e4 before the sum so deep-tail entries
    (p == 0, logp == -inf) contribute exactly 0 instead of a NaN product.
    """
    flat = logits.reshape(-1, logits.shape[-1])
    rows = flat.shape[0]
    out = torch.empty(rows, dtype=torch.float32, device=flat.device)
    log_cap = -1.0e4
    for start in range(0, rows, chunk_rows):
        block = flat[start : start + chunk_rows].to(torch.float32)
        log_softmax = torch.log_softmax(block, dim=-1).clamp_min(log_cap)
        out[start : start + chunk_rows] = -(log_softmax.exp() * log_softmax).sum(-1)
    return out


def top_fraction_mask(entropies: torch.Tensor, rho: float) -> torch.Tensor:
    """Bool mask of the top ``rho`` fraction of positions by entropy (paper sec 4.1).

    ``TopFrac({Ht}, rho)`` keeps the highest-entropy ``round(rho * T)``
    positions of one sequence (ties resolved toward earlier positions).
    """
    length = entropies.shape[0]
    keep = min(max(1, round(float(rho) * length)), length)
    values, indexes = torch.topk(entropies, keep, sorted=False)
    del values
    mask = torch.zeros(length, dtype=torch.bool, device=entropies.device)
    mask[indexes] = True
    del indexes
    return mask


@dataclass
class CaptureMeta:
    """Manifest of one capture run, written with the column files."""

    num_rows: int
    num_layers: int
    num_columns: int
    rho: float
    max_length: int
    target_model: str
    source_jsonl_paths: list[str]
    retained_counts: list[int] = field(default_factory=list)
    tokens_total: int = 0
    tokens_seen: int = 0
    wall_seconds: float = 0.0
    notes: str = (
        "raw folded decoder-layer outputs + post-final-norm stream (column "
        "num_layers) + per-token predictive entropies at softmax temperature 1; "
        "pooled chunk sums along the top-rho retained-token axis"
    )

    def to_json(self) -> dict[str, Any]:
        """Plain dict for the report/manifest file."""
        return asdict(self)


def build_accumulators(num_columns: int, dim: int) -> dict[int, StreamAccumulator]:
    """One chunk-sum accumulator per stream column (num_layers raw + post-norm)."""
    return {column: StreamAccumulator(dim) for column in range(num_columns)}


def make_layer_hooks(
    backbone: Any, num_layers: int, captured: dict[int, torch.Tensor]
) -> list[RemovableHandle]:
    """Hook every decoder layer plus the final norm; each stream folds to CPU.

    The final-norm hook fills column ``num_layers`` with the post-final-norm
    output (RMSNorm.forward may return ``(normed, residual)``; the helper takes
    the first tensor), matching what the cache stores for id 44.
    """

    def make_hook(layer_id: int) -> Callable[[Any, Any, Any], None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer_id] = fold_streams(_get_hook_tensor(output)).detach().cpu()

        return hook

    handles = [
        backbone.layers[layer_id].register_forward_hook(make_hook(layer_id))
        for layer_id in range(num_layers)
    ]
    handles.append(backbone.norm.register_forward_hook(make_hook(num_layers)))
    return handles


def get_target_backbone(target_model: Any) -> Any:
    """Text decoder stack of the loaded GLM-5.3 target.

    The architecture nests twice: ``ForConditionalGeneration.model`` is the
    composite ``Glm5NextModel`` whose ``language_model`` is the text stack
    carrying ``layers``/``norm``; the plain model resolves directly.
    """
    candidate = getattr(target_model, "language_model", None)
    if candidate is None:
        candidate = getattr(target_model, "model", target_model)
    if not hasattr(candidate, "layers"):
        candidate = getattr(candidate, "language_model", candidate)
    if not hasattr(candidate, "layers"):
        msg = "could not resolve the GLM-5.3 text decoder stack from the loaded target"
        raise AttributeError(msg)
    return candidate


def get_target_hidden_size(target_model: Any) -> int:
    """Hidden width of the text stack (same helper as the cache builder)."""
    text_config = getattr(target_model.config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None):
        return int(text_config.hidden_size)
    return int(target_model.config.hidden_size)


def accumulate_batch(
    accumulators: dict[int, StreamAccumulator],
    streams_cpu: list[torch.Tensor],
    entropies_cpu: torch.Tensor,
    *,
    row_lengths: list[int],
    rho: float,
    running: int,
) -> tuple[int, list[int]]:
    """Pool one batch's streams into per-column chunk sums; add in-place.

    ``streams_cpu`` holds ``[B, T, dim]`` bf16 per column (post-norm column
    last); ``entropies_cpu`` holds ``[B, T]`` predictive entropies. Each row is
    cut to its true length first (``row_lengths`` from the attention mask):
    without the cut, a short row padded to the batch max would let masked
    positions' garbage hiddens and logits enter the top-rho selection.
    Returns the next pooled offset plus each row's retained count.
    """
    batch_counts: list[int] = []
    for row, real_length in enumerate(row_lengths):
        mask = top_fraction_mask(entropies_cpu[row, :real_length], rho)
        nr = int(mask.sum())
        batch_counts.append(nr)
        if nr == 0:
            continue
        retained_positions = mask.nonzero().flatten()
        pooled = torch.arange(nr, dtype=torch.int64) + running
        for column, stream in enumerate(streams_cpu):
            selected = (
                stream[row, :real_length].index_select(0, retained_positions).to(torch.float32)
            )
            accumulators[column].add(pooled, selected)
        running += nr
    return running, batch_counts


def run_capture(
    target_model: str,
    train_data_path: list[str],
    output_dir: str,
    *,
    num_rows: int,
    max_length: int,
    rho: float,
    min_loss_tokens: int,
    local_batch_size: int,
) -> None:
    """Full capture run: load, hook, forward, entropy, pool, save columns."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dataset = JsonLineDataset(data_paths=train_data_path)
    rows = min(num_rows, len(dataset))
    subset = Subset(dataset, range(rows))
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template="glm",
        max_length=max_length,
        min_loss_tokens=min_loss_tokens,
    )
    loader = DataLoader(
        subset, batch_size=local_batch_size, collate_fn=collator, num_workers=2, drop_last=False
    )
    # the glm5_next family ships no CausalLM registry mapping: load the
    # architecture class directly so the full-vocab logits exist for the
    # predictive-entropy pass
    model = Glm5NextForConditionalGeneration.from_pretrained(
        target_model, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    backbone = get_target_backbone(model)
    num_layers = len(backbone.layers)
    hidden_size = get_target_hidden_size(model)
    log(f"[capture] model layers={num_layers}, hidden={hidden_size}")
    captured: dict[int, torch.Tensor] = {}
    handles = make_layer_hooks(backbone, num_layers, captured)
    num_columns = num_layers + 1
    accumulators = build_accumulators(num_columns, hidden_size)
    running = 0
    all_entropies: list[torch.Tensor] = []
    retained_counts: list[int] = []
    started = time.time()
    tokens_seen = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue
                moved = {
                    key: value.to(model.device, non_blocking=True) for key, value in batch.items()
                }
                output = model(
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                    use_cache=False,
                )
                logits = output.logits
                entropies_cpu = (
                    token_entropies(logits).reshape(logits.shape[0], logits.shape[1]).cpu()
                )
                del logits
                row_lengths = moved["attention_mask"].sum(dim=1).tolist()
                streams_cpu = [captured[column] for column in range(num_columns)]
                for column in range(num_columns):
                    captured.pop(column, None)
                all_entropies.extend(
                    entropies_cpu[row, :length].clone() for row, length in enumerate(row_lengths)
                )
                running, counts = accumulate_batch(
                    accumulators,
                    streams_cpu,
                    entropies_cpu,
                    row_lengths=row_lengths,
                    rho=rho,
                    running=running,
                )
                retained_counts.extend(counts)
                tokens_seen += int(moved["attention_mask"].sum())
                if (batch_idx + 1) % 8 == 0:
                    log(f"[capture] batch {batch_idx + 1}, pooled tokens {running}")
                del streams_cpu, output, moved
    finally:
        for handle in handles:
            handle.remove()
    log(f"[capture] done: {running} pooled tokens across {num_columns} columns")
    write_capture_outputs(
        out_path,
        accumulators,
        all_entropies,
        retained_counts,
        CaptureMeta(
            num_rows=rows,
            num_layers=num_layers,
            num_columns=num_columns,
            rho=rho,
            max_length=max_length,
            target_model=target_model,
            source_jsonl_paths=list(train_data_path),
            retained_counts=retained_counts,
            tokens_total=running,
            wall_seconds=time.time() - started,
            tokens_seen=tokens_seen,
        ),
    )


def write_capture_outputs(
    out_path: Path,
    accumulators: dict[int, StreamAccumulator],
    all_entropies: list[torch.Tensor],
    retained_counts: list[int],
    meta: CaptureMeta,
) -> None:
    """Persist per-column chunk sums, the entropy bundle, and the meta JSON."""
    for column, acc in accumulators.items():
        sums, counts = acc.snapshot()
        torch.save({"sums": sums, "counts": counts}, out_path / f"col_{column:03d}.pt")
    torch.save(
        {"entropies": all_entropies, "retained_counts": retained_counts},
        out_path / "entropies.pt",
    )
    (out_path / "capture_meta.json").write_text(json.dumps(meta.to_json(), indent=1) + "\n")
    log(f"[capture] wrote {len(accumulators)} columns + entropies.pt + capture_meta.json")


def main() -> None:
    """CLI surface for the capture stage (runs inside the deepspec image)."""
    parser = argparse.ArgumentParser(
        description="All-45-layer GLM capture with predictive-entropy pooling (issue 51 stage 2)."
    )
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",
    )
    parser.add_argument("--output-dir", required=True, help="capture output directory.")
    parser.add_argument("--target-model", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--num-rows", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--local-batch-size", type=int, default=2)
    args = parser.parse_args()
    run_capture(
        target_model=args.target_model,
        train_data_path=list(args.train_data_path),
        output_dir=args.output_dir,
        num_rows=args.num_rows,
        max_length=args.max_length,
        rho=args.rho,
        min_loss_tokens=args.min_loss_tokens,
        local_batch_size=args.local_batch_size,
    )


if __name__ == "__main__":
    main()
