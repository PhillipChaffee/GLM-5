# Corroborate the inherited drafter-training recipe against public drafter-training runs

Research for ticket #65 (map #1; feeds back to #41's checklist, which this doc does not edit). Compiled 2026-10-13. Inquiry anchor kept fixed: AdamW in fp32 master weights, warmup_ratio 0.04 then cosine annealing to total steps, global_batch_size 512, lr 6.0e-4, bf16 model precision, weight_decay 0, grad clip 1.0, ~28k optimizer steps planned — from `experiments/glm-drafter/deepspec_glm/config/dspark/glm5_flash.py`.

## Bottom line

Corroboration on every anchor value; no per-item change rationale emerged. The six LR/optimizer values are verbatim the upstream DeepSpec DSpark config and are run publicly, unchanged, on a GLM target (SpecForge's GLM-5.2 DSpark recipe: lr 0.0006, warmup 0.04, clip 1.0, global batch 512), on DeepSeek-MoE targets, and independently reproduced without instability reports. The only public instability report in the lineage attaches to the EAGLE-3 recipe shape, not ours; lineage training-failure reports are loss-mask/data bugs. On torch.compile/flex_attention: silent degradation is a designed default of the API and torch 2.9.1 wheels carry silent-correctness bugs in exactly our flex+BlockMask pattern that are fixed only in 2.10.0 — so keep-`torch_compile=True` should carry a one-item compile-audit addendum for #41 (wording left to the session; TODO-1 below). Compile-cache reuse across container restarts is not reachable as configured: the Modal app wires no `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR`, so caches live in-container and die with it.

## 1. LR / schedule precedent

### 1.1 Provenance of the six values

The values copy upstream verbatim: DeepSpec `config/dspark/dspark_qwen3_8b.py` carries lr 6.0e-4, warmup_ratio 0.04, weight_decay 0.0, precision bf16, global_batch_size 512, max_grad_norm 1.0, torch_compile True (confirmed against <https://github.com/deepseek-ai/DeepSpec/blob/main/config/dspark/dspark_qwen3_8b.py>). Our run inherits upstream `deepspec/trainer/base_trainer.py` + `deepspec/utils/optim.py`: an in-file-attributed port of SpecForge's `BF16Optimizer` — AdamW over fp32 master copies of the bf16 trainables, wrapped in `CosineAnnealingWarmupLR` with `warmup_steps = int(0.04 * total_steps)` (<https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/utils/optim.py>). The DSpark paper (arXiv 2607.05147, Section 4.1) publishes the 10-epoch budget and loss weights (ce 0.1 / tv 0.9 / conf 1.0) but no LR/batch figures — those live only in the repo configs.

### 1.2 Public runs at the anchor values

- DeepSpec DSpark configs (Qwen3-4/8/14B, Gemma4-12B): 6.0e-4 / cosine / 0.04 / clip 1.0 / gbs 512 / 10 epochs (repo configs, above).
- SpecForge DSpark Qwen3-8B offline: lr 6.0e-4, warmup 0.04, clip 1.0, default cosine, 6 epochs / max_steps 10000 — `examples/configs/offline/colocated/qwen3-8b-dspark-offline.yaml`.
- SpecForge DSpark **GLM-5.2** disaggregated: lr 0.0006, warmup_ratio 0.04, clip 1.0, global batch 512 ("portable one-rank equivalent of the source recipe's global batch 512"), 10 epochs — `examples/configs/online/disaggregated/external/glm-5.2-dspark-disaggregated.yaml`. Closest public match: GLM-family MoE target, identical six values.
- DFlash paper Appendix A.1 (arXiv 2602.06036v2): "The draft models are optimized for 6 epochs using AdamW with a learning rate of 6e-4, a gradient clipping threshold of 1.0, and a cosine schedule with a warmup ratio of 0.04", 512 anchors/sequence — the same trio from an independent line, pre-dating those configs.
- Our planned ~28k steps sits between the DeepSpec-scale public runs (10k-level; DeepSpec issue #84 discusses a step-12000 checkpoint) and RedHatAI's streamed 271k-step run below: every bracket of the scale has a public run at the anchor values, all inside the "tens of thousands of steps" question window.

### 1.3 Documented digressions in the lineage — chosen for regime, not instability

No publicly documented digression is framed as an LR-stability fix:

- SpecForge constant-LR for streamed big-target runs: Kimi-K3 and DeepSeek-V4-Flash DSpark recipes set `lr_scheduler: constant`, warmup 0, global batch 128, keeping lr 6.0e-4 (`kimi-k3-dspark-disaggregated.yaml`, `deepseek-v4-flash-dspark-disaggregated.yaml`). The Kimi-K3 recipe doc says the constant schedule is preserved from the prior colocated continuing run (`docs/recipes/kimi-k3-dspark-disaggregated.md`) — regime fidelity, not repair.
- RedHatAI speculators previews, same target class, different optimizer family. The GLM-5.3-Flash preview card (<https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview>) — our exact target — publishes: `optimizer: muon` (Muon over backbone + AdamW elsewhere, both lr 1e-3), `scheduler_type: linear` (speculators library default; unset warmup = total//100, per vllm-project/speculators `optimizers.py`/`trainer.py`), 3 epochs, released epoch-2 checkpoint at global step 270,979, block_size 8, seq len 8192, max_anchors 512, loss ce 0.1 / tv 0.9, streamed on-the-fly hidden-state extraction (vLLM + Mooncake, GB300 trays, 4 FSDP training ranks). Their GLM-5.2 run: AdamW, lr 3e-4, 1 epoch, anchors 1024; Kimi-K3: Muon 1e-4, cosine, warmup_ratio 0.03. Three runs, three optimizer shapes, zero instability notes on any card; the shared DSpark constants (anchors 512, ce 0.1/tv 0.9, rank-256 vanilla Markov head, confidence head with markov) match ours.
- EAGLE-3 lineage: official `eagle/traineagle3/ds_config.json` (<https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json>): AdamW betas (0.9, 0.95), WarmupDecayLR peak 5e-5, warmup 12,000 of 800,000 scheduled steps, clip 0.5. SpecForge's optimizer docstring calls this file its defaults source ("defaults copied from EAGLE traineagle3 ds_config.json"), and its schema defaults (cosine, warmup_ratio 0.015, clip 0.5) descend from it. EAGLE-3 recipes run lr 1e-4 (qwen3-8b-eagle3-offline.yaml) — order of magnitude below us with a different architecture/loss; its stability signal does not transfer to DSpark.

### 1.4 Instability reports found — all attached to shapes other than ours

- sgl-project/SpecForge#577 (open): "Qwen3-8B EAGLE3 example: learning_rate=1e-4 shows large grad-norm outliers in early training." User LR ablation found large grad-norm outliers at the EAGLE-3 recipe's default; smaller LRs calmer; no NaN. Scoped to the EAGLE-3 recipe (clip 0.5), not DSpark; no maintainer LR change made.
- SafeAILab/EAGLE#286 (open) plus #340/#317/#341: "Training loss does not decrease" — diagnosed as loss-mask/tokenizer mismatches and non-regenerated data setups. The lineage's failure mode is wrong mask/data, never LR-at-scale.
- deepseek-ai/DeepSpec#84 (open) is the best external datum for us: an independent 10-epoch repro of the exact 6e-4 schedule shows DSpark and DFlash monotone — "every per-1000-step increment stays positive through the final step" — with final checkpoints within ~1% of DeepSeek-released DSpark/DFlash; only EAGLE-3 turns over (peaks ~4.6 epochs then degrades). Relevant side-note for #41's epoch-budget item: turn-over belongs to EAGLE-3, not the DSpark schedule.
- DFlash: no public instability record found (z-lab/dflash repo is serving-only; the DFlash2 blog publishes no training hyperparameters, <https://inco.ai/blog/dflash2/>).
- No public record found of any mid-run LR reduction or schedule-instability post-mortem anywhere in the surveyed lineage.

### 1.5 Verdict for #41 (LR/optimizer items)

Corroboration to keep all six values. Public precedent exists at identical values on a GLM MoE target and the same drafter family, with a monotone-loss independent reproduction, and no public instability or LR-change record at our shape. Observability already exists regardless: the training loop logs lr and global grad_norm every logging_steps=10 (upstream `base_trainer.py` passes both to `training_logger.on_optimizer_step`), and per-epoch checkpoints every 3000 steps make a mid-run schedule problem stoppable, per #41's own acceptance criteria.

Per-item corroboration map for #41's checklist:

| #41 item | External precedent | Status |
|---|---|---|
| lr 6.0e-4 | Identical value in DeepSpec DSpark configs, SpecForge Qwen3-8B/GLM-5.2/Kimi-K3/Dv4-Flash DSpark recipes, DFlash paper A.1 (§1.2, §1.3) | keep |
| warmup_ratio 0.04 + cosine | Same pair in DeepSpec + SpecForge GLM-5.2 + DFlash paper A.1; optimizer code verified upstream (§1.1) | keep |
| AdamW, fp32 master weights | Upstream port of SpecForge BF16Optimizer; SpecForge default optimizer semantics (§1.1) | keep |
| weight_decay 0, grad clip 1.0 | Both verbatim in upstream config and SpecForge DSpark recipes (§1.2) | keep |
| global_batch_size 512 | GLM-5.2 recipe carries it verbatim; Kimi-K3/Dv4-Flash digress to 128 for a streamed regime, not stability (§1.3) | keep |
| num_train_epochs 10 | DSpark paper §4.1 "each drafter is trained for 10 epochs to ensure full convergence"; DeepSpec#84 monotone-DSpark repro; EAGLE-3 turn-over does not transfer (§1.4) | keep |
| torch_compile True | DeepSpec configs set it; 2.9.1 silent-correctness caveats → compile-audit addendum (TODO-1, §2.2/§2.4) | keep + addendum |

## 2. torch.compile + flex_attention (BlockMask) precedent

Our stack: Modal B200 (sm_100), upstream DeepSpec pins `torch==2.9.1` + `triton==3.5.1` (<https://github.com/deepseek-ai/DeepSpec/blob/main/requirements.txt>), cu12 wheels per the ticket anchor. The drafter's attention is `torch.nn.attention.flex_attention` with a per-batch `create_block_mask` closure capturing `anchor_positions`/`block_keep_mask` tensors (`deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py:26,153-157`; `TRAIN_ATTN_IMPLEMENTATION = "flex_attention"` in config.py), and the trainer compiles the whole draft model with `torch.compile(self.model, dynamic=True)` (upstream `base_trainer.py`).

### 2.1 Silent partial/eager degradation is the designed default, and it is auditable

- Graph breaks never error: the compiler FAQ states "the main reason you won't see the speedups you'd like ... is excessive graph breaks", and only `fullgraph=True` turns breaks into errors (<https://docs.pytorch.org/docs/2.9/torch.compiler_faq.html>). Our trainer uses the default fullgraph=False, so a degraded path surfaces as lost fusion/perf, not a failure.
- Recompiles are silent and capped: `torch._dynamo.config.recompile_limit = 8` per code object and `fail_on_recompile_limit_hit = False` — past the limit no error is raised (<https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py>).
- Official audit levers: `TORCH_LOGS="graph_breaks,recompiles"`, `torch._dynamo.explain`, and the `backend="eager"` first rung of the compiler FAQ's crash-bisection ladder.
- The FAQ's correctness section acknowledges downstream-compiler numerics "can be different in subtle ways yet have dramatic impact on your training stability" — compile quality is a training-stability variable, not just a perf knob.

### 2.2 Known flex_attention / BlockMask compile bugs around 2.9.x/2.10

Ordered by relevance to our exact call pattern (batch-dependent BlockMask closure, dynamic=True, packed anchor blocks):

- pytorch/pytorch#186775 (closed unconfirmed): "Flex Attention produces NaN with torch.compile(flex_attention, dynamic=True) and batch-dependent mask closures" (filed 2026-06-09). Reporter isolated the NaN to Inductor codegen for batch-dependent BlockMasks with complex partial blocks; NaN vanished under `backend="eager"` and with text-only batches. Maintainer closed with "try a newer version ... fixed since the release of 2.8" — never re-verified on a 2.9/2.10 wheel with the partial-block pattern. The closure pattern (per-batch mask tensors, partial blocks at sequence tails) is ours.
- pytorch/pytorch#198513 (closed) + fix PR #198543: silent wrong flex output under Inductor when query seq_len < block size (128) — the flex-decoding lowering read q's strides while unfrozen and baked them into an `as_strided` GQA view, so the kernel read q at wrong offsets. Fix merged 2026-09-26 → first stable containing it is torch 2.10.0; **2.9.1 carries the bug**. Trigger: query length < 128 plus a pointwise op feeding q/k.
- pytorch/pytorch#187547 (open) + #187272 (closed, PR #187273): Dynamo graph-breaks on raw unbacked SymInt BlockMask lengths in FlexAttention; the non-strict-tracing variant was fixed, the general Dynamo form remains a tracked TODO in `variables/builder.py`. Graph-break class (perf loss, not wrongness).
- pytorch/pytorch#188878 / #188871 (closed): FlashAttention-backend-of-flex silent wrongness — wrapped gather index misclassified as lane-uniform in vectorized score_mod groups, and Int64/Int32 dtype mismatch on captured-buffer negative-index wrap. Captured-buffer score_mod is our shape (the mask closure captures tensors). Fixes land in the 2.10-era line.
- pytorch/pytorch#199053 (open): compiled flex_attention returns wrong values with strided views / storage-offset inputs (inductor/CPU form); fix PR #199540 merged 2026-10-06 for the CPU template choice.
- pytorch/pytorch#180651 (open): flex autotuner uses a "misleading block-mask" → overhead and slower configs, silently (degradation without error).
- Adjacent sm_100 context: pytorch/pytorch#193893 (open) SDPA cuDNN backend silently corrupts on B200 for num_kv_heads < 8; pytorch/pytorch#190973 (open) Inductor SM100 ptxas VIMNMX3 fusion drops a negated integer-abs operand. Neither is our path, but both document silent-corruption climate on sm_100 in this window. B200 CI exercises the flex path across both versions (#162820 8xB200 ring-flex CI failures; #164593 tracking the B200-vs-triton#8328 regression class).
- The 2.9.0 release notes closed their own flex silent-correctness bugs ("Fix silent correctness w/ backpropping grads for FlexAttention" #163677; head broadcast #163426), showing 2.9-line-era flux. The 2.10.0 release notes add what 2.9.1 lacks: FlexAttention Flash-Attention support (#161118), Inductor Blackwell GPU templates including FlexAttention configs (#162916, #163147, #167340), "mechanism to get optimal autotune decision for FlexAttention" (#165817), and public `torch.compiler.config.force_disable_caches` (#166699). Upgrade BC note: nested-jagged support removed from flex_attention (#161734) — we do not use nested/jagged inputs; no impact found.

### 2.3 Restart-cache behavior (TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR on mounted volumes)

As configured, cross-restart reuse is not reachable: neither `experiments/glm-drafter/glm_drafter.py` nor the overlay sets those env vars, so Inductor/Triton caches land under the container's home (Linux default `~/.cache`) and vanish with each restart — slow cold starts, no corruption surface. The public hazard set applies only if cache dirs are later pointed at the Modal volume (a network filesystem):

- triton-lang/triton#11512 (open): "Concurrent multi-process compilation with a shared TRITON_CACHE_DIR on a network filesystem crashes: EBUSY from os.removedirs in cache put, FileNotFoundError reading back just-published files" — type-matched to our 4-rank FSDP training ranks sharing one mounted cache dir. Related: triton#2688 (open, FileNotFoundError while compiling Triton kernels in distributed training).
- Hash-key caveats: pytorch/pytorch#197519 (open) FX-graph cache key ignores tensor pinned-ness; #196728 → PR #196860 (closed 2026-09-18) inductor autotune cache crashed under the newer fp32_precision setting — a cache-reading error mode; triton#10107 `KeyError: "Unknown key: 'cubin'"` on version-mismatched cache reuse during serving.
- Official recognition of the failure class: the troubleshooting doc section "Cold Start Timing and Cache Corruption Debugging" — `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` / `torch.compiler.config.force_disable_caches = True` as debug/kill switch (<https://docs.pytorch.org/docs/2.14/torch.compiler_troubleshooting.html>).

### 2.4 Verdict for #41 (torch_compile item)

Corroboration to keep `torch_compile=True` as the checklist value: upstream and SpecForge DSpark public recipes ship and run it on this exact stack family. Not clean at 2.9.1, however — see 2.2. The #41 acceptance item ("verify whether torch_compile actually engaged in the smoke") already has app support: the sweep arms exist at `experiments/glm-drafter/glm_drafter.py:393-404`.

## Open items for the session (TODO — not decided here; the #41 checklist is deliberately untouched)

- TODO-1 (addendum for #41's torch_compile item; wording left to the session): extend the smoke's compile-engagement check with a one-time audit — first optimizer steps with `TORCH_LOGS="graph_breaks,recompiles"` (record break count/reasons), plus a short compiled-vs-`backend="eager"` loss-equivalence window to bound the #186775/#198513 silent-correctness classes. This is the only place where the external evidence argues for an action beyond the current checklist.
- TODO-2 (image decision, not a checklist value): whether to float the training image to torch 2.10.0 to inherit the flex-decoding stride fix (#198543), the FLASH captured-buffer fixes, and the Blackwell FlexAttention autotune templates — weighed against upstream-deps drift (upstream pins 2.9.1, and we inherit that pin through its requirements.txt).
- TODO-3 (cache policy, only if revisited): if compile caches are ever mounted on the Modal volume, per-run-unique subdirectories are required by the triton#11512 concurrent-network-FS failure mode (4 ranks sharing one dir), or use the corruption-debug kill switch on failure. As wired today, nothing to do.

## Source register

- deepseek-ai/DeepSpec: `config/dspark/dspark_qwen3_8b.py`; `deepspec/trainer/base_trainer.py`; `deepspec/utils/optim.py`; `deepspec/modeling/dspark/common.py` (create_block_mask per batch); `requirements.txt` (torch 2.9.1 / triton 3.5.1); issue #84; <https://github.com/deepseek-ai/DeepSpec>, <https://github.com/deepseek-ai/DeepSpec/issues/84>. DSpark paper: <https://arxiv.org/abs/2607.05147>.
- sgl-project/SpecForge: `specforge/optimizer.py`; `specforge/config/schema.py` (defaults cosine / 0.015 / clip 0.5 / flex_attention); example YAMLs cited inline in 1.2/1.3; `docs/recipes/kimi-k3-dspark-disaggregated.md`; issue #577 (<https://github.com/sgl-project/SpecForge/issues/577>).
- SafeAILab/EAGLE: `eagle/traineagle3/ds_config.json`; issue #286 (<https://github.com/SafeAILab/EAGLE/issues/286>). EAGLE-3 paper <https://arxiv.org/abs/2503.01840> (no optimizer figures).
- DFlash: arXiv 2602.06036 Appendix A.1 (anchor trio verbatim); z-lab/dflash (serving-only, <https://github.com/z-lab/dflash>); DFlash2 blog <https://inco.ai/blog/dflash2/> (no training figures).
- RedHatAI speculators previews: <https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark-preview> (GLM-5.2 and Kimi-K3 dspark cards in the same org); vllm-project/speculators `optimizers.py` / `schema.py` / `trainer.py` (defaults adamw / linear / warmup total//100); the preview card names training commit b9c51bc.
- pytorch/pytorch issues and PRs cited inline: #186775, #187272/#187273, #187547, #198513/#198543, #199053/#199540, #188878/#188871, #180651, #193893, #190973, #198224, #162820, #164593, #196728/#196860, #197519; releases v2.9.0/v2.9.1/v2.10.0.
- triton-lang/triton: #11512, #2688, #10107 (all cited inline).
- PyTorch docs: compiler FAQ (2.9) <https://docs.pytorch.org/docs/2.9/torch.compiler_faq.html>; troubleshooting cache-corruption section <https://docs.pytorch.org/docs/2.14/torch.compiler_troubleshooting.html>; `torch/_dynamo/config.py` recompile limits.
- In-repo anchors: `experiments/glm-drafter/deepspec_glm/config/dspark/glm5_flash.py`, `experiments/glm-drafter/glm_drafter.py`, `experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py`, and the audit this doc extends: `docs/research/deepspec-audit.md`.
