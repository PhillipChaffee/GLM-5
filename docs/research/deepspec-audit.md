# DeepSpec audit as the vehicle for training a DSpark-style drafter against GLM-5.3-Flash

Research for ticket #2 (map #1). Compiled 2026-09-27.

## Answer in one paragraph

DeepSpec is real, complete, and usable: `deepseek-ai/DeepSpec` (MIT, ~7.2k stars) ships the full data-regeneration, training, and evaluation pipeline behind the DSpark paper's Table 1, plus standalone DSpark/Eagle3/DFlash drafter checkpoints on Hugging Face for Qwen3-4B/8B/14B and Gemma4-12B targets — i.e., DeepSeek themselves have already proven the recipe on non-DeepSeek targets. A drafter needs exactly four target hooks: hidden states from a handful of target layers (KV injection), the target's frozen embedding layer and LM head, a mask token id in the target vocab, and a family-specific target wrapper/trainer/chat-template — GLM would need one new family implemented in `deepspec/modeling/dspark/` plus a trainer subclass, which is a bounded but nontrivial patch given GLM-5.3-Flash's mHC hyper-connections (multiple hidden-state streams per layer) and hybrid linear/sparse attention. The pipeline is: regenerate 1.3M Open-PerfectBlend responses with the target (inference spend), precompute a disk cache of target hidden states (the killer detail: ~38 TB even for a 4B target; roughly 50-60 TB at GLM's hidden size unless layers are cut), then train the small drafter for 10 epochs on one 8-GPU node. Nobody has adapted DeepSpec/DSpark to GLM or to GLM-5.3-Flash's exact architecture, but close relatives exist: a community DSpark Markov head trained on a 75%-SSM hybrid target, and TorchSpec (DSpark + EAGLE3/DFlash training with streamed hidden states instead of a disk cache) has produced drafters for Kimi K3 MoE and others. A single GLM drafter run looks like roughly 500-1,500 H200-class GPU-hours plus 50-60 TB scratch storage — feasible on Modal, with the disk cache being the main engineering risk that TorchSpec's streaming design eliminates.

## 1. Where DeepSpec lives and what is released

- Repo: <https://github.com/deepseek-ai/DeepSpec> (created 2026-06-26, Python, ~7.2k stars, ~667 forks, default branch `main`). Self-described as "a full-stack codebase for training and evaluating draft models for speculative decoding."
- Layout: `deepspec/` package (`data/`, `eval/`, `modeling/`, `trainer/`, `utils/`), `config/{dflash,dspark,eagle3}/` with one config per target (`dspark_qwen3_4b.py`, `dspark_qwen3_8b.py`, `dspark_qwen3_14b.py`, `dspark_gemma4_12b.py`), `train.py` (spawns one worker per visible GPU), `eval.py`, `eval_datasets/` (gsm8k, math500, aime25, humaneval, mbpp, livecodebench, mt-bench, alpaca, arena-hard-v2), and `scripts/{data,train,eval}/` entry scripts.
- License: MIT, with `NOTICE` attributing adapted code from SpecForge (Apache-2.0) and z-lab DFlash (MIT). The DSpark Markov head is DeepSpec's own; DFlash backbone design credited to z-lab.
- Checkpoints (Hugging Face, `deepseek-ai` org, "DeepSpec" collection): standalone drafters for the paper's Table 1 — `dspark_qwen3_{4b,8b,14b}_block7`, `dspark_gemma4_12b_block7`, plus matching `dflash_*_block7` and `eagle3_*_ttt7` variants. Separately, production serving bundles `DeepSeek-V4-Flash-DSpark` (165B safetensors: the FP8 target with the spec module attached, MIT) and `DeepSeek-V4-Pro-DSpark`. The V4 drafter weights are inside the serving bundle, not a standalone trainable artifact.
- Data: prompts come from public `mlabonne/open-perfectblend`; the target-regenerated responses are NOT released — you regenerate them yourself (stage 2 of the pipeline).
- Completeness: the pipeline is end-to-end and runnable (README documents the 38 TB cache warning, 8-GPU default, config override flags). The repo does not ship multi-node launchers; the paper's production-scale tricks (hidden-state communication over HAI-LLM, asynchronous scheduler) are not in the public code — the public path materializes hidden states on disk instead.

## 2. Target-model hooks a drafter needs

From `config/dspark/dspark_qwen3_4b.py` and the paper (Sections 2-3, lines 144-163, 402-448):

- KV injection layer selection: `target_layer_ids=[1, 9, 17, 25, 33]` — five layers spread shallow-to-mid across the target's 36. Hidden states from these layers are concatenated, RMSNormed, and projected (`W_c`) into the draft hidden space, then injected into every draft layer's K/V as extra context the block attends to bidirectionally. For GLM-5.3-Flash (45 layers + 1 MTP layer), you would pick an equivalent spread; the exact layer choice is a hyperparameter DeepSpec does not explain how to re-derive.
- Embedding + LM head sharing: the drafter reuses the target's embedding layer and LM head, both frozen. Only the 5 draft layers, the sequential block (Markov head, rank 256), and the confidence head are trained. For GLM: hidden 4096, vocab 154,880, `tie_word_embeddings=false` — embeddings and LM head are separate ~634M-param tensors that get loaded frozen.
- Vocab handling: full target vocabulary throughout (no draft-vocab pruning in DeepSpec; TorchSpec separately supports d2t/t2d pruning, and DFlash2 requires full vocab). The config carries `mask_token_id=151669` — a real token id that must exist in the target vocab. Whether GLM's 154,880 vocab has a reserved/unused id suitable as a mask token needs verification against the tokenizer.
- Family-specific code surface: `deepspec/modeling/dspark/{qwen3,gemma4}/` (target-side wrappers that extract and cache hidden states), a per-family trainer class (`Qwen3DSparkTrainer` referenced from the config), and `data.chat_template` ("qwen"). Supporting GLM means: a `deepspec/modeling/dspark/glm5/` wrapper, a `GLM5DSparkTrainer` subclass, a glm chat template, mask token choice, and a config file. The training math (losses, anchor packing) is family-agnostic.
- GLM-specific risk: mHC hyper-connections mean each layer has multiple hidden-state streams (`hc_attn_base/fn/scale`, `hc_ffn_*` per the config's `modules_to_not_convert`). "Hidden state at layer l" is not single-valued; the wrapper must choose or combine streams. The DSpark paper's production drafter itself uses mHC (3 MoE layers, sliding window 128), so mHC inside the drafter is proven — but injecting mHC-carrying target hidden states is not.
- FP8: GLM ships FP8 with lm_head and embed_tokens excluded from quantization, so the frozen shared modules are BF16 — no obstacle.

## 3. Data-regeneration + training pipeline and compute footprint

From `scripts/data/README.md`, `scripts/train/train.sh`, and paper Section 3.3 / 4.1 / 5.1:

1. Download and split `mlabonne/open-perfectblend` (1.3M samples; chat 17.6%, math 39.4%, code 38.9%, instruction-following 4.1%) with a 5% held-out split.
2. Regenerate responses: serve the target behind any OpenAI-compatible engine (SGLang example with 8 workers, concurrency 32), regenerate with the model's recommended sampling (`temperature 0.7, top_p 0.8, top_k 20` for Qwen3), non-thinking mode, with resume support. For GLM-5.3-Flash this means serving a 320B FP8 MoE (ticket #3's engine-support findings feed directly in here).
3. Build the target cache (`prepare_target_cache.py`): one forward pass of the target over the regenerated data, dumping per-token hidden states for the configured layers. Storage warning in-tree: roughly 38 TB for the default Qwen3-4B setting, scaling with dataset size, sequence length, hidden dim, and captured layers; fewer captured layers = proportionally less cache. Scaling to GLM (hidden 4096 vs Qwen3-4B's 2560, 5 layers, bf16, ~1.3M sequences): order 50-60 TB.
4. Train: `train.sh` -> `train.py`, one worker per GPU, default single node 8 GPUs, `sharding_strategy="no_shard"`, bf16, torch.compile, lr 6e-4, warmup 4%, global batch 512, 10 epochs, checkpoints every 3000 steps. Loss = 0.1 CE + 0.9 total-variation (L1) + 1.0 confidence BCE, position-weighted `w_k = exp(-(k-1)/4.0)`. Anchor packing: `num_anchors=512` anchors sampled per sequence, γ=7-token blocks packed via token-level attention indices (the paper's "anchor-bounded sequence packing"; the public repo implements this as the disk-cache + packing path).
5. Evaluate: `eval.sh` over the nine benchmarks; report accepted length τ (incl. bonus token).

Compute footprint decomposition: (a) regeneration = full target inference over 1.3M samples — the largest GPU spend; (b) cache build = one target forward-only pass — smaller than (a) in FLOPs but dominates storage; (c) drafter training = cheap per-GPU-hour (drafter ~1-2B trainable params) but iterates 10 epochs over packed anchor blocks; (d) eval = small. The paper's internal variant replaces the disk cache with hidden-state communication between target and draft workers (O(d) per token) — that optimization is not in the public repo.

## 4. Adaptations to non-DeepSeek targets, especially GLM / hybrid attention

- Official, by DeepSeek: Table-1 DSpark/Eagle3/DFlash drafters for Qwen3-{4B,8B,14B} and Gemma4-12B — both non-DeepSeek, dense-attention targets. The V4-Flash/Pro production drafters target DeepSeek's own hybrid CSA+HCA attention, proving DSpark on hybrid targets in production, but those are not standalone trainable artifacts.
- Community, training adaptations:
  - `hikarioyama/dspark-aeon-27b` (MIT): DFlash backbone warm-start + DeepSpec's VanillaMarkov head (rank 256) trained for Qwen3.6-27B-AEON — explicitly a 75%-SSM hybrid target — with vLLM 0.23 patches, on-policy/tool-call corpus, ABBA A/B eval. +10.9% tok/s over DFlash baseline, lossless. This is the closest published proof that the DeepSpec Markov head transfers to a hybrid-attention target; note it is a head-transfer recipe on a DFlash warm start, not a full DSpark backbone retrain.
  - `lightseekorg/TorchSpec` (PyTorch blog, Ray Summit 2026): torch-native DSpark/EAGLE3/DFlash training with hidden states streamed from vLLM/SGLang/TRT-LLM via Mooncake — no 38 TB disk cache. Adopted by DigitalOcean (MiniMax-M2.5 EAGLE3), vLLM's Artificial Analysis EAGLE3, CoreWeave (Kimi K2.7 DFlash), fal (DSpark for Ideogram V4, 16x throughput), Tencent (AngelSpec). Released drafters include `lightseekorg/kimi-k3-dspark` and `Inferact/Kimi-K3-DSpark` — DSpark trained against a large MoE (Kimi K3). TorchSpec also ships DFlash-to-DSpark warm-start conversion and draft-vocab pruning.
  - Serving-only ports (no training): `ARahim3/mlx-dspark` (Apple Silicon; Gemma-4, Qwen3.8, Nemotron, LFM2.5...), several DGX Spark / RTX DeepSeek-V4-Flash serving recipes, `MiaAI-Lab/Qwen3.8-27B-...-DSpark` on SGLang.
- GLM specifically: nothing. Zero issues mentioning GLM in `deepseek-ai/DeepSpec`, zero DSpark/DeepSpec discussion in `zai-org/GLM-5`, and the only GLM+DSpark hit anywhere is `will702/mlx-vision` (DSpark-style serving for OCR VLMs on MLX, 1 star, no training code). No one has trained a drafter against any GLM-5.x target, and nothing covers GLM-5.3-Flash's exact combination (MoE + glm5_next hybrid linear/sparse attention + mHC + MTP layer).

## 5. Rough compute estimate for a GLM-5.3-Flash drafter run

Derived, not measured (exact Modal pricing is ticket #4/#5's job). Assumptions: DeepSpec defaults — 1.3M prompts, 10 epochs, block 7, 5 captured layers, 5 draft layers, hidden 4096, vocab 154,880; ~1k avg regenerated tokens/sample; ~35-40% MFU on H200-class.

- Stage A, regeneration: ~1.3e9 output tokens from a 320B-A18B FP8 MoE. A single 8-GPU H200 node (or equivalent B200 pair) batch-serves this class of model at roughly 5-15k tok/s, so ~25-70 node-hours. Call it 200-600 GPU-hours, the dominant inference spend.
- Stage B, target cache: forward-only, ~1.3e9 tokens x 2 x 18B active ≈ 4.7e19 FLOPs ≈ 4-5 hours on one 8-GPU node (halves in FP8). The real cost is storage: ~1.3e9 tokens x 5 layers x 4096 x 2 bytes ≈ 50-60 TB bf16 (or ~32 TB with 3 layers). This is a first-order design decision: Modal object storage/scratch must hold it, or switch vehicles.
- Stage C, drafter training: trainable ≈ 5 dense draft layers at d=4096 (~1.0B) + Markov head (2 x 154,880 x 256 ≈ 79M) ≈ 1.1B params. Packed positions ≈ 10 epochs x 1.3M samples x ~2k positions ≈ 2.6e10; FLOPs ≈ 6 x 1.1e9 x 2.6e10 ≈ 1.7e20, plus big-vocab head/TV-loss projections ≈ 0.7e20 → ~2.4e20 FLOPs ≈ 20-25 hours on one 8-GPU node. Call it 150-250 GPU-hours.
- Eval + STS calibration + retries: tens of GPU-hours.
- Total: order 500-1,000 GPU-hours H200-class (roughly 3-6 days on one 8-GPU node end to end), plus 50-60 TB transient storage. Confidence: low-medium — regen throughput and avg response length dominate the spread; the training stage is small enough that a cheaper recipe (fewer epochs, subsampled prompts, 3 captured layers, TorchSpec-style warm start from a DFlash run) plausibly halves it.

## Decision-relevant takeaways

1. DeepSpec is a viable vehicle — MIT, complete pipeline, proven beyond DeepSeek targets (Qwen3, Gemma4) with released reference drafters to validate a port against.
2. The GLM family implementation is bounded work (target wrapper + trainer subclass + template + config) but mHC's multi-stream hidden states and the mask-token question are real unknowns that need a small spike before committing.
3. The 50-60 TB hidden-state disk cache is the biggest architectural objection to DeepSpec-as-is on Modal; TorchSpec's Mooncake-streamed DSpark training is the proven alternative and would need the same GLM family work but no cache. A hybrid plan (port the family into DeepSpec first for correctness reference, run on TorchSpec for the real training) is worth considering.
4. No GLM adaptation exists anywhere as of this research — this would be the first.

## Sources

- DeepSpec repo: <https://github.com/deepseek-ai/DeepSpec> (README, config/dspark/dspark_qwen3_4b.py @ sha c33fe02, scripts/data/README.md @ sha c7bf552, scripts/train/train.sh @ sha 3310a06, deepspec/ tree)
- HF deepseek-ai org and DeepSpec collection: <https://huggingface.co/deepseek-ai> ; checkpoints `deepseek-ai/dspark_qwen3_{4b,8b,14b}_block7`, `dspark_gemma4_12b_block7`, `DeepSeek-V4-Flash-DSpark` (model card, MIT, vLLM `--speculative-config '{"method":"dspark",...}'`)
- DSpark paper (local): papers/DSpark-2607.05147.txt — Section 3.3 Training (line 402), Section 4.1 setup (line 454), Section 5.1 Scalable and Flexible Training (line 785), KV injection architecture (lines 144-163), Markov/confidence heads (lines 239-300)
- GLM-5.3-Flash config: <https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json> (hidden 4096, vocab 154880, 45+1 layers, 288 experts/8 active, 34 linear + 11 sparse attention layers, mHC, FP8, tie_word_embeddings=false, num_nextn_predict_layers=1)
- Community adaptations: <https://github.com/hikarioyama/dspark-aeon-27b> ; <https://github.com/lightseekorg/TorchSpec> ; <https://github.com/ARahim3/mlx-dspark> ; <https://github.com/will702/mlx-vision>
- GLM-5 technical report (local): papers/GLM-5-technical-report-2602.15763.txt (MTP placement, lines 204-213, 398-401, 1885)

## Unknowns and confidence

- GLM mask token availability in the 154,880 vocab: unverified (medium confidence it exists as a reserved special token; must check tokenizer).
- Which mHC stream(s) to capture as "layer l hidden state": unknown; needs a code spike against `deepspec/modeling/dspark/qwen3/` to see what the Qwen3 wrapper extracts and what GLM's analogue is.
- Whether vLLM/SGLang's dspark proposer path (proven on DeepSeek-V4 and Qwen/Gemma drafters) accepts a GLM-target drafter without engine patches: unknown; DeepSpec ships no GLM serving glue, and ticket #3 owns the engine-support question.
- Avg regenerated response length for Open-PerfectBlend under GLM sampling: assumed ~1k tokens (low confidence; drives Stage A estimate).
- Compute figures are derived from published FLOP/throughput arithmetic, not benchmarks: low-medium confidence; treat as order-of-magnitude.
- Whether DeepSeek will accept GLM-family contributions upstream (README says contributions welcome): untested.
