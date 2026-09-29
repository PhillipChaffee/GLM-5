# GLM-5.3-Flash: target-side facts for drafter training and serving (ticket #3)

## Answer in one paragraph

GLM-5.3-Flash is a 320B-total / 18B-active natively multimodal MoE (arch `glm5_next` / `Glm5NextForConditionalGeneration`) with 45 text layers — 34 gated-delta (KDA) linear-attention layers and 11 DeepSeek-style sparse-attention (DSA) layers using NoPE sparse MLA with a lightweight indexer — a 154,880-token vocabulary, 288 routed + 1 shared experts with top-8 routing, 4-branch manifold-constrained hyper-connections (mHC), native FP8 (E4M3 dynamic) weights (~306 GiB), and a 1M-token context. It ships exactly one native MTP draft layer (`num_nextn_predict_layers: 1`, checkpoint layer index 45, DeepSeek-NextN style: `enorm`/`hnorm`/`eh_proj` + a full DSA/MoE decoder layer + shared LM head) that both SGLang and vLLM can drive natively today — SGLang via `--speculative-algorithm EAGLE` (NEXTN alias) at 5 steps / top-k 1 / 6 draft tokens, vLLM via `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'` — with measured accept lengths near the 6-token ceiling on GB300. For a custom (DSpark-style) drafter, SGLang has the deepest and most relevant surface (in-tree DSpark implementation with KV injection, planner, verify, STS calibration, plus EAGLE/EAGLE3/DFLASH/STANDALONE external-drafter paths and an access-gated GLM-5.3-Flash-DFlash2 checkpoint), TokenSpeed documents a complete external DSpark/DFLASH/EAGLE3 drafter contract, and vLLM offers a generic experimental `custom_class` proposer plus per-family DSpark modules for DeepSeek-V4/V4.1, Kimi-K3, Gemma-4 and Qwen3 — but nothing glm5next-specific, so vLLM would need either the custom proposer or a new family module. The z.ai blog itself never mentions MTP or speculative decoding; its serving story is ReplaySSM, W8A8 quantization, hybrid INT8/FP8/BF16 cache quantization, Layer Split and EPD disaggregation on Chinese AI chips, while the MTP/acceptance claims (+up to 20% acceptance length vs GLM-5.2) live in the repo README and the GLM-5 technical report.

## 1. Architecture from the HF config and model card

Authoritative source: `https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json` (the `/raw/` URL 404s; `/resolve/main/` works), plus the model card and vLLM recipe.

The default checkpoint is FP8 (`zai-org/GLM-5.3-Flash`); a BF16 variant (`zai-org/GLM-5.3-Flash-BF16`) and an NVFP4 variant (`RedHatAI/GLM-5.3-Flash-NVFP4`, Blackwell-only) exist. License: MIT.

### Architecture fact table (text tower; `text_config`)

| Fact | Value | Source |
|---|---|---|
| Architecture | `Glm5NextForConditionalGeneration`, model_type `glm5_next` (multimodal wrapper; text tower `glm5_next_text`) | config.json |
| Total / active params | ~321B / 18B ("320B-A18B") | model card, blog |
| Hidden dim | 4096 | config.json `hidden_size` |
| Text layers | 45 total: 3 dense (first_k_dense_replace 3) + 42 sparse MoE | config.json |
| Attention mix | 34 `linear_attention` (KDA) + 11 `deepseek_sparse_attention` (DSA), 3:1 repeating pattern | config.json `layer_types`, `linear_attn_config` |
| Vocab size | 154,880 | config.json `vocab_size` |
| Context | 1,048,576 tokens | config.json `max_position_embeddings` |
| KDA linear attention | 64 heads x 128 head_dim, short conv kernel 4, `gate_lower_bound` -5 | config.json `linear_attn_config` |
| DSA sparse MLA (NoPE) | kv_lora_rank 512, q_lora_rank 1536, qk_head_dim 256, qk_rope_head_dim 0 (NoPE, `mla_use_nope: true`), v_head_dim 256, 64 Q heads / 64 KV heads | config.json |
| DSA indexer | 32 heads x 128 dim, `index_topk` 2048, `index_kpool` 4 with compression (IndexPool: 4 key vectors pooled into 1), `always_select_tail`, `index_share_for_mtp_iteration: true` | config.json; blog |
| MoE | 288 routed + 1 shared experts, top-8 per token, moe_intermediate_size 2048, dense intermediate 12288, sigmoid scoring, noaux_tc, routed_scaling_factor 2.5, router dtype float32 | config.json |
| mHC | `mhc: true`, `hc_mult: 4` (4 branches), `hc_sinkhorn_iters: 20` | config.json |
| MTP | `num_nextn_predict_layers: 1` — one draft layer at index 45: `enorm`, `hnorm`, `eh_proj` (2*4096 -> 4096), a full DSA-MLA + MoE decoder layer (`mtp_block`), `shared_head` (norm + LM head, shared embed) | config.json; vLLM `vllm/models/glm5next/common/mtp.py` |
| FP8 | E4M3, dynamic activation scheme; kept unquantized: lm_head, embed_tokens, hc_* params, MLA a-projections (q_a/kv_a/o_proj held BF16), norms, gates, indexer norms/biases | config.json `quantization_config`; vLLM loader |
| Vision | ViT depth 24, hidden 1024, 16 heads, patch 14, 448px, out 4096, temporal_patch_size 2 (image + video) | config.json `vision_config` |
| Weight memory | FP8 ~306 GiB; BF16 ~2x that | vLLM recipe; KTransformers tutorial |
| Transformers version | 5.16.0 (`transformers_version` in config) | config.json |

### MTP layer shape

`num_nextn_predict_layers = 1`. The vLLM implementation (`vllm/models/glm5next/common/mtp.py`, file SHA 8361a05) shows the exact shape: EAGLE/NextN-style inputs — `enorm(embed(t))` fused with `hnorm(previous hidden)` concatenated through `eh_proj` into the MTP decoder block; the block is a full GLM5Next decoder layer with sparse-MLA attention (it builds the DSA `topk_indices_buffer` and reuses the indexer) and an MoE MLP (288-expert `Glm5NextMoE`); a `shared_head` (RMSNorm + LM head deferred to the target weights) produces logits. Drafting is greedy via a vocab-parallel argmax (`get_top_tokens`) that avoids materializing full-vocab logits. `index_share_for_mtp_iteration` means the indexer top-k is computed at step 0 and reused for steps 1+. Weight loading fails closed if the checkpoint lacks the MTP layer. Estimated MTP cost: one MoE decoder layer is ~7.3B params (288 experts x 3 x 4096x2048) plus ~0.6B LM head, i.e. roughly 8-9 GiB in FP8/BF16 mix — small against the 306 GiB target.

### MTP design lineage (GLM-5 technical report, `papers/GLM-5-technical-report-2602.15763.txt`)

- During training GLM-5 uses 3 MTP layers with shared parameters: same draft memory footprint as DeepSeek-V3's single layer, but higher acceptance because train/inference discrepancy is reduced (report lines ~204-215); "the acceptance length of GLM-5 is longer than DeepSeek-V3.2, given the same number of speculative steps".
- Table 10 confirms MTP Layers = 1 for both GLM-4.5 and GLM-5, and vocab 154,880 for GLM-5 (Flash inherits both).
- Flexible MTP placement under interleaved pipeline parallelism; the MTP output layer is co-located with the main output layer on the final stage (lines ~398-401).
- FP8 rollouts + MTP are used in RL (slime) to cut tail latency from stragglers (lines ~758-763).
- The repo README (GLM-5.3 section) states the family "improves GLM-5.2's MTP layer for speculative decoding, increasing the acceptance length by up to 20%", and adds IndexShare (arXiv 2603.12201): one indexer shared across every four sparse-attention layers, cutting per-token indexer FLOPs 2.9x at 1M context.

## 2. Serving engines with native MTP support today

| Engine | Native MTP today? | Flags |
|---|---|---|
| SGLang | Yes (v0.5.20+ for this model) | `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` (SGLang folds the older `NEXTN` spelling into EAGLE; the draft path is auto-handled from the in-checkpoint MTP head). DSA draft workers default to the `sgl-kernel` top-k backend, overridable with `--speculative-dsa-topk-backend`. MTP + DP-attention is explicitly not validated. Adaptive spec decoding (`--speculative-adaptive`) works on top (EAGLE/EAGLE3, top-k 1 only) and rescales steps per batch-size tier from EMA accept length. |
| vLLM | Yes (recipe says vLLM 0.29.0+) | `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`. FlashInfer 0.6.17+ (0.6.18 validated) required for NoPE sparse MLA. MTP survives prefill/decode disaggregation (must match `num_speculative_tokens` on both pools). AMD MI355X MTP validated at TP4 (vLLM PR #55239); Ascend 950PR single-node TP8. |
| TokenSpeed | Yes | `--draft-model-path-use-base --speculative-algorithm MTP --speculative-num-steps 2 --speculative-num-draft-tokens 3` with TP8 + EP; auto-configures KDA/DSA backends. |
| Transformers | Reference implementation only | `glm5_next` docs; the MTP weights ship in the checkpoint but this is not a serving path. |
| KTransformers | Not documented | CPU-GPU hybrid FP8 serving (SM89/SM120 + AVX-512 FP8 CPU expert kernel); no MTP mention in its tutorial. |

Measured SGLang acceptance (cookbook perf campaign, 4x GB300, upstream main): accept length ~5.98-6.0 (chat) and ~4.5-6.0 (summarization) at MTP 5/1/6; GSM8K 97.14% with MTP + BF16 KV + TileLang DSA; the cookbook suggests `SGLANG_SIMULATE_ACC_LEN=3` for reproducible throughput benchmarking and notes that at heavy batching, disabling speculation can be more efficient.

## 3. Custom-drafter integration surface per engine

**SGLang — deepest surface.** External drafters accepted at `--speculative-draft-model-path` under five algorithms: `EAGLE`/`EAGLE3` (feature/tap-based draft heads; per-draft overrides: `--speculative-draft-attention-backend`, `--speculative-draft-model-quantization`, `--speculative-draft-model-revision`, `--speculative-draft-load-format`, `--speculative-token-map` FR-Spec, torch.compile for the draft), `DFLASH` (block verifier; `--speculative-dflash-block-size`, `--speculative-draft-window-size`; checkpoint properties like LiLiCorr are picked up from the draft config, not flags), `STANDALONE` (any smaller draft LLM, token-level), `UNO` (LoRA on the target itself), `NGRAM`. Critically for this effort, SGLang already ships a full DSpark implementation in-tree: `python/sglang/srt/models/dspark.py` (generic drafter model class), `dspark_components/` (`dspark_draft.py`, `dspark_draft_sampler.py`, `dspark_config.py`, `dspark_kv_inject.py` — the KV-injection mechanism, `dspark_verify.py`, `dspark_planner.py`, `dspark_observability.py`), `dspark_disaggregation.py`, `benchmark/dspark_sts_fit.py` (Sequential Temperature Scaling calibration), and a `--speculative-dspark-block-size` flag plus `--enable-linear-replayssm-spec` (ReplaySSM: prefix replay for linear-attention state, the same problem DSpark's Prefix Replay solves). Existing per-model DSpark drafter classes: `deepseek_v4_dspark.py`, `lfm2_dspark.py` — there is no `glm5_next_dspark.py` yet, so a DSpark-style GLM drafter means writing one family module (drafter model class + wiring into the existing DSpark worker components). The cookbook also references `incoai/GLM-5.3-Flash-DFlash2`, an access-gated HF checkpoint ("request access"), launchable with `--speculative-algorithm DFLASH --speculative-draft-model-path ... --speculative-draft-attention-backend fa4`.

**vLLM — generic but shallow for this model.** Methods: `draft_model` (any HF draft, `draft_tensor_parallel_size`, `use_heterogeneous_vocab` TLI for cross-tokenizer drafts, `parallel_drafting` flag), `eagle3`, `dflash`, `mtp`, `pard` (Parallel Draft Model — low draft latency), MLP speculator, ngram, suffix. Two custom paths: (a) the experimental Custom Proposer Backend — `speculative_config.method = "custom_class"`, `speculative_config.model = "your_module.YourCustomProposerClass"`, class takes `VllmConfig` and implements `propose()`; (b) the `vllm-project/speculators` kit for training draft models and integrating them. vLLM already carries per-family DSpark modules — `vllm/models/deepseek_v4/{nvidia,amd,xpu}/dspark.py`, `deepseek_v41/{nvidia,amd}/dspark.py`, `kimi_k3/nvidia/dspark_mla.py`, `gemma4_dspark.py`, `qwen3_dspark.py` — and "Adaptive Verification (sizes verification per request from drafter confidence; currently DSpark only)". But searches found no glm5next EAGLE or DSpark wiring (`path:glm5next eagle` -> 0 results), so integrating a DSpark-style GLM drafter in vLLM means either a new family module mirroring `deepseek_v4/nvidia/dspark.py` or the experimental custom proposer. Generic constraint: pipeline parallelism is not composable with speculative decoding.

**TokenSpeed — most complete documented external block-drafter contract.** `--speculative-algorithm` accepts `EAGLE3`, `MTP`, `DFLASH`, `DSPARK`; external checkpoints via `--speculative-draft-model-path` (e.g. `nvidia/MiniMax-M3-DSpark`, described as "a six-layer Qwen3-shaped GQA block drafter with a vanilla Markov head"); drafter-specific backends via `--drafter-attention-backend` and `--draft-moe-backend`; EAGLE3 tap selection via `--eagle3-layers-to-capture` or the draft config's `eagle_aux_hidden_state_layer_ids`. Block conventions are enforced fail-fast at startup: DSpark checkpoints store the drafted-token count (`block_size` = `--speculative-num-steps`, verify width = steps+1); DFlash/DFlash2 store the verify width. Block drafts write KV at the target's cache locations and share the target's page table; a draft's sliding window is an attention mask its own layers apply, and only backends that forward that mask are accepted (`mla`/`tokenspeed_mla`/`gluon` for MLA drafts; `mha`/`fa3`/`fa4`/`triton`/`flashinfer`/`trtllm_mha` for GQA drafts). Same-checkpoint DSpark with Prefix Replay exists for DeepSeek V4/V4.1 (`--speculative-config '{"method":"dspark",...}'`, replay window from checkpoint config, fails closed if DSpark weights are missing and only MTP/NextN are present). Request temperature/top-k/top-p are applied by target verification, so the served distribution is the target's regardless of drafter greed.

**Takeaway for the DSpark-on-GLM-5.3-Flash plan:** the pattern to copy exists end to end in SGLang (per-model drafter class + KV injection + STS calibration), and TokenSpeed proves the checkpoint convention works across families; vLLM needs either `custom_class` or a new `glm5next` DSpark module. A drafter must reuse target hidden states (DSpark KV injection / EAGLE-style taps) and share the target's embedding + LM head (vocab 154,880).

## 4. What the z.ai blog says

Fetched full text from the blog's JS bundle (`https://z.ai/blog/assets/glm-5.3-flash--uSKeP6w.js`, dated 2026-08-26, "GLM-5.3-Flash: Frontier Intelligence, Flash Cost"). Findings:

- The blog does **not** mention MTP, speculative decoding, or acceptance rates anywhere. The MTP story is in the repo README ("improved MTP layer ... acceptance length by up to 20%") and the technical report (shared-parameter 3-layer MTP training; longer acceptance than DeepSeek-V3.2). This is a genuine negative finding for planning the narrative around the native drafter.
- Architecture emphasis: first GLM with hybrid sparse + linear attention ("linear attention captures local dependencies through state modeling, while sparse attention retrieves relevant global context through a lightweight indexer"); IndexPool compresses four indexer key vectors into one via weighted pooling to cut indexer latency/memory at 1M context; vs GLM-4.5 it nearly halves active params (18B vs 32B) and layers (45 vs 92); vs GLM-5.3 it uses 3.0x less attention compute and a 4.4x smaller KV cache, with the lowest attention compute among GLM-5.3 / DeepSeek-V4-Flash / Kimi-K3 (KV still slightly larger than those two).
- Serving story (on Chinese AI chips, custom engine built on SGLang): intra-node TP for linear attention + LM head, ReplaySSM, W8A8 quantization, hybrid INT8/FP8/BF16 cache quantization, Layer Split, and production-grade Encode-Prefill-Decode (EPD) disaggregation; 3x end-to-end improvement over their initial baseline on the same hardware. Framework support named: SGLang, vLLM, TokenSpeed.
- Misc: tested anonymously as `ox-alpha` on OpenCode/OpenRouter; ~$0.045/task on AA Intelligence Index v4.1.1; reasoning_effort low/high/max (default max).

## Memory math (8xH100 vs 8xH200 vs B200 node)

Anchor: FP8 weights ~306 GiB (vLLM recipe; KT tutorial: "reserve at least 350 GB of system memory"). Empirical KV pools from the vLLM recipe: TP4 on 4xMI355X (256 GiB each) -> 14.92M-token pool (FP8 checkpoint) / 8.87M-token pool (BF16 checkpoint). Bottom-up estimate: 11 DSA layers x (MLA compressed KV 512 elems + indexer pooled cache ~1024 elems) ~ 17 KB/token in FP8 KV, ~34 KB/token BF16 KV, plus amortized KDA recurrent state (64x128x128x2B = 2 MiB/layer x 34 layers ~ 68 MiB per request, ~4 KB/token-equivalent at 16K context) — call it ~20-25 KB/token FP8 KV and ~40-50 KB/token BF16 KV. Hopper caveat (vLLM recipe): this model does not support FP8 KV cache on Hopper — BF16 KV required; Blackwell supports FP8 KV.

Assumptions for the table: gpu_memory_utilization 0.90, ~30 GiB node reserve for CUDA graphs/activations/NCCL. Numbers are planning-grade estimates, not measurements.

| Node | Total VRAM | Weights (FP8) | KV budget | KV $/token | Token pool (est.) | Concurrent 16K-context calls (est.) |
|---|---|---|---|---|---|---|
| 8xH100 80GB (TP8) | 640 GiB | 306 GiB | ~275 GiB | ~45 KB (BF16 KV forced) | ~6M tokens | ~380 |
| 8xH200 141GB (TP8) | 1,128 GiB | 306 GiB | ~715 GiB | ~45 KB (BF16 KV forced) | ~16M tokens | ~1,000 |
| 8xB200 192GB (TP8) | 1,536 GiB | 306 GiB | ~1,080 GiB | ~22 KB (FP8 KV) | ~48M tokens | ~2,900 |

Cross-checks: the SGLang Hopper recipes run TP8 + EP8 with BF16 KV and mem-fraction 0.70 (H100) / 0.75 (H200); the BF16 checkpoint (~612 GiB weights) does not fit 8xH100 at all and is tight on H200; on Blackwell the NVFP4 variant (~half the weight bytes) plus FP8 KV gives the biggest headroom. The vLLM recipe's TP4-on-GB200-tray configuration and 4xGB300 SGLang validation both confirm a single 4-GPU Blackwell tray can host FP8 target + MTP.

For real-time phone-call agents the MTP draft layer adds ~8-9 GiB, and a DSpark-style drafter (six-layer scale, per the MiniMax-M3-DSpark precedent) would add roughly 1-3 GiB — immaterial on all three nodes.

## Engine-by-engine support matrix

| Engine | Target serving | Native MTP | External drafter paths | DSpark-class support | GLM-5.3-Flash specifics |
|---|---|---|---|---|---|
| SGLang | Yes (H100/H200/B200/B300/GB200/GB300 cells) | Yes, EAGLE-flagged 5/1/6 | EAGLE/EAGLE3, DFLASH, STANDALONE, UNO, NGRAM | Full in-tree implementation (kv_inject, planner, verify, STS fit, PD disagg); needs a new glm5_next drafter class | v0.5.20+; MTP+DP-attention unvalidated; DFlash2 checkpoint access-gated; DSA top-k backend flag for drafts |
| vLLM | Yes (v0.29.0+, Hopper/Blackwell/ROCm MI355X/Ascend 950PR) | Yes, `method: mtp`, 5 tokens | draft_model (TLI cross-vocab), eagle3, dflash, pard, mlp, custom_class (experimental), speculators kit | Per-family DSpark modules (DeepSeek-V4/V4.1, Kimi-K3, Gemma-4, Qwen3) + DSpark-only Adaptive Verification; none for glm5next | FlashInfer 0.6.17+; FP8 KV Blackwell-only; PD disagg w/ MTP both pools; ROCm MTP PR #55239 |
| TokenSpeed | Yes | Yes (2 steps / 3 draft) | EAGLE3, DFLASH, DSPARK external checkpoints, drafter attention/MoE backends | DSPARK algorithm + same-checkpoint DSpark w/ Prefix Replay; block-size conventions fail fast | `--draft-model-path-use-base`; auto KDA/DSA backends; TP8+EP |
| Transformers | Reference impl | MTP weights in checkpoint | n/a (not a server) | n/a | glm5_next docs |
| KTransformers | Yes (CPU-GPU hybrid, SM89/SM120) | No | No | No | 306 GiB FP8, 350 GB system RAM, 1M ctx, no speculation |

## Sources

- HF config (authoritative): <https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json>
- HF model card: <https://huggingface.co/zai-org/GLM-5.3-Flash>
- z.ai blog (text extracted from <https://z.ai/blog/assets/glm-5.3-flash--uSKeP6w.js>; page <https://z.ai/blog/glm-5.3-flash> is a JS-rendered SPA)
- SGLang cookbook: <https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash>
- SGLang speculative decoding docs: <https://docs.sglang.io/docs/advanced_features/speculative_decoding.md> and <https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding.md>
- vLLM recipe: <https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>
- vLLM speculative decoding docs: <https://docs.vllm.ai/en/latest/features/speculative_decoding/> (+ /mtp/ and /draft_model/)
- TokenSpeed model recipes and server parameters: <https://lightseek.org/tokenspeed/recipes/models> , <https://lightseek.org/tokenspeed/configuration/server>
- KTransformers tutorial: <https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md>
- vLLM GLM5Next MTP source: <https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/common/mtp.py> (SHA 8361a05)
- Engine source searches: GitHub code search on vllm-project/vllm (`dspark`, `path:glm5next eagle`, `Glm5Next num_nextn_predict_layers`) and sgl-project/sglang (`dspark`, `glm5_next mtp`)
- Local: `papers/GLM-5-technical-report-2602.15763.txt` (MTP lines ~204-215, 398-401, 758-763; Table 10 lines ~1876-1890), repo `README.md` (IndexShare + MTP +20% acceptance line 39; model table lines 79-80)

## Unknowns and confidence

- **High confidence**: all config.json facts (fetched raw JSON); vLLM/SGLang MTP flags (official recipes/docs); SGLang DSpark file inventory (GitHub code search); blog text (fetched bundle); ~306 GiB FP8 weights (two independent sources).
- **Medium-high**: SGLang accept-length numbers (cookbook-measured, GB300 campaign, exact context/effort pairing not fully extracted); engine version floors (vLLM 0.29.0+, SGLang v0.5.20+, FlashInfer 0.6.17/0.6.18).
- **Medium**: KV bytes/token estimates and the derived token pools / concurrency table (bottom-up estimate anchored on one empirical TP4 pool figure; utilization and reserve assumptions stated). Exact H100-vs-H200 cell pairing in the SGLang configurator (mem-fraction 0.70 vs 0.75 inferred from cell pairs).
- **Unresolved**: whether a public GLM-5.3-Flash EAGLE3/DSpark drafter checkpoint exists (none found; only the access-gated DFlash2); vLLM `custom_class` maturity for hybrid linear/sparse targets (experimental); whether TokenSpeed's DSPARK path accepts glm5_next-shaped draft checkpoints out of the box (its conventions are family-agnostic but validated models listed are MiniMax/Kimi/DeepSeek/LFM2); DFlash2 access-gate process. Resolving these needs either an HF access request or a follow-up ticket to smoke-test drafter loading in SGLang.
