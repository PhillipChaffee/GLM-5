# How production serves GLM-5.3-Flash: concurrency bounds, deployment recipes, and throughput anchors

Research supporting the regen-concurrency-ramp and cost-optimization work under [Train the GLM-5.3-Flash drafter via DeepSpec on Modal, end to end](https://github.com/PhillipChaffee/GLM-5/issues/23) (map #1). Compiled 2026-10-12 by web research (z.ai docs, SGLang cookbook, vLLM recipe, engine issue trackers, OpenRouter telemetry). Motivation: our regen smoke measured ~258 tok/s/node aggregate at concurrency 64, ~10x below every published anchor, and the plan's 5–10k tok/s/node assumption needed an external check.

## Answer in one paragraph

**Nobody publishes a concurrency cap for GLM-5.3-Flash; production bounds it with engine-side memory budgets, and the documented OOM failures are transient activation blowups — the same failure class as our cache stage, not KV exhaustion.** z.ai's rate limits live behind a per-key dashboard and their serving story (SGLang-derived engine, tens of thousands of domestic accelerators, encode–prefill–decode disaggregation, hybrid cache quantization) publishes no throughput or concurrency numbers. The concrete public recipes are the SGLang cookbook (4xGB300: `--mem-fraction-static 0.90 --max-running-requests 80 --chunked-prefill-size 8192`, FP8 KV) and the vLLM recipe (TP4 + FP8 KV + MTP on one GB200 tray; PD-disagg decode pool `--max-num-seqs 512`). Published throughput anchors: ~2,660 output tok/s per node at c64 spec-off (4xGB300), ~4,828 at c256, ~8,976 tok/s with MTP at c80 (accept 4.7). All are >=10x our smoke's 258 tok/s/node — the gap is not explained by any published config difference and the smoke's own token accounting must be audited before the cost model re-anchors.

## 1. Operator practice (z.ai and marketplaces)

- **No public rate-limit or max-concurrency figure exists.** The docs "Rate Limits" page redirects to the authenticated dashboard (`<https://z.ai/manage-apikey/rate-limits`;> redirect verified); the FAQ confirms limits live there. OpenRouter lists 30+ providers serving the model (FP4/FP8/NVFP4, 1M context) with `throughput_last_30m` telemetry null for all of them — no provider publishes concurrency metadata.
- **z.ai's production stack** (their docs): a dedicated inference engine built on SGLang over "tens of thousands of domestically developed accelerators"; encode–prefill–decode (EPD) disaggregation with independently scheduled worker pools; intra-node TP for linear attention and the LM head; W8A8 quantization; hybrid INT8/FP8/BF16 cache quantization; "3x improvement in end-to-end serving performance" vs their initial baseline — relative only, no absolute tok/s.
- The GLM-5 technical report (local `papers/GLM-5-technical-report-2602.15763.txt`, lines 753–780) documents the flagship's RL-rollout serving pattern: EP64/DP64 over 8 nodes for KV capacity, FP8 rollouts + MTP for tail latency, prefill–decode disaggregation.

## 2. Deployment recipes (primary sources)

- **SGLang cookbook GLM-5.3-Flash cells** (<https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash>):
  - 4xGB300 (low-latency): TP4 + FP8 KV (`fp8_e4m3`) + TRT-LLM DSA + `flashinfer_trtllm` MoE (+ optional EAGLE MTP 5/1/6); perf-campaign server flags `--mem-fraction-static 0.90 --max-running-requests 80 --cuda-graph-max-bs-decode 80 --context-length 16384`.
  - 8xH100: TP8/EP8, `--mem-fraction-static 0.70`; 8xH200: 0.75; both BF16 KV + TileLang DSA + deep_gemm. Hopper cannot do FP8 KV for this model in either engine.
  - 4xMI355X: TP4/EP4, `--mem-fraction-static 0.82 --max-running-requests 96 --chunked-prefill-size 8192`.
  - Multimodal EPD language process: `--mem-fraction-static 0.78 --max-running-requests 64 --chunked-prefill-size 8192`.
  - Structural caveats from the cookbook's own disable reasons: MTP with DP-Attention "not validated on this model"; DFLASH "does not support DP-Attention".
- **vLLM recipe** (<https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>, updated 2026-10-08): FP8 TP4 + MTP (`num_speculative_tokens: 5`) + FP8 KV on one GB200 tray; PD-disagg TP4 prefill + TP4 decode with decode pool `--max-num-seqs 512`; env pins `VLLM_SSM_CONV_STATE_LAYOUT=DS` / `VLLM_KV_CACHE_LAYOUT=HND` for KDA conv-state layout parity. FP8 KV is supported on Blackwell only (BF16 KV required on Hopper). MI355X TP4 reports a 14.92M-token KV pool and ~113.8x max concurrency at 128K context.
- **How the bounds work**: `--mem-fraction-static` sizes the static KV pool as a fraction of post-weights memory; the scheduler admits a request only when its worst-case KV footprint fits the free pool (queueing the rest); `--max-running-requests` caps admission explicitly; chunked prefill bounds per-step activation spikes. Serving never "rate-limits at the API" to avoid OOM — memory-budgeted admission control does it.

## 3. Throughput anchors (aggregate output tok/s per node)

- **SGLang, 4xGB300 TP4/EP4, spec decoding off** (cookbook benchmark block, 2026-09 perf campaign, 1,024 in/256 out, temp 0, prefix cache off): **1,161 / 2,660 / 4,828 tok/s at concurrency 16 / 64 / 256** (sublinear at 256 from prefill queueing). FP8-KV/TRT-LLM variant: 1,227 / 2,739 / 4,977. The FP8 pool holds "12.6M tokens per rank vs 7.0M at BF16".
- **SGLang, 4xGB300, MTP 5/1/6**: chat 1k/1k = 454 / 2,986 / **8,976 tok/s at c1/16/80** (accept 5.98/5.98/4.7); summarization 8k/1k = 458 / 2,467 / 3,803 (accept 6.0/6.0/4.5).
- **vLLM, 4xGB200 TP4, no MTP** (issue #56975): 1,892 / 2,607 / 3,377 tok/s at c32/64/128; p99 ITL 199/337/342 ms; SGLang on the same tray has 13/16/20 ms p99 ITL at 1.2–4.5x higher TTFT.
- **Hopper floor** (8,192 in/1,024 out, simulated accept 3): 8xH100 213 c1 / 1,176 c16; 8xB200 244 c1 / 1,811 c16; 8xB300 278 c1 / 2,079 c16.
- Per-user context from DSpark production serving (local [acceptance-rates.md](acceptance-rates.md)): 80–120 tok/s/user SLAs with verification budgets shrinking 4–6 → 2 under load.

## 4. Documented OOM incidents and memory tuning

- **SGLang #37712** (4xB300): a single **73.65 GiB transient allocation** in the DSA indexer's `fp8_mqa_logits` when several 300K–1M-token prefills ran concurrently with prefix-cache hits — KV pool only ~35% used. Transient activation blowup, not KV exhaustion; fixed by bounding (#38469) or chunking (#38507) the allocation. This is the serving-side twin of our cache-stage OOM.
- **vLLM #56975**: CUDA-graph capture OOM when `--max-cudagraph-capture-size` = chunk size = 8192 ("ran out of memory while capturing 531 graph sizes"); 4096 fits and cuts p99 ITL 334 → 117 ms at c64 for ~-7% throughput — a config cliff, not a scheduler bug.
- **SGLang #36830**: SGLang cannot use FP8 KV on CUDA for this model (no bf16-query x fp8-KV DSA backend) → BF16 KV doubles KV bytes in SGLang; vLLM does support FP8 KV. The two engines differ in KV capacity on identical hardware.
- Adjacent-family allocator OOMs worth watching for our spec-dec deployment: SGLang #43486 (DeepSeek-V4.1-Flash + DSPARK: caching allocator exhausts GPU memory under sustained max-batch decode while KV is 29% used), vLLM #47127 (GLM-5.2 profile_run OOM with PD disagg + FlashInfer CUTLASS MoE).

## 5. The 258 tok/s/node contradiction

Our regen smoke (c64, 8-minute client window) implied **1,033 tok/s aggregate / ~258 tok/s/node** vs published c64 anchors of 2,607–2,739 on 4-GPU GB200/GB300 trays — **9.6–10.6x below**; even 8xH100 at c16 is 4.6x above. Candidate explanations (hypotheses, unverified):

1. **Token accounting** — the cookbook notes retokenized counts can be ~1/4 of generated tokens when `reasoning_content` is not retokenized; if the regen path counted only visible tokens, expected deflation ~4x (258 x 4 ≈ 1,032, still ~2.5x low).
2. **DSA backend fallback** — SGLang #42012/#40286/#37105 show the DSA backends are fragile per SM arch (trtllm SM100-only, tilelang smem-limited, triton slow).
3. **Small effective decode batch** — `--max-running-requests` or a KV pool smaller than nominal c64 (low mem-fraction, or the 306 GiB weights squeezing the pool), or chunked prefill absorbing the concurrency.
4. **Missing CUDA graphs / eager decode.**

Nothing published explains 258 at c64 on a healthy Blackwell 4-GPU launch; the smoke config itself needs auditing (which regen flags, what was counted) before the 5–10k tok/s/node planning assumption is kept, re-anchored to ~2.5–5k spec-off, or replaced.

## Not found / no public info

- Any documented max-concurrent-requests or per-deployment concurrency cap for GLM-5.3-Flash (z.ai docs, OpenRouter, GitHub).
- Any absolute tok/s for z.ai's domestic-chip cluster (only the relative "3x").
- Any SGLang 4xB200-class FP8 recipe or benchmark (closest: 4xGB300 TP4, 8xB300 TP8, vLLM 4xGB200 TP4).
- `--mem-fraction-static` / chunked-prefill guidance in the vLLM recipe (it relies on defaults).
- The prior local claim that "disabling speculation at heavy batching is more efficient" ([glm-flash-engines.md](glm-flash-engines.md)) was **not** re-verified on today's cookbook page; treat as unconfirmed.

## Sources

- <https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash> (cells, benchmarks, EPD recipe)
- <https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>
- <https://docs.z.ai/guides/vlm/glm-5.3-flash.md>
- <https://docs.z.ai/help/faq.md> and <https://docs.z.ai/api-reference/rate-limit.md> (redirect to the per-key dashboard)
- <https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints>
- <https://github.com/vllm-project/vllm/issues/56975>
- <https://github.com/sgl-project/sglang/issues/37712> (refs #38469, #38507), #36830, #42012, #40286, #43486
- <https://github.com/vllm-project/vllm/issues/47127>
- <https://huggingface.co/zai-org/GLM-5.3-Flash>
- Local: `papers/GLM-5-technical-report-2602.15763.txt`; [acceptance-rates.md](acceptance-rates.md); [glm-flash-engines.md](glm-flash-engines.md); [modal-cost.md](modal-cost.md)
