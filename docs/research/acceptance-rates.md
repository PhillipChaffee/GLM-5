# Acceptance-rate research for GLM-5.3-Flash on real-time phone-call workloads (ticket #5)

## Answer in one paragraph

No published acceptance-length (tau) number exists for GLM-5.3-Flash itself. The closest family anchor is the GLM-5 technical report, which reports an MTP accept length of 2.76 (vs 2.55 for DeepSeek-V3.2) at 4 speculative steps on a private prompt set. Cross-domain data from DSpark and SPEED-Bench shows native-MTP tau of roughly 2.5-2.8 on mixed workloads, 1.9-2.1 on high-entropy conversational text at short draft lengths, and 4.9-6.2 on math and 4.5-5.6 on code for trained drafters at longer blocks. Phone-call chat is open-ended, high-entropy conversational text - the worst domain in every published table - and production voice stacks run with reasoning off, which removes the biggest tau and TTFT risk. A defensible planning range for GLM-5.3-Flash with its MTP-1 layer on phone-call chat is tau = 2.0-2.9 (point estimate 2.4-2.6); a DSpark-class parallel drafter would plausibly raise offline chat tau to 3.0-3.7, with production schedulers accepting an effective 2-4 tokens per round under load. The voice-agent SLA that matters is not throughput but end-to-end responsiveness: roughly 600 ms-1 s from end of user speech to first audio, of which the LLM TTFT budget is about 200-500 ms; sustained inter-token rate only needs to outrun speech (~5 tok/s) by a safe margin (10-30 tok/s/user).

## 1. Published acceptance rates for GLM-family MTP and comparable MoE targets

### GLM-family MTP

- GLM-5 technical report, Table 2: accept length 2.76 for GLM-5 vs 2.55 for DeepSeek-V3.2, "given the same number of speculative steps (4) in our private prompt set" (papers/GLM-5-technical-report-2602.15763.txt, lines 198-216). GLM-5 ships one MTP layer (Table 10, "# MTP Layers 1"), trained with parameter sharing across 3 MTP layers to cut the training-inference discrepancy that "reduces the acceptance rate of the second token."
- The report also notes MTP is "especially effective under the small-batch decoding regime typical in RL rollouts" and gives "disproportionately large benefits on the long tail" (lines 758-765) - directly relevant to a phone-call serving profile where per-request batch occupancy is small and tail latency is the SLA.
- No tau number is published for GLM-5.3-Flash; the model card and z.ai docs claim efficiency from architecture (hybrid sparse+linear attention, mHC, FP8) and FlashX is marketed at "200 tokens/s" inference, with no acceptance figures.

### Comparable MoE targets (SPEED-Bench, arXiv:2604.09557, Table 1: BS=32, draft length 3, production engines TensorRT-LLM/SGLang)

Acceptance length (AL, includes bonus token) by domain, native MTP columns:

| Model / drafter | Coding | Math | QA | RAG | Roleplay | Mean |
|---|---|---|---|---|---|---|
| GPT-OSS-120B MTP (temp 0) | 2.76 | 2.77 | 2.52 | 2.61 | 1.87 | 2.55 |
| Qwen3-Next MTP (temp 0) | 3.34 | 3.13 | 2.71 | 2.94 | 2.09 | 2.81 |
| Qwen3-Next MTP (temp 1) | - | - | - | - | - | 2.68 |

Domain finding (paper Section 8.1): "low-entropy domains, such as Coding and Math, yield larger ALs than high-entropy tasks like Roleplay." Roleplay - the closest published proxy to phone-call chat - is the worst domain everywhere. Temperature matters: at temp 1, mean AL drops ~5% for Qwen3-Next MTP (2.81 -> 2.68) and ~10-20% for post-trained drafters (EAGLE3 GPT-OSS 2.25 -> 2.07). The paper also confirms MoE targets are well suited to SD because they stay memory-bound even at high batch.

### DSpark (arXiv:2607.05147, Table 1: offline, fixed block, temp 1.0, non-thinking mode, tau includes bonus token)

Domain averages across benchmarks (Qwen3 4B/8B/14B and Gemma4-12B targets, trained DSpark drafter):

| Domain | tau range |
|---|---|
| Math (GSM8K/MATH500/AIME25) | 4.89-6.21 |
| Code (MBPP/HumanEval/LCB) | 4.51-5.64 |
| Chat (MT-Bench/Alpaca/Arena-Hard) | 2.92-3.72 |

For reference, Eagle3 (autoregressive) chat tau is 2.26-2.72 on the same targets. Figure 2 gives position-wise conditional acceptance for Qwen3-4B: position 1 on chat is 0.72 (parallel drafter) vs 0.53 (Eagle3); parallel drafting decays across positions (chat 0.72 -> 0.63, code 0.87 -> 0.78) while autoregressive rises (chat 0.53 -> 0.74). Figure 5 (threshold sweep): at zero threshold (fixed-length verification) the overall acceptance rate is 45.7% on chat, 76.9% math, 67.6% code - chat wastes about half of all verified draft tokens without confidence pruning. Production (Section 5): DSpark-5 vs MTP-1 on DeepSeek-V4 serving; the scheduler allocates 4-6 verification tokens per request under moderate concurrency (<200 requests) and shrinks it under load (Figure 8); at matched throughput DSpark raises per-user speed 60-85% (V4-Flash) and 57-78% (V4-Pro).

## 2. Thinking mode / reasoning_effort interaction with acceptance

What the model exposes (z.ai docs and model card):

- GLM-5.3-Flash thinking is forced: `thinking.type` only supports `enabled` and "cannot be disabled" (docs.z.ai/guides/capabilities/thinking-mode; docs.z.ai/guides/llm/glm-5.3-flash).
- `reasoning_effort` accepts low/high/max and defaults to max; pass low explicitly to cut the thinking budget.
- `clear_thinking` defaults to false; the model card says: "For chat scenarios, explicitly pass clear_thinking=true."

What this does to tau and latency:

1. Published caveat: DSpark's offline evaluation explicitly "adopt[s] the non-thinking mode" (arXiv:2607.05147, line 470). All chat tau numbers above are non-thinking. There is no published tau for GLM-family thinking traces.
2. Reasoning text plausibly accepts worse than chat: thinking traces make genuine decisions (branch selection, revising, computing) that carry target-conditional entropy the drafter must guess; exploratory tokens are exactly the high-entropy tokens that fail prefix verification. Adjacent evidence is weak but not encouraging: SPEED-Bench's reasoning model (DeepSeek R1) shows mean ALs at or slightly below non-reasoning peers, and DSpark's Figure 5 shows chat already wastes ~54% of verified tokens at fixed-length verification.
3. TTFT is the sharper effect: thinking tokens are generated before the first visible token. At reasoning_effort max, hundreds of thinking tokens sit between end-of-user-speech and first audible word - fatal for a phone-call budget regardless of tau. At low effort the thinking span is short; with `clear_thinking=true` the reasoning_content is stripped from returned context, so the drafter never conditions on thinking-style history and the conversational distribution stays close to the drafter's chat training data. (Note: `clear_thinking=false` "Preserved Thinking" returns reasoning_content and requires echoing it back exactly - that shifts context distribution toward reasoning text and would stress any drafter trained on chat.)
4. Practical conclusion for phone calls: `reasoning_effort: low` + `clear_thinking: true` (or equivalent) is the right serving configuration. Under it, the acceptance profile of the visible response stream should be close to the non-thinking chat numbers in Section 1; the residual risk is the short low-effort thinking prefix, whose tau is unmeasured. Vapi's methodology page independently confirms the industry norm: "Voice agents run LLMs with reasoning turned off to keep latency low."

## 3. FP8 target vs drafter precision effects on acceptance

- GLM-5.3-Flash ships FP8 weights (model card, F8_E4M3 tensors). GLM-5 uses "FP8 for rollout inference to reduce per-token latency" (GLM-5 TR lines 758-765). Z.ai's production stack adds W8A8 compute and hybrid INT8/FP8/BF16 cache quantization on their chips (z.ai blog/docs). DeepSeek-V4.1-Flash serves DSpark over FP8 embeddings/KV projections and even FP4 main KV cache (papers/DeepSeek-V4.1-Flash-2609.19969.txt, lines 605-607, 629-649).
- Correctness is not at stake: speculative decoding is lossless relative to the target as actually served - acceptance is computed against the FP8 target's own logits, not a BF16 reference. Every cited production system (DeepSeek-V3.2/V4/V4.1-Flash, GLM-5 rollouts) runs FP8 targets with speculation and publishes healthy tau (2.5+), so the operating point is proven.
- Effect on tau is a distribution-shift question: an FP8 target's argmax/greedy paths differ slightly from BF16, so a drafter distilled online from the FP8 target matches better than one trained against BF16 traces. If the drafter's own GEMMs or the target hidden states injected into it (KV injection) are quantized more aggressively than the target's verification path, draft quality degrades slightly. No publication quantifies this delta for GLM or DeepSeek targets; direction is small and second-order relative to the domain effect (chat vs code spans ~2 tau units; quantization is plausibly <0.1). Treat FP8 as a noise term in the cost model, not a driver. Confidence: low - no direct measurement.

## 4. Per-user tok/s and latency SLA for real-time voice agents

Vendor-published numbers (all vendor docs/blogs - flagged as such; no peer-reviewed SLA standard exists):

| Source | Number |
|---|---|
| Retell docs (check-estimated-latency) | "latency as low as 600ms" from user stops speaking to agent begins responding; default "Response Wait time" 0.4 s; accuracy-mode transcription adds ~200 ms |
| Vapi docs (understanding-latency, metrics-methodology) | Five sequential stages: transport, endpointing, transcription, model TTFT, TTS first audio; "Model time to first token is often the largest component"; voice agents run reasoning off |
| LiveKit docs (turn-taking tuning) | endpointing.min_delay default 0.5 s (adds to pre-LLM latency); preemptive generation offered to cut TTFT; default recommended LLM is a latency-optimized small model (Gemma 4 31B), not a frontier MoE |
| Pipecat docs (metrics) | Tracks TTFB, TTFA (first audio), TTFAT (first answer token); example values: LLM TTFB 0.84 s, TTS TTFB 0.17 s, text aggregation 0.21 s |
| DSpark production telemetry (arXiv:2607.05147 Sec. 5) | Interactivity SLA anchors used in production: 80-120 tok/s/user (V4-Flash), 35-50 tok/s/user (V4-Pro) |

Derived LLM-side budget for an STT -> LLM -> TTS phone agent: if total voice-to-voice is 600 ms-1 s, and endpointing (0.2-0.5 s) + final STT (~0.1-0.3 s) + TTS first audio (~0.1-0.2 s) consume the front and back, the LLM TTFT budget is roughly 200-500 ms. Per-call speech is ~150 wpm, about 5 tok/s (1.3 tok/s per spoken word x ~4), so sustained generation only needs to outrun speech: 10-30 tok/s/user avoids TTS starvation with margin; production voice SLAs like Retell/Vapi are achieved at far below DSpark's 35-120 tok/s/user anchors. The binding constraint is TTFT and early streaming, not steady-state tok/s. Two implications: (a) reasoning_effort must not add a thinking wall in front of TTFT; (b) at high concurrency the scheduler trades verification length for batch capacity (DSpark Figure 8 shrinks budgets under load), which directly trades away tok/s/user - so the cost model should check tok/s/user at the target concurrency, not only at light load.

## Tau table by domain and source

| Source (settings) | Target / drafter | Chat / conversational | Code | Math | Mixed |
|---|---|---|---|---|---|
| GLM-5 TR Table 2 (4 spec steps, private set) | GLM-5, native MTP | n/a | n/a | n/a | 2.76 |
| GLM-5 TR Table 2 (4 spec steps, private set) | DeepSeek-V3.2, native MTP | n/a | n/a | n/a | 2.55 |
| SPEED-Bench T1 (BS=32, DL=3, temp 0) | GPT-OSS-120B MTP | 1.87 (Roleplay) | 2.76 | 2.77 | 2.55 |
| SPEED-Bench T1 (BS=32, DL=3, temp 0) | Qwen3-Next MTP | 2.09 (Roleplay) | 3.34 | 3.13 | 2.81 |
| SPEED-Bench T1 (temp 1) | Qwen3-Next MTP | - | - | - | 2.68 |
| DSpark Table 1 (gamma=7-ish, temp 1.0, non-thinking) | Qwen3 4-14B, DSpark | 3.13-3.72 | 4.51-5.64 | 4.89-6.21 | - |
| DSpark Table 1 | same targets, Eagle3 | 2.26-2.72 | 3.69-5.37 | 3.71-5.87 | - |

## Defensible tau range for phone-call chat with reasoning

- Baseline (GLM-5.3-Flash MTP-1, reasoning_effort=low, clear_thinking=true): tau = 2.0-2.9, point estimate 2.4-2.6. Anchors: GLM-5's own 2.76 on a private mixed set; SPEED-Bench native MTP 2.55-2.81 mixed at DL=3 and 1.87-2.09 on the most phone-like (Roleplay) domain capped by the short draft length; DSpark/Eagle3 chat 2.3-3.7 at long blocks. Phone-call chat is high-entropy conversational text with noisy ASR context, so shade toward the bottom of chat benchmarks; GLM-5's parameter-shared 3-layer MTP training is claimed to specifically lift second-token acceptance.
- Stretch (a DSpark-class parallel drafter trained for GLM-5.3-Flash): offline chat tau 3.0-3.7 (DSpark chat benchmark range); in production with confidence-scheduled verification at hundreds of concurrent calls, effective accepted tokens per round compress toward 2-4 as budgets shrink under load. Speedup vs the MTP-1 baseline at matched throughput: 1.6-1.85x per-user (DSpark production: 60-85%).
- Cost-model input: use tau = 2.5 (MTP-1 baseline) and 3.0-3.5 (DSpark-class) with a sensitivity band 2.0-3.7; do not model thinking-heavy traffic (max effort) with any of these numbers - that regime is unmeasured and dominated by pre-TTFT thinking tokens.

## Sources

Local papers (full text in papers/):

- DSpark, arXiv:2607.05147 - Table 1 tau by domain; Figure 2 position-wise conditional acceptance; Figure 5 threshold sweep; Section 5 production results and Figure 8 verification budgets. <https://arxiv.org/abs/2607.05147>
- GLM-5 technical report, arXiv:2602.15763 - Table 2 accept lengths; MTP parameter sharing; FP8 rollouts and MTP tail-latency benefits; Table 10 (# MTP Layers 1). <https://arxiv.org/abs/2602.15763>
- DeepSeek-V4.1-Flash, arXiv:2609.19969 - DSpark integration, FP8/FP4 cache serving. <https://arxiv.org/abs/2609.19969>
- DeepSeek-V4, arXiv:2606.19348 - MTP configuration unchanged from DeepSeek-V3 (no new acceptance numbers; HTML full text checked). <https://arxiv.org/abs/2606.19348>
- SPEED-Bench, arXiv:2604.09557 - Table 1 ALs and speedups across domains at BS=32/DL=3, temp 0 and 1; domain-entropy and MoE findings. <https://arxiv.org/abs/2604.09557>

Web (vendor docs; flagged as vendor sources, not peer-reviewed):

- GLM-5.3-Flash model card: <https://huggingface.co/zai-org/GLM-5.3-Flash> (reasoning_effort low/high/max default max; clear_thinking for chat; FP8 weights)
- z.ai GLM-5.3-Flash docs (blog content mirrored here; the blog itself is a JS app that fetch returns empty): <https://docs.z.ai/guides/llm/glm-5.3-flash> (FlashX 200 tok/s claim; thinking.type enabled-only)
- z.ai thinking mode: <https://docs.z.ai/guides/capabilities/thinking-mode> (forced thinking; clear_thinking semantics)
- Retell: <https://docs.retellai.com/reliability/check-estimated-latency.md> (600 ms floor; 0.4 s response wait) and <https://docs.retellai.com/build/transcription-mode.md> (+200 ms accuracy mode)
- Vapi: <https://docs.vapi.ai/assistants/model-intelligence/understanding-latency.md> and .../metrics-methodology.md (five-stage model; TTFT largest; reasoning off)
- LiveKit: <https://docs.livekit.io/agents/logic/turns/tuning.md> (endpointing defaults; preemptive generation) and <https://docs.livekit.io/agents/models/llm/livekit.md> (latency-optimized default LLM)
- Pipecat: <https://docs.pipecat.ai/pipecat/fundamentals/metrics.md> (TTFB/TTFA/TTFAT definitions and example values)

## Unknowns and confidence

1. No published tau for GLM-5.3-Flash or its MTP layer under any workload - the whole range is an inference from family and cross-model anchors. Confidence: medium-low for the 2.4-2.6 point estimate, medium for the 2.0-2.9 band.
2. GLM-5's 2.76 comes from a private prompt set with unknown domain mix and unknown thinking settings. Confidence: medium that it transfers to phone-chat serving.
3. Tau on thinking traces (reasoning_effort high/max): unmeasured anywhere for GLM-family models; direction (lower than chat) is an inference from entropy arguments and DSpark's non-thinking-only evaluation. Confidence: low. Mitigation is operational: serve low effort + clear_thinking for calls.
4. FP8/quantization effect on tau: no direct publication found; expected sub-0.1 and second-order. Confidence: low; treat as sensitivity, not a driver.
5. Voice-agent SLA numbers are vendor docs/blogs (Retell 600 ms, LiveKit endpointing defaults, Vapi component medians), not independent measurements; the 200-500 ms TTFT budget is derived, not published. Confidence: medium for magnitudes, low for tails; validate against our own STT/TTS picks in the cost model.
6. Production effective accepted-length under our target concurrency (hundreds-thousands of calls) depends on DSpark-style load-adaptive scheduling, which is described but not tabulated per concurrency in the paper (Figure 8 is graphical). Confidence: medium for the 2-4 tokens/round compression under load.
