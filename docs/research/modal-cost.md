# Modal-side cost model for GLM-5.3-Flash serving research (ticket #4)

## Answer in one paragraph

Modal bills per second with no minimums: B300 $7.10/GPU-hr, B200 $6.25, H200 $4.54, H100 $3.95, up to 8 GPUs per container on one machine; function executions cap at 24 h, long-lived serving is done with deployed apps where `min_containers` containers stay warm and are billed 24/7 while idle, and GPU Memory Snapshots (alpha) do not help multi-GPU serving nodes, so a GLM-5.3-Flash node cannot cold-start into a phone-call SLA and any production-style deployment burns roughly $18k/month for a single warm 4xB200 node. The four experimental phases are cheap by contrast: proxy drafter training on Qwen3-4B ($150-400), response-regeneration serving ($500-1,300 for ~0.3-1B output tokens), drafter training with teacher hidden states ($750-2,900, or ~$100 to evaluate the already-released access-gated DFlash2 drafter instead of training), and concurrent benchmark campaigns ($1,300-2,900). Total: roughly $3k-8k, so $15k funds the whole research program with headroom. At a 50 tok/s/user phone SLA a 4xB300-class node holds ~112 concurrent conversations at measured acceptance length 4.7 (~28 conv/GPU, ~$0.25/conversation-hour), scaling ~linearly with tau: ~48 conv/node at tau=2, ~229 at tau=6. The budget dies first on always-warm serving, not on experiments; the cheapest de-risking order is (a) proxy pipeline, then a zero-training DFlash2 acceptance test on 4xB200, then data regeneration with thinking disabled, then drafter training only if DFlash2 disappoints, then benchmarks.

## 1. GPU catalog, pricing, and platform limits

Per-GPU pricing from <https://modal.com/pricing> (per-second rates converted to hourly):

| GPU | $/GPU-hr | Notes |
|---|---|---|
| B300 | $7.10 | Blackwell Ultra; needs CUDA 13.1+; closest Modal GPU to the SGLang GB300 benchmark hardware |
| B200 | $6.25 | `B200+` opt-in runs B200 or B300, billed as B200 |
| H200 SXM | $4.54 | 141 GB HBM3e |
| H100 SXM5 | $3.95 | Modal may silently upgrade H100 requests to H200 at no cost change; pass `H100!` to pin for benchmarking |
| A100-80GB | $2.50 | |
| L40S | $1.95 | adequate for drafter-only serving (~1-4B params) |
| T4/L4/A10 | $0.59-1.10 | irrelevant here except A10 caps at 4 GPUs/container |

CPU: $0.047/core-hr; memory $0.008/GiB-hr; Volumes $0.09/GiB-month with 1 TiB/month free (the 306 GiB FP8 weight set fits in the free tier). <https://modal.com/docs/guide/gpu>

Practical limits (docs: gpu, timeouts, scale, cold-start, memory-snapshots):

- Max 8 GPUs per container, always one physical machine, for B300/B200/H200/H100/A100/L40S/L4/T4; A10 max 4. Requests above 2 GPUs/container see longer allocation waits. Multi-node training is private beta (email <support@modal.com>) - all planned phases must (and can) fit one 8-GPU node.
- Function execution timeout: default 300 s, configurable 1 s to 24 h (max). `startup_timeout` separately bounds container boot/`enter`. A >24 h training run must checkpoint and resume; 12-48 node-hour drafter training fits in 1-2 executions.
- Long-lived serving: deploy an app with web endpoints; containers persist while traffic or `min_containers` demands. Keep-warm knobs: `min_containers` (floor, billed 24/7), `buffer_containers` (burst buffer while active), `scaledown_window` (2 s-20 min idle grace, default max 60 s), `max_containers`. The cold-start doc states idle warm containers are billed ("you will be billed for any resources used while the container is idle (e.g., GPU reservation ...)").
- Snapshots: CPU Memory Snapshots are mature; GPU Memory Snapshots are alpha, "generally incompatible with multi-GPU code", and "do not speed up model loading from storage". For a 306 GiB-weight engine this means pre-stage weights in a Volume and rely on warm pools - snapshots will not rescue scale-to-zero phone serving.
- Billing: per-second, no minimum increments, monthly invoices, workspace/environment spend budgets, tagged billing reports (docs/guide/billing).
- Plan caps that bind: Starter = 10 GPU concurrency (too small for 8-GPU serving plus anything else); Team $250/month = 50 GPU concurrency, 5,000 containers; Enterprise = custom, volume discounts. Non-preemptible execution 3x; region selection 1.15-1.75x. Hard limit 4,000 concurrent containers per Function. Whether the $250/month Team fee itself draws against credits is unconfirmed (see unknowns).

## 2. Phase-to-dollar mapping

Model facts used throughout (vLLM recipe + HF card): GLM-5.3-Flash FP8 checkpoint is ~306 GiB weights, minimum 386 GB VRAM (NVFP4 variant: 229 GB, Blackwell only; BF16: 772 GB). Verified serving paths on Modal hardware: SGLang 4xB200 (Low Latency / High Throughput TP4, tested in-repo), SGLang 8xH200 (TP8, tested in-repo), SGLang 8xH100 (TP8/EP8, measured in the 2026-09 campaign), vLLM H100 and B200 verified. FP8 KV cache is Blackwell-only; Hopper serves BF16 KV.

| Phase | Work | Modal hardware | Volume assumption | GPU-hours | Cost |
|---|---|---|---|---|---|
| (a) Proxy drafter training | Qwen3-4B target: hidden-state extraction (~100-300M tok), drafter train, STS/confidence calibration, ~3 debug iterations | 1x H100 | ~40-100 GPU-h | 40-100 | $160-400 |
| (b) Response-regeneration serving | Serve GLM-5.3-Flash, spec off, thinking off, batched generation | 8x H100 ($31.59/hr) or 4x B200 ($25.00/hr) | 0.3-1B output tokens at ~5-10k tok/s/node | 20-40 node-h | $500-1,300 |
| (c) Drafter training | Teacher hidden-state capture (same corpus) + drafter training on 8-GPU node + post-hoc calibration | 4x B200 teacher, 8x H100 trainer | 0.3-1B tok teacher pass; 12-48 node-h training | 30-90 node-h | $750-2,900 (or ~$100 to evaluate released DFlash2 instead) |
| (d) Concurrent serving benchmarks | Acceptance-vs-concurrency curves, SLA frontier, soak tests | 4x B200 + 8x H200 | ~50-90 node-h | 50-90 | $1,300-2,900 |
| **Total** | | | | | **~$2,700-7,600** |

Common costs are minor: the 306 GiB weights Volume is inside the 1 TiB/month free tier; image builds and CPU orchestration are noise at these GPU rates.

## 3. Conversations per GPU and $/conversation, tau in [2, 6]

Explicit SLA and model assumptions:

- Phone SLA: sustained per-user decode >= 50 tok/s/user. Human speech is ~150-180 wpm ~ 4-5 tokens/s, so 50 tok/s gives ~10x headroom for streaming TTS and barge-in. Sensitivity shown at 80 and 120 tok/s/user, the DSpark V4-Flash SLA anchors (papers/DSpark-2607.05147.txt, Sec 5.4). TTFT target <= 500 ms on short phone prompts (the measured H100 TTFT of 2,985 ms corresponds to 8,192-token inputs, not phone-length).
- Round-rate model: a verification round costs ~one target forward pass regardless of draft length, so aggregate tok/s ~ tau at fixed batch size. DSpark measured the sequential-head overhead at 0.2-1.3% of round latency. I linearly rescale measured throughput by tau.
- Per-user speed at concurrency C approximates aggregate/round-rate split: at GB300 c80, 8,976/80 = 112 tok/s/user >= 80 SLA, consistent with the DSpark regime where effective batch sits below compute saturation and per-user speed and aggregate throughput correlate.
- Thinking disabled (`reasoning_effort: low` or `thinking: false`) for phone serving; the default `max` reasoning mode multiplies token consumption several-fold and would inflate both phase (b) and serving cost.
- Capacity pricing: a conversation holds a slot for its duration regardless of speech duty cycle, so $/conversation is capacity-based, not token-based.

Anchor measurements (SGLang GLM-5.3-Flash cookbook, 2026-09 perf campaign, chat 1k-in/1k-out):

- 4x GB300 (TP4/EP1, FP8 KV, TRT-LLM DSA, fixed MTP 5/1/6, natural acceptance): 454 tok/s at c1, 2,986 at c16, 8,976 at c80, acceptance length 5.98 / 5.98 / 4.7. GB300 is not sold on Modal; B300 is the same Blackwell Ultra class and the closest proxy (flagged, ±30-40%).
- 8x H100 (TP8/EP8, BF16 KV, TileLang DSA, simulated accept length 3): 212.6 tok/s at c1, 1,176.4 at c16 (1,206 with HiCache L1+L2, TPOT 10.35 ms). Simulated acceptance makes these mechanism numbers, not quality-verified acceptance.

Throughput sensitivity at c80 on the B300-class node (rescaling 8,976 by tau, tau_meas = 4.7):

| tau | Aggregate tok/s (4-GPU node) | Conv/node @ SLA 50 | Conv/GPU @ 50 | Conv/GPU @ 80 | Conv/GPU @ 120 |
|---|---|---|---|---|---|
| 2 (MTP-1-like) | 3,820 | 76 | 19.1 | 11.9 | 8.0 |
| 4.7 (measured) | 8,976 | 180 | 44.8 | 28.1 | 18.7 |
| 6 | 11,460 | 229 | 57.3 | 35.9 | 23.9 |

$/conversation-hour = node hourly cost / conversations per node (4x B300 node, $28.40/hr):

| tau | SLA 50 | SLA 80 | SLA 120 |
|---|---|---|---|
| 2 | $0.37 | $0.60 | $0.89 |
| 4.7 | $0.16 | $0.25 | $0.38 |
| 6 | $0.12 | $0.20 | $0.30 |

Hopper anchor on Modal-native 8x H100 ($31.59/hr), measured c16 point only: 1,176 tok/s aggregate gives 23.5 conv/node at SLA 50 (2.9 conv/GPU, $1.34/conv-hr), 14.7 at SLA 80 (1.8/GPU, $2.15), 9.8 at SLA 120 (1.2/GPU, $3.22). If the H100 concurrency curve scales like the GB300 one (~5x from c16 to c80), these drop ~5x to $0.3-0.6/conv-hr, but that is extrapolation - the campaign did not publish H100 c80 numbers.

Reading the tau sensitivity honestly: conversations/GPU scales ~3x from tau=2 to tau=6, but DSpark's own scheduler data shows natural acceptance falling from 6.0 to 4.5-4.7 as concurrency saturates (the dynamic scheduler shrinks verification budgets from 2 static MTP-1 tokens to 4-6 at moderate load, then prunes under heavy load). Plan on tau ~ 4-5 under load, 6 at light load. The decision-relevant quantity is sustained acceptance at target concurrency, which is exactly what phase (d) must measure - acceptance-vs-concurrency, not just peak tok/s.

## 4. Where the $15k runs out first, and cheapest de-risking order

The budget dies on always-warm serving, not on experiments:

1. **Warm production serving.** One 4x B200 node held warm with `min_containers=1` costs $25/hr x 730 h = $18,250/month - over the entire $15k by itself, in ~25 days. 8x H200 warm = $26,500/month (~17 days). "Hundreds to thousands of concurrent calls" at SLA 80, tau 4.7 needs ~2-9 warm 4x B300 nodes ($57-256/hr), i.e. $42k-190k/month. $15k is an R&D budget; production serving needs committed-spend economics (Enterprise) or a different provider structure. Do not let any phase leave a warm pool running overnight.
2. **Hidden reasoning tokens.** `reasoning_effort` defaults to max. If data regeneration or serving forgets to disable thinking, token consumption inflates severalfold, silently multiplying both phase (b) corpus cost and per-conversation serving cost.
3. **Naive teacher co-residency in phase (c).** Running an 8-GPU target node live alongside the 8-GPU drafter trainer ($50-62/hr combined) wastes money; capture teacher hidden states offline from the phase (b) server (SGLang PR #36708 shipped exactly this hidden-state capture in v0.5.20).
4. **Long high-concurrency soaks** (c200+ for many hours) - cap soak durations in the benchmark harness.

Cheapest de-risking order:

1. **(a) Proxy pipeline on Qwen3-4B, single H100, ~$250.** Proves DSpark-style drafter training, confidence-head calibration, and STS end-to-end at trivial cost.
2. **Zero-training acceptance test (new option the plan did not include): serve GLM-5.3-Flash with the released DFlash2 drafter** (`incoai/GLM-5.3-Flash-DFlash2`, access-gated) on 4x B200 using the Low Latency recipe for ~8-16 hours ($200-400) and measure natural acceptance vs concurrency. If tau >= 4 sustains at target concurrency, phase (c) collapses to ~$100.
3. **(b) Data regeneration, serverless, thinking off, spec off, scale-to-zero:** $500-1,300.
4. **(c) Drafter training only if DFlash2 disappoints:** $750-2,900.
5. **(d) Benchmark campaign on 4x B200 + 8x H200:** $1,300-2,900, including the acceptance-vs-concurrency curve and an SLA frontier replication of the DSpark Figure 7 shape.

Total: ~$2,000-4,700 on the DFlash2 path, ~$2,700-7,600 with training - leaving $7k+ of the $15k for a second iteration or a bounded pilot warm window (e.g., one 4x B200 node warm 8 h/day on business days ~ $4k/month).

## Sources

- Modal pricing (GPU per-second rates, plan caps, multipliers): <https://modal.com/pricing>
- Modal GPU types/counts, H100->H200 upgrade, multi-node training beta: <https://modal.com/docs/guide/gpu>
- Modal function timeouts (1 s-24 h): <https://modal.com/docs/guide/timeouts>
- Modal autoscaling (min_containers, buffer_containers, scaledown_window, 4,000-container limit): <https://modal.com/docs/guide/scale>
- Modal cold start (idle warm containers billed): <https://modal.com/docs/guide/cold-start>
- Modal Memory Snapshots (GPU snapshots alpha, multi-GPU incompatible, no storage speedup): <https://modal.com/docs/guide/memory-snapshots>
- Modal billing (per-second, budgets, tagged reports): <https://modal.com/docs/guide/billing>
- SGLang GLM-5.3-Flash cookbook and 2026-09 benchmark campaign (GB300 and H100 numbers, DFlash2, PR #36708): <https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash> (benchmarks snippet: sgl-project/sglang docs/src/snippets/configs/zai-org/glm-5.3-flash-benchmarks.jsx)
- vLLM recipe (306 GiB FP8 weights, VRAM minimums, verified hardware, TP4+MTP): <https://recipes.vllm.ai/zai-org/GLM-5.3-Flash>
- GLM-5.3-Flash model card (320B/18B, FP8, reasoning_effort): <https://huggingface.co/zai-org/GLM-5.3-Flash>
- DSpark paper (local papers/DSpark-2607.05147.txt): Figure 7 SLA anchors 80/120 tok/s/user, +51%/+661% vs MTP-1; Figure 8 verification budget 2 -> 4-6 under load; 0.2-1.3% sequential-head overhead; V4-Flash frontier up to ~20k tok/s/GPU axis
- GLM-5 technical report (local papers/GLM-5-technical-report-2602.15763.txt): MTP parameter sharing; "acceptance length of GLM-5 is longer than DeepSeek-V3.2, given the same number of speculative steps"; FP8 rollouts + MTP for tail latency
- DeepSeek-V4.1-Flash paper (local papers/DeepSeek-V4.1-Flash-2609.19969.txt): inference-system context (EPD disaggregation, fused-kernel decode path)

## Unknowns and confidence

- **GB300 -> B300 transfer** (Modal sells B300, campaign ran GB300): medium confidence, same Blackwell Ultra class; SGLang lists a B300 recipe but its verification badge state is unclear; allow ±30-40%.
- **Hopper throughput at c80+**: only c1/c16 measured on H100; the ~5x c16->c80 scaling is extrapolated from the GB300 curve - low-medium confidence.
- **H100 numbers use simulated acceptance** (SGLANG_SIMULATE_ACC_LEN=3): mechanism numbers, not natural acceptance - medium confidence for planning.
- **Modal credit mechanics** (prepaid vs grant redemption, whether the $250/month Team fee draws from credits): the pricing FAQ is not machine-readable; confirm in the dashboard before committing. Low confidence; low stakes.
- **DFlash2 drafter on cookbook hardware**: supported and shipped but "not yet measured"; its acceptance-vs-concurrency curve is the single most decision-relevant unknown in the plan - medium confidence it works, zero confidence in its numbers until phase (d)-style measurement.
- **Spec-off high-throughput tok/s on Modal hardware**: unmeasured; estimated 5-10k tok/s/node from the GB300 curve plus the cookbook's guidance that spec off wins under sustained batches - low-medium confidence.
- **Memory-snapshot cold start for this engine**: high confidence it cannot rescue scale-to-zero phone serving (alpha feature, multi-GPU incompatible, weights load dominates regardless).
