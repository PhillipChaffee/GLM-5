# Sequential head for the GLM-5.3-Flash DSpark drafter: within-block prefix conditioning survey

**Research for ticket #13 (map #1)** — 2026-09-27
Claimed by session `research-seq-head-2026-09-27`. Feeds ticket #8 (drafter design decision). Companion docs: `docs/research/deepspec-audit.md` (#2), `docs/research/glm-flash-engines.md` (#3), `docs/research/acceptance-rates.md` (#5).

## TL;DR verdict

The ticket's goal — draft position k sees all of x₁…x_{k−1} — is exactly DeepSpec's **RNN head**, and the answer is: it already exists end to end. It is implemented in DeepSpec's training code (`markov_head_type="rnn"`, config-only switch), implemented in SGLang's in-tree DSpark stack (`SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla", "gated", "rnn")`), and evaluated by the DSpark authors themselves — who rejected it as the default: *"RNN head provides only marginal additional gains over the Markov head, mainly at longer proposal lengths. Given its higher implementation complexity and less favorable deployment properties, we use the Markov head as the default"* (DSpark paper, Sec 4.3.2). DeepSeek's own production V4.1-Flash drafter ships the vanilla Markov head, as does NVIDIA's Nemotron-3.5 DSpark drafter. For a phone-call-serving drafter at γ≈5–7 with a confidence scheduler pruning doubtful suffixes and TTFT (not throughput) binding the SLA, **ship vanilla first; carry the RNN head as the named fallback** — it costs zero SGLang patches, so the real trigger is a proxy-run acceptance-decay curve, not engineering fear. The ticket's framing premise is partly outdated: "a non-vanilla head likely needs SGLang proposer patches" is false for SGLang today (see §3, Engine integration).

## 1. What exists

### 1.1 DeepSpec's three heads (the family-port surface)

All in `deepspec/modeling/dspark/markov_head.py` (local clone `$OPENCODE_SCRATCH_DIR/DeepSpec`, read from `main`):

| Head | What it conditions on | Extra params | File:lines |
|---|---|---|---|
| `VanillaMarkov` | x_{k−1} only (first-order bigram bias, low-rank r=256: `B = W1[x_{k−1}]·W2`) | W1+W2 ≈ 2·r·V | markov_head.py:8–91 |
| `GatedMarkovHead` | x_{k−1} **plus backbone hidden h_k** via a sigmoid gate on `[h_k; W1[x_{k−1}]]` — still first-order in tokens | + (d+r)·r gate | markov_head.py:93–123 |
| `RNNHead` | **full within-block prefix** via GRU-style recurrent state s_k over `[s_{k−1}; W1[x_{k−1}]; h_k]`, joint projection (2r+d)→3r | + (2r+d)·3r ≈ 3.5M (GLM d=4096) | markov_head.py:125–284 |

The RNN head is the ticket's goal, verbatim: its docstring says "position k can access the full prefix history x_{<k} through a GRU-like recurrent state" (markov_head.py:126–129). Selection is config-driven: `build_markov_head(config)` dispatches on `config.markov_head_type` (markov_head.py:287–311); every family port (qwen3/config.py:30–55, gemma4/config.py:75–101) passes `markov_head_type` through, and the training script needs no code change. **Caveat:** every shipped DeepSpec config sets `'vanilla'` (config/dspark/dspark_qwen3_{4b,8b,14b}.py, dspark_gemma4_12b.py) — the RNN path is implemented but not exercised by released checkpoints, so it carries reproducibility risk (untrained-by-anyone-public at GLM scale).

Integration points that do not change with head choice: the backbone stays a single parallel pass (qwen3/modeling.py:361–427); corrected logits flow into verification identically — `sample_draft_tokens` → `draft_logits` → `draft_probs` → rejection sampling (eval/dspark/draft_ops.py:96–153). The confidence head already consumes the *sampled* within-block prefix (`prev_token_ids` from `sampled_tokens[:, :-1]`, draft_ops.py:65–79; qwen3/modeling.py:504–516) — so the scheduler side is prefix-aware even with the vanilla head. **Losslessness is unaffected**: the non-anticipating property the paper formalizes (DSpark paper lines 387–401) constrains *scheduler admission decisions*, not the head; the head's left-to-right conditional p_k(·|x₀, x_{<k}) is exactly the drafter distribution the verifier must see for standard rejection sampling to preserve the target distribution.

### 1.2 The DSpark paper's own position

Sec 3.1 (lines 208–260 of `papers/DSpark-2607.05147.txt`) defines the semi-AR factorization P(X|x₀) = ∏ₖ p_k(x_k|x₀, x_{<k}) with transition bias B_k — the *general* form already conditions on the full prefix; the Markov head is presented as "the simplest instantiation" restricting B_k to one token back, with the RNN head as the full-prefix instantiation. The design constraint the paper sets: T_sequential ≪ T_parallel so the block stays single-pass (lines 235–237). Sec 4.3.2 (lines 657–692) then ablates: both heads across proposal lengths {4, 8, 12, 16} on Qwen3-4B (Figure 4 — numbers are figure-only, not in the text sidecar), verdict quoted above, plus: the whole sequential loop adds only **0.2–1.3% to full-round latency at batch 128** while scaling γ from 4 to 16. Context: DSpark(markov) beats DFlash (pure parallel) by +16–18% τ and Eagle3 (fully autoregressive) by +27–31% τ (Table 1, lines 485–514); chat is the worst domain everywhere (τ 3.1–3.7 vs math 4.9–6.2).

### 1.3 The wider literature (as of late 2026)

Within-block conditioning on the parallel↔sequential spectrum:

- **Medusa** (arXiv:2401.10774): k parallel decoding heads + tree attention; heads are *sequentially independent* — the negative control Hydra starts from.
- **Hydra** (arXiv:2402.05109, MIT): "sequentially-dependent draft heads" — head k consumes the tokens sampled by earlier heads; Hydra++ up to 1.31× throughput vs Medusa. The classic within-block-conditioning citation. Code: github.com/zankner/Hydra.
- **Clover** (arXiv:2405.00263, Apple/Baichuan): Regressive Connection + Attention Decoder injecting "sequential knowledge from pre-speculated tokens" into parallel heads; +37–57% over Medusa. Closed-loop conditioning, Medusa-era hardware.
- **EAGLE-3** (arXiv:2503.01840): the fully-autoregressive extreme — feature-level AR drafting, T_draft ∝ γ; strong τ, but drafting cost grows with block size and it's the drafter DSpark beats by ~30% τ on equal training budget. In-tree in SGLang and vLLM; the "proven production" option for *rich* conditioning, at a different operating point.
- **Domino** (arXiv:2605.29707, May 2026) — closest 2026 sibling: parallel draft backbone + "lightweight Domino head [that] refine[s] them with prefix-dependent causal information," explicitly decoupling causal modeling from AR drafting; GRU-based correction per DominoTree's description; base-anchored training curriculum; 5.49×/5.8× (Transformers/SGLang). Confirms the semi-AR + causal-correcting-head recipe is reproducible outside DeepSeek.
- **xPress** (arXiv:2608.02438): "lightweight causal refiner that reconciles the whole diffusion block at once through parallel refinement" — restores cross-position causality **without a token-by-token loop**; +~30% acceptance length over dFlash (up to +56%). The most interesting *alternative to the sequential loop itself*: a second parallel pass instead of a serial head.
- **AngelSpec / DFly** (arXiv:2607.25852, Tencent AngelSlim): block-diffusion backbone + "**predecessor-conditioned autoregressive head** … keeping generation parallel"; co-specializes MTP (chat) vs block-diffusion (code/math) drafters; 10.5–11.8% throughput over DFlash on Hy3-A21B. Notably ships the structure choice "MTP for chat, block-parallel for code" that phone-workload analysis should absorb.
- **LiLiCorr** (arXiv:2608.20530): single-pass likelihood-correlation scoring of DFlash's per-position top-K tokens; +7–19% acceptance at ~3% block latency; explicitly benchmarks against "three concurrently developed methods that also restore coherence at draft time."
- **Tree family** (orthogonal axis, all built on parallel/backbone marginals): DDTree (arXiv:2604.12989), GRAFT (2608.20375), CaDDTree (2606.01813), **TreeSpark** (arXiv:2609.22098 — reads "a parent-conditioned distribution from the drafter's *existing* Markov head," +15–25% tokens/round vs tuned chain, collapses to chain under load), **DARTree** (arXiv:2608.13524 — training-free extension of an AR correction head from chains to trees; +27.9% τ over Domino), DominoTree (2607.08642 — its ablation attributes ~+10.1% τ to applying the GRU correction at all and ~+4.7% more to rescoring along realized tree paths).
- **Cross-round conditioning** (not within-block, adjacent): Carryover Drafting (arXiv:2609.14717 — recycles rejected target hidden states as drafter KV context, +6.5–14.7% τ on a "DSpark-derived semi-autoregressive drafter"), DFlow (2609.06498), ReTrace (2608.29748) — verifier-side rejected-suffix information flowing into the next round.
- **Production evidence**: DeepSeek-V4.1-Flash serves DSpark with a 3-block Transformer drafter (sliding window 128, 5 draft positions) and "a lightweight **Markov head** models dependencies among the draft tokens" (papers/DeepSeek-V4.1-Flash-2609.19969.txt lines 612–628) — vanilla, trained in a dedicated post-pretraining stage with frozen backbone. NVIDIA's Nemotron-3.5 DSpark drafter in SGLang is vanilla-only (`Nemotron35VanillaMarkov`, sglang models/dspark.py:187). No vendor has shipped a beyond-Markov sequential head in a production stack that we can find.

### 1.4 Negative space

No paper we found reports a head richer than the RNN/GRU formulation (e.g., transformer-based sequential block) at equal latency budget, and none reports within-block conditioning *beating* one-token-back by a large margin at γ ≤ 7. The 2026 energy went to (a) trees over existing heads, (b) cross-round state recycling, (c) training objectives (D-PACE 2605.18810, VAT 2608.30135, Draft-OPD 2605.29343, TTS 2605.09329), (d) parallel causality restoration (xPress, LiLiCorr). The field converged on "one cheap causal correction" rather than "rich conditioning."

## 2. Implementation tradeoffs

Quantified for the GLM port (vocab V=154,880, drafter hidden d=4096, r=256, γ≈5–7):

| Option | Spectrum position | Per-step draft cost | Block single-pass? | Expected τ vs vanilla | Training delta | SGLang integration | vLLM integration |
|---|---|---|---|---|---|---|---|
| **Vanilla Markov** (default) | parallel + 1st-order bias | embed + r·V matmul ≈ 39.7M MACs | yes | — baseline | — | in-tree, fused-greedy + folded-sampling fast paths, proven (V4.1-Flash, Nemotron 3.5) | in-tree per-family DSpark proposers (ticket #3); vanilla is what they implement |
| **Gated head** | parallel + hidden-gated 1st-order | + gate_proj (d+r)·r ≈ 1.1M | yes | small (paper doesn't headline it; no public number) | config-only | **in-tree** (`gated`), but loses fused-greedy fast path | unverified; likely proposer-class patch |
| **RNN head** | parallel + full-prefix GRU correction | + joint_proj (2r+d)·3r ≈ 3.5M MACs (~+9% of the r·V matmul) | yes (loop is γ tiny steps ≪ backbone pass; paper: +0.2–1.3% round latency at bs=128) | "marginal, mainly at longer γ" (paper, Figure 4; exact deltas figure-only) | +3.5M params, same data/epochs, teacher-forced unroll already implemented | **in-tree** (`rnn`), generic block sampler path; misses fused-greedy optimization | unverified; likely patch |
| xPress-style parallel refiner | parallel + parallel causal refine | one extra light forward | yes | +~30% AL over DFlash (no-conditioning baseline) — vs vanilla DSpark unknown | new training component | new proposer class + engine patches | patch |
| LiLiCorr scorer | parallel + single-pass correlation head | ~3% block latency | yes | +7–19% over vanilla DFlash | co-training | patch | patch |
| Domino-style head | semi-AR, own curriculum | ≈ RNN head | yes | +10.1% AL attributed to the correction (DominoTree decomposition) | curriculum change | patch (out-of-tree plugin exists) | patch |
| EAGLE-3 (full AR head) | sequential | T_draft ∝ γ | **no** | −27–31% τ vs DSpark at equal budget (Table 1) — but strong τ in production stacks | separate recipe (TTT) | in-tree | in-tree |
| Trees (TreeSpark/DDTree/DARTree…) | tree over chain | per-node scoring | yes (batched) | +15–25% tokens/round (TreeSpark vs tuned chain); DARTree +27.9% over Domino | mostly training-free | new verifier path (tree attention on DSpark stack) — the biggest surface of all | patch |

Two honest cost notes. First, the "marginal" RNN gains in Figure 4 are the DSpark authors' characterization of a figure we cannot read exact numbers from (text sidecar has no table for it) — treat the delta as unmeasured-but-small at γ ≤ 8. Second, the fused-greedy/fast-path asymmetry inside SGLang is real: the vanilla head gets a sharded greedy kernel and CUDA-graph-folded sampling; Gated/RNN take the generic block-sampler path, which is correct but slower at bs=1 latency-critical serving — precisely the phone-agent regime.

## 3. Do we even? (recommendation)

**Ranked options for ticket #8:**

1. **Ship vanilla Markov** (rank 1). It is what DeepSeek serves in production on V4.1-Flash, what NVIDIA ships on Nemotron 3.5, the only head with SGLang fast paths, and the proxy run (#6) is already training it. On phone calls — open-ended chat, the worst-acceptance domain, γ≈5–7, τ target 2.0–3.7 per ticket #5 — the first 2–3 positions carry most of τ, and the confidence-scheduled verifier prunes exactly the positions where richer conditioning would help most. The vanilla head's bias *is* a within-block conditional correction; the ticket's "first-order Markov bias only" framing understates it: TreeSpark demonstrates the Markov head already yields usable parent-conditioned distributions.
2. **RNN head as named fallback** (rank 2), decided by proxy-run evidence, not by engine fear — it is a config flag in both DeepSpec training and SGLang serving, so the incremental cost is one retrain of a 3.5M-param module plus re-measurement, not an engine project.
3. **Do not build now**: xPress/LiLiCorr-style coherence restorers (no DSpark-head integration, unknown interaction with the sequential bias), tree verification on the DSpark stack (largest engine surface, load-adaptive chain already collapses to chain under load anyway), cross-round state recycling (engine-level, orthogonal backlog), full-AR EAGLE-style head (abandons the single-pass block property that motivates DSpark).

**The honest argument for vanilla, stated plainly:** proven in two production stacks; zero engine patches; the verifier's confidence schedule already prunes doubtful positions, which is a *substitute* for within-block smarts at the tail; and the paper's own ablation says the full-prefix head buys "only marginal" τ at phone-relevant block sizes while giving up the vanilla head's kernel optimizations. The counter-argument for RNN: chat is the domain where suffix decay bites hardest and where TreeSpark-style evidence says conditioned information still pays (+15–25% over chain) — but that evidence is for *trees over* the head, not for the head alone.

**Proxy-run (#6) evidence that would change the ranking** (tell ticket #8 to look for exactly these): (a) position-wise conditional acceptance curves (Figure 2-style) on phone-call-like chat prompts — if conditional acceptance at positions 4–7 drops steeply and the scheduler's τ is capped by it, RNN-head A/B jumps to priority; (b) ceiling-clipped acceptance histograms (per Wu, arXiv:2608.30427) — a spike in the "whole block accepted" bin argues for longer blocks, where the RNN head's gains concentrate (paper: "mainly at longer proposal lengths"); (c) if phone transcripts show long verbatim spans (confirmations, order numbers, tool-call echoes), structured-text behavior rewards conditioning and longer γ. Conversely, if position-1/2 acceptance dominates τ and the scheduler prunes at 2–3, vanilla is settled and #8 should spend its budget on the GLM family port, not the head.

## Sources

**Code (read directly):**

- `$OPENCODE_SCRATCH_DIR/DeepSpec/deepspec/modeling/dspark/markov_head.py` (all three heads); `qwen3/modeling.py` (backbone/head/confidence integration); `qwen3/config.py` (head-type passthrough); `eval/dspark/draft_ops.py` (corrected logits → verification); shipped configs (all `'vanilla'`)
- SGLang @ main: `python/sglang/srt/models/dspark.py` (VanillaMarkov/GatedMarkovHead/RNNHead/Nemotron35VanillaMarkov, build_markov_head); `speculative/dspark_components/dspark_draft_sampler.py` ("Gated/RNN subclasses return None… fall through"); `dspark_config.py` (`SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla","gated","rnn")`) — fetched 2026-09-27 via GitHub API

**Papers (local sidecars, `rg`-checked):**

- DSpark: `papers/DSpark-2607.05147.txt` — Sec 3.1 lines 208–260 (heads), 3.2.1 lines 280–318 (confidence head), 3.2.2 + lines 387–401 (scheduler, non-anticipating property), 3.3 lines 402–437 (training), Table 1 lines 485–504, Sec 4.3.2 lines 657–692 (RNN ablation + latency)
- DeepSeek-V4.1-Flash: `papers/DeepSeek-V4.1-Flash-2609.19969.txt` lines 612–628 (production DSpark, vanilla Markov head, dedicated training stage)
- GLM-5 TR: `papers/GLM-5-technical-report-2602.15763.txt` Table 2 lines 198–216 (MTP accept length 2.76)

**Web (arXiv abs/search pages, fetched 2026-09-27):**

- Medusa arXiv:2401.10774 · Hydra arXiv:2402.05109 (+ github.com/zankner/Hydra) · GLIDE arXiv:2402.02082 · Clover arXiv:2405.00263 · PEARL arXiv:2408.11850 · EAGLE-3 arXiv:2503.01840 (+ github.com/SafeAILab/EAGLE) · DFlash arXiv:2602.06036 (ICML 2026, github.com/z-lab/dflash) · DDTree arXiv:2604.12989 · Domino arXiv:2605.29707 · D-PACE arXiv:2605.18810 · Draft-OPD arXiv:2605.29343 · DFlare arXiv:2606.02091 · WhiFlash arXiv:2606.07710 · CaDDTree arXiv:2606.01813 · AngelSpec/DFly arXiv:2607.25852 · DominoTree arXiv:2607.08642 · GRAFT arXiv:2608.20375 · xPress arXiv:2608.02438 · DARTree arXiv:2608.13524 · VAT arXiv:2608.30135 · LiLiCorr arXiv:2608.20530 · ReTrace arXiv:2608.29748 · TTS arXiv:2605.09329 · DFlow arXiv:2609.06498 · Carryover arXiv:2609.14717 · TreeSpark arXiv:2609.22098 (+ github.com/PopSoda2002/TreeSpark) · DPara arXiv:2609.27396 · H-Spec arXiv:2609.24197 · block-expansion histogram study arXiv:2608.30427

**Tracker (resolved tickets):** #2 (DeepSpec audit — family-port surface), #3 (engine support — SGLang in-tree DSpark, vLLM experimental), #5 (acceptance rates — phone τ bands, TTFT-binds finding)

**Confidence levels:**

- High: DeepSpec head code and integration; SGLang in-tree support for all three head types; DSpark paper's RNN-vs-Markov verdict (direct quote); V4.1-Flash production vanilla choice; GLM-5 MTP baseline.
- Medium: exact magnitude of RNN-vs-Markov τ delta (Figure 4 is figure-only; "marginal" is the authors' wording); 2026-paper τ claims (abstract-level, not reproduced by us); vLLM head-type support (unverified in repo — code search returned no hits; ticket #3's audit established experimental DSpark support but did not enumerate head types).
- Low: whether the 3.5M-param RNN module retraining behaves at GLM's 154,880 vocab (W1/W2 sharding dominates memory; joint_proj is negligible but untested) — flag for ticket #8.
