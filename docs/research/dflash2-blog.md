# DFlash2: "Keep Drafting Parallel" — Inco AI, August 2026

Source: <https://inco.ai/blog/dflash2/> (published 2026-08-18; fetched 2026-10-06).
This is an **archival digest, not a verbatim copy**: the structure, all quantitative
results, and key quotes are preserved; prose is condensed. The full text lives at the
URL above. The DFlash paper itself is in `papers/DFlash-2602.06036.pdf` (sidecar `.txt`).

Context for this repo: DSpark's parallel backbone *is* DFlash (one attention
modification); the GLM-5.3-Flash DFlash2 checkpoint (`incoai/GLM-5.3-Flash-DFlash2`)
is this effort's zero-training baseline (map ticket #12). All comparisons below are
**offline accepted-length (τ)** measurements — none includes a confidence scheduler,
which DFlash2 does not have (see the #7 resolution for why that matters).

## Ecosystem standing

DFlash runs in SGLang, vLLM, TensorRT-LLM, and llama.cpp. NVIDIA measured up to 15x
throughput on Blackwell; Google reported 3x on TPUs; CoreWeave's production Kimi K2.7
Code endpoint runs it by default. Official drafters ship from NVIDIA, Red Hat, Modal,
Meta (Muse Glimmer), Poolside, and Xiaomi. DFlash-family models: 3.5M+ HF downloads
(as of August 2026).

## The two modules (on a 5-layer Qwen3-4B DFlash backbone)

1. **Path selector** (+2.0M params, +0.6% draft-verify cycle latency). DFlash keeps
   top-16 candidates per position; the selector scores every adjacent pair with a
   low-rank bilinear match — `S_t(a,b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>` — where
   `U_t(b)` is DFlash's own logit, `A`/`B` are 256-dim token embeddings, and `H(h_t)`
   is a context gate. One greedy (or rejection-sampled) walk then traces a single
   coherent path. Fully parallel scoring; the only sequential work is the walk over
   precomputed scores.

2. **Two-tap dynamic convolution** (+16.5M params = 3%, +0.7% cycle latency). Inserted
   before and after every attention and MLP sublayer: `Conv_k(x)_t = k_{t,0} ⊙ x_t +
   k_{t,1} ⊙ x_{t-1}` — each coefficient is a learned base kernel plus a small
   correction from the hidden state, shared per 16 channels. Position 1 reads the last
   verified token; information crosses the block while all positions still compute in
   parallel. Rationale: DFlash's within-block attention share collapses with depth
   (30% in layer 1 to 8% in layer 5, concentrated in a few heads), so the local
   operator absorbs the local work.

## Why not a sequential head (the DSpark rebuttal)

Key measurement — recall at each draft position conditioned on all earlier positions
being right (5-layer Qwen3-4B DFlash on GSM8K): top pick is right 85.4% at position 1,
but the right token is in the top-16 candidates 99.5% of the time; even the *oracle*
selector decays (99.5% → 87.8% by position 7) because candidates run out — "suffix
decay is a backbone problem", and mostly a *local* one.

Head-to-head on the same backbone (Qwen3-4B, GSM8K, T=0 / T=1 acceptance length):

| Method | Params added | Cycle latency added | T=0 | T=1 |
|---|---|---|---|---|
| DFlash | — | — | 4.27 | 3.78 |
| + DSpark correction (Markov head) | +77.8M | +9.6% | 4.49 | 4.08 |
| + path selection (DFlash2) | +2.0M | +0.6% | **4.61** | **4.25** |

Their claim: "choosing is cheaper than predicting" — the selector beats the Markov
correction with ~40x fewer parameters and ~16x less latency. (The +77.8M figure
matches a rank-256 Markov head at Qwen3's vocab — the same design DeepSpec ships.)

Depth alone also helps late positions (15-layer DFlash: 78.7% recall at position 7 vs
72.9% for 5-layer) but costs 15.2% cycle latency; the convolutions reach nearly the
same recall at +3% params / +0.7% latency.

## Full comparisons (mean acceptance length, per-request, lossless rejection sampling)

Qwen3.5-4B (block size 7 implied; thinking enabled, temp 1.0, top-p 0.95, top-k 20,
presence penalty 1.5):

| Dataset | MTP | DFlash | DSpark | DFlash 2 |
|---|---|---|---|---|
| GSM8K | 4.78 | 4.99 | 5.69 | **6.20** |
| MATH-500 | 5.04 | 5.42 | 6.20 | **6.76** |
| HumanEval | 4.84 | 5.43 | 5.80 | **6.28** |
| MBPP | 4.16 | 4.49 | 4.96 | **5.41** |
| MT-Bench | 3.90 | 4.26 | 4.77 | **5.20** |
| Mean | 4.54 | 4.92 | 5.49 | **5.97** |

Qwen3.8-27B (block size 8): MTP 4.28, DSpark 3.62, **DFlash2 4.80** — DFlash2 beats
the model's native MTP and the community DSpark drafter (RadixArk) by wide margins;
2.7-3.4x autoregressive throughput at batch 1.

Muse Glimmer 30B (block size 16): DFlash 4.48, DSpark 4.48, **DFlash2 5.70** — more
than a full token ahead of both; 3.1-4.6x autoregressive throughput.

On MATH-500 position-by-position (Qwen3.5-4B), DFlash2 holds ~85-86% conditional
acceptance through position 14 while DSpark ends ~6-9 points lower.

## The GLM-5.3-Flash checkpoint

`incoai/GLM-5.3-Flash-DFlash2` — 1.17B params BF16, 5-layer Qwen3-shaped backbone
(sliding window 2048, GQA 32/8 heads, hidden 4096, full 154,880 vocab,
`is_causal: false`), `model_type: qwen3`, architecture `DFlash2DraftModel`. Config:
`block_size: 8`, `selector_rank: 256`, `selector_top_k: 16`, `conv_kernel_size: 2`,
`conv_group_size: 16`, `mask_token_id: 154856`, target hidden-state taps at target
layers `[5, 14, 24, 33, 42]`. **Not access-gated** (HF API `gated: false`, verified
2026-10-06). License: **CC BY-NC-ND 4.0** — research/evaluation use fine; no
derivatives, so this effort's trained drafter must not warm-start from its weights.
Vendor benchmark numbers for this checkpoint: unpublished ("being re-measured") —
ticket #12 produces them. SGLang launch (cookbook): install from PR #36708, then
`--speculative-algorithm DFLASH --speculative-draft-model-path
incoai/GLM-5.3-Flash-DFlash2 --speculative-draft-attention-backend fa4`.