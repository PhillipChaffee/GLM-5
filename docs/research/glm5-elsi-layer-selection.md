# Offline layer-selection analysis for the GLM-5.3-Flash drafter (ELSI/ELSI-C)

Research for [map ticket 51](https://github.com/PhillipChaffee/GLM-5/issues/51)
(the measurement stage of the layer-set decision [#33](https://github.com/PhillipChaffee/GLM-5/issues/33)).
Executed 2026-10-13. Code: `experiments/glm-drafter/scripts/data/elsi_analyze.py`
(statistics + beam searches), `experiments/glm-drafter/scripts/data/elsi_capture_glm.py`
(all-45 capture), `experiments/glm-drafter/elsi_glm.py` (Modal stages
`stats` / `capture` / `select`), tests
`experiments/glm-drafter/tests/test_glm5_elsi_{math,pipeline}.py`.
Recipe: arXiv:2610.12939 "Surfing the Activation Wave" (ELSI / ELSI-C);
full text at `papers/Surfing-Activation-Wave-2610.12939.txt`.

## Question

Which target-hidden-state sets are worth A/B-ing for the GLM-5.3-Flash
drafter, measured by joint informativeness (matrix entropy) and redundancy
(CKA) — stage 1 on the cached 7-stream smoke cache, stage 2 on a fresh
all-45-layer capture with the paper's predictive-entropy token filter.

## Method (and deliberate deviations)

- **Streaming chunk sums.** Every stream column is scatter-added into global
  32-token chunks along the retained-token axis, so a candidate subset's row
  Gram is the plain sum of per-[U,U] column Grams (the inner product of two
  concatenated rows decomposes over feature blocks). Two pooled granularities
  are computed from the same sums at whole-chunk edges: U=32 (the paper's
  B=32 reading) and U=256 (a finer check). A cross-check test
  (`test_glm5_elsi_math.py`) validates additivity against a direct
  concatenation-and-centering reference.
- **Token views.** Stage 1 runs on the existing cache, which stores no
  logits, so the paper's rho=0.3 predictive-entropy filter is unavailable
  there — the token views are `all` rendered positions and `assistant`
  (loss-mask). Stage 2 applies the paper's filter exactly (per-row top-30%
  by predictive entropy, computed from full-vocab logits at temperature 1).
- **Row-normalized variants.** The cached id-44 column is post-final-norm
  (RMS-normalized) while the other columns are raw residual folds; the raw
  view's set-entropy comparisons therefore conflate scale with structure
  (stage 2 in the raw view put `base4+44*` at 4-9x every other standing set).
  The row-normalized view rescales every pooled rep row to unit norm so
  mixed columns contribute comparable variance budgets. Reported twice,
  raw and rownorm, with rownorm as the fair read.
- **Chance baselines.** CKA between row-shuffled column pairs fixes the
  finite-sample noise floor (all u=256 bands: raw 0.001-0.009, rownorm
  0.004-0.017 — pooled reps with thousands of features do NOT have near-zero
  chance CKA, but 256 pooled rows make it small here).

## What was verified (the de-risk log)

1. Every math primitive has a known-answer unit test: scatter-add accuracy
   vs a dense independent reference; pooled rows equal direct token means;
   Renyi-1 entropy = ln n on flat spectra and 0 on rank-1; CKA identity = 1
   and row-permutation/rotation semantics pinned; **subset Gram additivity
   proven exact against direct concatenation**; duplicate-column sets leave
   entropy unchanged.
   The unit tests caught a real algebra bug pre-run: the centered Gram was
   initially computed with the feature-Gram centering form
   (`gram - n*mu mu^T`) on a row Gram — shapes happened to align in some
   fixtures, which is why the wrong formula can slip through naive testing.
   Both beam searches had a tuple-order bug caught by the same suite.
2. Stage-1 sanity: the cache's id-44 column is **bit-identical** to
   `target_last_hidden_states` for every checked sample (max abs diff 0.0),
   confirming the fold+re-norm equivalence claim in the capture research.
3. Capture pre-flight on GPU: 3-row dry run validated 46 column files,
   finite entropies (max 10.99 ≈ ln 154880), 4 nonzero chunks for 98 tokens.
4. A REAL bug surfaced at the first real-scale retention count: 0.430
   retained vs 0.300 expected. Cause: rows shorter than the batch max got
   their top-rho selection computed over padded positions. Fix: slice each
   row to `attention_mask` length before masking and pooling. The re-run
   measured retention = **0.3000 exactly**; the regression test
   (`test_capture_pooling_matches_reference`) now pins short-row behavior.
5. Pattern gates before trusting any ranking (paper-consistency): per-layer
   entropy wave shows a low layer-0, a layer-1 peak, a dip at layer 2, a
   climb peaking at ~27-28 (1.78 nats), a dip ~34-41, and a post-norm
   outlier at 2.93; transition peaks land at 44->45 (the final norm, delta
   3.45) and layer 2 boundaries (2.71) — the paper's activation-wave shape
   with compartments, on a different model family.

## Stage 1 (CPU-only, cached 7 streams; 686 samples, 529,899 tokens / 437,284 assistant)

Payload: `/vol/elsi/stage1.json` + `stage1_sums.pt` on Modal volume
`glm-drafter-run` (the JSON fetched into this session's scratch).

Headline reads (consistent across views):

- **CKA(42, 43) = 0.994-0.9999 in every view** — the adjacent-stage tail is
  near-totally redundant; one of the two is dispensable.
- Marginal S1 gains on base4 [1,12,23,34] (assistant, u=256 rownorm):
  +42 → +0.082, +43 → +0.068, +44* → +0.553. The post-norm column carries
  the single largest marginal, echoing the 5-spread hypothesis that 42/43
  are the weak members of the 7-union.
- Standing-set ranking (assistant u=256 rownorm): 5-spread 1.711 > 7-union
  1.656 > 6-stock 1.282 > 5-stock 1.240 > base4 1.158.

## Stage 2 (all-45 capture: 242 valid rows of 250, 89,432 tokens, 26,834 retained = exactly 0.3000)

`capture_meta.json` on the volume: max_length cap 2048, rho 0.3, target
`zai-org/GLM-5.3-Flash`, wall 7.5 min forward (126 batches at batch 2).
The full 46x46 centered-CKA matrix is `/vol/elsi/select.json`
(under `tables_by_u` → `256_rownorm` → `cka_matrix`).

Key numbers (u=256 rownorm — the fair view; chance band 0.004-0.017):

- **CKA(raw43, raw44) = 0.985** vs **CKA(raw43, postN) = 0.419** and
  **CKA(raw44, postN) = 0.334 raw / 0.409 rownorm** (verified from the
  shipped select.json; an earlier draft of this bullet swapped the two
  annotations). The final RMS norm is *not* a direction-only rescale at the
  Gram level: the per-token scalar s_i
  and gain g reweight each row pair by s_i s_j, visibly restructuring the
  Gram. The serving-side consequence is established: a pre-norm capture
  (the naive Kimi-seam trap) delivers genuinely different content than the
  trained post-norm column — matching the `glm5-final-layer-capture.md`
  warning that this mismatch class degrades acceptance silently.
- Standing-set entropies, u=256 rownorm (fair): 5-spread 1.877 > 7-union
  1.792 > 6-stock 1.497 > 5-stock 1.492 > DFlash2-set 1.488 > base4 1.478.
  Marginal on 5-stock: +43 → +0.005 (nothing), +44* → +0.338.
- ELSI's exact set picks are granularity-sensitive (u=32 rownorm prefers
  low-mid clusters {1,3,4,5,postN}; u=256 rownorm prefers a deep band
  {25-29}+postN) — the ranking prior is real but its set identity is not
  stable across the pooled granularity, consistent with treating ELSI as a
  prior and the smoke A/B as the decider (#33's own caveat).
- Ranking-wise (review-corrected wording, 2026-10-14): no single ELSI pick
  dominates the pre-registered 5-spread across all windows, and 5-spread
  ranks first among the pre-registered arms in every view. ELSI's
  single-window picks did beat the 5-spread score within individual
  windows — the original sentence on this bullet over-reached.

## The ≤3 proposed A/B arms (for the smoke A/B, ticket 52)

Arms must be subsets of the 7 cached columns (the cache stays 7-stream per
issue #33; out-of-union ELSI favorites are findings for #46, not capture
changes). The proposed arms:

1. **7-union [1,12,23,34,42,43,44\\*]** — the incumbent (step_20 trains it;
   zero extra cost), the reference arm. Needs the post-norm capture patch.
2. **5-spread [1,12,23,34,44\\*]** — ranked first among every standing set in
   every view (4/4), removes the measured-redundant 43 (CKA 42|43 ≥ 0.994)
   while keeping the single largest marginal (+44\* on base4), i.e. the
   ELSI/ELSI-C analysis's own first choice within the cache. Same patch
   surface as arm 1.
3. **5-stock [1,12,23,34,42]** — the zero-patch safe harbor. It strictly
   dominates 6-stock (same score band, fewer features) and base4 on the
   fair read; per #52's tiebreak rule (arms within ~0.1 tau → simplest
   wins), it wins ties by default.

Fallback positions the analysis adds: if the parity smoke of arm 1/2's
post-norm capture fails, 5-stock wins by default (#52's rule), and the
CKA(43,44)=0.985 / CKA(43,postN)=0.42 numbers are the direct evidence that
the parity gate checks for real content differences, not cosmetic roundoff.

## Capture engineering notes (stage 2)

- Offload: per-layer forward hook folds the mHC streams on GPU
  (`fold_streams`) and immediately moves a bf16 CPU copy; entropies are
  computed chunked from full-vocab logits (rows soft-clamped at -1e4 to keep
  deep-tail p*log(p) contributions exact instead of NaN); per-sample pooling
  happens CPU-side right after the forward. Peak GPU stays at model+batch;
  peak CPU ≈ a few GiB per batch.
- max_length cap = 2048 (paper-faithful shorter cap per the ticket); rows
  are per-row length-cut from the attention mask before the rho filter.
- rows: first 250 rows of `/vol/data/regen.jsonl`, 242 valid (min-loss
  filter, the cache builder's own 686/698 pattern).
- **GPU spend** (estimates; Modal bills in arrears — exact arrears on the
  workspace dashboard): the real capture ran twice (the unpatched run plus
  the fixed one), plus one 3-row preflight and one crashed B300 attempt.
  Measured counter values: H200 forward wall 7.5 min (126 batches, batch 2);
  load ≈ 2.5-3 min. At the published rate (Modal pricing page, H200 SXM
  $0.001261/s = $4.54 per GPU-hour, B200 $0.001736/s = $6.25) the two 4-GPU
  H200 runs ≈ 2 × (0.6 GPU-h) ≈ $5.4, the preflight ≈ $0.5, the B300 crash
  ≤ $1, B200 queue attempts billed nothing (no containers scheduled).
  **Total ≈ $7-12, inside the ticket's ~$5-25 cap.**
- Hardware note: exact-B200 could not be scheduled (queued > 30 min,
  "actively working on acquiring more capacity"; 2026-10-13) and B300
  (sm_103a) cannot run this stack (deepspec torch 2.9.1 cu12 Triton PTXAS
  rejection — the map already records the B300 rejection). H200:4 (sm_90)
  scheduled instantly and ran correctly; `elsi_glm.py` pins H200:4 with the
  reason in a comment.

## Harness corrections and decisions (GitHub issue 81, 2026-10-14)

From the five-way adversarial review of this analysis against
arXiv:2610.12939. Recorded-number reads above stay valid (every headline
re-verified from the shipped JSONs); the harness changes below are
prophylactic or fidelity-restoring and affect future runs.

- **NaN guard (matrix entropy).** Matrix-entropy eigen-share filtering now
  drops shares rounded to exactly 0.0 (a subnormal eigenvalue under a huge
  total used to produce `0 * log(0) = NaN` in the u>chunks regime).
  Bit-identical everywhere nothing underflows.
- **Accumulator capacity.** `StreamAccumulator` buffers are chunk rows, not
  token rows: the 32x RAM over-allocation is gone (recorded numbers never
  depended on it).
- **U-scale stability of the chance band.** The recorded chance band
  0.001-0.017 is valid only at u=256 (row count drives the finite-sample
  floor); the harness now recomputes the band at every pooled granularity.
- **Grid assert.** The column loader now raises unless every column shares
  one pooled chunk grid (counts identity), which the additive set-Gram
  identity assumes silently; previously an inconsistent capture would mix.
- **Search constants recorded.** select-report meta now carries
  `search_constants`: Kbeam = 10 is the paper's stated value (main text,
  citing supplementary fig. S3); Kexpand = Kpool = 10 are harness
  assumptions (no value stated in the accessible text, flagged by two
  reviewers).
- **ELSI-C transition rule — DECISION: implement the paper's rows.** The
  recorded elsi_c entries above were produced by a surrogate (Frobenius
  distance between consecutive candidates' *centered Grams*), which is
  scale/rotation sensitive: under it, a candidate duplicating a beam member
  scores ~16.0 while the paper's conditional-CKA row difference
  (`M_L(i,j) = CKA(¯Z_{L∪{i}}, ¯Z_{L∪{j}})` with strict set union,
  `δ_L(r) = ||M_L(r,:) − M_L(r−1,:)||₂`) makes the adjacent duplicate score
  0.0 and drop out of the candidate pool. The harness now implements the
  paper's rows at all pool sizes (feasible: N² CKA lookups per parent per
  depth; a parent member's own row uses the plain parent Gram, not a
  doubled one); **every elsi_c_top entry from the recorded runs should be
  read as surrogate-implemented**. The ELSI (entropy-only) searches never
  used the surrogate and reproduce bitwise.
- **Transition peaks follow Eq. 8 exactly.** The boundary deltas now
  concatenate the full M rows including self/adjacent entries. Measured on
  the recorded data: top-5 peak boundaries and their order unchanged, the
  large boundary deltas shifted ~2-9% (e.g. 44→45 by 2.3-9.3% across
  views), deep-tail deltas by under ~2.3%.
- **CKA matrix in fp64.** The printed 46x46 matrix is computed in float64 —
  diagonal reads exactly 1.0; fp32 roundoff had printed the two-sided
  ±5e-6 band (recorded diagonals 0.999996-1.000005, below-1.0 values
  included).
- **Fidelity caveats recorded for the future reader:** (a) pooled-grid
  reading — with 26,834 retained tokens the capture's chunk count is 839,
  so the new u=839 view is one rep row per 32-token chunk sum (the finest
  fidelity the saved sums support; pooled rep rows recover chunk means up
  to the fp32 cumsum/diff roundtrip); (b) tie handling — equal scores in
  enumerations and beams break by the harness's stable iteration order
  (candidate id / insertion), not a canonical tiebreak, so top-3 lists with
  near-equal scores can permute between harness versions with no math
  change; (c) fp32 noise — sub-1e-5 CKA and above-1.0 diagonals in any
  fp32 artifact are roundoff, not signal, and the fp64 matrix is the
  reference read.

## Implications for the downstream tickets

- **#45 (cost plan)**: the capture list is unchanged — the 7-union cache
  stands; the ELSI favorites outside the union (mid-band clusters per the
  u=256 view) are findings for the pre-train review, not capture edits.
- **#46 (pre-train review)**: cite this doc's stage-1/stage-2 tables, the
  42|43 redundancy read, the 44\* marginal, and the ELSI
  granularity-sensitivity caveat (ranking prior, not gospel).
- **#52 (smoke A/B)**: run the parity smoke first (unchanged), then the
  three arms above; the tiebreak rule stands; the $100-250 budget is
  unchanged. The evidence here does not shrink the A/B to two arms —
  5-stock remains the safe harbor the A/B exists to protect.
- The pre/post-norm mismatch is now measured as a CONTENT change (CKA
  0.33-0.42), not a scale-only quirk: the ~5-line post-norm return-value
  capture patch in glm5_next remains mandatory for any 44-including
  drafter, and the "Kimi seam delivers pre-norm" trap would have been a
  real quality loss, not a formality.

## Sources

- arXiv:2610.12939 (Surfing the Activation Wave) — Algs. 1-2, Eq. 4-10
  (matrix entropy in nats, CKA centering H = I - 11^T/T, per-row top-rho
  token filtering, block-averaged representatives); local sidecar at
  `papers/Surfing-Activation-Wave-2610.12939.txt`.
- Modal pricing page (2026-10-13 fetch): H200 SXM $0.001261/s, B200
  $0.001736/s, B300 $0.001972/s — the B200 rate matches the retro's cited
  $6.25/GPU-hour figure.
- `docs/research/glm5-final-layer-capture.md` (capture mechanics for
  glm5_next, pre/post-norm analysis, PR precedents).
- Repo code: the two data scripts + the Modal app listed above; the cache
  reader is DeepSpec's own `CacheDataset` (image-resident).
