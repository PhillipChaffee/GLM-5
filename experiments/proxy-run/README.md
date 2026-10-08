# Proxy run: DSpark on Qwen3-4B via DeepSpec, end to end

Work for map ticket #6 (PhillipChaffee/GLM-5). Reproduces DeepSpec's offline DSpark
pipeline against a Qwen3-4B target on Modal at proxy scale, measures accepted length
τ against the paper's Table 1 ballpark, and measures decode latency/throughput with
and without confidence-based pruning.

## What runs

DeepSpec pinned at `005e03b8` (research/deepspec-audit reference SHA). Pipeline stages
orchestrated by `dspark_proxy.py` (Modal app `dspark-proxy-run`, Volume `dspark-proxy-run`):

1. **split** (CPU): shuffled `mlabonne/open-perfectblend` subset → `data/perfectblend_train.jsonl`.
   Deviation from DeepSpec's `download_and_split.py`: shuffled (seed 42) instead of
   first-N rows, rows failing user-first validation are skipped not raised, no 5% test
   split (eval uses the repo's committed benchmark jsonl).
2. **regen** (1×H100): vLLM serves Qwen/Qwen3-4B in-process; DeepSpec's
   `generate_train_data.py` regenerates responses (temp 0.7, top_p 0.8, top_k 20,
   thinking off, resume support) → `data/regen.jsonl`.
3. **cache** (1×H100): DeepSpec's `prepare_target_cache.py` captures hidden states
   (layers [1, 9, 17, 25, 33]) → `cache/`.
4. **train** (4×H100): DeepSpec's `train.py` with the Qwen3-4B DSpark config
   (block γ=7, 5 draft layers, vanilla Markov head rank 256, confidence head, CE 0.1 +
   TV(L1) 0.9 + confidence BCE 1.0, lr 6e-4, global batch 512, bf16, torch.compile).
   Scale knobs only: epochs, local batch.
5. **eval** (8×H100): `timed_eval.py` — DeepSpec's evaluator protocol on a task subset
   (gsm8k, math500, humaneval, mbpp, mt-bench, alpaca; max_new_tokens 1024, temp 1.0),
   plus wall-clock timing → spec tok/s, ms/verify-round, and a target-only baseline
   tok/s. `--confidence-threshold > 0` runs the confidence-head pruning mode.
6. **push-hf** (CPU): uploads the cache, regen data, and the full-run drafter checkpoint
   to a public HF dataset repo (`<hf-user>/dspark-proxy-run`). HF public storage is
   free (8.7 TB quota); private would not fit (100 GB cap). Uses the `hf-token` Modal
   secret; the token never leaves Modal. Run this, verify, then delete `/vol/cache`
   so the 611 GB stops costing ~$54/mo on Modal. Ticket #14 downloads from HF instead:
   `uvx modal run ... --stage restore-cache-from-hf` (not implemented yet) or a plain
   `huggingface_hub.snapshot_download` in any function that needs the cache.

## Status log

- smoke split: 700 rows shuffled from open-perfectblend (0 skipped). CPU, ~1 min.
- smoke regen: vLLM Qwen3-4B on 1xH100, 700/700 regenerated, 0 errors; ~5 min vLLM
  boot + ~2:13 generation (concurrency 64).
- smoke cache: 694/700 valid samples (6 dropped by min_loss_tokens=14 filter),
  ~4 min on 1xH100.
- smoke train: completed on 1xH100 (10 steps, ~12 min incl. torch.compile). First
  attempt OOMed at local_batch_size=4 (full-vocab fp32 loss intermediates ~8 GiB per
  micro-batch) — DeepSpec's own guidance applies: keep local_batch_size=1 on 80 GB.
- **Harness validation**: released drafter `deepseek-ai/dspark_qwen3_4b_block7` through
  our timed_eval: gsm8k acceptance_length **6.12** vs paper Table 1 **6.11** — harness
  reproduces the published protocol. Position-wise acceptance 0.93/0.86/0.80/0.73/0.67/
  0.62/0.56 across the 7 draft positions (the suffix-decay curve the seq-head research
  flagged as the vanilla-vs-RNN arbiter). Confidence head: AUC ~0.88, ECE 0.09-0.13,
  overconfident (pred 0.83 vs observed 0.74) — threshold sweep must account for it.
- full split: 19,999 rows shuffled (seed 42), 1 skipped. CPU, ~2 min.
- full regen: 19,999 attempted, 12 errors (19,987 success rows) in **51.7 min** on
  1xH100 (~$3.40) — vLLM Qwen3-4B at concurrency 64, ~8.7 samples/s. This is the
  Qwen3-4B-scale number; the GLM-320B-scale regen is the expensive case, not this.
- full cache: **19,817/19,987 valid** samples (170 dropped by min_loss_tokens=14),
  ~55 min on 1xH100.
- full train: running on 4xH100 — 19,456 samples/epoch (floor to global batch 512),
  380 steps, local_batch_size=2 (4 OOMs; see gotchas), ~5 h projected, ~20 GPU-h.
  Loss 2.64 at step 20/380.
- smoke eval ours (694-sample/10-step drafter): gsm8k tau **1.06**, math500 tau
  **1.06**, ~37 tok/s single-stream — the floor of the data-scale curve; the full
  20k run gives the middle point vs released (1.3M-sample) 6.12/5.82.
- full train: **done** — 380/380 steps in 133.6 min on 4xH100 (~13.4 GPU-h), loss
  2.64 -> 1.79. Checkpoint: home/checkpoints/deepspec/dspark_block7_qwen3_4b_full/step_380.
- released eval (all six tasks, 1xH100): tau 6.12/5.82/5.42/5.09/3.67/3.54 vs Table 1
  6.11/5.70/5.38/5.13/3.64/3.54 — **all within +/-0.12, alpaca exact**. Harness validated.
- baseline (target-only, 8xH100 aggregate / 8): 335.8/356.6/334.2/371.5/291.7/331.5
  aggregate = ~42/45/42/46/37/41 tok/s per stream. Released-drafter speedups vs
  baseline: 5.65x (gsm8k), 5.13x (math500), 5.08x (humaneval), 4.29x (mbpp), 3.94x
  (mt-bench), 3.33x (alpaca). tok/s is ~linear in tau at ~40 tok/s per tau unit on
  1xH100 single-stream; ms/round ~25-29 ms.
- full eval ours (20k-sample drafter, 8xH100): tau 2.47/2.33/1.77/1.85/1.53/1.49
  (gsm8k/math500/humaneval/mbpp/mt-bench/alpaca), 29-54 tok/s per stream = **0.80-1.24x
  vs baseline** — a weak drafter is a net loss on chat/code; only math pays.
- confidence sweeps (8xH100, topology-matched): released gsm8k t=0.5: tau 6.16->6.10
  but tok/s 127.7->154.5 (**+21%**, efficiency win from pruning doomed trailing
  positions); t=0.8 over-prunes (tau 5.47, tok/s +3%); chat tasks: wash at t=0.5,
  harmful at t=0.8. Ours t=0.8: harmful everywhere (tau 1.80, tok/s -14%). The
  single-stream lens understates the mechanism — DSpark's scheduler wins are a
  concurrency/batch-capacity effect (paper Fig 8); phase (d) must measure
  acceptance-vs-concurrency.
- confidence calibration: ours ECE 0.005-0.008, AUC 0.91-0.95 (well-calibrated at low
  absolute survival); released ECE 0.06-0.13, overconfident.

## Sequential-head A/B (#14): RNN head vs vanilla Markov

Same data, cache, epochs, global batch (512), and eval protocol as the full vanilla
run — the sequential head is the only variable (`--opts model.markov_head_type=rnn`
in the `train` stage; DeepSpec builds `RNNHead` from that config key). Micro-batch 2
matches the vanilla full run (batch 4 OOMs for either head — the full-vocab fp32 loss
intermediate, not an RNN-specific cost). RNN adds one `joint_proj` (2r+d)x3r, r=256,
d=2560 and carries recurrent state across the 7 positions of a block.

- restore-hf (new stage): cache + checkpoint back from `phillipchaffee/dspark-proxy-run`,
  611 GB in 60.8 min at 3-way concurrency (~$1). Sequential single-stream took 13.8
  min/shard; 14 parallel streams killed the container's heartbeat twice. Resume works:
  `snapshot_download`-style per-file skip + per-file `volume.commit()`.
- smoke train: 5 optimizer steps (new `max_steps` knob), RNN head builds from the
  config override, checkpoint saves (`..._rnn_smoke/step_5`), ~14 min incl. compile.
- smoke eval (gsm8k:12): evaluator loads the RNN drafter from the checkpoint's saved
  config (`markov_head_type: "rnn"` in `config.json`) — drafting, verification, and
  confidence recording all run; tau 1.01 (untrained floor, as expected).
- full train (4xH100): **380/380 steps in 117.8 min (~$19), loss 2.62 -> 1.84** (vanilla
  full run: 133.6 min, loss 1.79). Checkpoint: `home/checkpoints/deepspec/
  dspark_block7_qwen3_4b_rnn/step_380` (on the Volume; reproducible config-only).
- full eval (8xH100, identical harness to full_ours): ~40 min (~$13).

Results (`results/ab_rnn.json` vs `results/full_ours.json`; released drafter from
`match_released_t00.json` for the tail comparison):

| task | tau vanilla | tau rnn | delta | |
|---|---|---|---|---|
| gsm8k | 2.468 | 2.434 | -0.034 | (-1.4%) |
| math500 | 2.329 | 2.294 | -0.035 | (-1.5%) |
| humaneval | 1.771 | 1.748 | -0.023 | (-1.3%) |
| mbpp | 1.850 | 1.823 | -0.026 | (-1.4%) |
| mt-bench | 1.531 | 1.526 | -0.005 | (-0.3%) |
| alpaca | 1.493 | 1.482 | -0.011 | (-0.7%) |
| mean | | | **-0.022** | |

Verdict against ticket #14's decision-relevant reads:

1. **tau delta vs eval noise (~0.1)**: every task delta is -0.005..-0.035, mean -0.022
   — uniformly slightly negative but well inside the noise bar. Vanilla stays the
   default; the novel-head bar stays high.
2. **Position-wise curves**: the RNN head does NOT lift the tail. Position 1 is
   identical (same backbone); positions 2-7 track vanilla down to slightly steeper
   (gsm8k pos2 0.364 -> 0.358, pos4 0.137 -> 0.131). The ours-vs-released tail gap
   (pos4-7 mass share: released 2.80x pos1 on gsm8k vs ours 0.53/0.49) is a **data-scale
   artifact** (1.3M vs 20k samples), not a head-architecture gap — at matched data the
   richer conditioner buys nothing.
3. **Confidence calibration**: AUC marginally better for RNN (+0.01-0.02 on several
   positions), ECE mixed (worse at pos1 on math-heavy tasks, better at some tails) —
   no decisive win.

Throughput is not decision-grade across these two runs: the pre-fix harness JSONs
carry aggregate fields only (per-stream = aggregate/8 per the gotcha above), and the
runs sat on different days/machines — per-stream tok/s moved vanilla ~51.9 -> rnn ~63.3
on gsm8k, which is machine-level variance, not a head effect. Serving-side throughput
belongs to the #25 campaign anyway.

Cost of the A/B: restore ~$1, smokes ~$3, failed batch-4 train attempt ~$2, full train
~$19, full eval ~$13 — **~$38 all-in**. Cache deleted again post-run (restorable from
HF in ~35 min); `home/checkpoints` keeps both step_380 drafters (~6 GB, <$1/mo).

## Cost and wall-clock (measured)

| Stage | Hardware | Wall | Cost |
|---|---|---|---|
| split (CPU, 700 + 20k) | CPU | ~3 min | ~$1 |
| smoke regen (700) | 1xH100 | ~8 min | ~$0.53 |
| smoke cache | 1xH100 | ~4 min | ~$0.26 |
| smoke train (2 attempts) | 1xH100 | ~15 min | ~$1 |
| full regen (19,999) | 1xH100 | **51.7 min** | **$3.40** |
| full cache (19,817 valid) | 1xH100 | **~55 min** | **$3.62** |
| full train (380 steps) | 4xH100 | **133.6 min** | **$35.20** |
| smoke evals (2x, full tasks, 1xH100) | 1xH100 | ~6 h | ~$24.5 |
| full eval ours + baseline | 8xH100 | ~70 min | ~$38 |
| sweeps + topology matches (5 runs) | 8xH100 | ~100 min | ~$53 |
| **GPU total** | | | **~$160** |

Volume: 611 GB cache + 43 GB checkpoints/results ≈ $54/mo if held. Cache kept for the
A/B ticket (#14 — same data/cache, head is the only variable); delete with
`uvx modal volume rm -r dspark-proxy-run /cache` if #14 slips past ~3 weeks (rebuild:
stage `cache`, ~$4 + 55 min). Grand total ≈ $165 + storage time. After the run: cache
and checkpoints moved to HF (see Storage decision) — storage now ~$0.10/mo.

Storage decision (post-run): the cache, regen data, and the full-run drafter checkpoint
moved to the public HF dataset repo `phillipchaffee/dspark-proxy-run` (free, 8.7 TB
public quota; 614 GB uploaded in 31.8 min at ~235 MB/s via per-file `upload_file`) and
`/vol/cache` plus `/vol/home/checkpoints` were deleted, capping Modal storage spend at
~$0.10/mo (regen data 0.13 GB + results). The 7.4 GB-per-rank optimizer resume states
were not uploaded (not needed once training completes). Ticket #14 fetches the cache
and checkpoint from HF (huggingface_hub is in the deepspec image; ~35 min + ~$1 to
restore the full cache, or ~1 min for the 2.8 GB checkpoint alone).

## Gotchas (all hit, all fixed)

- vLLM >= 0.11 removed `--disable-log-requests`; newer vLLM needs nvcc at engine init
  — build the regen image on a CUDA devel base.
- `python scripts/data/*.py` does not put the repo root on `sys.path` — set
  `PYTHONPATH=/deepspec` or `import deepspec` fails in cache/eval/train.
- DeepSpec refuses a non-empty cache output dir; delete stale dirs between runs.
- Training OOMs at `local_batch_size=4` on 80 GB (full-vocab fp32 loss intermediates,
  ~8 GiB alloc per micro-batch); batch 1-2 is safe. Paper default is 1.
- Trainer auto-resumes from the latest checkpoint under the same `exp_name` — isolate
  runs by `exp_name` (smoke vs full used different names for this reason).
- `$HOME` must be redirected to the Volume (`HOME=/vol/home`) or checkpoints and
  tensorboard logs die with the container.
- Modal GPU concurrency cap on this plan (~10 GPUs) silently queues 8-GPU calls
  behind each other — apps show "0 tasks" while queued; stage 8-GPU runs
  sequentially or in <=1 8-GPU batch at a time.
- `timed_eval.py` pre-fix `ms_per_round` divided wall by TOTAL rounds across ranks
  (8x understated at world=8); fixed runs carry per-stream fields; older JSONs: per-
  stream = aggregate/8 and ms/round x8.
- Volume views can lag commits by a minute+ across containers; verify with
  `modal volume ls` before dependent stages.

## Gotchas

- DeepSpec training asserts the cached dataset ≥ global_batch_size (512) after
  `min_loss_tokens` filtering — smoke split must exceed that.
- Trainer auto-resumes from the latest checkpoint under the same `exp_name`; every run
  gets a distinct `exp_name` to avoid resuming a smoke checkpoint into a full run.
- Trainer's checkpoint/tensorboard dirs derive from `$HOME` at import time; the app
  sets `HOME=/vol/home` so they land on the Volume.
- Full-scale numbers for reference: paper corpus 1.3M samples, ~38 TB hidden-state
  cache for Qwen3-4B, 10 epochs. This proxy downsizes the corpus only; cost model for
  the GLM-scale run lives in research/modal-cost.md.
