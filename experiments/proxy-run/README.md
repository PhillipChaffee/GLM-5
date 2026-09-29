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
stage `cache`, ~$4 + 55 min). Grand total ≈ $165 + storage time.

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
