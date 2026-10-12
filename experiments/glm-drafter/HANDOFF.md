# Handoff: GLM-5.3-Flash drafter training run (map ticket #23)

## Session state

- Repo: `~/git/GLM-5` (PhillipChaffee/GLM-5), main at `87e42a6` (was 5 commits stale, now synced).
- Worktree: `~/worktrees/GLM-5/23-glm-drafter-run`, branch `23-glm-drafter-run` — all new code lives here. Nothing committed or pushed yet (rule: never commit/push without explicit ask).
- Ticket #23 is CLAIMED (assigned via `gh issue edit 23 --add-assignee @me`). Do not re-claim; just continue it.
- Scratch from the prior session (`/tmp/opencode-scratch/GLM-5/opencode-91382/`) held a DeepSpec clone at pinned SHA `005e03b8...` and a venv with transformers 5.19.0 / torch 2.14.1 — recreate if needed, do not rely on it.

## What is decided (do not re-litigate — see tickets #8, #11, #22, #6, #12 on the map)

- Dense 5-layer drafter, block size 7, vanilla Markov head rank 256, confidence head kept + STS calibration.
- Data: full `mlabonne/open-perfectblend` (~1.42M rows), regenerated on-policy by GLM-5.3-Flash at `reasoning_effort=low` + `clear_thinking=True`, DeepSpec sampling defaults (temp 0.7, top_p 0.8, top_k 20), per-turn regeneration conditioned on prior regenerated turns, assistant turns stored as inline `<|think|>…</think>answer` (mgoin/open-perfectblend-glm5.2-regen is the pipeline template).
- Cache: 7-layer union `{1,12,23,34,42,43,44}` captured in one stage-B pass, ~85 TB, 2-week warm hold then delete; durable artifacts = corpus text (~5 GB) + checkpoints on HF.
- Eval gate: six-task DeepSpec suite, beat DFlash2 bar (chat τ ~3.14/3.35, math/code 4.2–5.6); baselines MTP-1 + DFlash2 + native MTP at serving config.
- Budget: run 1 ≈ $5–7k of $15k. GPU spec on Modal: `B200+:4` (B300 fallback at B200 rates). Long stages run detached via `nohup uvx modal run --detach … > log 2>&1 &`, poll with the app's peek/result stage.

## Spike findings (verified, do not rediscover)

1. GLM-5.3-Flash is `Glm5NextForConditionalGeneration` (VL wrapper) with nested `text_config` (model_type `glm5_next_text`). Backbone access: `model.language_model`, mirroring the Gemma4 branch in DeepSpec's cache script.
2. mHC: `hc_mult=4`; each decoder layer returns `(hidden_streams[B,S,4,D], topk_indices)`. Final collapse = unweighted mean over stream dim (GLM's own `HyperHead`), then final RMSNorm. So `last_hidden_state == norm(mean(streams_44, dim=2))`.
3. transformers 5.x `output_hidden_states` tuple: entry `i+1` is layer `i`'s output; the LAST entry is post-final-norm. Cache hooks capture raw layer outputs → the cache writer must fold+norm layer 44 so train and eval agree.
4. Mask token exists: id **154821** = `[MASK]` (reserved, special). Token range 154820–154855 is special tokens; eos ids `[154820, 154827, 154829]`, pad 154820.
5. Chat template: `apply_chat_template(..., reasoning_effort="low", clear_thinking=True)` works; effort values outside {low, high} silently fall back to max. Rendered prompt ends with an open `<|think|>`.
6. DeepSpec's base `run_dataset` hardcodes `enable_thinking=False` — the GLM evaluator overrides it (written).
7. DeepSpec's stock `prepare_target_cache.py` spawns one process per GPU, each loading the FULL model — impossible for a 320B FP8 target. The GLM variant must be single-process with `device_map="auto"` (4×B300) reusing the upstream writer/manifest helpers so the on-disk format stays canonical.

## Worktree contents (branch 23-glm-drafter-run, uncommitted)

Written and edited:

- `experiments/glm-drafter/deepspec_glm/deepspec/modeling/dspark/glm5/modeling.py` — Glm5DsparkConfig + Glm5DsparkModel + attention/MLP/layer classes (dense, NoPE, SDPA path). NOTE: `Glm5DsparkAttention.forward` signature was narrowed (no position_embeddings kwargs) — check callers pass nothing extra.
- `.../glm5/__init__.py` and `.../glm5/hidden_states.py` — fold + extraction helpers incl. `glm_target_hidden_states_from_hooks(captured, layer_ids, final_layer_id, final_norm)`.
- `.../deepspec/eval/dspark/glm5_eval.py` — Glm5DsparkEvaluator (overrides `build_models`, `_init_context`, `_update`, `run_dataset`; registers config with a `_CONFIG_REGISTERED` guard).

Still to write:

- Overlay copies with small patches: `deepspec/trainer/__init__.py` (export Glm5DsparkTrainer), `deepspec/modeling/dspark/__init__.py` (export glm5), `deepspec/utils/constant/public.py` (add `GLM_5_3_FLASH`).
- `deepspec_glm/deepspec/trainer/glm5_trainer.py` EXISTS (has `_WeightHolder`, `resolve_target_io_tensors` reading only embed/lm_head from safetensors shards — verify the exact checkpoint key names against the real index).
- `scripts/data/generate_train_data_glm.py` (completions endpoint, per-turn on-policy loop, resume support).
- `experiments/glm-drafter/glm_drafter.py` (Modal app; mirror dspark_proxy.py stages: probe/split/regen/cache/train/eval-smoke/eval/inspect/fetch/peek/push-hf; GLM constants; app name `glm-drafter-run`, new volume name).
- Local tests: tiny-config drafter forward; parser round-trip vs the real GLM tokenizer; hook-vs-hidden_states equivalence (CPU, tiny model from installed transformers).

## Verification gates before any paid GPU stage

1. Local tests green (template render, loss-mask boundaries, tiny drafter forward).
2. Lint gates on new files (repo uses ./run-gates.sh; needs the gate venv per AGENTS.md).
3. CPU-cheap Modal probe: transformers version in the serve image vs glm5_next availability; AutoConfig/AutoTokenizer load; DFlash2 gated config read via hf-token secret.
4. Only then: smoke regen (700 rows) on 4×B300+, verify output schema, THEN commit to longer stages. Probe first, spend later. Record actual GPU type per run.

## Failure mode observed this session (root-caused from opencode logs)

The stalls were NOT output-limit truncations and no local work was lost. In each case the tool calls completed and the turn died while generating the next assistant message: the SSE stream to the Together API closed mid-turn. Log evidence (`~/.local/share/opencode/log/opencode.log`, 2026-10-09 05:33–06:30 UTC, run `f1e5e7b3`): repeated `AI_APICallError: Cannot connect to API: The socket connection was closed unexpectedly` on both primary and subagent sessions. This is provider/network-side; no opencode config fixes it. Separately, at ~02:03 the same log shows `SQLiteError: database or disk is full` — the disk was full earlier in the night (fine since). Safeguards that actually help:

- Keep turns short: split large file writes into an initial write plus small follow-up edits.
- Emit a line of text between tool-call batches.
- Checkpoint progress into the ticket (comment on #23) after each completed unit, so a fresh session resumes from the tracker, not from memory.
- Prefer several small turns over one mega-turn.

## Suggested skills for the next session

- `wayfinder` (work-through-the-map mode; ticket #23 already claimed — do NOT re-claim, just continue).
- `implement` if the remaining files are written fresh.
- `diagnosing-bugs` if the smoke run fails.
