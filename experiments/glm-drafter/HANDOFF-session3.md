# Handoff: session 3 → next session (GLM-5.3-Flash drafter, map ticket #23)

Written 2026-10-10. Read AGENTS.md's "GLM control tokens: never emit, always mask"
section first; sanitize every read of token-bearing files (HANDOFF.md, parser.py,
tokenizer.json, chat_template.jinja) with the documented tr+sed pipeline. Masked
forms only everywhere: `[|think|]`-style for pipe forms, `[think]`/`[/think]` for
the ASCII forms, `?` for non-ASCII turn markers. Never retype token literals into
files — construct programmatically (`"<" + "think" + ">"` pattern is already used).

## Session state

- Worktree: `~/worktrees/GLM-5/23-glm-drafter-run`, branch `23-glm-drafter-run`,
  fast-forwarded to main @ b4821ca. NOTHING COMMITTED OR PUSHED (rule: never
  without explicit ask). `git status` shows: modified `.jscpd.json`,
  `experiments/proxy-run/dspark_proxy.py`, `pyproject.toml`, `run-gates.sh`;
  untracked `experiments/glm-drafter/`.
- Scratch (intact): `/tmp/opencode-scratch/GLM-5/opencode-91382/` — DeepSpec clone
  @ 005e03b8, venv (transformers 5.19.0/torch 2.14.1 + pytest/openai/tqdm/modal),
  gate venv (28 gates, run `PATH="<gate-venv>/bin:$PATH" ./run-gates.sh` from the
  worktree), `deepspec-combined/` (clone+overlay for tests), tokenizer assets,
  logs of every Modal run, `dspark-compat-report.md` (also copied to
  `experiments/glm-drafter/notes/dspark-compat-report.md`).
- Ticket #23 claimed (do NOT re-claim). Full unit-by-unit checkpoints live as
  comments on #23 (search "Checkpoint (session 3"); recovery must come from
  there, not session memory.

## What is proven (do not re-derive)

- Pipeline code all written + gates green (exit=0, 28 gates): regen script,
  single-process cache builder, Modal app (`experiments/glm-drafter/glm_drafter.py`,
  app `glm-drafter-run`, volume `glm-drafter-run`), config overlay
  (`deepspec_glm/config/dspark/glm5_flash.py`), timed_eval_glm, shared lib
  `experiments/lib/modal_pipeline.py` (both apps rewired onto it),
  `deepspec_glm/deepspec/*` overlay fixes (think-tag registry, RMSNorm import,
  flex-attention dispatch, eval world-1 dist + DynamicCache factory + no-op
  draft cache, trainer subclassing Qwen3DSparkTrainer, snapshot resolution).
- GPU smoke chain GREEN on Modal (hardware recorded per run):
  split 19,999 rows → regen 700 (698 success, 2 context-length 400s; schema
  verified: every assistant turn inline think-open+reasoning+think-close+answer)
  → cache 686/698 valid (canonical shards + manifest on the volume) →
  train-smoke 20 steps, loss 5.95→2.94, checkpoint
  `/vol/home/checkpoints/deepspec/glm_drafter_block7_glm53flash/step_20`
  (model.safetensors + config.json + train_config.py) + step_latest.
- Volume HF cache warm: /vol/home/.cache/huggingface has the full target
  (~30 GB/GPU load in ~2 min on later stages). All deepspec stages carry
  HF_HUB_CACHE + hf-token secret.
- GPU pins: B200:4 train/cache, B200:4 eval (B300 fallback rejected — deepspec
  torch 2.9.1 is cu12; regen image is CUDA 13.0.1). Expect B200 queue waits;
  poll with SHORT sleeps (30-60 s) until first container life, then lengthen.
- Local CPU tests: `experiments/glm-drafter/tests/` — 6 passed, 1 CUDA-skipped
  (flex-attention backward is CPU-unsupported; full-model backward runs on GPU).

## THE PIVOT (decision recorded on #23)

The eval gate goes through sglang's builtin DSPARK speculative decoding
(`--speculative-algorithm DSPARK`), NOT the offline DeepSpec harness. The hybrid
KDA rollback is solved in sglang's public kernels (kda_replayssm_spec_decode,
per-draft-token intermediate SSM states + accept-length commit/rollback, used
for Kimi-K3). The DFlash2 baseline (#12) was measured through this stack
(τ 3.142 mt-bench, 4×B300) — apples-to-apples requires the same path. The offline
harness is a dev tool only; the transformers-5.19 crop-semantics fix there is
optional polish (crop changed from keep-to-N to remove-N in 5.19).

## The compatibility report (read it: experiments/glm-drafter/notes/dspark-compat-report.md)

Bottom line — minimal change list to serve our checkpoint:

1. Config edits on the checkpoint (no retrain): `model_type → "qwen3"` (DFlash2
   precedent), keep `architectures ["Glm5DsparkModel"]`, `head_dim 0 → 64`
   (hard crash otherwise — inherited 0 from the target config), verify
   `intermediate_size` against the trained gate_proj shape (12288 inherited vs
   10240 class default), add `use_cache: false`, `dtype: "bfloat16"`,
   explicit `sample_from_anchor: true`.
2. NO weight remaps — sglang fuses q/k/v→qkv and gate/up→gate_up, skips our
   embed/lm_head for the target's, loads fc/hidden_norm/norm/markov_head/
   confidence_head verbatim.
3. New sglang file `python/sglang/srt/models/glm5_dspark.py` (~40 lines,
   lfm2_dspark.py pattern): Glm5DsparkModel(DSparkDraftModel) + NoPE attention
   subclass (rotary_emb=None, apply_k_rope=identity) + decoder layer with
   attention_cls override + EntryClass.
4. One sglang patch: glm5_next capture cannot deliver the 7th feature (layer-44
   post-final-norm) — patch mirroring Kimi-K3's capture, OR retrain with ids
   ≤42 (the shipped DFlash2 draft uses [5,14,24,33,42] — strong evidence the
   team avoided the last layer on purpose; this would need a retrain decision).
5. Image pin: `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` (main @
   961bcf48, CUDA 13) — NOT v0.5.21 (missing #42489/#42624 DSpark fixes).
6. Flags: `--speculative-algorithm DSPARK --speculative-draft-model-path <ckpt>
   --speculative-dspark-block-size 7 --speculative-draft-attention-backend
   <trtllm_mha|fa4>` (+ GLM recipe flags: dsa backend, fp8 kv, tp 4).
7. UNKNOWNS for the live test are listed in the report (E section).

## Next session's order

1. Read this handoff + the compat report + the #23 checkpoints.
2. Decide (with Phillip if needed): sglang patch for the 7th capture vs
   retrain-with-new-ids (the retrain is cheap-ish: cache rebuild + ~1 h train,
   but the sglang patch is ~10 lines and also fixes future DeepSpec-trained
   drafters).
3. Apply config edits to a COPY of the checkpoint (config.json only) on the
   volume; add the glm5_dspark.py class file (mount/patch into the image or
   SGLANG_EXTERNAL_MODEL_PACKAGE).
4. Serve with the pinned nightly on 4 GPUs; smoke with a handful of prompts;
   watch for the fc-shape ValueError (7th capture) and CUDA-graph capture.
5. Run the #12-style ladder/suite (reuse experiments/dflash2-smoke harness
   patterns) → compare against dflash.json/mtp.json/plain.json.
6. Full-run decision (regen full corpus ~1.42M rows, cache, long train) only
   after the eval path is confirmed live — budget note in #23.

## Gotchas that cost this session time

- GLM control tokens kill the session (AGENTS.md section; masked forms only).
- Modal `add_local_dir` before `run_commands` needs `copy=True`; the shared lib
  needs an explicit `/root/lib` mount; deepspec stages need PYTHONPATH=/deepspec
  (now on the image) and the GLM scripts are mounted at /deepspec/scripts too.
- transformers 5.19: `apply_chat_template(tokenize=True)` returns BatchEncoding
  → `return_dict=False`; `device_map="auto"` needs accelerate + `kernels==0.17.0`
  for the fp8 checkpoint; drafter configs with layer-count-coupled lists must
  override `mlp_layer_types` (5) to pass the strict validator.
- Volume cache: failed runs can still leave files (use `modal volume rm -r`);
  cache attempts OOMed until `logits_to_keep=1` + expandable_segments + batch 2.
- Gates: mirror-tree exemptions are documented in pyproject.toml (ruff/mypy/
  deptry) and run-gates.sh (complexipy --exclude); keep them documented when
  adding files. `experiments/lib/modal_pipeline.py` is the shared pipeline lib —
  add shared behavior THERE, not in either app.
- Scratch logs of every Modal run: /tmp/opencode-scratch/GLM-5/opencode-91382/logs/.
