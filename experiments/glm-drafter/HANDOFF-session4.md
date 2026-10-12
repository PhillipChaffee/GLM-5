# Handoff: session 4 → next session (GLM-5.3-Flash drafter, map ticket #23)

Written 2026-10-11. Read AGENTS.md's "GLM control tokens: never emit, always mask" section first; masked forms only; sanitize token-bearing reads (HANDOFF.md, parser.py, tokenizer.json, chat_template.jinja) with the documented tr+sed pipeline.

## Session state

- Worktree `~/worktrees/GLM-5/23-glm-drafter-run`, branch `23-glm-drafter-run` @ b4821ca. NOTHING committed or pushed. `git status`: the intentional session-3 set (modified `.jscpd.json`, `experiments/proxy-run/dspark_proxy.py`, `pyproject.toml`, `run-gates.sh`; untracked `experiments/glm-drafter/`) plus session-4 doc additions in the same tree: `experiments/glm-drafter/HANDOFF-session4.md`, `CONTEXT.md` glossary additions, a note in `docs/agents/issue-tracker.md`. No code changed this session.
- Scratch intact: `/tmp/opencode-scratch/GLM-5/opencode-91382/` (DeepSpec clone, venvs, logs) plus session-4 additions: measurement scripts (`measure_lengths.py`, `measure_gen_tokens.py`, `tokdir/`), issue-comment pulls, ticket-creator scripts, briefing/addendum/checkpoint markdown. No GPU spent this session; no Modal apps running.

## What session 4 did (all durable on the tracker — do not re-derive)

- Session-3 briefing ([comment 6104880313](https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6104880313)) + measured-numbers addendum ([6117284786](https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6117284786)).
- The pre-run ticket set (#33–#46: layer decision, ten pre-run issues, resume build, cost plan, review gate) and the execution leaves (#47 bring-up smoke, #48 full run, #49 findings report) — all sub-issues of #23 with native blocked-by edges; #23 is a full umbrella (17 sub-tickets) and closes on #49. Checkpoints: [6117550863](https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6117550863) (tickets), [6117671054](https://github.com/PhillipChaffee/GLM-5/issues/23#issuecomment-6117671054) (execution structure). #23's body carries the live index; map #1's Notes point at it.
- Created the four missing triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`).

## Findings a future session must know

- **The 7-union deviation**: #8/#22 recorded "cache 7, TRAIN the 5-spread [1,12,23,34,44]"; the implemented config and trained checkpoint TRAIN the 7-union — unrecorded deviation, adjudication lives in the layer-decision ticket (#33). The id-44 (post-final-norm) serving blocker exists under every planned variant.
- **Code-verified resume state**: cache build has NO resume (upstream writer fails closed; stage commits only at end) — #11's own resolution required a chunked resumable build; regen resume exists but was never exercised through a real interruption; train auto-resume exists upstream (`discover_latest_checkpoint` + state load) and needs only a drill; `CacheDataset` takes a single dir.
- **Measured (698-row smoke artifacts)**: rendered length mean 853 / p50 277 / p99 7,172 tokens; 4.3% truncated at the 4,096 cap; cache ≈ 62 TB at 1.42M rows (bf16, 7 streams; manifest pins dtype); regen implied 1,033 tok/s aggregate at c64 (~20–40× below the plan's assumed 5–10k tok/s/node); cache ~219 tok/s at batch 2; train ~1.6k tok/s/GPU at micro 2. These are utilization floors — the $45–65k briefing extrapolation is the worst case; the recorded $5–7k envelope is plausibly reachable after the three cheap throughput experiments (regen ramp #39, cache bucketing, train micro-batch #36).
- **Cost caveat**: the data-hunger curve (τ 6.12 at 1.3M vs 2.47 at 20k) makes corpus subsetting the LAST lever — recorded in the cost-optimization ticket (#45).

## Next session's frontier (wayfinder: one ticket per session; claim first, by name in all human-facing text)

- Phillip works the human tickets: the **layer decision (#33)** first — it gates the full cache build — then #34–#43.
- **#44 (resumable cache build + drills, ~$50–70 GPU)** is agent-ready with no blockers: claim via `gh issue edit 44 --add-assignee @me`, implement the slice loop + merge, run the three kill-and-restart drills on the warm volume.
- Everything else (#45–#49) blocks behind those.

## Gotchas learned this session

- Verify sub-issue parenting via **GraphQL** — REST issue objects return `parent: null` while the relationship is real (note added to `docs/agents/issue-tracker.md`).
- `gh label create` before `--add-label`: the triage labels were documented in `docs/agents/triage-labels.md` but absent from the repo.
- Scratch write-size guard: files over ~10k chars are rejected — split writes/edits.
- When scripting `gh` issue creation, write the created-number mapping to disk immediately (this session lost the mapping once to a mid-script crash; hardcoded recovery).
- Tickets #33–#49 all carry self-contained bodies with the facts inline — zoom a ticket instead of re-deriving from scratch.
