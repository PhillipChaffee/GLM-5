# Agent notes

Pointers for agent sessions in this repo.

## GLM control tokens: never emit, always mask

This repo's sessions run on the GLM-5.3-Flash serving stack. The model's own
chat-format control tokens — the angle-pipe think-open/think-close tags and the
special-token turn markers (non-ASCII; they render blank in terminal output) —
double as stop sequences for that stack. Emitting any of them in assistant
output (visible text or reasoning — reasoning rides the same completion) makes
the server cut the stream mid-turn: the tool calls landed, the turn dies, and
the log shows a socket-connection error. Several sessions were lost this way
while working the data pipeline, whose files are full of these tokens.

Rules for every session in this repo:

1. **Never write a GLM control-token literal.** Not in replies, not in
   reasoning, not in file content you author, not in comments. Spell it out
   ("the think-open tag", "the assistant-turn end marker") or use the masked
   form: replace the opening angle bracket with `[` — the think-open tag is
   then `[|think|]`.
2. **Never read a token-bearing file raw.** Mask first; this also reveals the
   otherwise-invisible turn markers as `?`:

   `LC_ALL=C tr -c '[:print:]\n\t' '?' < FILE | sed 's/</⟨/g'`

   Pipe any `rg` over such files through the same two filters. Known carriers:
   `experiments/glm-drafter/deepspec_glm/deepspec/data/parser.py`,
   `experiments/glm-drafter/HANDOFF.md`, `tokenizer.json`,
   `chat_template.jinja`, and anything else touching the GLM chat format.
3. **When a file must gain token literals** (a template, a parser entry, a
   script), never type them: copy from an existing carrier with `cp`/`sed`, or
   build the strings in Python from the tokenizer's special-token list.
4. **Keep turns small and checkpoint progress to the tracker** (issue-tracker
   section below) — a dead turn costs only what the last checkpoint didn't
   cover.

## Papers

The `papers/*.txt` sidecars are the readable full text of every paper; the `.pdf` files cannot be rendered by this harness. Search them with `rg`. To add a paper: drop the `.pdf` in `papers/` and run `scripts/extract-paper papers/<name>.pdf` — it writes the NUL-free `.txt` sidecar.

## Glossary

`CONTEXT.md` holds this effort's domain vocabulary; extend it when a term crystallizes (domain-modeling skill).

## Effort map

The DSpark-for-GLM-5.3-Flash wayfinder map is [issue #1](https://github.com/PhillipChaffee/GLM-5/issues/1) on PhillipChaffee/GLM-5; its tickets are that issue's sub-issues.

## Agent skills

### Modal runs

Run long Modal GPU stages detached, from a backgrounded process with a log file:
`nohup uvx modal run --detach experiments/<app>.py --stage <stage> ... > <log> 2>&1 &`,
then tail the log. An ephemeral `modal run` stops its app when the entrypoint
completes, which silently kills any `.spawn()`ed call (0 containers, no result,
no logs); `.remote()` under `--detach` streams logs and survives a dead local
process. Poll progress with the app's result-peek stage — an empty
`modal container list` can also mean an image build is still running. Quick CPU
stages (inspect, fetch) are fine as plain blocking `modal run`. Blackwell GPU
scarcity is real (2026-10-06: B200 unschedulable for hours): prefer the
`B200+:N` spec, which falls back to B300 at B200 rates, and record the
hardware each run actually got. Probe a fresh run with short sleeps (30-60 s)
until it shows first life — a container in `modal container list`, a log line,
or a committed result — and only then lengthen the cadence: a long first sleep
burns wall time on a dead or misconfigured run.

### Issue tracker

GitHub Issues on PhillipChaffee/GLM-5 via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the repo root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.

### Lint and gates

Full-lint gates are adopted in this fork; run every PR-blocking gate locally with `./run-gates.sh` from the repo root. It needs the gate venv on PATH: create one (Python 3.12) with `uv venv --python 3.12 <dir> && uv pip install --python <dir>/bin/python -r requirements-lock.txt && uv pip install --python <dir>/bin/python pip`, then run `PATH="<dir>/bin:$PATH" ./run-gates.sh`. The venv supplies the Python-side tools (ruff, mypy, pytest, complexipy, deptry, vulture, lint-imports); the binary gates also expect these on PATH (Homebrew/npm): shellcheck, shfmt, kcov, ast-grep, gitleaks, typos, markdownlint-cli2, lychee, jscpd, osv-scanner, actionlint, yamllint, jq. The extensionless `scripts/extract-paper` is shell-gated alongside the `*.sh` files: shellcheck, shfmt, kcov coverage ≥ 95%, ast-grep header comment, and the fail-closed TODO-policy grep. `papers/*.txt` is excluded from the typos, lychee, and jscpd scans. Mutation testing runs nightly in `.github/workflows/mutation.yml`, never on the PR path; after a local `mutmut run`, delete the generated `mutants/` directory and the `kcov-out/` report before re-running the gates — some gates scan untracked output. `.pre-commit-config.yaml` carries the quick hooks plus a pre-push guard (`base-current-gate.sh`) that refuses pushes while `origin/main` has moved; activate with `pre-commit install --hook-type pre-push`.
