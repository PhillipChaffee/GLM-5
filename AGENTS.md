# Agent notes

Pointers for agent sessions in this repo.

## Papers

The `papers/*.txt` sidecars are the readable full text of every paper; the `.pdf` files cannot be rendered by this harness. Search them with `rg`. To add a paper: drop the `.pdf` in `papers/` and run `scripts/extract-paper papers/<name>.pdf` — it writes the NUL-free `.txt` sidecar.

## Glossary

`CONTEXT.md` holds this effort's domain vocabulary; extend it when a term crystallizes (domain-modeling skill).

## Effort map

The DSpark-for-GLM-5.3-Flash wayfinder map is [issue #1](https://github.com/PhillipChaffee/GLM-5/issues/1) on PhillipChaffee/GLM-5; its tickets are that issue's sub-issues.

## Agent skills

### Issue tracker

GitHub Issues on PhillipChaffee/GLM-5 via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the repo root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.

### Lint and gates

Full-lint gates are adopted in this fork; run every PR-blocking gate locally with `./run-gates.sh` from the repo root. It needs the gate venv on PATH: create one (Python 3.12) with `uv venv --python 3.12 <dir> && uv pip install --python <dir>/bin/python -r requirements-lock.txt && uv pip install --python <dir>/bin/python pip`, then run `PATH="<dir>/bin:$PATH" ./run-gates.sh`. The venv supplies the Python-side tools (ruff, mypy, pytest, complexipy, deptry, vulture, lint-imports); the binary gates also expect these on PATH (Homebrew/npm): shellcheck, shfmt, kcov, ast-grep, gitleaks, typos, markdownlint-cli2, lychee, jscpd, osv-scanner, actionlint, yamllint, jq. The extensionless `scripts/extract-paper` is shell-gated alongside the `*.sh` files: shellcheck, shfmt, kcov coverage ≥ 95%, ast-grep header comment, and the fail-closed TODO-policy grep. `papers/*.txt` is excluded from the typos, lychee, and jscpd scans. Mutation testing runs nightly in `.github/workflows/mutation.yml`, never on the PR path; after a local `mutmut run`, delete the generated `mutants/` directory and the `kcov-out/` report before re-running the gates — some gates scan untracked output.
