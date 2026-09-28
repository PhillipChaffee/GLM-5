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