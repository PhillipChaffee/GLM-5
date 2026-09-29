#!/usr/bin/env bash
# Runs every PR-blocking gate (see AGENTS.md "Lint and gates"), in parallel.
# Keep this gate list in sync with the CI steps in shell.yml and
# quality.yml. Mutation testing is nightly only, so it is deliberately
# not here.
#
# The gate commands below are opaque strings that run_gates.sh evaluates
# at runtime; shellcheck sees them out of context here.
# shellcheck disable=SC2016,SC2027,SC2086,SC2154

set -u -o pipefail

log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

# Preflight: name any missing binary before the parallel gates run, where
# it would surface only as an exit-127 failure buried in one gate's log.
# The python-side tools come from the gate venv, the rest from brew/npm
# (AGENTS.md "Lint and gates").
for tool in ruff mypy pytest complexipy deptry vulture lint-imports pip python3 awk \
	typos markdownlint-cli2 lychee gitleaks jscpd osv-scanner \
	shellcheck shfmt kcov ast-grep actionlint yamllint git jq; do
	command -v "$tool" >/dev/null 2>&1 || {
		echo "run-gates: FAIL — missing '$tool' on PATH; install it first (see AGENTS.md \"Lint and gates\")" >&2
		exit 1
	}
done

names=()
cmds=()
add() {
	names+=("$1")
	cmds+=("$2")
}
add "lint" "ruff check ."
add "format" "ruff format --check ."
add "types" "mypy ."
add "tests+coverage" "pytest"
add "cognitive" "complexipy ."
add "file-length" "./effective-lines-gate.sh"

add "spell-check" "typos"
add "markdown-lint" 'markdownlint-cli2 "**/*.md"'
add "link-check" "lychee --no-progress ."
add "secret-scan" "gitleaks detect --no-git --redact"
add "duplication" "jscpd"
add "advisories" "osv-scanner scan --lockfile requirements-lock.txt && osv-scanner scan --no-resolve --lockfile requirements.txt"
add "license-check" "osv-scanner scan --lockfile requirements-lock.txt --licenses=MIT,Apache-2.0,ISC,BSD-3-Clause,BSD-2-Clause,MPL-2.0,PSF-2.0,Unicode-3.0,Python-2.0,Unlicense,CC0-1.0,0BSD,Apache-1.1,BSD-3-Clause-Clear,LGPL-3.0-only,BlueOak-1.0.0,CC-BY-3.0"
# The per-file loops accumulate rc: a for loop's exit status is the LAST
# command's, so a clean final file would mask an earlier failure — observed
# live when shfmt flagged a new gate script and the loop still passed.
add "shell-lint" 'rc=0; for sh in $(git ls-files --cached --others --exclude-standard "*.sh" scripts/extract-paper); do shellcheck "$sh" || rc=1; done; test "$rc" -eq 0'
add "shell-format" 'rc=0; for sh in $(git ls-files --cached --others --exclude-standard "*.sh" scripts/extract-paper); do shfmt -d "$sh" || rc=1; done; test "$rc" -eq 0'
add "shell-tests" "./test/run_tests.sh"
add "shell-coverage" "./coverage-gate.sh"
add "shell-file-length" "./shell-effective-lines-gate.sh"
add "shell-doc-header" 'rc=0; for sh in $(git ls-files --cached --others --exclude-standard "*.sh" scripts/extract-paper); do ast-grep scan --rule ast-grep/header-comment.yml "$sh" || rc=1; done; test "$rc" -eq 0'
# The marker pattern is quote-split ("TOD""O") because this runner and
# the extensionless scripts/extract-paper are themselves files the gate
# scans: unsplit, the literal regex bytes here would self-match and the
# gate could never go green. Options come BEFORE the pattern: git parses
# a post-pattern --untracked as a revision (exit 128, verified on git
# 2.50.1), and --no-recurse-submodules neutralizes a local
# submodule.recurse=true that would otherwise reject --untracked. The
# captured exit code makes the gate fail-closed: git grep exits 0 on
# matches, 1 on none, and errors above that — only 1 may pass.
add "shell-todo-policy" 'rc=0; git grep --untracked --no-recurse-submodules -nE "TOD""O|FIX""ME" -- "*.sh" scripts/extract-paper || rc=$?; test "$rc" -eq 1'
add "workflow-yaml-lint" "yamllint ./.github/workflows/*.yml ./.github/ISSUE_TEMPLATE/*.yaml ./.pre-commit-config.yaml"
add "workflow-lint" "actionlint ./.github/workflows/*.yml"
add "pin-digests" "./pin-digest-gate.sh"
add "unused-deps" "deptry ."
add "dead-code" "vulture scripts vulture-allowlist.py"
add "import-layers" "lint-imports"
add "lockfile" "pip install --require-hashes --dry-run -r requirements-lock.txt"
for i in "${!names[@]}"; do
	name="${names[$i]}"
	cmd="${cmds[$i]}"
	(
		if eval "$cmd" >"$log_dir/$name.log" 2>&1; then
			echo "PASS  $name" >"$log_dir/$name.status"
		else
			echo "FAIL  $name" >"$log_dir/$name.status"
			printf '%s\n' "--- $name output ---" >>"$log_dir/failures.log"
			cat "$log_dir/$name.log" >>"$log_dir/failures.log"
		fi
	) &
done
wait

cat "$log_dir"/*.status 2>/dev/null
if [ -f "$log_dir/failures.log" ]; then
	echo "=== failing gate output ==="
	cat "$log_dir/failures.log"
	exit 1
fi
echo "all gates pass"
