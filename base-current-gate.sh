#!/usr/bin/env bash
# Pre-push guard: refuses the push when origin/main has commits this
# branch does not. CI checks out the merge of a PR branch with main, so a
# stale base can pass every local gate and still fail CI on code this
# branch never saw (observed on the full-lint adoption branch: green
# locally, four gates red on CI over newly-merged experiments code).
# Remedy: merge origin/main and re-run ./run-gates.sh before pushing.
# Wired through .pre-commit-config.yaml (pre-push stage, which consumes
# the pushed refs itself; the guard therefore runs once per push and
# checks the branch, not individual pushed refs).
set -u -o pipefail

if ! git fetch --no-tags origin main --quiet 2>/dev/null; then
	if git ls-remote --exit-code origin refs/heads/main >/dev/null 2>&1; then
		echo "base-current-gate: FAIL — cannot fetch origin/main; network or remote problem, not a stale base" >&2
		exit 1
	fi
	echo "base-current-gate: PASS — origin has no main branch yet (first push)"
	exit 0
fi

ahead=$(git rev-list --count HEAD..refs/remotes/origin/main 2>&1) || {
	echo "base-current-gate: FAIL — cannot compare HEAD with origin/main: $ahead" >&2
	exit 1
}

if [ "$ahead" -gt 0 ]; then
	echo "base-current-gate: FAIL — origin/main is $ahead commit(s) ahead of this branch; merge origin/main and re-run ./run-gates.sh before pushing" >&2
	exit 1
fi
echo "base-current-gate: PASS — base is current with origin/main"
