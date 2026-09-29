#!/usr/bin/env bash
#
# Coverage gate for a strict shell repository: fails when total line
# coverage from kcov is below 95%.
#
# kcov instruments the suite with bash's debug trap (PS4/BASH_XTRACEFD)
# and writes a coverage.json with a percent_covered value — but it ships
# no fail-under switch (the --limits flag only colorizes HTML), so this
# script is the gate: it parses every report kcov produced and compares
# the minimum against the threshold.
#
# A failing suite fails the gate too — coverage is never reported on an
# unproven suite. The suite runs twice: once plainly for a trustworthy
# exit status (kcov's exit-code propagation for the covered program is
# undocumented), then under kcov.
#
# This is line coverage only: kcov counts executed lines, and no
# branch-coverage mode exists for shell (kcov issue #27, open). The reports
# are kept on both branches: on
# failure for kcov/index.html debugging, and on success so the CI's
# Coveralls upload step can turn the cobertura.xml report into the free
# coverage badge.
#
# Usage: run from the repository root: ./coverage-gate.sh
set -u -o pipefail

readonly required=95
# The out-directory must not be named "coverage": kcov v42 resolves the bare
# name through PATH when wiring its report base, and the coverage.py console
# script (installed beside the gate python by pytest-cov) shadows it — the
# run then dies with "Can't write helper" / "Can't start/attach" / "Can't
# open directory <python-bin>/coverage/" (first observed on ubuntu-24.04 CI;
# any other name runs clean).
readonly out_dir=kcov

if ! command -v jq >/dev/null 2>&1; then
	echo "coverage-gate: FAIL — jq is required to read kcov's coverage.json and is not installed" >&2
	exit 1
fi

if ! ./test/run_tests.sh; then
	echo "coverage-gate: FAIL — test suite failed" >&2
	exit 1
fi

if ! kcov --clean --exclude-path=test --include-path=scripts "${out_dir}" ./test/run_tests.sh; then
	echo "coverage-gate: FAIL — kcov run failed; reports kept at ${out_dir}" >&2
	exit 1
fi

# One coverage.json per covered program; a suite that spawns scripts
# writes their directories too. Fail closed when none was produced, and
# gate on the minimum across reports — the strictest honest number when
# more than one program was instrumented.
min=""
while IFS= read -r report; do
	pct=$(jq -r '.percent_covered' "${report}")
	if [ -z "${pct}" ] || [ "${pct}" = "null" ]; then
		echo "coverage-gate: FAIL — unreadable percent_covered in ${report}" >&2
		exit 1
	fi
	if [ -z "${min}" ] || awk -v a="${pct}" -v b="${min}" 'BEGIN { exit !(a + 0 < b + 0) }'; then
		min="${pct}"
	fi
done < <(find "${out_dir}" -name coverage.json)
if [ -z "${min}" ]; then
	echo "coverage-gate: FAIL — kcov produced no coverage.json under ${out_dir}" >&2
	exit 1
fi

if ! awk -v got="${min}" -v need="${required}" 'BEGIN { exit !(got + 0 >= need + 0) }'; then
	echo "coverage-gate: FAIL — total line coverage is ${min}%, required ${required}%; reports kept at ${out_dir}" >&2
	exit 1
fi

echo "coverage-gate: PASS — total line coverage ${min}% ≥ ${required}% (reports kept at ${out_dir} for the CI upload)"
