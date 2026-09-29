#!/usr/bin/env bash
#
# File-length gate for a strict shell repository: fails when any shell
# file exceeds the maximum effective-line count.
#
# "Effective lines" counts a physical line unless it is blank or its
# first non-whitespace character is "#" — the same definition eslint gives
# TypeScript's max-lines (skipBlankLines + skipComments). One exception,
# shell-specific: inside a heredoc body a "#" is content, not a comment,
# so a state machine tracks open heredocs (a queue, because one command
# can open several) and reclassifies "#" lines inside them as code. The
# machine models bash exactly on the common cases: the body starts on the
# line after the command, only the head of the queue can terminate a
# line, and a "<<-" body's terminator may be indented with tabs. It is
# deliberately line-shaped, and that shape carries a documented
# imprecision (the coverage-gate.sh style): a "<<" inside a string
# literal or an arithmetic shift ("$((x << 2))") is mistaken for a
# heredoc operator, so the following lines count until a line matching
# the fake delimiter (or end of file) — always the conservative
# (higher-count) direction for a maximum. Delimiter parsing follows
# bash: whitespace is allowed between the operator and the word, a
# quoted word may contain spaces, and an unquoted word ends at a shell
# metacharacter; a "<<<" here-string is skipped explicitly;
# backslash-escaped delimiters are parsed as unquoted. A file
# that cannot be read fails the gate: fail closed, never silently.
#
# Scope: every *.sh file and scripts/extract-paper, from the repository root — the same
# surface the lint gate walks. Test files are capped identically: no test
# carve-out. Shell generates no code, so there is no generated-file
# exemption.
#
# Threshold 200: ratified per-language — shell idiom is short files, and
# no tracked .sh in the strictest-setups reference tree was near it when
# the gate shipped. Raise it only with a written reason here, never
# silently. Remedy for a violation: split the script.
#
# Usage: run from the repository root: ./effective-lines-gate.sh
set -u -o pipefail

readonly max_effective_lines=200
# SC2016: the awk program is single-quoted shell text; its "$0" and "$1"
# are awk fields, not shell expansions — nothing here may expand.
# shellcheck disable=SC2016
readonly awk_prog='
function pop_head(    i) {
	for (i = 1; i < np; i++) { d[i] = d[i + 1]; dt[i] = dt[i + 1] }
	np--
}
function scan_heredocs(s,    n, i, c, j, q, delim, tabstrip) {
	n = length(s)
	i = 1
	while (i <= n - 1) {
		if (substr(s, i, 1) == "<" && substr(s, i + 1, 1) == "<") {
			if (substr(s, i + 2, 1) == "<") { i = i + 3; continue }
			j = i + 2
			tabstrip = 0
			if (substr(s, j, 1) == "-") { tabstrip = 1; j++ }
			while (j <= n && (substr(s, j, 1) == " " || substr(s, j, 1) == "\t")) j++
			q = substr(s, j, 1)
			delim = ""
			if (q == "\x27" || q == "\"") {
				j++
				while (j <= n && substr(s, j, 1) != q) { delim = delim substr(s, j, 1); j++ }
				j++
			} else {
				if (q == "\\") j++
				while (j <= n) {
					c = substr(s, j, 1)
					if (c == " " || c == "\t" || c == ";" || c == "&" || c == "|" ||
					    c == "(" || c == ")" || c == "<" || c == ">") break
					delim = delim c
					j++
				}
			}
			if (delim != "") { np++; d[np] = delim; dt[np] = tabstrip }
			i = j
			continue
		}
		i++
	}
}
BEGIN { np = 0 }
{
	line = $0
	if (np > 0) {
		term = line
		if (dt[1]) sub(/^\t+/, "", term)
		if (term == d[1]) {
			pop_head()
			count++
		} else if (line !~ /^[[:space:]]*$/) {
			count++
		}
		next
	}
	if (line ~ /^[[:space:]]*$/) next
	stripped = line
	sub(/^[[:space:]]+/, "", stripped)
	if (substr(stripped, 1, 1) == "#") next
	count++
	scan_heredocs(stripped)
}
END { print count + 0 }
'

status=0
while IFS= read -r file; do
	count=$(awk "$awk_prog" "$file") || {
		echo "effective-lines-gate: FAIL — cannot read $file; fix the tooling, never skip the gate" >&2
		exit 1
	}
	if [ "$count" -gt "$max_effective_lines" ]; then
		echo "$file: $count effective lines exceeds the maximum of $max_effective_lines"
		status=1
	fi
done < <(git ls-files --cached --others --exclude-standard '*.sh' scripts/extract-paper)

if [ "$status" -ne 0 ]; then
	echo "effective-lines-gate: FAIL — the file(s) above exceed the ${max_effective_lines} effective-line maximum" >&2
	exit 1
fi
echo "effective-lines-gate: PASS — no shell file exceeds ${max_effective_lines} effective lines"
