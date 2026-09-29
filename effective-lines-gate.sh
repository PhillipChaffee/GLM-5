#!/usr/bin/env bash
#
# File-length gate for a strict Python repository: fails when any Python
# file exceeds the maximum effective-line count.
#
# "Effective lines" counts a physical line unless it is blank, is a
# whole-line comment, or lies inside a module, class, or function
# docstring — the same definition eslint gives TypeScript's max-lines
# (skipBlankLines + skipComments), with docstrings in the role doc
# comments play for eslint: python documentation lives in docstrings
# (PEP 257) and this stack mandates Google docstrings, so counting them
# would penalize required behavior. The docstring spans come from a
# stdlib `ast` parse and the whole-line-comment classification from
# stdlib `tokenize` COMMENT tokens, so a "#" inside a string literal is
# never mistaken for a comment and a triple-quoted string that is not a
# docstring still counts as code, line by line. A file that does not
# tokenize counts every non-blank line: fail closed, never silently.
#
# Scope: the scan walks every *.py file from the repository root, skipping
# exactly the precursor's EXCLUDED_DIRS (.venv, .git, __pycache__,
# .mypy_cache, .pytest_cache — caches and VCS metadata a gate must never
# measure). Test files are capped identically: no test carve-out (a table
# too big for the cap is data and belongs in a fixture).
#
# Threshold 1000: pylint's max-module-lines and Sonar's per-language file
# defaults are 1000 *raw* lines; 1000 *effective* is stricter than both
# because blanks, whole-line comments, and docstrings drop out. Raise it
# only with a written reason here, never silently. Remedy for a
# violation: split the file by responsibility (PLR0915 stays as the
# per-function size axis).
#
# Usage: run from the repository root: ./effective-lines-gate.sh
set -u -o pipefail

readonly max_effective_lines=1000

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

root="$(pwd -P)"

# The counter is embedded: the gate carries its own tool and adds no
# Python source to the repository it gates — so it cannot trip the
# linters, coverage, vulture, or dependency gates it runs beside. Stdlib
# only; it runs on the interpreter python/ already requires.
cat >"$tmp/effective_lines.py" <<'EOF'
"""Effective-lines counter for the house file-length gate.

A line counts unless it is blank, is a whole-line comment, or lies inside
a module, class, or function docstring; ast supplies the docstring spans
and tokenize's COMMENT tokens supply the whole-line-comment lines, so a
"#" inside a string literal is never mistaken for a comment. Usage:
effective-lines <max> <root>. Exit 0 clean, 1 violations, 2 tooling or
usage error.
"""

import ast
import io
import sys
import tokenize
from pathlib import Path

# Kept from the b7ac4ca precursor: caches and VCS metadata a gate run
# must never measure.
EXCLUDED_DIRS = frozenset(
    {".venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache"}
)

DOCSTRING_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def docstring_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Return (first, last) line pairs for every docstring in the tree."""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_NODES) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.append((first.lineno, first.end_lineno))
    return spans


def comment_only_lines(src: str) -> set[int]:
    """Return the line numbers a COMMENT token occupies alone.

    A comment is whole-line when everything before its start column is
    whitespace; a trailing comment leaves the line counted as code.
    """
    whole: set[int] = set()
    tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in tokens:
        if tok.type == tokenize.COMMENT and not tok.line[: tok.start[1]].strip():
            whole.add(tok.start[0])
    return whole


def effective_lines(
    src: str,
    spans: list[tuple[int, int]],
    comments: set[int],
) -> int:
    """Count the physical lines the definition above keeps."""
    count = 0
    for number, line in enumerate(src.splitlines(), start=1):
        if not line.strip() or number in comments:
            continue
        if any(lo <= number <= hi for lo, hi in spans):
            continue
        count += 1
    return count


def run() -> int:
    """Walk the tree, report every over-cap file, and exit by the verdict."""
    if len(sys.argv) != 3:
        print(
            "usage: effective-lines <max-effective-lines> <root>", file=sys.stderr
        )
        return 2
    try:
        max_lines = int(sys.argv[1])
    except ValueError:
        max_lines = 0
    if max_lines < 1:
        print(
            "effective-lines: <max> must be a positive integer", file=sys.stderr
        )
        return 2
    root = Path(sys.argv[2]).resolve()
    if not root.is_dir():
        print(f"effective-lines: not a directory: {root}", file=sys.stderr)
        return 2
    violations = 0
    for path in sorted(root.rglob("*.py")):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        try:
            src = path.read_text()
        except OSError as err:
            print(f"effective-lines: cannot read {path}: {err}", file=sys.stderr)
            return 2
        try:
            spans = docstring_spans(ast.parse(src))
            comments = comment_only_lines(src)
            count = effective_lines(src, spans, comments)
        except (SyntaxError, tokenize.TokenError):
            # Fail closed: a file that does not tokenize counts every
            # non-blank line — comment and docstring spans are unreliable
            # there.
            count = sum(1 for line in src.splitlines() if line.strip())
        if count > max_lines:
            violations += 1
            print(f"{path}: {count} effective lines exceeds the maximum of {max_lines}")
    if violations > 0:
        print(
            f"effective-lines: FAIL — {violations} file(s) exceed the "
            f"{max_lines} effective-line maximum",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> None:
    """Entry point: run and exit with the verdict."""
    sys.exit(run())


if __name__ == "__main__":
    main()
EOF

python3 "$tmp/effective_lines.py" "$max_effective_lines" "$root" >"$tmp/violations.txt" 2>"$tmp/toolerr.txt"
rc=$?

if [ "$rc" -eq 1 ]; then
	sed 's/^/effective-lines: /' "$tmp/violations.txt" >&2
	cat "$tmp/toolerr.txt" >&2
	exit 1
fi
if [ "$rc" -ne 0 ]; then
	echo "effective-lines-gate: FAIL — the counter could not run; fix the tooling, never skip the gate" >&2
	sed 's/^/  /' "$tmp/toolerr.txt" >&2
	exit 1
fi
echo "effective-lines-gate: PASS — no Python file exceeds ${max_effective_lines} effective lines"
