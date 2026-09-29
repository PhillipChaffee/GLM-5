#!/usr/bin/env bash
# Exercise the paper extractor's success and error paths.
set -euo pipefail

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if ./scripts/extract-paper "$tmp/missing.pdf" >"$tmp/error.log" 2>&1; then
	printf 'expected missing PDF to fail\n' >&2
	exit 1
fi

cp papers/DSpark-2607.05147.pdf "$tmp/paper.pdf"
./scripts/extract-paper "$tmp/paper.pdf"
test -s "$tmp/paper.txt"

# The script must also work when called outside the repository root.
repo_root=$PWD
(
	cd "$tmp"
	"$repo_root/scripts/extract-paper" "$tmp/paper.pdf"
)
