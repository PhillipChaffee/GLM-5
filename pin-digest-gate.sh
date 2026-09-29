#!/usr/bin/env bash
# Format gate for the sha256-pinned downloads embedded in the CI
# workflows: every digest in an `echo "<digest>  <path>" | sha256sum
# --check` line must be exactly 64 hex characters. A shortened digest is
# malformed at the format level — the class that shipped a 63-character
# ast-grep pin through actionlint, shellcheck, yamllint, and review
# before CI caught it. The digest's VALUE is verified by the download
# itself; this gate owns only the format, and fails closed when the
# embed pattern stops matching the workflows.
set -u -o pipefail

status=0
count=0
while IFS= read -r digest; do
	count=$((count + 1))
	if ! [[ $digest =~ ^[0-9a-fA-F]{64}$ ]]; then
		echo "pin-digest-gate: FAIL — digest '${digest}' is not 64 hex characters (${#digest})" >&2
		status=1
	fi
done < <(awk '/sha256sum/ && match($0, /"[^"]+  [^"]+"/) {
	s = substr($0, RSTART + 1, RLENGTH - 2)
	print substr(s, 1, index(s, "  ") - 1)
}' ./.github/workflows/*.yml)

if [ "$count" -eq 0 ]; then
	echo "pin-digest-gate: FAIL — no embedded digests found; the scan pattern no longer matches the workflows" >&2
	exit 1
fi

if [ "$status" -ne 0 ]; then
	echo "pin-digest-gate: FAIL — fix the malformed digest(s) above" >&2
	exit 1
fi
echo "pin-digest-gate: PASS — ${count} embedded digest(s), all 64 hex characters"
