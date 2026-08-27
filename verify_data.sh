#!/usr/bin/env bash
# Verify the catalog against the organizer's published SHA-256 checksums.
#
# SHA256SUMS is a RELEASE ASSET, not a file in this repository. bootstrap.sh
# originally guarded this step behind `[ -f SHA256SUMS ]`, which is never true
# after a plain `git clone`, so the check was skipped in silence -- which reads
# exactly like passing. Hence this script, which fails loudly instead.
set -euo pipefail

REPO="https://github.com/TechJam2026/techjam-conversational-search"
SUMS_URL="$REPO/releases/download/participant-kit/SHA256SUMS"
DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"

command -v sha256sum >/dev/null && SHA="sha256sum" || SHA="shasum -a 256"

cd "$DATA_DIR"

[ -f catalog.jsonl.gz ] || { echo "!! data/catalog.jsonl.gz missing — run bootstrap.sh first"; exit 1; }

echo "==> fetching official SHA256SUMS from the release"
curl -sL --fail -o SHA256SUMS.official "$SUMS_URL" || { echo "!! could not download SHA256SUMS"; exit 1; }

echo "==> verifying catalog.jsonl.gz"
if $SHA -c SHA256SUMS.official --ignore-missing; then
  echo "==> checksum OK"
else
  echo "!! CHECKSUM MISMATCH — do not trust any score produced from this catalog"
  rm -f SHA256SUMS.official
  exit 1
fi
rm -f SHA256SUMS.official

ROWS=$(wc -l < catalog.jsonl | tr -d ' ')
echo "==> catalog.jsonl rows: $ROWS (expected 50000)"
[ "$ROWS" = "50000" ] || { echo "!! wrong row count"; exit 1; }

echo "==> decompressed catalog.jsonl sha256 (not covered by the official sums,"
echo "    but every team member should see the same value):"
$SHA catalog.jsonl | sed 's/^/    /'
echo "==> all data checks passed"
