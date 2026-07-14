#!/usr/bin/env bash
# Classify a Relax sglang.patch against a slime reference patch into:
#   A) identical change content (covered by slime skeleton — no work)
#   B) shared file, Relax modified beyond slime (port the delta)
#   C) Relax-only file (port / may be an upstream-renamed file)
#
# Comparison ignores patch @@ line-number offsets: only real +/- change lines
# are compared, so files that differ only by hunk-header offsets count as identical.
#
# Usage:
#   classify_patch.sh <relax_patch> <slime_patch>
# Example:
#   classify_patch.sh docker/patch/latest/sglang.patch /tmp/slime_skeleton.patch

set -euo pipefail

RELAX="${1:?usage: classify_patch.sh <relax_patch> <slime_patch>}"
SLIME="${2:?usage: classify_patch.sh <relax_patch> <slime_patch>}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/relax" "$work/slime"

# Split a unified diff into one file per touched path (path -> __-joined key).
split_patch() {
  awk -v dir="$2" '
    /^diff --git/ { f=$0; sub(/^diff --git a\//,"",f); sub(/ b\/.*$/,"",f);
                    gsub("/","__",f); cur=dir"/"f; next }
    cur { print > cur }
  ' "$1"
}
split_patch "$RELAX" "$work/relax"
split_patch "$SLIME" "$work/slime"

# Keep only real change lines (drop +++/---/@@/diff/index headers).
norm() { grep -E '^[+-]' "$1" | grep -vE '^(\+\+\+|---)' || true; }

cd "$work"
relax_files=$(ls relax 2>/dev/null | sort || true)
slime_files=$(ls slime 2>/dev/null | sort || true)

echo "Relax files: $(echo "$relax_files" | grep -c . || true) | slime files: $(echo "$slime_files" | grep -c . || true)"
echo

echo "### A. identical change content (slime skeleton already covers — no work) ###"
a=0
for f in $(comm -12 <(echo "$relax_files") <(echo "$slime_files")); do
  if diff -q <(norm "relax/$f") <(norm "slime/$f") >/dev/null 2>&1; then
    a=$((a+1))
  fi
done
echo "count: $a"
echo

echo "### B. shared file, Relax modified beyond slime (port the delta) ###"
for f in $(comm -12 <(echo "$relax_files") <(echo "$slime_files")); do
  if ! diff -q <(norm "relax/$f") <(norm "slime/$f") >/dev/null 2>&1; then
    n=$(diff <(norm "relax/$f") <(norm "slime/$f") | grep -cE '^[<>]' || true)
    printf '  %s  (~%s real diff lines)\n' "$(echo "$f" | sed 's/__/\//g')" "$n"
  fi
done | sort -t'~' -k2 -rn
echo

echo "### C. Relax-only files (port; may be upstream-renamed) ###"
comm -23 <(echo "$relax_files") <(echo "$slime_files") | sed 's/__/\//g' | sed 's/^/  /'
