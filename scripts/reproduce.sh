#!/bin/bash
# The README result: three seeds, each trained (about 15.5 min on an RTX 4090) and evaluated on all 129,491 clips.
# Train on an otherwise idle GPU: the minutes depend on it.
# Usage: bash scripts/reproduce.sh logs/reproduce
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-logs/reproduce}
for s in 1 2 3; do
  bash scripts/train.sh "$OUT/s$s" --seed "$s"
  bash scripts/eval_full.sh "$OUT/s$s/policy"
done
for s in 1 2 3; do
  echo "seed $s: $(cut -d: -f2 "$OUT/s$s/time.txt") | $(uv run python tools/merge_eval.py "$OUT/s$s/policy")"
done
