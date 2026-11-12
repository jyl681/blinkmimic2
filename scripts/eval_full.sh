#!/bin/bash
# Evaluate a deployed policy on all 129,491 clips: four shards in parallel (about 3 GB of GPU memory each), then merged.
# Usage: bash scripts/eval_full.sh logs/run_s1/policy
set -euo pipefail
cd "$(dirname "$0")/.."
A=${1:?policy dir}
LIST=${LIST:-data/bones_seed/keep.txt}
mkdir -p "$A/shards"
split -n l/4 -d -a 1 "$LIST" "$A/shards/part_"
for k in 0 1 2 3; do
  uv run python -u tools/evaluate.py --run_dir "$A" --bones_list "$A/shards/part_$k" --tag "full$k" > "$A/eval_full$k.log" 2>&1 &
  sleep 2
done
wait
uv run python tools/merge_eval.py "$A"
