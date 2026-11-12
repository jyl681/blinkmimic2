#!/bin/bash
# Train one seed and build the deployed policy, the average of the last five checkpoints (one every 500 iterations):
# <run_dir>/policy. With the defaults this is the result in the README (about 15.5 min on an RTX 4090).
# Usage: bash scripts/train.sh logs/run_s1 --seed 1 [more train flags, e.g. ${PRESET[baseline]}]
set -euo pipefail
cd "$(dirname "$0")/.."
R=${1:?run dir}; shift
mkdir -p "$R"
date +%s > "$R/process_start.txt"  # training time = process start -> last checkpoint of the average
uv run python -u -m blinkmimic2.train --log_dir "$R" "$@" 2>&1 | tee "$R/stdout.log"
T=$(python3 -c "import json; print(json.load(open('$R/args.json'))['iterations'])")
rm -rf "$R/policy"
uv run python tools/avg_ckpt.py --run_dir "$R" --ckpts "$(seq -s, $((T - 2000)) 500 $T)" --out_dir "$R/policy"
s=$(( $(stat -c %Y "$R/ckpt_$(printf %06d $T).pt") - $(cat "$R/process_start.txt") ))
echo "training time: $((s / 60)) min $((s % 60)) s from process start" | tee "$R/time.txt"
