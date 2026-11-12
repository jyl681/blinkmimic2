"""Merge the four shard evaluations of scripts/eval_full.sh: tracking score J (mean normalized return over all
clips) and number of failed clips F.

    python tools/merge_eval.py logs/run_s1/policy
"""
import glob
import json
import sys

A = sys.argv[1]
for ck in sorted(glob.glob(f"{A}/ckpt_*_eval_full0.json")):
    parts = [json.load(open(ck.replace("_eval_full0", f"_eval_full{k}"))) for k in range(4)]
    per_clip = {c: r for p in parts for c, r in p["per_clip"].items()}
    failed = sorted(c for p in parts for c in p["failed"])
    J = sum(p["norm_return"] * len(p["per_clip"]) for p in parts) / len(per_clip)  # (per_clip values are rounded)
    print(f"{ck.split('/')[-1][:11]}: {len(per_clip)} clips, tracking score J = {J:.4f}, failed clips F = {len(failed)}")
    out = {"clips": len(per_clip), "J": J, "F": len(failed), "failed": failed}
    json.dump(out, open(ck.replace("_eval_full0", "_eval_full"), "w"), indent=1)
