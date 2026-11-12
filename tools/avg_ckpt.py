"""Average the deployed (EMA) policy weights of several checkpoints of one run into a new run directory, which the
evaluator reads like any checkpoint. Weights and biases are averaged with equal weights; the observation / goal
normalizers and other buffers are taken from the last checkpoint. The output is named after the last iteration.

    python tools/avg_ckpt.py --run_dir logs/run_s1 --ckpts 4500,5000,5500,6000,6500 --out_dir logs/run_s1/policy
"""
import argparse
import json
import os
import shutil

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--ckpts", required=True)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
its = [int(x) for x in a.ckpts.split(",")]
cks = [torch.load(os.path.join(a.run_dir, f"ckpt_{it:06d}.pt"), map_location="cpu", weights_only=False) for it in its]
last = cks[-1]
keep = lambda k: ("_norm." in k) or not torch.is_floating_point(last["pi"][k])  # noqa: E731
avg = {}
for k, v in last["pi"].items():
    avg[k] = v.clone() if keep(k) else torch.stack([c["pi"][k].float() for c in cks]).mean(0).to(v.dtype)
os.makedirs(a.out_dir, exist_ok=False)
shutil.copy(os.path.join(a.run_dir, "args.json"), os.path.join(a.out_dir, "args.json"))
out = dict(last, pi=avg, averaged_from=its, averaged_run=a.run_dir)
torch.save(out, os.path.join(a.out_dir, f"ckpt_{its[-1]:06d}.pt"))
json.dump({"run_dir": a.run_dir, "ckpts": its, "kept_from_last": [k for k in avg if keep(k)]},
          open(os.path.join(a.out_dir, "avg_info.json"), "w"), indent=1)
print(f"averaged {len(its)} checkpoints -> {a.out_dir}/ckpt_{its[-1]:06d}.pt; kept from last: {[k for k in avg if keep(k)]}")
