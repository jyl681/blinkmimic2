"""Evaluate checkpoints on a list of BONES-SEED clips: every clip once, from its first frame, in the evaluation-mode
simulator (no randomization, no reset noise, a_prev = 0), with the checkpoint's deployed policy. Return = tracking reward
summed while alive, normalized by the clip length; failure = termination before the clip ends.

A world that finishes its clip is reset onto the next clip of a queue (longest first), so a pass takes about
max(max_len, sum_len / W) steps of W worlds per checkpoint. The simulator is not bitwise reproducible across batch
layouts, so borderline clips can flip between two evaluations of the same checkpoint.

    python tools/evaluate.py --run_dir logs/run_s1/policy --bones_list data/bones_seed/keep.txt --tag full

Results go to <run_dir>/eval_<tag>.csv and ckpt_*_eval_<tag>.json (per-clip returns, failed clips and failure frames).
"""

import argparse
import glob
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--robot", type=str, default="mjlab_g1")
parser.add_argument("--bones_list", type=str, default=None)
parser.add_argument("--bones_lib", type=str, default=None)
parser.add_argument("--tag", type=str, required=True, help="results go to eval_<tag>.csv and ckpt_*_eval_<tag>.json")
parser.add_argument("--ckpts", type=str, default=None, help="comma-separated iterations (default: all)")
parser.add_argument("--group", type=int, default=1, help="checkpoints evaluated together (W worlds each)")
parser.add_argument("--worlds", type=int, default=0,
                    help="worlds per checkpoint (default: sum of clip lengths / max clip length, rounded up to 256)")
parser.add_argument("--danger", action="store_true", help="per clip episode maximum of the danger ratio")
parser.add_argument("--fp32", action="store_true", help="diagnostic: policy in fp32 even if the run trained with --amp bf16")
args = parser.parse_args()

import torch

import blinkmimic2.networks as networks
from blinkmimic2.dr import XiLayout
from blinkmimic2.env import TrackingSim, make_sim
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MAX_STEP_REWARD, MDP
from blinkmimic2.motion import MotionLib
from blinkmimic2.networks import Policy, policy_kwargs
from blinkmimic2.robot import get_robot

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
_run_args = json.load(open(os.path.join(args.run_dir, "args.json"))) if os.path.exists(os.path.join(args.run_dir, "args.json")) else {}
if not _run_args:  # a downloaded policy file alone: the flags it was trained with are stored inside it
    _run_args = torch.load(sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))[0], map_location="cpu", weights_only=False)["args"]
motion = MotionLib.load(robot, kin, dev, _run_args.get("motion_dir"), args.bones_list or _run_args.get("bones_list"),
                         args.bones_lib or _run_args.get("bones_lib"), _run_args.get("motion_store", "full"))
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
C = len(motion.names)
L = motion.clip_len.cpu()
# with fewer than 1024 worlds the same checkpoint scores about 0.001 lower (256 / 512 worlds: 0.9478 / 0.9479
# on pairs_train2000, 1024 / 2000 / 4000 / 32000: 0.9488 / 0.9487 / 0.9487 / 0.9487; not the bf16 policy). Keep W >= 2048
W = args.worlds or max(2048, int(-(-int(L.sum()) // int(L.max())) + 255) // 256 * 256)
W = min(W, C)
assert W >= 1024 or W == C or args.worlds, "fewer than 1024 worlds changes the result"
ORDER = torch.argsort(L, descending=True, stable=True).to(dev)  # longest first


def load_policy(ck):
    a_ = ck["args"]
    widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    return pi.eval()


@torch.no_grad()
def evaluate_group(env: TrackingSim, cks: list[dict]) -> tuple[list[dict], int]:
    M = len(cks)
    N = env.N
    pis = [load_policy(ck) for ck in cks]
    a0 = cks[0]["args"]
    K, Hh = a0["goal_frames"], a0["hist"]
    assert all(ck["args"]["goal_frames"] == K and ck["args"]["hist"] == Hh for ck in cks)
    networks.AMP_DTYPE = torch.bfloat16 if a0.get("amp", "none") == "bf16" and not args.fp32 else None
    blk = torch.arange(N, device=dev) // W  # checkpoint of each world
    # queue position of each world's current clip (per checkpoint block); world j of a block starts with ORDER[j]
    qpos = torch.arange(N, device=dev) % W
    nxt = torch.full((M,), W, dtype=torch.long, device=dev)  # next queue position per block
    cur = ORDER[qpos]  # clip of each world
    active = torch.ones(N, dtype=torch.bool, device=dev)
    ret_c = torch.zeros(M, C, device=dev)
    fail_c = torch.zeros(M, C, dtype=torch.bool, device=dev)
    failf_c = torch.full((M, C), -1, dtype=torch.long, device=dev)
    dmax_c = torch.zeros(M, C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(N, device=dev), motion.clip_start[cur].clone())
    ret = torch.zeros(N, device=dev)
    dmax = torch.zeros(N, device=dev)
    fail_f = torch.full((N,), -1, dtype=torch.long, device=dev)
    a = torch.zeros(N, robot.num_joints, device=dev)
    o = env.observe()
    hist = o.unsqueeze(1).repeat(1, Hh, 1)
    steps = 0
    while bool(active.any()):
        g = motion.goal(env.f, K)
        for m, pi in enumerate(pis):
            sl = slice(m * W, (m + 1) * W)
            a[sl] = pi(hist[sl], g[sl])
        alive = active & ~env.done_eval
        out = env.step(a)
        steps += 1
        if args.danger:
            dmax = torch.where(alive, torch.maximum(dmax, mdp.danger(env.state(), env.f)), dmax)
        new = env.failed_eval & (fail_f < 0) & active
        fail_f = torch.where(new, env.f - motion.clip_start[cur], fail_f)
        ret += torch.where(alive, out["r"], 0.0)
        fin = active & env.done_eval  # episodes that ended this step
        ids = fin.nonzero().flatten()
        if len(ids) > 0:
            b, c = blk[ids], cur[ids]
            ret_c[b, c] = ret[ids]
            fail_c[b, c] = env.failed_eval[ids]
            failf_c[b, c] = fail_f[ids]
            dmax_c[b, c] = dmax[ids]
            # hand out the next clips of each block's queue, in world order
            rank = torch.zeros_like(ids)
            for m in b.unique().tolist():
                sel = (b == m).nonzero().flatten()
                rank[sel] = torch.arange(len(sel), device=dev)
            pos = nxt[b] + rank
            ok = pos < C
            nxt.index_add_(0, b, torch.ones_like(b))
            r_ids, r_pos = ids[ok], pos[ok]
            active[ids[~ok]] = False
            if len(r_ids) > 0:
                cur[r_ids] = ORDER[r_pos]
                env.reset(r_ids, motion.clip_start[cur[r_ids]].clone())
                env.done_eval[r_ids] = False
                env.failed_eval[r_ids] = False
                ret[r_ids] = 0.0
                dmax[r_ids] = 0.0
                fail_f[r_ids] = -1
        env.f = torch.minimum(env.f, motion.clip_end[cur] - 1)  # finished / idle worlds keep stepping; keep f+1 in range
        o = env.observe()
        hist = torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        if len(ids) > 0 and bool(ok.any()):  # a fresh episode starts with its first observation repeated
            hist[r_ids] = o[r_ids].unsqueeze(1).repeat(1, Hh, 1)
    norm = ret_c / ((L.to(dev) - 1).float() * MAX_STEP_REWARD)
    res = []
    for m in range(M):
        failed = fail_c[m]
        fi = failed.nonzero().flatten().tolist()
        res.append({
            "norm_return": norm[m].mean().item(),
            "norm_return_framew": (ret_c[m].sum() / ((L - 1).sum().item() * MAX_STEP_REWARD)).item(),
            "n_failed": len(fi),
            "failed": [motion.names[i] for i in fi],
            "fail_at": {motion.names[i]: [int(failf_c[m, i]), int(L[i])] for i in fi},
            "per_clip": {motion.names[i]: round(norm[m, i].item(), 4) for i in range(C)},
        })  # fmt: skip
        if args.danger:
            res[-1]["per_clip_danger"] = {motion.names[i]: round(dmax_c[m, i].item(), 4) for i in range(C)}
    return res, steps


out_csv = os.path.join(args.run_dir, f"eval_{args.tag}.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
todo = [c for c in sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt"))) if int(os.path.basename(c)[5:11]) not in done]
if args.ckpts:
    todo = [c for c in todo if int(os.path.basename(c)[5:11]) in {int(x) for x in args.ckpts.split(",")}]
print(f"{len(todo)} checkpoints, {C} clips (max len {int(L.max())}, sum {int(L.sum())}), {W} worlds per checkpoint, "
      f"groups of {args.group}", flush=True)  # fmt: skip
MOTION_ID = {"clips": C, "frames": motion.num_frames, "names": motion.names}
env = None
for s in range(0, len(todo), args.group):
    paths = todo[s : s + args.group]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    if env is None or env.N != W * len(paths):
        env = TrackingSim(make_sim(robot), W * len(paths), robot, motion, mdp, dev, eval_mode=True)
    t0 = time.time()
    results, steps = evaluate_group(env, cks)
    for p, ck, res in zip(paths, cks, results):
        it = int(os.path.basename(p)[5:11])
        line = (f"{it},{ck['wall_time'] / 3600:.6f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.6f},"
                f"{res['norm_return_framew']:.6f},{res['n_failed']:g},{'|'.join(res['failed'])}\n")  # fmt: skip
        res["layout"] = {"refill": True, "worlds_per_ckpt": W, "group": [int(os.path.basename(q)[5:11]) for q in paths],
                         "steps": steps, "motion": MOTION_ID}  # fmt: skip
        open(out_csv, "a").write(line)
        json.dump(res, open(p.replace(".pt", f"_eval_{args.tag}.json"), "w"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C}", flush=True)
    print(f"group of {len(paths)} in {time.time() - t0:.0f}s ({steps} steps)", flush=True)
