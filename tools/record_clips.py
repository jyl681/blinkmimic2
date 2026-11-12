"""Record evaluation rollouts of a few named BONES-SEED clips for rendering (tools/render_gif.py).

Same protocol as tools/evaluate.py. The clips are replicated so that the batch has at least --min_worlds worlds (fewer
than 1024 worlds changes the result); the trajectory of replica 0 is saved, with every replica's failure frame and return.

    python tools/record_clips.py --ckpt logs/run_s1/policy/ckpt_006500.pt --clips flip_360_002__A305 --out traj.npz
"""

import argparse
import json
import os
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--clips", nargs="+", required=True, help="BONES-SEED clip names")
parser.add_argument("--out", required=True, help="output npz")
parser.add_argument("--robot", default="mjlab_g1")
parser.add_argument("--bones_lib", default=None, help="default: the run's (args.json next to the checkpoint)")
parser.add_argument("--min_worlds", type=int, default=2048)
args = parser.parse_args()

import numpy as np
import torch

import blinkmimic2.networks as networks
from blinkmimic2.dr import XiLayout
from blinkmimic2.env import TrackingSim, make_sim
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MAX_STEP_REWARD, MDP
from blinkmimic2.motion import MotionLib
from blinkmimic2.networks import Policy, policy_kwargs
from blinkmimic2.robot import get_robot
from blinkmimic2.rotations import mat_to_quat

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
run_dir = os.path.dirname(os.path.abspath(args.ckpt))
_run_args = json.load(open(os.path.join(run_dir, "args.json"))) if os.path.exists(os.path.join(run_dir, "args.json")) else {}
if not _run_args:  # a downloaded policy file alone: the flags it was trained with are stored inside it
    _run_args = torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"]
clips = list(dict.fromkeys(args.clips))
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
    fh.write("\n".join(clips) + "\n")
    list_path = fh.name
motion = MotionLib.load(robot, kin, dev, None, list_path, args.bones_lib or _run_args.get("bones_lib"), _run_args.get("motion_store", "full"))
os.unlink(list_path)
assert motion.names == clips
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
C = len(clips)
R = -(-args.min_worlds // C)
N = C * R
clip_of = torch.arange(N, device=dev) % C

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
a_ = ck["args"]
widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
            enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
pi.load_state_dict(ck["pi"])
pi.eval()
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
K, Hh = a_["goal_frames"], a_["hist"]

env = TrackingSim(make_sim(robot), N, robot, motion, mdp, dev, eval_mode=True)
with torch.no_grad():
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(N, device=dev), motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    ret = torch.zeros(N, device=dev)
    fail_f = torch.full((N,), -1, dtype=torch.long, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    rec = slice(0, C)  # replica 0

    def snap(alive):
        S = env.state()
        traj["root_p"].append(S[0][rec].cpu()), traj["root_quat"].append(mat_to_quat(S[1][rec]).cpu())
        traj["q"].append(S[4][rec].cpu()), traj["f"].append(env.f[rec].cpu()), traj["alive"].append(alive[rec].cpu())

    while not bool(env.done_eval.all()):
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        snap(alive)
        out = env.step(a)
        new = env.failed_eval & (fail_f < 0)
        fail_f = torch.where(new, env.f - motion.clip_start[clip_of], fail_f)
        ret += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)
    snap(~env.done_eval)  # the terminating state (alive False)

clip_len = motion.clip_len[clip_of]
norm = (ret / ((clip_len - 1).float() * MAX_STEP_REWARD)).view(R, C).cpu().numpy()
ff = fail_f.view(R, C).cpu().numpy()
np.savez_compressed(args.out, names=np.array(clips), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                    ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R[torch.arange(motion.num_frames, device=dev)].cpu().numpy(),
                    ref_q=motion.q[torch.arange(motion.num_frames, device=dev)].cpu().numpy(),
                    clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy(),
                    rep_fail_f=ff, rep_norm=norm, ckpt=np.array(args.ckpt))  # fmt: skip
print(f"{C} clips x {R} replicas = {N} worlds; wrote {args.out}")
for c, n in enumerate(clips):
    L = int(motion.clip_len[c])
    nf = int((ff[:, c] >= 0).sum())
    fr = sorted(set(ff[ff[:, c] >= 0, c].tolist()))
    print(f"  {n:70s} len {L:5d} ({L * 0.02:5.1f}s)  return r0 {norm[0, c]:.4f}  failed in {nf}/{R} replicas"
          + (f" at frames {fr[:6]}" if nf else ""))
