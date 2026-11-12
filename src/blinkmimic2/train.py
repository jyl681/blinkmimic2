"""BlinkMimic 2 training: SuperMimic's learner (collect -> fit world model -> BPTT policy + critic update) with exams,
frontier practice, a deployment stream and a step size planned for the budget. The defaults are the released recipe."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The defaults are the BlinkMimic 2 recipe (scripts/presets.sh: PRESET[main]); PRESET[baseline] turns the late-phase rules off.
FINAL_PHASE = ("0:exam_frac=0,front_share=0,fail_penalty=0,reset_aprev_ref=0,lr_decay_iters=250000,lr_end=0.03,wstream_frac=0;"
               "2000:exam_frac=0.5,front_share=1,fail_penalty=3,reset_aprev_ref=1,lr_decay_iters=5757,lr_end=0.3;"
               "3000:exam_frac=0.5,lr_decay_iters=6200,wstream_frac=0.25,critic_epochs=1;"
               "4500:exam_frac=0.9")
D = os.path.join(REPO, "data", "bones_seed")
parser = argparse.ArgumentParser()
# ---- run ----
parser.add_argument("--iterations", type=int, default=6500)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--ckpt_every", type=int, default=500, help="deployed (averaged) policy checkpoints")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--bones_lib", type=str, default=os.path.join(D, "lib_all.npz"), help="packed library (tools/prepare_bones.py)")
parser.add_argument("--bones_list", type=str, default=os.path.join(D, "keep.txt"), help="clip names to train on, one per line")
parser.add_argument("--motion_store", type=str, default="fp32", choices=["full", "fp32", "fp16"],
                    help="full: every per-frame quantity precomputed; fp32 / fp16: root pose and joints only, the rest computed when indexed")
parser.add_argument("--robot", type=str, default="mjlab_g1", help="robot package under blinkmimic2/ (mjlab_g1: the G1 as mjlab ships it)")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
# ---- learner (SuperMimic with the BlinkMimic changes) ----
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--buffer_steps", type=int, default=1000, help="replay buffer length (simulator steps)")
parser.add_argument("--warmup_steps", type=int, default=64, help="simulator steps before the first update")
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32, help="imagined steps H")
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8, help="world-model training window (steps)")
parser.add_argument("--wm_batches", type=int, default=5, help="world-model updates per iteration")
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--wm_hidden", type=str, default="1024,1024,1024", help="world-model MLP widths")
parser.add_argument("--wm_mirror", type=float, default=0.5, help="share of each world-model batch replaced by its left-right mirror")
parser.add_argument("--enc_hidden", type=str, default="2048,1024,512,256", help="motion-encoder MLP widths")
parser.add_argument("--dec_hidden", type=str, default="2048,1024,512,256", help="shared-decoder MLP widths")
parser.add_argument("--pi_batch", type=int, default=2048, help="imagined rollouts per policy update")
parser.add_argument("--pi_ref_skip", action=argparse.BooleanOptionalAction, default=True,
                    help="actions on top of the reference pose, a = (q_ref - q0) / alpha + network output")
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--pi_lr_mult", type=float, default=1.6, help="policy learning rate = --lr x this")
parser.add_argument("--lr_decay_iters", type=int, default=250_000, help="the cosine reaches its floor here (also a --phase key)")
parser.add_argument("--lr_end", type=float, default=0.03, help="floor of the cosine, x --lr (also a --phase key)")
parser.add_argument("--pi_beta1", type=float, default=0.9)
parser.add_argument("--pi_clip_value", type=float, default=0.1, help="element-wise policy gradient clip after the norm clip")
parser.add_argument("--pi_ema", type=float, default=0.995, help="EMA of the policy weights; checkpoints store the average")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5, help="also a --phase key")
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--contact_reward", type=float, default=0.25, help="contact-matching term in imagination (evaluation reward unchanged)")
parser.add_argument("--fail_reset", type=float, default=0.3, help="share of practice resets one bin before recent failure points")
# ---- schedule ----
parser.add_argument("--phase", type=str, default=FINAL_PHASE,
                    help="'IT:key=v,...;IT:key=v' sets, from iteration IT on, the keys exam_frac, front_share, fail_penalty, "
                         "start_prac_frac, reset_aprev_ref (0/1), lr_decay_iters, lr_end, wstream_frac, critic_epochs, wm_batches "
                         "(each entry keeps the earlier ones' values). The flags give the capacity: --exam_frac is the largest "
                         "exam share, --fail_penalty / --front_share > 0 and --reset_aprev_ref switch those paths on")
parser.add_argument("--phase_refill", action=argparse.BooleanOptionalAction, default=True,
                    help="when the exam worlds change, collect a full buffer before the updates resume")
parser.add_argument("--share_ramp", type=str, default="0.5,0.9,50,0.1",
                    help="'lo,hi,n0,step': after each pre-exam, exam share = hi - (hi - lo) min(1, n1 / n0) on the step grid, "
                         "n1 = clips the pre-exam predicts to fail; '' = off")
# ---- exams and frontier practice ----
parser.add_argument("--exam_frac", type=float, default=0.9,
                    help="largest share of the simulated robots that play whole clips from their first frame with the EMA "
                         "policy (exams); 0 = off")
parser.add_argument("--exam_mix", type=str, default="0.6,0.2",
                    help="'p,q': shares of the exam clips drawn by margin priority (p), by training failure rate (q) and "
                         "uniformly (rest); '' = uniform")
parser.add_argument("--exam_s0", type=float, default=0.15, help="--exam_mix: noise of one exam's danger ratio")
parser.add_argument("--exam_v", type=float, default=5e-5, help="--exam_mix: margin drift variance per iteration")
parser.add_argument("--exam_every", type=int, default=250, help="exam clip table refresh interval (iterations)")
parser.add_argument("--exam_pre_every", type=int, default=500,
                    help="pre-exams: every this many iterations every exam clip is played in the world model from its first "
                         "frame with the deployed policy; its max danger ratio is a second estimate of the clip's margin; 0 = off")
parser.add_argument("--exam_pre_steps", type=int, default=300, help="imagined steps of a pre-exam")
parser.add_argument("--exam_pre_s", type=float, default=0.2, help="noise of a pre-exam score as a margin estimate")
parser.add_argument("--exam_pre_near", type=float, default=0.3,
                    help="re-score only clips whose margin estimate is >= this, plus a rotating 1/--exam_pre_rot of the rest")
parser.add_argument("--exam_pre_rot", type=int, default=8)
parser.add_argument("--front_share", type=float, default=1.0,
                    help="share of practice resets sent to the frontier of clips that failed (their last --front_win frames "
                         "before the failure to the clip end); 0 = off")
parser.add_argument("--front_win", type=int, default=50)
parser.add_argument("--front_every", type=int, default=50, help="frontier refresh interval (iterations)")
parser.add_argument("--start_prac_frac", type=float, default=0.1,
                    help="share of imagination starts from practice transitions (the rest from exams); < 0 = uniform")
parser.add_argument("--fail_penalty", type=float, default=3.0, help="penalty x soft termination probability per imagined step")
parser.add_argument("--pen_horizon_only", action=argparse.BooleanOptionalAction, default=True,
                    help="the penalty acts inside the imagined horizon only (the critic learns the tracking reward)")
parser.add_argument("--reset_aprev_ref", action=argparse.BooleanOptionalAction, default=True,
                    help="a reset in the middle of a clip sets the previous action to the one that holds the reset pose")
# ---- deployment stream ----
parser.add_argument("--wstream_frac", type=float, default=0.25,
                    help="share of imagination starts taken from robots that run the EMA policy in the world model as the "
                         "evaluation does (first frame, no noise); also a --phase key")
parser.add_argument("--wstream_worlds", type=int, default=8192)
parser.add_argument("--wstream_steps", type=int, default=2, help="stream steps per iteration")
parser.add_argument("--wstream_burn", type=int, default=300, help="stream advances when it starts")
args = parser.parse_args()

import torch
import torch.nn as nn


from blinkmimic2.buffer import ReplayBuffer
from blinkmimic2.dr import XiLayout
from blinkmimic2.env import PHYSICS_DT, TrackingSim, make_sim
from blinkmimic2.kinematics import Kinematics
from blinkmimic2.mdp import MAX_STEP_REWARD, MDP, inv_dim, unflatten_state
from blinkmimic2.motion import CONTROL_DT, MotionLib
import blinkmimic2.networks as networks
import blinkmimic2.rotations as rotations
from blinkmimic2.networks import Critic, Policy, WorldModel, policy_kwargs
from blinkmimic2.mirror import Mirror
from blinkmimic2.objective import actor_objective
from blinkmimic2.mdp import flatten_state
from blinkmimic2.robot import get_robot
from blinkmimic2.sensing import ObsPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

robot = get_robot(args.robot)
J = robot.num_joints
kin = Kinematics(robot, dev)
motion = MotionLib.load(robot, kin, dev, None, args.bones_list, args.bones_lib, args.motion_store)
print(f"motion library: {motion.merge_tail_bins()} clip-end bins merged, {motion.num_bins} bins", flush=True)
layout = XiLayout(robot, dev)
normalize_xi = layout.normalize
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
pipe = ObsPipeline(robot, mdp, layout, dev, PHYSICS_DT)
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = motion.goal_dim(K)
CRITIC_IN = inv_dim(J) + mdp.critic_extra + GOAL_DIM

widths = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
networks.RunningNorm.no_sync = True
rotations.ELEMENTWISE = True

wm = WorldModel(J, layout.dim, use_prev_action=robot.has_action_delay, hidden=widths(args.wm_hidden),
                contact_modes=False, structured=True, dt=CONTROL_DT).to(dev)  # fmt: skip
pi = Policy(mdp.obs_dim, GOAL_DIM, J, hist=Hh, enc_hidden=widths(args.enc_hidden), dec_hidden=widths(args.dec_hidden),
            **policy_kwargs(vars(args), robot)).to(dev)
N_EX = int(round(args.exam_frac * args.num_envs))  # --exam_frac: envs N - N_EX .. N - 1 run exam episodes
PEN = args.fail_penalty
PH_TYPES = {"exam_frac": float, "front_share": float, "fail_penalty": float, "start_prac_frac": float,
            "reset_aprev_ref": lambda v: bool(int(v)), "lr_decay_iters": int, "lr_end": float, "wstream_frac": float, "critic_epochs": int, "wm_batches": int}
PHASES = sorted(((int(e.split(":")[0]), {k: PH_TYPES[k](v) for k, v in (kv.split("=") for kv in e.split(":")[1].split(",") if kv)})
                for e in args.phase.split(";") if e.strip()), key=lambda p: p[0])  # fmt: skip
PHASE = bool(PHASES)  # --phase; off: code path and random stream unchanged
PH_ITS = {p[0] for p in PHASES}
if PHASE:
    PEN = torch.full((), args.fail_penalty, device=dev)


PH_BASE = {k: getattr(args, k) for k in PH_TYPES}  # flag values (phase_apply changes some args fields later)
RAMP = tuple(float(x) for x in args.share_ramp.split(",")) if args.share_ramp else None  # --share_ramp (lo, hi, n0, step)
RAMP_OV = {}  # --share_ramp: the exam_frac it set last (applied over the --phase entries while exams run)


def phase_at(it: int) -> dict:
    """--phase: the settings in force at iteration it (flag values, then every entry with IT <= it in order)."""
    cur = dict(PH_BASE)
    for p_it, ov in PHASES:
        if p_it <= it:
            cur.update(ov)
    if RAMP_OV and cur["exam_frac"] > 0:
        cur.update(RAMP_OV)
    return cur


if PHASE:
    _all = [phase_at(p[0]) for p in PHASES]
    assert N_EX > 0, "--phase: needs --exam_frac > 0"
    assert all(round(c["exam_frac"] * args.num_envs) <= N_EX for c in _all), "--phase: exam_frac above the --exam_frac capacity"
    assert args.start_prac_frac >= 0
    assert args.fail_penalty > 0 or all(c["fail_penalty"] == 0 for c in _all), "--phase fail_penalty > 0 needs --fail_penalty > 0"
    assert args.front_share > 0 or all(c["front_share"] == 0 for c in _all), "--phase front_share > 0 needs --front_share > 0"
    assert args.reset_aprev_ref or not any(c["reset_aprev_ref"] for c in _all), "--phase reset_aprev_ref needs --reset_aprev_ref"
    assert all(c["wstream_frac"] in (0.0, args.wstream_frac) for c in _all), "--phase wstream_frac: 0 or the --wstream_frac value"
assert N_EX == 0 or (args.pi_ema > 0 and not pipe.stateful), "--exam_frac: needs --pi_ema; not with stateful sensing"
assert not args.front_share or (N_EX > 0), \
    "--front_share needs --exam_frac"
critic = Critic(CRITIC_IN).to(dev)
VMAX = (MAX_STEP_REWARD + max(args.contact_reward, 0.0) * 2 * CONTROL_DT) / (1 - args.gamma)  # bound on the critic value (largest per-step reward over 1 - gamma)
VCL = torch.zeros(4, device=dev)  # summed clamped fractions (GAE V, targets, actor bootstrap), count
print(f"critic value bound V_max = {VMAX:.4g}", flush=True)
GRAPH = True
assert not pipe.stateful, "CUDA graphs: not with stateful sensing"
fused = ({"fused": True})
# CUDA graphs: the learning rate is a device tensor (filled every iteration) and Adam keeps its step on the GPU, so the
# captured optimizer steps follow the schedule
adam_kw = lambda: ({"fused": True, "capturable": True, "lr": torch.tensor(args.lr, device=dev)})  # noqa: E731
from blinkmimic2.optim import MuonAdam
opt_wm = MuonAdam(wm, [wm.net], lr=args.lr, betas=(0.9, 0.999), fused=True, capturable=GRAPH)
opt_pi = MuonAdam(pi, [pi.enc, pi.dec], lr=args.lr, betas=(args.pi_beta1, 0.95), fused=True, capturable=GRAPH)
NONFINITE = torch.zeros((), dtype=torch.long, device=dev)  # skipped non-finite policy updates
pi_avg = torch.optim.swa_utils.AveragedModel(pi, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.pi_ema),
                                             use_buffers=True) if args.pi_ema > 0 else None  # fmt: skip
opt_v = torch.optim.Adam(critic.parameters(), **adam_kw())

start_it, wall_offset, sim_steps = 0, 0.0, 0


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((J,), 3e-3), torch.full((J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, a_prev, xi):
    S = unflatten_state(s[:, 0])
    preds, logits = [], []
    for j in range(args.wm_window):
        S, c = wm.step_c(S, a[:, j], a_prev[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
        logits.append(c)
    S_hat = tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)
    return S_hat, torch.stack(logits, 1) if wm.contact_modes else None


FEET = [robot.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")] if robot.name in ("g1", "mjlab_g1") else []
FOOT_CONTACT_Z = 0.045  # ankle-roll link origin below this height: the foot is on the ground (flat foot ~0.035 m)
mdp.feet = FEET
assert args.contact_reward == 0 or FEET, "--contact_reward needs foot bodies (g1)"
motion.set_ref_contact(FEET)


def _contact_bce(logits, S1):
    """BCE of the predicted next-step foot contact against the logged next state: (B, k)."""
    B, k = S1[0].shape[:2]
    bp = kin.forward(*(x.flatten(0, 1) for x in S1))[0]
    on_ground = (bp[:, FEET, 2] < FOOT_CONTACT_Z).float().view(B, k, 2)
    return nn.functional.binary_cross_entropy_with_logits(logits, on_ground, reduction="none").sum(-1)


def _wm_err(S_hat, logits, s1, mask, weight=None):
    std = torch.maximum(wm.inc_norm.std(), ERR_FLOOR)
    S1 = unflatten_state(s1)
    e = (wm.state_error(S_hat, S1) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    loss = e
    if logits is not None:
        loss = loss + 0.0 * _contact_bce(logits, S1)
    if weight is not None:
        loss = loss * weight.unsqueeze(-1)
    return (loss * mask).sum() / s1.shape[0], e[:, -1]


CLIP_STATS = []


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K)


def _imag_step_z(S, o_hist, a_prev, f, xi, xi_n, g, z, g1):
    """``_imag_step`` with the motion latent z = encode(g) and the next goal g1 computed beforehand."""
    a = pi(o_hist, g, z)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, g1


def _imag_step_sensed(S, o_hist, a_prev, f, xi, xi_n, g, st):
    """Imagination step with the robot's stateful signal chain (filters, delay) at the control rate."""
    a = pi(o_hist, g)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    gyro, accel = pipe.imu_from_states(S, S1, CONTROL_DT) if robot.sensing.imu is not None else (None, None)
    st = pipe.filter_update(st, S1[5], gyro, accel, xi, CONTROL_DT, noise=True, ctrl_rate=True)
    o1, st = pipe.observe(st, pipe.raw_obs(st, S1, a, xi, f1), noise=True, delay=True)
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K), st


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev, w_contact=args.contact_reward), mdp.soft_termination(S, f, scale=1.0)


def _gae_targets(V, rews, dones):
    """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
    R = V[-1]
    targets = []
    for t in range(HZ - 1, -1, -1):
        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
        targets.append(R)
    return torch.stack(targets[::-1])


def _critic_loss(X, Y):
    return ((critic(X) - Y) ** 2).mean()


def _wm_norm_data(S0, S1, a, a_prev, xi):
    return wm.inputs(S0, a, a_prev, normalize_xi(xi)), wm.targets(S0, S1), WorldModel.increments(S0, S1)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
imag_step_z = torch.compile(_imag_step_z)
imag_step_sensed = torch.compile(_imag_step_sensed)
reward_term = torch.compile(_reward_term)
critic_input_c = torch.compile(_critic_input)
wm_norm_data_c = torch.compile(_wm_norm_data)
gae_targets_c = torch.compile(_gae_targets)
critic_loss_c = torch.compile(_critic_loss)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * args.lr_end
    t0 = 0
    if it < t0:
        return args.lr
    span = 0.97 if args.lr_end == 0.03 else 1.0 - args.lr_end  # literal kept: default bitwise unchanged
    return args.lr * (args.lr_end + span * 0.5 * (1 + math.cos(math.pi * (it - t0) / (args.lr_decay_iters - t0))))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
mirror = Mirror(robot, layout.slices, dev)


def _mirror_windows(w: dict, n: int) -> dict:
    """Replace the first n windows by their mirror images (states, actions, sim context)."""
    out = dict(w)
    ms = lambda x: flatten_state(mirror.state(unflatten_state(x)))  # noqa: E731
    for k, fn in (("s", ms), ("s1", ms), ("a", mirror.joints), ("a_prev", mirror.joints), ("xi", mirror.xi)):
        out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
    return out


def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    wt = None
    if mirror is not None:
        w = _mirror_windows(w, int(args.wm_batch * args.wm_mirror))
    S_hat, logits = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"])
    loss, e_last = wm_err(S_hat, logits, w["s1"], w["mask"], wt)
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int, mix: bool = True, max_draws: int = 1000):
    """B starts with HZ frames left in their clip; rejection sampling of 2B candidates at a time, at most ``max_draws``
    rounds (an error, not a hang, if the proposal has (almost) no valid start; normally one round suffices)."""
    s, h = buffer.sample_starts(2 * B, Hh, 0.0, 0.5)
    f = s["f"]
    ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
    idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
    s, h = {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}
    return dstart_mix(s, h) if mix else (s, h)


WS = args.wstream_frac > 0  # --wstream_frac: deployment stream
assert not WS or (args.exam_pre_every > 0), "--wstream_frac: needs --exam_pre_every (nominal context)"
WS_ON = torch.zeros((), dtype=torch.bool, device=dev)  # device-side switch of the mixing
WS_ACTIVE, WST = [False], {}  # host-side switch of the advance; the stream's fixed tensors


def ws_mix(s, h):
    """--wstream_frac: the last int(B * frac) starts are replaced by uniform draws of stream worlds (worlds with fewer than H
    frames left in their clip keep the buffer draw)."""
    k = int(len(s["f"]) * args.wstream_frac)
    j = (torch.rand(k, device=dev) * args.wstream_worlds).long().clamp(max=args.wstream_worlds - 1)
    f = WST["s"]["f"][j]
    ok = WS_ON & (f + HZ <= motion.clip_end[motion.clip_id[f]])
    for d, b in ((s, WST["s"]), (h, WST["h"])):
        for key in d:
            v = b[key][j]
            d[key][-k:] = torch.where(ok.view(-1, *([1] * (v.dim() - 1))), v, d[key][-k:])
    return s, h


def dstart_mix(s, h):
    """Imagination starts: the deployment stream replaces a random subset of the buffer draws."""
    if WS:
        s, h = ws_mix(s, h)
    return s, h


CLIP_FRAC = [torch.zeros(())]


OPT = {"batch_goal_enc": False, "graph": GRAPH}  # code paths of the update


@torch.no_grad()
def ema_update():
    """The EMA step of ``AveragedModel.update_parameters`` (same lerp) without its host-side first-call test; the
    host-side counter n_averaged is advanced by the training loop."""
    avg = list(pi_avg.module.parameters()) + list(pi_avg.module.buffers())
    live = [t.detach() for t in list(pi.parameters()) + list(pi.buffers())]
    assert all(t.is_floating_point() for t in avg)
    torch._foreach_lerp_(avg, live, 1 - args.pi_ema)


class Graphed:
    """Run ``fn`` eagerly until ``ready()``; then eagerly on a side stream for ``warmup`` calls, capture it as a CUDA
    graph on the next call and replay it from then on. Each call is one update; outputs are cloned out of the graph pool."""

    def __init__(self, fn, ready, warmup: int = 3):
        self.fn, self.ready, self.warmup, self.calls, self.graph, self.out = fn, ready, warmup, 0, None, None

    def __call__(self):
        if self.graph is None and not self.ready():
            return self.fn()
        self.calls += 1
        if self.calls <= self.warmup:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = self.fn()
            torch.cuda.current_stream().wait_stream(side)
            return out
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # return the eager phase's cached blocks before the graph takes its own pool
            t0 = time.time()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.fn()
            torch.cuda.synchronize()
            print(f"cuda_graph: captured {self.fn.__name__} in {time.time() - t0:.1f} s, "
                  f"{torch.cuda.memory_reserved() / 2**30:.1f} GB reserved", flush=True)  # fmt: skip
        self.graph.replay()
        return tuple(x.clone() for x in self.out)


def policy_update():
    BGE, GR = OPT["batch_goal_enc"], OPT["graph"]
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    if pipe.stateful:  # the observations the policy actually saw, and the sensing state after the last one
        o_hist, st = h["o"], pipe.unpack(s["obs_state"])
    else:
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    if BGE and not pipe.stateful:
        # the goals of an imagined rollout depend only on the reference frames f, f+1, ..., f+H, so they and the motion
        # latents are computed once for the whole horizon: one large GEMM chain instead of H small ones (same gradient)
        fH = f.unsqueeze(0) + torch.arange(HZ + 1, device=dev).unsqueeze(-1)  # (H+1, B)
        G_all = motion.goal(fH.flatten(), K).view(HZ + 1, B, -1)
        Z_all = pi.encode(G_all[:HZ].flatten(0, 1)).view(HZ, B, -1)
    else:
        BGE = False
    for t in range(HZ):
        Aps.append(a_prev)
        if pipe.stateful:
            S, a_prev, o_hist, f, g, st = imag_step_sensed(S, o_hist, a_prev, f, xi, xi_n, g, st)
        elif BGE:
            S, a_prev, o_hist, f, g = imag_step_z(S, o_hist, a_prev, f, xi, xi_n, G_all[t], Z_all[t], G_all[t + 1])
        else:
            S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    r_log = r  # the tracking reward alone (log column imag_r)
    if args.fail_penalty > 0:
        r = r - PEN * d
    rews, dones = (r_log if args.pen_horizon_only else r).detach(), d.detach()

    def critic_fit():
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros(())
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            Vx = critic(X)
            VCL[0].add_((Vx > VMAX).float().mean())
            Vx = Vx.clamp(max=VMAX)
            targets = (gae_targets_c)(Vx, rews, dones)  # (H, B)
            VCL[1].add_((targets > VMAX).float().mean())
            VCL[3].add_(1.0)
            targets = targets.clamp(max=VMAX)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = (critic_loss_c)(Xf[idx], Yf[idx])
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
        return v_loss

    v_boot = critic(critic_input_c(S, f, xi_n, g))
    if v_boot is not None:
        VCL[2].add_((v_boot.detach() > VMAX).float().mean())
        v_boot = v_boot.clamp(max=VMAX)
    # CUDA graphs: survival products as a loop (cumprod's backward reads the host)
    J, c_t, c = actor_objective(r, d, v_boot, args.gamma, 1.0, loop_cumprod=GR)
    r_sum = (c_t * r_log).detach().sum(0)
    loss = -(J).mean()
    opt_pi.zero_grad(set_to_none=True)
    CLIP_STATS.clear()
    loss.backward()
    if CLIP_STATS:  # share of live samples whose adjoint was scaled down, over all hooked tensors and imagined steps
        CLIP_FRAC[0] = torch.stack([a for a, _ in CLIP_STATS]).sum() / torch.stack([b for _, b in CLIP_STATS]).sum().clamp(min=1)
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), args.pi_clip_value)
    finite = torch.isfinite(gn)
    NONFINITE.add_((~finite).long())
    for p in pi.parameters():
        if p.grad is not None:
            p.grad.copy_(torch.where(finite, p.grad, 0.0))
    if isinstance(opt_pi, torch.optim.Optimizer) and not opt_pi.defaults.get("fused"):
        opt_pi.step()
    elif isinstance(opt_pi, torch.optim.Optimizer):
        opt_pi.found_inf = (~finite).float()  # the fused kernel skips the step and undoes the step count
        try:
            opt_pi.step()
        finally:
            opt_pi.found_inf = None
    else:
        opt_pi.step(ok=finite)
    if pi_avg is not None:
        if GR and int(pi_avg.n_averaged) > 0:
            ema_update()
        else:
            pi_avg.update_parameters(pi)
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    v_loss = critic_fit()
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(robot)
mdp.reward, mdp.termination, mdp.obs = torch.compile(mdp.reward), torch.compile(mdp.termination), torch.compile(mdp.obs)
motion.goal = torch.compile(motion.goal)
env = TrackingSim(sim_cfg, args.num_envs, robot, motion, mdp, dev, gamma=args.gamma, fail_reset=args.fail_reset, bin_fix=True)
env.aprev_ref = args.reset_aprev_ref
if PHASE:
    env.aprev_gate = torch.ones((), dtype=torch.bool, device=dev)
env.alloc_uniform = 'legacy' in ("e", "u")
if N_EX:
    _idx = {n: i for i, n in enumerate(motion.names)}
    EX_CLIPS = (torch.tensor([_idx[ln.strip()] for ln in open(None) if ln.strip() in _idx], device=dev)
                if None else torch.arange(len(motion.names), device=dev))  # fmt: skip
    env.set_exam(N_EX, EX_CLIPS)
    EX_MASK = torch.zeros(len(motion.names), dtype=torch.bool, device=dev)
    EX_MASK[EX_CLIPS] = True
    EX_D = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam's max danger ratio per clip
    EX_T = torch.zeros(len(motion.names), device=dev)  # iteration of the last exam
    EX_K = torch.zeros(len(motion.names), device=dev)  # current failure streak
    EX_RUN = torch.zeros(N_EX, device=dev)  # running max danger of each exam world's episode
    EX_P = torch.zeros(len(motion.names), device=dev)  # exams of each clip in flight (diminishing returns per clip)
    EX_SRC = torch.zeros(3, device=dev)  # exam episodes started by source (priority, training failure, uniform)
    IT_T = torch.full((), float(start_it), device=dev)  # the current iteration (filled at the top of each)
    EX_PD = torch.full((len(motion.names),), float("nan"), device=dev)  # --exam_pre_every: last WM pre-exam score
    EX_PT = torch.zeros(len(motion.names), device=dev)  # iteration of the last pre-exam
    EX_FF = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam: failure frame, -1 = passed
    EX_PF = torch.full((len(motion.names),), float("nan"), device=dev)  # last pre-exam: imagined failure frame, -1 = none
    EX_KF = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam: max-danger (or failure) frame
    EX_PK = torch.full((len(motion.names),), float("nan"), device=dev)  # last pre-exam: max-danger frame
    EX_RUNF = torch.zeros(N_EX, dtype=torch.long, device=dev)  # frame of the running max danger of each exam world
    EX_WPI = torch.full((N_EX,), 1.0 / len(motion.names), device=dev)  # inclusion probability of each exam world's clip
    EX_IW = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam: 1 / its inclusion probability


    EX_PIC = torch.full((len(motion.names) + 1,), 1.0 / len(motion.names), device=dev)  # per clip, the inclusion probability of its last draw (+ a dump slot)
    if args.exam_mix:
        _p, _q = (float(x) for x in args.exam_mix.split(","))
        EX_MIXW = torch.tensor([_p, _q, max(0.0, 1.0 - _p - _q)], device=dev)

        def exam_fused():
            """The margin estimate per clip: D, its sd, and whether the clip has any measurement."""
            now = IT_T
            sd = torch.sqrt(args.exam_s0**2 + args.exam_v * (now - EX_T).clamp(min=0))
            if args.exam_pre_every:  # two measurements of the margin, each drifting with its age
                sp = torch.sqrt(args.exam_pre_s**2 + args.exam_v * (now - EX_PT).clamp(min=0))
                pr = torch.where(torch.isfinite(EX_D), sd**-2, 0.0)
                pp = torch.where(torch.isfinite(EX_PD), sp**-2, 0.0)
                prec = (pr + pp).clamp(min=1e-12)
                D = (pr * torch.nan_to_num(EX_D) + pp * torch.nan_to_num(EX_PD)) / prec
                sd = prec.rsqrt()
                seen = ((pr + pp) > 0) & EX_MASK
            else:
                D = torch.nan_to_num(EX_D)
                seen = torch.isfinite(EX_D) & EX_MASK
            return D, sd, seen

        def exam_weights():
            """--exam_mix: priority weights wp and training-failure weights prior over all clips (0: not drawable)."""
            now = IT_T
            D, sd, seen = exam_fused()
            z = (1 - D) / sd
            # one exam per clip in flight (its result, which updates the table, is not back yet); clips more than 3 sd
            # from the boundary are left to the other sources
            wp = torch.where(seen & (z <= 3) & (EX_P <= 0), torch.exp(-0.5 * z**2) / sd, 0.0)
            prior = torch.zeros(len(motion.names), device=dev).scatter_reduce_(0, motion.bin_clip, env.bin_fail, "amax")
            prior = prior * EX_MASK
            return wp, prior

        def exam_sample(n):
            """--exam_mix: n exam clips from the priority / training-failure / uniform mixture (a source without
            weight falls back to uniform)."""
            src = torch.multinomial(EX_MIXW, n, replacement=True)
            c = EX_CLIPS[torch.randint(0, len(EX_CLIPS), (n,), device=dev)]
            wp, prior = exam_weights()
            pr_s = prior
            # priority draws without replacement (one per clip); its unfilled draws go to the training-failure source,
            # then to uniform
            m0 = (src == 0).nonzero(as_tuple=False).squeeze(-1)
            k0 = min(len(m0), int((wp > 0).sum()))
            if k0 > 0:
                c[m0[:k0]] = torch.multinomial(wp, k0, replacement=False)
                src[m0[k0:]] = 1
            else:
                src[m0] = 1
            prior = prior * (EX_P <= 0)  # one exam per clip in flight here too
            prior[c[m0[:k0]]] = 0
            m1 = (src == 1).nonzero(as_tuple=False).squeeze(-1)
            k1 = min(len(m1), int((prior > 0).sum()))
            if k1 > 0:
                c[m1[:k1]] = torch.multinomial(prior, k1, replacement=False)
            src[m1[k1:]] = 2
            EX_SRC.add_(torch.bincount(src, minlength=3).float())
            EX_P.index_add_(0, c, torch.ones_like(c, dtype=EX_P.dtype))
            # per clip, the probability of one draw (as exam_sample_masked)
            EX_PIC[c] = (EX_MIXW[0] * wp[c] / wp.sum().clamp(min=1e-30) + EX_MIXW[1] * pr_s[c] / pr_s.sum().clamp(min=1e-30)
                         + EX_MIXW[2] / len(EX_CLIPS)).clamp(min=1e-12)  # fmt: skip
            return c

        def _gumbel_top(w, k):
            """k draws without replacement with probability proportional to w (Gumbel top-k, the same distribution as
            torch.multinomial without replacement, no host sync): clip ids, and whether each draw exists (w > 0)."""
            u = torch.rand_like(w).clamp(min=1e-20)
            key = torch.where(w > 0, torch.log(w.clamp(min=1e-30)) - torch.log(-torch.log(u)), float("-inf"))
            v, i = torch.topk(key, k)
            return i, torch.isfinite(v)


        def exam_sample_masked(dn):
            """exam_sample for env.reset_masked (capturable): a clip for every exam world; only the worlds in dn (being
            reset) count. The priority draws (without replacement) go to the reset worlds of that source in order;
            draws that do not exist fall to the training-failure source, then to uniform, as exam_sample."""
            n = N_EX
            src = torch.multinomial(EX_MIXW, n, replacement=True)
            c = EX_CLIPS[(torch.rand(n, device=dev) * len(EX_CLIPS)).long().clamp(max=len(EX_CLIPS) - 1)]
            wp, prior = exam_weights()
            wp_s, pr_s = wp, prior
            i0, ok0 = _gumbel_top(wp, n)
            m0 = dn & (src == 0)
            r0 = (torch.cumsum(m0.long(), 0) - 1).clamp(min=0)
            use0 = m0 & ok0[r0]
            c = torch.where(use0, i0[r0], c)
            src = torch.where(m0 & ~use0, 1, src)
            prior = prior * (EX_P <= 0)
            prior = prior.scatter(0, c, torch.where(use0, 0.0, prior[c]))  # a clip drawn by priority is not drawn again
            i1, ok1 = _gumbel_top(prior, n)
            m1 = dn & (src == 1)
            r1 = (torch.cumsum(m1.long(), 0) - 1).clamp(min=0)
            use1 = m1 & ok1[r1]
            c = torch.where(use1, i1[r1], c)
            src = torch.where(m1 & ~use1, 2, src)
            EX_SRC.add_((torch.where(dn, src, 3).unsqueeze(1) == torch.arange(3, device=dev)).sum(0).float())  # no bincount (host sync)
            EX_P.index_add_(0, c, dn.to(EX_P.dtype))
            # probability that one exam draw picks clip c (the source mixture; draws without replacement and the
            # fallbacks ignored)
            pi = (EX_MIXW[0] * wp_s[c] / wp_s.sum().clamp(min=1e-30) + EX_MIXW[1] * pr_s[c] / pr_s.sum().clamp(min=1e-30)
                  + EX_MIXW[2] / len(EX_CLIPS))  # fmt: skip
            EX_WPI.copy_(torch.where(dn, pi.clamp(min=1e-12), EX_WPI))
            EX_PIC.index_put_((torch.where(dn, c, len(motion.names)),), pi.clamp(min=1e-12))
            return c

        env.exam_sampler = exam_sample
        env.exam_sampler_masked = exam_sample_masked
    EX_DIAG = None
    EXAM_NOISE = args.action_noise
    EX_FLIP = torch.tensor([n.startswith("flip_") for n in motion.names], device=dev)
    EX_ENDS, EX_FAILS = torch.zeros((), device=dev), torch.zeros(len(motion.names), device=dev)  # since the last log
    if args.exam_pre_every:
        assert args.exam_mix, "--exam_pre_every feeds the --exam_mix priority"
        _e1 = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev, eval_mode=True)  # only for the nominal context
        EX_XI1, EX_XIN1 = _e1.xi[:1].clone(), layout.normalize(_e1.xi)[:1].clone()
        del _e1
        _g = torch.Generator().manual_seed(args.seed + 7919)  # own stream: the training random stream is unchanged
        EX_ROTK, EX_PRE_N = torch.randperm(len(motion.names), generator=_g).to(dev), [0]

        @torch.no_grad()
        def exam_preexam(it):
            """--exam_pre_every: every exam clip deployed in the world model as the evaluation runs it;
            EX_PD = the imagined episode's max danger ratio."""
            torch.cuda.synchronize()
            t0 = time.time()
            pol = pi_avg.module
            todo = EX_CLIPS
            if args.exam_pre_near > 0 and bool(torch.isfinite(EX_PD[EX_CLIPS]).all()):
                # the fused estimate as exam_sample forms it; clips below the band only in their rotation slice
                now = float(it)
                sd = torch.sqrt(args.exam_s0**2 + args.exam_v * (now - EX_T).clamp(min=0))
                sp = torch.sqrt(args.exam_pre_s**2 + args.exam_v * (now - EX_PT).clamp(min=0))
                pr = torch.where(torch.isfinite(EX_D), sd**-2, 0.0)
                pp = sp**-2
                D = (pr * torch.nan_to_num(EX_D) + pp * EX_PD) / (pr + pp)
                near = D[EX_CLIPS] >= args.exam_pre_near
                rot = (EX_ROTK[EX_CLIPS] % args.exam_pre_rot) == (EX_PRE_N[0] % args.exam_pre_rot)
                todo = EX_CLIPS[near | rot]
            EX_PRE_N[0] += 1
            for b0 in range(0, len(todo), 16384):
                c = todo[b0 : b0 + 16384]
                n = len(c)
                f, end = motion.clip_start[c].clone(), motion.clip_end[c]
                S = tuple(x.clone() for x in motion.ref_state(f))
                a_prev = torch.zeros(n, J, device=dev)
                xi, xin = EX_XI1.expand(n, -1), EX_XIN1.expand(n, -1)
                hist = mdp.obs(S, a_prev, xi, f).unsqueeze(1).repeat(1, Hh, 1)
                dmax = torch.zeros(n, device=dev)
                alive = torch.ones(n, dtype=torch.bool, device=dev)
                ff = torch.full((n,), -1, dtype=torch.long, device=dev)
                pk = f.clone()
                for t in range(args.exam_pre_steps):
                    a = pol(hist, motion.goal(f, K))
                    S = wm.step(S, a, a_prev, xin)
                    f1 = torch.minimum(f + 1, end)
                    d = mdp.danger(S, f1)
                    pk = torch.where(alive & (d > dmax), f1, pk)
                    dmax = torch.where(alive, torch.maximum(dmax, d), dmax)
                    ff = torch.where(alive & (d >= 1), f1, ff)
                    alive &= (f1 < end) & (d < 1)
                    if t % 50 == 49 and not bool(alive.any()):
                        break
                    f = torch.minimum(f1, end - 1)
                    hist = torch.cat((hist[:, 1:], mdp.obs(S, a, xi, f).unsqueeze(1)), 1)
                    a_prev = a
                EX_PD[c] = dmax
                EX_PF[c] = ff.float()
                EX_PK[c] = pk.float()
                EX_PT[c] = float(it)
            torch.cuda.synchronize()
            p = EX_PD[EX_CLIPS]
            dg = (f" | diag list >= 0.5 {int((EX_PD[EX_DIAG] >= 0.5).sum())}/{int(EX_DIAG.sum())}"
                  if EX_DIAG is not None else "")  # fmt: skip
            print(f"--exam_pre it {it}: {len(todo)}/{len(EX_CLIPS)} clips, {args.exam_pre_steps} steps, {time.time() - t0:.1f} s | "
                  f"score >= 1 {int((p >= 1).sum())}, 0.8-1 {int(((p >= 0.8) & (p < 1)).sum())}, "
                  f"0.5-0.8 {int(((p >= 0.5) & (p < 0.8)).sum())}" + dg, flush=True)  # fmt: skip

        if not PHASE or phase_at(start_it)["exam_frac"] > 0:
            exam_preexam(start_it)
    print(f"--exam: {N_EX} exam worlds over {len(EX_CLIPS)} clips, noise {EXAM_NOISE}"
          + (f", bank from their failures (window {48}, last {args.exam_every} its, rebuilt every "
             f"{50} its, share {0.0})" if False else ""), flush=True)  # fmt: skip
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, J, layout.dim,
                      mdp.obs_dim if pipe.stateful else None, pipe.state_dim if pipe.stateful else None, danger=False)  # fmt: skip
obs_hist = torch.zeros(args.num_envs, Hh, mdp.obs_dim, device=dev)
if args.start_prac_frac >= 0:
    assert 0 < N_EX < args.num_envs, "--start_prac_frac needs --exam_frac in (0, 1)"
    buffer.start_split = (args.num_envs - N_EX, args.start_prac_frac)
    if PHASE:  # device scalars, set in place by phase_apply (the captured policy update follows them)
        SPLIT_N0, SPLIT_P = torch.full((), args.num_envs - N_EX, device=dev), torch.full((), args.start_prac_frac, device=dev)
        buffer.start_split = (SPLIT_N0, SPLIT_P)


@torch.no_grad()
def exam_record(out):
    """Count the exam episodes' ends and failures per clip and update the clip table."""
    global EX_ENDS
    n = N_EX
    term, f, done_ex = out["term"][-n:], out["f"][-n:], out["done"][-n:]
    if PHASE:  # only the exam slots inside the current share
        term, done_ex = term & env.exam[-n:], done_ex & env.exam[-n:]
    EX_ENDS += done_ex.float().sum()
    EX_FAILS.index_add_(0, motion.clip_id[f], term.float())
    if args.exam_mix:  # the clip table: last exam's max danger ratio (the exam measures the margin)
        rho = mdp.danger(unflatten_state(out["s"][-n:]), f)  # pre-step state (s1 at a clip end is scored against the next clip); a failure counts via term
        new = (out["ep_step"][-n:] == 0) | (rho > EX_RUN)
        EX_RUNF.copy_(torch.where(new, f, EX_RUNF))
        EX_RUN.copy_(torch.where(out["ep_step"][-n:] == 0, rho, torch.maximum(EX_RUN, rho)))
        dn = done_ex
        cl = motion.clip_id[f]
        k = torch.where(term, EX_K[cl] + 1, torch.zeros_like(EX_K[cl]))
        d = torch.where(term, 1.0 + 0.1 * (k - 1), EX_RUN.clamp(max=0.999))
        now = IT_T
        EX_P.index_add_(0, cl, -dn.float())
        idx = torch.where(dn, cl, len(motion.names))  # dropped: worlds whose episode goes on
        for tab, val in ((EX_D, d), (EX_K, k), (EX_T, now.expand_as(d)), (EX_FF, torch.where(term, f.float(), -1.0)),
                         (EX_KF, torch.where(term, f, EX_RUNF).float()), (EX_IW, 1.0 / EX_WPI)):
            tab.copy_(torch.cat((tab, tab[:1])).index_put_((idx,), val)[:-1])
    return


@torch.no_grad()
def exam_refresh(it):
    """Every --exam_every iterations: log the exam episodes."""
    global EX_ENDS
    n_ok = 0
    if (it - start_it) % args.exam_every:
        return
    if WS and WS_ACTIVE[0]:
        print(f"--wstream it {it}: {int(WST['ends'])} episode ends, {int(WST['fails'])} imagined failures since the last log", flush=True)
        WST["ends"].zero_(), WST["fails"].zero_()
    fc = EX_FAILS > 0
    if args.exam_mix:
        seen = torch.isfinite(EX_D)
        s = EX_SRC / EX_SRC.sum().clamp(min=1)
        extra_mix = (f" | table: {int(seen.sum())} clips examined, {int((EX_D >= 1).sum())} failed last time, "
                     f"{int(((EX_D >= 0.8) & (EX_D < 1)).sum())} at 0.8-1 | sources {s[0]:.2f} / {s[1]:.2f} / {s[2]:.2f}")
        EX_SRC.zero_()
    else:
        extra_mix = ""
    gap = ""
    print(f"--exam it {it}: {float(EX_ENDS):.0f} episodes, {float(EX_FAILS.sum()):.0f} failures on {int(fc.sum())} clips "
          f"({int((fc & ~EX_FLIP).sum())} non-flip, {int((fc & EX_FLIP).sum())} flip)"
          + ("") + extra_mix, flush=True)  # fmt: skip
    EX_ENDS.zero_(), EX_FAILS.zero_()


@torch.no_grad()
def front_refresh(it):
    """--front_share: reset focus on each failing clip's frontier region (equal mass per clip, uniform over its frames)."""
    F = torch.where(torch.isfinite(EX_FF), EX_FF, torch.nan_to_num(EX_PF, nan=-1.0))  # real exam first
    Fb = F[motion.bin_clip]
    lo = torch.maximum(motion.bin_start.float(), Fb - args.front_win)
    w = torch.where(Fb >= 0, (motion.bin_end.float() - lo).clamp(min=0), 0.0)
    per = torch.zeros(len(motion.names), device=dev).index_add_(0, motion.bin_clip, w)
    q = w / per[motion.bin_clip].clamp(min=1e-9)
    n_real, n_pre = int((torch.isfinite(EX_FF) & (EX_FF >= 0)).sum()), int((~torch.isfinite(EX_FF) & (EX_PF >= 0)).sum())
    on = float(q.sum()) > 0
    env.set_focus(q / q.sum() if on else torch.full_like(q, 1.0 / len(q)), args.front_share if on else 0.0)
    if (it - start_it) % args.exam_every == 0:
        print(f"--front it {it}: {n_real + n_pre} frontier clips ({n_real} from exams, {n_pre} from pre-exams), "
              f"{int((q > 0).sum())} bins, share {args.front_share if on else 0.0}", flush=True)


CG = {"calls": 0}  # captured collection step: static tensors and the graph


@torch.no_grad()
def _collect_core():
    """One collection step up to the episode ends, on the static tensors of CG (the captured body)."""
    o = env.observe()
    fresh = env.ep_step == 0
    oh = CG["obs_hist"]
    oh_new = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(oh), torch.cat((oh[:, 1:], o.unsqueeze(1)), 1))
    g = motion.goal(env.f, K)
    pi.obs_norm.update(o)
    pi.goal_norm.update(g)
    a = pi(oh_new, g)
    a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
    if N_EX:  # exam worlds: the deployed (Polyak) policy plus noise
        ae = pi_avg.module(oh_new[-N_EX:], g[-N_EX:]).to(a.dtype)
        ae = torch.clamp(ae + EXAM_NOISE * torch.randn_like(ae), -20, 20)
        if PHASE:  # exam slots outside the current share run practice episodes
            ae = torch.where(env.exam[-N_EX:, None], ae, a[-N_EX:])
        a = torch.cat((a[:-N_EX], ae))
    out = env.step_core(a)
    if N_EX:  # before the resets below, which draw the next exam clips from the updated table
        exam_record(out)
    buffer.add_device(out)
    S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
    wm_in, wm_out, wm_inc = (wm_norm_data_c)(S0, S1, out["a"], out["a_prev"], out["xi"])
    wm.in_norm.update(wm_in)
    wm.out_norm.update(wm_out)
    wm.inc_norm.update(wm_inc)
    oh.copy_(oh_new)
    # the simulator wrapper rebinds these per step: copy them back into the static tensors the graph reads
    CG["f"].copy_(env.f), CG["ep_step"].copy_(env.ep_step), CG["a_prev"].copy_(env.a_prev)
    env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    core = {"done": out["done"], "term": out["term"], "f": out["f"], "r_mean": out["r"].mean() / 0.1}
    out = env.step_tail_masked(out)
    n = out["n_done"]
    nan = torch.full((), float("nan"), device=dev)
    core.update(ep_len=torch.where(n > 0, out["ep_len_sum"] / n.clamp(min=1), nan),
                ep_ret=torch.where(n > 0, out["ep_ret_norm_sum"] / n.clamp(min=1), nan),
                term_sum=out["term"].float().sum(), done_sum=n)  # fmt: skip
    return core


@torch.no_grad()
def collect_graphed_step():
    import warp as wp
    from mjlab.sim.sim_data import TorchArray

    global obs_hist
    if CG["calls"] == 0:
        CG.update(obs_hist=obs_hist.clone(), f=env.f.clone(), ep_step=env.ep_step.clone(), a_prev=env.a_prev.clone())
        env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    CG["calls"] += 1
    if CG["calls"] <= 3:  # warmup on the default stream (Warp's own stream orders with it implicitly)
        out = _collect_core()
    elif "graph" not in CG:
        torch.cuda.synchronize()
        t0 = time.time()
        CG["graph"] = torch.cuda.CUDAGraph()
        setitem, env.sim.use_cuda_graph = TorchArray.__setitem__, False
        TorchArray.__setitem__ = lambda self, idx, value: self._tensor.__setitem__(idx, value)
        try:
            own = None
            with torch.cuda.graph(CG["graph"], stream=own, capture_error_mode="thread_local"):
                ws = wp.stream_from_torch(torch.cuda.current_stream())
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    wp.capture_begin(stream=ws, external=True)
                    try:
                        CG["out"] = _collect_core()
                    finally:
                        CG["wp_graph"] = wp.capture_end(stream=ws)  # keep: it holds the captured kernels' modules
        finally:
            TorchArray.__setitem__, env.sim.use_cuda_graph = setitem, True
        torch.cuda.synchronize()
        print(f"captured one collection step in {time.time() - t0:.1f} s", flush=True)
        CG["graph"].replay()  # the capture itself does not run the step
        out = CG["out"]
    else:
        CG["graph"].replay()
        out = CG["out"]
    buffer.advance_host()
    obs_hist = CG["obs_hist"]
    return out


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        core = collect_graphed_step()
        sim_steps += args.num_envs
        stats["r"].append(core["r_mean"].clone())
        for k_s, k_c in (("ep_len", "ep_len"), ("ep_ret", "ep_ret"), ("term", "term_sum"), ("done", "done_sum")):
            stats[k_s].append(core[k_c].clone())
        continue
    return

env.use_static_focus()
assert not env.push_enabled and not pipe.stateful, "captured collection: no pushes, no sensing state"
buffer.use_device_ptr(device_size=True)
# captured sampling reads the buffer size on the device, so the graphs are captured at once
full = (lambda: True)
wm_update_g = Graphed(wm_update, full)
policy_update_g = Graphed(policy_update, lambda: full() and int(pi_avg.n_averaged if pi_avg is not None else 1) > 0)


if WS:
    _V = args.wstream_worlds
    _f0 = motion.clip_start[torch.zeros(_V, dtype=torch.long, device=dev)]
    _fl = flatten_state(tuple(x.clone() for x in motion.ref_state(_f0)))
    _xi = EX_XI1.expand(_V, -1).clone()
    WST["s"] = {"s": _fl.clone(), "a_prev": torch.zeros(_V, J, device=dev), "xi": _xi, "f": _f0.clone()}
    WST["h"] = {"s": _fl.unsqueeze(1).repeat(1, Hh, 1), "a_prev": torch.zeros(_V, Hh, J, device=dev),
                "xi": _xi.unsqueeze(1).repeat(1, Hh, 1), "f": _f0.unsqueeze(1).repeat(1, Hh)}  # fmt: skip
    WST["o"] = torch.zeros(_V, Hh, mdp.obs_dim, device=dev)
    WST["xin"] = EX_XIN1.expand(_V, -1).clone()
    _len = (motion.clip_end - motion.clip_start + 1).to(torch.float64)
    WST["cdf"] = torch.cumsum(_len ** (-1.0) / (_len ** (-1.0)).sum(), 0)  # equal time per clip
    WST["ends"], WST["fails"] = torch.zeros((), device=dev), torch.zeros((), device=dev)


    @torch.no_grad()
    def ws_draw():
        """A first frame for every stream world (clip probability WST['cdf']), its reference state and observation."""
        u = torch.rand(args.wstream_worlds, device=dev, dtype=torch.float64) * WST["cdf"][-1]
        c = torch.searchsorted(WST["cdf"], u, right=True).clamp(max=len(WST["cdf"]) - 1)
        f = motion.clip_start[c]
        S = tuple(x.clone() for x in motion.ref_state(f))
        return f, S, mdp.obs(S, torch.zeros(args.wstream_worlds, J, device=dev), WST["xi_s"], f)


    WST["xi_s"] = WST["s"]["xi"]

    @torch.no_grad()
    def ws_advance():
        """--wstream_steps steps of the stream in the world model (fixed size, capturable); as exam_preexam."""
        V, st, hi = args.wstream_worlds, WST["s"], WST["h"]
        for _ in range(args.wstream_steps):
            S, f, end = unflatten_state(st["s"]), st["f"], motion.clip_end[motion.clip_id[st["f"]]]
            a = pi_avg.module(WST["o"], motion.goal(f, K)).float()
            S1 = wm.step(S, a, st["a_prev"], WST["xin"])
            f1 = torch.minimum(f + 1, end)
            fail = mdp.danger(S1, f1) >= 1
            done = fail | (f1 >= end)
            f_new, S_new, o_new = ws_draw()
            S2 = tuple(torch.where(done.view(-1, *([1] * (x.dim() - 1))), xn, x) for xn, x in zip(S_new, S1))
            f2 = torch.where(done, f_new, f1)
            a2 = torch.where(done.unsqueeze(-1), 0.0, a)
            fl = flatten_state(S2)
            o2 = torch.where(done.unsqueeze(-1), o_new, mdp.obs(S2, a2, st["xi"], f2))
            d2 = done.view(V, 1, 1)
            WST["o"].copy_(torch.where(d2, o2.unsqueeze(1).expand_as(WST["o"]), torch.cat((WST["o"][:, 1:], o2.unsqueeze(1)), 1)))
            hi["s"].copy_(torch.where(d2, fl.unsqueeze(1).expand_as(hi["s"]), torch.cat((hi["s"][:, 1:], fl.unsqueeze(1)), 1)))
            hi["a_prev"].copy_(torch.where(d2, 0.0, torch.cat((hi["a_prev"][:, 1:], a2.unsqueeze(1)), 1)))
            d1 = done.view(V, 1)
            hi["f"].copy_(torch.where(d1, f2.unsqueeze(1).expand_as(hi["f"]), torch.cat((hi["f"][:, 1:], f2.unsqueeze(1)), 1)))
            st["s"].copy_(fl), st["a_prev"].copy_(a2), st["f"].copy_(f2)
            WST["ends"] += done.float().sum()
            WST["fails"] += fail.float().sum()
        return (WST["ends"],)


    ws_advance_g = Graphed(ws_advance, lambda: True)


    @torch.no_grad()
    def ws_set(on: bool, it: int):
        """Switch the stream's mixing; when it starts, fresh episodes and a burn-in so that the worlds spread along their clips."""
        if on and not WS_ACTIVE[0]:
            t0 = time.time()
            f, S, o = ws_draw()
            fl = flatten_state(S)
            WST["s"]["s"].copy_(fl), WST["s"]["a_prev"].zero_(), WST["s"]["f"].copy_(f)
            WST["o"].copy_(o.unsqueeze(1).expand_as(WST["o"]))
            WST["h"]["s"].copy_(fl.unsqueeze(1).expand_as(WST["h"]["s"]))
            WST["h"]["a_prev"].zero_(), WST["h"]["f"].copy_(f.unsqueeze(1).expand_as(WST["h"]["f"]))
            for _ in range(args.wstream_burn):
                ws_advance_g()
            torch.cuda.synchronize()
            print(f"--wstream it {it}: on, {args.wstream_worlds} worlds, share {args.wstream_frac}, burn-in {time.time() - t0:.1f} s", flush=True)
        WS_ACTIVE[0] = on
        WS_ON.fill_(on)


assert not RAMP or (PHASE and args.exam_pre_every and 0 < RAMP[0] <= RAMP[1] <= args.exam_frac and RAMP[2] > 0 and RAMP[3] > 0), \
    "--share_ramp: needs --phase, --exam_pre_every and lo <= hi <= --exam_frac"

PH_STATE = {"cur": None, "pre_at": None}  # --phase: settings in force; iteration of a pre-exam run by phase_apply


def ph_exams() -> bool:
    """--phase: whether any exam world is active (always true without --phase)."""
    return not PHASE or PH_STATE["cur"]["exam_frac"] > 0


@torch.no_grad()
def ramp_eval(it: int) -> bool:
    """--share_ramp: set RAMP_OV from the pre-exam just run; True when the share changes."""
    lo, hi, n0, st = RAMP
    n1 = int((EX_PD[EX_CLIPS] >= 1).sum())
    s = hi - (hi - lo) * min(1.0, n1 / n0)
    s = min(hi, lo + st * round((s - lo) / st))
    old = PH_STATE["cur"]["exam_frac"] if PH_STATE["cur"] is not None else None
    RAMP_OV["exam_frac"] = s
    print(f"--share_ramp it {it}: n1 {n1} -> exam_frac {s:.3f}" + ("" if old is None or abs(old - s) < 1e-9 else f" (was {old:.3f})"), flush=True)
    return old is not None and abs(old - s) >= 1e-9


def phase_apply(it: int, initial: bool = False, ramp: bool = False):
    """--phase: apply the settings in force at iteration it (initial: at the start of this process; ramp: a --share_ramp
    change, without a refill)."""
    global policy_update_g
    new, old = phase_at(it), PH_STATE["cur"]
    PH_STATE["cur"] = new
    args.wm_batches = new["wm_batches"]  # host-side loop over the captured world-model update
    if new["critic_epochs"] != args.critic_epochs:  # the epoch loop is unrolled in the captured update: capture it anew
        args.critic_epochs = new["critic_epochs"]
        if isinstance(policy_update_g, Graphed):
            ready = policy_update_g.ready
            policy_update_g = None  # frees the old graph's pool before the new capture
            policy_update_g = Graphed(policy_update, ready)
        print(f"--phase it {it}: critic epochs {args.critic_epochs}", flush=True)
    n_act = int(round(new["exam_frac"] * args.num_envs))
    n_old = N_EX if old is None else int(round(old["exam_frac"] * args.num_envs))
    args.lr_decay_iters, args.lr_end, args.front_share = new["lr_decay_iters"], new["lr_end"], new["front_share"]
    PEN.fill_(new["fail_penalty"])
    env.aprev_gate.fill_(new["reset_aprev_ref"])
    if WS:
        ws_set(new["wstream_frac"] > 0, it)
    if n_act > 0 and n_old == 0 and not initial and args.exam_pre_every:  # exams start: as a process started here
        exam_preexam(it)
        PH_STATE["pre_at"] = it
        if RAMP:
            ramp_eval(it)
            new = phase_at(it)
            PH_STATE["cur"] = new
            n_act = int(round(new["exam_frac"] * args.num_envs))
    if n_act != n_old or initial:
        ex_old = env.exam.clone()
        changed = env.set_exam_active(n_act)
        if not initial:  # exam episodes in flight of the worlds that stop examining (the setup draws are not counted)
            gone = (changed & ex_old)[-N_EX:]
            EX_P.index_add_(0, motion.clip_id[env.f[-N_EX:]], -gone.float())
        if bool(changed.any()):
            if buffer.size > 0:  # the reset ends these episodes: mark their last transition (world-model windows stop there)
                last = (buffer.ptr_t - 1) % buffer.T if buffer.ptr_t is not None else torch.tensor((buffer.ptr - 1) % buffer.T, device=dev)
                for k in ("cut", "done"):
                    buffer.data[k][last.view(1)] = buffer.data[k][last.view(1)] | changed
            env.reset_masked(changed)
            if args.phase_refill and not initial and not ramp:  # as a process restarted here: the whole buffer comes from the new roles
                t_fill = time.time()
                if args.front_share > 0 and old["front_share"] == 0:  # the frontier first, as a restarted process has it
                    front_refresh(it)
                SPLIT_N0.fill_(args.num_envs - n_act)
                collect(args.buffer_steps, {k: [] for k in ("r", "ep_len", "ep_ret", "term", "done")})
                print(f"--phase it {it}: buffer refilled with {args.buffer_steps} steps in {time.time() - t_fill:.0f} s", flush=True)
    else:
        changed = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
    p = new["start_prac_frac"]
    SPLIT_N0.fill_(args.num_envs - n_act)
    SPLIT_P.fill_(1.0 if n_act == 0 else (p if p >= 0 else (args.num_envs - n_act) / args.num_envs))
    if args.front_share == 0:
        env.set_focus(torch.full_like(env.focus_q, 1.0 / len(env.focus_q)), 0.0)
    elif n_act > 0 and (initial or old["front_share"] == 0) and (it - start_it) % args.front_every:
        front_refresh(it)
    print(f"--phase it {it}: " + ", ".join(f"{k}={v}" for k, v in new.items()) + f" | {n_act} exam worlds, "
          f"{int(changed.sum())} worlds reset, start split ({int(SPLIT_N0)}, {float(SPLIT_P):.3f})", flush=True)


stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
if PHASE:
    phase_apply(start_it, initial=True)
elif WS:
    ws_set(True, start_it)
collect(args.warmup_steps, stats)  # the replay buffer is not checkpointed, so a resumed run refills it too
log_path = os.path.join(args.log_dir, "train_log.csv")
new_log = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
log_f = open(log_path, "a")
anchor_cols = ""
anchor_cols += ""
log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr"
            + anchor_cols + ",pi_lr,pi_mult,nonfinite\n")  # fmt: skip
log_f.flush()
LOG_EXTRA = "pi_lr" in open(log_path).readline()  # lr = base LR; pi_lr = policy LR (x pi_mult)
# checkpoints: one background writer; a failed write is re-raised in the training loop (or at the end), at most 2 queued
CKPT_POOL, CKPT_PENDING = (ThreadPoolExecutor(max_workers=1)), []


def ckpt_wait(drain=False):
    while CKPT_PENDING and (drain or CKPT_PENDING[0].done() or len(CKPT_PENDING) > 2):
        CKPT_PENDING.pop(0).result()


def _save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}


def _to_cpu(o):
    if torch.is_tensor(o):
        return o.detach().to("cpu", copy=True)
    if isinstance(o, dict):
        return {k: _to_cpu(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_to_cpu(v) for v in o)
    return o


for it in range(start_it, args.iterations):
    if N_EX:
        IT_T.fill_(it)
    if PHASE and it in PH_ITS and it != start_it:
        phase_apply(it)
    lr = lr_at(it)
    pi_mult = args.pi_lr_mult
    for o, mult in ((opt_wm, 1.0), (opt_pi, pi_mult), (opt_v, 1.0)):
        for gparam in o.param_groups:
            if torch.is_tensor(gparam["lr"]):
                gparam["lr"].fill_(lr * mult)
            else:
                gparam["lr"] = lr * mult
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    # exams: refresh the exam clip table, pre-exams in the world model, frontier practice regions
    if N_EX and it > start_it and (it - start_it) % args.exam_every == 0 and ph_exams():
        exam_refresh(it)
    if (N_EX and args.exam_pre_every and it > start_it and (it - start_it) % args.exam_pre_every == 0 and ph_exams()
            and PH_STATE["pre_at"] != it):
        exam_preexam(it)
        if RAMP and ramp_eval(it):
            phase_apply(it, ramp=True)
    if args.front_share and (it - start_it) % args.front_every == 0:
        front_refresh(it)
    collect(n, stats)
    for b_ in range(args.wm_batches):
        l, e8 = wm_update_g()
    agg["wm"].append(l), agg["wm8"].append(e8)
    n_avg0 = int(pi_avg.n_averaged) if pi_avg is not None else 0
    if WS and WS_ACTIVE[0]:
        ws_advance_g()
    pl, vl, ir, gn, surv = policy_update_g()
    if (pi_avg is not None and int(pi_avg.n_averaged) == n_avg0):
        pi_avg.n_averaged += 1  # ema_update (and a graph replay) does not advance the host-side counter
    agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        mean_nn = lambda x: mean([v for v in (float(v) for v in x) if v == v])  # noqa: E731  (NaN = no episode ended)
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean_nn(stats["ep_len"]), mean_nn(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
        extra = [lr * pi_mult, float(pi_mult), int(NONFINITE)] if LOG_EXTRA else []
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row + extra) + "\n")
        log_f.flush()
        vc = VCL.tolist()
        VCL.zero_()
        print(f"   vclamp | V {vc[0] / max(vc[3], 1):.4f} targets {vc[1] / max(vc[3], 1):.4f} boot {vc[2] / max(vc[3], 1):.4f} above V_max", flush=True)
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
              f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    wall = time.time() - t_start
    final_ck = it + 1 == args.iterations
    if (not final_ck and (it + 1) % args.ckpt_every == 0):
        sd = pi_avg.module.state_dict() if pi_avg is not None else pi.state_dict()
        small = {"pi": {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}, "iteration": it + 1,
                 "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                 "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        ckpt_wait()
        CKPT_PENDING.append(CKPT_POOL.submit(_save_atomic, small, path))
    elif ((it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations):
        ck = {"wm": wm.state_dict(), "pi": pi_avg.module.state_dict() if pi_avg is not None else pi.state_dict(), "pi_live": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "pi_avg": pi_avg.state_dict() if pi_avg is not None else None, "wm_avg": None,
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "pi_live", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)
if CKPT_POOL is not None:  # every background checkpoint written (a failed write raises here: non-zero exit)
    ckpt_wait(drain=True)
    CKPT_POOL.shutdown()
