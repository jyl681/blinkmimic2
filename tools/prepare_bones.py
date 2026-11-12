"""BONES-SEED (G1 retarget) -> 50 Hz packed motion library and the list of clips to train on.

Streams g1.tar.gz (no extraction), converts every CSV (120 fps; root cm, root XYZ-Euler degrees, 29 joints in degrees)
to 50 Hz root position (m), root quaternion (wxyz) and joint angles (rad), and records the filter verdict of each clip.

Filter, as in SuperMimic (Appendix E) and the open reproductions (keyword list from LaoYang1994/SuperMimic-Rep, MIT):
the SONIC keywords plus box_, jump_off_50cm and look_over_fence, then feasibility on MuJoCo forward kinematics of the
mjlab G1. Result: 129,491 clips, the list used by realmimic and in our paper;
sha256 of keep.txt = d05ea3ef335802de8b80d467e5b8dd3af0029a3405cc0770bc26565528e3375d.

  python tools/prepare_bones.py convert --archive RAW/g1.tar.gz --out data/bones_seed/shards --workers 12
  python tools/prepare_bones.py pack --shards data/bones_seed/shards --out data/bones_seed   # manifest.csv, keep.txt, lib_all.npz
"""

from __future__ import annotations

import csv
import io
import multiprocessing as mp
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

SRC_FPS = 120.0
DT = 0.02  # = motion.CONTROL_DT
FPS = 50

SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "acrobatics_", "box_HSPU", "cartwheel",
    "50cm_box_", "on_box", "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m", "230m",
    "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll", "kozak_dance", "stair", "handstand",
    "box_jump", "monkey_jump", "safety_roll", "box_dips", "walking_on_edge", "push_obstacle",
]  # fmt: skip
EXTRA_KEYWORDS = ["box_", "jump_off_50cm", "look_over_fence"]
KEYWORDS = [k.lower() for k in SONIC_KEYWORDS + EXTRA_KEYWORDS]

ELEVATED_Z, ELEVATED_S = 0.20, 0.6
PENETRATE_Z_HARD, PENETRATE_Z_SOFT, PENETRATE_FRAC = -0.12, -0.08, 0.80
FLOAT_Z, FLOAT_S = 0.02, 0.5
UPRIGHT_GZ, STAND_Z, STILL_V, STILL_QD = 0.9, (0.60, 0.85), 0.1, 0.5

_M = _D = _FOOT = None


def name_filter(name: str) -> str | None:
    n = name.lower()
    for k in KEYWORDS:
        if k in n:
            return f"name:{k}"
    return None


def _model_path() -> str:
    import mjlab

    return os.path.join(os.path.dirname(mjlab.__file__), "asset_zoo/robots/unitree_g1/xmls/g1.xml")


def _init() -> None:
    global _M, _D, _FOOT
    import mujoco

    from blinkmimic2.g1.g1_constants import G1_JOINT_NAMES

    _M = mujoco.MjModel.from_xml_path(_model_path())
    assert tuple(_M.joint(j).name for j in range(1, _M.njnt)) == tuple(G1_JOINT_NAMES)
    _D = mujoco.MjData(_M)
    _FOOT = [g for g in range(_M.ngeom) if (_M.geom(g).name or "").endswith("_collision") and "_foot" in _M.geom(g).name]
    assert len(_FOOT) == 14 and all(_M.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in _FOOT)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), None, 1.0)
    th = np.arccos(dot)
    s = np.sin(th)
    t = t[:, None]
    small = s < 1e-6
    w0 = np.where(small, 1 - t, np.sin((1 - t) * th) / (s + 1e-12))
    w1 = np.where(small, t, np.sin(t * th) / (s + 1e-12))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def parse_csv(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """120 fps CSV -> 50 Hz root position (m), quaternion wxyz, joints (rad), float64."""
    from scipy.spatial.transform import Rotation

    raw = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, ndmin=2)
    p = raw[:, 1:4] / 100.0
    q_xyzw = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat()
    q = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)  # hemisphere continuity is handled by slerp; keep w >= 0
    j = np.deg2rad(raw[:, 7:36])
    T = raw.shape[0]
    n = int(np.floor((T - 1) / SRC_FPS / DT)) + 1
    t = np.arange(n) * DT * SRC_FPS
    i0 = np.minimum(np.floor(t).astype(np.int64), T - 2)
    a = np.clip(t - i0, 0.0, 1.0)
    p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
    j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
    q = _slerp(q[i0], q[i0 + 1], a)
    return p, q, j


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return int((np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]).max())


def feasibility(p: np.ndarray, q: np.ndarray, j: np.ndarray) -> str | None:
    import mujoco

    T = p.shape[0]
    body_z = np.empty((T, _M.nbody - 1))
    for t in range(T):
        _D.qpos[:3], _D.qpos[3:7], _D.qpos[7:] = p[t], q[t], j[t]
        mujoco.mj_kinematics(_M, _D)
        body_z[t] = _D.xpos[1:, 2]
    zmin = body_z.min(1)
    if _longest_run(zmin > ELEVATED_Z) >= ELEVATED_S * FPS:
        return "elevated_support"
    if (zmin < PENETRATE_Z_HARD).any() or (zmin < PENETRATE_Z_SOFT).mean() > PENETRATE_FRAC:
        return "ground_penetration"
    r22 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)  # z-axis of pelvis . world z
    upright = r22 > UPRIGHT_GZ
    standing = (p[:, 2] > STAND_Z[0]) & (p[:, 2] < STAND_Z[1])
    v = np.gradient(p, DT, axis=0) if T > 1 else np.zeros_like(p)
    qd = np.gradient(j, DT, axis=0) if T > 1 else np.zeros_like(j)
    still = (np.linalg.norm(v, axis=1) < STILL_V) & (np.sqrt((qd**2).mean(1)) < STILL_QD)
    gate = upright & standing & still
    if _longest_run(gate) >= FLOAT_S * FPS:
        radius, half = _M.geom_size[_FOOT, 0], _M.geom_size[_FOOT, 1]
        floating = np.zeros(T, bool)
        for t in np.nonzero(gate)[0]:
            _D.qpos[:3], _D.qpos[3:7], _D.qpos[7:] = p[t], q[t], j[t]
            mujoco.mj_kinematics(_M, _D)
            cz = _D.geom_xpos[_FOOT, 2]
            axis_z = _D.geom_xmat[_FOOT].reshape(-1, 3, 3)[:, 2, 2]
            floating[t] = (cz - half * np.abs(axis_z) - radius).min() > FLOAT_Z
        if _longest_run(floating) >= FLOAT_S * FPS:
            return "floating_feet"
    return None


def _convert(item: tuple[str, bytes]):
    name, data = item
    try:
        p, q, j = parse_csv(data)
        reason = name_filter(name)
        if reason is None:
            reason = feasibility(p, q, j)
        return name, (p.astype(np.float32), q.astype(np.float32), j.astype(np.float32)), reason or "", ""
    except Exception as e:  # noqa: BLE001
        return name, None, "", repr(e)


def _members(archive: Path, skip: set[str]):
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".csv"):
                name = Path(m.name).stem
                if name not in skip:
                    yield name, tf.extractfile(m).read()


def convert(archive: Path, out: Path, workers: int, shard_size: int = 2000, read_ahead: int = 256) -> None:
    out.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for f in sorted(out.glob("shard_*.csv")):
        with open(f, newline="") as fh:
            done |= {r["name"] for r in csv.DictReader(fh)}
    n_shard = len(list(out.glob("shard_*.csv")))
    print(f"{len(done)} clips already in {n_shard} shards", flush=True)
    rows, arrs = [], {}
    errors = open(out / "errors.txt", "a")
    t0 = time.time()

    def flush():
        nonlocal n_shard, rows, arrs
        if not rows:
            return
        stem = out / f"shard_{n_shard:04d}"
        np.savez(str(stem) + ".tmp.npz", **arrs)
        os.replace(str(stem) + ".tmp.npz", str(stem) + ".npz")
        with open(str(stem) + ".csv", "w", newline="") as fh:  # written last = shard complete
            w = csv.writer(fh)
            w.writerow(["name", "frames", "reason"])
            w.writerows(rows)
        n_shard += 1
        print(f"[{(time.time() - t0) / 60:.1f} min] shard {n_shard - 1}: total {len(done)}", flush=True)
        rows, arrs = [], {}

    batch = []
    with mp.Pool(workers, initializer=_init) as pool:

        def run(batch):
            for name, a, reason, err in pool.map(_convert, batch, chunksize=4):
                if a is None:
                    errors.write(f"{name}\t{err}\n")
                    errors.flush()
                    continue
                arrs[f"{name}/p"], arrs[f"{name}/q"], arrs[f"{name}/j"] = a
                rows.append((name, a[0].shape[0], reason))
                done.add(name)
                if len(rows) >= shard_size:
                    flush()

        for item in _members(archive, done):
            batch.append(item)
            if len(batch) >= read_ahead:
                run(batch)
                batch = []
        if batch:
            run(batch)
    flush()
    print(f"done: {len(done)} clips, {n_shard} shards, {(time.time() - t0) / 60:.1f} min", flush=True)


def pack(shards: Path, out: Path) -> None:
    """Merge shard manifests (manifest.csv), write keep.txt (sorted names) and lib_all.npz (all clips, sorted)."""
    rows = []
    for f in sorted(shards.glob("shard_*.csv")):
        with open(f, newline="") as fh:
            rows += [(r["name"], int(r["frames"]), r["reason"], f.stem) for r in csv.DictReader(fh)]
    rows.sort()
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "frames", "reason"])
        w.writerows([r[:3] for r in rows])
    keep = [r for r in rows if not r[2]]
    (out / "keep.txt").write_text("".join(r[0] + "\n" for r in keep))
    n_name = sum(1 for r in rows if r[2].startswith("name:"))
    reasons: dict[str, int] = {}
    for r in rows:
        if r[2] and not r[2].startswith("name:"):
            reasons[r[2]] = reasons.get(r[2], 0) + 1
    print(f"raw {len(rows)} -> name filter {len(rows) - n_name} -> feasibility {len(keep)} (expected 129,491)")
    print(f"feasibility removed {reasons}; kept {sum(r[1] for r in keep) / FPS / 3600:.2f} h")
    # one packed library of every clip (sorted by name); a subset is a list of names
    N = sum(r[1] for r in rows)
    P, Q, J = np.empty((N, 3), np.float32), np.empty((N, 4), np.float32), np.empty((N, 29), np.float32)
    starts, s = [], 0
    by_shard: dict[str, list] = {}
    for i, r in enumerate(rows):
        by_shard.setdefault(r[3], []).append(i)
    off = np.cumsum([0] + [r[1] for r in rows])
    for stem, idx in by_shard.items():
        with np.load(shards / f"{stem}.npz") as z:
            for i in idx:
                n, a = rows[i][0], off[i]
                P[a : a + rows[i][1]], Q[a : a + rows[i][1]], J[a : a + rows[i][1]] = z[f"{n}/p"], z[f"{n}/q"], z[f"{n}/j"]
    np.savez(out / "lib_all.npz", names=np.array([r[0] for r in rows]), starts=off[:-1], lengths=np.array([r[1] for r in rows]), p=P, q=Q, j=J)
    print(f"saved {out / 'lib_all.npz'}: {N} frames, {(P.nbytes + Q.nbytes + J.nbytes) / 1e9:.2f} GB")


if __name__ == "__main__":
    cmd = sys.argv[1]
    kw = dict(zip([a.lstrip("-") for a in sys.argv[2::2]], sys.argv[3::2]))
    if cmd == "convert":
        convert(Path(kw["archive"]), Path(kw["out"]), int(kw.get("workers", 8)))
    elif cmd == "pack":
        pack(Path(kw["shards"]), Path(kw["out"]))
    else:
        raise SystemExit(__doc__)
