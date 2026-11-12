"""Reference motion library: retargeted clips resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np
import torch

from .kinematics import Kinematics
from .robot import MotionArrays, RobotSpec
from .rotations import mat_to_6d, mm3, mv3, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


BONES_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "bones_seed", "lib_all.npz")
GOAL_Q_OFFSET = 15  # index of the joint positions q(f+1) in the goal vector (first future frame: dp 3, rot 6, v 3, w 3, then q)


class MotionLib:
    def __init__(self, files: list[str], kin: Kinematics, device: str, load_motion: Callable[[str], MotionArrays]):
        names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js = [], [], []
        for f in files:
            p, quat, j, src_fps = load_motion(f)
            p, quat, j = (torch.tensor(x, dtype=torch.float32) for x in (p, quat, j))
            # resample to 50 Hz
            T_src = p.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
        self._build(names, ps, qs, js, kin, device)

    @classmethod
    def from_clips(cls, names: list[str], ps, qs, js, kin: Kinematics, device: str) -> MotionLib:
        """Clips already at 50 Hz: root position (T,3), root quaternion wxyz (T,4), joint positions (T,J)."""
        self = cls.__new__(cls)
        self._build(names, ps, qs, js, kin, device)
        return self

    @classmethod
    def from_bones(cls, lib_path: str, list_path: str, kin: Kinematics, device: str) -> MotionLib:
        """BONES-SEED clips named in ``list_path`` (one per line) from the packed 50 Hz library of tools/prepare_bones.py."""
        names = [s.strip() for s in open(list_path) if s.strip()]
        lib = _npz_mmap(lib_path)
        index = {n: i for i, n in enumerate(lib["names"].tolist())}
        starts, lengths = lib["starts"], lib["lengths"]
        ps, qs, js = [], [], []
        for n in names:
            i = index[n]
            s, e = int(starts[i]), int(starts[i]) + int(lengths[i])
            ps.append(torch.from_numpy(np.array(lib["p"][s:e]))), qs.append(torch.from_numpy(np.array(lib["q"][s:e])))
            js.append(torch.from_numpy(np.array(lib["j"][s:e])))
        return cls.from_clips(names, ps, qs, js, kin, device)

    def _build(self, names: list[str], ps, qs, js, kin: Kinematics, device: str) -> None:
        self.device = device
        self.names = names
        lens = [p.shape[0] for p in ps]
        self._index(lens, device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)

    def _index(self, lens: list[int], device: str) -> None:
        """Clip ranges, per-frame clip id and the 1 s bins (vectorized; the same integers as per-clip loops)."""
        L = torch.tensor(lens, dtype=torch.long)
        self.clip_len = L.to(device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), L[:-1])), 0).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(L.sum())
        C = len(lens)
        self.clip_id = torch.repeat_interleave(torch.arange(C), L).to(device)
        nb = (L + BIN_FRAMES - 1) // BIN_FRAMES
        bin_off = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), nb[:-1])), 0)
        bin_clip = torch.repeat_interleave(torch.arange(C), nb)
        self.bin_clip = bin_clip.to(device)
        self.bin_start = (self.clip_start.cpu()[bin_clip] + (torch.arange(int(nb.sum())) - bin_off[bin_clip]) * BIN_FRAMES).to(device)
        self.num_bins = int(nb.sum())
        cid = self.clip_id.cpu()
        local = torch.arange(self.num_frames) - self.clip_start.cpu()[cid]
        self.frame_bin = (bin_off[cid] + local // BIN_FRAMES).to(device)
        self._set_bin_end()

    def _set_bin_end(self) -> None:
        """bin_end: exclusive end of each bin's episode starts (env.sample_frames draws in [bin_start, bin_end)); a
        clip's last bin ends one frame before the clip end (keep >= 1 step), the others after BIN_FRAMES starts."""
        last = torch.ones_like(self.bin_clip, dtype=torch.bool)
        last[:-1] = self.bin_clip[1:] != self.bin_clip[:-1]
        ce = self.clip_end[self.bin_clip]
        self.bin_end = torch.where(last, ce, torch.minimum(self.bin_start + BIN_FRAMES, ce))
        w = (self.bin_end - self.bin_start).clamp(min=0).float()  # episode starts per bin
        self.bin_w = w / w.sum()  # --alloc e: uniform over episode starts

    def merge_tail_bins(self, min_starts: int = BIN_FRAMES // 2) -> int:
        """A clip's last bin with fewer than ``min_starts``
        episode starts is merged into the bin before it. Unmerged, a last bin can hold a single start (or none: the
        start is the clip's last frame), its episodes last 1-4 steps, the first step's action-rate penalty
        (a_prev = 0) extrapolated by the mean-reward bootstrap gives a very low G, and the return softmax sends a
        third of all resets there. Clips with a single bin are unchanged. Returns the number of merged bins."""
        dev = self.clip_len.device
        L = self.clip_len.cpu()
        nb = (L + BIN_FRAMES - 1) // BIN_FRAMES
        tail = (L - 1) - (nb - 1) * BIN_FRAMES  # starts in the last bin: local frames (nb-1)*BIN_FRAMES .. L-2
        merge = (nb > 1) & (tail < min_starts)
        nb = nb - merge.long()
        C = len(L)
        bin_off = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), nb[:-1])), 0)
        bin_clip = torch.repeat_interleave(torch.arange(C), nb)
        self.bin_clip = bin_clip.to(dev)
        self.bin_start = (self.clip_start.cpu()[bin_clip] + (torch.arange(int(nb.sum())) - bin_off[bin_clip]) * BIN_FRAMES).to(dev)
        self.num_bins = int(nb.sum())
        cid = self.clip_id.cpu()
        local = torch.arange(self.num_frames) - self.clip_start.cpu()[cid]
        self.frame_bin = (bin_off[cid] + torch.minimum(local // BIN_FRAMES, nb[cid] - 1)).to(dev)
        self._set_bin_end()
        return int(merge.sum())

    def ref_bodies(self, f: torch.Tensor):
        """Tracked-body world positions, rotations, linear and angular velocities of reference frames f."""
        return self.body_p[f], self.body_R[f], self.body_v[f], self.body_w[f]

    @staticmethod
    def from_dir(path: str, robot: RobotSpec, kin: Kinematics, device: str) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, robot.motion_glob)))
        return MotionLib(files, kin, device, robot.load_motion)

    @staticmethod
    def load(robot: RobotSpec, kin: Kinematics, device: str, motion_dir: str | None = None, bones_list: str | None = None,
             bones_lib: str | None = None, store: str = "full") -> MotionLib:
        """The robot's clip directory (or ``motion_dir``), or the BONES-SEED clips named in ``bones_list``. ``store``:
        "full" precomputes every per-frame quantity (MotionLib); "fp32" / "fp16" keep only root pose and joint positions
        and compute the rest when indexed (CompactMotionLib)."""
        cls = MotionLib if store == "full" else CompactMotionLib
        if store != "full":
            CompactMotionLib.STORE_DTYPE = {"fp32": torch.float32, "fp16": torch.float16}[store]
        if bones_list:
            return cls.from_bones(bones_lib or BONES_LIB, bones_list, kin, device)
        files = sorted(glob.glob(os.path.join(motion_dir or robot.motion_dir, robot.motion_glob)))
        return cls(files, kin, device, robot.load_motion)

    def set_ref_contact(self, feet: list[int], margin: float = 0.015, vmax: float = 0.5, q: float = 0.05) -> None:
        """Speed-branch reference foot contact (num_frames, len(feet)): height below the clip's own floor (the
        q-quantile of that foot's height in the clip) + margin, and horizontal speed below vmax. A fixed 4.5 cm
        threshold misses most stance frames of clips whose retargeted feet sit ~1 cm higher (LAFAN1 subject1:
        floor ~4.5 cm against ~3.5 cm for the others)."""
        zs, vs = [], []
        with torch.no_grad():
            for a in range(0, self.num_frames, 1 << 20):
                bp, _, bv, _ = self.ref_bodies(torch.arange(a, min(a + (1 << 20), self.num_frames), device=self.device))
                zs.append(bp[:, feet, 2]), vs.append(bv[:, feet, :2].norm(dim=-1))
        z, vh = torch.cat(zs), torch.cat(vs)
        # per-clip quantile, vectorized (clips are contiguous frame ranges); equal to torch.quantile on each clip:
        # sort every column, a stable sort by clip id keeps each clip's values ordered, then the linear interpolation
        start, lens = self.clip_start.to(z.device).long(), self.clip_len.to(z.device).long()
        cid = torch.repeat_interleave(torch.arange(len(lens), device=z.device), lens)
        v, i = z.sort(dim=0)
        _, j = cid[i].sort(dim=0, stable=True)
        v = v.gather(0, j)
        pos = q * (lens - 1).to(z.dtype)
        lo, hi = pos.floor().long(), pos.ceil().long()
        w = (pos - lo).unsqueeze(1).expand(-1, z.shape[1]).contiguous()
        floor = torch.lerp(v[start + lo], v[start + hi], w)[cid]
        self.ref_contact = ((z < floor + margin) & (vh < vmax)).float()

    def goal_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.q.shape[-1])

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = mv3(Ht, self.root_p[fut] - self.root_p[f].unsqueeze(1))
        rot = mat_to_6d(mm3(Ht, self.root_R[fut]))
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,15+2J)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


class _Lazy:
    """Per-frame quantity computed when indexed: ``x[f]`` calls ``fn(f)``."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, f):
        return self.fn(f)


class CompactMotionLib(MotionLib):
    """MotionLib that stores per frame only the root position (fp32), root quaternion and joint positions (STORE_DTYPE)
    and computes root/joint velocities (central differences, one-sided at clip ends, as MotionLib) and tracked-body FK
    when indexed: ~100 B per frame in fp16 instead of ~1.4 KB, so the full BONES-SEED list (47 M frames) fits on the GPU
    next to training. Same attributes, indexed the same way (``m.root_R[f]``, ``m.ref_bodies(f)``, ...)."""

    STORE_DTYPE = torch.float16

    def _build(self, names: list[str], ps, qs, js, kin: Kinematics, device: str) -> None:
        self.device, self.names, self.kin = device, names, kin
        self._index([p.shape[0] for p in ps], device)
        dt = self.STORE_DTYPE
        self._p = torch.cat(ps).float().to(device)
        self._quat = torch.cat(qs).to(dtype=dt).to(device)
        self._j = torch.cat(js).to(dtype=dt).to(device)
        self.root_p = self._p
        self.root_R = _Lazy(self._R)
        self.root_v = _Lazy(lambda f: self._vel(f)[0])
        self.root_w = _Lazy(lambda f: self._vel(f)[1])
        self.q = _Lazy(lambda f: self._j[f].float())
        self.qd = _Lazy(lambda f: self._vel(f)[2])
        self.root_v_local = _Lazy(lambda f: self._local(f)[0])
        self.root_w_local = _Lazy(lambda f: self._local(f)[1])
        self.root_gz = _Lazy(lambda f: self._R(f)[..., 2, 2])
        self.body_p, self.body_R, self.body_v, self.body_w = (_Lazy(lambda f, k=k: self.ref_bodies(f)[k]) for k in range(4))

    def _R(self, f):
        return quat_to_mat(self._quat[f].float())

    def _vel(self, f):
        """World root linear / angular velocity and joint velocities of frames f (MotionLib's finite differences)."""
        c = self.clip_id[f]
        fm = torch.maximum(f - 1, self.clip_start[c])
        fp = torch.minimum(f + 1, self.clip_end[c])
        h = ((fp - fm).float() * CONTROL_DT).unsqueeze(-1)
        v = (self._p[fp] - self._p[fm]) / h
        w = so3_log(self._R(fp) @ self._R(fm).transpose(-1, -2)) / h
        qd = (self._j[fp].float() - self._j[fm].float()) / h
        return v, w, qd

    def _local(self, f):
        v, w, _ = self._vel(f)
        Rt = self._R(f).transpose(-1, -2)
        return (Rt @ v.unsqueeze(-1)).squeeze(-1), (Rt @ w.unsqueeze(-1)).squeeze(-1)

    def ref_state(self, f: torch.Tensor):
        v, w, qd = self._vel(f)
        return self._p[f], self._R(f), v, w, self._j[f].float(), qd

    def ref_bodies(self, f: torch.Tensor):
        return self.kin.forward(*self.ref_state(f))

    def goal_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self._j.shape[-1])

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """MotionLib.goal with every future-frame quantity computed once."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self._R(f))
        Ht = H.transpose(-1, -2).unsqueeze(1)
        p_f, R_f, v_f, w_f, q_f, qd_f = self.ref_state(fut)
        Rt = R_f.transpose(-1, -2)
        dp = mv3(Ht, p_f - self._p[f].unsqueeze(1))
        rot = mat_to_6d(mm3(Ht, R_f))
        v_loc, w_loc = (Rt @ v_f.unsqueeze(-1)).squeeze(-1), (Rt @ w_f.unsqueeze(-1)).squeeze(-1)
        feats = torch.cat((dp, rot, v_loc, w_loc, q_f, qd_f * 0.1), -1)
        return feats.reshape(f.shape[0], -1)


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def _npz_mmap(path: str) -> dict[str, np.ndarray]:
    """Memory-mapped members of an uncompressed .npz (np.load ignores mmap_mode for archives)."""
    import zipfile

    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as fh:
        for info in z.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, f"{info.filename} is compressed"
            fh.seek(info.header_offset)
            hdr = fh.read(30)
            off = info.header_offset + 30 + int.from_bytes(hdr[26:28], "little") + int.from_bytes(hdr[28:30], "little")
            fh.seek(off)
            version = np.lib.format.read_magic(fh)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read(fh)
            key = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            if dtype.hasobject:
                raise ValueError(key)
            out[key] = np.memmap(path, dtype=dtype, mode="r", offset=fh.tell(), shape=shape, order="F" if fortran else "C")
    return out
