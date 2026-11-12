"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .mdp import state_dim


def fields(num_j: int, xi_dim: int, obs_dim: int | None = None, obs_state_dim: int | None = None,
           danger: bool = False) -> dict:
    out = {
        "s": (state_dim(num_j), torch.float32),
        "s1": (state_dim(num_j), torch.float32),
        "a": (num_j, torch.float32),
        "a_prev": (num_j, torch.float32),
        "xi": (xi_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
        "cut": ((), torch.bool),
    }
    if obs_dim is not None:  # stateful sensing: the observation as seen and the sensing state after it
        out["o"] = (obs_dim, torch.float32)
        out["obs_state"] = (obs_state_dim, torch.float32)
    if danger:  # --anchor_coef: danger ratio of the state s against its frame f (MDP.danger), one float per transition
        out["rho"] = ((), torch.float32)
    return out


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, num_j: int, xi_dim: int, obs_dim: int | None = None,
                 obs_state_dim: int | None = None, danger: bool = False):  # fmt: skip
        self.T, self.N, self.device = T, N, device
        self.fields = fields(num_j, xi_dim, obs_dim, obs_state_dim, danger)
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0
        # --cuda_graph: a device copy of ``ptr``, so that sampling captured in a CUDA graph follows the write head
        self.ptr_t = None
        # --graph_early: a device copy of ``size`` too; sampling bounds follow it (uniform draws scaled by the fill level
        # instead of randint with a host bound), so the captured updates stay valid while the buffer fills
        self.size_t = None

    start_split = None  # train.py --start_prac_frac: (n0, p)
    start_iw = None  # train.py --start_iw: weight of a transition from its frame f (the [n0, N) part of the starts)

    def use_device_ptr(self, device_size: bool = False):
        self.ptr_t = torch.full((), self.ptr, dtype=torch.long, device=self.device)
        if device_size:
            self.size_t = torch.full((), self.size, dtype=torch.long, device=self.device)

    def _randint(self, n, B: int) -> torch.Tensor:
        """Uniform integers in [0, n) with n a host int, or (--graph_early) a device tensor."""
        if self.size_t is None:
            return torch.randint(0, n, (B,), device=self.device)
        return torch.minimum((torch.rand(B, device=self.device) * n).long(), n - 1)

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        if self.ptr_t is not None:
            self.ptr_t.fill_(self.ptr)
        if self.size_t is not None:
            self.size_t.fill_(self.size)

    def add_device(self, tr: dict):
        """add() at the device-side write head (train.py --collect_graph: capturable). Advances ptr_t / size_t on the
        device; the host-side ptr / size are advanced by advance_host() after each replay."""
        row = self.ptr_t.view(1)
        for k in self.fields:
            self.data[k].index_copy_(0, row, tr[k].unsqueeze(0).to(self.data[k].dtype))  # add() casts the same way
        self.ptr_t.copy_((self.ptr_t + 1) % self.T)
        self.size_t.copy_(torch.clamp(self.size_t + 1, max=self.T))

    def advance_host(self):
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        ptr = self.ptr_t if self.ptr_t is not None else self.ptr
        size = self.size_t if self.size_t is not None else self.size
        return (ptr - size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = self._randint((self.size_t if self.size_t is not None else self.size) - k + 1, B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "a_prev", "xi", "cut", "f")}
        # transition j is valid if the state stayed continuous (no episode end, no push) after 0..j-1
        d = out["cut"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int, danger_frac: float = 0.0, danger_tau: float = 0.5):
        """Start points for imagination plus the indices of their observation history (oldest first).

        ``danger_frac`` > 0 (train.py --imag_danger_frac): that share of the starts is drawn uniformly among the filled
        transitions whose danger ratio is >= ``danger_tau`` (uniform starts when there is none), without a host sync;
        the two parts are shuffled together so that a later truncation of the candidate list keeps the mix. Also
        returns the share of the buffer with danger >= tau (0-dim tensor) in ``self.danger_share``."""
        age = self._randint(self.size_t if self.size_t is not None else self.size, B)
        if self.start_split is None:
            env = torch.randint(0, self.N, (B,), device=self.device)
        else:  # train.py --start_prac_frac: share p of the starts from envs [0, n0), the rest from [n0, N)
            n0, p = self.start_split
            e0 = (torch.rand(B, device=self.device) * n0).long().clamp(max=n0 - 1)
            e1 = n0 + (torch.rand(B, device=self.device) * (self.N - n0)).long().clamp(max=self.N - n0 - 1)
            if self.start_iw is not None:  # the [n0, N) part drawn with probability proportional to start_iw(f)
                size = self.size_t if self.size_t is not None else self.size
                rows = self._row(torch.arange(self.T, device=self.device))  # age order; ages >= size are not filled
                filled = torch.arange(self.T, device=self.device) < size
                w = self.start_iw(self.data["f"][rows, n0:]) * filled.unsqueeze(-1)  # (T, N - n0)
                cs = w.flatten().double().cumsum(0)
                idx = torch.searchsorted(cs, torch.rand(B, device=self.device, dtype=torch.float64) * cs[-1], right=True)
                idx = idx.clamp(max=cs.numel() - 1)
                a1, e1 = idx // (self.N - n0), n0 + idx % (self.N - n0)
                self.start_ess = cs[-1] ** 2 / ((w.double() ** 2).sum() * (filled.sum() * (self.N - n0))).clamp(min=1e-30)
            pick0 = torch.rand(B, device=self.device) < p
            env = torch.where(pick0, e0, e1)
            if self.start_iw is not None:
                age = torch.where(pick0, age, a1)
        Bd = int(B * danger_frac)
        if Bd > 0:
            size = self.size_t if self.size_t is not None else self.size
            rows = self._row(torch.arange(self.T, device=self.device))  # age order; ages >= size are not filled
            filled = torch.arange(self.T, device=self.device) < size
            m = (self.data["rho"][rows] >= danger_tau) & filled.unsqueeze(-1)  # (T, N) in age order
            cs = m.flatten().cumsum(0)
            n = cs[-1]
            k = (torch.rand(Bd, device=self.device) * n).long().clamp(max=(n - 1).clamp(min=0))
            idx = torch.searchsorted(cs, k, right=True).clamp(max=cs.numel() - 1)
            has = n > 0
            age = torch.cat((age[: B - Bd], torch.where(has, idx // self.N, age[B - Bd :])))
            env = torch.cat((env[: B - Bd], torch.where(has, idx % self.N, env[B - Bd :])))
            perm = torch.randperm(B, device=self.device)
            age, env = age[perm], env[perm]
            self.danger_share = n / (filled.sum() * self.N).clamp(min=1)
        return self.starts_at(age, env, hist)

    def starts_at(self, age: torch.Tensor, env: torch.Tensor, hist: int):
        """Start points at the transitions (age, env) (age 0 = oldest) plus their observation history (oldest first)."""
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        stateful = "o" in self.data
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f") + (("obs_state",) if stateful else ())}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f") + (("o",) if stateful else ())}
        return start, hist_d

    # ---- --anchor_coef (self-anchoring): dangerous transitions the episode survived ----
    def critical_mask(self, tau: float, survive: int) -> torch.Tensor:
        """(max(size - survive + 1, 0), N) bool in age order (row a = age a, 0 = oldest). The transition at age a is
        critical if its danger ratio is >= tau and none of the transitions a .. a + survive - 1 is cut (episode end
        or push): the episode went on for at least ``survive`` more steps, all of them inside the filled buffer."""
        n = self.size - survive + 1
        if n <= 0:
            return torch.zeros(0, self.N, dtype=torch.bool, device=self.device)
        rows = self._row(torch.arange(self.size, device=self.device))
        cut = self.data["cut"][rows].to(torch.int32)  # (size, N), oldest first
        C = torch.cat((torch.zeros_like(cut[:1]), cut.cumsum(0, dtype=torch.int32)), 0)  # C[a] = cuts among ages < a
        ok = (C[survive:] - C[:n]) == 0
        return ok & (self.data["rho"][rows[:n]] >= tau)

    def sample_critical(self, B: int, hist: int, tau: float, survive: int):
        """B critical transitions drawn uniformly with replacement, without a host sync, plus their observation
        history (as sample_starts). Returns (start, hist, n_valid, n_danger), the counts as 0-dim tensors: critical
        transitions, and transitions with danger >= tau. With n_valid = 0 the draws are arbitrary (the caller must
        give them zero weight)."""
        mask = self.critical_mask(tau, survive)
        rows = self._row(torch.arange(self.size, device=self.device))
        n_danger = (self.data["rho"][rows] >= tau).sum()
        if mask.shape[0] == 0:
            mask = torch.zeros(1, self.N, dtype=torch.bool, device=self.device)
        cs = mask.flatten().cumsum(0)
        n_valid = cs[-1]
        k = (torch.rand(B, device=self.device) * n_valid).long().clamp(max=(n_valid - 1).clamp(min=0))  # rank 0 .. n-1
        idx = torch.searchsorted(cs, k, right=True)  # first position whose cumulative count exceeds the rank
        idx = idx.clamp(max=cs.numel() - 1)
        return (*self.starts_at(idx // self.N, idx % self.N, hist), n_valid, n_danger)
