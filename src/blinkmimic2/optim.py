"""Optimizers: Muon (hidden matrices) + Adam, and SOAP.

- ``MuonAdam``: Muon (Jordan et al. 2024; torch.optim.Muon) on the hidden weight matrices of the given MLPs, Adam on
  their first and last Linear and on all vectors. Muon's step is rescaled to the update RMS of Adam
  (adjust_lr_fn='match_rms_adamw', Liu et al. 2025), so the Adam learning rate is reused unchanged.
- ``SOAP`` (Vyas et al. 2024, "SOAP: Improving and Stabilizing Shampoo using Adam"): for every weight matrix, Adam run
  in the eigenbasis of Shampoo's Kronecker factors L = E[G G^T], R = E[G^T G]; the bases are refreshed every
  ``freq`` steps (eigh the first time, then one power-iteration step + QR, the paper's Algorithm 4). Vectors get Adam.
  With the bases held at the identity it is exactly Adam.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _linears(m: nn.Module) -> list[nn.Linear]:
    return [x for x in m.modules() if isinstance(x, nn.Linear)]


class MuonAdam:
    """Muon on hidden matrices of ``mlps`` (all Linear layers except each MLP's first and last), Adam elsewhere.

    ``capturable`` (--cuda_graph): the learning rate is a 0-dim device tensor (set it with ``fill_``), Adam is the
    capturable fused kernel, and the Muon step is done here with the same arithmetic as torch.optim.Muon but with the
    tensor learning rate (torch.optim.Muon reads a tensor lr on the host, so a captured step would freeze it)."""

    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95), fused: bool = True,
                 capturable: bool = False):  # fmt: skip
        hidden = {id(l.weight) for m in mlps for l in _linears(m)[1:-1]}
        muon = [p for p in model.parameters() if id(p) in hidden]
        adam = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
        self.capturable = capturable
        if capturable:
            dev = muon[0].device
            self.muon = torch.optim.Muon(muon, lr=torch.tensor(float(lr), device=dev), weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
            self.adam = torch.optim.Adam(adam, lr=torch.tensor(float(lr), device=dev), betas=betas, fused=True, capturable=True)
        else:
            self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
            self.adam = torch.optim.Adam(adam, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self):
        # a property: Optimizer.load_state_dict replaces the group dicts, so a list built in __init__ goes stale on --resume
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none), self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _muon_step_tensor_lr(self, ok: torch.Tensor | None = None):
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        for g in self.muon.param_groups:
            assert g["weight_decay"] == 0.0
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.muon.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                buf = st["momentum_buffer"]
                if ok is None:
                    buf.lerp_(p.grad, 1 - g["momentum"])
                else:
                    buf.copy_(torch.where(ok, buf.lerp(p.grad, 1 - g["momentum"]), buf))
                update = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                update = _zeropower_via_newtonschulz(update, g["ns_coefficients"], g["ns_steps"], g["eps"])
                ratio = _adjust_lr(1.0, g["adjust_lr_fn"], p.shape)
                d = update.to(p.dtype) * (g["lr"] * -ratio)  # fp32 product (a bf16 update x 0-dim tensor stays bf16)
                p.add_(d if ok is None else torch.where(ok, d, 0.0))

    @torch.no_grad()
    def step(self, ok: torch.Tensor | None = None):
        """``ok``: optional 0-dim bool device tensor; where it is false the whole step is skipped without a host read
        (parameters, momenta, Adam moments and step counts unchanged). The gradients must be finite (zero them first)."""
        if ok is not None:
            if self.capturable:
                self._muon_step_tensor_lr(ok)
            else:  # torch.optim.Muon's own step (unchanged arithmetic), undone where ok is false
                ps = [p for g in self.muon.param_groups for p in g["params"] if p.grad is not None]
                bufs = lambda: [self.muon.state[p]["momentum_buffer"] for p in ps]  # noqa: E731
                for p in ps:
                    self.muon.state[p].setdefault("momentum_buffer", torch.zeros_like(p.grad, memory_format=torch.preserve_format))
                keep = [t.clone() for t in ps + bufs()]
                self.muon.step()
                for t, k in zip(ps + bufs(), keep):
                    t.copy_(torch.where(ok, t, k))
            assert self.adam.defaults["fused"], "a gated step needs the fused Adam (--fused_adam)"
            self.adam.found_inf = (~ok).float()  # the fused kernel skips the step and undoes the step count
            try:
                self.adam.step()
            finally:
                self.adam.found_inf = None
            return
        if self.capturable:
            self._muon_step_tensor_lr()
        else:
            self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd: dict):
        lrs = [g["lr"] for g in self.param_groups]
        self.muon.load_state_dict(sd["muon"]), self.adam.load_state_dict(sd["adam"])
        if self.capturable:  # keep the device learning-rate tensors (the training loop fills them)
            for g, lr in zip(self.param_groups, lrs):
                lr.fill_(float(g["lr"]))
                g["lr"] = lr


class SOAP(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas=(0.9, 0.95), shampoo_beta: float = 0.95, eps: float = 1e-8, freq: int = 10,
                 max_dim: int = 4096):  # fmt: skip
        super().__init__(params, dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps, freq=freq, max_dim=max_dim))

    @staticmethod
    def _basis(P: torch.Tensor, Q: torch.Tensor | None) -> torch.Tensor:
        if Q is None:
            return torch.linalg.eigh(P + 1e-30 * torch.eye(P.shape[0], device=P.device, dtype=P.dtype))[1].flip(-1)
        return torch.linalg.qr(P @ Q)[0]  # one power-iteration step from the previous basis

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                G = p.grad.float()
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(G)
                    st["v"] = torch.zeros_like(G)
                    if G.dim() == 2 and max(G.shape) <= g["max_dim"]:
                        st["L"] = torch.zeros(G.shape[0], G.shape[0], device=G.device)
                        st["R"] = torch.zeros(G.shape[1], G.shape[1], device=G.device)
                        st["QL"] = st["QR"] = None
                mat = "L" in st
                # the first step only accumulates the factors (paper: no update before the first basis)
                if mat:
                    st["L"].lerp_(G @ G.T, 1 - g["shampoo_beta"])
                    st["R"].lerp_(G.T @ G, 1 - g["shampoo_beta"])
                    if st["QL"] is None:
                        st["QL"], st["QR"] = self._basis(st["L"], None), self._basis(st["R"], None)
                        continue
                st["t"] += 1
                t = st["t"]
                st["m"].lerp_(G, 1 - b1)  # momentum in the original space, projected each step
                Gr = st["QL"].T @ G @ st["QR"] if mat else G
                st["v"].lerp_(Gr * Gr, 1 - b2)
                mr = st["QL"].T @ st["m"] @ st["QR"] if mat else st["m"]
                step_r = (mr / (1 - b1**t)) / ((st["v"] / (1 - b2**t)).sqrt() + g["eps"])
                upd = st["QL"] @ step_r @ st["QR"].T if mat else step_r
                p.add_(upd.to(p.dtype), alpha=-g["lr"])
                if mat and t % g["freq"] == 0:
                    # official implementation: order the current basis by the estimated eigenvalues diag(Q^T P Q)
                    # (and the second moment with it), then one power-iteration step + QR
                    for key, dim in (("L", 0), ("R", 1)):
                        Q = st["Q" + key]
                        idx = torch.argsort(((st[key] @ Q) * Q).sum(0), descending=True)
                        st["v"] = st["v"].index_select(dim, idx)
                        st["Q" + key] = self._basis(st[key], Q[:, idx])


def check():
    """Unit checks: SOAP with identity bases equals Adam; SOAP beats Adam on an ill-conditioned matrix least squares;
    MuonAdam puts exactly the hidden matrices on Muon."""
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. identity bases: SOAP == Adam (bias-corrected) step for step
    W0 = torch.randn(16, 8, device=dev)
    X = torch.randn(64, 8, device=dev)
    Y = torch.randn(64, 16, device=dev)
    w1, w2 = nn.Parameter(W0.clone()), nn.Parameter(W0.clone())
    adam = torch.optim.Adam([w1], lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    soap = SOAP([w2], lr=1e-2, betas=(0.9, 0.95), freq=10**9)
    soap.state[w2].update(t=0, m=torch.zeros_like(W0), v=torch.zeros_like(W0), L=torch.zeros(16, 16, device=dev),
                          R=torch.zeros(8, 8, device=dev), QL=torch.eye(16, device=dev), QR=torch.eye(8, device=dev))  # fmt: skip
    for _ in range(30):
        for w, o in ((w1, adam), (w2, soap)):
            o.zero_grad()
            ((X @ w.T - Y) ** 2).mean().backward()
            o.step()
    chk("SOAP with identity bases equals Adam", torch.allclose(w1, w2, atol=1e-5), f"max diff {float((w1 - w2).abs().max()):.1e}")
    # 2. Kronecker-structured quadratic 0.5 tr((W-T)^T A (W-T) B), A and B ill-conditioned with random eigenbases
    #    (the structure Shampoo / SOAP model): SOAP converges, and to a lower loss than Adam at the same lr
    torch.backends.cuda.matmul.allow_tf32 = False

    def spd(k):
        Q = torch.linalg.qr(torch.randn(k, k, device=dev))[0]
        return Q @ torch.diag(torch.logspace(0, -2, k, device=dev)) @ Q.T

    A, B, T = spd(32), spd(24), torch.randn(32, 24, device=dev)
    res = {}
    for name in ("adam", "soap"):
        w = nn.Parameter(torch.zeros(32, 24, device=dev))
        o = torch.optim.Adam([w], lr=1e-2, betas=(0.9, 0.95)) if name == "adam" else SOAP([w], lr=1e-2, betas=(0.9, 0.95))
        for _ in range(1000):
            o.zero_grad()
            (0.5 * ((w - T) * (A @ (w - T) @ B)).sum()).backward()
            o.step()
        res[name] = float(0.5 * ((w - T) * (A @ (w - T) @ B)).sum())
    chk("SOAP converges on a Kronecker quadratic (start 22)", res["soap"] < 1e-3 and res["soap"] < res["adam"], f"{res}")
    # 3. MuonAdam split
    from blinkmimic2.networks import Policy

    pi = Policy(20, 30, 5, hist=3, enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16))
    oa = MuonAdam(pi, [pi.enc, pi.dec], lr=1e-3, fused=False)
    n_muon = sum(len(g["params"]) for g in oa.muon.param_groups)
    n_adam = sum(len(g["params"]) for g in oa.adam.param_groups)
    chk("MuonAdam: hidden matrices on Muon (2 per MLP), the rest on Adam", n_muon == 4 and n_muon + n_adam == len(list(pi.parameters())),
        f"muon {n_muon}, adam {n_adam}")  # fmt: skip
    # 4. capturable MuonAdam (tensor lr, own Muon step, capturable fused Adam) == MuonAdam, eagerly and inside a CUDA graph.
    #    Muon orthogonalizes in bf16, so a 1e-7 difference flips bf16 roundings after a few steps; the yardstick is the
    #    spread of plain MuonAdam itself under a 1e-6 relative lr perturbation.
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        pis = [Policy(20, 30, 5, hist=3, enc_hidden=(64, 48, 16), dec_hidden=(64, 32, 16)).to(dev) for _ in range(4)]
        for q in pis[1:]:
            q.load_state_dict(pis[0].state_dict())
        init = [p.detach().clone() for p in pis[0].parameters()]
        mk = lambda q, cap: MuonAdam(q, [q.enc, q.dec], lr=1e-3, capturable=cap)  # noqa: E731
        opts = [mk(pis[0], False), mk(pis[1], True), mk(pis[2], True), mk(pis[3], False)]
        xo, xg = torch.randn(256, 3, 20, device=dev), torch.randn(256, 30, device=dev)
        lrs = [1e-3 * (1 - 0.02 * i) for i in range(20)]

        def stepf(k):
            opts[k].zero_grad(set_to_none=True)
            (pis[k](xo, xg) ** 2).mean().backward()
            opts[k].step()

        graph = None
        for i, lr in enumerate(lrs):
            for k in range(4):
                lr_k = lr * (1 + 1e-6) if k == 3 else lr
                for gr in opts[k].param_groups:
                    gr["lr"].fill_(lr_k) if torch.is_tensor(gr["lr"]) else gr.__setitem__("lr", lr_k)
            stepf(0), stepf(1), stepf(3)
            if i < 3:  # warm-up on a side stream, then capture on the next step and replay
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stepf(2)
                torch.cuda.current_stream().wait_stream(side)
            else:
                if graph is None:
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        stepf(2)
                graph.replay()

        def rel(k):  # max over tensors of |p_0 - p_k| / |total change of p_0|
            return max(float((p.detach() - q.detach()).norm() / (p.detach() - p0).norm().clamp(min=1e-12))
                       for p, q, p0 in zip(pis[0].parameters(), pis[k].parameters(), init))  # fmt: skip

        floor = rel(3)
        chk("capturable MuonAdam == MuonAdam (20 steps, changing lr)", rel(1) < max(3 * floor, 1e-4),
            f"max diff / total change {rel(1):.1e} (noise floor {floor:.1e})")  # fmt: skip
        chk("capturable MuonAdam in a CUDA graph == MuonAdam", rel(2) < max(3 * floor, 1e-4), f"{rel(2):.1e}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
