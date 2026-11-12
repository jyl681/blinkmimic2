"""Actor objective over an imagined horizon: discounted rewards with survival weights, closed by the critic."""

from __future__ import annotations

import torch


def actor_objective(r: torch.Tensor, d: torch.Tensor, v_boot: torch.Tensor | None, gamma: float, boot_coef: float = 1.0,
                    loop_cumprod: bool = False):  # fmt: skip
    """SHAC objective J = sum_{t<H} gamma^t c_t r_t + boot_coef * gamma^H c_H V(z_H), c_t = prod_{k<t} (1 - d_k).
    r, d: (H, B); v_boot: (B,) critic value of the last imagined state, or None (--no_critic). Returns (J, c_t, c_H).
    boot_coef = 1 runs exactly the operations of train.py before this option (bitwise identical).
    ``loop_cumprod`` (--cuda_graph): the survival products as a loop of multiplications (same products; cumprod's
    backward tests for zeros on the host, which a CUDA graph cannot capture)."""
    H, B = r.shape
    if loop_cumprod:
        acc, cs = torch.ones_like(d[0]), []
        for t in range(H):
            acc = acc * (1 - d[t])
            cs.append(acc)
        alive = torch.stack(cs)
    else:
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=r.device), alive[:-1]), 0)
    disc = gamma ** torch.arange(H, device=r.device, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    if v_boot is not None:
        boot = (gamma**H) * c * v_boot
        J = J + (boot if boot_coef == 1.0 else boot_coef * boot)
    return J, c_t, c


def clip_sample_grad(x: torch.Tensor, chi: float, stats: list | None = None) -> torch.Tensor:
    """Identity in the forward pass; in the backward pass each sample's gradient (dim 0 = batch) is scaled down to at
    most ``chi`` times the batch median of the per-sample gradient norms (samples that still carry a gradient).
    ``stats``: the hook appends (number of clipped samples, number of live samples) as 0-dim tensors (no host sync)."""
    if not x.requires_grad:
        return x

    def hook(g):
        if g is None:  # a state component the rest of the graph does not use (seen under torch.compile)
            return g
        n = g.flatten(1).float().norm(dim=1)
        # median over the samples that still carry a gradient: samples that died earlier in the imagined
        # horizon have (near-)zero gradients and would otherwise drag the cap to ~0 for everyone
        live = n > 1e-6 * n.max()
        cap = chi * torch.nanmedian(torch.where(live, n, torch.nan))
        cap = torch.where(torch.isfinite(cap), cap, torch.inf)  # no live sample: leave the gradient as is
        s = torch.clamp(cap / n.clamp(min=1e-12), max=1.0)
        if stats is not None:
            stats.append(((live & (s < 1.0)).sum(), live.sum()))
        return g * s.view(-1, *([1] * (g.dim() - 1))).to(g.dtype)

    x.register_hook(hook)
    return x
