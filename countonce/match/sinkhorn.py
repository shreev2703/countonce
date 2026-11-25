"""Log-space Sinkhorn with a learnable dustbin (SuperGlue-style), PLAN §10.3.

torch is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    import torch

    return torch


def log_sinkhorn(scores: Any, bin_score: Any, iters: int = 50) -> Any:
    """Augment `scores` (..., M, N) with a dustbin row/column and normalise in log space.

    Returns the log-assignment, shape (..., M+1, N+1), scaled as in SuperGlue: each real row and
    column sums to 1 in probability space, the dustbin row sums to N and the dustbin column to M.
    """
    torch = _torch()
    m, n = scores.shape[-2:]
    bs = bin_score
    row = bs.expand(*scores.shape[:-2], m, 1)
    col = bs.expand(*scores.shape[:-2], 1, n)
    corner = bs.expand(*scores.shape[:-2], 1, 1)
    z = torch.cat([torch.cat([scores, row], -1), torch.cat([col, corner], -1)], -2)

    norm = -torch.log(torch.tensor(float(m + n), dtype=z.dtype, device=z.device))
    log_mu = torch.cat([norm.expand(m), (norm + torch.log(torch.tensor(float(n)))).expand(1)])
    log_nu = torch.cat([norm.expand(n), (norm + torch.log(torch.tensor(float(m)))).expand(1)])
    log_mu = log_mu.to(z).expand(*z.shape[:-2], m + 1)
    log_nu = log_nu.to(z).expand(*z.shape[:-2], n + 1)

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(-2), dim=-1)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(-1), dim=-2)
    return z + u.unsqueeze(-1) + v.unsqueeze(-2) - norm


def log_sinkhorn_padded(
    scores: Any, ma: Any, mb: Any, m: Any, n: Any, bin_score: Any, iters: int = 50
) -> Any:
    """Batched ``log_sinkhorn`` over pairs padded to a common (M, N).

    ``scores`` is (P, M, N); ``ma`` (P, M) and ``mb`` (P, N) mark the real rows/columns and ``m``,
    ``n`` (P,) are their float counts. The dustbin sits at fixed index M / N, so the result is
    (P, M+1, N+1); a pair's real block is ``[:m, :n]`` plus row M / column N. Padded entries are
    pushed to a large negative value so they drop out of every logsumexp. For each pair the
    real entries match ``log_sinkhorn`` on the unpadded scores.
    """
    torch = _torch()
    p, mm, nn = scores.shape
    bs = bin_score
    s = scores.to(bs.dtype)
    neg = -1e9
    valid = ma.unsqueeze(-1) & mb.unsqueeze(-2)
    row = torch.where(ma, bs, bs.new_tensor(neg)).unsqueeze(-1)
    col = torch.where(mb, bs, bs.new_tensor(neg)).unsqueeze(-2)
    top = torch.cat([torch.where(valid, s, bs.new_tensor(neg)), row], -1)
    bottom = torch.cat([col, bs.expand(p, 1, 1)], -1)
    z = torch.cat([top, bottom], -2)

    rm = torch.cat([ma, ma.new_ones(p, 1)], -1)
    cm = torch.cat([mb, mb.new_ones(p, 1)], -1)
    norm = -torch.log(m + n).to(z.dtype)  # (P,)
    zeros_m, zeros_n = z.new_zeros(p, mm), z.new_zeros(p, nn)
    log_mu = norm[:, None] + torch.cat([zeros_m, torch.log(n).to(z.dtype)[:, None]], -1)
    log_nu = norm[:, None] + torch.cat([zeros_n, torch.log(m).to(z.dtype)[:, None]], -1)

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = torch.where(rm, log_mu - torch.logsumexp(z + v.unsqueeze(-2), dim=-1), 0.0)
        v = torch.where(cm, log_nu - torch.logsumexp(z + u.unsqueeze(-1), dim=-2), 0.0)
    return z + u.unsqueeze(-1) + v.unsqueeze(-2) - norm[:, None, None]


class DustbinSinkhorn:
    """Module factory: built lazily so importing this file never needs torch."""

    def __new__(cls, iters: int = 50, init_bin: float = 1.0) -> Any:
        torch = _torch()
        module_base: Any = torch.nn.Module  # torch is an optional dependency

        class _Module(module_base):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.iters = iters
                self.bin_score = torch.nn.Parameter(torch.tensor(float(init_bin)))

            def forward(self, scores: Any) -> Any:
                return log_sinkhorn(scores, self.bin_score, self.iters)

        return _Module()
