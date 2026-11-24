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
