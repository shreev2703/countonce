"""CountOnce-Match v3: multi-view attentional GNN with a dustbin Sinkhorn head (PLAN §10.3).

One node per detection. Layers alternate within-view self-attention and cross-view attention;
both add a learned bias computed from the 3D relative position of the two detections. Pairwise
scores between views go through the learnable-dustbin Sinkhorn of ``match/sinkhorn.py``.

Nothing here depends on view *order*: the view enters only through the camera centre and the
ray direction, never through a learned index embedding, so permuting views permutes the output.

torch is imported lazily; ``DetInputs`` and ``episode_inputs`` are numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from countonce.match.sinkhorn import log_sinkhorn

NOMINAL_OBJECT_SIZE = 0.1  # world units; the synthetic generator has no per-object size
# xyz 3, size 1, box 4, score 1, lift_conf 1, ray 3, camera centre 3, log-depth 1
BASE_FEATURES = 17


@dataclass
class DetInputs:
    """Per-detection model inputs (numpy). ``view`` holds arbitrary integer view ids."""

    xyz: np.ndarray  # (N,3) centred and divided by scene_scale
    size: np.ndarray  # (N,) divided by scene_scale
    box: np.ndarray  # (N,4) cx, cy, w, h normalised by the image size
    score: np.ndarray  # (N,)
    lift_conf: np.ndarray  # (N,)
    ray: np.ndarray  # (N,3) unit world-frame ray direction
    cam: np.ndarray  # (N,3) camera centre, centred and divided by scene_scale
    view: np.ndarray  # (N,) int
    appearance: np.ndarray | None = None  # (N,A) optional

    @property
    def n(self) -> int:
        return int(self.view.shape[0])

    def features(self) -> np.ndarray:
        depth = np.linalg.norm(self.xyz - self.cam, axis=1, keepdims=True)
        cols = [
            self.xyz,
            self.size[:, None],
            self.box,
            self.score[:, None],
            self.lift_conf[:, None],
            self.ray,
            self.cam,
            np.log(depth + 1e-3),
        ]
        return np.concatenate(cols, axis=1).astype(np.float32)

    def permuted(self, order: np.ndarray) -> DetInputs:
        app = None if self.appearance is None else self.appearance[order]
        return DetInputs(
            self.xyz[order],
            self.size[order],
            self.box[order],
            self.score[order],
            self.lift_conf[order],
            self.ray[order],
            self.cam[order],
            self.view[order],
            app,
        )


FEATURE_GROUPS = ("3d", "3d_rays", "full")


def mask_inputs(d: DetInputs, group: str = "full", appearance_dim: int = 0) -> DetInputs:
    """Zero the input groups outside ``group`` (PLAN §10.5 feature-group attribution).

    ``3d``: lifted position and size only. ``3d_rays``: adds ray direction and camera centre.
    ``full``: everything. The log-depth column is derived from position and camera, so under
    ``3d`` it degrades to distance from the scene centre. When the model has an appearance
    slot (``appearance_dim`` > 0) and the detections carry none, the slot is masked with zeros.
    """
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group {group!r}; expected one of {FEATURE_GROUPS}")
    app = d.appearance
    if appearance_dim and app is None:
        app = np.zeros((d.n, appearance_dim))
    if group == "full":
        return DetInputs(d.xyz, d.size, d.box, d.score, d.lift_conf, d.ray, d.cam, d.view, app)
    keep_rays = group == "3d_rays"
    return DetInputs(
        xyz=d.xyz,
        size=d.size,
        box=np.zeros_like(d.box),
        score=np.zeros_like(d.score),
        lift_conf=np.zeros_like(d.lift_conf),
        ray=d.ray if keep_rays else np.zeros_like(d.ray),
        cam=d.cam if keep_rays else np.zeros_like(d.cam),
        view=d.view,
        appearance=None if app is None else np.zeros_like(app),
    )


def episode_inputs(ep: Any) -> DetInputs:
    """Model inputs from a ``synth.geometry.Episode`` (unprojects uv + depth with K, R, t)."""
    v = ep.view
    k, r, t = ep.K[v], ep.R[v], ep.t[v]
    pix = np.concatenate([ep.uv, np.ones((ep.n_detections, 1))], axis=1)
    ray_cam = np.einsum("nij,nj->ni", np.linalg.inv(k), pix)
    xyz_cam = ray_cam * ep.depth[:, None]
    xyz = np.einsum("nji,nj->ni", r, xyz_cam - t)  # R^T (x_cam - t)
    ray = np.einsum("nji,nj->ni", r, ray_cam)
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    cam_centres = -np.einsum("vji,vj->vi", ep.R, ep.t)

    scale = float(np.median(np.linalg.norm(xyz - cam_centres[v], axis=1))) or 1.0
    shift = xyz.mean(axis=0) if ep.n_detections else np.zeros(3)
    img = 640.0
    half = k[:, 0, 0] * NOMINAL_OBJECT_SIZE / np.maximum(ep.depth, 1e-6)
    box = np.stack([ep.uv[:, 0] / img, ep.uv[:, 1] / img, half / img, half / img], axis=1)
    return DetInputs(
        xyz=(xyz - shift) / scale,
        size=np.full(ep.n_detections, NOMINAL_OBJECT_SIZE / scale),
        box=box,
        score=ep.score.astype(float),
        lift_conf=np.ones(ep.n_detections),
        ray=ray,
        cam=(cam_centres[v] - shift) / scale,
        view=v.astype(np.int64),
    )


def _torch() -> Any:
    import torch

    return torch


def make_model(
    width: int = 128,
    layers: int = 4,
    heads: int = 4,
    appearance_dim: int = 0,
    init_bin: float = 1.0,
    sinkhorn_iters: int = 30,
) -> Any:
    """Build the v3 network (a ``torch.nn.Module``). ``layers`` alternate self / cross."""
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional
    if width % heads:
        raise ValueError("width must be divisible by heads")
    hparams = {
        "width": width,
        "layers": layers,
        "heads": heads,
        "appearance_dim": appearance_dim,
        "init_bin": init_bin,
        "sinkhorn_iters": sinkhorn_iters,
    }

    class Block(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self, cross: bool) -> None:
            super().__init__()
            self.cross = cross
            self.qkv = nn.Linear(width, 3 * width)
            self.out = nn.Linear(width, width)
            self.bias = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, heads))
            self.norm1 = nn.LayerNorm(width)
            self.norm2 = nn.LayerNorm(width)
            self.ffn = nn.Sequential(
                nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
            )

        def forward(self, x: Any, rel: Any, allow: Any, has_key: Any) -> Any:
            n = x.shape[0]
            q, k, v = self.qkv(x).view(n, 3, heads, width // heads).permute(1, 2, 0, 3)
            bias = self.bias(rel).permute(2, 0, 1)  # (H,N,N)
            bias = bias.masked_fill(~allow, -1e4)
            h = functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            h = h.permute(1, 0, 2).reshape(n, width)
            h = self.out(h) * has_key.unsqueeze(-1)  # no valid partner -> no update
            x = self.norm1(x + h)
            return self.norm2(x + self.ffn(x))

    class MatchNet(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.hparams = hparams
            self.inp = nn.Sequential(
                nn.Linear(BASE_FEATURES, width), nn.ReLU(), nn.Linear(width, width)
            )
            self.app = nn.Linear(appearance_dim, width) if appearance_dim else None
            self.blocks = nn.ModuleList([Block(cross=i % 2 == 1) for i in range(layers)])
            self.proj = nn.Linear(width, width)
            # score = <desc_i, desc_j> / sqrt(w) + MLP(relative 3D position), a geometric term
            self.pair_head = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
            self.bin_score = nn.Parameter(torch.tensor(float(init_bin)))
            self.iters = sinkhorn_iters

        def encode(self, d: DetInputs) -> Any:
            dev = self.bin_score.device
            feats = torch.as_tensor(d.features(), device=dev)
            x = self.inp(feats)
            if self.app is not None:
                if d.appearance is None:
                    raise ValueError("model has an appearance slot but inputs carry none")
                x = x + self.app(torch.as_tensor(d.appearance, dtype=torch.float32, device=dev))
            xyz = torch.as_tensor(d.xyz, dtype=torch.float32, device=dev)
            diff = xyz[None, :, :] - xyz[:, None, :]
            rel = torch.cat([diff, diff.norm(dim=-1, keepdim=True)], dim=-1)
            vw = torch.as_tensor(d.view, device=dev)
            same = vw[:, None] == vw[None, :]
            for blk in self.blocks:
                allow = same if not blk.cross else ~same
                x = blk(x, rel, allow.unsqueeze(0).expand(heads, -1, -1), allow.any(-1).float())
            return self.proj(x)

        def _pair_scores(self, desc: Any, xyz: Any, ia: Any, ib: Any) -> Any:
            diff = xyz[ib][None, :, :] - xyz[ia][:, None, :]
            dist = diff.norm(dim=-1, keepdim=True)
            # even in the offset, so swapping the two views of a pair cannot change the score
            geo = (
                self.pair_head(torch.cat([diff, dist], -1))
                + self.pair_head(torch.cat([-diff, dist], -1))
            ) / 2
            return desc[ia] @ desc[ib].T / width**0.5 + geo.squeeze(-1)

        def _pairs(self, d: DetInputs) -> Any:
            """Yield (a, b, ia, ib, scores) per non-empty view pair (a, b: sorted view ranks)."""
            desc = self.encode(d)
            xyz = torch.as_tensor(d.xyz, dtype=torch.float32, device=desc.device)
            vw = np.asarray(d.view)
            ids = sorted(set(vw.tolist()))
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    na, nb = np.flatnonzero(vw == ids[a]), np.flatnonzero(vw == ids[b])
                    if na.size and nb.size:
                        ia = torch.as_tensor(na, device=desc.device)
                        ib = torch.as_tensor(nb, device=desc.device)
                        yield a, b, na, nb, self._pair_scores(desc, xyz, ia, ib)

        def pair_log_assignments(self, d: DetInputs) -> dict[tuple[int, int], Any]:
            """Log-assignment (m+1, n+1) per view pair, keyed by positions in ``sorted(views)``."""
            return {
                (a, b): log_sinkhorn(s, self.bin_score, self.iters)
                for a, b, _, _, s in self._pairs(d)
            }

        def match_probs(self, d: DetInputs) -> Any:
            """(N,N) symmetric match probability; zero within a view and on the diagonal.

            Truncated Sinkhorn is not exactly transpose-symmetric, so both orientations are
            averaged: the result then cannot depend on how the views happen to be ordered.
            """
            p = torch.zeros(d.n, d.n, device=self.bin_score.device)
            for _, _, ia, ib, s in self._pairs(d):
                fwd = log_sinkhorn(s, self.bin_score, self.iters)[:-1, :-1].exp()
                bwd = log_sinkhorn(s.T, self.bin_score, self.iters)[:-1, :-1].exp().T
                core = (fwd + bwd) / 2
                p[np.ix_(ia, ib)] = core
                p[np.ix_(ib, ia)] = core.T
            return p

    return MatchNet()


def pair_nll(logp: Any, match: Any, unmatched_a: Any, unmatched_b: Any) -> Any:
    """NLL of the true partial assignment (incl. dustbin) for one view pair."""
    torch = _torch()
    match = torch.as_tensor(match, device=logp.device)
    ua = torch.as_tensor(unmatched_a, device=logp.device)
    ub = torch.as_tensor(unmatched_b, device=logp.device)
    total = -(logp[:-1, :-1][match].sum() + logp[:-1, -1][ua].sum() + logp[-1, :-1][ub].sum())
    count = int(match.sum() + ua.sum() + ub.sum())
    return total / max(count, 1)


def episode_loss(model: Any, ep: Any, d: DetInputs | None = None) -> Any:
    """Mean pair NLL over all view pairs of a synthetic episode."""
    d = d or episode_inputs(ep)
    ids = sorted(set(ep.view.tolist()))
    losses = []
    for (a, b), logp in model.pair_log_assignments(d).items():
        m, ua, ub = ep.pair_targets(ids[a], ids[b])
        losses.append(pair_nll(logp, m, ua, ub))
    return _torch().stack(losses).mean()


def save_checkpoint(
    model: Any, path: str | Path, meta: dict[str, Any] | None = None
) -> None:
    """Save hparams + weights; `meta` (e.g. the dataset version) rides along and is ignored by
    `load_checkpoint`. Read it back with `checkpoint_meta`."""
    blob: dict[str, Any] = {"hparams": model.hparams, "state_dict": model.state_dict()}
    if meta:
        blob["meta"] = dict(meta)
    _torch().save(blob, path)


def checkpoint_meta(path: str | Path) -> dict[str, Any]:
    blob = _torch().load(path, map_location="cpu", weights_only=True)
    return dict(blob.get("meta") or {})


def load_checkpoint(path: str | Path, device: str = "cpu") -> Any:
    torch = _torch()
    blob = torch.load(path, map_location=device, weights_only=True)
    model = make_model(**blob["hparams"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()
