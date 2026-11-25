"""CPU diagnostic for the Fuji Core protocol (sixteenth audit): replays cached windows.

Run from the repo root with ``CUDA_VISIBLE_DEVICES= PYTHONPATH=.:scripts .venv-gpu/bin/python
scripts/diag_fuji_protocol.py Tree06:west:5 ...``. Everything is read from ``runs/fuji_core/cache``
(a cache miss raises: no GPU inference); dumps go to ``runs/fuji_core/diag/`` (git-ignored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from run_fuji_core import PROMPT, WORK, _window, base_config, load_recon

from bench import fuji_core as fc
from bench.datasets.fuji_sfm import load_fuji_sfm
from countonce.pipeline import build_detector, build_reconstructor, run
from countonce.recon.align import camera_centers, transform_extrinsics
from countonce.recon.chunked import align_chunk

DIAG = WORK / "diag"


def _in(p: np.ndarray, recon: object, ext: np.ndarray, v: int, w: int, h: int) -> bool:
    from bench.gt_matching import project_points

    uv, z = project_points(p[None], recon.intrinsics[v], ext[v])
    return bool(z[0] > 0 and 0 <= uv[0, 0] < w - 1 and 0 <= uv[0, 1] < h - 1)


def main(specs: list[str]) -> None:
    scene = load_fuji_sfm("data/fuji_sfm", validate=False)
    imgs = fc.prepare_images(scene, WORK / "img")
    ov = {s: json.loads((WORK / f"overview_{s}.json").read_text()) for s in ("east", "west")}
    box = fc.roi_box(scene.gt_xyz)
    base = base_config(score_threshold=0.2, merge_k=0.3)
    import torch

    torch.cuda.get_device_capability = lambda *a, **k: (12, 0)  # CUDA hidden: pick bf16 as in the run
    det, rec = build_detector(base), build_reconstructor(base)

    def _no_infer(*a: object, **k: object) -> None:
        raise RuntimeError("cache miss: this diagnostic never runs a model")

    det.detect = _no_infer  # type: ignore[method-assign]
    rec.method.reconstruct = _no_infer  # type: ignore[method-assign]
    DIAG.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        tree, side, k = spec.split(":")
        start, stop, paths, gt_ext = _window(scene, imgs, ov, side, tree, int(k))
        cfg = base_config(assoc="v1", score_threshold=0.2, merge_k=0.3)
        res = run(paths, PROMPT, cfg, detector=det, reconstructor=rec, export=False)
        recon = load_recon(cfg, res.recon_cache_key or "")
        sim = align_chunk(gt_ext, recon.extrinsics)
        ext = transform_extrinsics(recon.extrinsics, sim)
        depth = recon.depth * sim.s
        vis = fc.visible_gt(scene.gt_xyz, recon.intrinsics, ext, depth)
        c_gt, c_al = camera_centers(gt_ext), camera_centers(ext)
        from bench.gt_matching import project_points

        h, w = depth.shape[1:]
        tmask = scene.tree_mask(tree)
        info = {
            "spec": spec, "frames": [start, stop], "scale": sim.s,
            "cam_rms_m": float(np.sqrt(((c_gt - c_al) ** 2).sum(1).mean())),
            "cam_y": c_gt[:, 1].tolist(), "tree_y": float(scene.gt_xyz[tmask][:, 1].mean()),
            "n_inst": len(res.instances), "n_det": len(res.detections),
            "n_vis_per_view": [len(v) for v in vis], "depth_med": float(np.median(depth)),
            "views": [],
        }  # fmt: skip
        for v in range(recon.n_views):
            uv, z = project_points(scene.gt_xyz, recon.intrinsics[v], ext[v])
            inf = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            near = inf & (z < fc.GT_MAX_DEPTH_M)
            surf = np.full(len(z), np.nan)
            ii = np.flatnonzero(near & (uv[:, 0] < w - 1) & (uv[:, 1] < h - 1))
            surf[ii] = depth[v, uv[ii, 1].astype(int), uv[ii, 0].astype(int)]
            dd = [d for d in res.detections if d.view == v and d.uv is not None]
            info["views"].append({
                "in_image": int(inf.sum()), "in_depth": int(near.sum()), "visible": len(vis[v]),
                "z_med_in_image": float(np.median(z[inf])) if inf.any() else None,
                "z_minus_surf_med": float(np.nanmedian(z - surf)) if np.isfinite(surf).any() else None,
                "n_dets": len(dd),
            })  # fmt: skip
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                if v in (0, recon.n_views // 2):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(depth[v], cmap="gray")
                    ax.scatter(*uv[inf].T, s=8, c="r", label="GT in image")
                    ax.scatter(*uv[near].T, s=8, c="y", label=f"<{fc.GT_MAX_DEPTH_M} m")
                    if vis[v].size:
                        ax.scatter(*uv[vis[v]].T, s=10, c="lime", label="visible")
                    if dd:
                        ax.scatter(*np.array([d.uv for d in dd]).T, s=14, c="c", marker="x", label="det")
                    ax.legend(fontsize=6)
                    fig.savefig(DIAG / f"{tree}_{side}_k{k}_v{v}.png", dpi=110)
                    plt.close(fig)
            except Exception as e:  # plotting is best-effort
                info["plot_error"] = str(e)
        ixyz = sim.apply(np.array([i.xyz for i in res.instances])) if res.instances else np.zeros((0, 3))
        info["roi_box"] = [box[0].tolist(), box[1].tolist()]
        info["inst_in_roi"] = int(fc.in_roi(ixyz, box).sum())
        if len(ixyz):
            info["inst_xyz_min_max"] = [ixyz.min(0).tolist(), ixyz.max(0).tolist()]
        info["gt_min_max"] = [scene.gt_xyz.min(0).tolist(), scene.gt_xyz.max(0).tolist()]
        info["cam_centres_gt"] = c_gt.tolist()
        info["cam_centres_aligned"] = c_al.tolist()
        rr = np.concatenate([
            np.array([project_points(scene.gt_xyz[[i]], recon.intrinsics[v], ext[v])[1][0]
                      / max(float(depth[v, int(project_points(scene.gt_xyz[[i]], recon.intrinsics[v], ext[v])[0][0, 1]),
                                       int(project_points(scene.gt_xyz[[i]], recon.intrinsics[v], ext[v])[0][0, 0])]), 1e-9)
                      for i in np.flatnonzero(tmask) if _in(scene.gt_xyz[i], recon, ext, v, w, h)])
            for v in range(recon.n_views)
        ] or [np.zeros(0)])
        info["tree_z_over_surf_p10_p50"] = [float(np.percentile(rr, 10)), float(np.median(rr))] if len(rr) else None
        info["cam_span_m"] = float(np.linalg.norm(np.ptp(c_gt, axis=0)))
        (DIAG / f"{tree}_{side}_k{k}.json").write_text(json.dumps(info, indent=1))
        print(spec, {a: b for a, b in info.items() if a not in ("views", "cam_y", "cam_centres_gt", "cam_centres_aligned")})



if __name__ == "__main__":
    main(sys.argv[1:])
