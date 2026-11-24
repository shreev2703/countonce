"""Exact backbone-grid pixel mapping (PLAN 8.1): crop/pad maps, lifting, outside_recon."""

import numpy as np
import pytest
from PIL import Image

from countonce.ingest.frames import backbone_grid_map
from countonce.lift import lift_detection
from countonce.pipeline import make_frames
from countonce.types import Detection, Frame, Recon

SIZES = {"landscape": (1000, 400), "portrait": (300, 900), "square": (640, 640)}
Z, FOCAL = 5.0, 400.0


@pytest.mark.parametrize("mode", ["crop", "pad"])
@pytest.mark.parametrize("name", SIZES)
def test_round_trip_original_grid_original(name, mode):
    w, h = SIZES[name]
    gm = backbone_grid_map(w, h, mode)
    for x, y in [(0, 0), (w / 2, h / 2), (w - 1, h - 1), (w * 0.23, h * 0.71)]:
        u, v = gm.to_grid(x, y)
        x2, y2 = gm.to_original(u, v)
        assert abs(x2 - x) < 0.5 and abs(y2 - y) < 0.5
    # image centre lands on the grid centre in both modes
    u, v = gm.to_grid(w / 2, h / 2)
    assert (u, v) == (
        pytest.approx(gm.grid_width / 2, abs=1),
        pytest.approx(gm.grid_height / 2, abs=1),
    )


def test_crop_and_pad_match_vggt_load_fn_arithmetic():
    gm = backbone_grid_map(300, 900, "crop")  # new_h = round(900*518/300/14)*14 = 1554
    assert (gm.grid_width, gm.grid_height) == (518, 518)
    assert gm.sy == pytest.approx(1554 / 900) and gm.oy == -(1554 - 518) // 2
    gm = backbone_grid_map(300, 900, "pad")  # new_h=518, new_w=round(300*518/900/14)*14=173->168
    assert gm.sx == pytest.approx(168 / 300) and gm.ox == (518 - 168) // 2 and gm.oy == 0
    gm = backbone_grid_map(1000, 400, "pad")  # new_h=210, padded to 518 with 154 rows on top
    assert gm.sy == pytest.approx(210 / 400) and gm.oy == (518 - 210) // 2
    with pytest.raises(ValueError):
        backbone_grid_map(10, 10, "stretch")


def test_make_frames_fills_grid_map(tmp_path):
    p = tmp_path / "tall.png"
    Image.new("RGB", (300, 900)).save(p)
    (f,) = make_frames([p], "crop")
    assert f.grid_map == backbone_grid_map(300, 900, "crop")
    assert make_frames([p])[0].grid_map is None


def _plane_recon(gm):
    """Fronto-parallel plane at depth Z seen by a pinhole camera on the backbone grid."""
    gw, gh = gm.grid_width, gm.grid_height
    ys, xs = np.mgrid[0:gh, 0:gw]
    pts = np.stack([(xs + 0.5 - gw / 2) / FOCAL * Z, (ys + 0.5 - gh / 2) / FOCAL * Z,
                    np.full((gh, gw), Z)], -1)  # fmt: skip
    k = np.array([[FOCAL, 0, gw / 2], [0, FOCAL, gh / 2], [0, 0, 1.0]])
    return Recon(
        backbone="vggt", intrinsics=k[None], extrinsics=np.eye(4)[None, :3],
        depth=np.full((1, gh, gw), Z, np.float32), depth_conf=np.full((1, gh, gw), 5.0, np.float32),
        world_points=pts[None].astype(np.float32),
        world_points_conf=np.full((1, gh, gw), 5.0, np.float32),
    )  # fmt: skip


@pytest.mark.parametrize("mode", ["crop", "pad"])
def test_lift_on_portrait_plane_hits_right_3d_point(mode):
    w, h = SIZES["portrait"]
    gm = backbone_grid_map(w, h, mode)
    recon = _plane_recon(gm)
    frame = Frame(0, "x.png", w, h, "h", grid_map=gm)
    cx, cy = 100.0, 520.0  # off-centre in both axes
    det = Detection("0:0", 0, 0.9, box_xyxy=np.array([cx - 20, cy - 20, cx + 20, cy + 20]))
    ld = lift_detection(det, recon, 0, frame, min_pixels=5)
    assert ld.ok
    u, v = gm.to_grid(cx, cy)
    expect = np.array(
        [(u - gm.grid_width / 2) / FOCAL * Z, (v - gm.grid_height / 2) / FOCAL * Z, Z]
    )
    assert np.abs(ld.xyz - expect).max() < 0.03
    # the old scale-only mapping lands visibly off for the same detection
    naive = lift_detection(det, recon, 0, Frame(0, "x.png", w, h, "h"), min_pixels=5)
    assert naive.ok and np.abs(naive.xyz - expect).max() > 0.2


def test_detection_in_cropped_region_is_outside_recon():
    w, h = SIZES["portrait"]  # crop removes 518 px rows top and bottom (grid rows)
    gm = backbone_grid_map(w, h, "crop")
    recon = _plane_recon(gm)
    frame = Frame(0, "x.png", w, h, "h", grid_map=gm)
    box = Detection("0:0", 0, 0.9, box_xyxy=np.array([100.0, 10.0, 200.0, 60.0]))
    pt = Detection("0:1", 0, 0.9, point_xy=np.array([150.0, 880.0]))
    inside = Detection("0:2", 0, 0.9, box_xyxy=np.array([100.0, 400.0, 200.0, 500.0]))
    assert lift_detection(box, recon, 0, frame).fail_reason == "outside_recon"
    assert lift_detection(pt, recon, 0, frame).fail_reason == "outside_recon"
    assert lift_detection(inside, recon, 0, frame).ok
    # padding is also outside the recon
    gp = backbone_grid_map(w, h, "pad")
    pad_frame = Frame(0, "x.png", w, h, "h", grid_map=gp)
    left_pad = Detection("0:3", 0, 0.9, point_xy=np.array([-10.0, 450.0]))
    assert lift_detection(left_pad, _plane_recon(gp), 0, pad_frame).fail_reason == "outside_recon"


def test_v2_reprojection_uses_inverse_map():
    from countonce.merge.match import _Views

    w, h = SIZES["portrait"]
    gm = backbone_grid_map(w, h, "crop")
    geo = _Views(_plane_recon(gm), {0: Frame(0, "x.png", w, h, "h", grid_map=gm)}, None)
    u, v = gm.to_grid(120.0, 700.0)
    x, y = geo.to_frame(0, np.array([u, v]))
    assert abs(x - 120.0) < 0.5 and abs(y - 700.0) < 0.5
