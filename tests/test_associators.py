"""v2 associator, shared Associator protocol and Hypothesis properties (PLAN §8.6, §18 v4)."""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from countonce.merge.appearance import FeatureExtractor, cosine_sim
from countonce.merge.base import Associator
from countonce.merge.cluster import V1Associator
from countonce.merge.match import V2Associator, V2Config, _Views
from countonce.types import Detection, LiftedDetection, Recon

H, W, F, Z, SIZE = 300, 400, 200.0, 10.0, 0.6


def make_recon(n_views, z=Z, depth=None):
    k = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    ext = np.zeros((n_views, 3, 4))
    for v in range(n_views):
        ext[v, :, :3] = np.eye(3)
        ext[v, 0, 3] = -0.2 * v  # camera slides along +x
    d = np.full((n_views, H, W), z) if depth is None else depth
    return Recon(
        backbone="vggt",
        intrinsics=np.tile(k, (n_views, 1, 1)),
        extrinsics=ext,
        depth=d,
        depth_conf=np.full((n_views, H, W), 5.0),
        world_points=np.zeros((n_views, H, W, 3)),
        world_points_conf=np.full((n_views, H, W), 5.0),
    )


def _lifted(det_id, frame, xyz, view_tx, box_noise=None, conf=0.9):
    cam = np.asarray(xyz, float) - [view_tx, 0, 0]
    c = np.array([F * cam[0] / cam[2] + W / 2, F * cam[1] / cam[2] + H / 2])
    half = SIZE * F / cam[2] / 2
    if box_noise is not None:
        c = c + box_noise
    box = np.array([c[0] - half, c[1] - half, c[0] + half, c[1] + half])
    det = Detection(id=det_id, frame_id=frame, score=conf, box_xyxy=box)
    return LiftedDetection(det=det, xyz=np.asarray(xyz, float), size=SIZE, lift_conf=conf)


def make_scene(seed, n_obj=15, n_views=5, noise=0.03, drop=0.25, dup=0.0, fp=1):
    """Objects on a jittered grid seen by sliding cameras; returns (recon, lifted, truth_ids)."""
    rng = np.random.default_rng(seed)
    cols = 5
    base = np.array([(i % cols, i // cols) for i in range(n_obj)], float) * 3.0 - [6.0, 4.0]
    pts = np.column_stack([base + rng.uniform(-0.3, 0.3, (n_obj, 2)), np.full(n_obj, Z)])
    lifted, truth, count = [], {}, 0

    def add(v, o, xyz):
        nonlocal count
        did = f"{v}:{count}"
        count += 1
        ld = _lifted(did, v, xyz, 0.2 * v)
        lifted.append(ld)
        truth[did] = o

    for v in range(n_views):
        for o in range(n_obj):
            if rng.random() < drop:
                continue
            add(v, o, pts[o] + rng.normal(0, noise * SIZE, 3) * [1, 1, 0.2])
            if rng.random() < dup:  # double detection of the same object in the same frame
                add(v, o, pts[o] + rng.normal(0, 0.1 * SIZE, 3) * [1, 1, 0.2])
        for _ in range(fp):  # isolated false positives
            add(v, -1 - count, np.array([rng.uniform(-9, 9), rng.uniform(-6, 6), Z]))
    return make_recon(n_views), lifted, truth


def partition(res):
    return {frozenset(i.members) for i in res.instances}


def assocs():
    return [V1Associator(k=0.5), V2Associator()]


# --- protocol / basics
def test_protocol_shared():
    assert all(isinstance(a, Associator) for a in assocs())
    assert [a.name for a in assocs()] == ["v1", "v2"]


@pytest.mark.parametrize("noise", [0.0, 0.03, 0.08])
def test_v2_synthetic_geometry_recovers_count(noise):
    recon, lifted, truth = make_scene(0, n_obj=20, n_views=6, noise=noise, fp=2)
    res = V2Associator().associate(lifted, recon)
    seen = {o for o in truth.values() if o >= 0}
    confident = [i for i in res.instances if i.n_views >= 2]
    assert abs(len(confident) - len(seen)) <= 0.1 * 20
    for inst in res.instances:
        frames = [m.split(":")[0] for m in inst.members]
        assert len(frames) == len(set(frames))
    # purity: members of one instance share a ground-truth object
    for inst in confident:
        assert len({truth[m] for m in inst.members}) == 1


def test_v2_fills_reprojection_evidence():
    recon, lifted, _ = make_scene(1, noise=0.03)
    res = V2Associator().associate(lifted, recon)
    merged = [e for e in res.evidence if e.merged]
    assert merged and all(e.reason == "matched" for e in merged)
    assert all(e.reproj_error_px >= 0 for e in res.evidence)
    assert any(e.reproj_error_px > 0 for e in merged)
    assert all(0.0 < e.match_prob <= 1.0 for e in merged)


def test_v2_same_frame_never_merged_even_if_coincident():
    recon = make_recon(2)
    a = _lifted("0:0", 0, [0, 0, Z], 0.0)
    b = _lifted("0:1", 0, [0, 0, Z], 0.0)
    c = _lifted("1:0", 1, [0, 0, Z], 0.2)
    res = V2Associator().associate([a, b, c], recon)
    assert len(res.instances) == 2
    assert all(len({m.split(":")[0] for m in i.members}) == len(i.members) for i in res.instances)


def test_v2_failed_lifts_rejected():
    recon = make_recon(1)
    bad = LiftedDetection(
        det=Detection(id="0:0", frame_id=0, score=0.1, point_xy=np.zeros(2)), fail_reason="low_conf"
    )
    res = V2Associator().associate([bad], recon)
    assert res.detection_status == {"0:0": "rejected"} and not res.instances


def test_v2_requires_recon():
    with pytest.raises(ValueError):
        V2Associator().associate([], None)


# --- visibility
def test_visibility_inside_image_and_depth_test():
    depth = np.full((2, H, W), Z)
    depth[1, 140:160, 190:210] = 3.0  # an occluder in front of view 1's image center
    recon = make_recon(2, depth=depth)
    geo = _Views(recon, None, None)
    xyz = np.array([0.0, 0.0, Z])
    assert geo.project(xyz, SIZE, 0, 1.0)[0]
    assert not geo.project(xyz, SIZE, 1, 1.0)[0]  # occluded in view 1 (cam x = -0.2 -> u~196)
    assert not geo.project(np.array([50.0, 0, Z]), SIZE, 0, 1.0)[0]  # outside the image
    assert not geo.project(np.array([0.0, 0, -1.0]), SIZE, 0, 1.0)[0]  # behind the camera


def test_occluded_track_is_not_matched():
    depth = np.full((2, H, W), Z)
    depth[1, 100:200, 150:250] = 3.0
    recon = make_recon(2, depth=depth)
    a = _lifted("0:0", 0, [0, 0, Z], 0.0)
    b = _lifted("1:0", 1, [0, 0, Z], 0.2)
    res = V2Associator(V2Config(hallucination="off")).associate([a, b], recon)
    assert len(res.instances) == 2  # occluded object cannot be the same instance


# --- hallucination / split / appearance
def _hallu_scene():
    recon = make_recon(4)
    real = [_lifted(f"{v}:0", v, [-3, 0, Z], 0.2 * v) for v in range(4)]
    ghost = _lifted("0:1", 0, [3, 0, Z], 0.0)  # visible in all views, detected only in one
    return recon, real + [ghost]


def test_hallucination_modes():
    recon, lifted = _hallu_scene()
    off = V2Associator(V2Config(hallucination="off")).associate(lifted, recon)
    down = V2Associator(V2Config(hallucination="downweight")).associate(lifted, recon)
    drop = V2Associator(V2Config(hallucination="drop")).associate(lifted, recon)
    conf = lambda r: {i.members[0]: i.conf for i in r.instances}  # noqa: E731
    assert len(off.instances) == len(down.instances) == 2 and len(drop.instances) == 1
    assert conf(down)["0:1"] == pytest.approx(0.5 * conf(off)["0:1"])
    assert drop.detection_status["0:1"] == "rejected"
    assert drop.reject_reasons["0:1"] == "hallucination"


class DictExtractor:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def extract(self, frame, dets):
        self.calls += 1
        return np.stack([self.table[d.id] for d in dets])


def test_appearance_term_and_split_check():
    recon = make_recon(2)
    a = _lifted("0:0", 0, [0, 0, Z], 0.0)
    b = _lifted("1:0", 1, [0, 0, Z], 0.2)
    table = {"0:0": np.array([1.0, 0, 0]), "1:0": np.array([0.0, 1, 0])}  # orthogonal
    ext = DictExtractor(table)
    assert isinstance(ext, FeatureExtractor)
    # Default gate (sim < 0 rejects) keeps them together; split check separates them.
    split = V2Associator(V2Config(hallucination="off"), ext).associate([a, b], recon)
    assert ext.calls == 2 and len(split.instances) == 2
    nosplit = V2Associator(V2Config(hallucination="off", split_check=False), ext).associate(
        [a, b], recon
    )
    assert len(nosplit.instances) == 1
    assert nosplit.evidence[0].appearance_sim == pytest.approx(0.0)
    # Strict appearance gate rejects the pair outright.
    strict = V2Associator(V2Config(hallucination="off", min_appearance_sim=0.5), ext)
    assert len(strict.associate([a, b], recon).instances) == 2
    # Matching appearance merges and reports sim = 1.
    same = DictExtractor({"0:0": np.ones(3), "1:0": 2 * np.ones(3)})
    res = V2Associator(V2Config(hallucination="off"), same).associate([a, b], recon)
    assert len(res.instances) == 1
    assert res.evidence[0].appearance_sim == pytest.approx(1.0)
    assert cosine_sim(np.zeros(3), np.ones(3)) == 0.0


def test_split_on_extent():
    recon = make_recon(2)
    # Two members whose xyz are far apart but project to the same pixels (depth ambiguity).
    a = _lifted("0:0", 0, [0, 0, Z], 0.0)
    b = _lifted("1:0", 1, [0, 0, Z], 0.2)
    b.xyz = np.array([0.0, 0.0, Z + 5.0])
    merged = V2Associator(V2Config(hallucination="off", split_check=False)).associate(
        [a, b], recon
    )
    split = V2Associator(V2Config(hallucination="off")).associate([a, b], recon)
    assert len(merged.instances) == 1 and len(split.instances) == 2


# --- Hypothesis properties
scene_args = st.fixed_dictionaries(
    {
        "seed": st.integers(0, 10_000),
        "n_obj": st.integers(3, 15),
        "n_views": st.integers(2, 5),
        "noise": st.floats(0.01, 0.15),
        "drop": st.floats(0.0, 0.5),
        "dup": st.floats(0.0, 0.5),
    }
)
prop = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@pytest.mark.parametrize("assoc", assocs(), ids=lambda a: a.name)
@prop
@given(args=scene_args, perm_seed=st.integers(0, 10_000))
def test_invariant_to_view_and_detection_order(assoc, args, perm_seed):
    recon, lifted, _ = make_scene(**args)
    base = assoc.associate(lifted, recon)
    rng = np.random.default_rng(perm_seed)
    # Shuffle detection order and, separately, view order (whole-frame blocks reordered).
    shuffled = [lifted[i] for i in rng.permutation(len(lifted))]
    views = list(rng.permutation(args["n_views"]))
    blocks = [[ld for ld in lifted if ld.det.frame_id == v] for v in views]
    by_view = [ld for block in blocks for ld in block]
    for variant in (shuffled, by_view):
        out = assoc.associate(variant, recon)
        assert partition(out) == partition(base)
        assert out.detection_status == base.detection_status


@pytest.mark.parametrize("assoc", assocs(), ids=lambda a: a.name)
@prop
@given(args=scene_args)
def test_never_violates_cannot_link(assoc, args):
    recon, lifted, _ = make_scene(**args)
    frame_of = {ld.det.id: ld.det.frame_id for ld in lifted}
    res = assoc.associate(lifted, recon)
    seen = set()
    for inst in res.instances:
        frames = [frame_of[m] for m in inst.members]
        assert len(frames) == len(set(frames))
        assert inst.n_views == len(frames)
        seen.update(inst.members)
    assert seen == set(frame_of)  # every detection lands in exactly one instance
    assert sum(len(i.members) for i in res.instances) == len(frame_of)
