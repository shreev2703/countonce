"""CPU tests of the merging-gate sweep helpers (bench.order_invariance)."""

from bench.order_invariance import gate_sweep, n_merges, smallest_merging_gate
from countonce.merge.match import V2Associator, V2Config
from tests.test_associators import make_scene


def test_n_merges():
    assert n_merges(20, 20) == 0
    assert n_merges(20, 14) == 6
    assert n_merges(5, 9) == 0  # never negative


def test_gate_sweep_is_monotone_on_a_synthetic_scene():
    scenes = [make_scene(s, n_obj=8, n_views=5, noise=0.15)[:2] for s in (1, 2, 3)]
    windows = [(recon, lifted) for recon, lifted in scenes]
    gates = (1e-6, 0.5, 4.0)
    rows = gate_sweep(lambda g: V2Associator(V2Config(max_reproj_norm=g)), windows, gates)
    assert [r["gate"] for r in rows] == list(gates)
    assert all(r["n_windows"] == 3 for r in rows)
    assert rows[0]["n_with_merge"] == 0 and rows[0]["total_merges"] == 0  # nothing passes the gate
    assert rows[-1]["n_with_merge"] == 3  # views of the same object merge at a loose gate
    merges = [r["total_merges"] for r in rows]
    assert merges == sorted(merges)


def _sw(gate_fracs):
    return [{"gate": g, "frac_with_merge": f} for g, f in gate_fracs]


def test_smallest_merging_gate_needs_every_sweep():
    a = _sw([(0.5, 0.0), (1.0, 0.5), (2.0, 1.0)])
    b = _sw([(0.5, 0.0), (1.0, 0.33), (2.0, 0.5)])
    assert smallest_merging_gate([a]) == 1.0
    assert smallest_merging_gate([a, b]) == 2.0
    assert smallest_merging_gate([a, _sw([(0.5, 0.0), (1.0, 0.1), (2.0, 0.2)])]) is None
    assert smallest_merging_gate([]) is None
