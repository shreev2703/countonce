"""Cold-start rows from fake loaders, safetensors mmap path, and serving report rendering."""

import csv
import importlib.util
import io
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.serving.coldstart import SCHEMA, append_rows, load_weights, run_coldstart, touch_warm

_spec = importlib.util.spec_from_file_location(
    "make_serving_report", Path(__file__).parent.parent / "scripts" / "make_serving_report.py"
)
assert _spec and _spec.loader
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _rows(clock: Clock) -> list:
    return run_coldstart(
        {"a": lambda: clock.advance(2.0), "b": lambda: clock.advance(3.0)},
        lambda m: clock.advance(0.1),
        importer=lambda: clock.advance(1.5),
        process_start=90.0,
        warm_iters=5,
        run_id="r1",
        clock=clock,
        wall=lambda: 100.0,
    )


def test_timing_rows_from_fakes() -> None:
    rows = _rows(Clock())
    by = {(r.phase, r.name): r for r in rows}
    assert by[("process_start", "interpreter")].seconds == pytest.approx(10.0)
    assert by[("import", "pipeline")].seconds == pytest.approx(1.5)
    assert by[("model_load", "a")].seconds == pytest.approx(2.0)
    assert by[("model_load", "b")].seconds == pytest.approx(3.0)
    assert by[("first_inference", "warm")].seconds == pytest.approx(0.1)
    assert by[("warm_p50", "warm")].seconds == pytest.approx(0.1)
    assert by[("warm_p95", "warm")].n == 5
    assert {r.run_id for r in rows} == {"r1"}


def test_append_only_schema(tmp_path: Path) -> None:
    p = tmp_path / "r" / "coldstart.csv"
    append_rows(_rows(Clock()), p)
    append_rows(_rows(Clock()), p)
    lines = list(csv.reader(io.StringIO(p.read_text())))
    assert tuple(lines[0]) == SCHEMA
    assert sum(1 for ln in lines if ln[0] == "run_id") == 1
    assert len(lines) == 1 + 2 * 7


def test_mmap_safetensors(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    st = pytest.importorskip("safetensors.numpy")
    f = tmp_path / "tiny.safetensors"
    st.save_file({"w": np.arange(6, dtype=np.float32).reshape(2, 3)}, str(f))
    handle = load_weights(f)
    assert list(handle.keys()) == ["w"]
    assert touch_warm({"tiny": handle}) == 6


def test_plain_file_read(tmp_path: Path) -> None:
    f = tmp_path / "w.bin"
    f.write_bytes(b"abcd")
    assert load_weights(f) == b"abcd"


def test_cli_coldstart(tmp_path: Path) -> None:
    w = tmp_path / "w.bin"
    w.write_bytes(b"xyz")
    out = tmp_path / "cs.csv"
    assert main(["coldstart", "--model", f"m={w}", "--warm-iters", "3", "--out", str(out)]) == 0
    assert "model_load" in out.read_text()
    assert main(["coldstart", "--model", "bad", "--out", str(out)]) == 2


def test_report_with_and_without_rows(tmp_path: Path) -> None:
    empty = report.render([], None)
    assert "TODO: fill from results" in empty
    assert "TODO: set rate" in empty
    csv_path = tmp_path / "cs.csv"
    append_rows(_rows(Clock()), csv_path)
    full = report.render(report.latest_run(csv_path), 3.6)
    assert "| Model load | a | 2.000 |" in full
    assert "| 0.1000 | 0.1000 |" not in full  # latency table uses 3 decimals
    assert "| 3.6000 | 0.1000 |" in full  # 0.1 s * 1000 / 3600 * 3.6
    cost = full.split("## Cost")[1].split("\n## ")[0]  # later sections may keep their own TODOs
    assert "TODO" not in cost.split("|---|---|")[1]


def test_report_cli(tmp_path: Path) -> None:
    out = tmp_path / "SERVING.md"
    assert report.main(["--csv", str(tmp_path / "none.csv"), "--out", str(out)]) == 0
    assert "TODO: fill from results" in out.read_text()
