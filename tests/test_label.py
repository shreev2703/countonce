"""Identity labeling tool: HTML generation, label I/O validation, association F1 vs labels."""

import json
import shutil
from pathlib import Path

import pytest

from bench.label import io as lio
from bench.label.html import write_label_html
from countonce.cli import main
from countonce.explain import load_result_dir

FIX = Path(__file__).parent / "fixtures" / "schemas" / "v1.1"


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    from PIL import Image

    out = tmp_path / "scene_a"
    (out / "sanitized" / "0").mkdir(parents=True)
    shutil.copy(FIX / "result.json", out / "result.json")
    for i in range(4):
        Image.new("RGB", (64, 48), (10 * i, 80, 120)).save(out / "sanitized" / "0" / f"v{i}.png")
    return out


def _ids(run: Path) -> set[str]:
    return {d.id for d in load_result_dir(run).detections}


def _truth(run: Path) -> lio.IdentityLabels:
    """Labels equal to the run's own instances."""
    return lio.seed_from_result(load_result_dir(run), "scene_a", "t")


def test_html_generation(run_dir: Path) -> None:
    p = write_label_html(run_dir)
    text = p.read_text()
    assert p.name == "label.html" and text.startswith("<!doctype html>")
    assert text.count("data:image/jpeg;base64,") == 4  # inline thumbnails, no external refs
    assert (
        "http://" not in text and "https://" not in text and 'src="' not in text.split("<script")[0]
    )
    blob = json.loads(
        text.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    )
    assert len(blob["views"]) == 4 and sum(len(v["dets"]) for v in blob["views"]) == 20
    assert blob["seed"]["labels"] == {}  # plain mode starts empty
    d0 = blob["views"][0]["dets"][0]
    assert (d0["x"], d0["y"]) == (22.0, 19.0)  # grid 64x48 == image 64x48 -> identity scale


def test_prelabel_and_resume(run_dir: Path, tmp_path: Path) -> None:
    p = write_label_html(run_dir, prelabel=True)
    blob = json.loads(p.read_text().split('type="application/json">')[1].split("</script>")[0])
    assert len(blob["seed"]["labels"]) == 20 and blob["seed"]["labels"]["0:0"] == "i0"
    lab = _truth(run_dir)
    lab.labels["0:0"] = "other"
    f = lio.save_labels(lab, tmp_path / "l.json")
    p2 = write_label_html(run_dir, tmp_path / "x.html", labels_path=f)
    assert '"other"' in p2.read_text()


def test_round_trip(run_dir: Path, tmp_path: Path) -> None:
    lab = lio.IdentityLabels(
        "s", {"0:0": "a", "1:0": "a", "0:1": "b"}, ["2:0"], "me", lio.now_iso()
    )
    f = lio.save_labels(lab, tmp_path / "l.json", _ids(run_dir))
    back = lio.load_labels(f, _ids(run_dir))
    assert back == lab and back.schema_version == "1"
    gt = lio.to_gt(back, _ids(run_dir))
    assert gt["0:0"] == gt["1:0"] == "a" and gt["2:0"] is None and gt["3:4"] is None
    assert len(gt) == 20


def _write(tmp_path: Path, **over: object) -> Path:
    doc: dict[str, object] = {"schema_version": "1", "scene": "s", "labels": {}, "rejected": [],
                              "labeler": "x", "timestamp": "t"}  # fmt: skip
    doc.update(over)
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(doc))
    return f


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"labels": {"0:0": "a", "0:1": "a"}}, "cannot-link"),
        ({"labels": {"9:9": "a"}}, "unknown detection"),
        ({"rejected": ["9:9"]}, "unknown detection"),
        ({"labels": {"0:0": "a"}, "rejected": ["0:0"]}, "both labeled and rejected"),
        ({"rejected": ["0:0", "0:0"]}, "duplicate"),
        ({"schema_version": "2"}, "schema_version"),
        ({"labels": []}, "must be an object"),
        ({"labels": {"bad": "a"}}, "unknown detection"),
    ],
)
def test_validation_errors(run_dir: Path, tmp_path: Path, over: dict, match: str) -> None:
    with pytest.raises(lio.LabelError, match=match):
        lio.load_labels(_write(tmp_path, **over), _ids(run_dir))


def test_missing_key_and_bad_json(tmp_path: Path) -> None:
    f = tmp_path / "m.json"
    f.write_text('{"scene": "s"}')
    with pytest.raises(lio.LabelError, match="missing keys"):
        lio.load_labels(f)
    f.write_text("{nope")
    with pytest.raises(lio.LabelError, match="not valid JSON"):
        lio.load_labels(f)


def test_association_f1_against_labels(run_dir: Path) -> None:
    res = load_result_dir(run_dir)
    clusters = lio.clusters_from_result(res)
    perfect = lio.association_scores(clusters, _truth(run_dir))
    assert perfect["f1"] == 1.0 and perfect["purity"] == 1.0 and perfect["n_scored"] == 20
    # Human says instances 0 and 1 are one object: precision stays 1, recall drops.
    lab = _truth(run_dir)
    lab.labels = {d: ("i0" if v == "i1" else v) for d, v in lab.labels.items()}
    # that merges same-view ids, so validate would refuse it; scoring itself is metric-only
    s = lio.association_scores(clusters, lab)
    assert s["precision"] == 1.0 and s["recall"] < 1.0 and 0 < s["f1"] < 1
    # rejected / unlabelled detections are not scored
    lab2 = _truth(run_dir)
    for d in ("0:0", "1:0"):
        del lab2.labels[d]
    lab2.rejected = ["0:0"]
    assert lio.association_scores(clusters, lab2)["n_scored"] == 18


def test_cli_label(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "out.html"
    assert main(["label", str(run_dir), "--prelabel", "--dest", str(dest)]) == 0
    assert dest.is_file() and "label:" in capsys.readouterr().out
    assert main(["label", str(tmp_path / "nope")]) == 2
