"""Correction tools on the v1.1 fixture: actions, cannot-link, idempotence, consent, CLI."""

import json
import shutil
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.corrections import (
    Correction,
    apply_corrections,
    export_training_pairs,
    write_training_pairs_jsonl,
)
from countonce.errors import CorrectionError, ErrorCode
from countonce.schemas import load_result

FIX = Path(__file__).parent / "fixtures" / "schemas" / "v1.1"


@pytest.fixture
def res():
    return load_result(json.loads((FIX / "result.json").read_text()))


def C(action, *ids, consent=False):
    return Correction("job1", action, ids, user="u", consent=consent)


def _inst_of(r, d):
    return next(i for i in r.instances if d in i.members)


def test_split(res):
    out = apply_corrections(res, [C("split", "2:0", "3:0")])
    assert out.count == 6 and res.count == 5
    new = _inst_of(out, "2:0")
    assert set(new.members) == {"2:0", "3:0"} and _inst_of(out, "0:0").id != new.id
    assert set(_inst_of(out, "0:0").members) == {"0:0", "1:0"}
    assert out.interval.low == 6 and out.config["corrections_applied"][0]["action"] == "split"
    assert out.schema_version == "1.1"


def test_merge(res):
    s2 = apply_corrections(res, [C("split", "1:1", "2:1", "3:1")])  # leaves 0:1 alone
    a, b = _inst_of(s2, "0:1"), _inst_of(s2, "1:1")
    assert a.id != b.id
    out = apply_corrections(s2, [C("merge", "0:1", "1:1")])
    assert out.count == res.count and len(_inst_of(out, "0:1").members) == 4
    assert _inst_of(out, "0:1").id == min(a.id, b.id)


def test_reject(res):
    out = apply_corrections(res, [C("reject", "3:4")])
    assert out.detection_status["3:4"] == "rejected" and out.reject_reasons["3:4"] == "user_reject"
    assert "3:4" not in _inst_of(out, "0:4").members and out.count == 5
    full = apply_corrections(res, [C("reject", "0:2", "1:2", "2:2", "3:2")])
    assert full.count == 4 and full.interval.high == 4
    assert full.per_frame_counts["0"] == 4


def test_cannot_link_refusal(res):
    with pytest.raises(CorrectionError) as e:
        apply_corrections(res, [C("merge", "0:0", "0:1")])
    assert e.value.code == ErrorCode.ERR_CANNOT_LINK and not e.value.retryable
    with pytest.raises(CorrectionError):
        apply_corrections(res, [C("split", "0:0", "0:1")])


def test_invalid_refs(res):
    with pytest.raises(CorrectionError):
        apply_corrections(res, [C("reject", "9:9")])
    with pytest.raises(CorrectionError):
        apply_corrections(res, [C("split", "0:0", "1:0", "2:0", "3:0")])  # empties instance


def test_idempotent(res):
    cs = [C("split", "2:0", "3:0"), C("reject", "3:4")]
    once = apply_corrections(res, cs)
    twice = apply_corrections(once, cs)
    assert twice.model_dump() == once.model_dump()
    assert len(once.config["corrections_applied"]) == 2


def test_consent_filtering(res):
    cs = [
        C("split", "2:0", "3:0", consent=True),
        C("merge", "0:1", "1:1", consent=True),  # already one instance: no pairs
        C("reject", "3:4", consent=False),
        C("reject", "3:3", consent=True),
    ]
    pairs = export_training_pairs(cs, {"job1": res})
    assert {p.kind for p in pairs} == {"hard_negative", "false_positive"}
    split = apply_corrections(res, [C("split", "1:1", "2:1", "3:1")])
    pos = export_training_pairs([C("merge", "0:1", "1:1", consent=True)], {"job1": split})
    assert {p.kind for p in pos} == {"hard_positive"} and len(pos) == 3
    fp = [p for p in pairs if p.kind == "false_positive"]
    assert [p.det_a for p in fp] == ["3:3"]  # unconsented 3:4 excluded
    neg = [p for p in pairs if p.kind == "hard_negative"]
    assert {(p.det_a, p.det_b) for p in neg} == {
        ("2:0", "0:0"),
        ("2:0", "1:0"),
        ("3:0", "0:0"),
        ("3:0", "1:0"),
    }
    assert export_training_pairs([C("reject", "3:4")], {"job1": res}) == []


def test_cli_leaves_original_untouched(tmp_path, res):
    out = tmp_path / "run"
    shutil.copytree(FIX, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps([C("split", "2:0", "3:0", consent=True).to_dict()]))
    pairs = tmp_path / "pairs.jsonl"
    assert main(["correct", str(out), "--file", str(cf), "--pairs", str(pairs)]) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before
    cdir = out / "corrected"
    assert {p.name for p in cdir.iterdir()} == {"result.json", "markers.json", "counts.csv"}
    assert json.loads((cdir / "result.json").read_text())["count"] == 6
    assert json.loads((cdir / "markers.json").read_text())["count"] == 6
    assert (cdir / "counts.csv").read_text().strip().splitlines()[-1].startswith("total")
    rows = [json.loads(line) for line in pairs.read_text().splitlines()]
    assert len(rows) == 4 and all(r["kind"] == "hard_negative" for r in rows)


def test_cli_error(tmp_path):
    out = tmp_path / "run"
    shutil.copytree(FIX, out)
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps([C("merge", "0:0", "0:1").to_dict()]))
    assert main(["correct", str(out), "--file", str(cf)]) == 2
    assert not (out / "corrected").exists()


def test_jsonl_writer(tmp_path, res):
    p = write_training_pairs_jsonl(
        export_training_pairs([C("reject", "3:3", consent=True)], {"job1": res}),
        tmp_path / "a" / "p.jsonl",
    )
    assert json.loads(p.read_text())["geometry"]["3:3"]["frame_id"] == 3
