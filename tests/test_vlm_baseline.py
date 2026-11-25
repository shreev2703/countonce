"""VLM baseline harness: only the fake client is used; no network client may be built."""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from bench import runner
from bench.runner import Job, SkipRun, read_rows, run_jobs
from bench.sweep import BenchConfig, MethodSpec, build_jobs, make_executor
from bench.vlm import FakeVLMClient, load_prompt, parse_answer
from bench.vlm.fig10 import fig_vlm_vs_countonce
from bench.vlm.harness import SpendTracker, make_vlm_executor, paid_gate

pytest.importorskip("PIL")


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("ANSWER: 12", 12),
        ("There are about 3 per photo.\nobj1..\nANSWER: 1,204", 1204),
        ("**Answer: 9**", 9),
        ("I count a total of 14 objects.", 14),
        ("Final count: twenty-three", 23),
        ("ten", 10),
        ("Seen 5 in photo 1 and 8 in photo 2, so 11", 11),
    ],
)
def test_parse_integers(text, want):
    a = parse_answer(text)
    assert a.value == want and not a.refusal


def test_parse_refusal_and_empty():
    a = parse_answer("I'm sorry, I can't help with counting people in photos.")
    assert a.value is None and a.refusal
    assert parse_answer("") == parse_answer("hmm")  # no number, no refusal wording
    assert not parse_answer("hmm").refusal


def test_prompts_versioned():
    for name in ("count", "enumerate"):
        text, tag = load_prompt(name)
        assert tag == f"{name}.v1" and "{object}" in text and "ANSWER:" in text
    with pytest.raises(ValueError):
        load_prompt("nope")
    with pytest.raises(FileNotFoundError):
        load_prompt("count", 99)


def _scene(tmp_path, n_img=5, count=7):
    from PIL import Image

    d = tmp_path / "scenes" / "s0"
    (d / "images").mkdir(parents=True)
    for i in range(n_img):
        Image.new("RGB", (16, 16), (i * 20, 0, 0)).save(d / "images" / f"{i:03d}.jpg")
    return [{"id": "s0", "dir": d, "tier": "easy", "gt": {"count": count}, "overlap": "high"}]


def _cfg(tmp_path, **kw):
    return BenchConfig(
        csv=tmp_path / "bench.csv", views=[3], random_samples=1, even_sample=False,
        order=["ordered"], methods=[MethodSpec(name="vlm_fake", kind="vlm",
        config={"provider": "fake", "model": "fake-1", "prompt": "count"})], **kw,
    )  # fmt: skip


def test_gate_requires_all_three(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    kw = {"allow_paid": True, "spend_cap_usd": 1.0, "env": {}}
    assert "allow-paid" in paid_gate("anthropic", **{**kw, "allow_paid": False})
    assert "cap" in paid_gate("anthropic", **{**kw, "spend_cap_usd": None})
    assert "cap" in paid_gate("anthropic", **{**kw, "spend_cap_usd": 0})
    assert "ANTHROPIC_API_KEY" in paid_gate("anthropic", **kw)
    assert paid_gate("anthropic", **{**kw, "env": {"ANTHROPIC_API_KEY": "x"}}) is None


def test_no_client_without_allow_paid(tmp_path):
    scenes = _scene(tmp_path)
    cfg = _cfg(tmp_path, spend_cap_usd=5.0)  # cap set, allow_paid missing
    built = []

    def factory(p, m):
        built.append((p, m))
        return FakeVLMClient()

    fn = make_vlm_executor(cfg, scenes, client_factory=factory, call_log=tmp_path / "calls.csv")
    ran, skipped, errored = run_jobs(build_jobs(cfg, scenes), fn, cfg.csv)
    assert (ran, skipped, errored) == (0, 1, 0) and built == []
    assert [r["status"] for r in read_rows(cfg.csv)] == ["skipped_paid_disabled"]
    assert not (tmp_path / "calls.csv").exists()

    # via the default executor: real build_client would raise for a missing SDK/key, so no call
    # may reach it either; the row is skipped, not errored.
    cfg2 = _cfg(tmp_path, spend_cap_usd=None, allow_paid=True)
    cfg2 = cfg2.model_copy(update={"csv": tmp_path / "b2.csv"})
    run_jobs(build_jobs(cfg2, scenes), make_executor(cfg2, scenes), cfg2.csv)
    assert read_rows(cfg2.csv)[0]["status"] == "skipped_paid_disabled"


def test_paid_run_with_fake_logs_call(tmp_path):
    scenes = _scene(tmp_path, n_img=8)
    cfg = _cfg(tmp_path, allow_paid=True, spend_cap_usd=5.0)
    fake = FakeVLMClient(["ANSWER: 6"])
    log = tmp_path / "calls.csv"
    fn = make_vlm_executor(cfg, scenes, client_factory=lambda p, m: fake, call_log=log)
    assert run_jobs(build_jobs(cfg, scenes), fn, cfg.csv) == (1, 0, 0)
    row = read_rows(cfg.csv)[0]
    assert row["status"] == "ok" and float(row["pred"]) == 6 and float(row["gt"]) == 7
    assert len(fake.calls[0][1]) == 3
    (call,) = list(csv.DictReader(io.StringIO(log.read_text())))
    assert call["provider"] == "fake" and call["model"] == "fake-1" and call["answer"] == "6"
    assert call["prompt_version"] == "count.v1" and call["n_images"] == "3"
    assert call["input_tokens"] == "100" and float(call["est_cost_usd"]) > 0 and call["date"]


def test_image_count_capped_at_provider_limit(tmp_path, monkeypatch):
    from bench.vlm import client

    monkeypatch.setitem(client.PROVIDERS, "fake", client.ProviderInfo(None, 2, 1.0, 2.0, 1000))
    scenes = _scene(tmp_path, n_img=5)
    cfg = _cfg(tmp_path, allow_paid=True, spend_cap_usd=5.0)
    fake = FakeVLMClient()
    log = tmp_path / "calls.csv"
    fn = make_vlm_executor(cfg, scenes, client_factory=lambda p, m: fake, call_log=log)
    run_jobs(build_jobs(cfg, scenes), fn, cfg.csv)
    (call,) = list(csv.DictReader(io.StringIO(log.read_text())))
    assert call["n_images"] == "2" and call["n_images_requested"] == "3"


def test_refusal_is_logged_and_skipped(tmp_path):
    scenes = _scene(tmp_path)
    cfg = _cfg(tmp_path, allow_paid=True, spend_cap_usd=5.0)
    log = tmp_path / "calls.csv"
    fake = FakeVLMClient(["I'm sorry, I can't help with that."])
    fn = make_vlm_executor(cfg, scenes, client_factory=lambda p, m: fake, call_log=log)
    assert run_jobs(build_jobs(cfg, scenes), fn, cfg.csv) == (0, 1, 0)
    assert read_rows(cfg.csv)[0]["status"] == "refused"
    (call,) = list(csv.DictReader(io.StringIO(log.read_text())))
    assert call["refusal"] == "1" and call["answer"] == ""


def test_unparseable_reply_is_an_error_row(tmp_path):
    scenes = _scene(tmp_path)
    cfg = _cfg(tmp_path, allow_paid=True, spend_cap_usd=5.0)
    fn = make_vlm_executor(cfg, scenes, client_factory=lambda p, m: FakeVLMClient(["hmm"]),
                           call_log=tmp_path / "c.csv")  # fmt: skip
    assert run_jobs(build_jobs(cfg, scenes), fn, cfg.csv) == (0, 0, 1)


def test_cost_cap_stops_and_latches(tmp_path):
    t = SpendTracker(1.0)
    assert t.allows(0.4)
    t.add(0.9)
    assert not t.allows(0.2) and t.stopped
    assert not t.allows(0.0001)  # latched

    scenes = _scene(tmp_path)
    cfg = _cfg(tmp_path, allow_paid=True, spend_cap_usd=0.001)  # fake call projects ~0.005
    fake = FakeVLMClient()
    fn = make_vlm_executor(cfg, scenes, client_factory=lambda p, m: fake,
                           call_log=tmp_path / "c.csv")  # fmt: skip
    assert run_jobs(build_jobs(cfg, scenes), fn, cfg.csv) == (0, 1, 0)
    assert read_rows(cfg.csv)[0]["status"] == "skipped_cost_cap" and fake.calls == []


def test_vlm_method_requires_provider_and_model():
    with pytest.raises(ValueError):
        MethodSpec(name="x", kind="vlm", config={"provider": "fake"})


def test_skiprun_status_exported():
    assert issubclass(SkipRun, Exception) and runner.COLUMNS


def test_fig10_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "bench.csv"
    rng = np.random.default_rng(0)
    jobs = []
    for s in range(6):
        for m, kind in (("vlm_fake", "vlm"), ("co_v2", "countonce")):
            for ov in ("low", "high"):
                cfg = {"kind": kind, "overlap": ov}
                jobs.append(Job(f"{s}{ov}", m, 5, "even", True, "test", cfg))

    def fn(job):
        gt = 20.0
        sd = 8.0 if job.config["kind"] == "vlm" else 1.0
        return runner.RunOutput(pred=gt + rng.normal(0, sd), gt=gt)

    run_jobs(jobs, fn, path)
    out = fig_vlm_vs_countonce(path, tmp_path / "fig10.png")
    assert out.stat().st_size > 0
    with pytest.raises(ValueError):  # no VLM rows
        run_jobs([Job("z", "co_v2", 5, "even", True, "test", {"kind": "countonce"})], fn,
                 tmp_path / "b2.csv")  # fmt: skip
        fig_vlm_vs_countonce(tmp_path / "b2.csv", tmp_path / "x.png")
