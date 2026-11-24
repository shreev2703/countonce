"""Retention, deletion, share tokens, API keys, rate limits and training opt-in."""

import json
import sqlite3
from pathlib import Path

import pytest

from countonce.cli import main
from countonce.corrections import Correction, export_training_pairs
from countonce.jobs.store import DAY_S, RateLimitError, SqliteJobStore
from countonce.schemas import load_result

FIX = Path(__file__).parent / "fixtures" / "schemas" / "v1.1"
RESULT = {"count": 3}


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def _finished(store, tmp_path, name, clock):
    """Submit a job with a real input file and an output dir, then complete it."""
    inp = tmp_path / f"{name}.jpg"
    inp.write_bytes(b"x")
    jid, _ = store.submit("u", "apples", [inp])
    out = tmp_path / "outputs" / jid
    out.mkdir(parents=True)
    (out / "result.json").write_text("{}")
    store.register_worker("w")
    job = store.claim("w")
    assert job["id"] == jid
    assert store.complete(jid, "w", RESULT)
    return jid, inp, out


def test_purge_expired_keeps_live(tmp_path):
    clock = Clock()
    store = SqliteJobStore(tmp_path / "j.db", clock=clock, output_root=tmp_path / "outputs")
    old, old_in, old_out = _finished(store, tmp_path, "old", clock)
    clock.t += 40 * DAY_S
    new, new_in, new_out = _finished(store, tmp_path, "new", clock)
    queued_in = tmp_path / "q.jpg"
    queued_in.write_bytes(b"x")
    qid, _ = store.submit("u", "queued", [queued_in])

    assert store.purge_expired() == [old]
    assert not old_in.exists() and not old_out.exists()
    assert store.get(old)["status"] == "deleted" and store.get_result(old) is None
    assert new_in.exists() and new_out.exists() and store.get(new)["status"] == "done"
    assert queued_in.exists() and store.get(qid)["status"] == "queued"
    # configurable window
    assert store.purge_expired(retention_days=0.0) == [new]
    assert store.purge_expired() == []


def test_delete_job(tmp_path):
    store = SqliteJobStore(tmp_path / "j.db", output_root=tmp_path / "outputs")
    jid, inp, out = _finished(store, tmp_path, "a", Clock())
    tok = store.create_share_token(jid)
    assert store.delete_job(jid) and not store.delete_job(jid)
    assert not inp.exists() and not out.exists()
    assert store.resolve_share_token(tok) is None
    assert store.get(jid)["status"] == "deleted"
    assert not store.delete_job("nope")


def test_cannot_delete_running(tmp_path):
    store = SqliteJobStore(tmp_path / "j.db")
    jid, _ = store.submit("u", "p", [])
    store.register_worker("w")
    store.claim("w")
    with pytest.raises(ValueError):
        store.delete_job(jid)


def test_share_token_revoke(tmp_path):
    store = SqliteJobStore(tmp_path / "j.db", output_root=tmp_path / "outputs")
    jid, *_ = _finished(store, tmp_path, "a", Clock())
    t1, t2 = store.create_share_token(jid), store.create_share_token(jid)
    assert t1 != t2 and len(t1) >= 32
    view = store.resolve_share_token(t1)
    assert view["job"]["id"] == jid and view["result"]["count"] == 3
    assert "worker_id" not in view["job"] and "user_id" not in view["job"]
    assert store.revoke_share_token(t1) and not store.revoke_share_token(t1)
    assert store.resolve_share_token(t1) is None
    assert store.resolve_share_token(t2) is not None
    assert store.resolve_share_token("garbage") is None
    with pytest.raises(KeyError):
        store.create_share_token("nope")


def test_api_key_plaintext_never_stored(tmp_path):
    clock = Clock()
    db = tmp_path / "j.db"
    store = SqliteJobStore(db, clock=clock)
    key_id, plain = store.create_api_key("alice")
    assert store.verify_api_key(plain) == "alice"
    assert store.api_keys("alice")[0]["last_used_at"] == clock.t
    assert store.verify_api_key(plain + "x") is None
    assert store.verify_api_key("co_unknown_zzz") is None
    secret = plain.split("_", 2)[2]
    conn = sqlite3.connect(db)
    dump = "\n".join(conn.iterdump())
    conn.close()
    assert plain not in dump and secret not in dump
    assert "hashed_key" in dump
    assert "hashed_key" not in store.api_keys()[0]
    assert store.revoke_api_key(key_id) and not store.revoke_api_key(key_id)
    assert store.verify_api_key(plain) is None
    # salted: two keys never share a salt/hash
    _, p2 = store.create_api_key("alice")
    assert p2 != plain


def test_rate_limit_rejects_n_plus_1(tmp_path):
    clock = Clock()
    store = SqliteJobStore(tmp_path / "j.db", clock=clock, rate_limit=3, rate_window_s=60)
    for _ in range(3):
        store.submit("u", "p", [])
    with pytest.raises(RateLimitError) as ei:
        store.submit("u", "p", [])
    assert 0 < ei.value.retry_after_s <= 20
    assert len(store.list_jobs()) == 3  # rejected submit left no row
    store.submit("other", "p", [])  # per-user
    clock.t += 20  # one token refilled
    store.submit("u", "p", [])
    with pytest.raises(RateLimitError):
        store.submit("u", "p", [])


def test_idempotent_duplicate_costs_no_token(tmp_path):
    store = SqliteJobStore(tmp_path / "j.db", clock=Clock(), rate_limit=1)
    a, created = store.submit("u", "p", [], idempotency_key="k")
    b, dup = store.submit("u", "p", [], idempotency_key="k")
    assert created and not dup and a == b


def test_opted_out_corrections_excluded(tmp_path):
    store = SqliteJobStore(tmp_path / "j.db")
    store.set_training_opt_in("in", True)
    store.set_training_opt_in("out", False)
    res = load_result(json.loads((FIX / "result.json").read_text()))
    cs = [
        Correction("job1", "reject", ("3:4",), user=u, consent=True)
        for u in ("in", "out", "unknown", None)
    ]
    pairs = export_training_pairs(cs, {"job1": res}, store.opted_in_users())
    assert [p.user for p in pairs] == ["in"]
    assert len(export_training_pairs(cs, {"job1": res})) == 4  # no gate: consent only
    store.set_training_opt_in("in", False)
    assert export_training_pairs(cs, {"job1": res}, store.opted_in_users()) == []


def test_cli(tmp_path, capsys):
    db = tmp_path / "j.db"
    store = SqliteJobStore(db, output_root=tmp_path / "outputs")
    jid, inp, _ = _finished(store, tmp_path, "a", Clock())
    base = ["jobs", "--db", str(db)]
    assert main([*base, "share", jid]) == 0
    token = capsys.readouterr().out.strip()
    assert store.resolve_share_token(token)
    assert main([*base, "revoke-share", token]) == 0
    assert store.resolve_share_token(token) is None
    assert main([*base, "apikey", "create", "bob"]) == 0
    out = capsys.readouterr().out
    key_id = out.split("id: ")[1].split()[0]
    plain = out.split("key: ")[1].split()[0]
    assert store.verify_api_key(plain) == "bob"
    assert main([*base, "apikey", "revoke", key_id]) == 0
    assert store.verify_api_key(plain) is None
    assert main([*base, "purge", "--retention-days", "0"]) == 0
    assert not inp.exists()
    assert main([*base, "delete", jid]) == 1  # already purged


def test_cli_dash_leading_token(tmp_path, capsys, monkeypatch):
    """A token_urlsafe value starting with '-' must not break the CLI round trip."""
    import secrets

    monkeypatch.setattr(secrets, "token_urlsafe", lambda n=32: "-" + "a" * 42)
    db = tmp_path / "j.db"
    store = SqliteJobStore(db, output_root=tmp_path / "outputs")
    jid, _, _ = _finished(store, tmp_path, "a", Clock())
    base = ["jobs", "--db", str(db)]
    assert main([*base, "share", jid]) == 0
    token = capsys.readouterr().out.strip()
    assert not token.startswith("-")
    assert store.resolve_share_token(token)
    assert main([*base, "revoke-share", token]) == 0
    assert store.resolve_share_token(token) is None


def test_cli_positionals_after_double_dash(tmp_path, capsys):
    db = tmp_path / "j.db"
    store = SqliteJobStore(db, output_root=tmp_path / "outputs")
    jid, _, _ = _finished(store, tmp_path, "a", Clock())
    legacy = "-legacytoken"
    with store._tx() as conn:
        conn.execute(
            "INSERT INTO share_tokens (token, job_id, created_at) VALUES (?, ?, ?)",
            (legacy, jid, 0.0),
        )
    assert store.resolve_share_token(legacy)  # old unprefixed tokens still resolve
    base = ["jobs", "--db", str(db)]
    assert main([*base, "revoke-share", "--", legacy]) == 0
    assert store.resolve_share_token(legacy) is None
    assert main([*base, "apikey", "revoke", "--", "-nokey"]) == 1  # parsed, just unknown
