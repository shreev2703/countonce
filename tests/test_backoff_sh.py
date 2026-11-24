"""scripts/lib/backoff.sh: limit classification, retry hints and exponential backoff."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "scripts" / "lib" / "backoff.sh"


def sh(call: str, **env: str) -> str:
    r = subprocess.run(
        ["bash", "-c", f'source "{LIB}"; {call}'],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "BACKOFF_JITTER": "0", **env},
        timeout=20,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def log(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "run.log"
    p.write_text(text)
    return p


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('{"type":"result","is_error":true,"result":"API Error: 429 rate_limit_error"}', "rate"),
        ("HTTP 429 Too Many Requests", "rate"),
        ('{"error":{"status": 429}}', "rate"),
        ("Claude AI usage limit reached|1999999999", "quota"),
        ("Error: insufficient_quota - You exceeded your current quota", "quota"),
        ("API Error: 529 overloaded_error", "overloaded"),
        ("all tests passed", "none"),
    ],
)
def test_classify_limit(tmp_path: Path, text: str, kind: str) -> None:
    assert sh(f'classify_limit "{log(tmp_path, text)}"') == kind


def test_only_the_tail_is_inspected(tmp_path: Path) -> None:
    # A transcript that merely discusses rate limits early on is not a rate-limited run.
    body = "we handle 429 and rate limit errors in the router\n" + "ok\n" * 50
    assert sh(f'classify_limit "{log(tmp_path, body)}"') == "none"


def test_retry_after_hints(tmp_path: Path) -> None:
    def hint(text: str) -> str:
        return sh(f'retry_after_secs "{log(tmp_path, text)}"')

    assert hint("retry-after: 42") == "42"
    assert hint("Please retry in 30s") == "30"
    assert hint("try again in 5 minutes") == "300"
    assert hint("try again in 20 seconds") == "20"
    assert hint("nothing useful") == ""
    reset = int(time.time()) + 600
    assert 600 <= int(hint(f"usage limit reached|{reset}")) <= 700
    assert hint("usage limit reached|1000000000") == ""  # reset already in the past


def test_backoff_is_exponential_and_capped() -> None:
    rate = [sh(f"backoff_seconds rate {n}") for n in (1, 2, 3, 4)]
    assert rate == ["30", "60", "120", "240"]
    assert sh("backoff_seconds quota 1") == "300"
    assert sh("backoff_seconds overloaded 1") == "60"
    assert sh("backoff_seconds rate 30") == "1800"  # capped, no overflow


def test_backoff_prefers_hint_and_respects_cap() -> None:
    assert sh("backoff_seconds rate 5 45") == "45"
    assert sh("backoff_seconds quota 1 99999") == "1800"


def test_jitter_stays_within_bound() -> None:
    for _ in range(20):
        v = int(sh("backoff_seconds rate 1", BACKOFF_JITTER="10"))
        assert 30 <= v <= 33
