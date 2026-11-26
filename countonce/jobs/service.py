"""Run the laptop worker as a long-lived service: supervise, restart, one-command start.

``supervise`` restarts the worker process when it exits abnormally (a CUDA crash, an OOM kill)
with exponential backoff that resets after a healthy stretch, and forwards SIGINT/SIGTERM so a
stop is graceful. ``systemd_unit`` renders a user unit for machines with systemd (WSL2 included).
"""

from __future__ import annotations

import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

HEALTHY_AFTER_S = 120.0
MAX_BACKOFF_S = 60.0


def worker_command(argv: Sequence[str]) -> list[str]:
    """The child command: the same interpreter running ``countonce worker`` with ``argv``."""
    return [sys.executable, "-m", "countonce.cli", "worker", *argv]


def supervise(
    cmd: Sequence[str],
    *,
    max_restarts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True),
) -> int:
    """Run ``cmd`` until it exits 0 or a signal asks us to stop. Returns the last exit code."""
    stopping = False
    child: subprocess.Popen[bytes] | None = None

    def _stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)  # the worker treats SIGINT as a graceful stop

    old = {s: signal.signal(s, _stop) for s in (signal.SIGINT, signal.SIGTERM)}
    backoff, restarts, code = 1.0, 0, 0
    try:
        while not stopping:
            started = clock()
            child = subprocess.Popen(list(cmd))
            code = child.wait()
            if stopping or code == 0:
                break
            if clock() - started >= HEALTHY_AFTER_S:
                backoff = 1.0
            restarts += 1
            if max_restarts is not None and restarts > max_restarts:
                log(f"worker service: giving up after {restarts - 1} restart(s), exit {code}")
                break
            log(f"worker service: worker exited {code}; restarting in {backoff:g}s")
            sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    return 0 if stopping else code


def systemd_unit(argv: Sequence[str], workdir: Path) -> str:
    """A systemd user unit that runs the worker and restarts it on failure."""
    exec_line = " ".join(shlex.quote(a) for a in worker_command(argv))
    return f"""[Unit]
Description=CountOnce worker
After=network-online.target

[Service]
WorkingDirectory={workdir}
ExecStart={exec_line}
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=120

[Install]
WantedBy=default.target
"""
