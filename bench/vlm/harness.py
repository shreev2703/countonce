"""Paid-call gate, spend cap, per-call log and the ``kind: vlm`` bench executor.

A VLM run happens only if ``--allow-paid`` is passed, a per-run spending cap is set and the
provider's API key variable exists. Otherwise each job is recorded as
``status=skipped_paid_disabled`` and no client is constructed.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.runner import Job, RunOutput, SkipRun
from bench.vlm.client import PROVIDERS, VLMClient, build_client
from bench.vlm.parse import parse_answer
from bench.vlm.prompts import load_prompt, render

DEFAULT_CALL_LOG = Path("results/vlm_calls.csv")
CALL_COLUMNS = [
    "run_key", "scene_id", "provider", "model", "date", "n_images", "n_images_requested",
    "prompt_version", "input_tokens", "output_tokens", "est_cost_usd", "answer", "refusal",
    "parse_method", "reply_excerpt",
]  # fmt: skip


def paid_gate(
    provider: str,
    *,
    allow_paid: bool,
    spend_cap_usd: float | None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """None when paid calls may proceed, else the reason they are disabled."""
    if provider not in PROVIDERS:
        return f"unknown provider {provider!r}"
    if not allow_paid:
        return "--allow-paid not passed"
    if spend_cap_usd is None or spend_cap_usd <= 0:
        return "no positive per-run spending cap set"
    var = PROVIDERS[provider].env_var
    if var is not None and not (os.environ if env is None else env).get(var):
        return f"API key env var {var} is not set"
    return None


class SpendTracker:
    """Cumulative estimated spend for one bench run; latches once a call would exceed the cap."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.spent = 0.0
        self.stopped = False

    def allows(self, projected_usd: float) -> bool:
        if self.stopped or self.spent + projected_usd > self.cap:
            self.stopped = True
            return False
        return True

    def add(self, usd: float) -> None:
        self.spent += usd


@dataclass(frozen=True)
class Prices:
    usd_in: float
    usd_out: float

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_in + tokens_out * self.usd_out) / 1e6


def _prices(provider: str, opts: Mapping[str, Any]) -> Prices:
    p = PROVIDERS[provider]
    return Prices(
        float(opts.get("price_in", p.usd_per_mtok_in)),
        float(opts.get("price_out", p.usd_per_mtok_out)),
    )


def append_call_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CALL_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in CALL_COLUMNS})


def make_vlm_executor(
    cfg: Any,
    scenes: list[dict[str, Any]],
    *,
    client_factory: Callable[[str, str], VLMClient] = build_client,
    call_log: Path = DEFAULT_CALL_LOG,
    env: Mapping[str, str] | None = None,
    max_output_tokens: int = 1024,
) -> Callable[[Job], RunOutput]:
    """Executor for ``kind: vlm`` jobs. ``cfg`` is a ``BenchConfig`` (needs ``allow_paid``,
    ``spend_cap_usd``, ``prompt_key``, ``prompt``, ``seed``, ``random_samples``/``even_sample``)."""
    from bench.sweep import select_views, view_samples
    from countonce.ingest.frames import list_images

    by_id = {s["id"]: s for s in scenes}
    seeds = dict(view_samples(cfg))
    tracker = SpendTracker(cfg.spend_cap_usd or 0.0)
    clients: dict[tuple[str, str], VLMClient] = {}

    def fn(job: Job) -> RunOutput:
        opts = job.config["options"]
        provider, model = opts["provider"], opts["model"]
        reason = paid_gate(
            provider, allow_paid=cfg.allow_paid, spend_cap_usd=cfg.spend_cap_usd, env=env
        )
        if reason:
            raise SkipRun("skipped_paid_disabled", reason)
        info = PROVIDERS[provider]
        prices = _prices(provider, opts)
        scene = by_id[job.scene_id]
        gt = scene["gt"]
        n_req = job.n_views or 0
        paths = select_views(
            list_images(scene["dir"]), n_req, seeds[job.sample], cfg.seed, job.ordered
        )[: info.max_images]
        template, version = load_prompt(opts.get("prompt", "count"), int(opts.get("version", 1)))
        prompt = render(template, gt.get(cfg.prompt_key, cfg.prompt))
        projected = prices.cost(len(paths) * info.tokens_per_image + len(prompt) // 3,
                                max_output_tokens)  # fmt: skip
        if not tracker.allows(projected):
            raise SkipRun(
                "skipped_cost_cap",
                f"spent ${tracker.spent:.4f} of ${tracker.cap:.2f}; "
                f"next call projects ${projected:.4f}",
            )
        key = (provider, model)
        if key not in clients:  # the only place a client is built, after the gate
            clients[key] = client_factory(provider, model)
        t0 = time.perf_counter()
        resp = clients[key].complete(prompt, paths, max_output_tokens)
        dt = time.perf_counter() - t0
        cost = prices.cost(resp.input_tokens, resp.output_tokens)
        tracker.add(cost)
        parsed = parse_answer(resp.text)
        append_call_row(
            call_log,
            {
                "run_key": job.key(), "scene_id": job.scene_id, "provider": provider,
                "model": model, "date": datetime.now(UTC).date().isoformat(),
                "n_images": len(paths), "n_images_requested": n_req, "prompt_version": version,
                "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "est_cost_usd": f"{cost:.6f}",
                "answer": "" if parsed.value is None else parsed.value,
                "refusal": int(parsed.refusal), "parse_method": parsed.method,
                "reply_excerpt": json.dumps(resp.text[:200])[1:-1],
            },
        )  # fmt: skip
        if parsed.refusal:
            raise SkipRun("refused", f"{provider}/{model} refused (logged in {call_log})")
        if parsed.value is None:
            raise ValueError(f"no integer answer in reply: {resp.text[:80]!r}")
        return RunOutput(pred=float(parsed.value), gt=float(gt["count"]), runtime_s=dt)

    return fn
