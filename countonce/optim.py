"""Inference-optimization switches (PLAN 13 items 2-5).

Everything here imports torch lazily and is a no-op on CPU: ``maybe_compile`` returns the module
unchanged without CUDA, ``check_sdpa`` reports ``available=False``, the preloader skips pinning.
Nothing here is a result; speedups only exist after a real ``countonce profile`` run.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from countonce.config import Config
    from countonce.detect.base import Exemplar
    from countonce.types import Detection, Frame

log = logging.getLogger("countonce.optim")

CompileMode = Literal["off", "default", "reduce-overhead"]
SDPA_KERNELS = ("flash", "efficient", "cudnn", "math")  # preference order
FAST_KERNELS = ("flash", "efficient", "cudnn")


def _cuda_ok() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def maybe_compile(module: Any, mode: CompileMode) -> Any:
    """``torch.compile(module, mode=...)`` on CUDA; the module itself when off or on CPU.

    A compile failure at wrap time is logged and falls back to eager (compilation itself is
    lazy, so errors at first call surface in the stage, where the OOM ladder / caller sees them).
    """
    if mode == "off" or not _cuda_ok():
        return module
    import torch

    try:
        return torch.compile(module, mode=None if mode == "default" else mode)
    except Exception as exc:  # pragma: no cover - needs a GPU
        log.warning("torch.compile(%s) failed, staying eager: %s", mode, exc)
        return module


@dataclass
class SdpaReport:
    available: bool  # False on CPU / without torch: nothing was probed
    kernel: str | None = None  # best kernel that ran: flash | efficient | cudnn | math
    usable: dict[str, bool] = field(default_factory=dict)
    math_fallback: bool = False
    note: str = ""


def pick_sdpa_kernel(probe: Callable[[str], bool]) -> SdpaReport:
    """Choose the first kernel in preference order for which ``probe(name)`` is True.

    Pure so it can be tested with a fake probe; ``check_sdpa`` supplies the real CUDA probe.
    """
    usable = {k: bool(probe(k)) for k in SDPA_KERNELS}
    kernel = next((k for k in SDPA_KERNELS if usable[k]), None)
    fallback = kernel in (None, "math")
    note = ""
    if fallback:
        note = (
            "attention would use the math SDPA fallback (slow, O(n^2) memory): check dtype "
            "(bf16/fp16), head dim and torch/GPU support"
        )
    return SdpaReport(True, kernel, usable, fallback, note)


def _cuda_probe(kernel: str, dtype: Any = None, head_dim: int = 64) -> bool:  # pragma: no cover
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }[kernel]
    dtype = dtype or torch.bfloat16
    q = torch.randn(1, 2, 16, head_dim, device="cuda", dtype=dtype)
    try:
        with sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return True
    except Exception:
        return False


def check_sdpa(name: str = "model", *, head_dim: int = 64) -> SdpaReport:
    """Log which SDPA kernel VGGT/DINOv2-style attention (bf16, ``head_dim``) can use; warn on math.

    Probes tiny bf16 tensors on the GPU; returns ``available=False`` (no log noise) on CPU.
    """
    if not _cuda_ok():
        return SdpaReport(False, note="no CUDA: SDPA kernel check skipped")
    rep = pick_sdpa_kernel(lambda k: _cuda_probe(k, head_dim=head_dim))
    if rep.math_fallback:
        log.warning("%s: %s", name, rep.note)
    else:
        log.info("%s: SDPA kernel = %s (usable: %s)", name, rep.kernel, rep.usable)
    return rep


class BatchedDetector:
    """Runs a detector over frame batches of ``batch_size`` (<= 0 or >= n: one call).

    Results are concatenated in frame order, so output equals the unbatched call for detectors
    that treat frames independently.
    """

    def __init__(self, inner: Any, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.inner, self.batch_size = inner, batch_size
        self.name = getattr(inner, "name", type(inner).__name__)

    def config(self) -> dict[str, object]:
        return {**self.inner.config(), "batch_size": self.batch_size}

    def detect(
        self,
        frames: Sequence[Frame],
        prompt: str,
        exemplars: Sequence[Exemplar] | None = None,
    ) -> list[Detection]:
        out: list[Detection] = []
        for i in range(0, len(frames), self.batch_size):
            out.extend(self.inner.detect(frames[i : i + self.batch_size], prompt, exemplars))
        return out


def preload(
    paths: Sequence[Path],
    load: Callable[[Path], Any],
    workers: int = 4,
    *,
    pin: bool = False,
) -> list[Any]:
    """Load ``paths`` with ``load`` on a thread pool, in order; pin tensors when asked and on CUDA.

    Overlaps decode/resize (PIL and numpy release the GIL) with the caller's GPU work. Pinning is
    skipped silently on CPU or for non-tensor results. ``workers <= 1`` loads serially.
    """
    if workers <= 1:
        items = [load(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            items = list(pool.map(load, paths))
    if pin and _cuda_ok():
        torch = sys.modules["torch"]
        items = [x.pin_memory() if isinstance(x, torch.Tensor) else x for x in items]
    return items


def configure_inference(obj: Any, config: Config, kind: str) -> Any:
    """Hand the optimization switches to a backend that declares them; returns ``obj``.

    ``kind`` is "recon" or "detector". A backend opts in by having a ``compile_mode`` attribute
    (and, for reconstructors, ``preload_workers`` / ``pin_memory``). Fakes simply lack them.
    """
    mode = config.compile_reconstructor if kind == "recon" else config.compile_detector
    if hasattr(obj, "compile_mode"):
        obj.compile_mode = mode
    if kind == "recon":
        if hasattr(obj, "preload_workers"):
            obj.preload_workers = config.preload_workers
        if hasattr(obj, "pin_memory"):
            obj.pin_memory = config.pin_memory
    return obj


def wrap_detector(det: Any, config: Config) -> Any:
    return BatchedDetector(det, config.detector_batch_size) if config.detector_batch_size else det
