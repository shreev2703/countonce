"""Detector adaptation harness (PLAN §11 Track D): pure-torch LoRA plus a training loop.

Works on any torch module with ``forward(images (B,3,H,W)) -> density (B,1,H,W)``; the count of
an image is the sum of its density map. ``inject_lora`` swaps named ``nn.Linear`` layers for
``LoRALinear`` (frozen base weight, trainable low-rank ``B @ A``), ``merge_lora`` folds the update
into the base weight and ``unmerge_lora`` takes it back out. ``head_only`` trains the named head
layers instead. ``ModuleDetector`` exposes such a module as a ``countonce.detect.base.Detector``
so the fine-tuned model runs through the normal pipeline.

Nothing here downloads weights. The real CountGD fine-tune is gated on its weights (docs/GATES.md).
Torch is imported at module load: import this module lazily.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from countonce.detect.base import density_to_points, points_to_detections
from countonce.types import Detection, Frame


class LoRALinear(nn.Module):
    """``y = x W^T + b + (alpha / r) * x A^T B^T`` with ``W``, ``b`` frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        self.rank = rank
        self.scale = (alpha if alpha is not None else float(rank)) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False
        for p in self.base.parameters():
            p.requires_grad_(False)

    def delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scale

    def merge(self) -> None:
        if not self.merged:
            with torch.no_grad():
                self.base.weight += self.delta()
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            with torch.no_grad():
                self.base.weight -= self.delta()
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.base(x)
        if self.merged:
            return out
        low: torch.Tensor = x @ self.lora_A.t() @ self.lora_B.t()
        return out + low * self.scale


def _match(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) or name == p for p in patterns)


def inject_lora(
    model: nn.Module, targets: Sequence[str], rank: int = 4, alpha: float | None = None
) -> list[str]:
    """Replace every ``nn.Linear`` whose dotted name matches a glob in ``targets``.

    Freezes all non-adapter parameters. Returns the replaced names; raises if none matched.
    """
    replaced: list[str] = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and _match(name, targets):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(mod, rank, alpha))
            replaced.append(name)
    if not replaced:
        raise ValueError(f"no nn.Linear layer matched target_layers={list(targets)}")
    for n, p in model.named_parameters():
        p.requires_grad_(".lora_" in n)
    return replaced


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def merge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.unmerge()


def head_only(model: nn.Module, head_layers: Sequence[str]) -> list[str]:
    """Freeze everything except parameters of modules matching ``head_layers`` (globs)."""
    trainable: list[str] = []
    for n, p in model.named_parameters():
        mod_name = n.rpartition(".")[0]
        keep = _match(mod_name, head_layers)
        p.requires_grad_(keep)
        if keep:
            trainable.append(n)
    if not trainable:
        raise ValueError(f"no parameter matched head_layers={list(head_layers)}")
    return trainable


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Trainable parameters only (LoRA A/B, or head weights in head-only mode)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise KeyError(f"adapter keys not in model: {missing}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v)


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash of a tensor dict (names + values), independent of the container format."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def points_to_density(
    points_xy: np.ndarray, height: int, width: int, sigma: float = 2.0
) -> np.ndarray:
    """(H,W) density with one unit-mass Gaussian per point (mass drops only at the border)."""
    d = np.zeros((height, width), dtype=np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    for x, y in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2):
        g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
        s = g.sum()
        if s > 0:
            d += (g / s).astype(np.float32)
    return d


@dataclass
class FinetuneConfig:
    mode: str = "lora"  # lora | head
    target_layers: list[str] = field(default_factory=lambda: ["*"])
    head_layers: list[str] = field(default_factory=lambda: ["head"])
    rank: int = 4
    alpha: float | None = None
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 4
    weight_decay: float = 0.0
    sigma: float = 2.0
    seed: int = 0
    image_size: int = 256
    prompt: str = "apple"
    splits: dict[str, Any] = field(default_factory=dict)
    registry_root: str = "models/detector_adapters"
    out_dir: str = "runs/finetune"
    report_csv: str = "results/detector_adaptation.csv"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FinetuneConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown finetune config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.mode not in ("lora", "head"):
            raise ValueError(f"mode must be lora or head, got {cfg.mode!r}")
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuneConfig:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})


@dataclass
class TrainResult:
    losses: list[float]
    trainable: list[str]
    n_trainable_params: int
    n_total_params: int


def prepare(model: nn.Module, cfg: FinetuneConfig) -> list[str]:
    """Apply the configured adaptation mode; returns replaced layers / trainable params."""
    if cfg.mode == "lora":
        return inject_lora(model, cfg.target_layers, cfg.rank, cfg.alpha)
    return head_only(model, cfg.head_layers)


def train(
    model: nn.Module,
    images: torch.Tensor,
    densities: torch.Tensor,
    cfg: FinetuneConfig,
    on_epoch: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """MSE on density maps, over the trainable parameters only. ``model`` must be ``prepare``d."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters; call prepare() first")
    unmerge_lora(model)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = images.shape[0]
    losses: list[float] = []
    model.train()
    for epoch in range(cfg.epochs):
        order = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(images[idx]), densities[idx])
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / n)
        if on_epoch:
            on_epoch(epoch, losses[-1])
    model.eval()
    return TrainResult(
        losses,
        [n for n, p in model.named_parameters() if p.requires_grad],
        sum(p.numel() for p in params),
        sum(p.numel() for p in model.parameters()),
    )


@torch.no_grad()
def predict_counts(model: nn.Module, images: torch.Tensor) -> list[float]:
    model.eval()
    return [float(v) for v in model(images).sum(dim=(1, 2, 3))]


def base_checkpoint_hash(model: nn.Module) -> str:
    """Hash of the full state dict; call BEFORE injecting adapters."""
    return state_sha256({k: v for k, v in model.state_dict().items()})


def register_adapter(
    state: dict[str, torch.Tensor],
    cfg: FinetuneConfig,
    *,
    base_checkpoint_sha256: str,
    dataset_version: str,
    metrics: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Save the adapter tensors and register them with dataset version and base-checkpoint hash."""
    from dataclasses import asdict

    from countonce.match.registry import Registry

    out = Path(out_dir or cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"adapter_{state_sha256(state)[:12]}.pt"
    torch.save({"state": state, "mode": cfg.mode}, path)
    return Registry(cfg.registry_root).register(
        path,
        dataset_version=dataset_version,
        recipe_version="detector-adapter-v1",
        config={**asdict(cfg), "base_checkpoint_sha256": base_checkpoint_sha256, "kind": "adapter"},
        metrics=metrics or {},
    )


class ModuleDetector:
    """Wrap a density module as a ``Detector``: peaks of the density map become detections."""

    def __init__(
        self,
        model: nn.Module,
        name: str = "torch-density",
        version: str = "0",
        size: int = 256,
        min_peak: float = 0.0,
    ) -> None:
        self.model, self.name, self.version, self.size, self.min_peak = (
            model,
            name,
            version,
            size,
            min_peak,
        )

    def config(self) -> dict[str, object]:
        return {
            "size": self.size,
            "min_peak": self.min_peak,
            "merged": any(m.merged for m in lora_modules(self.model)),
        }

    def detect(
        self, frames: Sequence[Frame], prompt: str, exemplars: Any = None
    ) -> list[Detection]:
        from PIL import Image

        out: list[Detection] = []
        self.model.eval()
        for f in frames:
            with Image.open(f.path) as im:
                arr = np.asarray(im.convert("RGB").resize((self.size, self.size)), dtype=np.float32)
            x = torch.from_numpy(arr / 255.0).permute(2, 0, 1)[None]
            with torch.no_grad():
                d = self.model(x)[0, 0].numpy()
            pts, sc = density_to_points(d, self.min_peak)
            pts = pts * np.array([f.width / self.size, f.height / self.size])
            out += points_to_detections(f, pts, sc)
        return out


def load_image_set(
    images: Iterable[Any], size: int, sigma: float
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Dataset images (``path``, ``points_xy``) -> resized tensors, density maps, GT counts."""
    from PIL import Image

    xs, ds, counts = [], [], []
    for img in images:
        with Image.open(img.path) as im:
            w, h = im.size
            arr = np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        pts = np.asarray(img.points_xy, dtype=np.float64).reshape(-1, 2) * [size / w, size / h]
        xs.append(torch.from_numpy(arr).permute(2, 0, 1))
        ds.append(torch.from_numpy(points_to_density(pts, size, size, sigma))[None])
        counts.append(int(len(pts)))
    if not xs:
        raise ValueError("no images in the set")
    return torch.stack(xs), torch.stack(ds), counts


REPORT_SCHEMA = (
    "run_id", "timestamp", "split", "dataset", "variant", "kind", "n", "metric",
    "estimate", "lo", "hi", "adapter_version",
)  # fmt: skip


def _bootstrap(values: Sequence[float], n_boot: int = 2000) -> tuple[float, float, float]:
    from bench.stats import bootstrap_ci

    ci = bootstrap_ci(np.asarray(values, dtype=float), np.mean, n_boot=n_boot, seed=0)
    return ci.estimate, ci.low, ci.high


def append_report_rows(rows: Sequence[Sequence[object]], path: str | Path) -> Path:
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(REPORT_SCHEMA)
        w.writerows(rows)
    return p


def _rows(
    run_id: str,
    split: str,
    dataset: str,
    variant: str,
    kind: str,
    gt: Sequence[float],
    pred: Sequence[float],
    adapter_version: str,
) -> list[list[object]]:
    import time

    err = [abs(p - g) for g, p in zip(gt, pred, strict=True)]
    est, lo, hi = _bootstrap(err)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    return [
        [run_id, ts, split, dataset, variant, kind, len(err), "mae", est, lo, hi, adapter_version]
    ]


E2E = Callable[[nn.Module], list[tuple[float, float]]]


def run_adaptation(
    model: nn.Module,
    cfg: FinetuneConfig,
    train_set: tuple[torch.Tensor, torch.Tensor],
    eval_sets: Mapping[str, tuple[str, torch.Tensor, Sequence[int]]],
    *,
    dataset_version: str,
    e2e: dict[str, tuple[str, E2E]] | None = None,
    register: bool = True,
) -> dict[str, Any]:
    """Zero-shot eval -> prepare -> train -> fine-tuned eval -> registry + report rows.

    ``eval_sets``: split name ("seen"/"unseen") -> (dataset, images, GT counts).
    ``e2e``: split -> (dataset, fn(model) -> [(gt, pred)] per scene), the end-to-end deduplicated
    count error; supplied by the caller because it needs the full pipeline and real data.
    """
    import uuid

    run_id = uuid.uuid4().hex[:12]
    base_hash = base_checkpoint_hash(model)
    rows: list[list[object]] = []

    def evaluate(variant: str, version: str) -> None:
        for split, (ds, imgs, gts) in eval_sets.items():
            pred = predict_counts(model, imgs)
            rows.extend(_rows(run_id, split, ds, variant, "per_image", gts, pred, version))
        for split, (ds, fn) in (e2e or {}).items():
            pairs = fn(model)
            rows.extend(
                _rows(run_id, split, ds, variant, "e2e", [a for a, _ in pairs],
                      [b for _, b in pairs], version)
            )  # fmt: skip

    evaluate("zero_shot", "")
    prepare(model, cfg)
    res = train(model, *train_set, cfg)
    entry: dict[str, Any] = {}
    state = adapter_state(model)
    if register:
        entry = register_adapter(
            state,
            cfg,
            base_checkpoint_sha256=base_hash,
            dataset_version=dataset_version,
            metrics={"final_loss": res.losses[-1], "first_loss": res.losses[0]},
        )
    evaluate("finetuned", str(entry.get("version", "")))
    append_report_rows(rows, cfg.report_csv)
    return {"run_id": run_id, "train": res, "registry_entry": entry, "base_sha256": base_hash}
