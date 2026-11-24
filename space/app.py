"""CountOnce Hugging Face Space: photos + prompt -> count + overlays (PLAN §16).

Gradio is imported lazily inside ``build_demo`` so ``count_handler`` is importable and testable
without it. Nothing here touches the network except what the pipeline's own backends do.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def _paths(images: Any) -> list[Path]:
    """Normalise Gradio's gallery/file values (paths, objects with .name, (path, caption))."""
    out: list[Path] = []
    for item in images or []:
        if isinstance(item, (tuple, list)):
            item = item[0]
        out.append(Path(getattr(item, "name", item)))
    return out


def count_handler(
    images: Any,
    prompt: str,
    *,
    config: Any = None,
    detector: Any = None,
    reconstructor: Any = None,
    workdir: Path | str | None = None,
) -> tuple[str, list[str], str]:
    """Return (count text, overlay paths, result json); errors become text, not crashes."""
    from countonce import run
    from countonce.config import Config

    paths = _paths(images)
    if not paths:
        return "Upload at least one image.", [], "{}"
    if not prompt or not prompt.strip():
        return "Enter what to count, e.g. 'red apples'.", [], "{}"
    out_dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="countonce_space_"))
    cfg = config if config is not None else Config(out_dir=out_dir)
    if config is not None:
        cfg = cfg.model_copy(update={"out_dir": out_dir})
    try:
        result = run(
            paths, prompt.strip(), cfg, detector=detector, reconstructor=reconstructor, export=True
        )
    except Exception as exc:  # shown in the UI; the Space must not crash on bad input
        return f"Error: {exc}", [], json.dumps({"error": str(exc)})
    lo, hi = result.interval.low, result.interval.high
    text = f"{result.count} {prompt.strip()} (interval {lo}-{hi})"
    if result.zero_count:
        text = f"0 {prompt.strip()} found"
    overlays = sorted(str(p) for p in (out_dir / "overlays").glob("*") if p.is_file())
    return text, overlays, json.dumps(result.to_dict(), indent=2, default=str)


def build_demo() -> Any:
    import gradio as gr

    with gr.Blocks(title="CountOnce") as demo:
        gr.Markdown("# CountOnce\nCount distinct objects across overlapping photos.")
        imgs = gr.File(file_count="multiple", file_types=["image"], label="Photos")
        prompt = gr.Textbox(label="What to count", placeholder="red apples")
        btn = gr.Button("Count")
        count = gr.Textbox(label="Count")
        gallery = gr.Gallery(label="Overlays")
        raw = gr.Code(language="json", label="result.json")
        btn.click(count_handler, [imgs, prompt], [count, gallery, raw])
    return demo


if __name__ == "__main__":
    build_demo().launch()
