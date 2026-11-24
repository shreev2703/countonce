"""One-click inspection report (PLAN 14.8): a self-contained static ``report.html``.

Reads a stored ``result.json`` (any schema version; older ones are upgraded on load). Everything is
inlined: CSS, base64 thumbnails of annotated overlays and a top-down scatter of instance markers
(drawn with the ported plotting style), so the file can be emailed or archived as is. Styling is
deliberately minimal and neutral; the product UI design is gated on mockup approval (PLAN 17).
PDF output goes through ``weasyprint`` only when it is installed.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

from countonce.explain import load_result_dir

DEFAULT_TOP_N = 6
THUMB_MAX_PX = 480
_IMG_EXT = {".png", ".jpg", ".jpeg"}

_CSS = """
body{font-family:system-ui,sans-serif;color:#222;max-width:900px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.4rem}h2{font-size:1.1rem;border-bottom:1px solid #ccc;padding-bottom:.2rem;
margin-top:1.8rem}
.card{border:1px solid #ccc;border-radius:6px;padding:1rem;display:inline-block;min-width:14rem}
.count{font-size:2.6rem;font-weight:600}.muted{color:#666;font-size:.9rem}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.3rem .5rem;
text-align:left;vertical-align:top;font-size:.9rem}
.warn{border-left:4px solid #b7791f;padding:.3rem .8rem;margin:.5rem 0;background:#faf6ee}
.thumbs img,.plot img{max-width:100%;border:1px solid #ddd}
.thumbs figure{display:inline-block;margin:0 .5rem .5rem 0;width:31%}
figcaption{font-size:.8rem;color:#666}
"""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _thumbnail(path: Path) -> str | None:
    """Downscaled JPEG data URI for an overlay image, or None if it cannot be read."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
    except Exception:  # unreadable/corrupt image or no Pillow: skip it, the report still renders
        return None
    return _data_uri(buf.getvalue(), "image/jpeg")


def _overlay_thumbs(out_dir: Path, top_n: int) -> list[tuple[str, str]]:
    folder = out_dir / "overlays"
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXT)
    # "Top" overlays: the ones with the most annotation ink are hard to know, so rank by
    # file size (busier frames compress larger) and break ties by name for determinism.
    files.sort(key=lambda p: (-p.stat().st_size, p.name))
    out: list[tuple[str, str]] = []
    for p in files:
        if len(out) >= top_n:
            break
        uri = _thumbnail(p)
        if uri is not None:
            out.append((p.name, uri))
    return out


def _scatter_png(instances: list[Any]) -> str | None:
    """Top-down (x vs z) scatter of instance markers, in the shared plotting style."""
    pts = [i for i in instances if i.xyz is not None]
    if not pts:
        return None
    from countonce import plotting

    plotting.apply_style()
    plt = plotting.plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        [i.xyz[0] for i in pts],
        [i.xyz[2] for i in pts],
        s=[max(20.0, 400.0 * float(i.size or 0.05)) for i in pts],
        color=plotting.METHOD_COLORS["vggt"],
        alpha=0.8,
    )
    for i in pts:
        ax.annotate(
            f"#{i.id}", (i.xyz[0], i.xyz[2]), fontsize=7, xytext=(3, 3), textcoords="offset points"
        )
    ax.set_xlabel("x")
    ax.set_ylabel("z (depth)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Top-down instance markers")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return _data_uri(buf.getvalue(), "image/png")


def _kv_rows(d: dict[str, Any]) -> str:
    return "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in sorted(d.items()))


def render_report_html(out_dir: Path | str, top_n: int = DEFAULT_TOP_N) -> str:
    out_dir = Path(out_dir)
    res = load_result_dir(out_dir)
    iv = res.interval
    cfg: dict[str, Any] = res.config or {}
    settings = cfg.get("config", {}) if isinstance(cfg.get("config"), dict) else {}
    prov = {k: v for k, v in cfg.items() if k != "config" and not isinstance(v, dict)}
    n_txt = f", n={iv.n_calibration}" if iv.n_calibration is not None else ""

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>CountOnce report: {_esc(res.prompt)}</title><style>{_CSS}</style></head><body>",
        f"<h1>CountOnce inspection report</h1><p class='muted'>Prompt: {_esc(res.prompt)}</p>",
        "<div class='card'><div class='muted'>Count</div>"
        f"<div class='count'>{res.count}</div>"
        f"<div>{int(iv.level * 100)}% interval: {iv.low} to {iv.high}</div>"
        f"<div class='muted'>method: {_esc(iv.method)}{_esc(n_txt)}</div></div>",
    ]
    if res.zero_count:
        parts.append("<p>Zero-count result.</p>")

    parts.append("<h2>Warnings</h2>")
    if res.warnings:
        for w in res.warnings:
            fix = f"<br><em>Fix:</em> {_esc(w.fix)}" if w.fix else ""
            parts.append(
                f"<div class='warn'><strong>{_esc(w.code)}</strong>: {_esc(w.message)}{fix}</div>"
            )
    else:
        parts.append("<p class='muted'>No warnings.</p>")

    scatter = _scatter_png(list(res.instances))
    if scatter:
        parts.append(
            f"<h2>Instance markers</h2><div class='plot'><img alt='top-down scatter' "
            f"src='{scatter}'></div>"
        )

    thumbs = _overlay_thumbs(out_dir, top_n)
    if thumbs:
        parts.append("<h2>Annotated overlays</h2><div class='thumbs'>")
        parts += [
            f"<figure><img alt='{_esc(n)}' src='{u}'><figcaption>{_esc(n)}</figcaption></figure>"
            for n, u in thumbs
        ]
        parts.append("</div>")

    parts.append("<h2>Model versions</h2><table>")
    parts.append(_kv_rows(res.model_versions.model_dump()) + "</table>")
    parts.append("<h2>Settings</h2><table>" + _kv_rows(settings) + "</table>")
    parts.append("<h2>Provenance</h2><table>")
    parts.append(
        _kv_rows({"schema_version": res.schema_version, "input_hash": res.input_hash, **prov})
        + "</table></body></html>"
    )
    return "".join(parts)


def write_report(out_dir: Path | str, pdf: bool = False, top_n: int = DEFAULT_TOP_N) -> list[Path]:
    """Write ``report.html`` (and ``report.pdf`` when asked and weasyprint is installed)."""
    out_dir = Path(out_dir)
    doc = render_report_html(out_dir, top_n)
    path = out_dir / "report.html"
    path.write_text(doc, encoding="utf-8")
    written = [path]
    if pdf:
        try:
            from weasyprint import HTML  # type: ignore[import-not-found,unused-ignore]
        except ImportError:
            print("countonce: weasyprint not installed; skipping PDF")
        else:
            pdf_path = out_dir / "report.pdf"
            HTML(string=doc).write_pdf(str(pdf_path))
            written.append(pdf_path)
    return written
