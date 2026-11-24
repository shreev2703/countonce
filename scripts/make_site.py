# ruff: noqa: E501
"""Build a self-contained static project page ``site/index.html`` (PLAN §16B.4).

    python scripts/make_site.py [--csv results/bench.csv] [--figures results/figures] \
        [--readme README.md] [--out site/index.html]

Sections: headline chart, results table, failure gallery, limitations. Images are inlined as
base64 so the page is one file with no network references. Anything missing renders a visible
``TODO: fill from results`` placeholder; no number is invented.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # bench/ lives at the repo root

TODO = "TODO: fill from results"
HEADLINE = "mape_vs_views.png"


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def _placeholder(what: str) -> str:
    return f'<p class="todo">{TODO} ({html.escape(what)})</p>'


def headline(figures: Path) -> str:
    p = figures / HEADLINE
    return (
        _img(p, "Count error versus number of views")
        if p.exists()
        else _placeholder(f"missing {p}")
    )


def results_table(csv_path: Path) -> str:
    if not csv_path.exists():
        return _placeholder(f"missing {csv_path}")
    from bench.figures import load_rows, rel_error, scene_level
    from bench.stats import bootstrap_ci

    rows = [r for r in load_rows(csv_path) if r.ordered]
    if not rows:
        return _placeholder("no usable rows in bench.csv")
    body = []
    for m in sorted({r.method for r in rows}):
        g = [r for r in rows if r.method == m]
        vals = list(scene_level(g, rel_error).values())
        if vals:
            ci = bootstrap_ci(vals, n_boot=2000, seed=0)
            cell = f"{ci.estimate:.1f}% [{ci.low:.1f}, {ci.high:.1f}]"
        else:
            cell = TODO
        body.append(
            f"<tr><td>{html.escape(m)}</td><td>{len({r.scene for r in g})}</td><td>{cell}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Method</th><th>Scenes</th><th>MAPE (95% CI)</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
    )  # fmt: skip


def failure_gallery(csv_path: Path, runs_dir: Path | None, limit: int = 6) -> str:
    if not csv_path.exists():
        return _placeholder(f"missing {csv_path}")
    from bench.figures import load_rows, rel_error
    from bench.gallery import contact_sheet, failure_cause, overlay_paths, worst_fraction

    rows = load_rows(csv_path)
    if not rows:
        return _placeholder("no usable rows in bench.csv")
    items = []
    for r in worst_fraction(rows)[:limit]:
        e = rel_error(r)
        err = f"{e:.1f}%" if e == e else f"{abs(r.pred - r.gt):g} (gt=0)"
        files = overlay_paths(r, runs_dir)
        pic = _placeholder("no overlay for this run")
        if files:
            from PIL import Image

            buf = io.BytesIO()
            sheet = contact_sheet(files, _tmp_png(r))
            with Image.open(sheet) as im:
                im.save(buf, "PNG")
            pic = f'<img alt="overlays" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
        items.append(
            f"<figure>{pic}<figcaption>{html.escape(r.scene)} / {html.escape(r.method)}, "
            f"{r.n_views} views: predicted {r.pred:g}, truth {r.gt:g} ({err}); "
            f"{html.escape(failure_cause(r))}</figcaption></figure>"
        )
    return (
        '<div class="gallery">' + "".join(items) + "</div>"
        if items
        else _placeholder("no failures")
    )


def _tmp_png(r) -> Path:  # type: ignore[no-untyped-def]
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="site_sheet_"))
    return d / (
        re.sub(r"[^A-Za-z0-9.-]", "-", f"{r.scene}-{r.method}-{r.n_views}-{r.sample}") + ".png"
    )


def limitations(readme: Path) -> str:
    """Bullets under the README's ``## Limitations`` heading, rendered as plain text."""
    if not readme.exists():
        return _placeholder(f"missing {readme}")
    m = re.search(r"^## Limitations\s*\n(.*?)(?=^## |\Z)", readme.read_text(), re.S | re.M)
    items = re.findall(r"^- (.+?)(?=^- |\Z)", m.group(1), re.S | re.M) if m else []
    if not items:
        return _placeholder("no Limitations section in README")
    out = []
    for it in items:
        t = " ".join(it.split())
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        out.append(f"<li>{t}</li>")
    return "<ul>" + "".join(out) + "</ul>"


CSS = """body{font:16px/1.6 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fff}
@media(prefers-color-scheme:dark){body{background:#121212;color:#e6e6e6}}
h1,h2{line-height:1.2}img{max-width:100%}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #8884;padding:.4rem;text-align:left}.todo{color:#b45309;font-style:italic}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(18rem,1fr));gap:1rem}figcaption{font-size:.9rem}"""


def build(csv_path: Path, figures: Path, readme: Path, runs_dir: Path | None = None) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CountOnce</title><style>{CSS}</style></head><body>
<h1>CountOnce</h1>
<p>Count objects across unordered photos without counting any of them twice.</p>
<h2>Headline</h2>{headline(figures)}
<h2>Results</h2>{results_table(csv_path)}
<h2>Failure gallery</h2>
<p>Worst runs by error, sampled systematically, not hand-picked.</p>
{failure_gallery(csv_path, runs_dir)}
<h2>Limitations</h2>{limitations(readme)}
<h2>Links</h2><p>Code, model, data and demo: TODO: fill from results</p>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=Path("results/bench.csv"))
    ap.add_argument("--figures", type=Path, default=Path("results/figures"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("site/index.html"))
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build(a.csv, a.figures, a.readme, a.runs_dir))
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
