"""Render docs/mockups/*.html to PNG at desktop and phone widths.

Uses playwright if installed (and a browser is available); otherwise prints a message and exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

MOCK_DIR = Path(__file__).resolve().parents[1] / "docs" / "mockups"
SIZES = {"desktop": (1280, 900), "phone": (375, 812)}
SCHEMES = ("light", "dark")


def main(argv: list[str] | None = None) -> int:
    out = Path(argv[0]) if argv else MOCK_DIR / "screenshots"
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed; skipping mockup screenshots "
            "(pip install playwright && playwright install chromium)."
        )
        return 0
    out.mkdir(parents=True, exist_ok=True)
    pages = sorted(MOCK_DIR.glob("*.html"))
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # browser binary missing
            print(f"could not launch chromium ({exc}); skipping mockup screenshots.")
            return 0
        for scheme in SCHEMES:
            for name, (w, h) in SIZES.items():
                ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme)
                page = ctx.new_page()
                for html in pages:
                    page.goto(html.as_uri())
                    dest = out / f"{html.stem}-{name}-{scheme}.png"
                    page.screenshot(path=str(dest), full_page=True)
                    print("wrote", dest)
                ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
