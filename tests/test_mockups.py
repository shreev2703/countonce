"""Static UI mockups (PLAN 17) stay valid: parse, titled, alt text, local assets, real errors."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from countonce.errors import ErrorCode, user_tip

MOCK = Path(__file__).resolve().parents[1] / "docs" / "mockups"
PAGES = ["upload", "progress", "results", "history", "states"]
VOID = {"meta", "link", "img", "input", "br", "hr"}


class _Scan(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.title = ""
        self._in_title = False
        self.imgs: list[dict[str, str | None]] = []
        self.refs: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "img":
            self.imgs.append(a)
        for key in ("href", "src"):
            if a.get(key) is not None and tag in {"a", "link", "script", "img"}:
                self.refs.append(a[key] or "")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def _scan(name: str) -> _Scan:
    s = _Scan()
    s.feed((MOCK / f"{name}.html").read_text(encoding="utf-8"))
    s.close()
    return s


@pytest.mark.parametrize("name", PAGES)
def test_page_parses_titled_alt_and_local(name):
    s = _scan(name)
    assert not s.errors and not s.stack, (s.errors, s.stack)
    assert s.title.strip()
    for img in s.imgs:
        assert (img.get("alt") or "").strip(), f"{name}: img without alt: {img}"
    for ref in s.refs:
        assert not re.match(r"^([a-z][a-z0-9+.-]*:|//)", ref, re.I), f"{name}: non-local {ref}"
        target = ref.split("#")[0]
        if target:
            assert (MOCK / target).is_file(), f"{name}: missing local file {target}"


@pytest.mark.parametrize("name", PAGES)
def test_no_network_assets_in_css_or_page(name):
    text = (MOCK / f"{name}.html").read_text(encoding="utf-8") + (MOCK / "style.css").read_text()
    assert "@import" not in text
    assert not re.search(r"url\(\s*['\"]?(https?:)?//", text)


def test_error_messages_come_from_taxonomy():
    text = html.unescape((MOCK / "states.html").read_text(encoding="utf-8"))
    for code in (
        ErrorCode.ERR_UNSUPPORTED_CODEC,
        ErrorCode.ERR_TOO_FEW_FRAMES,
        ErrorCode.ERR_NO_OVERLAP,
        ErrorCode.ERR_FILE_TOO_LARGE,
        ErrorCode.ERR_GPU_OOM,
        ErrorCode.ERR_CANNOT_LINK,
    ):
        assert user_tip(code) in text and code.value in text


def test_results_page_has_required_pieces():
    text = (MOCK / "results.html").read_text(encoding="utf-8")
    for needle in (
        "47",
        "44",
        "51",
        "NEW",
        "DUPLICATE",
        "REJECTED",
        "Split",
        "Merge",
        "Reject",
        "share",
        "MOCKUP",
    ):
        assert needle in text
    assert "no worker is online" in (MOCK / "progress.html").read_text(encoding="utf-8")


def _lum(hex_):
    r, g, b = (int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_palettes_meet_wcag_aa():
    css = (MOCK / "style.css").read_text(encoding="utf-8")
    blocks = re.findall(r"(:root[^{]*)\{([^}]*)\}", css)
    palettes = []
    for _sel, body in blocks:
        v = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", body))
        if "bg" in v:
            palettes.append(v)
    assert len(palettes) == 3  # light, dark via media query, dark via toggle
    pairs = [
        ("text", "bg"),
        ("text", "surface"),
        ("muted", "bg"),
        ("muted", "surface"),
        ("accent", "bg"),
        ("accent", "surface"),
        ("on-accent", "accent"),
        ("new-fg", "new-bg"),
        ("dup-fg", "dup-bg"),
        ("rej-fg", "rej-bg"),
        ("warn-fg", "warn-bg"),
        ("err-fg", "err-bg"),
        ("focus", "bg"),
    ]
    for p in palettes:
        for fg, bg in pairs:
            assert _ratio(p[fg], p[bg]) >= 4.5, (fg, bg, p[fg], p[bg])
