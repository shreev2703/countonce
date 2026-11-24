"""Robust integer extraction from free-text VLM replies, plus refusal detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}  # fmt: skip
_NUM = r"\d{1,3}(?:,\d{3})+|\d+"
_MARKED = re.compile(rf"answer\s*(?:is)?\s*[:=\-]?\s*\**\s*({_NUM})", re.I)
_TOTAL = re.compile(
    rf"(?:total|count)\s*(?:is|of|=|:)\s*\**\s*(?:about|approximately|~)?\s*({_NUM})", re.I
)
_WORD_RE = re.compile(
    r"\b(" + "|".join([*_TENS, *_WORDS]) + r")"
    r"(?:[\s-]+(one|two|three|four|five|six|seven|eight|nine))?\b",
    re.I,
)  # fmt: skip
_APOS = "['’]"
_REFUSAL = re.compile(
    rf"\b(i can{_APOS}?t|i cannot|i can not|i{_APOS}m unable|i am unable"
    rf"|unable to (?:count|determine|provide|assist|help)|i{_APOS}m sorry|i apologi[sz]e"
    rf"|i won{_APOS}t|not able to (?:count|determine|assist))",
    re.I,
)


@dataclass(frozen=True)
class ParsedAnswer:
    value: int | None
    refusal: bool
    method: str  # "marker", "total", "last_line", "words", "last_number", "none"


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


def _word_value(text: str) -> int | None:
    last = None
    for m in _WORD_RE.finditer(text):
        w, unit = m.group(1).lower(), m.group(2)
        if w in _TENS:
            last = _TENS[w] + (_WORDS[unit.lower()] if unit else 0)
        else:
            last = _WORDS[w]
    return last


def parse_answer(text: str) -> ParsedAnswer:
    """Priority: last ``ANSWER: N`` marker, last "total/count is N", an integer on the last
    non-empty line, a number word on the last line, then the last integer anywhere.
    A reply with no number that reads as a refusal is flagged ``refusal=True``."""
    text = text or ""
    if m := list(_MARKED.finditer(text)):
        return ParsedAnswer(_to_int(m[-1].group(1)), False, "marker")
    if m := list(_TOTAL.finditer(text)):
        return ParsedAnswer(_to_int(m[-1].group(1)), False, "total")
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    if nums := re.findall(_NUM, last):
        return ParsedAnswer(_to_int(nums[-1]), False, "last_line")
    if (w := _word_value(last)) is not None:
        return ParsedAnswer(w, False, "words")
    if nums := re.findall(_NUM, text):
        return ParsedAnswer(_to_int(nums[-1]), False, "last_number")
    return ParsedAnswer(None, bool(_REFUSAL.search(text)), "none")
