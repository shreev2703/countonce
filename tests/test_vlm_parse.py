"""Table-driven tests for the hardened VLM count parser (PLAN §9.3 item 8, §9.8b)."""

from __future__ import annotations

import pytest

from bench.vlm import parse_answer

CASES = [
    # (reply, value, refusal, method)
    ("There are 5 apples.\nConfidence: 80%", 5, False, "last_line"),
    ("I count 12 apples. I'm 95% sure.", 12, False, "last_line"),
    ("There are 7 apples (confidence: 90 percent).", 7, False, "last_line"),
    ("Apples: 14\n**Certainty:** 70", 14, False, "last_line"),
    ("There are 9 cups.\nProbability: 0.8 (85%)", 9, False, "last_line"),
    ("No apples are visible.", 0, False, "zero_phrase"),
    ("There are none.", 0, False, "zero_phrase"),
    ("Zero.", 0, False, "zero_phrase"),
    ("There are zero apples in this scene.", 0, False, "zero_phrase"),
    ("I don't see any apples. There aren't any.", 0, False, "zero_phrase"),
    ("There aren't any mugs on the table.", 0, False, "zero_phrase"),
    ("The table does not contain any cups.", 0, False, "zero_phrase"),
    ("**No** bottles can be seen.", 0, False, "zero_phrase"),
    ("The count is 0.", 0, False, "total"),
    ("There are three apples. None are red.", 3, False, "words"),
    ("I'm sorry, I can't count the people in this photo.", None, True, "none"),
    ("I cannot help with that. There are none I'd identify.", None, True, "none"),
    ("Sorry, I am unable to tell. No apples can be confirmed.", None, True, "none"),
    ("Image 3 is blurry, but there are 12 apples.", 12, False, "last_line"),
    ("There are 8 apples in view 4.", 8, False, "last_line"),
    ("Looking at photo 2, I see 15 apples.\nImages 1-3 agree.", 15, False, "last_number"),
    ("In images 1-3 there are 20 pears in total.", 20, False, "last_line"),
    ("About 40-45 apples.", 43, False, "last_line"),
    ("Roughly 40 to 45 apples.", 43, False, "last_line"),
    ("ANSWER: 100-120", 110, False, "marker"),
    ("**Answer: 37**", 37, False, "marker"),
    ("Total: **48** apples", 48, False, "total"),
    ("There are about 1,250 grains.", 1250, False, "last_line"),
    ("Between 1,000-1,200 beads.", 1100, False, "last_line"),
    ("Final count: twenty-three", 23, False, "words"),
    ("hmm", None, False, "none"),
]


@pytest.mark.parametrize(("text", "value", "refusal", "method"), CASES)
def test_parse_table(text, value, refusal, method):
    a = parse_answer(text)
    assert (a.value, a.refusal, a.method) == (value, refusal, method), text


def test_table_has_enough_cases():
    assert len(CASES) >= 25
