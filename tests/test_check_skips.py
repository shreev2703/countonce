from scripts.check_skips import check

LOG = """SKIPPED [1] tests/a.py:1: could not import 'pycolmap': No module named 'pycolmap'
SKIPPED [2] tests/b.py:2: could not import 'torch': No module named 'torch'
"""


def test_allow_list_and_offenders():
    total, bad = check(LOG)
    assert total == 3
    assert len(bad) == 1 and "torch" in bad[0]


def test_clean_log():
    assert check(LOG.splitlines()[0])[1] == []
