"""CountGD / CountGD++ are planned: selecting them fails loudly; gdino is the default."""

import pytest

from countonce.config import Config
from countonce.errors import PlannedComponentError
from countonce.pipeline import build_detector
from countonce.routing import LiveRoute


def test_default_detector_is_gdino():
    assert Config().detector == "gdino"
    assert LiveRoute().detector == "gdino"


@pytest.mark.parametrize("name", ["countgd", "countgdpp"])
def test_planned_detectors_raise_clear_error(name):
    with pytest.raises(PlannedComponentError, match="planned"):
        build_detector(Config(detector=name))


def test_default_builds_hf_gdino_lazily():
    det = build_detector(Config())
    assert det.name == "groundingdino"
    assert det.backend == "hf"
