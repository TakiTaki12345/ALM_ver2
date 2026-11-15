import importlib

import pytest

SUBPACKAGES = ["core", "scenarios", "banking", "insurance", "analysis"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name: str) -> None:
    importlib.import_module(f"alm.{name}")
