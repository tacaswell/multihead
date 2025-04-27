from __future__ import annotations

import importlib.metadata

import hrd_test as m


def test_version():
    assert importlib.metadata.version("hrd_test") == m.__version__
