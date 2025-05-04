from __future__ import annotations

import importlib.metadata

import multihead as m


def test_version():
    assert importlib.metadata.version("multihead") == m.__version__
