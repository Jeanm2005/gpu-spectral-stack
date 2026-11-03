"""Shared pytest setup: the gpu marker and a reproducible random generator for each test."""

import zlib

import numpy as np
import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skips every test marked `gpu` when no CUDA device is visible (as on CI runners)."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng(request) -> np.random.Generator:
    """A generator seeded from the test's id, so each test (and each parameter case) draws the
    same numbers on every run, whatever order the tests run in."""
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))
