"""
Shared pytest fixtures.

The dataset tests need the real archive at data/raw/cifar-10-python.tar.gz.
If it is missing they are SKIPPED (not failed) with a clear message, so the
model tests still run on a machine without the data.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import ARCHIVE, build_datasets  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load(open(REPO_ROOT / "configs" / "baseline.yaml"))


@pytest.fixture(scope="session")
def data():
    """Loaded once per test session (~5 s), shared by all dataset tests."""
    if not ARCHIVE.exists():
        pytest.skip(f"dataset archive not found: {ARCHIVE}")
    return build_datasets()
