import json
from pathlib import Path

import pytest

from twinpix import demo


@pytest.fixture(scope="session")
def library(tmp_path_factory) -> tuple[Path, dict]:
    """A small synthetic photo library with known duplicate groups (read-only)."""
    root = tmp_path_factory.mktemp("lib") / "photos"
    truth = demo.build(root, scenes=12, seed=3)
    return root, truth


@pytest.fixture()
def library_copy(library, tmp_path) -> tuple[Path, dict]:
    """A private copy for tests that move files around."""
    import shutil

    src, truth = library
    dst = tmp_path / "photos"
    shutil.copytree(src, dst)
    return dst, json.loads((dst / "truth.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINPIX_CACHE", str(tmp_path / "cache.sqlite"))
