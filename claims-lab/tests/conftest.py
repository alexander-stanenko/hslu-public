import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator"))


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("CLAIMS_LAB_NONINTERACTIVE", "1")
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "anthropic")
    yield
