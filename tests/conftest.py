from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    base = Path.cwd() / '.test_tmp'
    base.mkdir(exist_ok=True)
    path = base / f'pytest-{uuid.uuid4().hex}'
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
