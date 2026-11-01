"""Shared fixtures: fake clients plus env -i style minimum configuration.

No test reads a real key or makes a paid call. The autouse fixture strips
secret-bearing variables so tests must inject fakes explicitly.
"""

from __future__ import annotations

import pytest

from app.contracts import MetaFilesClient, SamClient, SparkClient
from app.settings import Settings


@pytest.fixture(scope="session", autouse=True)
def _film_frames():
    from tools.film_frames import unpack

    unpack()


@pytest.fixture(autouse=True)
def _clean_process_env(monkeypatch):
    for var in (
        "META_API_KEY",
        "BASETEN_API_KEY",
        "META_FILES_BASE_URL",
        "PARTFIDELITY_PRODUCTION",
        "PARTFIDELITY_DATA_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _fresh_open_readback_workers(monkeypatch):
    """Each test gets its own second-check pool, stopped afterwards."""
    from app.providers import open_readback

    runner = open_readback._Runner()
    monkeypatch.setattr(open_readback, "_RUNNER", runner)
    yield
    runner.shutdown(0)


class FakeSparkClient:
    """Test double for SparkClient."""


class FakeSamClient:
    """Test double for SamClient."""


class FakeMetaFilesClient:
    """Test double for MetaFilesClient."""

    def upload_file(self, path, *, expires_after_hours, idempotency_key):
        raise NotImplementedError

    def find_upload(self, idempotency_key):
        raise NotImplementedError

    def delete_file(self, file_id):
        raise NotImplementedError

    def get_file(self, file_id):
        raise NotImplementedError


assert isinstance(FakeSparkClient(), SparkClient)
assert isinstance(FakeSamClient(), SamClient)
assert isinstance(FakeMetaFilesClient(), MetaFilesClient)


@pytest.fixture
def fake_spark() -> FakeSparkClient:
    return FakeSparkClient()


@pytest.fixture
def fake_sam() -> FakeSamClient:
    return FakeSamClient()


@pytest.fixture
def fake_meta_files() -> FakeMetaFilesClient:
    return FakeMetaFilesClient()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_root=tmp_path, db_path=tmp_path / "test.sqlite3", production=False)


@pytest.fixture(scope="session")
def synthetic_sample_bundle(tmp_path_factory):
    """A tiny stand-in for tests that exercise the guide, never the footage."""
    import shutil
    import cv2
    import numpy as np
    from pathlib import Path
    from app import service
    from tools.sample_media import FILES

    source = Path(__file__).resolve().parents[1] / "app/sample_real"
    dest = tmp_path_factory.mktemp("sample-media")
    for name in ("guide.json", "receipts.json", "step_notes.json"):
        shutil.copy(source / name, dest / name)
    for name in FILES[1:]:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))
    service.write_sample_clip(dest / "sample.mp4")
    return dest


@pytest.fixture(autouse=True)
def _sample_fixture_when_media_absent(monkeypatch, synthetic_sample_bundle):
    from app import service
    from tools.sample_media import ROOT, FILES

    if any(not (ROOT / "app/sample_real" / name).is_file() for name in FILES):
        monkeypatch.setattr(service, "_sample_bundle", lambda settings: synthetic_sample_bundle)
