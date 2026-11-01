"""Live smoke: one Spark call on the 60 s clip, one SAM call on one still.

Skipped unless META_API_KEY, META_SMOKE_CLIP and META_SMOKE_STILL are all
set in the environment at collection time (the suite-wide fixture
strips the key before each test body runs, so this module captures the
three values at import). Footage paths stay outside the repo; the key
is never printed, logged, or stored. Total ledger spend must stay
under USD 0.50.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from decimal import Decimal
from pathlib import Path

import pytest

_LIVE_KEY = os.environ.get("META_API_KEY", "").strip()
_SMOKE_CLIP = os.environ.get("META_SMOKE_CLIP", "").strip()
_SMOKE_STILL = os.environ.get("META_SMOKE_STILL", "").strip()

pytestmark = pytest.mark.skipif(
    not (_LIVE_KEY and _SMOKE_CLIP and _SMOKE_STILL),
    reason="live smoke needs META_API_KEY, META_SMOKE_CLIP and META_SMOKE_STILL",
)


def _multipart_upload(base_url: str, key: str, path: Path) -> str:
    """Test-only fallback upload (spike-measured multipart shape).

    Used only when the existing meta_files client cannot upload to the
    real endpoint; the app itself keeps exactly one uploader.
    """
    import uuid

    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    def field(name: str, value: object) -> None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n".encode()
        )

    field("purpose", "user_data")
    field("expires_after[anchor]", "created_at")
    field("expires_after[seconds]", 3600)
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{path.name}"\r\nContent-Type: video/mp4\r\n\r\n'.encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        base_url + "/files",
        data=b"".join(parts),
        method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
    )
    with urllib.request.urlopen(request, timeout=180) as reply:
        payload = json.loads(reply.read().decode("utf-8"))
    file_id = payload.get("id")
    assert isinstance(file_id, str) and file_id, f"upload gave {payload!r}"
    return file_id


def _direct_delete(base_url: str, key: str, file_id: str) -> None:
    request = urllib.request.Request(
        base_url + "/files/" + file_id,
        method="DELETE",
        headers={"Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(request, timeout=60):
        pass


def test_live_spark_and_sam_smoke(tmp_path, capsys):
    from app import db
    from app.admission import admit_job
    from app import evidence as _evidence
    from app.contracts import MetaFileId
    from app.meta_files import MetaFilesError, ProductionMetaFilesClient
    from app.providers import select_providers
    from app.sam import segment
    from app.settings import load_settings

    clip = Path(_SMOKE_CLIP)
    still_src = Path(_SMOKE_STILL)
    assert clip.is_file() and still_src.is_file()
    database = tmp_path / "live.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    ctx = admit_job("anon-live", "ip-live", "video-live")
    settings = load_settings(
        {"PARTFIDELITY_PROVIDER_MODE": "live", "META_API_KEY": _LIVE_KEY}
    )
    trio = select_providers(settings, ctx)
    base_url = settings.providers.meta_api_base_url

    files = ProductionMetaFilesClient(
        _LIVE_KEY, base_url=base_url, timeout_s=180
    )
    file_id: str | None = None
    upload_path = "meta_files_client"
    try:
        try:
            uploaded = files.upload_file(
                clip, expires_after_hours=1, idempotency_key="live-smoke-1"
            )
            file_id = str(uploaded.file_id)
        except MetaFilesError as error:
            upload_path = f"test_multipart_fallback(meta_files_client:{error})"
            file_id = _multipart_upload(base_url, _LIVE_KEY, clip)

        hold = ctx.reserve("spark_steps", Decimal("0.09"))
        try:
            spark = trio.spark.run_steps(file_id)
        except Exception:
            hold.settle(Decimal("0"), "failed")
            raise
        hold.settle(spark.cost_usd, "succeeded")

        still = tmp_path / "smoke_still.png"
        shutil.copyfile(still_src, still)
        detections = segment(ctx, still, "battery", client=trio.sam)

        total = spark.cost_usd + Decimal("0.0025")
        calls = _evidence.list_calls(ctx)
        print(f"smoke upload path: {upload_path}")
        print(f"smoke file id: {file_id}")
        print(f"smoke spark cost/latency: {spark.cost_usd} / {spark.latency_ms}ms")
        print(f"smoke sam detections: {len(detections)}")
        print(f"smoke total settled: {total}")
        print(f"smoke evidence rows: {len(calls)}")
        assert total < Decimal("0.50")
        assert len(calls) == 2
        assert total == sum(
            (c.estimated_cost_usd for c in calls), Decimal("0")
        )
    finally:
        if file_id is not None:
            try:
                files.delete_file(MetaFileId(file_id))
            except MetaFilesError:
                _direct_delete(base_url, _LIVE_KEY, file_id)
