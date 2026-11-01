"""Upload limits: settled refusal before persist or Meta call."""

from __future__ import annotations

import io
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import db
from app.admission import admit_job
from app.uploads import (
    MAX_BYTES,
    MAX_DURATION_S,
    OVER_LIMIT_MESSAGE,
    OverLimitError,
    check_upload_bounds,
    prepare_preview,
)
from app.uploads_store import UploadStoreImpl


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start if start is not None else db.utcnow()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> datetime:
        self.now = self.now + timedelta(**kwargs)
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc))


@pytest.fixture
def uploads(lane_db: Path, clock: FakeClock) -> UploadStoreImpl:
    return UploadStoreImpl(lane_db, clock=clock)


@pytest.fixture(autouse=True)
def _use_fake_clock(monkeypatch, clock: FakeClock):
    monkeypatch.setattr(db, "utcnow", clock)


def _make_clip(path: Path, *, duration: float = 1.0, with_audio: bool = True) -> Path:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc=size=160x120:rate=10:duration={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-shortest", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "short.mp4"
    return _make_clip(path, duration=1.0, with_audio=True)


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory, short_clip: Path) -> Path:
    """A real 301-second mp4 built by stream-copy concat (fast)."""
    workdir = tmp_path_factory.mktemp("long")
    listing = workdir / "list.txt"
    listing.write_text("".join(f"file '{short_clip}'\n" for _ in range(301)))
    out = workdir / "long.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)],
        check=True, capture_output=True, timeout=120,
    )
    return out


@pytest.fixture(scope="module")
def png_file(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("media") / "still.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=1:duration=1",
         "-frames:v", "1", str(out)],
        check=True, capture_output=True, timeout=120,
    )
    return out


def _admit(tag: str):
    return admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")


def _count(lane_db: Path, table: str, where: str = "") -> int:
    conn = sqlite3.connect(lane_db)
    try:
        query = f"SELECT COUNT(*) FROM {table} {where}"
        return conn.execute(query).fetchone()[0]
    finally:
        conn.close()


def _job_files(tmp_path: Path, job_id: str) -> list[Path]:
    root = tmp_path / "data" / "jobs" / str(job_id)
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def test_over_limit_refused_before_persist_or_call(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, long_clip: Path
) -> None:
    # Oversized input: refused from the stream before any persist or call.
    ctx = _admit("big")
    with pytest.raises(OverLimitError) as excinfo:
        prepare_preview(ctx, io.BytesIO(b"\x00" * (MAX_BYTES + 1)), "big.mp4",
                        uploads=uploads)
    assert excinfo.value.message == OVER_LIMIT_MESSAGE == str(excinfo.value)
    assert uploads.load_job(ctx.job_id) is None
    assert _count(lane_db, "artifacts") == 0
    assert _job_files(tmp_path, ctx.job_id) == []
    # Over-duration input: same refusal after the probe, job failed.
    ctx2 = _admit("long")
    with long_clip.open("rb") as handle:
        with pytest.raises(OverLimitError) as excinfo2:
            prepare_preview(ctx2, handle, "long.mp4", uploads=uploads)
    assert excinfo2.value.message == OVER_LIMIT_MESSAGE
    assert uploads.load_job(ctx2.job_id) is None
    assert _count(lane_db, "artifacts") == 0
    assert _job_files(tmp_path, ctx2.job_id) == []
    conn = sqlite3.connect(lane_db)
    try:
        states = {
            row[0]
            for row in conn.execute("SELECT state FROM jobs").fetchall()
        }
    finally:
        conn.close()
    assert states == {"failed"}


def test_non_video_container_or_missing_video_stream_is_refused(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, png_file: Path,
    short_clip: Path,
) -> None:
    # A PNG is refused even when it arrives named like a video: the
    # probed container decides, never the filename or content type.
    assert png_file.stat().st_size <= MAX_BYTES
    ctx = _admit("png")
    with png_file.open("rb") as handle:
        with pytest.raises(OverLimitError) as excinfo:
            prepare_preview(ctx, handle, "clip.mp4", uploads=uploads)
    assert excinfo.value.message == OVER_LIMIT_MESSAGE
    assert excinfo.value.reason == "type"
    assert uploads.load_job(ctx.job_id) is None
    assert _count(lane_db, "artifacts") == 0
    # Plain non-media bytes are refused the same settled way.
    ctx2 = _admit("txt")
    with pytest.raises(OverLimitError) as excinfo2:
        prepare_preview(ctx2, io.BytesIO(b"not a video" * 100), "clip.mp4",
                        uploads=uploads)
    assert excinfo2.value.message == OVER_LIMIT_MESSAGE
    assert uploads.load_job(ctx2.job_id) is None
    # And a real clip still passes the same bounds check.
    info = check_upload_bounds(short_clip, size_bytes=short_clip.stat().st_size)
    assert info.duration_s <= MAX_DURATION_S
    assert info.has_video_stream


def test_post_directory_failure_finishes_job_and_leaves_no_live_artifact(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, short_clip: Path,
    monkeypatch,
) -> None:
    from app import uploads as uploads_mod

    # Forced hash failure after the job directory exists.
    ctx = _admit("hashfail")

    def _boom(path: Path) -> str:
        raise RuntimeError("hash backend down")

    monkeypatch.setattr(uploads_mod, "hash_file", _boom)
    with short_clip.open("rb") as handle:
        with pytest.raises(RuntimeError, match="hash backend down"):
            prepare_preview(ctx, handle, "clip.mp4", uploads=uploads)
    assert uploads.load_job(ctx.job_id) is None
    assert _count(lane_db, "artifacts", "WHERE delete_state = 'live'") == 0
    assert _job_files(tmp_path, ctx.job_id) == []
    # Forced artifact-registry failure: same failed job, no live row.
    monkeypatch.undo()
    ctx2 = _admit("regfail")

    def _reg_boom(kind: str, path: Path):
        raise RuntimeError("registry down")

    monkeypatch.setattr(ctx2.artifacts, "register", _reg_boom)
    with short_clip.open("rb") as handle:
        with pytest.raises(RuntimeError, match="registry down"):
            prepare_preview(ctx2, handle, "clip.mp4", uploads=uploads)
    assert uploads.load_job(ctx2.job_id) is None
    assert _count(lane_db, "artifacts", "WHERE delete_state = 'live'") == 0
    assert _job_files(tmp_path, ctx2.job_id) == []
    conn = sqlite3.connect(lane_db)
    try:
        states = {
            row[0] for row in conn.execute("SELECT state FROM jobs").fetchall()
        }
    finally:
        conn.close()
    assert states == {"failed"}


def test_settled_refusal_message_is_exact() -> None:
    assert OVER_LIMIT_MESSAGE == (
        "Trim it to 300 seconds or less and try again. "
        "A 720p phone video is usually under 100 MB."
    )
    assert MAX_DURATION_S == 300.0
    assert MAX_BYTES == 100 * 1024 * 1024


def test_unknown_codec_is_refused_even_in_an_allowed_container(
    tmp_path: Path, monkeypatch,
) -> None:
    from app import uploads as uploads_mod

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"probe-is-mocked")
    mocked = uploads_mod.MediaInfo(
        duration_s=5.0,
        containers=frozenset({"mp4"}),
        has_video_stream=True,
        video_codec="bogus-codec",
        video_frames=50,
    )
    monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: mocked)
    with pytest.raises(OverLimitError) as excinfo:
        check_upload_bounds(clip, size_bytes=clip.stat().st_size)
    assert excinfo.value.message == OVER_LIMIT_MESSAGE
    assert excinfo.value.reason == "type"


def test_non_finite_or_non_positive_duration_is_refused(
    tmp_path: Path, monkeypatch,
) -> None:
    from app import uploads as uploads_mod

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"probe-is-mocked")
    for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
        mocked = uploads_mod.MediaInfo(
            duration_s=bad,
            containers=frozenset({"mp4"}),
            has_video_stream=True,
            video_codec="h264",
            video_frames=50,
        )
        monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: mocked)
        with pytest.raises(OverLimitError) as excinfo:
            check_upload_bounds(clip, size_bytes=clip.stat().st_size)
        assert excinfo.value.message == OVER_LIMIT_MESSAGE
        assert excinfo.value.reason == "duration"


def test_near_empty_clip_is_refused(tmp_path: Path, monkeypatch) -> None:
    from app import uploads as uploads_mod

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"probe-is-mocked")
    too_short = uploads_mod.MediaInfo(
        duration_s=0.2,
        containers=frozenset({"mp4"}),
        has_video_stream=True,
        video_codec="h264",
        video_frames=2,
    )
    monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: too_short)
    with pytest.raises(OverLimitError) as excinfo:
        check_upload_bounds(clip, size_bytes=clip.stat().st_size)
    assert excinfo.value.reason == "duration"
    one_frame = uploads_mod.MediaInfo(
        duration_s=1.0,
        containers=frozenset({"mov"}),
        has_video_stream=True,
        video_codec="h264",
        video_frames=1,
    )
    monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: one_frame)
    with pytest.raises(OverLimitError) as excinfo2:
        check_upload_bounds(clip, size_bytes=clip.stat().st_size)
    assert excinfo2.value.reason == "duration"


def test_unknown_frame_count_is_refused_without_rate_proof(
    tmp_path: Path, monkeypatch,
) -> None:
    from app import uploads as uploads_mod

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"probe-is-mocked")
    mocked = uploads_mod.MediaInfo(
        duration_s=1.0,
        containers=frozenset({"mp4"}),
        has_video_stream=True,
        video_codec="h264",
        video_frames=None,
    )
    monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: mocked)
    with pytest.raises(OverLimitError) as excinfo:
        check_upload_bounds(clip, size_bytes=clip.stat().st_size)
    assert excinfo.value.message == OVER_LIMIT_MESSAGE
    assert excinfo.value.reason == "duration"


def test_frame_count_prefers_nb_frames_then_trustworthy_rate() -> None:
    from app.uploads import _video_stream_facts

    counted = [{"codec_type": "video", "codec_name": "h264",
                "nb_frames": "25", "avg_frame_rate": "10/1"}]
    assert _video_stream_facts(counted, 1.0) == (True, "h264", 25)
    estimated = [{"codec_type": "video", "codec_name": "h264",
                  "avg_frame_rate": "10/1"}]
    assert _video_stream_facts(estimated, 1.0) == (True, "h264", 10)
    unusable = [{"codec_type": "video", "codec_name": "h264",
                 "avg_frame_rate": "0/0"}]
    assert _video_stream_facts(unusable, 1.0) == (True, "h264", None)


def test_frame_rate_needs_finite_strictly_positive_components() -> None:
    from app.uploads import _parse_frame_rate, _video_stream_facts

    # Probe: a positive quotient of two negatives is not a
    # frame rate, and neither is any other non-positive/non-finite part.
    for raw in (
        "-30/-1", "-30/1", "30/-1", "0/1", "30/0", "0/0",
        "nan/1", "1/nan", "inf/1", "1/inf", "-inf/2",
        "30", "", "10/1/2", "a/b", None, 30, b"10/1",
    ):
        assert _parse_frame_rate(raw) is None, raw
    assert _parse_frame_rate("10/1") == 10.0
    assert _parse_frame_rate("30000/1001") == pytest.approx(29.97, rel=1e-3)
    probe = [{"codec_type": "video", "codec_name": "h264",
              "avg_frame_rate": "-30/-1"}]
    assert _video_stream_facts(probe, 1.0) == (True, "h264", None)


def test_other_probe_numbers_stay_refused(tmp_path: Path, monkeypatch) -> None:
    from app import uploads as uploads_mod
    from app.uploads import _video_stream_facts

    # Sweep: an explicit count stays authoritative, so a negative or zero
    # nb_frames is refused downstream rather than re-estimated from rate.
    for raw_count, expected in (("-5", -5), ("0", 0)):
        streams = [{"codec_type": "video", "codec_name": "h264",
                    "nb_frames": raw_count, "avg_frame_rate": "10/1"}]
        assert _video_stream_facts(streams, 1.0) == (True, "h264", expected)
    # And non-positive/non-finite durations never pass the bounds check.
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"probe-is-mocked")
    for duration, frames in ((-1.0, 50), (0.0, 50), (float("nan"), 50)):
        mocked = uploads_mod.MediaInfo(
            duration_s=duration,
            containers=frozenset({"mp4"}),
            has_video_stream=True,
            video_codec="h264",
            video_frames=frames,
        )
        monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: mocked)
        with pytest.raises(OverLimitError) as excinfo:
            check_upload_bounds(clip, size_bytes=clip.stat().st_size)
        assert excinfo.value.reason == "duration"
    for frames in (-5, 0):
        mocked = uploads_mod.MediaInfo(
            duration_s=1.0,
            containers=frozenset({"mp4"}),
            has_video_stream=True,
            video_codec="h264",
            video_frames=frames,
        )
        monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: mocked)
        with pytest.raises(OverLimitError) as excinfo:
            check_upload_bounds(clip, size_bytes=clip.stat().st_size)
        assert excinfo.value.reason == "duration"


def test_one_frame_mjpeg_mov_is_refused(tmp_path_factory) -> None:
    out = tmp_path_factory.mktemp("media") / "still.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=10:duration=1",
         "-frames:v", "1", "-c:v", "mjpeg", str(out)],
        check=True, capture_output=True, timeout=120,
    )
    with pytest.raises(OverLimitError) as excinfo:
        check_upload_bounds(out, size_bytes=out.stat().st_size)
    assert excinfo.value.message == OVER_LIMIT_MESSAGE
    assert excinfo.value.reason == "type"


def test_too_short_real_clip_is_refused(tmp_path_factory) -> None:
    out = tmp_path_factory.mktemp("media") / "blink.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=10:duration=0.2",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True, timeout=120,
    )
    with pytest.raises(OverLimitError) as excinfo:
        check_upload_bounds(out, size_bytes=out.stat().st_size)
    assert excinfo.value.message == OVER_LIMIT_MESSAGE
    assert excinfo.value.reason == "duration"


def _drive_asgi(app, *, body: bytes, content_length: bytes | None):
    """Run one ASGI request; return (status, inner_ran, inner_seen)."""
    import asyncio

    seen: list[bytes] = []
    ran = False

    async def inner(scope, receive, send):
        nonlocal ran
        ran = True
        while True:
            message = await receive()
            assert message["type"] == "http.request"
            seen.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-length", b"2")]})
        await send({"type": "http.response.body", "body": b"ok"})

    wrapped = app(inner)
    headers = []
    if content_length is not None:
        headers.append((b"content-length", content_length))
    scope = {"type": "http", "method": "POST", "path": "/upload",
             "headers": headers}
    sent: list[dict] = []
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(wrapped(scope, receive, send))
    status = next(m["status"] for m in sent
                  if m["type"] == "http.response.start")
    return status, ran, b"".join(seen)


def test_body_cap_rejects_by_content_length_before_any_read() -> None:
    from app.body_limit import MAX_BODY_BYTES, BodyLimitMiddleware
    from app.uploads import MAX_BYTES as UPLOAD_MAX

    assert MAX_BODY_BYTES > UPLOAD_MAX  # bounded multipart overhead only
    assert MAX_BODY_BYTES <= UPLOAD_MAX + 2 * 1024 * 1024
    status, ran, _ = _drive_asgi(
        lambda inner: BodyLimitMiddleware(inner, max_body_bytes=1024),
        body=b"x" * 2048, content_length=b"2048",
    )
    assert status == 413
    assert ran is False


def test_body_cap_counts_streamed_bytes_without_a_length() -> None:
    from app.body_limit import BodyLimitMiddleware

    status, _, _ = _drive_asgi(
        lambda inner: BodyLimitMiddleware(inner, max_body_bytes=1024),
        body=b"x" * 2048, content_length=None,
    )
    assert status == 413
    ok_status, ok_ran, ok_seen = _drive_asgi(
        lambda inner: BodyLimitMiddleware(inner, max_body_bytes=1024),
        body=b"x" * 512, content_length=None,
    )
    assert (ok_status, ok_ran, ok_seen) == (200, True, b"x" * 512)


def test_body_cap_ignores_non_http_scopes() -> None:
    import asyncio

    from app.body_limit import BodyLimitMiddleware

    calls: list[str] = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    async def _noop():
        return {"type": "lifespan.startup"}

    async def _send(message):
        pass

    asyncio.run(
        BodyLimitMiddleware(inner)({"type": "lifespan"}, _noop, _send)
    )
    assert calls == ["lifespan"]
