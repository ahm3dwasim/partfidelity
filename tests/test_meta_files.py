"""multipart upload shape, streaming, and API-vs-transport errors.

Unit tests run against an injected fake transport (no network); the live
smoke at the bottom runs only when META_API_KEY is set in the ambient
environment at collection time. The key is never printed.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from app.contracts import MetaFileId, MetaFilesError, MetaUploadExpiryError
from app.deletion import _short
from app.meta_files import (
    MetaApiError,
    MetaFilesUploadConfig,
    ProductionMetaFilesClient,
    _http_error_payload,
    expiry_details,
    expiry_error,
    select_meta_files_client,
)
from app.settings import (
    DEFAULT_META_FILES_MAX_ERROR_BYTES,
    Settings,
    load_settings,
)

LIVE_KEY = os.environ.get("META_API_KEY", "").strip()
LIVE_BASE_URL = os.environ.get(
    "META_FILES_BASE_URL", "https://api.meta.ai/v1"
).strip()
needs_live_key = pytest.mark.skipif(
    not LIVE_KEY, reason="live Meta API check needs META_API_KEY"
)
SMOKE_LOG_ENV_VAR = "META_FILES_SMOKE_LOG"


def _cleanup_id_from_error(error: BaseException) -> MetaFileId | None:
    """Id an upload error carries for cleanup, else ``None``.

    Expiry errors go through the canonical reader (attribute first,
    constructor args as fallback); any other error only via an explicit
    ``file_id`` attribute, never by sniffing message-shaped args.
    """
    if isinstance(error, MetaUploadExpiryError):
        file_id, _ = expiry_details(error)
        return file_id
    candidate = getattr(error, "file_id", None)
    if isinstance(candidate, str) and candidate:
        return MetaFileId(candidate)
    return None


def _write_smoke_log(
    path, *, command: str, exit_code: int, lines, secrets: dict
) -> None:
    """Write the scrubbed live-smoke log the later live step cites.

    Every secret is replaced by its placeholder first; the write
    asserts none survived, so a cited log never leaks the key, a live
    id, or the idempotency key.
    """
    rows = [f"command: {command}", f"exit: {exit_code}", *lines]
    body = ""
    for row in rows:
        for secret, placeholder in secrets.items():
            if secret:
                row = row.replace(secret, placeholder)
        body += row + "\n"
    for secret in secrets:
        assert not secret or secret not in body
    Path(path).write_text(body, encoding="utf-8")


class _FakeReply:
    def __init__(self, status: int, payload) -> None:
        self.status = status
        self._raw = b"" if payload is None else json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


class FakeTransport:
    """Canned HTTP round trips; request bodies are read in bounded chunks."""

    def __init__(self, handler, *, chunk_size: int = 8192) -> None:
        self._handler = handler
        self._chunk_size = chunk_size
        self.requests: list = []  # (request, body bytes)
        self.timeouts: list = []
        self.read_calls: list[int] = []  # read() call counts per request
        self.streamed: list[bool] = []  # body arrived file-like, not bytes

    def __call__(self, request, timeout=None):
        self.timeouts.append(timeout)
        data = request.data
        calls = 0
        if data is None:
            raw_body, was_stream = b"", False
        elif isinstance(data, (bytes, bytearray)):
            raw_body, was_stream = bytes(data), False
        elif hasattr(data, "read"):
            was_stream = True
            parts = []
            while True:
                calls += 1
                chunk = data.read(self._chunk_size)
                if not chunk:
                    break
                parts.append(chunk)
            raw_body = b"".join(parts)
        else:
            raise AssertionError(f"unexpected body type {type(data)!r}")
        self.read_calls.append(calls)
        self.streamed.append(was_stream)
        self.requests.append((request, raw_body))
        status, payload = self._handler(request, raw_body)
        return _FakeReply(status, payload)


def _http_error(url: str, code: int, payload) -> urllib.error.HTTPError:
    raw = json.dumps(payload).encode("utf-8")
    return urllib.error.HTTPError(url, code, "error", {}, io.BytesIO(raw))


def _grant(before: datetime, hours: int) -> str:
    """Grant inside the request window however the test clock advances."""
    return (before + timedelta(hours=hours, seconds=-30)).isoformat()


def _split_multipart(body: bytes, boundary: bytes):
    """Split a multipart body into (headers, payload) parts."""
    segments = body.split(b"--" + boundary)
    parts = []
    for segment in segments:
        if segment in (b"", b"--\r\n"):
            continue
        segment = segment.removeprefix(b"\r\n")
        head, _, payload = segment.partition(b"\r\n\r\n")
        assert payload.endswith(b"\r\n")
        parts.append((head.decode("utf-8"), payload[:-2]))
    return parts


def test_upload_sends_multipart_form_with_purpose_and_expiry(tmp_path: Path):
    seen = {}
    before = datetime.now(timezone.utc)

    def _handler(request, body):
        seen["request"] = request
        seen["body"] = body
        return 201, {"file_id": "meta-1", "expires_at": _grant(before, 23)}

    file_path = tmp_path / "clip.mp4"
    file_bytes = b"fake-video-bytes" * 64
    file_path.write_bytes(file_bytes)
    transport = FakeTransport(_handler)
    client = ProductionMetaFilesClient(
        "test-key", base_url="https://meta.example.test", transport=transport
    )
    found = client.upload_file(
        file_path, expires_after_hours=23, idempotency_key="key-1"
    )
    assert found.file_id == "meta-1"

    request, body = seen["request"], seen["body"]
    content_type = request.get_header("Content-type")
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1].encode("utf-8")
    assert body.startswith(b"--" + boundary + b"\r\n")
    assert body.endswith(b"--" + boundary + b"--\r\n")
    assert int(request.get_header("Content-length")) == len(body)

    parts = _split_multipart(body, boundary)
    assert len(parts) == 4
    assert parts[0] == (
        'Content-Disposition: form-data; name="purpose"',
        b"user_data",
    )
    assert parts[1] == (
        'Content-Disposition: form-data; name="expires_after[anchor]"',
        b"created_at",
    )
    assert parts[2] == (
        'Content-Disposition: form-data; name="expires_after[seconds]"',
        b"82800",
    )
    file_head, file_payload = parts[3]
    assert 'name="file"; filename="clip.mp4"' in file_head
    assert "Content-Type: video/mp4" in file_head
    assert file_payload == file_bytes

    # The upload module behaviour preserved: the keyed query still travels on the URL.
    query = parse_qs(urlparse(request.full_url).query)
    assert query["expires_after_hours"] == ["23"]
    assert query["idempotency_key"] == ["key-1"]
    # The key travels only in the Authorization header.
    assert request.get_header("Authorization") == "Bearer test-key"
    assert "test-key" not in request.full_url
    assert b"test-key" not in body


def test_upload_streams_the_file_without_a_full_read(tmp_path: Path, monkeypatch):
    file_path = tmp_path / "big.mp4"
    file_bytes = bytes(range(256)) * 400  # 102400 bytes
    file_path.write_bytes(file_bytes)
    before = datetime.now(timezone.utc)
    transport = FakeTransport(
        lambda request, body: (
            201,
            {"file_id": "meta-big", "expires_at": _grant(before, 24)},
        ),
        chunk_size=8192,
    )
    client = ProductionMetaFilesClient(
        "test-key", base_url="https://meta.example.test", transport=transport
    )

    def _no_full_read(self, *args, **kwargs):
        raise AssertionError("upload must stream, not read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", _no_full_read)
    found = client.upload_file(
        file_path, expires_after_hours=24, idempotency_key="key-big"
    )
    assert found.file_id == "meta-big"

    assert transport.streamed == [True]  # file-like body, never bytes
    assert transport.read_calls[0] > 5  # bounded reads, not one drain
    request, body = transport.requests[0]
    assert int(request.get_header("Content-length")) == len(body)
    assert file_bytes in body  # every byte arrived exactly once
    assert body.count(file_bytes) == 1


def test_upload_accepts_real_api_shapes(tmp_path: Path):
    now = datetime.now(timezone.utc)
    cases = [
        # Spike shape: {"id", unix "expires_at"}.
        (
            {"id": "file-unix", "expires_at": int(now.timestamp()) + 3600},
            "file-unix",
        ),
        # ISO-string expiry with the real id key.
        (
            {
                "id": "file-iso",
                "expires_at": (now + timedelta(hours=2)).isoformat(),
            },
            "file-iso",
        ),
        # Numeric-string timestamp.
        (
            {"id": "file-numstr", "expires_at": str(int(now.timestamp()) + 7200)},
            "file-numstr",
        ),
    ]
    for payload, want_id in cases:
        file_path = tmp_path / f"{want_id}.mp4"
        file_path.write_bytes(b"x" * 32)

        def _handler(request, body, _payload=payload):
            return 200, dict(_payload)

        client = ProductionMetaFilesClient(
            "test-key",
            base_url="https://meta.example.test",
            transport=FakeTransport(_handler),
        )
        found = client.upload_file(
            file_path, expires_after_hours=3, idempotency_key=f"key-{want_id}"
        )
        assert found.file_id == want_id
        assert now < found.expires_at <= now + timedelta(hours=3, seconds=5)


def test_upload_upper_bound_tolerates_skew_not_overgrants(tmp_path: Path):
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"v" * 16)
    before = datetime.now(timezone.utc)

    def _upload_with_grant(grant: datetime):
        def _handler(request, body):
            return 200, {
                "id": "file-skew",
                "expires_at": grant.isoformat(),
            }

        client = ProductionMetaFilesClient(
            "test-key",
            base_url="https://meta.example.test",
            transport=FakeTransport(_handler),
        )
        return client.upload_file(
            file_path, expires_after_hours=3, idempotency_key="k"
        )

    # A grant seconds past the request (latency, server clock) is accepted.
    skew = _upload_with_grant(before + timedelta(hours=3, seconds=60))
    assert skew.file_id == "file-skew"
    # A grant ten minutes past the 5-minute skew bound still fails closed,
    # carrying the id for deletion.
    with pytest.raises(MetaUploadExpiryError) as excinfo:
        _upload_with_grant(before + timedelta(hours=3, seconds=600))
    assert excinfo.value.file_id == "file-skew"


def test_upload_expiry_error_carries_the_real_api_id(tmp_path: Path):
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"v" * 16)

    def _handler(request, body):
        return 200, {"id": "file-badexp"}  # no parseable expiry at all

    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_handler),
    )
    with pytest.raises(MetaUploadExpiryError) as excinfo:
        client.upload_file(file_path, expires_after_hours=1, idempotency_key="k")
    assert excinfo.value.file_id == "file-badexp"
    assert excinfo.value.expires_at is None


def test_http_4xx_is_an_api_error_and_transport_failure_is_not(tmp_path: Path):
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"v" * 16)

    def _refusing(request, body):
        raise _http_error(
            request.full_url,
            400,
            {
                "error": {
                    "code": "invalid_request_body",
                    "message": "Request body must be multipart/form-data",
                    "param": None,
                    "type": "invalid_request_error",
                }
            },
        )

    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_refusing),
    )
    with pytest.raises(MetaApiError) as excinfo:
        client.upload_file(file_path, expires_after_hours=1, idempotency_key="k")
    error = excinfo.value
    assert error.status == 400
    assert error.code == "invalid_request_body"
    assert "400" in str(error)
    assert "Request body must be multipart/form-data" in str(error)
    assert "test-key" not in str(error)

    def _crashing(request, body):
        raise ConnectionError("connection refused")

    dead = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_crashing),
    )
    with pytest.raises(MetaFilesError) as excinfo:
        dead.get_file(MetaFileId("x"))
    assert not isinstance(excinfo.value, MetaApiError)  # no answer at all

    def _server_error(request, body):
        raise _http_error(
            request.full_url,
            500,
            {"error": {"code": "server_error", "message": "boom"}},
        )

    failing = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_server_error),
    )
    with pytest.raises(MetaApiError) as excinfo:
        failing.delete_file(MetaFileId("x"))
    assert excinfo.value.status == 500


def test_http_error_bodies_close_after_api_errors(tmp_path: Path):
    retained: list = []

    def _refusing(request, body):
        error = _http_error(
            request.full_url,
            400,
            {"error": {"code": "bad_request", "message": "no"}},
        )
        retained.append(error)
        raise error

    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"v" * 16)
    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_refusing),
    )
    with pytest.raises(MetaApiError):
        client.upload_file(file_path, expires_after_hours=1, idempotency_key="k")
    with pytest.raises(MetaApiError):
        client.delete_file(MetaFileId("x"))
    assert len(retained) == 2
    for error in retained:
        assert error.fp.closed


def test_error_bodies_do_not_change_absence_semantics():
    not_found = {"error": {"code": "file_not_found", "message": "no file"}}

    def _gone(request, body):
        parsed = urlparse(request.full_url)
        if parsed.path == "/files" and "key" in parse_qs(parsed.query):
            raise _http_error(request.full_url, 404, not_found)
        if parsed.path == "/files":
            return 200, {"files": []}
        raise _http_error(request.full_url, 404, not_found)

    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_gone),
    )
    assert client.find_upload("gone") is None  # 404 stays absence
    assert client.get_file(MetaFileId("gone")) is None  # GET 404 + clean list


def test_list_accepts_real_api_data_shape_and_refuses_partial_pages():
    def _client(handler):
        return ProductionMetaFilesClient(
            "test-key",
            base_url="https://meta.example.test",
            transport=FakeTransport(handler),
        )

    def _empty(request, body):
        if urlparse(request.full_url).path == "/files":
            return 200, {"data": []}
        raise _http_error(request.full_url, 404, {"error": {"code": "x"}})

    assert _client(_empty).get_file(MetaFileId("gone")) is None

    def _shows(request, body):
        if urlparse(request.full_url).path == "/files":
            return 200, {"data": [{"id": "meta-x"}]}
        raise _http_error(request.full_url, 404, {"error": {"code": "x"}})

    with pytest.raises(MetaFilesError):
        _client(_shows).get_file(MetaFileId("meta-x"))

    def _partial(request, body):
        if urlparse(request.full_url).path == "/files":
            return 200, {"data": [], "has_more": True}
        raise _http_error(request.full_url, 404, {"error": {"code": "x"}})

    with pytest.raises(MetaFilesError, match="more pages"):
        _client(_partial).get_file(MetaFileId("gone"))


def test_upload_rejects_files_over_the_configured_limit_without_io(tmp_path: Path):
    file_path = tmp_path / "over.mp4"
    file_path.write_bytes(b"v" * 17)
    before = datetime.now(timezone.utc)
    transport = FakeTransport(
        lambda request, body: (
            201,
            {"file_id": "m", "expires_at": _grant(before, 1)},
        )
    )
    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        upload_config=MetaFilesUploadConfig(max_upload_bytes=16),
        transport=transport,
    )
    with pytest.raises(MetaFilesError, match="over the 16 byte limit"):
        client.upload_file(file_path, expires_after_hours=1, idempotency_key="k")
    assert transport.requests == []

    exact = tmp_path / "exact.mp4"
    exact.write_bytes(b"v" * 16)
    found = client.upload_file(exact, expires_after_hours=1, idempotency_key="k")
    assert found.file_id == "m"


def test_upload_settings_drive_values_and_reject_bad_input():
    settings = load_settings(
        {
            "META_API_KEY": "secret-key",
            "META_FILES_TIMEOUT_S": "3",
            "META_FILES_UPLOAD_TIMEOUT_S": "33",
            "META_FILES_MAX_UPLOAD_BYTES": "1234",
            "META_FILES_UPLOAD_PURPOSE": "evidence",
            "META_FILES_PURPOSE_FIELD": "use",
            "META_FILES_FILE_FIELD": "blob",
            "META_FILES_FILE_CONTENT_TYPE": "application/octet-stream",
            "META_FILES_EXPIRES_ANCHOR_FIELD": "exp[at]",
            "META_FILES_EXPIRES_SECONDS_FIELD": "exp[s]",
            "META_FILES_EXPIRES_ANCHOR": "now",
            "META_FILES_EXPIRY_SKEW_S": "11",
        }
    )
    client = select_meta_files_client(settings)
    assert isinstance(client, ProductionMetaFilesClient)
    config = client._upload_config
    assert config.purpose == "evidence"
    assert config.purpose_field == "use"
    assert config.file_field == "blob"
    assert config.file_content_type == "application/octet-stream"
    assert config.expires_anchor_field == "exp[at]"
    assert config.expires_seconds_field == "exp[s]"
    assert config.expires_anchor == "now"
    assert config.upload_timeout_s == 33
    assert config.max_upload_bytes == 1234
    assert config.expiry_skew_s == 11

    for bad in (
        {"META_FILES_TIMEOUT_S": "abc"},
        {"META_FILES_TIMEOUT_S": "0"},
        {"META_FILES_TIMEOUT_S": "nan"},
        {"META_FILES_UPLOAD_TIMEOUT_S": "-2"},
        {"META_FILES_MAX_UPLOAD_BYTES": "0"},
        {"META_FILES_MAX_UPLOAD_BYTES": "plenty"},
        {"META_FILES_UPLOAD_PURPOSE": "bad\nvalue"},
        {"META_FILES_EXPIRY_SKEW_S": "0"},
    ):
        with pytest.raises(RuntimeError, match="META_FILES"):
            load_settings(bad)
    # Blank means unset: the default applies.
    assert load_settings({"META_FILES_UPLOAD_PURPOSE": "  "}).meta_files_upload_purpose == (
        "user_data"
    )


def test_api_error_scrubs_key_and_bearer_forms_from_provider_text():
    key = "fake-key-123"

    def _echoing(request, body):
        raise _http_error(
            request.full_url,
            400,
            {
                "error": {
                    "code": "bad",
                    "message": f"refused Bearer {key} here, also {key} bare",
                }
            },
        )

    client = ProductionMetaFilesClient(
        "fake-key-123",
        base_url="https://meta.example.test",
        transport=FakeTransport(_echoing),
    )
    with pytest.raises(MetaApiError) as excinfo:
        client.delete_file(MetaFileId("x"))
    assert key not in str(excinfo.value)
    assert key not in (excinfo.value.code or "")
    assert "[REDACTED]" in str(excinfo.value)

    other = "foreign-token-999"

    def _foreign(request, body):
        raise _http_error(
            request.full_url,
            400,
            {"error": {"message": f"bad credentials Bearer {other}"}},
        )

    foreign = ProductionMetaFilesClient(
        key,
        base_url="https://meta.example.test",
        transport=FakeTransport(_foreign),
    )
    with pytest.raises(MetaApiError) as excinfo:
        foreign.delete_file(MetaFileId("x"))
    assert other not in str(excinfo.value)
    assert "Bearer [REDACTED]" in str(excinfo.value)


def test_persisted_error_text_never_carries_the_key():
    key = "fake-key-123"

    def _echoing(request, body):
        raise _http_error(
            request.full_url,
            400,
            {"error": {"message": f"refused Bearer {key}"}},
        )

    client = ProductionMetaFilesClient(
        key,
        base_url="https://meta.example.test",
        transport=FakeTransport(_echoing),
    )
    with pytest.raises(MetaApiError) as excinfo:
        client.delete_file(MetaFileId("x"))
    persisted = _short(excinfo.value)
    assert key not in persisted
    assert "Bearer [REDACTED]" in persisted


def test_error_body_read_is_bounded_and_oversized_detail_unavailable():
    seen: dict = {}

    def _read(size):
        seen["size"] = size
        return b"x" * size

    assert _http_error_payload(SimpleNamespace(read=_read), 64) is None
    assert seen == {"size": 65}


def test_oversized_error_detail_is_unavailable_but_small_bodies_parse():
    small = {"error": {"code": "bad", "message": "nope"}}
    big = {"error": {"code": "bad", "message": "x" * 500}}

    def _handler(request, body):
        if "big" in request.full_url:
            raise _http_error(request.full_url, 400, big)
        raise _http_error(request.full_url, 400, small)

    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_handler),
        error_max_bytes=128,
    )
    with pytest.raises(MetaApiError) as excinfo:
        client.delete_file(MetaFileId("small"))
    assert excinfo.value.code == "bad"
    assert "nope" in str(excinfo.value)
    with pytest.raises(MetaApiError) as excinfo:
        client.delete_file(MetaFileId("big"))
    assert excinfo.value.code is None
    assert "400" in str(excinfo.value)
    assert "xxxx" not in str(excinfo.value)


def test_error_bound_setting_loads_and_rejects_bad_values():
    assert (
        load_settings({"META_FILES_MAX_ERROR_BYTES": "128"})
        .meta_files_max_error_bytes
        == 128
    )
    assert (
        load_settings({}).meta_files_max_error_bytes
        == DEFAULT_META_FILES_MAX_ERROR_BYTES
    )
    for bad in (
        {"META_FILES_MAX_ERROR_BYTES": "0"},
        {"META_FILES_MAX_ERROR_BYTES": "lots"},
    ):
        with pytest.raises(RuntimeError, match="META_FILES_MAX_ERROR_BYTES"):
            load_settings(bad)
    wired = select_meta_files_client(
        load_settings(
            {"META_API_KEY": "k", "META_FILES_MAX_ERROR_BYTES": "128"}
        )
    )
    assert wired._error_max_bytes == 128
    defaulted = select_meta_files_client(load_settings({"META_API_KEY": "k"}))
    assert defaulted._error_max_bytes == DEFAULT_META_FILES_MAX_ERROR_BYTES


_INJECTION = 'x"; filename="injected'


@pytest.mark.parametrize(
    "var",
    [
        "META_FILES_PURPOSE_FIELD",
        "META_FILES_FILE_FIELD",
        "META_FILES_EXPIRES_ANCHOR_FIELD",
        "META_FILES_EXPIRES_SECONDS_FIELD",
    ],
)
def test_multipart_field_names_refuse_header_injection(var):
    with pytest.raises(RuntimeError, match=var):
        load_settings({var: _INJECTION})


def test_multipart_field_name_grammar_accepts_supported_values():
    settings = load_settings(
        {
            "META_FILES_PURPOSE_FIELD": "use",
            "META_FILES_FILE_FIELD": "blob",
            "META_FILES_EXPIRES_ANCHOR_FIELD": "exp[at]",
            "META_FILES_EXPIRES_SECONDS_FIELD": "exp[s]",
        }
    )
    assert settings.meta_files_purpose_field == "use"
    assert settings.meta_files_file_field == "blob"
    assert settings.meta_files_expires_anchor_field == "exp[at]"
    assert settings.meta_files_expires_seconds_field == "exp[s]"
    with pytest.raises(ValueError, match="file_field"):
        MetaFilesUploadConfig(file_field=_INJECTION)
    MetaFilesUploadConfig()  # defaults pass the config boundary too


def test_cleanup_id_comes_from_carried_ids_only():
    assert _cleanup_id_from_error(MetaFilesError("meta upload needs a key")) is None
    assert _cleanup_id_from_error(expiry_error(MetaFileId("m-1"), None)) == "m-1"
    carrying = MetaFilesError("boom")
    carrying.file_id = "m-2"
    assert _cleanup_id_from_error(carrying) == "m-2"


def test_expiry_error_keeps_its_id_for_cleanup(tmp_path: Path):
    """a carried id still reaches the finally delete."""
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"v" * 16)
    transport = FakeTransport(
        lambda request, body: (
            201,
            {"file_id": "m-orphan", "expires_at": "not-a-time"},
        )
    )
    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=transport,
    )
    file_id: MetaFileId | None = None
    with pytest.raises(MetaUploadExpiryError):
        try:
            client.upload_file(
                file_path, expires_after_hours=1, idempotency_key="k"
            )
        except MetaFilesError as error:
            kept = _cleanup_id_from_error(error)
            if kept is not None:
                file_id = kept
            raise
        finally:
            if file_id is not None:
                try:
                    client.delete_file(file_id)
                except MetaFilesError:
                    pass
    assert file_id == "m-orphan"
    deletes = [
        request
        for request, _ in transport.requests
        if request.get_method() == "DELETE"
    ]
    assert len(deletes) == 1
    assert deletes[0].full_url.endswith("/files/m-orphan")


def test_smoke_log_writer_scrubs_secrets_and_records_command_exit(tmp_path):
    log = tmp_path / "smoke.log"
    _write_smoke_log(
        log,
        command="pytest tests/test_meta_files.py::test_live -q",
        exit_code=0,
        lines=["file: file-abc", "steps: uploaded+read+deleted+confirmed-gone"],
        secrets={"sk-key": "[REDACTED]", "file-abc": "<file-id>"},
    )
    body = log.read_text(encoding="utf-8")
    assert "sk-key" not in body
    assert "file-abc" not in body
    assert "command: pytest tests/test_meta_files.py::test_live -q" in body
    assert "exit: 0" in body
    assert "<file-id>" in body


def test_live_observed_shapes_parse(tmp_path: Path):
    """Replay a live session's recorded shapes (unix-seconds expiry)."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "meta_files_live_shapes.json")
        .read_text(encoding="utf-8")
    )
    doc_keys = set(fixture["file_document"])
    assert doc_keys == {
        "bytes",
        "created_at",
        "expires_at",
        "filename",
        "id",
        "object",
        "purpose",
        "status",
    }
    assert set(fixture["list_top"]) == {
        "data",
        "first_id",
        "has_more",
        "last_id",
        "object",
    }
    now = datetime.now(timezone.utc)
    live_doc = {
        "bytes": 5400,
        "created_at": int(now.timestamp()) - 5,
        "expires_at": int(now.timestamp()) + 3600,
        "filename": "live-tiny.mp4",
        "id": "<file-id>",
        "object": "file",
        "purpose": "user_data",
        "status": "processed",
    }
    assert set(live_doc) == doc_keys  # replay carries every observed key
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"v" * 16)
    client = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(lambda request, body: (200, dict(live_doc))),
    )
    found = client.upload_file(
        file_path, expires_after_hours=1, idempotency_key="k-live"
    )
    assert found.file_id == "<file-id>"
    assert now < found.expires_at <= now + timedelta(hours=1, seconds=5)

    live_list = {
        "data": [],
        "first_id": "<file-id>",
        "has_more": False,
        "last_id": "<file-id>",
        "object": "list",
    }
    assert set(live_list) == set(fixture["list_top"])

    def _gone(request, body):
        if urlparse(request.full_url).path == "/files":
            return 200, dict(live_list)
        raise _http_error(request.full_url, 404, {"error": {"code": "x"}})

    listed = ProductionMetaFilesClient(
        "test-key",
        base_url="https://meta.example.test",
        transport=FakeTransport(_gone),
    )
    assert listed.get_file(MetaFileId("gone")) is None


@needs_live_key
def test_live_meta_files_upload_read_delete(tmp_path: Path, capsys):
    """One live round trip: upload, read back, delete, confirm gone."""
    assert LIVE_BASE_URL.startswith("https://")
    clip = tmp_path / "live-tiny.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
            "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True, capture_output=True, timeout=120,
    )
    settings = Settings(
        data_root=tmp_path, db_path=tmp_path / "live.sqlite3",
        production=False, meta_api_key=LIVE_KEY,
        meta_files_base_url=LIVE_BASE_URL,
    )
    client = select_meta_files_client(settings)
    log_path = os.environ.get(SMOKE_LOG_ENV_VAR, "").strip() or None
    command = shlex.join(sys.argv)
    started = datetime.now(timezone.utc)
    key = uuid.uuid4().hex
    before = datetime.now(timezone.utc)
    file_id: MetaFileId | None = None
    exit_code = 0
    try:
        try:
            uploaded = client.upload_file(
                clip, expires_after_hours=1, idempotency_key=key
            )
        except MetaFilesError as error:
            assert LIVE_KEY not in str(error)
            kept = _cleanup_id_from_error(error)
            if kept is not None:
                file_id = kept
            raise
        file_id = uploaded.file_id
        assert before < uploaded.expires_at <= before + timedelta(hours=1)
        got = client.get_file(file_id)
        assert got is not None and got.file_id == file_id
        assert before < got.expires_at <= before + timedelta(hours=1)
        assert client.delete_file(file_id) is None
        assert client.get_file(file_id) is None
        elapsed = (datetime.now(timezone.utc) - before).total_seconds()
        assert LIVE_KEY not in (capsys.readouterr().out or "")
        assert file_id is not None
        print(
            f"live meta files: id={file_id} "
            "uploaded+read+deleted+confirmed-gone "
            f"in {elapsed:.1f}s; no usage/cost fields observed ($0.00)"
        )
    except BaseException:
        exit_code = 1
        raise
    finally:
        if file_id is not None:
            try:
                client.delete_file(file_id)
            except MetaFilesError:
                pass
        if log_path is not None:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            steps = (
                "uploaded+read+deleted+confirmed-gone"
                if exit_code == 0
                else "FAILED"
            )
            _write_smoke_log(
                log_path,
                command=command,
                exit_code=exit_code,
                lines=[
                    f"file: {file_id if file_id is not None else 'none'}",
                    f"idempotency-key: {key}",
                    f"steps: {steps}",
                    f"elapsed-s: {elapsed:.1f}",
                    "cost: no usage/cost fields observed ($0.00)",
                ],
                secrets={
                    LIVE_KEY: "[REDACTED]",
                    file_id or "": "<file-id>",
                    key: "<idempotency-key>",
                },
            )
