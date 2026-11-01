"""Meta files surface. The upload module owns the method surface; tests inject fakes.

The four-method :class:`contracts.MetaFilesClient` protocol is the only
client surface upload callers and fakes use. Production reads ``META_API_KEY``
from the process environment only (via the settings); the key is never
persisted, never logged, and never appears in an exception message.
Uploads reach the real API as streaming multipart/form-data; an HTTP
error answer surfaces as :class:`MetaApiError` with status and scrubbed
provider detail, while a transport failure stays a plain
:class:`MetaFilesError`.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app import db
from app.contracts import (
    META_UPLOAD_MAX_HOURS,
    MetaFileId,
    MetaFilesClient,
    MetaFilesError,
    MetaReadUnsupported,
    MetaUpload,
    MetaUploadExpiryError,
)
from app.settings import (
    DEFAULT_META_FILES_EXPIRY_SKEW_S,
    DEFAULT_META_FILES_EXPIRES_ANCHOR,
    DEFAULT_META_FILES_EXPIRES_ANCHOR_FIELD,
    DEFAULT_META_FILES_EXPIRES_SECONDS_FIELD,
    DEFAULT_META_FILES_FILE_CONTENT_TYPE,
    DEFAULT_META_FILES_FILE_FIELD,
    DEFAULT_META_FILES_MAX_ERROR_BYTES,
    DEFAULT_META_FILES_MAX_UPLOAD_BYTES,
    DEFAULT_META_FILES_PURPOSE_FIELD,
    DEFAULT_META_FILES_TIMEOUT_S,
    DEFAULT_META_FILES_UPLOAD_PURPOSE,
    DEFAULT_META_FILES_UPLOAD_TIMEOUT_S,
    is_multipart_field_name,
)

META_RECIPIENT = "Meta"
LOCAL_RETENTION_HOURS = 24

# Deployment configuration: the integration code points this at the real
# Meta files endpoint. The placeholder never resolves, so until then every
# production call fails closed as a retryable MetaFilesError.
DEFAULT_META_FILES_BASE_URL = "https://files.meta.example/v1"
HTTP_TIMEOUT_S = DEFAULT_META_FILES_TIMEOUT_S
MAX_LIST_PAGES = 100

VERIFIED_ABSENT_COPY = (
    "Deleted at Meta \u2014 confirmed absent via the API "
    "(GET and list show no such file). "
    "This confirms API-level absence, not physical erasure."
)
ACKNOWLEDGED_ONLY_COPY = (
    "Deletion requested and acknowledged; Meta\u2019s copy also expires "
    "within 24 hours. We could not confirm absence via the API."
)
DELETION_FAILURE_COPY = (
    "We could not delete your video. Your guide is hidden until deletion succeeds."
)
UPLOAD_BLOCKED_COPY = (
    "Uploads are paused while we fix deletion. You can still try the sample guide."
)
DELETE_REQUEST_ACK_COPY = (
    "Request received. We will reply at the contact you gave. "
    "This is a request inbox, not a compliance claim."
)
DELETE_REQUEST_FULL_COPY = (
    "Our request inbox is full right now. Please try again later."
)
DELETE_REQUEST_THROTTLED_COPY = (
    "Too many delete requests from this device. Please try again tomorrow."
)
META_UNAVAILABLE_COPY = (
    "Uploads are unavailable right now. Your preview is kept; please try again later."
)


def expiry_error(
    file_id: MetaFileId, expires_at: datetime | None
) -> MetaUploadExpiryError:
    """Build the id-carrying expiry error with its attributes attached.

    The frozen mirror declares ``__init__`` but stores nothing, so the
    attributes are set explicitly here; the ``(file_id, expires_at)``
    constructor args always carry the same pair as a fallback.
    """
    error = MetaUploadExpiryError(file_id, expires_at)
    error.file_id = file_id
    error.expires_at = expires_at
    return error


def expiry_details(error: MetaUploadExpiryError) -> tuple[MetaFileId | None, Any]:
    """Read ``(file_id, expires_at)`` off an expiry error, args as fallback."""
    file_id = getattr(error, "file_id", None)
    expires_at = getattr(error, "expires_at", None)
    args = getattr(error, "args", ())
    if (not isinstance(file_id, str) or not file_id) and len(args) >= 1:
        file_id = args[0]
    if expires_at is None and len(args) >= 2:
        expires_at = args[1]
    if not isinstance(file_id, str) or not file_id:
        return None, expires_at
    return MetaFileId(file_id), expires_at


class MetaClientUnavailableError(MetaFilesError):
    """No authenticated Meta client exists (missing key/config). Retryable."""


class MetaApiError(MetaFilesError):
    """The Meta API answered with an HTTP error status. Not a transport failure.

    Carries the numeric ``status`` plus the provider ``code`` (if any), so a
    refused request (an answer with a status and usually a body) is
    distinguishable from a request that never got an answer (a transport
    failure, which stays a plain :class:`MetaFilesError` with no status).
    The message never carries the API key.
    """

    def __init__(
        self, message: str, *, status: int, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class MetaFilesUploadConfig:
    """Multipart upload knobs. Every value comes from the settings.

    Field names, the purpose value, the expiry anchor, the upload
    timeout, and the client-side size bound all live in
    :mod:`app.settings`, so an API change is a config change, never a
    code change. ``from_settings`` builds this; tests may override it.
    """

    purpose_field: str = DEFAULT_META_FILES_PURPOSE_FIELD
    purpose: str = DEFAULT_META_FILES_UPLOAD_PURPOSE
    file_field: str = DEFAULT_META_FILES_FILE_FIELD
    file_content_type: str = DEFAULT_META_FILES_FILE_CONTENT_TYPE
    expires_anchor_field: str = DEFAULT_META_FILES_EXPIRES_ANCHOR_FIELD
    expires_seconds_field: str = DEFAULT_META_FILES_EXPIRES_SECONDS_FIELD
    expires_anchor: str = DEFAULT_META_FILES_EXPIRES_ANCHOR
    upload_timeout_s: float = DEFAULT_META_FILES_UPLOAD_TIMEOUT_S
    max_upload_bytes: int = DEFAULT_META_FILES_MAX_UPLOAD_BYTES
    expiry_skew_s: float = DEFAULT_META_FILES_EXPIRY_SKEW_S

    def __post_init__(self) -> None:
        for name in (
            "purpose_field",
            "purpose",
            "file_field",
            "file_content_type",
            "expires_anchor_field",
            "expires_seconds_field",
            "expires_anchor",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"meta files upload config needs a non-empty {name}"
                )
            if "\r" in value or "\n" in value:
                raise ValueError(
                    f"meta files upload config {name} must not contain a newline"
                )
            if name in (
                "purpose_field",
                "file_field",
                "expires_anchor_field",
                "expires_seconds_field",
            ) and not is_multipart_field_name(value):
                raise ValueError(
                    f"meta files upload config {name} must use letters, "
                    "digits, '_' or '[...]'"
                )
        if (
            isinstance(self.upload_timeout_s, bool)
            or not isinstance(self.upload_timeout_s, (int, float))
            or not math.isfinite(self.upload_timeout_s)
            or self.upload_timeout_s <= 0
        ):
            raise ValueError(
                "meta files upload config needs a positive finite upload_timeout_s"
            )
        if (
            isinstance(self.expiry_skew_s, bool)
            or not isinstance(self.expiry_skew_s, (int, float))
            or not math.isfinite(self.expiry_skew_s)
            or self.expiry_skew_s <= 0
        ):
            raise ValueError(
                "meta files upload config needs a positive finite expiry_skew_s"
            )
        if (
            isinstance(self.max_upload_bytes, bool)
            or not isinstance(self.max_upload_bytes, int)
            or self.max_upload_bytes <= 0
        ):
            raise ValueError(
                "meta files upload config needs a positive max_upload_bytes"
            )


class MissingMetaClient:
    """Protocol-shaped client that always fails closed. Never reports absence."""

    def _fail(self) -> MetaFilesError:
        return MetaClientUnavailableError("meta files client is unavailable")

    def upload_file(
        self, path: Path, *, expires_after_hours: int, idempotency_key: str
    ) -> MetaUpload:
        raise self._fail()

    def find_upload(self, idempotency_key: str) -> MetaUpload | None:
        raise self._fail()

    def delete_file(self, file_id: MetaFileId) -> None:
        raise self._fail()

    def get_file(self, file_id: MetaFileId) -> MetaUpload | None:
        raise self._fail()


assert isinstance(MissingMetaClient(), MetaFilesClient)


def _parse_expiry(value: Any) -> datetime | None:
    """Return the aware UTC expiry for a provider value, else ``None``.

    ``None`` covers missing, wrong-typed, unparseable, and naive values —
    the caller then raises the id-carrying expiry error or a plain
    :class:`MetaFilesError`, never a partial result. ISO-8601 strings and
    unix timestamps (numbers or numeric strings) both parse.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    try:
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_upload(payload: Any) -> MetaUpload | None:
    """Return the exact ``MetaUpload`` for a provider payload, else ``None``.

    A document carrying any no-match marker is contradictory, not a
    match: it parses as ``None`` so the caller raises and stays
    retryable instead of trusting half the document.
    """
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("found") is False
        or ("match" in payload and payload["match"] is None)
        or ("file" in payload and payload["file"] is None)
        or ("files" in payload and payload["files"] == [])
    ):
        return None
    file_id = payload.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        file_id = payload.get("id")  # real API shape: {"id": ...}
    if not isinstance(file_id, str) or not file_id.strip():
        return None
    expires_at = _parse_expiry(payload.get("expires_at"))
    if expires_at is None:
        return None
    return MetaUpload(file_id=MetaFileId(file_id), expires_at=expires_at)


@dataclass(frozen=True)
class _Response:
    status: int
    payload: Any


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: the Authorization header never leaves home.

    A provider open redirect must not carry ``Bearer <key>`` to another
    origin, so no 3xx is followed at all — the caller sees a retryable
    :class:`MetaFilesError` instead of a cross-origin response.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise MetaFilesError(f"meta files refused redirect {code}")


def _is_explicit_no_match(payload: Any) -> bool:
    """True only for one of the four exact singleton no-match documents.

    ``404`` plus exactly ``{"found": false}``, ``{"match": null}``,
    ``{"file": null}``, or ``{"files": []}`` are the only absence
    proofs. Any error field, any unexpected field, any present match
    field (even a null, empty, or non-string id), and any marker
    combination stays retryable instead of discarding the only
    provider key on a contradictory document.
    """
    if not isinstance(payload, dict):
        return False
    keys = set(payload.keys())
    if keys == {"found"}:
        return payload["found"] is False
    if keys == {"match"}:
        return payload["match"] is None
    if keys == {"file"}:
        return payload["file"] is None
    if keys == {"files"}:
        return isinstance(payload["files"], list) and payload["files"] == []
    return False


def _http_error_payload(error: urllib.error.HTTPError, max_bytes: int) -> Any:
    """Best-effort parsed body of an HTTP error; ``None`` when unreadable.

    Reads at most ``max_bytes``: an oversized body yields ``None``
    (detail unavailable) instead of allocating an unbounded refusal.
    """
    try:
        raw = error.read(max_bytes + 1)
    except Exception:
        return None
    if not isinstance(raw, (bytes, bytearray)) or not bytes(raw).strip():
        return None
    if len(raw) > max_bytes:
        return None
    try:
        return json.loads(bytes(raw).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _error_details(payload: Any) -> tuple[str | None, str | None]:
    """Return the ``(code, message)`` of a provider error document, if any."""
    if not isinstance(payload, dict):
        return None, None
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        return (
            code if isinstance(code, str) and code.strip() else None,
            message if isinstance(message, str) and message.strip() else None,
        )
    if isinstance(error, str) and error.strip():
        return None, error.strip()
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return None, message.strip()
    return None, None


_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/=]+", re.IGNORECASE)


def _scrub_secret(text: str, secret: str | None) -> str:
    """Remove credential forms from provider-controlled text.

    The exact key goes first (a provider echo of ``Bearer <key>`` then
    reads ``Bearer [REDACTED]``), then any remaining ``Bearer <token>``
    form, so neither the key nor another token survives into an
    exception message or a persisted error string.
    """
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return _BEARER_RE.sub("Bearer [REDACTED]", text)


def _api_error(
    action: str, reply: _Response, secret: str | None
) -> MetaApiError:
    """Build the clear API error for an unexpected HTTP error status.

    Provider code and message are scrubbed of the key and any Bearer
    token first, so the exception (and anything persisting its text)
    never carries a credential.
    """
    code, message = _error_details(reply.payload)
    if code is not None:
        code = _scrub_secret(code, secret)
    text = f"meta {action} refused with status {reply.status}"
    if code is not None:
        text += f" ({code})"
    if message is not None:
        text += f": {_scrub_secret(message, secret)}"
    return MetaApiError(text, status=reply.status, code=code)


def _setting_text(settings: object, name: str, default: str) -> str:
    """Read a non-empty text knob off settings, defaulting when absent."""
    value = getattr(settings, name, default)
    if not isinstance(value, str) or not value.strip():
        return default
    return value


def _multipart_preamble(
    *,
    boundary: str,
    config: MetaFilesUploadConfig,
    filename: str,
    expires_after_hours: int,
) -> tuple[bytes, bytes]:
    """Return the ``(preamble, epilogue)`` around the streamed file bytes."""
    safe_name = "".join("_" if char in '\\"\r\n' else char for char in filename)

    def _field(name: str, value: str) -> bytes:
        return (
            "--"
            + boundary
            + '\r\nContent-Disposition: form-data; name="'
            + name
            + '"\r\n\r\n'
            + value
            + "\r\n"
        ).encode("utf-8")

    preamble = (
        _field(config.purpose_field, config.purpose)
        + _field(config.expires_anchor_field, config.expires_anchor)
        + _field(
            config.expires_seconds_field, str(expires_after_hours * 3600)
        )
        + (
            "--"
            + boundary
            + '\r\nContent-Disposition: form-data; name="'
            + config.file_field
            + '"; filename="'
            + safe_name
            + '"\r\nContent-Type: '
            + config.file_content_type
            + "\r\n\r\n"
        ).encode("utf-8")
    )
    return preamble, ("\r\n--" + boundary + "--\r\n").encode("utf-8")


class _MultipartUploadBody:
    """File-like multipart/form-data reader: preamble + file bytes + tail.

    ``read(n)`` serves the in-memory preamble, then the open file in
    bounded chunks, then the closing boundary, so an upload never holds
    the file in memory. ``len()`` is the exact Content-Length, computed
    from the stat size without reading the file. One upload each; the
    caller closes it. A file that shrank since the stat raises instead of
    sending a truncated body under a longer Content-Length.
    """

    _DRAIN_BLOCK = 65536

    def __init__(
        self,
        *,
        preamble: bytes,
        path: Path,
        file_size: int,
        epilogue: bytes,
    ) -> None:
        self._preamble = preamble
        self._preamble_offset = 0
        self._file = path.open("rb")
        self._file_remaining = file_size
        self._epilogue = epilogue
        self._epilogue_offset = 0
        self._length = len(preamble) + file_size + len(epilogue)
        self._closed = False

    def __len__(self) -> int:
        return self._length

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("read of closed multipart body")
        if size is None or size < 0:
            out = bytearray()
            while True:
                chunk = self.read(self._DRAIN_BLOCK)
                if not chunk:
                    return bytes(out)
                out += chunk
        if size == 0:
            return b""
        out = bytearray()
        while len(out) < size:
            need = size - len(out)
            if self._preamble_offset < len(self._preamble):
                take = min(need, len(self._preamble) - self._preamble_offset)
                out += self._preamble[
                    self._preamble_offset : self._preamble_offset + take
                ]
                self._preamble_offset += take
            elif self._file_remaining > 0:
                chunk = self._file.read(min(need, self._file_remaining))
                if not chunk:
                    raise MetaFilesError("meta upload hit a short local file")
                self._file_remaining -= len(chunk)
                out += chunk
            elif self._epilogue_offset < len(self._epilogue):
                take = min(need, len(self._epilogue) - self._epilogue_offset)
                out += self._epilogue[
                    self._epilogue_offset : self._epilogue_offset + take
                ]
                self._epilogue_offset += take
            else:
                break
        return bytes(out)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._file.close()


class ProductionMetaFilesClient:
    """Key-safe HTTP adapter over the injectable four-method protocol.

    Implements the exact ``MetaFilesClient`` absence/error behavior: an
    invalid upload expiry with a parseable id raises
    :class:`MetaUploadExpiryError` so the upload module can queue that id for deletion,
    delete treats an absent id as idempotent success, definitive absence
    returns ``None``, unsupported reads raise :class:`MetaReadUnsupported`,
    and every other failure raises :class:`MetaFilesError` and stays
    retryable. The API key travels only in the Authorization header.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_META_FILES_BASE_URL,
        timeout_s: float = HTTP_TIMEOUT_S,
        upload_config: MetaFilesUploadConfig | None = None,
        transport: Callable[..., Any] | None = None,
        error_max_bytes: int = DEFAULT_META_FILES_MAX_ERROR_BYTES,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise MetaClientUnavailableError(
                "META_API_KEY is required for the production client"
            )
        if (
            isinstance(error_max_bytes, bool)
            or not isinstance(error_max_bytes, int)
            or error_max_bytes <= 0
        ):
            raise ValueError(
                "production meta files client needs a positive error_max_bytes"
            )
        self._api_key = api_key
        self._error_max_bytes = error_max_bytes
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._upload_config = (
            upload_config if upload_config is not None else MetaFilesUploadConfig()
        )
        self._opener = urllib.request.build_opener(_NoRedirectHandler)
        self._transport = transport if transport is not None else self._opener.open

    @classmethod
    def from_settings(
        cls,
        settings: object,
        *,
        base_url: str = DEFAULT_META_FILES_BASE_URL,
        transport: Callable[..., Any] | None = None,
    ) -> ProductionMetaFilesClient:
        """Build the production client from the settings, or fail closed.

        The endpoint comes from settings (``META_FILES_BASE_URL``) when
        configured; the argument stays for tests pointing at a stub.
        Timeouts and every multipart knob come from settings too. An
        injected opener-shaped ``transport`` replaces urllib (tests);
        production always passes none.
        """
        key = getattr(settings, "meta_api_key", None)
        if not isinstance(key, str) or not key.strip():
            raise MetaClientUnavailableError(
                "META_API_KEY is required for the production client"
            )
        endpoint = getattr(settings, "meta_files_base_url", None)
        if isinstance(endpoint, str) and endpoint.strip():
            base_url = endpoint
        timeout = getattr(settings, "meta_files_timeout_s", HTTP_TIMEOUT_S)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            timeout = HTTP_TIMEOUT_S
        error_max_bytes = getattr(
            settings,
            "meta_files_max_error_bytes",
            DEFAULT_META_FILES_MAX_ERROR_BYTES,
        )
        if (
            isinstance(error_max_bytes, bool)
            or not isinstance(error_max_bytes, int)
            or error_max_bytes <= 0
        ):
            error_max_bytes = DEFAULT_META_FILES_MAX_ERROR_BYTES
        try:
            config = MetaFilesUploadConfig(
                purpose_field=_setting_text(
                    settings,
                    "meta_files_purpose_field",
                    DEFAULT_META_FILES_PURPOSE_FIELD,
                ),
                purpose=_setting_text(
                    settings,
                    "meta_files_upload_purpose",
                    DEFAULT_META_FILES_UPLOAD_PURPOSE,
                ),
                file_field=_setting_text(
                    settings,
                    "meta_files_file_field",
                    DEFAULT_META_FILES_FILE_FIELD,
                ),
                file_content_type=_setting_text(
                    settings,
                    "meta_files_file_content_type",
                    DEFAULT_META_FILES_FILE_CONTENT_TYPE,
                ),
                expires_anchor_field=_setting_text(
                    settings,
                    "meta_files_expires_anchor_field",
                    DEFAULT_META_FILES_EXPIRES_ANCHOR_FIELD,
                ),
                expires_seconds_field=_setting_text(
                    settings,
                    "meta_files_expires_seconds_field",
                    DEFAULT_META_FILES_EXPIRES_SECONDS_FIELD,
                ),
                expires_anchor=_setting_text(
                    settings,
                    "meta_files_expires_anchor",
                    DEFAULT_META_FILES_EXPIRES_ANCHOR,
                ),
                upload_timeout_s=getattr(
                    settings,
                    "meta_files_upload_timeout_s",
                    DEFAULT_META_FILES_UPLOAD_TIMEOUT_S,
                ),
                max_upload_bytes=getattr(
                    settings,
                    "meta_files_max_upload_bytes",
                    DEFAULT_META_FILES_MAX_UPLOAD_BYTES,
                ),
                expiry_skew_s=getattr(
                    settings,
                    "meta_files_expiry_skew_s",
                    DEFAULT_META_FILES_EXPIRY_SKEW_S,
                ),
            )
        except ValueError as error:
            raise MetaClientUnavailableError(
                "meta files upload settings are invalid"
            ) from error
        return cls(
            key,
            base_url=base_url,
            timeout_s=timeout,
            upload_config=config,
            transport=transport,
            error_max_bytes=error_max_bytes,
        )

    def __repr__(self) -> str:  # never leak the key
        return f"{type(self).__name__}(base_url={self._base_url!r})"

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | _MultipartUploadBody | None = None,
        content_type: str | None = None,
        timeout_s: float | None = None,
    ) -> _Response:
        """One HTTP round trip. Transport failures raise ``MetaFilesError``.

        Redirects are never followed (see :class:`_NoRedirectHandler`),
        and any 3xx that still surfaces is refused rather than trusted.
        HTTP error answers keep their parsed body, so callers raise a
        clear :class:`MetaApiError` instead of a bare status.
        """
        headers = {"Authorization": "Bearer " + self._api_key}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if body is not None and not isinstance(body, (bytes, bytearray)):
            # Streaming body: the exact length up front, so neither urllib
            # nor http.client falls back to chunked encoding.
            headers["Content-Length"] = str(len(body))
        request = urllib.request.Request(
            self._base_url + path,
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with self._transport(
                request,
                timeout=timeout_s
                if timeout_s is not None
                else self._timeout_s,
            ) as reply:
                raw = reply.read()
                status = int(getattr(reply, "status", 200) or 200)
        except urllib.error.HTTPError as error:
            try:
                payload = _http_error_payload(error, self._error_max_bytes)
            finally:
                # The failed open never entered the context manager, so the
                # error response still owns its body; release it here.
                try:
                    error.close()
                except Exception:
                    pass
            return _Response(status=int(error.code), payload=payload)
        except MetaFilesError:
            raise
        except Exception as error:
            raise MetaFilesError(f"meta files {method} failed") from error
        if 300 <= status < 400:
            raise MetaFilesError(f"meta files {method} refused redirect {status}")
        if not raw.strip():
            return _Response(status=status, payload=None)
        try:
            return _Response(status=status, payload=json.loads(raw.decode("utf-8")))
        except (ValueError, UnicodeDecodeError) as error:
            raise MetaFilesError(f"meta files {method} returned bad JSON") from error

    def _idempotency_key(self, idempotency_key: str) -> str:
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise MetaFilesError("meta upload needs a non-empty idempotency key")
        return idempotency_key

    def upload_file(
        self, path: Path, *, expires_after_hours: int, idempotency_key: str
    ) -> MetaUpload:
        """Upload one file as streamed multipart; enforce the expiry bounds."""
        if expires_after_hours < 1 or expires_after_hours > META_UPLOAD_MAX_HOURS:
            raise MetaFilesError(
                f"expires_after_hours must be within 1..{META_UPLOAD_MAX_HOURS}"
            )
        key = self._idempotency_key(idempotency_key)
        config = self._upload_config
        try:
            file_size = path.stat().st_size
            is_file = path.is_file()
        except OSError as error:
            raise MetaFilesError("meta upload cannot read the local file") from error
        if not is_file:
            raise MetaFilesError("meta upload cannot read the local file")
        if file_size > config.max_upload_bytes:
            raise MetaFilesError(
                f"meta upload is {file_size} bytes, "
                f"over the {config.max_upload_bytes} byte limit"
            )
        boundary = uuid.uuid4().hex
        preamble, epilogue = _multipart_preamble(
            boundary=boundary,
            config=config,
            filename=path.name,
            expires_after_hours=expires_after_hours,
        )
        try:
            body = _MultipartUploadBody(
                preamble=preamble,
                path=path,
                file_size=file_size,
                epilogue=epilogue,
            )
        except OSError as error:
            raise MetaFilesError("meta upload cannot read the local file") from error
        request_time_utc = db.utcnow()
        try:
            reply = self._request(
                "POST",
                "/files?expires_after_hours="
                + str(expires_after_hours)
                + "&idempotency_key="
                + urllib.parse.quote(key, safe=""),
                body=body,
                content_type="multipart/form-data; boundary=" + boundary,
                timeout_s=config.upload_timeout_s,
            )
        finally:
            body.close()
        now = db.utcnow()
        if reply.status not in (200, 201):
            raise _api_error("upload", reply, self._api_key)
        payload = reply.payload if isinstance(reply.payload, dict) else {}
        parsed = _parse_upload(payload)
        # The grant must outlive both clock reads but not the request by
        # more than clock skew: the upper prong anchors on the post-send
        # read, since the server grants after the bytes arrive, not
        # before they leave. Hours-long over-grants still fail.
        if parsed is not None and (
            request_time_utc < parsed.expires_at
            and now < parsed.expires_at
            and parsed.expires_at
            <= now
            + timedelta(hours=expires_after_hours, seconds=config.expiry_skew_s)
        ):
            return parsed
        # Invalid expiry with a parseable id: the id must be queued for
        # deletion, so carry it in the expiry error, never a partial result.
        file_id = payload.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            file_id = payload.get("id")  # real API shape: {"id": ...}
        if isinstance(file_id, str) and file_id.strip():
            raise expiry_error(
                MetaFileId(file_id), _parse_expiry(payload.get("expires_at"))
            )
        raise MetaFilesError("meta upload returned no parseable file id")

    def find_upload(self, idempotency_key: str) -> MetaUpload | None:
        """Return the one exact key match, ``None`` on definitive no-match.

        Only a 404 or an explicit no-match document proves absence; an
        empty or malformed 200 stays retryable so recovery never
        discards the only provider key on a read glitch.
        """
        key = self._idempotency_key(idempotency_key)
        reply = self._request(
            "GET", "/files?key=" + urllib.parse.quote(key, safe="")
        )
        if reply.status == 404:
            return None
        if reply.status != 200:
            raise _api_error("lookup", reply, self._api_key)
        if _is_explicit_no_match(reply.payload):
            return None
        found = _parse_upload(reply.payload)
        if found is None:
            raise MetaFilesError("meta lookup returned a malformed match")
        return found

    def delete_file(self, file_id: MetaFileId) -> None:
        """Delete one id; an already-absent id is idempotent success."""
        if not isinstance(file_id, str) or not file_id.strip():
            raise MetaFilesError("meta delete needs a file id")
        reply = self._request(
            "DELETE", "/files/" + urllib.parse.quote(file_id, safe="")
        )
        if reply.status in (200, 202, 204, 404):
            return None
        raise _api_error("delete", reply, self._api_key)

    def _list_page_ids(self, payload: Any) -> tuple[list[str], str | None]:
        """Return one page's ids plus the next same-origin path, if any.

        Pages carry ``files`` (or the real API's ``data``) plus an optional
        ``next`` path or ``cursor``/``next_cursor`` token. Only same-origin
        paths are ever followed; an absolute URL is refused rather than
        followed with the Authorization header attached.
        """
        if isinstance(payload, dict):
            if "files" in payload:
                items: Any = payload["files"]
            elif "data" in payload:  # real API list shape: {"data": [...]}
                items = payload["data"]
            else:
                items = payload
        else:
            items = payload
        if not isinstance(items, list):
            raise MetaFilesError("meta list returned a malformed payload")
        ids: list[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                ids.append(item)
            elif isinstance(item, dict) and isinstance(item.get("file_id"), str):
                ids.append(item["file_id"])
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])  # real API list items carry "id"
            else:
                raise MetaFilesError("meta list returned a malformed entry")
        if isinstance(payload, dict) and payload.get("has_more"):
            # More pages than this client knows how to follow: fail closed
            # instead of treating one page as the whole list, since a
            # partial list must never prove absence.
            raise MetaFilesError("meta list has more pages than this client reads")
        follow: str | None = None
        if isinstance(payload, dict):
            raw_next = payload.get("next")
            if isinstance(raw_next, str) and raw_next.strip():
                follow = raw_next.strip()
            else:
                for token_key in ("cursor", "next_cursor"):
                    token = payload.get(token_key)
                    if isinstance(token, str) and token.strip():
                        follow = "/files?cursor=" + urllib.parse.quote(
                            token.strip(), safe=""
                        )
                        break
        if follow is not None and not follow.startswith("/"):
            raise MetaFilesError("meta list continuation leaves this origin")
        return ids, follow

    def _list_ids(self) -> tuple[str, ...] | None:
        """Return every listed id, or ``None`` when listing is unsupported.

        Exhausts pagination: an empty first page carrying a continuation
        is not proof of anything until the last page is read. A page
        cap fails closed instead of looping forever.
        """
        path = "/files"
        ids: list[str] = []
        for _ in range(MAX_LIST_PAGES):
            reply = self._request("GET", path)
            if reply.status in (405, 501):
                if path == "/files":
                    return None
                raise MetaFilesError("meta list pagination is unsupported")
            if reply.status != 200:
                raise _api_error("list", reply, self._api_key)
            page_ids, follow = self._list_page_ids(reply.payload)
            ids.extend(page_ids)
            if follow is None:
                return tuple(ids)
            path = follow
        raise MetaFilesError("meta list pagination exceeded its page bound")

    def get_file(self, file_id: MetaFileId) -> MetaUpload | None:
        """Return the present file, ``None`` on definitive GET/list absence.

        Absence needs both proofs: a GET 404 *and* a fully paginated
        list without the id. One-sided evidence — an unsupported GET, an
        unsupported list, or a partial page — raises instead of
        verifying, so callers record acknowledged-only or retry.
        """
        if not isinstance(file_id, str) or not file_id.strip():
            raise MetaFilesError("meta get needs a file id")
        reply = self._request("GET", "/files/" + urllib.parse.quote(file_id, safe=""))
        if reply.status == 200:
            found = _parse_upload(reply.payload)
            if found is None:
                raise MetaFilesError("meta get returned a malformed file")
            return found
        if reply.status in (405, 501):
            # GET unsupported: a listed id is still present evidence and
            # beats the unsupported read (retry); only a fully clean
            # list degrades to acknowledged-only, never to verified.
            listed = self._list_ids()
            if listed is not None and file_id in listed:
                raise MetaFilesError("meta list contradicts an unsupported get")
            raise MetaReadUnsupported("meta file GET reads are unsupported")
        if reply.status != 404:
            raise _api_error("get", reply, self._api_key)
        # GET says absent: the full list must agree before absence is
        # definitive, so a contradicting or unreadable list stays
        # retryable instead of verified.
        listed = self._list_ids()
        if listed is None:
            raise MetaReadUnsupported("meta file list reads are unsupported")
        if file_id in listed:
            raise MetaFilesError("meta list contradicts an absent get")
        return None


assert isinstance(
    ProductionMetaFilesClient.__new__(ProductionMetaFilesClient), MetaFilesClient
)


def select_meta_files_client(
    settings: object,
    *,
    base_url: str = DEFAULT_META_FILES_BASE_URL,
    transport: Callable[..., Any] | None = None,
) -> ProductionMetaFilesClient:
    """Select the production client from settings, or fail closed.

    Returns the keyed production adapter pointed at the configured
    ``META_FILES_BASE_URL`` endpoint when ``META_API_KEY`` is present;
    otherwise raises :class:`MetaClientUnavailableError` so callers render
    an honest unavailable page and record retry, never verified absence.
    An injected opener-shaped ``transport`` replaces urllib (tests);
    production always passes none.
    """
    return ProductionMetaFilesClient.from_settings(
        settings, base_url=base_url, transport=transport
    )
