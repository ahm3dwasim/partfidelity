"""Shared HTTP for the live Meta providers: one POST shape, bounded retries.

Every provider POSTs JSON to ``{base_url}/responses`` with a bearer key.
Redirects are never followed (the Authorization header must not leave
the API origin). ``/responses`` honors no idempotency key (the spike
scripts send only Authorization and Content-Type, and neither the spike
results nor the docs name one for this endpoint), so a paid request is
never re-POSTed once its bytes may have flown: only a connect failure
before send and a 429 refusal retry with backoff up to ``MAX_ATTEMPTS``
attempts, honoring the server's ``retry-after`` up to
``RETRY_MAX_WAIT_S``. Timeouts, post-send transport failures and 5xx
responses raise an indeterminate error at once (billing outcome
unknown: the spike never measured whether failed requests are billed);
any other 4xx fails fast as a determinate refusal. One injected
end-to-end deadline covers every attempt and all backoff: each attempt
runs under the remaining time and the call stops when it is gone. Tests
inject a scripted ``HttpTransport``; production uses
:class:`UrllibTransport`.
"""

from __future__ import annotations

import base64
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Spike-measured retry shape (s3_sam.py): a handful of attempts, and the
# 429 path honors retry-after (60 s seen) under a cap. Only 429 retries:
# a 5xx may already have been billed (the spike never measured whether
# failed requests are billed), so 5xx is indeterminate, never retried.
MAX_ATTEMPTS = 4
RETRY_MAX_WAIT_S = 65.0
RETRYABLE_STATUSES = frozenset({429})
INDETERMINATE_STATUSES = frozenset(range(500, 600))

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


class ProviderError(RuntimeError):
    """A live provider call failed (transport, HTTP, or shape).

    ``indeterminate`` means the request bytes may have reached the API,
    so the billing outcome is unknown and the caller must settle
    conservatively instead of recording $0. Indeterminate errors never
    retry: without an idempotency key a retry would risk a second charge.
    """

    def __init__(
        self, message: str, *, retryable: bool = True, indeterminate: bool = False
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.indeterminate = indeterminate


class ProviderTimeout(ProviderError):
    """No response within the attempt timeout; the server may have billed it."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, indeterminate=True)


class ProviderConnectError(ProviderError):
    """The connection failed before any request byte was sent (safe to retry)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True, indeterminate=False)


class ProviderHttpError(ProviderError):
    """The API answered with an unretryable or exhausted HTTP status."""

    def __init__(self, status: int, message: str, *, indeterminate: bool = False) -> None:
        super().__init__(
            f"HTTP {status}: {message}",
            retryable=not indeterminate,
            indeterminate=indeterminate,
        )
        self.status = status


class ProviderRateLimited(ProviderHttpError):
    """429s persisted through every attempt (pacing lives in the caller)."""


class ProviderMalformed(ProviderError):
    """A 200 body missed the measured response shape (no retry: deterministic)."""


class ProviderDeadline(ProviderError):
    """The job deadline passed before any request byte was sent ($0, no retry)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, indeterminate=False)


def call_budget_s(
    timeout_s: float, deadline_at: float | None, monotonic: Callable[[], float]
) -> float:
    """Return one logical call's end-to-end budget under the job deadline.

    ``deadline_at`` is the job's end on the client's ``monotonic`` clock
    (None: no job bound, the per-call timeout alone). The budget is the
    smaller of the two, so every attempt and backoff of the call ends by
    the job deadline; a deadline already gone (for example after a long
    pacing wait) refuses the call before send with :class:`ProviderDeadline`.
    """
    if deadline_at is None:
        return timeout_s
    remaining = deadline_at - monotonic()
    if not remaining > 0:
        raise ProviderDeadline("job deadline reached before the request was sent")
    return min(timeout_s, remaining)


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP round trip. Header names are lowercased."""

    status: int
    body: bytes
    headers: Mapping[str, str]


class HttpTransport(Protocol):
    """Injectable POST seam. Tests script this; production uses urllib."""

    def post(
        self, url: str, *, headers: Mapping[str, str], body: bytes, timeout_s: float
    ) -> HttpResponse: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: the bearer key never leaves the API origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError(
            f"refused redirect {code} from the Model API", retryable=False
        )


class UrllibTransport:
    """Default transport over urllib. Timeouts map to :class:`ProviderTimeout`."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_NoRedirectHandler)

    def post(
        self, url: str, *, headers: Mapping[str, str], body: bytes, timeout_s: float
    ) -> HttpResponse:
        request = urllib.request.Request(
            url, data=body, method="POST", headers=dict(headers)
        )
        try:
            with self._opener.open(request, timeout=timeout_s) as reply:
                raw = reply.read()
                status = int(getattr(reply, "status", 200) or 200)
                response_headers = {
                    str(key).lower(): str(value)
                    for key, value in reply.headers.items()
                }
        except urllib.error.HTTPError as error:
            try:
                raw = error.read()
            except Exception:
                raw = b""
            try:
                items = error.headers.items() if error.headers else ()
                response_headers = {
                    str(key).lower(): str(value) for key, value in items
                }
            except Exception:
                response_headers = {}
            return HttpResponse(
                status=int(error.code), body=bytes(raw), headers=response_headers
            )
        except (socket.timeout, TimeoutError) as error:
            raise ProviderTimeout(f"timed out after {timeout_s:g}s") from error
        except ProviderError:
            raise
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise ProviderTimeout(f"timed out after {timeout_s:g}s") from error
            if isinstance(reason, (ConnectionRefusedError, socket.gaierror)):
                # DNS and refused connections fail before any request byte
                # is sent: determinate, and the only transport errors that
                # may retry. Anything else may have reached the API after
                # the request was sent, so it is indeterminate.
                raise ProviderConnectError(
                    f"connection failed before send: {reason}"
                ) from error
            raise ProviderError(
                f"request failed: {reason}", indeterminate=True
            ) from error
        except Exception as error:
            raise ProviderError(
                f"request failed: {error}", indeterminate=True
            ) from error
        if 300 <= status < 400:
            raise ProviderError(f"refused redirect {status} from the Model API")
        return HttpResponse(status=status, body=raw, headers=response_headers)


def _retry_wait_s(
    response: HttpResponse | None, attempt: int, cap_s: float = RETRY_MAX_WAIT_S
) -> float:
    """Wait before the next attempt: server retry-after wins, capped; else 2/4/8..."""
    if response is not None:
        seen = response.headers.get("retry-after", "").strip()
        if seen:
            try:
                return max(0.0, min(float(seen), cap_s))
            except ValueError:
                pass
    return min(2.0 * (2.0**attempt), cap_s)


def post_with_retry(
    transport: HttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes,
    timeout_s: float,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    deadline_s: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> HttpResponse:
    """POST once per attempt; return the first 200 or raise the failure.

    Only pre-send connect failures and 429 refusals retry; timeouts,
    post-send failures and 5xx responses raise at once as indeterminate
    (the billing outcome is unknown, so a retry could double-charge).
    ``deadline_s`` bounds the whole logical call — every attempt plus
    all backoff — on the injected ``monotonic`` clock: each attempt runs
    under the remaining time, backoff is truncated to it, and the call
    stops with the last retryable error once it is gone. ``None`` keeps
    the legacy unbounded shape for direct callers; the clients always
    pass their per-call timeout as the deadline.
    """
    if deadline_s is not None and (
        isinstance(deadline_s, bool)
        or not isinstance(deadline_s, (int, float))
        or not deadline_s > 0
    ):
        raise ValueError("deadline_s must be a positive number")
    started = monotonic()
    last: ProviderError | None = None
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        if deadline_s is None:
            attempt_timeout = timeout_s
        else:
            remaining = deadline_s - (monotonic() - started)
            if remaining <= 0:
                if last is None:
                    raise ProviderTimeout("end-to-end deadline already exhausted")
                break
            attempt_timeout = min(timeout_s, remaining)
        try:
            response = transport.post(
                url, headers=headers, body=body, timeout_s=attempt_timeout
            )
        except ProviderError as error:
            if error.indeterminate or not error.retryable:
                raise
            last = error
            response = None
        else:
            if response.status == 200:
                return response
            if response.status in RETRYABLE_STATUSES:
                last = ProviderRateLimited(
                    response.status, "rate limited; backing off"
                )
            elif response.status in INDETERMINATE_STATUSES:
                raise ProviderHttpError(
                    response.status,
                    "server error after send; billing outcome unknown",
                    indeterminate=True,
                )
            else:
                raise ProviderHttpError(response.status, "request refused")
        if attempt + 1 < attempts:
            wait = _retry_wait_s(response, attempt)
            if deadline_s is not None:
                remaining = deadline_s - (monotonic() - started)
                if remaining <= 0:
                    break
                wait = min(wait, remaining)
            sleep(wait)
    assert last is not None
    raise last


def response_json(response: HttpResponse) -> Any:
    """Decode a 200 JSON body, or raise :class:`ProviderMalformed`."""
    try:
        return json.loads(response.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ProviderMalformed("the API returned a non-JSON 200 body") from error


def output_text(payload: Any) -> str:
    """Concatenate the ``output_text`` spans of a ``/responses`` payload."""
    if not isinstance(payload, Mapping):
        raise ProviderMalformed("the API returned no output object")
    chunks: list[str] = []
    output = payload.get("output")
    if not isinstance(output, list):
        raise ProviderMalformed("the API returned no output list")
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for span in content:
            if (
                isinstance(span, Mapping)
                and span.get("type") == "output_text"
                and isinstance(span.get("text"), str)
            ):
                chunks.append(span["text"])
    if not chunks:
        raise ProviderMalformed("the API returned no output text")
    return "".join(chunks)


def usage_counts(payload: Any) -> tuple[int, int]:
    """Return ``(input_tokens, output_tokens)`` or raise :class:`ProviderMalformed`.

    Unmeasurable spend fails the call: settling an unknown actual would
    silently under-bill the ledger.
    """
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    if not isinstance(usage, Mapping):
        raise ProviderMalformed("the API returned no usage object")
    counts: list[int] = []
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProviderMalformed(f"the API returned a bad usage count: {key}")
        counts.append(value)
    return counts[0], counts[1]


def still_data_url(path: Path) -> str:
    """Return a base64 data URL for a PNG/JPEG still, or raise (fail closed)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ProviderError(f"cannot read the still: {error}") from error
    if raw.startswith(PNG_MAGIC):
        mime = "image/png"
    elif raw.startswith(JPEG_MAGIC):
        mime = "image/jpeg"
    else:
        raise ProviderError("the still is neither PNG nor JPEG bytes")
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")
