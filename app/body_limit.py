"""Pre-multipart request-body cap (the upload module).

FastAPI parses and spools the whole multipart body before a route
handler runs, so a route-level size check is too late: a huge upload
is already on disk. This pure-ASGI middleware rejects over-cap
requests before the body is read — by ``Content-Length`` when one is
declared, and by counting streamed bytes otherwise (chunked or lying
senders). Only a bounded multipart overhead above ``MAX_BYTES`` is
allowed. The route-level check stays as defense in depth.

Owned by the upload module; registered by integration (``app/main.py`` belongs to the settings module)::

    from app.body_limit import BodyLimitMiddleware
    app.add_middleware(BodyLimitMiddleware)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.uploads import MAX_BYTES

MULTIPART_OVERHEAD_BYTES = 1024 * 1024
MAX_BODY_BYTES = MAX_BYTES + MULTIPART_OVERHEAD_BYTES

_TOO_LARGE_BODY = b"Request body too large."


class _BodyTooLarge(Exception):
    """Internal signal: streamed bytes passed the cap mid-request."""


async def _send_413(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(_TOO_LARGE_BODY)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _TOO_LARGE_BODY})


class BodyLimitMiddleware:
    """Reject HTTP requests whose body exceeds ``max_body_bytes``."""

    def __init__(self, app: Any, *, max_body_bytes: int = MAX_BODY_BYTES) -> None:
        self._app = app
        self._max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_length = int(declared.decode("latin-1"))
            except (ValueError, UnicodeError):
                declared_length = None
            if declared_length is not None and declared_length > self._max_body_bytes:
                await _send_413(send)
                return
        seen = 0
        started = False

        async def capped_receive() -> dict[str, Any]:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > self._max_body_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self._app(scope, capped_receive, tracking_send)
        except _BodyTooLarge:
            if started:  # pragma: no cover - buffered routes never start early
                raise
            await _send_413(send)
