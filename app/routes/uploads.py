"""Upload/consent/deletion routes (the upload module). Self-contained; integration mounts it.

Every lookup reloads through :meth:`UploadStore.load_job` — no
process-local preview state, no client-supplied path. Rendering uses
Jinja2 with autoescape; only settled constant copy is ever shown, never
a raw exception. The Meta client comes from the replay-aware service
(recorded adapter in replay mode, keyed production adapter in live mode)
unless tests install an override.
"""

from __future__ import annotations

import dataclasses
import hashlib
import html
import json
import logging
import re
import secrets
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from app.routes.site import CachedStaticFiles, add_html_cache_control
from app.brand_templates import make_env
from app.brand_config import icon_links

from app import db, service, uploads as uploads_mod
from app.admission import BudgetExhausted, QueueFullError, RateLimitedError, admit_job
from app.contracts import META_UPLOAD_MAX_HOURS, UPLOAD_PREVIEW_TTL_HOURS, JobId
from app.deletion import CancellationRetryError, cancel_and_delete
from app.ledger import LedgerError, LifecycleRepositoryImpl
from app.pipeline import (
    STEP_EXTRACTION_OVER_BUDGET,
    RecordingMetaClient,
    chunk_budget_refused,
)
from app.meta_files import (
    ACKNOWLEDGED_ONLY_COPY,
    DELETE_REQUEST_ACK_COPY,
    DELETE_REQUEST_FULL_COPY,
    DELETE_REQUEST_THROTTLED_COPY,
    DELETION_FAILURE_COPY,
    META_RECIPIENT,
    META_UNAVAILABLE_COPY,
    UPLOAD_BLOCKED_COPY,
    VERIFIED_ABSENT_COPY,
    MetaClientUnavailableError,
    MissingMetaClient,
    select_meta_files_client,
)
from app.uploads import (
    CONFIRM_CONFLICT_COPY,
    CONFIRM_WAITING_COPY,
    MAX_BYTES,
    MAX_DURATION_S,
    PROBE_TIMEOUT_MS,
    ConfirmationConflictError,
    ConfirmationDeadlineExceeded,
    ConfirmationFailedError,
    OverLimitError,
    claim_for_send,
    confirm_and_send,
    ensure_active_confirmation_job,
    send_claimed_confirmation,
)
from app.uploads_store import (
    DeleteRequestThrottled,
    UploadError,
    UploadStoreImpl,
)
from app import upload_config
from app.routes.site import duration_text
from app.settings import Settings

router = APIRouter()
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

_env = make_env(TEMPLATES_DIR)

_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")

# Server-generated anonymous ids are exactly token_hex(8): 16 lowercase
# hex chars. Anything else in the cookie was not minted here and is
# replaced, never persisted.
_ANON_RE = re.compile(r"[0-9a-f]{16}\Z")

# Injectable for tests; production resolves everything from settings.
_uploads_override = None
_lifecycle_override = None
_meta_client_override: object | None = None


def set_test_overrides(*, uploads=None, lifecycle=None, meta_client=None) -> None:
    """Install test doubles (store, lifecycle, Meta client factory value)."""
    global _uploads_override, _lifecycle_override, _meta_client_override
    _uploads_override = uploads
    _lifecycle_override = lifecycle
    _meta_client_override = meta_client


def clear_test_overrides() -> None:
    """Remove test doubles installed by :func:`set_test_overrides`."""
    global _uploads_override, _lifecycle_override, _meta_client_override
    _uploads_override = None
    _lifecycle_override = None
    _meta_client_override = None


ADMISSION_BUSY_COPY = (
    "The build queue is full right now. Please try again in a minute."
)
ADMISSION_LIMIT_COPY = (
    "Too many uploads at once. Please wait a minute and try again."
)
ADMISSION_PAUSED_COPY = (
    "Uploads are paused right now. You can still try the sample guide."
)
PROCESSING_FAILED_COPY = (
    "We could not process that clip. Try a different file."
)
CONSENT_REQUIRED_COPY = "Tick both boxes to confirm and send."
GONE_COPY = "That preview is gone."
HIDDEN_COPY = DELETION_FAILURE_COPY
CANCELLED_COPY = "Cancelled. Your files were deleted."
CANCELLED_ACK_COPY = (
    "Cancelled. " + ACKNOWLEDGED_ONLY_COPY
)
CANCEL_PENDING_COPY = (
    "Cancelled. Deletion is still running; hidden files stay hidden until it succeeds."
)
CANCEL_RETRY_COPY = (
    "That cancellation did not go through. "
    "Your preview is kept; please try again."
)
CONFIRM_UNAVAILABLE_COPY = META_UNAVAILABLE_COPY
CONFIRM_RETRY_COPY = (
    "That confirmation did not go through. "
    "Your preview is kept; please try again."
)


def _settings(request: Request):
    return getattr(request.app.state, "settings", None)


def _data_root(request: Request) -> Path:
    settings = _settings(request)
    if settings is not None and getattr(settings, "data_root", None) is not None:
        return Path(settings.data_root)
    return Path("var")


def _db_path(request: Request) -> Path:
    settings = _settings(request)
    if settings is not None and getattr(settings, "db_path", None) is not None:
        return Path(settings.db_path)
    return _data_root(request) / "partfidelity.sqlite3"


def _uploads(request: Request) -> UploadStoreImpl:
    if _uploads_override is not None:
        return _uploads_override
    return UploadStoreImpl(_db_path(request))


def _lifecycle(request: Request) -> LifecycleRepositoryImpl:
    if _lifecycle_override is not None:
        return _lifecycle_override
    return LifecycleRepositoryImpl(_db_path(request))


def _meta_client(request: Request):
    if _meta_client_override is not None:
        return _meta_client_override
    settings = _settings(request)
    if settings is not None and service.replay_flag(settings):
        return service.resolve_clients(settings).meta
    return select_meta_files_client(settings)


def _max_duration_s(request: Request) -> float:
    settings = _settings(request)
    if settings is not None and getattr(settings, "max_video_s", None) is not None:
        return float(settings.max_video_s)
    return float(MAX_DURATION_S)


def _limits_json(max_duration_s: float) -> str:
    return json.dumps(
        {
            "maxBytes": MAX_BYTES,
            "maxDuration": float(max_duration_s),
            "message": uploads_mod.over_limit_message(max_duration_s),
            "probeTimeoutMs": PROBE_TIMEOUT_MS,
        }
    )


def _copy(max_duration_s: float) -> dict:
    """Upload and confirmation copy with the enforced limits and hours filled in."""
    return upload_config.render(
        {
            "duration": duration_text(max_duration_s),
            "size": f"{MAX_BYTES // (1024 * 1024)} MB",
            "recipient": META_RECIPIENT,
            "meta_hours": str(META_UPLOAD_MAX_HOURS),
            "local_hours": str(UPLOAD_PREVIEW_TTL_HOURS),
        }
    )


def _upload_page(
    *, error: str = "", blocked: bool = False, max_duration_s: float | None = None
) -> str:
    template = _env.get_template("upload.html")
    cap = float(MAX_DURATION_S) if max_duration_s is None else float(max_duration_s)
    return template.render(
        c=_copy(cap),
        error=error,
        blocked=blocked,
        blocked_copy=UPLOAD_BLOCKED_COPY if blocked else "",
        limits_json=_limits_json(cap),
        max_duration=f"{cap:g}",
        max_duration_text=duration_text(cap),
    )


def _preview_page(
    *,
    job_id: str,
    duration: str,
    size: str,
    video_url: str,
    state: str,
    show_video: bool,
    notice: str = "",
) -> str:
    template = _env.get_template("preview.html")
    cancelled_copy = ""
    if state == "cancelled":
        cancelled_copy = notice
        notice = ""
    return template.render(
        job_id=job_id,
        duration=duration,
        size=size,
        video_url=video_url,
        show_video=show_video,
        state=state,
        notice=notice,
        cancelled_copy=cancelled_copy,
        hidden_copy=HIDDEN_COPY,
        c=_copy(float(MAX_DURATION_S)),  # the confirmation copy has no length placeholder
        meta_recipient=META_RECIPIENT,
        verified_copy=VERIFIED_ABSENT_COPY,
        acknowledged_copy=ACKNOWLEDGED_ONLY_COPY,
    )


def _valid_job_id(job_id: str) -> bool:
    return bool(_JOB_ID_RE.match(job_id or ""))


@router.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request) -> HTMLResponse:
    """Render the placeholder upload form, or the sample-only gate."""
    try:
        blocked = _uploads(request).public_uploads_blocked()
    except Exception:
        blocked = True
    return HTMLResponse(
        _upload_page(blocked=blocked, max_duration_s=_max_duration_s(request))
    )


@router.post("/upload")
def upload_submit(request: Request, clip: UploadFile = File(...)):
    """Stage, validate, admit, and prepare one clip, then redirect.

    A plain sync handler: FastAPI runs it in the threadpool, so the
    ffprobe/ffmpeg/SQLite work never blocks the event loop.
    """
    uploads = _uploads(request)
    cap = _max_duration_s(request)
    try:
        blocked = uploads.public_uploads_blocked()
    except Exception:
        blocked = True
    if blocked:
        return HTMLResponse(
            _upload_page(blocked=True, max_duration_s=cap), status_code=503
        )
    root = _data_root(request)
    try:
        # Lazy startup cleanup: the first request after a SIGKILL or
        # reboot heals crashed-upload leftovers (root staging plus job
        # orphans); the settings-registered startup hook and the hourly sweep
        # phase cover the rest.
        uploads_mod.run_startup_cleanup(data_root=root)
    except Exception:
        pass
    staged: Path | None = None
    try:
        # Bounded sync read into private staging; nothing is admitted,
        # persisted under a job dir, or sent before the limits pass.
        staging_dir = root / uploads_mod.STAGING_DIRNAME
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            staging_dir.chmod(0o700)
        except OSError:
            pass
        staged = staging_dir / f"up-{secrets.token_hex(8)}.bin"
        size = 0
        try:
            with staged.open("wb") as handle:
                while True:
                    chunk = clip.file.read(uploads_mod.CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise OverLimitError("size")
                    handle.write(chunk)
        except BaseException:
            uploads_mod.discard_staged(staged)
            staged = None
            raise
        try:
            uploads_mod.check_upload_bounds(
                staged, size_bytes=size, max_duration_s=cap
            )
        except OverLimitError as error:
            return HTMLResponse(
                _upload_page(error=error.message, max_duration_s=cap),
                status_code=400,
            )
        video_hash = uploads_mod.hash_file(staged)
        raw_anon = request.cookies.get("pf_anon") or ""
        if _ANON_RE.match(raw_anon):
            anon_id = raw_anon
            fresh_anon = False
        else:
            anon_id = secrets.token_hex(8)
            fresh_anon = True
        client_host = request.client.host if request.client else "unknown"
        ip_hash = hashlib.sha256(client_host.encode("utf-8")).hexdigest()
        try:
            ctx = admit_job(
                anon_id, ip_hash, video_hash,
                guide_filename=getattr(_settings(request), "guide_filename", Settings.guide_filename),
                corrections_filename=getattr(_settings(request), "corrections_filename", Settings.corrections_filename),
            )
        except QueueFullError:
            return HTMLResponse(
                _upload_page(error=ADMISSION_BUSY_COPY, max_duration_s=cap),
                status_code=503,
            )
        except RateLimitedError:
            return HTMLResponse(
                _upload_page(error=ADMISSION_LIMIT_COPY, max_duration_s=cap),
                status_code=429,
            )
        except BudgetExhausted:
            return HTMLResponse(
                _upload_page(error=ADMISSION_PAUSED_COPY, max_duration_s=cap),
                status_code=503,
            )
        except (LedgerError, ValueError):
            return HTMLResponse(
                _upload_page(error=ADMISSION_PAUSED_COPY, max_duration_s=cap),
                status_code=503,
            )
        try:
            with staged.open("rb") as handle:
                preview = uploads_mod.prepare_preview(
                    ctx, handle, clip.filename or "clip.mp4", uploads=uploads,
                    max_duration_s=cap,
                )
        except OverLimitError as error:
            return HTMLResponse(
                _upload_page(error=error.message, max_duration_s=cap),
                status_code=400,
            )
        except Exception:
            # Covers staged.open failures before prepare_preview runs
            # as well as prepare failures: the admitted job, its hold,
            # and its empty directory never leak without a terminal row.
            try:
                ctx.finish("failed")
            except Exception:
                pass
            return HTMLResponse(
                _upload_page(error=PROCESSING_FAILED_COPY, max_duration_s=cap),
                status_code=500,
            )
        except BaseException:
            try:
                ctx.finish("failed")
            except Exception:
                pass
            raise
    except OverLimitError as error:
        return HTMLResponse(
            _upload_page(error=error.message, max_duration_s=cap), status_code=400
        )
    finally:
        if staged is not None:
            uploads_mod.discard_staged(staged)
    response = RedirectResponse(f"/preview/{preview.job_id}", status_code=303)
    if fresh_anon:
        response.set_cookie("pf_anon", anon_id, httponly=True, samesite="lax")
    return response


def _preview_context(job_id: str, record) -> dict:
    preview = record.preview
    return {
        "job_id": job_id,
        "duration": f"{preview.duration_s:.1f}",
        "size": str(preview.size_bytes),
        "video_url": f"/preview/{job_id}/video",
    }


def _expire_preview_during_lookup(uploads, job_id: str):
    """Enforce the 24-hour local-preview promise on read; every state.

    At or after ``preview_expires_at`` a still-visible record is hidden before
    anything renders: an unclaimed ``preview`` terminalizes through
    ``recover_job`` (failed, job terminal, footage due), while any
    other state hides through ``hide_expired_preview``, which touches
    no state, claim, artifact, or Meta row. The record is reloaded, so
    callers render the hidden state instead of a live preview. When the
    persistence write fails, the lookup fails closed with an in-memory
    hidden view: an expired preview is never rendered or served.
    """
    loaded = uploads.load_job(JobId(job_id))
    if loaded is None:
        return None
    _, record = loaded
    if record.hidden:
        return loaded
    try:
        expired = db.utcnow() >= record.preview_expires_at
    except Exception:
        expired = False
    if not expired:
        return loaded
    now = db.utcnow()
    try:
        if record.state == "preview" and record.claim_id is None:
            uploads.recover_job(JobId(job_id), now)
        else:
            uploads.hide_expired_preview(JobId(job_id), now)
    except Exception:
        pass
    loaded = uploads.load_job(JobId(job_id))
    if loaded is None:
        return None
    ctx, fresh = loaded
    if not fresh.hidden:
        fresh = dataclasses.replace(fresh, hidden=True)
        return ctx, fresh
    return loaded


# One table maps each fresh durable state to its canonical page, video,
# status, and copy. A caller intent names one of those states; it may pick
# a copy/status variant only when the fresh record shows that same state,
# otherwise the fresh state's canonical row wins. No caller passes copy.
_FRESH_CANONICAL: dict[str, tuple[str, bool, int, str]] = {
    # fresh_key -> (page_state, show_video, status, canonical_notice)
    "cancelled": ("cancelled", False, 200, CANCELLED_COPY),
    "pending": ("pending", False, 202, CANCEL_PENDING_COPY),
    "hidden": ("hidden", False, 410, ""),
    "confirmed": ("confirmed", True, 200, ""),
    "consent": ("consent", True, 200, ""),
}

# Which fresh state each caller intent names. Consent intents share the
# live page; cancelled_ack is the acknowledged-only variant of cancelled.
_INTENT_PAGE = {
    "show": "consent",
    "consent_required": "consent",
    "conflict": "consent",
    "waiting": "consent",
    "unavailable": "consent",
    "retry": "consent",
    "cancel_retry": "consent",
    "confirmed": "confirmed",
    "cancelled": "cancelled",
    "cancelled_ack": "cancelled",
    "pending": "pending",
    "hidden": "hidden",
}

_CONSENT_INTENT_STATUS = {
    "show": 200,
    "consent_required": 400,
    "conflict": 409,
    "waiting": 409,
    "unavailable": 503,
    "cancel_retry": 503,
}

_CONSENT_INTENT_COPY = {
    "show": "",
    "consent_required": CONSENT_REQUIRED_COPY,
    "conflict": CONFIRM_CONFLICT_COPY,
    "waiting": CONFIRM_WAITING_COPY,
    "unavailable": CONFIRM_UNAVAILABLE_COPY,
    "retry": CONFIRM_RETRY_COPY,
    "cancel_retry": CANCEL_RETRY_COPY,
}


def _fresh_key(record) -> str:
    if record.state == "cancelled":
        return "cancelled"
    if record.state == "cancelling":
        return "pending"
    if record.hidden or record.state == "failed":
        return "hidden"
    if record.state == "confirmed" and record.meta_file_id is not None:
        return "confirmed"
    return "consent"


def _consent_status(*, intent: str, record) -> int:
    """Status for the consent page: intent selects, the record confirms.

    A retry renders 409 only while the (fresh, or stale-fallback) record
    still holds a live ``confirming`` claim, else 503. Terminal intents
    (confirmed/cancelled/pending/hidden) on a still-live preview fall back
    to 200: the record wins, so the page is honest about what is durable.
    """
    if intent == "retry":
        return 409 if record.state == "confirming" else 503
    return _CONSENT_INTENT_STATUS.get(intent, 200)


def _consent_notice(*, intent: str) -> str:
    return _CONSENT_INTENT_COPY.get(intent, "")


def _render_preview_response(
    uploads,
    job_id: str,
    decision_record,
    *,
    intent: str,
    max_duration_s: float,
) -> HTMLResponse:
    """Render one preview page from the freshly reloaded durable record.

    Single choke point for every preview render in this module (GET, POST,
    success, conflict, failure, retry, cancel): reloads through
    :func:`_expire_preview_during_lookup` immediately before building the
    template context, and the fresh record alone selects the page, the
    video, and the canonical copy/status from :data:`_FRESH_CANONICAL`.
    Callers pass only an intent naming the operation outcome: a consent
    intent picks its variant only on a live record, ``cancelled_ack``
    picks the acknowledged-only variant only on a cancelled record, and
    every other mismatch renders the fresh canonical. A losing
    confirmation whose competitor completes before the reload therefore
    renders the confirmed page, and a cancel whose sweeper completes
    before the reload renders the cancelled page — never a stale copy.
    Direct :func:`_preview_page` / :func:`_preview_context` calls outside
    this helper are a bug.
    """
    try:
        loaded = _expire_preview_during_lookup(uploads, job_id)
    except Exception:
        loaded = None
        unreadable = True
    else:
        unreadable = False
    if loaded is None and not unreadable:
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=max_duration_s),
            status_code=404,
        )
    record = loaded[1] if loaded is not None else None
    if record is None:
        # Unreadable reload: fall back to the decision-time record. A stale
        # record that already proves hidden, terminal, or past-expiry fails
        # closed; a stale confirmed-and-attached record renders confirmed;
        # anything else renders the consent page with the intent's variant
        # only when it names the live page, so a transient store error
        # stays retryable without ever showing a stale terminal copy.
        record = decision_record
        if record is None:
            return HTMLResponse(
                _upload_page(error=GONE_COPY, max_duration_s=max_duration_s),
                status_code=404,
            )
        try:
            stale_expired = db.utcnow() >= record.preview_expires_at
        except Exception:
            stale_expired = False
        if (
            record.hidden
            or record.state in ("failed", "cancelled", "cancelling")
            or stale_expired
        ):
            base = _preview_context(job_id, record)
            page, video, status, _ = _FRESH_CANONICAL["hidden"]
            return HTMLResponse(
                _preview_page(
                    state=page, show_video=video, notice="", **base
                ),
                status_code=status,
            )
        base = _preview_context(job_id, record)
        if record.state == "confirmed" and record.meta_file_id is not None:
            page, video, status, _ = _FRESH_CANONICAL["confirmed"]
            return HTMLResponse(
                _preview_page(
                    state=page, show_video=video, notice="", **base
                ),
                status_code=status,
            )
        if _INTENT_PAGE.get(intent) == "consent":
            notice = _consent_notice(intent=intent)
            status = _consent_status(intent=intent, record=record)
        else:
            page, video, status, notice = _FRESH_CANONICAL["consent"]
            return HTMLResponse(
                _preview_page(
                    state=page, show_video=video, notice=notice, **base
                ),
                status_code=status,
            )
        return HTMLResponse(
            _preview_page(
                state="consent", show_video=True, notice=notice, **base
            ),
            status_code=status,
        )
    base = _preview_context(job_id, record)
    fresh = _fresh_key(record)
    if fresh == "cancelled":
        page, video, status, canonical = _FRESH_CANONICAL["cancelled"]
        notice = (
            CANCELLED_ACK_COPY if intent == "cancelled_ack" else canonical
        )
        return HTMLResponse(
            _preview_page(
                state=page, show_video=video, notice=notice, **base
            ),
            status_code=status,
        )
    if fresh == "pending":
        page, video, status, notice = _FRESH_CANONICAL["pending"]
        return HTMLResponse(
            _preview_page(
                state=page, show_video=video, notice=notice, **base
            ),
            status_code=status,
        )
    if fresh == "hidden":
        page, video, status, notice = _FRESH_CANONICAL["hidden"]
        return HTMLResponse(
            _preview_page(state=page, show_video=video, notice=notice, **base),
            status_code=status,
        )
    if fresh == "confirmed":
        page, video, status, notice = _FRESH_CANONICAL["confirmed"]
        return HTMLResponse(
            _preview_page(state=page, show_video=video, notice=notice, **base),
            status_code=status,
        )
    if _INTENT_PAGE.get(intent) == "consent":
        return HTMLResponse(
            _preview_page(
                state="consent", show_video=True,
                notice=_consent_notice(intent=intent), **base
            ),
            status_code=_consent_status(intent=intent, record=record),
        )
    page, video, status, notice = _FRESH_CANONICAL["consent"]
    return HTMLResponse(
        _preview_page(
            state=page, show_video=video, notice=notice, **base
        ),
        status_code=status,
    )


@router.get("/preview/{job_id}", response_class=HTMLResponse)
def preview_show(request: Request, job_id: str) -> HTMLResponse:
    """Reload the durable record and render consent, status, or hidden."""
    cap = _max_duration_s(request)
    if not _valid_job_id(job_id):
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=cap), status_code=404
        )

    uploads = _uploads(request)
    loaded = _expire_preview_during_lookup(uploads, job_id)
    if loaded is None:
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=cap), status_code=404
        )
    _, record = loaded
    if record.state == "confirmed" and not record.hidden:
        return _render_preview_response(
            uploads, job_id, record, intent="confirmed", max_duration_s=cap
        )
    if record.state == "cancelled":
        return _render_preview_response(
            uploads, job_id, record, intent="cancelled", max_duration_s=cap,
        )
    if record.hidden or record.state in ("failed", "cancelling"):
        return _render_preview_response(
            uploads, job_id, record, intent="hidden", max_duration_s=cap,
        )
    return _render_preview_response(
        uploads, job_id, record, intent="show", max_duration_s=cap
    )


@router.get("/preview/{job_id}/video")
def preview_video(request: Request, job_id: str):
    """Serve only the registered preview path; hidden previews stay hidden."""
    if not _valid_job_id(job_id):
        return HTMLResponse("Gone.", status_code=404)

    loaded = _expire_preview_during_lookup(_uploads(request), job_id)
    if loaded is None:
        return HTMLResponse("Gone.", status_code=404)
    _, record = loaded
    if record.hidden:
        return HTMLResponse("Hidden.", status_code=410)
    if not record.preview.local_video.is_file():
        return HTMLResponse("Gone.", status_code=404)
    return FileResponse(str(record.preview.local_video), media_type="video/mp4")


def _confirm_failure_response(
    uploads, job_id: str, decision_record, *, max_duration_s: float
):
    """Decide a failed confirmation from a reload; render via the helper.

    The decision reload goes through the expiry-aware loader, and the
    final render goes through :func:`_render_preview_response`, which
    reloads again immediately before rendering — a failure never serves
    the expired video URL, even when expiry crosses mid-operation.
    A failure write can itself fail, and a transient store error can
    abort the claim before anything was written, so an unexpired
    preview may still be live and visible on refresh. The decision
    branches below only select the intent; the fresh reload inside the
    helper selects the page, copy, and status, so only a durable reload
    proving hidden/terminal state renders the permanent 410, anything
    still live renders retryable (409 while a claim is in flight, else
    503), an unreadable reload retries rather than claiming hidden, and
    a raced-through confirmation renders its success page.
    """
    try:
        loaded = _expire_preview_during_lookup(uploads, job_id)
    except Exception:
        loaded = None
        unreadable = True
    else:
        unreadable = False
    if unreadable:
        return _render_preview_response(
            uploads, job_id, decision_record, intent="retry",
            max_duration_s=max_duration_s,
        )
    if loaded is None:
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=max_duration_s),
            status_code=404,
        )
    _, record = loaded
    if record.hidden or record.state in ("failed", "cancelled", "cancelling"):
        return _render_preview_response(
            uploads, job_id, record, intent="hidden",
            max_duration_s=max_duration_s,
        )
    if record.state == "confirmed" and record.meta_file_id is not None:
        return _render_preview_response(
            uploads, job_id, record, intent="confirmed",
            max_duration_s=max_duration_s,
        )
    return _render_preview_response(
        uploads, job_id, record, intent="retry", max_duration_s=max_duration_s,
    )


def _refuse_over_budget_confirm(
    request, uploads, job_id: str, ctx, record, *, max_duration_s: float
):
    """Refuse an over-cap confirmation before anything reaches Meta.

    Runs the configured chunk-count cost preflight from the stored
    preview duration against the same client set the pipeline would
    use, so the route and the pipeline refuse identically. Over cap,
    the job fails closed with the budget message and no key use or
    footage committed, and the caller redirects to the job report.
    Returns the response, or ``None`` when the preflight passes or
    cannot run (no settings: the standalone router has no pipeline).
    When no client set can be built, nothing reaches Meta either: the
    cause is logged and the preview stays, with the honest unavailable
    page. A refusal that cannot persist renders the failure page
    instead of uploading.
    """
    settings = _settings(request)
    if settings is None:
        return None
    try:
        staged = service.resolve_clients(settings, ctx)
    except service.PipelineUnavailable as error:
        logger.warning("confirm job=%s refused before upload: %s", job_id, service.safe_unavailable_cause(error))
        return _render_preview_response(
            uploads, job_id, record, intent="unavailable",
            max_duration_s=max_duration_s,
        )
    refused, _, _, _ = chunk_budget_refused(
        record.preview.duration_s, staged.spark, settings
    )
    if not refused:
        return None
    try:
        uploads.record_failure(
            JobId(job_id), f"confirm-budget:{job_id}", "meta_upload",
            STEP_EXTRACTION_OVER_BUDGET, db.utcnow(),
        )
    except Exception:
        return _confirm_failure_response(
            uploads, job_id, record, max_duration_s=max_duration_s
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/preview/{job_id}/confirm")
def preview_confirm(
    request: Request,
    job_id: str,
    consent: str = Form(default=""),
    ownership: str = Form(default=""),
) -> HTMLResponse:
    """Require both consents, then confirm once and render the result.

    The initial load is expiry-aware, so an expired preview hides —
    or fails closed when the hide write fails — before any consent or
    confirmation render. FastAPI runs this sync handler in a threadpool;
    hosted Meta I/O runs later on the job runner.
    """
    cap = _max_duration_s(request)
    if not _valid_job_id(job_id):
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=cap), status_code=404
        )

    uploads = _uploads(request)
    loaded = _expire_preview_during_lookup(uploads, job_id)
    if loaded is None:
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=cap), status_code=404
        )
    ctx, record = loaded
    if (
        record.state == "confirmed"
        and record.meta_file_id is not None
        and not record.hidden
    ):
        return _render_preview_response(
            uploads, job_id, record, intent="confirmed", max_duration_s=cap
        )
    if record.hidden or record.state not in ("preview", "confirming"):
        return _render_preview_response(
            uploads, job_id, record, intent="hidden", max_duration_s=cap,
        )
    if consent != "on" or ownership != "on":
        return _render_preview_response(
            uploads, job_id, record, intent="consent_required",
            max_duration_s=cap,
        )
    try:
        blocked = uploads.public_uploads_blocked()
    except Exception:
        blocked = True
    if blocked:
        return HTMLResponse(
            _upload_page(blocked=True, max_duration_s=cap), status_code=503
        )
    over_budget = _refuse_over_budget_confirm(
        request, uploads, job_id, ctx, record, max_duration_s=cap
    )
    if over_budget is not None:
        return over_budget
    try:
        client = _meta_client(request)
    except MetaClientUnavailableError:
        return _render_preview_response(
            uploads, job_id, record, intent="unavailable", max_duration_s=cap,
        )
    settings = _settings(request)
    runner = getattr(request.app.state, "job_runner", None)
    runner_lock = None
    if settings is not None and runner is not None:
        runner_lock = service.claim_job_runner_lock(settings, job_id)
        if runner_lock is None:
            return RedirectResponse(f"/jobs/{job_id}", status_code=303)
    try:
        if settings is None or runner is None:
            confirm_and_send(
                JobId(job_id), uploads=uploads,
                client=RecordingMetaClient(client, ctx),
            )
        else:
            claim = claim_for_send(JobId(job_id), uploads=uploads)
    except ConfirmationConflictError as error:
        if runner_lock is not None:
            runner_lock.close()
        waiting = error.message == CONFIRM_WAITING_COPY
        return _render_preview_response(
            uploads, job_id, record,
            intent="waiting" if waiting else "conflict",
            max_duration_s=cap,
        )
    except ConfirmationFailedError:
        if runner_lock is not None:
            runner_lock.close()
        return _confirm_failure_response(
            uploads, job_id, record, max_duration_s=cap
        )
    except (UploadError, LedgerError):
        if runner_lock is not None:
            runner_lock.close()
        return _confirm_failure_response(
            uploads, job_id, record, max_duration_s=cap
        )
    if settings is None or runner is None:
        return _render_preview_response(
            uploads, job_id, record, intent="confirmed", max_duration_s=cap,
        )
    if not claim.acquired:
        if runner_lock is not None:
            runner_lock.close()
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)
    return _start_confirmed_job(
        request, uploads, job_id, record, client, claim.record.claim_id,
        runner_lock=runner_lock,
    )


def _start_confirmed_job(
    request, uploads, job_id: str, decision_record, client,
    claim_id: str | None = None, *, runner_lock=None,
):
    """Queue Meta upload and then the pipeline; redirect at once.

    The run happens on the app's job runner, never on this request: the
    job-status page reports its progress and links the guide when done.
    Without settings (the standalone router, which has no pipeline) the
    confirmation itself still stands and renders as before.
    """
    settings = _settings(request)
    runner = getattr(request.app.state, "job_runner", None)
    if settings is None or runner is None:
        return _render_preview_response(
            uploads, job_id, decision_record, intent="confirmed",
            max_duration_s=_max_duration_s(request),
        )
    lifecycle = _lifecycle(request)
    if runner_lock is None:
        runner_lock = service.claim_job_runner_lock(settings, job_id)
        if runner_lock is None:
            return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    def run_inner() -> None:
        try:
            if claim_id is not None:
                loaded = uploads.load_job(JobId(job_id))
                if loaded is None:
                    raise ConfirmationFailedError()
                send_claimed_confirmation(
                    JobId(job_id), claim_id, uploads=uploads,
                    client=RecordingMetaClient(client, loaded[0]),
                )
            ensure_active_confirmation_job(JobId(job_id))
        except Exception as error:
            logger.warning("confirm job=%s background upload failed: %s", job_id, type(error).__name__)
            try:
                current = uploads.load_job(JobId(job_id))
                if current is None or current[1].cancel_requested_at is None:
                    service.fail_confirmed_upload(
                        JobId(job_id),
                        ("job deadline expired before processing"
                         if isinstance(error, ConfirmationDeadlineExceeded)
                         else "upload not confirmed"),
                        settings, uploads=uploads, lifecycle=lifecycle,
                        meta_client=client,
                    )
            except Exception as failure:
                logger.error("confirm job=%s failure settlement failed: %s", job_id, type(failure).__name__)
            return
        service.run_confirmed_upload(
            JobId(job_id), settings, uploads=uploads, lifecycle=lifecycle,
            meta_client=client,
        )

    def run() -> None:
        try:
            run_inner()
        finally:
            runner_lock.close()

    try:
        accepted = runner.submit(job_id, run)
        if not accepted:
            raise RuntimeError("job runner rejected duplicate submission")
    except Exception as error:
        runner_lock.close()
        logger.warning("confirm job=%s could not queue: %s", job_id, type(error).__name__)
        try:
            service.fail_confirmed_upload(
                JobId(job_id), "background job could not start", settings,
                uploads=uploads, lifecycle=lifecycle, meta_client=client,
            )
        except Exception as failure:
            logger.error("confirm job=%s failure settlement failed: %s", job_id, type(failure).__name__)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/preview/{job_id}/cancel")
def preview_cancel(request: Request, job_id: str) -> HTMLResponse:
    """Cancel through the durable path and render the actual outcome.

    A plain sync handler: FastAPI runs it in the threadpool, so the
    SQLite and Meta I/O never blocks the event loop. Success renders
    only for the durable ``cancelled`` state; anything still
    cancelling or retrying renders an honest pending page instead.
    """
    cap = _max_duration_s(request)
    if not _valid_job_id(job_id):
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=cap), status_code=404
        )

    uploads = _uploads(request)
    if uploads.load_job(JobId(job_id)) is None:
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=cap), status_code=404
        )
    try:
        client = _meta_client(request)
    except MetaClientUnavailableError:
        # No authenticated client: local deletion still runs and Meta rows
        # stay pending for retry — never reported as verified.
        client = MissingMetaClient()
    try:
        result = cancel_and_delete(
            JobId(job_id),
            uploads=uploads,
            lifecycle=_lifecycle(request),
            client=client,
            guide_filename=getattr(_settings(request), "guide_filename", Settings.guide_filename),
            corrections_filename=getattr(_settings(request), "corrections_filename", Settings.corrections_filename),
        )
    except CancellationRetryError:
        return _render_preview_response(
            uploads, job_id, uploads.load_job(JobId(job_id))[1],
            intent="cancel_retry", max_duration_s=cap,
        )
    except (UploadError, LedgerError):
        return HTMLResponse(
            _upload_page(error=GONE_COPY, max_duration_s=cap), status_code=404
        )
    loaded = uploads.load_job(JobId(job_id))
    decision_record = loaded[1] if loaded is not None else None
    durable_cancelled = loaded is not None and loaded[1].state == "cancelled"
    if (
        not durable_cancelled
        or not result.local_deleted
        or result.meta_outcome == "retry"
    ):
        return _render_preview_response(
            uploads, job_id, decision_record, intent="pending",
            max_duration_s=cap,
        )
    if result.meta_outcome == "acknowledged_only":
        intent = "cancelled_ack"
    else:
        intent = "cancelled"
    return _render_preview_response(
        uploads, job_id, decision_record, intent=intent, max_duration_s=cap,
    )


def _delete_request_page(*, error: str = "") -> str:
    err = f"<p>{html.escape(error)}</p>" if error else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"{icon_links()}"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Request deletion</title></head><body><main>"
        "<h1>Request deletion</h1>"
        f"{err}"
        "<form action='/delete-request' method='post'>"
        "<p><label>Job id (if you still have it) <input name='job_id'></label></p>"
        "<p><label>Contact <input name='contact' required></label></p>"
        "<p><button type='submit'>Send request</button></p>"
        "</form>"
        "<p><small>This is a request inbox, not a compliance claim.</small></p>"
        "</main></body></html>"
    )


@router.get("/delete-request", response_class=HTMLResponse)
def delete_request_form() -> HTMLResponse:
    """Render the bounded delete-request inbox form."""
    return HTMLResponse(_delete_request_page())


@router.post("/delete-request", response_class=HTMLResponse)
def delete_request_submit(
    request: Request, job_id: str = Form(default=""), contact: str = Form(default="")
) -> HTMLResponse:
    """Queue one bounded delete request; refuse invalid or full honestly.

    A plain sync handler: FastAPI runs it in the threadpool, so the
    SQLite work never blocks the event loop.
    """
    uploads = _uploads(request)
    contact_text = (contact or "").strip()
    if not contact_text:
        return HTMLResponse(
            _delete_request_page(error="A contact address is required."),
            status_code=400,
        )
    job_text = (job_id or "").strip()
    if job_text and not _valid_job_id(job_text):
        return HTMLResponse(
            _delete_request_page(error="That job id is invalid."), status_code=400
        )
    client_host = request.client.host if request.client else "unknown"
    source_hash = hashlib.sha256(client_host.encode("utf-8")).hexdigest()
    raw_session = request.cookies.get("pf_anon") or ""
    session_hash = raw_session if _ANON_RE.match(raw_session) else None
    try:
        request_id = uploads.submit_delete_request(
            JobId(job_text) if job_text else None,
            contact_text,
            db.utcnow(),
            source_hash=source_hash,
            session_hash=session_hash,
        )
    except ValueError:
        return HTMLResponse(
            _delete_request_page(error="That request was invalid."), status_code=400
        )
    except DeleteRequestThrottled:
        return HTMLResponse(
            _delete_request_page(error=DELETE_REQUEST_THROTTLED_COPY),
            status_code=429,
        )
    except (UploadError, LedgerError):
        return HTMLResponse(
            _delete_request_page(error=DELETE_REQUEST_FULL_COPY), status_code=503
        )
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"{icon_links()}"
        "<title>Request received</title></head><body><main>"
        f"<h1>Request {html.escape(str(request_id))} received</h1>"
        f"<p>{html.escape(DELETE_REQUEST_ACK_COPY)}</p>"
        "</main></body></html>"
    )


def create_uploads_app() -> FastAPI:
    """Standalone app for tests and screenshots: /static mount + router.

    Mount-first order, exactly like the integration mount: ``/static/*``
    resolves from the ``static/`` tree on disk.
    """
    app = FastAPI(title="PartFidelity")
    statics = TEMPLATES_DIR.parent / "static"
    app.mount("/static", CachedStaticFiles(directory=statics, html=False))
    add_html_cache_control(app)
    app.include_router(router)
    return app
