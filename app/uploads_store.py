"""Durable upload lifecycle over the ledger SQLite database.

``UploadStoreImpl`` is the sole owner of durable preview/consent/
confirmation/cancellation/failure state, Meta-id attachment, per-job
recovery, the aggregate public-upload block, and the bounded delete-request
queue. The upload module owns media and Meta I/O but no
process-global lifecycle state: it holds a returned immutable record for one
request and reloads through :meth:`load_job`.

Every state change is one short ``BEGIN IMMEDIATE`` transaction on the same
WAL connection policy as admission and the ledger. A crash between steps is
recoverable through :meth:`list_recoverable` / :meth:`recover_job`: the
committed ``claim_id`` doubles as the provider idempotency/lookup key, and
the persisted ``guard_deadline_utc`` bounds every unknown-id claim. Exact
replays are idempotent; anything outside the closed transition table or with
a conflicting payload fails closed with no change.
"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

from app import db
from app.admission import MAX_RUNNING, job_dir
from app.settings import DEFAULT_GUIDE_FILENAME, DEFAULT_CORRECTIONS_FILENAME
from app.contracts import (
    DELETE_REQUEST_CONTACT_MAX_CHARS,
    DELETE_REQUEST_MAX_PENDING,
    DELETE_REQUEST_PAGE_SIZE,
    DELETE_REQUEST_RETENTION_DAYS,
    GUIDE_CLIP_ARTIFACT_KIND,
    JOB_DEADLINE_S,
    META_EXPIRY_GUARD_MARGIN_HOURS,
    META_UPLOAD_MAX_HOURS,
    UPLOAD_ERROR_MAX_CHARS,
    UPLOAD_FAILURE_BLOCK_THRESHOLD,
    UPLOAD_PREVIEW_TTL_HOURS,
    ArtifactId,
    ConfirmationClaim,
    DeleteRequestId,
    DeleteRequestRecord,
    JobContext,
    JobId,
    MetaFileId,
    UploadPreview,
    UploadRecord,
)
from app.ledger import (
    ContentionError,
    JobContextImpl,
    LedgerError,
    _finalize_job_failed,
    _expire_overdue,
    _promote_waiting,
    queue_guide_images,
    _locked,
)

_FAILURE_STAGES = ("meta_upload", "meta_persist", "local_delete", "meta_delete", "recovery")
logger = logging.getLogger(__name__)
_DELETE_FAILURE_STAGES = ("local_delete", "meta_delete", "recovery")


class UploadError(LedgerError):
    """An upload-store refusal: unknown id, closed-table edge, or conflict."""


class ClaimWaitingError(UploadError):
    """Claim refused: the admitted job is still waiting for a worker."""


class ClaimExpiredError(UploadError):
    """Claim refused: the preview expired or the job passed its deadline."""


class ClaimTerminalError(UploadError):
    """Claim refused: the job already terminalized before confirmation."""


class DeleteRequestThrottled(UploadError):
    """Delete request refused: the source/session recently asked too often."""


# At most this many delete requests per source hash and per session
# hash inside the throttle window. Delete requests are rare and human,
# so five a day per source/session stops queue-filling scripts while a
# legitimate retry still fits; the global 1000-row bound is unchanged.
DELETE_REQUEST_THROTTLE_LIMIT = 5
DELETE_REQUEST_THROTTLE_WINDOW_HOURS = 24
_DELETE_THROTTLE_BUCKET_MAX_CHARS = 256


def _throttle_bucket(raw: str | None, label: str) -> str | None:
    """Validate one optional throttle key, returning its bucket or ``None``."""
    if raw is None:
        return None
    if type(raw) is not str or not raw.strip():
        raise ValueError(f"delete-request {label} must be a string")
    if len(raw) > _DELETE_THROTTLE_BUCKET_MAX_CHARS:
        raise ValueError(f"delete-request {label} exceeds the length bound")
    return raw


def _aware(moment: datetime, label: str) -> datetime:
    """Require an exact aware datetime; return a canonical plain-UTC copy.

    Awareness means a real (exact-``timedelta``) ``utcoffset()`` — a
    ``tzinfo`` whose offset is ``None`` is naive and would raise at the
    first comparison. The offset is read exactly once then the value is
    rebuilt as a NEW plain ``datetime`` from its UTC fields, so every
    later comparison, arithmetic step, and format touches only that
    snapshot and never the caller's ``tzinfo`` again.
    """
    if type(moment) is not datetime:
        raise ValueError(f"{label} must be an aware UTC datetime")
    try:
        offset = moment.utcoffset()
        if offset is None or type(offset) is not timedelta:
            raise ValueError(f"{label} must be an aware UTC datetime")
        return (moment.replace(tzinfo=None) - offset).replace(tzinfo=timezone.utc)
    except Exception:
        raise ValueError(f"{label} must be an aware UTC datetime") from None


def _canonical_or_none(value: datetime) -> datetime | None:
    """Return the canonical plain-UTC copy of ``value``, or ``None``.

    Never raises and never touches the caller's ``tzinfo`` more than
    once: any unusable value (wrong type, naive, hostile offset) maps
    to ``None`` so the caller fails closed without a second look.
    """
    try:
        return _aware(value, "datetime")
    except ValueError:
        return None


def _non_empty(value: str, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _usable_expiry_text(expiry: datetime | None, now: datetime) -> str | None:
    """Canonical UTC text for a usable supplied provider expiry, else ``None``.

    A supplied provider expiry is trusted only when it is an exact
    aware datetime, still in the future, and at most
    ``META_UPLOAD_MAX_HOURS`` after ``now``. The aware check, the
    window check, the plain-UTC rebuild, a second window check on the
    persisted canonical, and the persisted text all happen in ONE
    guarded block; ANY failure — wrong type, naive, a raising offset,
    a window miss — means ``None`` and the caller persists ``now`` as
    the immediate conservative cleanup deadline instead. Callers reuse
    the returned text for persistence AND replay comparison, so the
    caller's datetime is never touched again after this block.
    """
    try:
        if type(expiry) is not datetime:
            return None
        if expiry.utcoffset() is None:
            return None
        if not (now < expiry <= now + timedelta(hours=META_UPLOAD_MAX_HOURS)):
            return None
        plain = expiry.astimezone(timezone.utc)
        if not (now < plain <= now + timedelta(hours=META_UPLOAD_MAX_HOURS)):
            return None
        return db.format_dt(plain)
    except Exception:
        return None


class UploadStoreImpl:
    """SQLite ``UploadStore``. The ledger module owns the implementation; tests inject fakes."""

    def __init__(
        self,
        database: Path | str | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._db = Path(database) if database is not None else db.db_path()
        self._clock = clock if clock is not None else db.utcnow

    def _now(self) -> datetime:
        """Return the injected clock's aware UTC time, failing closed."""
        return _aware(self._clock(), "upload clock")

    def _record_for(self, conn: sqlite3.Connection, job_id: str) -> UploadRecord:
        """Rebuild the immutable record for one upload row plus its joins."""
        up = conn.execute("SELECT * FROM uploads WHERE job_id = ?", (job_id,)).fetchone()
        if up is None:
            raise UploadError(f"unknown upload job {job_id}")
        art = conn.execute(
            "SELECT job_id, relative_path FROM artifacts WHERE id = ?",
            (up["preview_artifact_id"],),
        ).fetchone()
        if art is None:  # pragma: no cover - RESTRICT foreign key prevents this
            raise LedgerError(f"upload {job_id} lost its preview artifact")
        # The path resolves under the artifact's owning job: a chunk
        # sub-job's row references its parent's chunk file.
        directory = job_dir(art["job_id"], root=db.data_root())
        confirmed = up["confirmed_at"]
        preview = UploadPreview(
            job_id=JobId(job_id),
            local_video=directory / art["relative_path"],
            duration_s=float(up["duration_s"]),
            size_bytes=int(up["size_bytes"]),
            audio_removed=bool(up["audio_removed"]),
            confirmed_at=None if confirmed is None else db.parse_dt(confirmed),
        )
        meta_expires: datetime | None = None
        if up["meta_file_id"] is not None:
            meta_row = conn.execute(
                "SELECT expires_at FROM meta_files WHERE id = ?", (up["meta_file_id"],)
            ).fetchone()
            if meta_row is None:  # pragma: no cover - RESTRICT foreign key
                raise LedgerError(f"upload {job_id} lost its meta file")
            meta_expires = db.parse_dt(meta_row["expires_at"])
        failures = conn.execute(
            "SELECT error FROM upload_failures WHERE job_id = ? AND resolved_at IS NULL"
            " ORDER BY created_at DESC, rowid DESC",
            (job_id,),
        ).fetchall()

        def _opt(column: str) -> datetime | None:
            value = up[column]
            return None if value is None else db.parse_dt(value)

        return UploadRecord(
            preview=preview,
            preview_artifact_id=ArtifactId(up["preview_artifact_id"]),
            preview_expires_at=db.parse_dt(up["preview_expires_at"]),
            state=up["state"],
            consented_at=_opt("consented_at"),
            claim_id=up["claim_id"],
            claim_started_at=_opt("claim_started_at"),
            guard_deadline_utc=_opt("guard_deadline_utc"),
            meta_file_id=None if up["meta_file_id"] is None else MetaFileId(up["meta_file_id"]),
            meta_expires_at=meta_expires,
            confirmed_at=None if confirmed is None else db.parse_dt(confirmed),
            cancel_requested_at=_opt("cancel_requested_at"),
            hidden=bool(up["hidden"]),
            failure_count=len(failures),
            last_error=failures[0]["error"] if failures else None,
        )

    def register_preview(
        self,
        ctx: JobContext,
        preview: UploadPreview,
        artifact_id: ArtifactId,
        expires_at: datetime,
    ) -> UploadRecord:
        """Attach a registered preview artifact to its admitted job's upload row.

        Verifies the job is admitted and still waiting/active, the artifact
        belongs to that job with kind ``upload_preview`` and a live
        regular-file row, and the supplied expiry is aware UTC and no later
        than ``UPLOAD_PREVIEW_TTL_HOURS`` out; then shortens the artifact
        expiry and inserts the ``preview`` row. An exact replay returns the
        existing row; a different preview for the job is refused.
        """
        expires_at = _aware(expires_at, "expires_at")
        ctx_job_id = str(ctx.job_id)
        preview_job_id = preview.job_id
        preview_duration = preview.duration_s
        preview_size = preview.size_bytes
        preview_audio_removed = preview.audio_removed
        if type(preview_job_id) is not str or preview_job_id != ctx_job_id:
            raise ValueError("preview belongs to a different job than the context")
        if type(preview_duration) not in (int, float):
            raise ValueError("preview duration_s must be a number")
        if type(preview_size) is not int:
            raise ValueError("preview size_bytes must be an int")
        if preview_duration < 0 or preview_size < 0:
            raise ValueError("preview duration and size must be non-negative")
        if not isinstance(preview_audio_removed, bool):
            raise ValueError("preview audio_removed must be a bool")
        artifact_text = str(artifact_id)
        now = self._now()
        latest = now + timedelta(hours=UPLOAD_PREVIEW_TTL_HOURS)
        if expires_at > latest:
            raise ValueError(
                f"preview expiry exceeds the {UPLOAD_PREVIEW_TTL_HOURS}h preview TTL"
            )
        expires_text = db.format_dt(expires_at)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    job = conn.execute(
                        "SELECT state FROM jobs WHERE id = ?", (ctx_job_id,)
                    ).fetchone()
                    if job is None:
                        raise UploadError(f"unknown job {ctx_job_id}")
                    if job["state"] not in ("waiting", "active"):
                        raise UploadError(
                            f"job {ctx_job_id} is {job['state']}, not open"
                        )
                    art = conn.execute(
                        "SELECT job_id, kind, relative_path, delete_state FROM artifacts"
                        " WHERE id = ?",
                        (artifact_text,),
                    ).fetchone()
                    if art is None or art["job_id"] != ctx_job_id:
                        raise UploadError(
                            f"unknown preview artifact {artifact_text} for job {ctx_job_id}"
                        )
                    if art["kind"] != "upload_preview":
                        raise UploadError(
                            f"preview artifact {artifact_text} has kind {art['kind']!r}"
                        )
                    if art["delete_state"] != "live":
                        raise UploadError(
                            f"preview artifact {artifact_text} is {art['delete_state']}"
                        )
                    absolute = (
                        job_dir(ctx_job_id, root=db.data_root())
                        / art["relative_path"]
                    )
                    if absolute.is_symlink() or not absolute.is_file():
                        raise UploadError(
                            f"preview artifact {artifact_text} has no regular file"
                        )
                    existing = conn.execute(
                        "SELECT * FROM uploads WHERE job_id = ?", (ctx_job_id,)
                    ).fetchone()
                    if existing is not None:
                        same = (
                            existing["preview_artifact_id"] == artifact_text
                            and float(existing["duration_s"]) == float(preview_duration)
                            and int(existing["size_bytes"]) == preview_size
                            and bool(existing["audio_removed"]) == preview_audio_removed
                            and existing["preview_expires_at"] == expires_text
                        )
                        if not same:
                            raise UploadError(
                                f"job {ctx_job_id} already registered a different preview"
                            )
                        conn.execute("COMMIT")
                        return self._record_for(conn, ctx_job_id)
                    moment = db.format_dt(now)
                    conn.execute(
                        "UPDATE artifacts SET expires_at = ? WHERE id = ?",
                        (expires_text, artifact_text),
                    )
                    conn.execute(
                        "INSERT INTO uploads (job_id, preview_artifact_id, state,"
                        " preview_expires_at, duration_s, size_bytes, audio_removed,"
                        " consented_at, claim_id, claim_started_at, guard_deadline_utc,"
                        " meta_file_id, confirmed_at, cancel_requested_at, hidden,"
                        " created_at, updated_at)"
                        " VALUES (?, ?, 'preview', ?, ?, ?, ?, NULL, NULL, NULL, NULL,"
                        " NULL, NULL, NULL, 0, ?, ?)",
                        (
                            ctx_job_id,
                            artifact_text,
                            expires_text,
                            float(preview_duration),
                            preview_size,
                            int(preview_audio_removed),
                            moment,
                            moment,
                        ),
                    )
                    _promote_waiting(conn, moment, MAX_RUNNING)
                    conn.execute("COMMIT")
                    return self._record_for(conn, ctx_job_id)
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "register_preview lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def load_job(self, job_id: JobId) -> tuple[JobContext, UploadRecord] | None:
        """Reload the context and current record for one job, or ``None``.

        Joins uploads, artifacts, jobs, and optional meta_files; never infers
        state from the filesystem. ``None`` for an unknown job or a job with
        no upload row.
        """
        job_text = str(job_id)
        conn = db.connect(self._db)
        try:
            found = conn.execute(
                "SELECT 1 FROM uploads WHERE job_id = ?", (job_text,)
            ).fetchone()
            if found is None:
                return None
            record = self._record_for(conn, job_text)
        finally:
            conn.close()
        ctx = JobContextImpl(
            self._db, JobId(job_text), job_dir(job_text, root=db.data_root())
        )
        return ctx, record

    def claim_confirmation(
        self, job_id: JobId, consented_at: datetime
    ) -> ConfirmationClaim:
        """Single-winner ``BEGIN IMMEDIATE`` claim; only the winner calls Meta.

        Requires an admitted open job. A preview acquires a running slot and
        starts its run deadline atomically with consent; a full queue is retryable.
        A ``confirming``/``confirmed`` row then returns the
        durable record with ``acquired=False`` without re-checking the
        deadline or preview expiry, so a contender racing a live claim
        stands down instead of erroring. Acquiring needs an unhidden
        ``preview`` upload before its preview expiry: the guarded update
        persists the random 128-bit
        provider-queryable ``claim_id`` with ``claim_started_at`` and
        ``guard_deadline_utc`` (claim start + 24h + the 1h margin) before
        any external I/O. Exactly one contender receives ``acquired=True``;
        a lost race to cancellation or failure is refused as well.
        """
        consented_at = _aware(consented_at, "consented_at")
        job_text = str(job_id)
        claim_id = secrets.token_hex(16)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    now = self._now()
                    now_text = db.format_dt(now)
                    up = conn.execute(
                        "SELECT state, preview_expires_at, hidden FROM uploads WHERE job_id = ?",
                        (job_text,),
                    ).fetchone()
                    if up is None:
                        raise UploadError(f"unknown upload job {job_text}")
                    job = conn.execute(
                        "SELECT state, deadline_at FROM jobs WHERE id = ?",
                        (job_text,),
                    ).fetchone()
                    if job is None or job["state"] not in ("active", "waiting"):
                        raise ClaimTerminalError(
                            f"job {job_text} is not active"
                        )
                    if up["state"] in ("confirming", "confirmed"):
                        conn.execute("COMMIT")
                        return ConfirmationClaim(
                            record=self._record_for(conn, job_text), acquired=False
                        )
                    if up["state"] != "preview":
                        raise UploadError(
                            f"job {job_text} upload is {up['state']}, not preview"
                        )
                    if now_text >= up["preview_expires_at"]:
                        raise ClaimExpiredError(
                            f"job {job_text} preview expired"
                        )
                    if up["hidden"]:
                        raise UploadError(f"job {job_text} upload is hidden")
                    # Claim the worker in this same transaction, before any Meta I/O.
                    _expire_overdue(conn, now_text)
                    _promote_waiting(conn, now_text, MAX_RUNNING)
                    active = conn.execute(
                        "SELECT COUNT(*) FROM jobs WHERE state = 'active'"
                    ).fetchone()[0]
                    if active >= MAX_RUNNING:
                        raise ClaimWaitingError(f"job {job_text} is waiting for a worker")
                    conn.execute(
                        "UPDATE jobs SET state = 'active', deadline_at = ? WHERE id = ?",
                        (db.format_dt(now + timedelta(seconds=JOB_DEADLINE_S)), job_text),
                    )
                    guard_text = db.format_dt(now + timedelta(
                        hours=META_UPLOAD_MAX_HOURS + META_EXPIRY_GUARD_MARGIN_HOURS
                    ))
                    updated = conn.execute(
                        "UPDATE uploads SET state = 'confirming', claim_id = ?,"
                        " claim_started_at = ?, guard_deadline_utc = ?, consented_at = ?,"
                        " updated_at = ? WHERE job_id = ? AND state = 'preview'"
                        " AND hidden = 0",
                        (
                            claim_id,
                            now_text,
                            guard_text,
                            db.format_dt(consented_at),
                            now_text,
                            job_text,
                        ),
                    )
                    if updated.rowcount != 1:
                        # Lost the compare-and-set: re-read the durable row.
                        raced = conn.execute(
                            "SELECT state FROM uploads WHERE job_id = ?", (job_text,)
                        ).fetchone()
                        if raced is not None and raced["state"] in (
                            "confirming",
                            "confirmed",
                        ):
                            conn.execute("COMMIT")
                            return ConfirmationClaim(
                                record=self._record_for(conn, job_text),
                                acquired=False,
                            )
                        raise UploadError(
                            f"job {job_text} lost the confirmation race"
                        )
                    conn.execute("COMMIT")
                    return ConfirmationClaim(
                        record=self._record_for(conn, job_text), acquired=True
                    )
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "claim_confirmation lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def claim_chunk_upload(
        self,
        job_id: JobId,
        claim_id: str,
        artifact_id: ArtifactId,
        duration_s: float,
        size_bytes: int,
        now: datetime,
    ) -> UploadRecord:
        """Persist one chunk's pre-upload claim under the caller's key.

        Inserts the sub-job's uploads row in ``confirming`` with the
        caller-supplied provider idempotency key (``claim_id``),
        ``claim_started_at``, and the same guard deadline the preview
        claim uses (claim start + 24h + the 1h margin), before any
        external I/O. The row is hidden: chunk uploads are internal and
        never render. The chunk's own artifact row backs the preview
        join; its duration and size are the cut file's real values, and
        audio stays present (a chunk is a stream copy, never stripped).
        The uploader attaches the Meta id through
        :meth:`complete_confirmation`; a crash between upload and attach
        leaves the committed key listed by :meth:`list_recoverable`, so
        the sweeper recovers the provider file by key exactly like a
        crashed preview confirmation. A reissued key is refused: the
        first claim owns the provider file and its sweeper recovery.
        """
        _non_empty(claim_id, "claim_id")
        artifact_text = str(artifact_id)
        _non_empty(artifact_text, "artifact_id")
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not duration_s > 0
        ):
            raise ValueError("chunk duration_s must be a number > 0")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError("chunk size_bytes must be an int")
        if size_bytes < 0:
            raise ValueError("chunk size_bytes must be non-negative")
        now = _aware(now, "now")
        now_text = db.format_dt(now)
        job_text = str(job_id)
        guard_text = db.format_dt(
            now + timedelta(hours=META_UPLOAD_MAX_HOURS + META_EXPIRY_GUARD_MARGIN_HOURS)
        )
        preview_expires_text = db.format_dt(
            now + timedelta(hours=UPLOAD_PREVIEW_TTL_HOURS)
        )
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    job = conn.execute(
                        "SELECT state FROM jobs WHERE id = ?", (job_text,)
                    ).fetchone()
                    if job is None:
                        raise UploadError(f"unknown upload job {job_text}")
                    if job["state"] != "active":
                        raise UploadError(
                            f"job {job_text} is {job['state']}, not active"
                        )
                    art = conn.execute(
                        "SELECT id FROM artifacts WHERE id = ?",
                        (artifact_text,),
                    ).fetchone()
                    if art is None:
                        raise UploadError(
                            f"unknown chunk artifact {artifact_text}"
                        )
                    try:
                        conn.execute(
                            "INSERT INTO uploads (job_id, preview_artifact_id, state,"
                            " preview_expires_at, duration_s, size_bytes, audio_removed,"
                            " consented_at, claim_id, claim_started_at, guard_deadline_utc,"
                            " meta_file_id, confirmed_at, cancel_requested_at, hidden,"
                            " created_at, updated_at)"
                            " VALUES (?, ?, 'confirming', ?, ?, ?, 0,"
                            " NULL, ?, ?, ?, NULL, NULL, NULL, 1, ?, ?)",
                            (
                                job_text,
                                artifact_text,
                                preview_expires_text,
                                float(duration_s),
                                size_bytes,
                                claim_id,
                                now_text,
                                guard_text,
                                now_text,
                                now_text,
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise UploadError(
                            f"chunk claim {claim_id} is already taken"
                        ) from error
                    conn.execute("COMMIT")
                    return self._record_for(conn, job_text)
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "claim_chunk_upload lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def _check_completion_window(
        self,
        claim_started_at: str,
        guard_deadline_utc: str,
        expires_at: datetime,
        request_time_utc: datetime,
        now: datetime,
    ) -> datetime:
        """Enforce the lower bounds plus the request-plus-24h cap.

        Canonicalizes every supplied time first and checks only the
        snapshots; returns the canonical expiry for persistence.
        """
        expires_at = _aware(expires_at, "expires_at")
        request_time_utc = _aware(request_time_utc, "request_time_utc")
        now = _aware(now, "now")
        if not (
            db.parse_dt(claim_started_at) <= request_time_utc <= now
        ) or request_time_utc > db.parse_dt(guard_deadline_utc) - timedelta(
            hours=META_UPLOAD_MAX_HOURS
        ):
            raise UploadError("confirmation request time is outside the claim window")
        if not (request_time_utc < expires_at and now < expires_at):
            raise UploadError("confirmation Meta expiry is already reached")
        if expires_at > request_time_utc + timedelta(hours=META_UPLOAD_MAX_HOURS):
            raise UploadError("confirmation Meta expiry exceeds request plus 24 hours")
        return expires_at

    def chunk_claim_taken(self, claim_id: str) -> bool:
        """True when a chunk claim id is already taken.

        The chunked pipeline probes this on re-entry so a resumed
        chunk's next attempt uses a fresh idempotency key instead of
        reissuing a key the first claim still owns.
        """
        _non_empty(claim_id, "claim_id")
        conn = db.connect(self._db)
        try:
            row = conn.execute(
                "SELECT 1 FROM uploads WHERE claim_id = ?", (claim_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def load_chunk_claim(
        self, claim_id: str
    ) -> tuple[JobId, MetaFileId | None] | None:
        """Return ``(job, attached Meta id)`` for one chunk claim id.

        The chunked pipeline reads this on re-entry: the job prices
        the attempt's durable ledger spend, and the Meta id (None
        before attach) is drained as superseded cleanup state. None
        when the id was never claimed; a store read failure
        propagates so the chunk fails closed before any side effect.
        """
        _non_empty(claim_id, "claim_id")
        conn = db.connect(self._db)
        try:
            row = conn.execute(
                "SELECT job_id, meta_file_id FROM uploads WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
            if row is None:
                return None
            attached = row["meta_file_id"]
            return (
                JobId(row["job_id"]),
                None if attached is None else MetaFileId(attached),
            )
        finally:
            conn.close()

    def complete_confirmation(
        self,
        claim_id: str,
        meta_file_id: MetaFileId,
        expires_at: datetime,
        request_time_utc: datetime,
        now: datetime,
    ) -> UploadRecord:
        """Atomically attach one Meta row to its claim; cancellation can win.

        One ``BEGIN IMMEDIATE`` transaction verifies the claim, enforces
        ``claim_started_at <= request_time_utc <= now`` with the request no
        later than the derived pre-call deadline (guard minus 24h) plus the
        independent ``request_time_utc < expires_at``, ``now < expires_at``,
        and request-plus-24h bounds, inserts the ``meta_files`` row, attaches
        it, records confirmation, and returns ``confirmed`` — or
        ``cancelling`` when cancellation was requested mid-call, in which
        case the new Meta row is marked due in the same transaction.
        Replaying the same claim/id/expiry returns the same record after
        revalidating the supplied times; a different id or expiry, a failed
        claim, or any bound violation is refused with nothing attached.
        """
        _non_empty(claim_id, "claim_id")
        meta_text = str(meta_file_id)
        _non_empty(meta_text, "meta_file_id")
        now = _aware(now, "now")
        now_text = db.format_dt(now)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    up = conn.execute(
                        "SELECT * FROM uploads WHERE claim_id = ?", (claim_id,)
                    ).fetchone()
                    if up is None:
                        raise UploadError("unknown confirmation claim")
                    job_id = up["job_id"]
                    attached = up["meta_file_id"]
                    if attached is not None:
                        meta_row = conn.execute(
                            "SELECT expires_at FROM meta_files WHERE id = ?", (attached,)
                        ).fetchone()
                        canonical_replay = _canonical_or_none(expires_at)
                        if (
                            attached == meta_text
                            and meta_row is not None
                            and canonical_replay is not None
                            and db.parse_dt(meta_row["expires_at"]) == canonical_replay
                        ):
                            self._check_completion_window(
                                up["claim_started_at"],
                                up["guard_deadline_utc"],
                                canonical_replay,
                                request_time_utc,
                                now,
                            )
                            conn.execute("COMMIT")
                            return self._record_for(conn, job_id)
                        raise UploadError("confirmation conflicts with the attached id")
                    if up["state"] not in ("confirming", "cancelling"):
                        raise UploadError(
                            f"claim is {up['state']}, not confirming"
                        )
                    canonical_expires = self._check_completion_window(
                        up["claim_started_at"],
                        up["guard_deadline_utc"],
                        expires_at,
                        request_time_utc,
                        now,
                    )
                    try:
                        conn.execute(
                            "INSERT INTO meta_files (id, job_id, expires_at, delete_state,"
                            " delete_requested_at, verified_absent_at, last_error)"
                            " VALUES (?, ?, ?, 'live', NULL, NULL, NULL)",
                            (
                                meta_text,
                                job_id,
                                db.format_dt(canonical_expires),
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise UploadError(
                            f"meta file {meta_text} is already attached elsewhere"
                        ) from error
                    cancelled = up["cancel_requested_at"] is not None
                    conn.execute(
                        "UPDATE uploads SET meta_file_id = ?, confirmed_at = ?,"
                        " state = ?, updated_at = ? WHERE job_id = ?",
                        (
                            meta_text,
                            now_text,
                            "cancelling" if cancelled else "confirmed",
                            now_text,
                            job_id,
                        ),
                    )
                    if cancelled:
                        conn.execute(
                            "UPDATE meta_files SET delete_state = 'pending',"
                            " delete_requested_at = ? WHERE id = ?",
                            (now_text, meta_text),
                        )
                    conn.execute(
                        "UPDATE jobs SET confirmed_at = ? WHERE id = ?", (now_text, job_id)
                    )
                    conn.execute("COMMIT")
                    return self._record_for(conn, job_id)
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "complete_confirmation lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def request_cancellation(
        self, job_id: JobId, now: datetime,
        *, guide_filename: str = DEFAULT_GUIDE_FILENAME,
        corrections_filename: str = DEFAULT_CORRECTIONS_FILENAME,
    ) -> UploadRecord:
        """Hide first, then delete: cancellation wins any confirmation race.

        One ``BEGIN IMMEDIATE`` transaction commits ``hidden``,
        ``cancel_requested_at`` (earliest wins on replay), and ``cancelling``,
        marks every already-attached Meta row for the job pending and due,
        queues the preview and guide clips (a done job's too) for the
        sweeper, due now, and applies the shared failed-job finalization to waiting/active
        jobs. Replay is idempotent and never moves a pending/retry row out
        of the due queue; a later ``complete_confirmation`` still attaches
        but returns ``cancelling``, never handed to the readback module.
        """
        now_text = db.format_dt(_aware(now, "now"))
        job_text = str(job_id)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    up = conn.execute(
                        "SELECT state, cancel_requested_at FROM uploads WHERE job_id = ?",
                        (job_text,),
                    ).fetchone()
                    if up is None:
                        raise UploadError(f"unknown upload job {job_text}")
                    if up["state"] == "cancelled":
                        conn.execute("COMMIT")
                        self._write_cancel_reason(job_text)
                        return self._record_for(conn, job_text)
                    cancel_at = up["cancel_requested_at"]
                    if cancel_at is None or now_text < cancel_at:
                        cancel_at = now_text
                    conn.execute(
                        "UPDATE uploads SET hidden = 1, cancel_requested_at = ?,"
                        " state = 'cancelling', updated_at = ? WHERE job_id = ?",
                        (cancel_at, now_text, job_text),
                    )
                    conn.execute(
                        "UPDATE meta_files SET delete_state = 'pending',"
                        " delete_requested_at = ? WHERE job_id = ?"
                        " AND delete_state IN ('live', 'requested')",
                        (now_text, job_text),
                    )
                    guide_dir = job_dir(job_text, root=db.data_root())
                    resources = (
                        ("cancel_guide", guide_filename, (guide_dir / guide_filename).exists()),
                        ("cancel_corrections", corrections_filename,
                         (guide_dir / guide_filename).exists()
                         or (guide_dir / corrections_filename).exists()),
                        ("cancel_export_links", guide_filename,
                         (guide_dir / guide_filename).exists()
                         or conn.execute("SELECT 1 FROM export_links WHERE guide_id = ? LIMIT 1",
                                         (job_text,)).fetchone() is not None),
                    )
                    # The preview and clips go with the cancel, due now; the
                    # sweeper unlinks them once the second check is quiet.
                    conn.execute(
                        "UPDATE artifacts SET delete_state = 'pending_delete',"
                        " expires_at = ?, deleted_at = NULL, hidden = 1"
                        " WHERE job_id = ? AND delete_state != 'tombstoned' AND (id ="
                        " (SELECT preview_artifact_id FROM uploads WHERE job_id = ?)"
                        " OR kind = ?)",
                        (now_text, job_text, job_text, GUIDE_CLIP_ARTIFACT_KIND),
                    )
                    for kind, relative, exists in resources:
                        if exists:
                            conn.execute(
                                "INSERT OR IGNORE INTO artifacts"
                                " (id, job_id, kind, relative_path, expires_at, hidden,"
                                " kept_at, delete_state, deleted_at, created_at)"
                                " VALUES (?, ?, ?, ?, ?, 1, NULL, 'pending_delete', NULL, ?)",
                                (f"{kind}:{job_text}", job_text, kind, relative,
                                 now_text, now_text),
                            )
                    queue_guide_images(conn, job_text, now_text, guide_dir, guide_filename)
                    _finalize_job_failed(conn, job_text, now_text, MAX_RUNNING)
                    conn.execute("COMMIT")
                    self._write_cancel_reason(job_text)
                    return self._record_for(conn, job_text)
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "request_cancellation lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def _write_cancel_reason(self, job_id: str) -> None:
        marker = job_dir(job_id, root=db.data_root()) / "refusal.json"
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            with marker.open("x", encoding="utf-8") as handle:
                json.dump({"kind": "cancelled"}, handle)
        except FileExistsError:
            pass
        except OSError as error:
            logger.warning("cancellation reason job=%s unwritten: %s", job_id, type(error).__name__)

    def _attach_known_meta(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        meta_file_id: str,
        meta_expires_at: datetime | None,
        now: datetime,
        now_text: str,
    ) -> None:
        """Insert-or-exact-replay one known Meta id and queue it for deletion.

        Uses the canonical text of the supplied aware UTC provider
        expiry when it is future and within the 24 h Meta bound,
        otherwise ``now`` as an immediate conservative cleanup deadline
        (never represented as a confirmed provider expiry). A
        conflicting attached id or valid provider expiry fails closed.
        """
        supplied_text = _usable_expiry_text(meta_expires_at, now)
        up = conn.execute(
            "SELECT meta_file_id FROM uploads WHERE job_id = ?", (job_id,)
        ).fetchone()
        attached = up["meta_file_id"] if up is not None else None
        if attached is not None and attached != meta_file_id:
            raise UploadError("failure carries a conflicting meta id")
        meta_row = conn.execute(
            "SELECT job_id, expires_at FROM meta_files WHERE id = ?", (meta_file_id,)
        ).fetchone()
        if meta_row is not None and meta_row["job_id"] != job_id:
            raise UploadError("failure meta id belongs to another job")
        if supplied_text is not None and meta_row is not None:
            if meta_row["expires_at"] != supplied_text:
                raise UploadError("failure carries a conflicting meta expiry")
        if attached is None:
            if meta_row is not None:
                conn.execute(
                    "UPDATE uploads SET meta_file_id = ? WHERE job_id = ?",
                    (meta_file_id, job_id),
                )
            else:
                stored = supplied_text if supplied_text is not None else now_text
                conn.execute(
                    "INSERT INTO meta_files (id, job_id, expires_at, delete_state,"
                    " delete_requested_at, verified_absent_at, last_error)"
                    " VALUES (?, ?, ?, 'pending', ?, NULL, NULL)",
                    (meta_file_id, job_id, stored, now_text),
                )
                conn.execute(
                    "UPDATE uploads SET meta_file_id = ? WHERE job_id = ?",
                    (meta_file_id, job_id),
                )
        # Otherwise the same id is already attached (a retry carrying an
        # invalid expiry must not fight the stored conservative deadline).
        conn.execute(
            "UPDATE meta_files SET delete_state = 'pending', delete_requested_at = ?"
            " WHERE id = ? AND delete_state IN ('live', 'requested')",
            (now_text, meta_file_id),
        )

    @staticmethod
    def _check_replay_meta(
        conn: sqlite3.Connection,
        job_id: str,
        meta_file_id: str,
        meta_expires_at: datetime | None,
        now: datetime,
    ) -> None:
        """Refuse a replay that adds or changes the recorded Meta identity.

        A supplied id must already be attached to this upload, and a
        supplied *usable* expiry must match the stored one; anything else
        fails closed, so a replay can never silently succeed while leaving
        a remote file unqueued. An unusable supplied expiry carries no
        identity (``now`` was stored instead), so it never conflicts.
        """
        attached = conn.execute(
            "SELECT meta_file_id FROM uploads WHERE job_id = ?", (job_id,)
        ).fetchone()
        if attached is None or attached["meta_file_id"] != meta_file_id:
            raise UploadError("failure key carries a conflicting meta id")
        supplied_text = _usable_expiry_text(meta_expires_at, now)
        if supplied_text is not None:
            meta_row = conn.execute(
                "SELECT expires_at FROM meta_files WHERE id = ?", (meta_file_id,)
            ).fetchone()
            if meta_row is not None and meta_row["expires_at"] != supplied_text:
                raise UploadError("failure key carries a conflicting meta expiry")

    def record_failure(
        self,
        job_id: JobId,
        failure_key: str,
        stage: Literal["meta_upload", "meta_persist", "local_delete", "meta_delete", "recovery"],
        error: str,
        now: datetime,
        *,
        meta_file_id: MetaFileId | None = None,
        meta_expires_at: datetime | None = None,
    ) -> UploadRecord:
        """Record one idempotent keyed failure; queue any known Meta id.

        One transaction inserts ``upload_failures`` by unique ``failure_key``
        (an exact replay returns the recomputed record without touching the
        count; a same-key different payload — including a Meta id or valid
        expiry that was not recorded — is refused), hides the upload, and
        moves processing per the closed table. With a known id it attaches
        the ``meta_files`` row, marks it pending and due, applies the shared
        failed-job finalization, and leaves the upload ``cancelling``.
        Without one, ``meta_upload``/``meta_persist``/recovery move
        ``preview``/``confirming`` to ``failed``; a claimed row keeps its
        claim as the listed provider-lookup handle, while an unclaimed
        preview (no handle, never listed once failed) is terminalized in
        this same transaction exactly like an expired preview in
        ``recover_job``. ``local_delete``/``meta_delete``/recovery preserve
        ``confirmed`` or ``cancelling`` so the durable id remains retryable.
        A Meta expiry without an id is refused.
        """
        _non_empty(failure_key, "failure_key")
        if type(stage) is not str or stage not in _FAILURE_STAGES:
            raise ValueError(f"failure stage {stage!r} is unknown")
        if type(error) is not str:
            raise ValueError("failure error must be a string")
        now = _aware(now, "now")
        now_text = db.format_dt(now)
        job_text = str(job_id)
        trimmed = error[:UPLOAD_ERROR_MAX_CHARS]
        meta_text: str | None = None
        if meta_file_id is not None:
            meta_text = str(meta_file_id)
            _non_empty(meta_text, "meta_file_id")
        elif meta_expires_at is not None:
            raise ValueError("failure meta expiry without a meta id is refused")
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    up = conn.execute(
                        "SELECT state, claim_id, preview_artifact_id FROM uploads"
                        " WHERE job_id = ?",
                        (job_text,),
                    ).fetchone()
                    if up is None:
                        raise UploadError(f"unknown upload job {job_text}")
                    prior = conn.execute(
                        "SELECT job_id, stage, error FROM upload_failures WHERE failure_key = ?",
                        (failure_key,),
                    ).fetchone()
                    if prior is not None:
                        if (
                            prior["job_id"] != job_text
                            or prior["stage"] != stage
                            or prior["error"] != trimmed
                        ):
                            raise UploadError("failure key carries a conflicting payload")
                        if meta_text is not None:
                            self._check_replay_meta(
                                conn,
                                job_text,
                                meta_text,
                                meta_expires_at,
                                now,
                            )
                        conn.execute("COMMIT")
                        return self._record_for(conn, job_text)
                    if up["state"] == "cancelled" and stage == "recovery":
                        conn.execute("COMMIT")
                        return self._record_for(conn, job_text)
                    conn.execute(
                        "INSERT INTO upload_failures (failure_key, job_id, stage, error,"
                        " created_at, resolved_at) VALUES (?, ?, ?, ?, ?, NULL)",
                        (failure_key, job_text, stage, trimmed, now_text),
                    )
                    state = up["state"]
                    if meta_text is not None:
                        if state == "cancelled":
                            raise UploadError("cancelled uploads record no failures")
                        self._attach_known_meta(
                            conn,
                            job_text,
                            meta_text,
                            meta_expires_at,
                            now,
                            now_text,
                        )
                        _finalize_job_failed(conn, job_text, now_text, MAX_RUNNING)
                        state = "cancelling"
                    elif state in ("preview", "confirming"):
                        state = "failed"
                    elif state == "failed":
                        pass
                    elif state in ("confirmed", "cancelling"):
                        if stage not in _DELETE_FAILURE_STAGES:
                            raise UploadError(
                                f"{stage} failure cannot move a {state} upload"
                            )
                    else:  # cancelled
                        raise UploadError("cancelled uploads record no failures")
                    conn.execute(
                        "UPDATE uploads SET hidden = 1, state = ?, updated_at = ?"
                        " WHERE job_id = ?",
                        (state, now_text, job_text),
                    )
                    if (
                        meta_text is None
                        and state == "failed"
                        and up["state"] == "preview"
                        and up["claim_id"] is None
                    ):
                        # An unclaimed preview has no claim key for the
                        # sweeper to look up and is never listed once
                        # failed, so terminalize it here — the same
                        # transaction, reusing the shared finalizer exactly
                        # like recover_job does for an expired preview.
                        _finalize_job_failed(conn, job_text, now_text, MAX_RUNNING)
                        conn.execute(
                            "UPDATE artifacts SET expires_at = ?, hidden = 1"
                            " WHERE id = ? AND kept_at IS NULL AND expires_at > ?",
                            (now_text, up["preview_artifact_id"], now_text),
                        )
                    conn.execute("COMMIT")
                    return self._record_for(conn, job_text)
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "record_failure lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def list_recoverable(self, now: datetime) -> tuple[UploadRecord, ...]:
        """Return every recovery-due upload, whatever its state.

        State never suppresses provider-key recovery: every row with a
        committed ``claim_id`` and no local ``meta_file_id`` is listed,
        including ``failed`` and ``cancelling`` rows — plus every expired,
        unclaimed ``preview`` row.
        """
        now_text = db.format_dt(_aware(now, "now"))
        conn = db.connect(self._db)
        try:
            rows = conn.execute(
                "SELECT job_id FROM uploads"
                " WHERE (claim_id IS NOT NULL AND meta_file_id IS NULL)"
                " OR (state = 'preview' AND claim_id IS NULL AND preview_expires_at <= ?)"
                " ORDER BY updated_at, job_id",
                (now_text,),
            ).fetchall()
            return tuple(self._record_for(conn, row["job_id"]) for row in rows)
        finally:
            conn.close()

    def _has_pending_deletes(self, conn: sqlite3.Connection, job_id: str) -> bool:
        """True while any local/Meta row for the job awaits a delete outcome."""
        artifact = conn.execute(
            "SELECT 1 FROM artifacts WHERE job_id = ?"
            " AND delete_state IN ('pending_delete', 'retry') LIMIT 1",
            (job_id,),
        ).fetchone()
        if artifact is not None:
            return True
        meta = conn.execute(
            "SELECT 1 FROM meta_files WHERE job_id = ?"
            " AND delete_state IN ('pending', 'retry', 'requested') LIMIT 1",
            (job_id,),
        ).fetchone()
        return meta is not None

    def recover_job(
        self, job_id: JobId, now: datetime, *, no_match_claim_id: str | None = None
    ) -> UploadRecord:
        """Re-evaluate one named job after sweeper lookups and delete outcomes.

        One transaction, scoped to this job only. An expired unclaimed
        preview is hidden, moved to ``failed``, its job terminalized, and
        its non-kept preview artifact made due (the same pass's artifact
        queue then deletes the footage and records the tombstone). An
        unknown-id claim resolves only on a definitive provider no-match
        supplied with the exact looked-up claim id after the persisted guard
        deadline; an omitted key, a mismatch, an attached claim, a lookup
        error, or ``now`` at or before the guard leaves the row queued. A
        ``cancelling`` upload closes to ``cancelled`` and this job's failure
        rows resolve only once no local/Meta row remains pending/retry and
        no unknown-id claim remains unresolved.
        """
        now_text = db.format_dt(_aware(now, "now"))
        if no_match_claim_id is not None:
            _non_empty(no_match_claim_id, "no_match_claim_id")
        job_text = str(job_id)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    up = conn.execute(
                        "SELECT * FROM uploads WHERE job_id = ?", (job_text,)
                    ).fetchone()
                    if up is None:
                        raise UploadError(f"unknown upload job {job_text}")
                    claim = up["claim_id"]
                    attached = up["meta_file_id"]
                    if no_match_claim_id is not None and (
                        claim is None or no_match_claim_id != claim or attached is not None
                    ):
                        raise UploadError("no-match key does not match the open claim")
                    guard_passed = (
                        no_match_claim_id is not None
                        and up["guard_deadline_utc"] is not None
                        and now_text > up["guard_deadline_utc"]
                    )
                    unresolved_claim = (
                        claim is not None and attached is None and not guard_passed
                    )
                    state = up["state"]
                    if (
                        state == "preview"
                        and claim is None
                        and now_text >= up["preview_expires_at"]
                    ):
                        state = "failed"
                        conn.execute(
                            "UPDATE uploads SET hidden = 1, state = 'failed',"
                            " updated_at = ? WHERE job_id = ?",
                            (now_text, job_text),
                        )
                        _finalize_job_failed(conn, job_text, now_text, MAX_RUNNING)
                        conn.execute(
                            "UPDATE artifacts SET expires_at = ?, hidden = 1 WHERE id = ?"
                            " AND kept_at IS NULL AND expires_at > ?",
                            (
                                now_text,
                                up["preview_artifact_id"],
                                now_text,
                            ),
                        )
                    elif guard_passed and state in (
                        "confirming",
                        "failed",
                        "cancelling",
                        "cancelled",
                    ):
                        # A definitive no-match observed after the guard
                        # resolves the unknown-id claim: the spent key is
                        # cleared so the row leaves the recoverable set, and
                        # the job is terminalized. The claim timestamps stay
                        # as history; the key itself is never reissued.
                        if state == "confirming":
                            state = "failed"
                        conn.execute(
                            "UPDATE uploads SET state = ?, claim_id = NULL,"
                            " updated_at = ? WHERE job_id = ?",
                            (state, now_text, job_text),
                        )
                        _finalize_job_failed(conn, job_text, now_text, MAX_RUNNING)
                    if state == "cancelling" and not unresolved_claim:
                        if not self._has_pending_deletes(conn, job_text):
                            state = "cancelled"
                            conn.execute(
                                "UPDATE uploads SET state = 'cancelled', updated_at = ?"
                                " WHERE job_id = ?",
                                (now_text, job_text),
                            )
                        _finalize_job_failed(conn, job_text, now_text, MAX_RUNNING)
                    if not unresolved_claim and not self._has_pending_deletes(
                        conn, job_text
                    ):
                        conn.execute(
                            "UPDATE upload_failures SET resolved_at = ? WHERE job_id = ?"
                            " AND resolved_at IS NULL",
                            (now_text, job_text),
                        )
                    conn.execute("COMMIT")
                    return self._record_for(conn, job_text)
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("recover_job lost a write race; retry") from error
                raise
        finally:
            conn.close()

    def hide_expired_preview(self, job_id: JobId, now: datetime) -> UploadRecord:
        """Hide one preview-expired upload without touching any lifecycle.

        One ``BEGIN IMMEDIATE`` transaction sets ``hidden`` when ``now``
        is at or after ``preview_expires_at`` and the row is still visible. It
        changes no state, writes no failure row, and touches neither the
        artifact nor the Meta row, so a ``confirming`` claim still
        completes (completion keys off cancellation, not visibility) and
        a ``confirmed`` id stays attached and due-queue eligible: only
        the local display goes dark. Already-hidden and unexpired rows
        are exact no-ops returning the current record.
        """
        now_text = db.format_dt(_aware(now, "now"))
        job_text = str(job_id)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    up = conn.execute(
                        "SELECT hidden, preview_expires_at FROM uploads"
                        " WHERE job_id = ?",
                        (job_text,),
                    ).fetchone()
                    if up is None:
                        raise UploadError(f"unknown upload job {job_text}")
                    if not up["hidden"] and now_text >= up["preview_expires_at"]:
                        conn.execute(
                            "UPDATE uploads SET hidden = 1, updated_at = ?"
                            " WHERE job_id = ?",
                            (now_text, job_text),
                        )
                    conn.execute("COMMIT")
                    return self._record_for(conn, job_text)
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "hide_expired_preview lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def public_uploads_blocked(self) -> bool:
        """True once unresolved durable failure rows reach the threshold.

        A plain database query, so both workers agree and recover together
        when the responsible jobs resolve; no process memory involved.
        """
        conn = db.connect(self._db)
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM upload_failures WHERE resolved_at IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()
        return total >= UPLOAD_FAILURE_BLOCK_THRESHOLD

    @staticmethod
    def _check_throttle_bucket(
        conn: sqlite3.Connection, bucket: str, now: datetime, now_text: str
    ) -> None:
        """Consume one throttle token, or raise :class:`DeleteRequestThrottled`.

        Fixed windows anchored at first use: a bucket outside the
        window restarts at one, otherwise the count must stay under the
        limit. Runs inside the caller's ``BEGIN IMMEDIATE`` transaction.
        """
        row = conn.execute(
            "SELECT window_start, count FROM delete_throttle WHERE bucket = ?",
            (bucket,),
        ).fetchone()
        window_start_text = db.format_dt(
            now - timedelta(hours=DELETE_REQUEST_THROTTLE_WINDOW_HOURS)
        )
        if row is None or row["window_start"] <= window_start_text:
            conn.execute(
                "INSERT OR REPLACE INTO delete_throttle (bucket, window_start, count)"
                " VALUES (?, ?, 1)",
                (bucket, now_text),
            )
            return
        if row["count"] >= DELETE_REQUEST_THROTTLE_LIMIT:
            raise DeleteRequestThrottled(
                "delete-request source is sending too many requests"
            )
        conn.execute(
            "UPDATE delete_throttle SET count = count + 1 WHERE bucket = ?",
            (bucket,),
        )

    def submit_delete_request(
        self,
        job_id: JobId | None,
        contact: str,
        now: datetime,
        *,
        source_hash: str | None = None,
        session_hash: str | None = None,
    ) -> DeleteRequestId:
        """Queue one bounded operator delete request: contact address only.

        One ``BEGIN IMMEDIATE`` transaction purges expired rows, returns
        the live row for an idempotent ``(job_id, contact)`` replay
        instead of inserting a duplicate, enforces the per-source and
        per-session throttle windows, refuses when
        ``DELETE_REQUEST_MAX_PENDING`` live rows already exist, and
        otherwise inserts an expiry exactly
        ``DELETE_REQUEST_RETENTION_DAYS`` later. The job id is nullable
        so a person who lost it can still request deletion. Callers
        that cannot identify the source pass nothing and get the
        global bound only.
        """
        if type(contact) is not str or not contact.strip():
            raise ValueError("delete-request contact must be a non-empty string")
        if len(contact) > DELETE_REQUEST_CONTACT_MAX_CHARS:
            raise ValueError("delete-request contact exceeds the length bound")
        job_text: str | None = None
        if job_id is not None:
            job_text = str(job_id)
            _non_empty(job_text, "job_id")
        source_bucket = _throttle_bucket(source_hash, "source_hash")
        session_bucket = _throttle_bucket(session_hash, "session_hash")
        now = _aware(now, "now")
        now_text = db.format_dt(now)
        expires_text = db.format_dt(
            now + timedelta(days=DELETE_REQUEST_RETENTION_DAYS)
        )
        request_id = DeleteRequestId(secrets.token_hex(16))
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # Databases created before migration 003 gain the
                    # throttle table here; init_db-created ones no-op.
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS delete_throttle ("
                        " bucket TEXT PRIMARY KEY,"
                        " window_start TEXT NOT NULL,"
                        " count INTEGER NOT NULL)"
                    )
                    conn.execute(
                        "DELETE FROM delete_requests WHERE expires_at <= ?", (now_text,)
                    )
                    conn.execute(
                        "DELETE FROM delete_throttle WHERE window_start <= ?",
                        (
                            db.format_dt(
                                now
                                - timedelta(
                                    hours=DELETE_REQUEST_THROTTLE_WINDOW_HOURS
                                )
                            ),
                        ),
                    )
                    replay = conn.execute(
                        "SELECT id FROM delete_requests WHERE contact = ?"
                        " AND expires_at > ?"
                        " AND ((job_id IS NULL AND ? IS NULL) OR job_id = ?)"
                        " ORDER BY requested_at, id LIMIT 1",
                        (contact, now_text, job_text, job_text),
                    ).fetchone()
                    if replay is not None:
                        conn.execute("COMMIT")
                        return DeleteRequestId(replay["id"])
                    if source_bucket is not None:
                        self._check_throttle_bucket(
                            conn, "ip:" + source_bucket, now, now_text
                        )
                    if session_bucket is not None:
                        self._check_throttle_bucket(
                            conn, "sess:" + session_bucket, now, now_text
                        )
                    live = conn.execute("SELECT COUNT(*) FROM delete_requests").fetchone()[0]
                    if live >= DELETE_REQUEST_MAX_PENDING:
                        raise UploadError("delete-request queue is full")
                    conn.execute(
                        "INSERT INTO delete_requests (id, job_id, contact, requested_at,"
                        " expires_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            str(request_id),
                            job_text,
                            contact,
                            now_text,
                            expires_text,
                        ),
                    )
                    conn.execute("COMMIT")
                    return request_id
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "submit_delete_request lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def list_delete_requests(self, limit: int) -> tuple[DeleteRequestRecord, ...]:
        """Return queued delete requests, oldest first (operator-only)."""
        if type(limit) is not int:
            raise ValueError("delete-request limit must be an int")
        if not 1 <= limit <= DELETE_REQUEST_PAGE_SIZE:
            raise ValueError("delete-request limit is outside the page size")
        conn = db.connect(self._db)
        try:
            rows = conn.execute(
                "SELECT id, job_id, contact, requested_at, expires_at"
                " FROM delete_requests ORDER BY requested_at, id LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return tuple(
            DeleteRequestRecord(
                id=DeleteRequestId(row["id"]),
                job_id=None if row["job_id"] is None else JobId(row["job_id"]),
                contact=row["contact"],
                requested_at=db.parse_dt(row["requested_at"]),
                expires_at=db.parse_dt(row["expires_at"]),
            )
            for row in rows
        )

    def resolve_delete_request(self, request_id: DeleteRequestId) -> None:
        """Delete one queued delete request immediately."""
        request_text = str(request_id)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    deleted = conn.execute(
                        "DELETE FROM delete_requests WHERE id = ?", (request_text,)
                    )
                    if deleted.rowcount != 1:
                        raise UploadError(f"unknown delete request {request_text}")
                    conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "resolve_delete_request lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def purge_delete_requests(self, now: datetime) -> int:
        """Delete expired delete requests; return the purged count."""
        now_text = db.format_dt(_aware(now, "now"))
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    deleted = conn.execute(
                        "DELETE FROM delete_requests WHERE expires_at <= ?", (now_text,)
                    )
                    count = deleted.rowcount
                    conn.execute("COMMIT")
                    return count
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "purge_delete_requests lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()
