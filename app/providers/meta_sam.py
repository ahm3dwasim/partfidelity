"""Live hosted-SAM 3.1 still segmentation over the Meta Model API.

Fan and battery only, with the best-measured API phrasings from config.
Boxes parse from the measured ``<|box;...|>`` response text; each box
becomes a frame-sized mask PNG (box filled) because the measured API
shape carries no mask payload. Pacing (55 starts/min) and the 32-call
job bound stay with the ``app.sam`` wrapper; this client retries 429s
with backoff and fails the call on timeout rather than hanging the
job. Every attempted call persists one ``segment`` receipt in the evidence module's
vocabulary at the configured per-call price. Only boxes at or above
the configured threshold are returned; the receipt keeps the best seen
score either way. The client carries its job's immutable
:class:`SamStageConfig` (the same thresholds, the per-call price and
the configured pace), which the ``app.sam`` wrapper reads for its
threshold, its ledger reserve/settle and its pacing slot.

Billing outcomes: a determinate refusal (4xx, exhausted 429s or a
pre-send connect failure) raises, and the wrapper settles $0. Anything
past the point of possible charge — a timeout, a post-send failure, a
5xx, a 200 whose text is unusable, or an unexpected local error after
the request may have flown — returns an empty result with a
non-``ok`` status, which the wrapper settles at the configured
per-call price as failed, never $0; the receipt carries the same
price. A cancel records its receipt first, then re-raises (the wrapper
settles the price as failed); a receipt that will not persist after a
retry is logged as lost and the measured result is still failed
through, so the wrapper settles the actual spend before the failure
surfaces. Every call runs under the job's remaining deadline
(``deadline_at``): the wrapper bounds its in-flight and pacing waits
by :meth:`MetaSamClient.remaining_s` and records a refusal through
:meth:`MetaSamClient.record_refusal`; a call whose wait still ran past
it refuses before send ($0).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from app.contracts import PART_CHECK_THRESHOLDS, JobContext, JobId
from app.parts import normalize_phrase
from app.providers import _records
from app.providers._transport import (
    HttpTransport,
    ProviderError,
    ProviderMalformed,
    UrllibTransport,
    call_budget_s,
    output_text,
    post_with_retry,
    response_json,
    still_data_url,
)
from app.sam import SamMask, SamStageConfig, SamStillResult
from app.settings import SAM_MODEL_RECORD_NAMES

logger = logging.getLogger(__name__)

# Measured box token (s3_sam.py); confidence rides along only when the
# request asked for it, so the c group stays optional.
BOX_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)"
    r"(?:;c=([\d.]+))?\|>"
)


def _require_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _require_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not value > 0
    ):
        raise ValueError("timeout_s must be a positive number")
    return float(value)


def _require_money(value: object, what: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or not value > 0:
        raise ValueError(f"{what} must be a finite positive Decimal")
    return value


class _Poison(Exception):
    """Internal: the call passed the point of possible charge but failed.

    Carries the provider status the wrapper settles at the full
    per-call price as failed, plus a reconcile note. Never escapes the
    client: the public method turns it into an empty result carrying
    that status.
    """

    def __init__(self, status: str, note: str) -> None:
        super().__init__(note)
        self.status = status
        self.note = note


class MetaSamClient:
    """Live ``SamClient``: one HTTP call per ``segment_still`` invocation.

    Instances are per job: the constructor binds the job context the
    ``segment`` receipts persist through. The API key travels only in
    the Authorization header and never reaches a message, receipt, or
    repr.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        price_per_call: Decimal,
        timeout_s: float,
        phrases: Mapping[str, str],
        thresholds: Mapping[str, float] | None = None,
        ctx: JobContext,
        max_starts_per_minute: int | None = None,
        deadline_at: float | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("MetaSamClient needs a non-empty api_key")
        missing = [
            part
            for part in PART_CHECK_THRESHOLDS
            if not phrases.get(part, "").strip()
        ]
        if missing:
            raise ValueError(
                f"SAM client has no API phrasing for: {', '.join(missing)}"
            )
        active_thresholds = (
            dict(PART_CHECK_THRESHOLDS) if thresholds is None else dict(thresholds)
        )
        for part in PART_CHECK_THRESHOLDS:
            value = active_thresholds.get(part)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    f"SAM client has no usable threshold for: {part}"
                )
        self._api_key = api_key
        self._base_url = _require_text(base_url, "base_url").rstrip("/")
        self._model_id = _require_text(model_id, "model_id")
        try:
            self._record_model = SAM_MODEL_RECORD_NAMES[self._model_id]
        except KeyError:
            raise ValueError(f"unknown SAM model id: {self._model_id!r}") from None
        self._price_per_call = _require_money(price_per_call, "price_per_call")
        self._timeout_s = _require_timeout(timeout_s)
        if deadline_at is not None and (
            isinstance(deadline_at, bool) or not isinstance(deadline_at, (int, float))
        ):
            raise ValueError("deadline_at must be a number on the monotonic clock")
        self._deadline_at = deadline_at
        self._phrases = {part: phrases[part].strip() for part in PART_CHECK_THRESHOLDS}
        self._thresholds = {
            part: float(active_thresholds[part]) for part in PART_CHECK_THRESHOLDS
        }
        self.sam_stage_config = SamStageConfig(
            thresholds=self._thresholds,
            price_per_call=self._price_per_call,
            max_starts_per_minute=max_starts_per_minute,
        )
        self._ctx = ctx
        self._transport = transport if transport is not None else UrllibTransport()
        self._sleep = sleep
        self._monotonic = monotonic

    def __repr__(self) -> str:  # never leak the key
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, "
            f"model_id={self._model_id!r})"
        )

    @property
    def bound_job_id(self) -> JobId:
        """The job whose context this client's receipts persist through."""
        return self._ctx.job_id

    def _frame_size(self, image_path: Path) -> tuple[int, int]:
        try:
            raw = Path(image_path).read_bytes()
        except OSError as error:
            raise ProviderError(f"cannot read the still: {error}") from error
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ProviderError("the still is not a decodable image")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ProviderError("the still has no pixels")
        return width, height

    def _request(
        self, image_url: str, api_phrase: str, *, include_confidence: bool
    ) -> str:
        body = json.dumps(
            {
                "model": self._model_id,
                "metadata": {
                    "include_confidence": "true" if include_confidence else "false"
                },
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": api_phrase},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
            }
        ).encode("utf-8")
        budget_s = call_budget_s(self._timeout_s, self._deadline_at, self._monotonic)
        try:
            response = post_with_retry(
                self._transport,
                self._base_url + "/responses",
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                },
                body=body,
                timeout_s=self._timeout_s,
                sleep=self._sleep,
                deadline_s=budget_s,
                monotonic=self._monotonic,
            )
        except ProviderError as error:
            if error.indeterminate:
                raise _Poison(
                    "indeterminate",
                    f"indeterminate SAM call ({error}); settles the "
                    "per-call price; reconcile against provider usage",
                ) from error
            raise
        try:
            return output_text(response_json(response))
        except ProviderMalformed as error:
            raise _Poison(
                "indeterminate",
                f"indeterminate SAM call (200 {error}); settles the "
                "per-call price; reconcile against provider usage",
            ) from error

    def _masks(
        self, text: str, width: int, height: int
    ) -> tuple[tuple[SamMask, ...], float | None]:
        """Parse every box token; rescale, clamp lows, drop inside-out boxes."""
        found: list[SamMask] = []
        best: float | None = None
        for match in BOX_RE.finditer(text):
            x1, y1, x2, y2, sent_w, sent_h = (
                int(match.group(i)) for i in range(1, 7)
            )
            if sent_w <= 0 or sent_h <= 0:
                continue
            box = (
                round(x1 * width / sent_w),
                round(y1 * height / sent_h),
                round(x2 * width / sent_w),
                round(y2 * height / sent_h),
            )
            box = (max(0, box[0]), max(0, box[1]), box[2], box[3])
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            raw_c = match.group(7)
            try:
                score = float(raw_c) if raw_c is not None else None
            except ValueError:
                continue
            if score is not None and not 0.0 <= score <= 1.0:
                continue
            mask = np.zeros((height, width), np.uint8)
            mask[box[1] : min(height, box[3]), box[0] : min(width, box[2])] = 255
            ok, encoded = cv2.imencode(".png", mask)
            if not ok:
                # Post-paid local failure: the 200 already did the work.
                raise _Poison(
                    "indeterminate",
                    "indeterminate SAM call (paid 200, mask would not "
                    "encode); settles the per-call price; reconcile "
                    "against provider usage",
                )
            found.append(
                SamMask(box_xyxy=box, c=score, mask_png=bytes(encoded))
            )
            if score is not None and (best is None or score > best):
                best = score
        return tuple(found), best

    def _failed_record(
        self,
        norm: str,
        latency_ms: int,
        score: float | None,
        cost_usd: Decimal,
    ) -> None:
        _records.persist_receipt(
            self._ctx,
            _records.segment_record(
                job_id=self._ctx.job_id,
                model=self._record_model,
                phrase=norm,
                score=score,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                status="failed",
            ),
        )

    def remaining_s(self) -> float | None:
        """Seconds left before the job deadline (None: no job bound).

        The ``app.sam`` wrapper bounds its in-flight and pacing waits by it.
        """
        if self._deadline_at is None:
            return None
        return self._deadline_at - self._monotonic()

    def record_refusal(self, phrase: str, latency_ms: int) -> None:
        """Persist the $0 failed receipt for a call refused before send."""
        self._failed_record(normalize_phrase(phrase), latency_ms, None, Decimal("0"))

    def _poisoned(self, status: str, note: str) -> SamStillResult:
        """Return an empty result the wrapper settles as failed at full price.

        Any non-``ok`` status takes the wrapper's provider-failure path,
        which settles the full per-call price as failed (never $0). The
        status names the cause; the note carries the reconcile detail.
        """
        logger.warning("poisoned SAM result (%s): %s", status, note)
        return SamStillResult(masks=(), status=status)

    def segment_still(
        self, image_path: Path, phrase: str, *, include_confidence: bool = True
    ) -> SamStillResult:
        """Locate one part in a still; boxes at/above threshold return."""
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("phrase must be a non-empty string")
        norm = normalize_phrase(phrase)
        try:
            api_phrase = self._phrases[norm]
        except KeyError:
            raise ValueError(f"unsupported SAM phrase: {phrase!r}") from None
        started = self._monotonic()
        may_have_sent = False
        try:
            width, height = self._frame_size(Path(image_path))
            image_url = still_data_url(Path(image_path))
            may_have_sent = True
            text = self._request(
                image_url, api_phrase, include_confidence=include_confidence
            )
            seen, best = self._masks(text, width, height)
        except _Poison as poison:
            latency_ms = int((self._monotonic() - started) * 1000)
            self._failed_record(norm, latency_ms, None, self._price_per_call)
            return self._poisoned(poison.status, poison.note)
        except ProviderError:
            # Determinate only: _request converts every indeterminate
            # error to _Poison, and the local ProviderErrors (unreadable
            # still) fire before any request byte. Refusals carry no
            # charge, so $0 is exact here.
            latency_ms = int((self._monotonic() - started) * 1000)
            self._failed_record(norm, latency_ms, None, Decimal("0"))
            raise
        except Exception as error:
            # Unexpected local failure. Before send nothing was spent: $0
            # and the bug surfaces as a raise. Once bytes may have flown
            # (say, a paid 200 then a local mask failure) it is a
            # cost-carrying failed result, so the wrapper settles the
            # price, never $0; the log keeps the traceback.
            latency_ms = int((self._monotonic() - started) * 1000)
            if not may_have_sent:
                self._failed_record(norm, latency_ms, None, Decimal("0"))
                raise
            logger.exception("unexpected SAM client failure after send")
            self._failed_record(norm, latency_ms, None, self._price_per_call)
            return self._poisoned(
                "indeterminate",
                f"unexpected SAM client failure ({type(error).__name__}); "
                f"settles {self._price_per_call}; reconcile against provider usage",
            )
        except BaseException:
            latency_ms = int((self._monotonic() - started) * 1000)
            cost = self._price_per_call if may_have_sent else Decimal("0")
            self._failed_record(norm, latency_ms, None, cost)
            raise
        floor = self._thresholds[norm]
        masks = tuple(
            mask for mask in seen if mask.c is None or mask.c >= floor
        )
        latency_ms = int((self._monotonic() - started) * 1000)
        if not _records.persist_receipt(
            self._ctx,
            _records.segment_record(
                job_id=self._ctx.job_id,
                model=self._record_model,
                phrase=norm,
                score=best,
                latency_ms=latency_ms,
                cost_usd=self._price_per_call,
                status="succeeded",
            ),
        ):
            return self._poisoned(
                "receipt_failed",
                f"segment receipt lost after a retry; settles {self._price_per_call}",
            )
        return SamStillResult(masks=masks)
