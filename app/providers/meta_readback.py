"""Live Spark read-back over the Meta Model API: yes/no on one frame.

The spike-measured question (part visibility, strict ``visible`` schema)
plus the wrapper's position hint, sent with the standard model id —
never contributor mode. Cost settles from the returned usage; a call
whose spend cannot be measured fails instead of under-billing. Each
call persists one ``readback`` receipt in the evidence module's vocabulary;
reserve/settle and the confidence cutoff band stay with the ``app.readback``
wrapper.

Billing outcomes: a determinate refusal (4xx, exhausted 429s or a
pre-send connect failure) raises, and the wrapper settles $0. Anything
past the point of possible charge — a timeout, a post-send failure, a
5xx, a 200 whose shape is unusable, or an unexpected local error
after the request may have flown — returns a poisoned result
carrying the measured cost, or the wrapper's reserved $0.04 when the
spend cannot be measured: the wrapper's own result validation rejects
the answer and settles that cost as failed, never $0. The failed
receipt carries the same cost, and the reconcile note goes to the log
(and rides the wrapper's rejection message). A cancel records its
receipt first, then re-raises (the wrapper settles its hold as
failed); a receipt that will not persist after a retry is logged as
lost and the measured cost is still returned, so the wrapper settles
the actual spend before the failure surfaces. Every call runs under
the job's remaining deadline (``deadline_at``): a call starting after
it refuses before send ($0).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.contracts import READBACK_COST_PER_CALL_USD, JobContext, JobId
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
    usage_counts,
)
from app.readback import ReadbackResult
from app.settings import SPARK_MODEL_RECORD_NAMES

logger = logging.getLogger(__name__)

READBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["visible"],
    "properties": {"visible": {"type": "boolean"}},
}

# spike-measured: a bool needs no reasoning room
READBACK_MAX_OUTPUT_TOKENS = 2000
# Bounded usage for the worst-case reservation. Output is the binding
# request cap; input is one still plus the question, measured near
# $0.004 a call all-in, so 8k input tokens is several times over.
READBACK_MAX_INPUT_TOKENS = 8000


def readback_worst_case(input_price: Decimal, output_price: Decimal) -> Decimal:
    """Return the per-call worst-case reservation for the active prices."""
    return (
        Decimal(READBACK_MAX_INPUT_TOKENS) * input_price
        + Decimal(READBACK_MAX_OUTPUT_TOKENS) * output_price
    )


class _Poison(Exception):
    """Internal: the call passed the point of possible charge but failed.

    Carries the cost the wrapper must settle as failed (measured when
    the 200 carried usable usage, else the wrapper's reserved amount)
    plus a reconcile note. Never escapes the client: the public method
    turns it into a poisoned result the wrapper's own validation
    rejects.
    """

    def __init__(self, cost_usd: Decimal, note: str) -> None:
        super().__init__(note)
        self.cost_usd = cost_usd
        self.note = note


@dataclass(frozen=True)
class _Priced:
    answer: str
    units_in: int
    units_out: int
    cost_usd: Decimal


def _question(phrase: str, position_hint: str) -> str:
    base = (
        f"Is a '{phrase}' clearly visible in this laptop photo? "
        "Answer visible true or false."
    )
    if position_hint.strip():
        return (
            f"Is a '{phrase}' clearly visible in this laptop photo? "
            f"It was labelled near '{position_hint.strip()}'. "
            "Answer visible true or false."
        )
    return base


def _require_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _require_money(value: object, what: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or not value > 0:
        raise ValueError(f"{what} must be a finite positive Decimal")
    return value


def _require_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not value > 0
    ):
        raise ValueError("timeout_s must be a positive number")
    return float(value)


class MetaReadbackClient:
    """Live ``SparkReadbackClient``: one HTTP call per ``read_back`` invocation.

    Instances are per job: the constructor binds the job context the
    ``readback`` receipts persist through. The API key travels only in
    the Authorization header and never reaches a message, receipt, or
    repr.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        input_price: Decimal,
        output_price: Decimal,
        timeout_s: float,
        ctx: JobContext,
        deadline_at: float | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("MetaReadbackClient needs a non-empty api_key")
        self._api_key = api_key
        self._base_url = _require_text(base_url, "base_url").rstrip("/")
        self._model_id = _require_text(model_id, "model_id")
        try:
            self._record_model = SPARK_MODEL_RECORD_NAMES[self._model_id]
        except KeyError:
            raise ValueError(f"unknown Spark model id: {self._model_id!r}") from None
        if deadline_at is not None and (
            isinstance(deadline_at, bool) or not isinstance(deadline_at, (int, float))
        ):
            raise ValueError("deadline_at must be a number on the monotonic clock")
        self._deadline_at = deadline_at
        self._input_price = _require_money(input_price, "input_price")
        self._output_price = _require_money(output_price, "output_price")
        self._timeout_s = _require_timeout(timeout_s)
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

    def _request(
        self, image_url: str, phrase: str, position_hint: str
    ) -> _Priced:
        body = json.dumps(
            {
                "model": self._model_id,
                "max_output_tokens": READBACK_MAX_OUTPUT_TOKENS,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "rb",
                        "schema": READBACK_SCHEMA,
                        "strict": True,
                    }
                },
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": _question(phrase, position_hint),
                            },
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
                    READBACK_COST_PER_CALL_USD,
                    f"indeterminate read-back ({error}); settles "
                    f"{READBACK_COST_PER_CALL_USD}; reconcile against provider usage",
                ) from error
            raise
        try:
            payload = response_json(response)
        except ProviderMalformed as error:
            raise _Poison(
                READBACK_COST_PER_CALL_USD,
                f"indeterminate read-back (200 {error}); settles "
                f"{READBACK_COST_PER_CALL_USD}; reconcile against provider usage",
            ) from error
        try:
            units_in, units_out = usage_counts(payload)
        except ProviderMalformed:
            units_in, units_out = None, None
        measured = (
            Decimal(units_in) * self._input_price
            + Decimal(units_out) * self._output_price
            if units_in is not None and units_out is not None
            else None
        )
        try:
            answer_raw = json.loads(output_text(payload))
        except (ProviderMalformed, ValueError) as error:
            cost = measured if measured is not None else READBACK_COST_PER_CALL_USD
            raise _Poison(
                cost,
                "indeterminate read-back "
                f"(200 with unusable answer: {error}); settles {cost}; "
                "reconcile against provider usage",
            ) from error
        if not isinstance(answer_raw, dict) or answer_raw.get("visible") not in (
            True,
            False,
        ):
            cost = measured if measured is not None else READBACK_COST_PER_CALL_USD
            raise _Poison(
                cost,
                "indeterminate read-back (200 with no visible boolean); "
                f"settles {cost}; reconcile against provider usage",
            )
        if measured is None:  # 200, good answer, no usable usage: unmeasurable
            raise _Poison(
                READBACK_COST_PER_CALL_USD,
                "indeterminate read-back (200 with no usable usage); "
                f"settles {READBACK_COST_PER_CALL_USD}; reconcile against "
                "provider usage",
            )
        assert units_in is not None and units_out is not None
        return _Priced(
            answer="yes" if answer_raw["visible"] else "no",
            units_in=units_in,
            units_out=units_out,
            cost_usd=measured,
        )

    def _failed_record(
        self,
        phrase: str,
        latency_ms: int,
        units_in: int | None,
        units_out: int | None,
        cost_usd: Decimal,
    ) -> None:
        _records.persist_receipt(
            self._ctx,
            _records.readback_record(
                job_id=self._ctx.job_id,
                model=self._record_model,
                phrase=phrase,
                decision=None,
                latency_ms=latency_ms,
                units_in=units_in,
                units_out=units_out,
                cost_usd=cost_usd,
                status="failed",
            ),
        )

    def _poisoned(self, cost_usd: Decimal, note: str, latency_ms: int) -> ReadbackResult:
        """Return a result the wrapper settles as failed at ``cost_usd``.

        The answer is deliberately neither yes nor no, so the wrapper's
        own result validation rejects it and settles the carried cost
        as failed (never $0); the rejection message quotes the answer,
        so the note rides it to the caller.
        """
        logger.warning("poisoned read-back result: %s", note)
        return ReadbackResult(
            answer=f"indeterminate: {note}",
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    def read_back(
        self, image_path: Path, phrase: str, position_hint: str
    ) -> ReadbackResult:
        """Ask Spark whether the part shows in the rendered frame; record it."""
        _require_text(phrase, "phrase")
        if not isinstance(position_hint, str):
            raise ValueError("position_hint must be a string")
        started = self._monotonic()
        may_have_sent = False
        try:
            image_url = still_data_url(Path(image_path))
            may_have_sent = True
            priced = self._request(image_url, phrase, position_hint)
        except _Poison as poison:
            latency_ms = int((self._monotonic() - started) * 1000)
            self._failed_record(phrase, latency_ms, None, None, poison.cost_usd)
            return self._poisoned(poison.cost_usd, poison.note, latency_ms)
        except ProviderError:
            # Determinate only: _request converts every indeterminate
            # error to _Poison, and the still read fires before any
            # request byte. Refusals carry no charge, so $0 is exact.
            latency_ms = int((self._monotonic() - started) * 1000)
            self._failed_record(phrase, latency_ms, None, None, Decimal("0"))
            raise
        except Exception as error:
            # Unexpected local failure. Before send nothing was spent: $0
            # and the bug surfaces as a raise. Once bytes may have flown
            # it is a poisoned result at the reserved amount, so the
            # wrapper settles that, never $0; the log keeps the traceback.
            latency_ms = int((self._monotonic() - started) * 1000)
            if not may_have_sent:
                self._failed_record(phrase, latency_ms, None, None, Decimal("0"))
                raise
            logger.exception("unexpected read-back client failure after send")
            cost = READBACK_COST_PER_CALL_USD
            self._failed_record(phrase, latency_ms, None, None, cost)
            return self._poisoned(
                cost,
                f"unexpected read-back client failure ({type(error).__name__}); "
                f"settles {cost}; reconcile against provider usage",
                latency_ms,
            )
        except BaseException:
            latency_ms = int((self._monotonic() - started) * 1000)
            cost = READBACK_COST_PER_CALL_USD if may_have_sent else Decimal("0")
            self._failed_record(phrase, latency_ms, None, None, cost)
            raise
        latency_ms = int((self._monotonic() - started) * 1000)
        if not _records.persist_receipt(
            self._ctx,
            _records.readback_record(
                job_id=self._ctx.job_id,
                model=self._record_model,
                phrase=phrase,
                decision=priced.answer,
                latency_ms=latency_ms,
                units_in=priced.units_in,
                units_out=priced.units_out,
                cost_usd=priced.cost_usd,
                status="succeeded",
            ),
        ):
            return self._poisoned(
                priced.cost_usd,
                f"readback receipt lost after a retry; settles actual {priced.cost_usd}",
                latency_ms,
            )
        return ReadbackResult(
            answer=priced.answer, cost_usd=priced.cost_usd, latency_ms=latency_ms
        )
