"""the evidence repository tests: round-trip, vocabulary, summaries, gaps."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import evidence
from app.contracts import CallRecord, JobId
from app.routes.calls import router


@dataclass
class FakeCtx:
    job_id: JobId = JobId("job-l5-1")
    evidence_store: object = None

    def record_call_with_event(self, key, record, event) -> None:
        """Explicit fake receipt operation over this context's store."""
        evidence.get_store(self).record_call_with_event(key, record, event)


@pytest.fixture(autouse=True)
def _clean_registry():
    evidence.reset_stores()
    yield
    evidence.reset_stores()


def make_record(job_id="job-l5-1", **over) -> CallRecord:
    base = dict(
        job_id=JobId(job_id),
        stage="locate",
        model="sam-3.1",
        endpoint="/locate",
        latency_ms=4200,
        units_in=1,
        units_out=2,
        estimated_cost_usd=Decimal("0.0025"),
        status="succeeded",
        evidence={"phrase": "fan", "score": 0.66},
    )
    base.update(over)
    return CallRecord(**base)


def test_call_record_round_trip_decimal_cost():
    ctx = FakeCtx(evidence_store=evidence.InMemoryEvidenceStore())
    record = make_record(estimated_cost_usd=Decimal("0.0025"))
    evidence.record_call(ctx, record)  # type: ignore[arg-type]
    (found,) = evidence.list_calls(ctx)  # type: ignore[arg-type]
    assert found.estimated_cost_usd == Decimal("0.0025")
    assert isinstance(found.estimated_cost_usd, Decimal)
    assert found.evidence["phrase"] == "fan"
    assert found.latency_ms == 4200
    # A float cost is refused: money is fixed-point text/decimal, never float.
    with pytest.raises(ValueError):
        evidence.record_call(ctx, make_record(estimated_cost_usd=0.0025))  # type: ignore[arg-type]


def test_record_call_rejects_other_job_result():
    ctx = FakeCtx(job_id=JobId("job-a"))
    with pytest.raises(ValueError, match="different job"):
        evidence.record_call(ctx, make_record(job_id="job-b"))  # type: ignore[arg-type]
    assert evidence.list_calls(ctx) == ()  # type: ignore[arg-type]


def test_minimum_event_vocabulary_only():
    # Independently written: an accidental extra event must fail here.
    assert evidence.EVENT_VOCABULARY == frozenset(
        {
            "job.queued",
            "job.admitted",
            "job.refused",
            "job.cancelled",
            "job.completed",
            "job.failed",
            "consent.recorded",
            "model.call_recorded",
            "model.call_failed",
            "readback.counted",
        }
    )
    ctx = FakeCtx()
    for event in sorted(evidence.EVENT_VOCABULARY):
        evidence.record_event(ctx, event, {"stage": "locate"})  # type: ignore[arg-type]
    assert len(evidence.list_events(ctx)) == len(evidence.EVENT_VOCABULARY)  # type: ignore[arg-type]
    for banned in (
        "gallery.viewed",
        "share.clicked",
        "page.viewed",
        "counter.incremented",
        "analytics.dashboard",
        "return_visit.attributed",
        "job.started",  # plausible but not in the minimum set
    ):
        with pytest.raises(ValueError, match="vocabulary"):
            evidence.record_event(ctx, banned, {})  # type: ignore[arg-type]


def test_run_count_median_worst_not_small_sample_p95():
    calls = (
        make_record(latency_ms=4600),
        make_record(latency_ms=6700),
        make_record(latency_ms=5200),
    )
    summary = evidence.summarize_calls(calls)
    assert summary.run_count == 3
    assert summary.median_latency_ms == 5200
    assert summary.worst_latency_ms == 6700
    assert summary.total_cost_usd == Decimal("0.0075")
    # No percentile surface exists to misuse on small samples.
    assert "p95" not in evidence.CallSummary.__dataclass_fields__
    assert "percentile" not in "".join(evidence.CallSummary.__dataclass_fields__).lower()


def test_missing_evidence_not_fabricated():
    ctx = FakeCtx(job_id=JobId("job-empty"))
    assert evidence.list_calls(ctx) == ()  # type: ignore[arg-type]
    assert evidence.select_sample_readback(()) is None
    app = FastAPI()
    app.include_router(router)
    body = TestClient(app).get("/calls/job-empty").text
    assert "No model calls recorded for this job yet" in body
    assert "No measured read-back for this run: none was recorded, so none is shown" in body
    assert "Checked" not in body  # neither "Checked" nor a badge row exists


def test_failed_call_records_failed_event():
    ctx = FakeCtx(evidence_store=evidence.InMemoryEvidenceStore())
    evidence.record_call(ctx, make_record(status="failed"))  # type: ignore[arg-type]
    events = evidence.list_events(ctx)  # type: ignore[arg-type]
    assert [name for name, _ in events] == ["model.call_failed"]


def test_non_finite_costs_are_rejected():
    ctx = FakeCtx()
    for bad in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        with pytest.raises(ValueError):
            evidence.record_call(ctx, make_record(estimated_cost_usd=bad))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evidence.record_call(  # type: ignore[arg-type]
            ctx, make_record(estimated_cost_usd=Decimal("-0.01"))
        )
    assert evidence.list_calls(ctx) == ()  # type: ignore[arg-type]


def test_even_sized_median_is_exact_not_truncated():
    summary = evidence.summarize_calls(
        (make_record(latency_ms=1), make_record(latency_ms=2))
    )
    assert summary.median_latency_ms == 1.5


def test_call_and_event_commit_atomically_on_failure_and_retry():
    store = evidence.InMemoryEvidenceStore()
    real_op = store.record_call_with_event
    state = {"failures": 1}

    def flaky_op(key, record, event) -> None:
        name, detail = event
        staged = (copy.deepcopy(record.evidence), copy.deepcopy(dict(detail)))
        assert staged[0] is not None and name
        if state["failures"]:
            state["failures"] -= 1
            raise RuntimeError("injected store failure")
        real_op(key, record, event)

    # Instance patch: the failure injection rides on the exact owned type,
    # so get_store still accepts the store.
    store.record_call_with_event = flaky_op  # type: ignore[method-assign]
    ctx = FakeCtx(evidence_store=store)
    with pytest.raises(RuntimeError, match="injected store failure"):
        evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
    assert evidence.list_calls(ctx) == ()  # type: ignore[arg-type]
    assert evidence.list_events(ctx) == ()  # type: ignore[arg-type]
    evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
    assert len(evidence.list_calls(ctx)) == 1  # type: ignore[arg-type]
    events = evidence.list_events(ctx)  # type: ignore[arg-type]
    assert [name for name, _ in events] == ["model.call_recorded"]


class LegacyStore:
    """Old attached-store surface: no record_call_with_event.

    Mirrors the audit exploit shape: ``save_call`` would succeed and
    ``save_event`` would raise, orphaning the call on the legacy path.
    """

    def __init__(self) -> None:
        self.calls: list = []
        self.events: list = []

    def save_call(self, record) -> None:
        self.calls.append(record)

    def calls_for(self, job_id):
        return tuple(c for c in self.calls if str(c.job_id) == str(job_id))

    def save_event(self, job_id, event, detail) -> None:
        raise RuntimeError("injected event failure")

    def events_for(self, job_id):
        return tuple(self.events)


class DuckStore:
    """Foreign store with every method right: still refused by exact type."""

    def __init__(self) -> None:
        self.calls: list = []
        self.events: list = []

    def save_call(self, record) -> None:
        self.calls.append(record)

    def calls_for(self, job_id):
        return tuple(c for c in self.calls if str(c.job_id) == str(job_id))

    def save_event(self, job_id, event, detail) -> None:
        self.events.append((event, dict(detail)))

    def events_for(self, job_id):
        return tuple(self.events)

    def record_call_with_event(self, key, record, event) -> None:
        name, detail = event
        self.save_call(record)
        self.save_event(str(record.job_id), name, detail)


@pytest.mark.parametrize("foreign", [LegacyStore(), DuckStore()])
def test_foreign_store_is_refused_before_any_write(foreign):
    ctx = FakeCtx(evidence_store=foreign)
    with pytest.raises(ValueError, match="must be InMemoryEvidenceStore"):
        evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
    assert foreign.calls == [] and foreign.events == []
    # Events and reads refuse the same way: no write, no read, no receipt.
    with pytest.raises(ValueError, match="must be InMemoryEvidenceStore"):
        evidence.record_event(ctx, "job.queued", {"stage": "locate"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be InMemoryEvidenceStore"):
        evidence.list_calls(ctx)  # type: ignore[arg-type]
    assert foreign.calls == [] and foreign.events == []


class SubStore(evidence.InMemoryEvidenceStore):
    """Exact-method subclass: still a foreign type."""


def test_subclassed_owned_store_is_refused():
    ctx = FakeCtx(evidence_store=SubStore())
    with pytest.raises(ValueError, match="must be InMemoryEvidenceStore"):
        evidence.record_call(ctx, make_record())  # type: ignore[arg-type]


def test_concurrent_retries_complete_one_pair_without_recovery_path():
    store = evidence.InMemoryEvidenceStore()
    store.save_call(make_record())  # orphan: receipt stored, event missing

    def no_recovery(job_id, event_name, detail) -> None:
        raise AssertionError("receipt path must not call save_event")

    # Receipt retries must route through the keyed atomic op only; any
    # check-then-append recovery trips this guard.
    store.save_event = no_recovery  # type: ignore[method-assign]
    ctx = FakeCtx(evidence_store=store)
    barrier = threading.Barrier(2)
    errors: list = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
        except Exception as exc:  # re-raised on the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not [thread for thread in threads if thread.is_alive()]
    assert errors == []
    assert len(evidence.list_calls(ctx)) == 1  # type: ignore[arg-type]
    events = evidence.list_events(ctx)  # type: ignore[arg-type]
    assert [name for name, _ in events] == ["model.call_recorded"]


def test_identical_retry_is_a_noop_on_reference_store():
    ctx = FakeCtx(evidence_store=evidence.InMemoryEvidenceStore())
    evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
    evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
    assert len(evidence.list_calls(ctx)) == 1  # type: ignore[arg-type]
    events = evidence.list_events(ctx)  # type: ignore[arg-type]
    assert [name for name, _ in events] == ["model.call_recorded"]
    # A differing record is a distinct call, not a retry.
    evidence.record_call(ctx, make_record(latency_ms=4201))  # type: ignore[arg-type]
    assert len(evidence.list_calls(ctx)) == 2  # type: ignore[arg-type]


def test_reference_store_upserts_by_idempotency_key():
    store = evidence.InMemoryEvidenceStore()
    record = make_record()
    key = evidence.call_key(record)
    pair = ("model.call_recorded", {"stage": "locate", "status": "succeeded"})
    store.record_call_with_event(key, record, pair)
    store.record_call_with_event(key, record, pair)
    assert len(store.calls_for("job-l5-1")) == 1
    assert len(store.events_for("job-l5-1")) == 1


class _Uncopyable:
    def __deepcopy__(self, memo):
        raise RuntimeError("injected copy failure")


def test_reference_store_leaves_nothing_after_copy_failure():
    store = evidence.InMemoryEvidenceStore()
    record = make_record()
    with pytest.raises(RuntimeError, match="copy failure"):
        store.record_call_with_event(
            evidence.call_key(record),
            record,
            ("model.call_recorded", {"poison": _Uncopyable()}),
        )
    assert store.calls_for("job-l5-1") == ()
    assert store.events_for("job-l5-1") == ()


def test_record_call_with_event_rejects_a_foreign_key_and_stores_nothing():
    # a caller-supplied key that is not the stored record's own
    # idempotency key — here a forbidden email address — is refused before
    # any mutation, even with an otherwise valid receipt.
    store = evidence.InMemoryEvidenceStore()
    record = make_record()
    pair = ("model.call_recorded", {"stage": "locate", "status": "succeeded"})
    with pytest.raises(ValueError, match="does not match the stored record"):
        store.record_call_with_event("jane.doe@example.com", record, pair)
    assert store.calls_for("job-l5-1") == ()
    assert store.events_for("job-l5-1") == ()
    assert "jane.doe@example.com" not in repr(store.calls_for("job-l5-1"))
    assert "jane.doe@example.com" not in repr(store.events_for("job-l5-1"))


# The bypass triple: an email, a media filename, and a
# consent sentence. None may persist through any writable entry point.
_FORBIDDEN_TRIPLE = (
    "jane.doe@example.com",
    "clip.mp4",
    "I, Jane Doe, agree to the recording terms",
)


@pytest.mark.parametrize("value", _FORBIDDEN_TRIPLE)
def test_forbidden_values_rejected_on_every_entry_point(value):
    # save_call: unknown keys and wrong-shaped known keys alike are dropped.
    store = evidence.InMemoryEvidenceStore()
    store.save_call(
        make_record(
            evidence={"phrase": "fan", "score": 0.66, "note": value, "file": value}
        )
    )
    (found,) = store.calls_for("job-l5-1")
    assert found.evidence == {"phrase": "fan", "score": 0.66}
    assert value not in repr(store.calls_for("job-l5-1"))
    # save_event: detail keeps only its declared keys and shapes.
    store.save_event("job-l5-1", "job.queued", {"stage": "locate", "note": value})
    ((_, detail),) = store.events_for("job-l5-1")
    assert detail == {"stage": "locate"}
    assert value not in repr(store.events_for("job-l5-1"))
    # record_call_with_event: both halves are allowlisted before anything
    # is written, so neither half can carry the value.
    pair_store = evidence.InMemoryEvidenceStore()
    record = make_record(evidence={"phrase": value, "note": value})
    # The key is the stored (allowlisted) record's own idempotency key:
    # the forbidden values never reach storage, so they are not in it.
    pair_store.record_call_with_event(
        evidence.call_key(evidence._validated_call_record(record)),
        record,
        ("model.call_recorded", {"stage": "locate", "note": value}),
    )
    (pair_found,) = pair_store.calls_for("job-l5-1")
    assert value not in repr(pair_found.evidence)
    ((_, pair_detail),) = pair_store.events_for("job-l5-1")
    assert value not in repr(pair_detail)


def test_store_entry_points_enforce_the_field_allowlists():
    good_pair = ("model.call_recorded", {"stage": "locate", "status": "succeeded"})
    for over in (
        {"stage": "exfiltrate"},
        {"model": "Jane Doe"},
        {"endpoint": "/var/data/clip.mp4"},
        {"status": "leaked"},
    ):
        store = evidence.InMemoryEvidenceStore()
        with pytest.raises(ValueError):
            store.save_call(make_record(**over))
        assert store.calls_for("job-l5-1") == ()
        with pytest.raises(ValueError):
            store.record_call_with_event(
                "key", make_record(**over), good_pair
            )
        assert store.calls_for("job-l5-1") == ()
        assert store.events_for("job-l5-1") == ()
    store = evidence.InMemoryEvidenceStore()
    with pytest.raises(ValueError, match="vocabulary"):
        store.save_event("job-l5-1", "gallery.viewed", {})
    assert store.events_for("job-l5-1") == ()
    with pytest.raises(ValueError, match="vocabulary"):
        store.record_call_with_event(
            "key", make_record(), ("gallery.viewed", {})
        )
    assert store.calls_for("job-l5-1") == ()
    assert store.events_for("job-l5-1") == ()


def test_record_event_writes_on_the_owned_store():
    store = evidence.InMemoryEvidenceStore()
    ctx = FakeCtx(evidence_store=store)
    evidence.record_event(ctx, "job.queued", {"stage": "locate"})  # type: ignore[arg-type]
    assert evidence.list_events(ctx) == (("job.queued", {"stage": "locate"}),)  # type: ignore[arg-type]


def test_l5_creates_no_tables_and_commits_nothing():
    # The evidence module owns no tables and no connections, so no other module may
    # define a schema, hold a connection, or commit one. The scan covers
    # only the evidence-module source set: later modules (e.g. a ledger adapter
    # committing its own connection) must not fail the evidence module's verification.
    assert not hasattr(evidence, "SqliteEvidenceStore")
    app_dir = Path(evidence.__file__).resolve().parent
    sources = [
        Path(evidence.__file__).resolve(),
        app_dir / "routes" / "calls.py",
    ]
    for path in sources:
        assert path.is_file(), path.name  # the scan ran over real files
        text = path.read_text(encoding="utf-8")
        assert "CREATE TABLE" not in text.upper(), path.name
        assert ".commit(" not in text, path.name
    assert "sqlite3" not in sources[0].read_text(encoding="utf-8")


def test_record_call_delegates_to_the_context_receipt_operation():
    received = []
    ctx = FakeCtx()
    ctx.record_call_with_event = (  # type: ignore[method-assign]
        lambda key, record, event: received.append((key, record, event))
    )
    evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
    assert len(received) == 1
    key, record, (name, detail) = received[0]
    assert name == "model.call_recorded"
    assert detail == {"stage": "locate", "status": "succeeded"}
    assert key == evidence.call_key(record)
    assert evidence.list_calls(ctx) == ()  # type: ignore[arg-type]


def test_non_callable_context_persistence_is_refused():
    ctx = FakeCtx()
    ctx.record_call_with_event = "not-a-method"  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="callable"):
        evidence.record_call(ctx, make_record())  # type: ignore[arg-type]
    assert evidence.list_calls(ctx) == ()  # type: ignore[arg-type]


@dataclass
class NoReceiptCtx:
    """A context without the receipt operation: a contract violation."""

    job_id: JobId = JobId("job-no-receipt")


def test_context_without_receipt_operation_fails_loudly_and_stores_nothing():
    ctx = NoReceiptCtx()
    with pytest.raises(ValueError, match="record_call_with_event"):
        evidence.record_call(ctx, make_record(job_id="job-no-receipt"))  # type: ignore[arg-type]
    assert evidence.list_calls(ctx) == ()  # type: ignore[arg-type]
    assert evidence.list_events(ctx) == ()  # type: ignore[arg-type]
