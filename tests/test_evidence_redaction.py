"""Evidence allowlist tests: only declared keys/shapes persist; the rest is dropped."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import evidence
from app.contracts import CallRecord, JobId
from app.routes.calls import resolve_job_context, router


@dataclass
class FakeCtx:
    job_id: JobId = JobId("job-red")
    evidence_store: object = None

    def record_call_with_event(self, key, record, event) -> None:
        """Explicit fake receipt operation over this context's store."""
        evidence.get_store(self).record_call_with_event(key, record, event)


@pytest.fixture(autouse=True)
def _clean_registry():
    evidence.reset_stores()
    yield
    evidence.reset_stores()


_JOB_SEQ = [0]


def _record_with(evidence_map, **over):
    _JOB_SEQ[0] += 1
    # Bare context: receipts land in the module registry, which is the
    # production read path these page-render assertions exercise.
    ctx = FakeCtx(JobId(f"job-red-{_JOB_SEQ[0]}"))
    base = dict(
        job_id=ctx.job_id,
        stage="extract",
        model="spark-1.3",
        endpoint="/extract",
        latency_ms=100,
        units_in=None,
        units_out=None,
        estimated_cost_usd=Decimal("0.01"),
        status="succeeded",
        evidence=evidence_map,
    )
    base.update(over)
    evidence.record_call(ctx, CallRecord(**base))  # type: ignore[arg-type]
    return ctx


def _stored(ctx) -> CallRecord:
    (found,) = evidence.list_calls(ctx)  # type: ignore[arg-type]
    return found


def test_keys_headers_paths_ips_and_consent_are_rejected_or_redacted():
    secrets = {
        "META_API_KEY": "fake-meta-key",
        "auth_header": "Bearer [REDACTED]",
        "ip": "192.168.4.22",
        "consent": "I, Jane Doe, agree to the recording terms",
        "email": "jane.doe@example.com",
        "apikey": "shh",
        "userConsentText": "yes I agree",
        "note": "contact jane.doe@example.com or Bearer [REDACTED]",
        "reason": "tried 192.168.4.22 then /var/data/clip.mp4",
    }
    # Unknown keys are dropped before storage, never kept or rendered.
    for key, value in secrets.items():
        ctx = _record_with({key: value})
        assert _stored(ctx).evidence == {}

    # Rendered output carries neither the secrets nor the identifiers.
    ctx = _record_with(dict(secrets))
    app = FastAPI()
    app.include_router(router)
    body = TestClient(app).get(f"/calls/{ctx.job_id}").text
    for leaked in (
        "192.168.4.22",
        "/var/data/clip.mp4",
        "jane.doe@example.com",
        "Bearer [REDACTED]",
        "mk-live-",
        "Jane Doe",
    ):
        assert leaked not in body


# The bypass values: a name, a consent sentence, a file name,
# a bearer string, an SSN-like number, and a phone number.
_BYPASSES = (
    "Jane Doe",
    "I agree to the recording terms",
    "PXL_001.MP4",
    "Bearer [SECURITY_DATA]",
    "123-45-6789",
    "(415) 555-1212",
)


def test_bypass_values_never_store_or_render():
    for value in _BYPASSES:
        # In the known phrase key (wrong shape) and in unknown keys alike.
        ctx = _record_with(
            {"phrase": value, "note": value, "detail": value},
            stage="readback",
            endpoint="/readback",
        )
        assert value not in repr(_stored(ctx).evidence)
        assert "phrase" not in _stored(ctx).evidence
        assert evidence.select_sample_readback(evidence.list_calls(ctx)) is None  # type: ignore[arg-type]

    # One read-back attempt carrying every bypass at once: nothing sticks,
    # the page falls back, and the render leaks nothing.
    ctx = _record_with(
        {
            "phrase": _BYPASSES[0],
            "note": _BYPASSES[1],
            "file": _BYPASSES[2],
            "auth": _BYPASSES[3],
            "ssn": _BYPASSES[4],
            "phone": _BYPASSES[5],
            "decision": "yes",
            "score": 0.66,
        },
        stage="readback",
        endpoint="/readback",
    )
    assert _stored(ctx).evidence == {"decision": "yes", "score": 0.66}
    app = FastAPI()
    app.include_router(router)
    body = TestClient(app).get(f"/calls/{ctx.job_id}").text
    for leaked in _BYPASSES:
        assert leaked not in body
    assert "No measured read-back for this run" in body

    # Contrast: a vocabulary phrase still stores and renders.
    good = _record_with(
        {"phrase": "battery", "decision": "no", "score": 0.52},
        stage="readback",
        endpoint="/readback",
    )
    assert _stored(good).evidence == {
        "phrase": "battery",
        "decision": "no",
        "score": 0.52,
    }
    assert "battery" in TestClient(app).get(f"/calls/{good.job_id}").text


def test_wrong_shaped_known_keys_are_dropped():
    bad_readback = {
        "phrase": "Jane Doe",  # not in the part-phrase vocabulary
        "decision": "not_run",
        "score": 2.5,
    }
    ctx = _record_with(bad_readback, stage="readback", endpoint="/readback")
    assert _stored(ctx).evidence == {}
    for bad_score in (-0.1, float("nan"), float("inf"), True, "0.66", None):
        ctx = _record_with(
            {"phrase": "fan", "score": bad_score},
            stage="locate",
            endpoint="/locate",
        )
        assert _stored(ctx).evidence == {"phrase": "fan"}
    for bad_phrase in ("", "fan ", "FAN", "cooler", None, 5):
        ctx = _record_with(
            {"phrase": bad_phrase, "score": 0.66},
            stage="locate",
            endpoint="/locate",
        )
        assert _stored(ctx).evidence == {"score": 0.66}
    for bad_decision in ("", "maybe", None, True):
        ctx = _record_with(
            {"phrase": "fan", "decision": bad_decision, "score": 0.66},
            stage="readback",
            endpoint="/readback",
        )
        assert _stored(ctx).evidence == {"phrase": "fan", "score": 0.66}
    for bad_index in (-1, True, 1_000_001, "1", 1.5, None):
        ctx = _record_with({"run_index": bad_index})
        assert _stored(ctx).evidence == {}
    # Boundary values that pass: vocabulary read-back phrasing, 0/1 scores,
    # and the index bound itself.
    ctx = _record_with(
        {"phrase": "cooling fan", "decision": "yes", "score": 1},
        stage="readback",
        endpoint="/readback",
    )
    assert _stored(ctx).evidence == {
        "phrase": "cooling fan",
        "decision": "yes",
        "score": 1,
    }
    ctx = _record_with({"run_index": 1_000_000})
    assert _stored(ctx).evidence == {"run_index": 1_000_000}


def test_raw_media_and_paths_are_not_storable():
    for bad in (b"\x89PNG-binary", Path("/tmp/clip.mp4"), {"nested", "set"}):
        ctx = _record_with({"note": bad})
        assert _stored(ctx).evidence == {}
        ctx = _record_with({"score": bad}, stage="locate", endpoint="/locate")
        assert _stored(ctx).evidence == {}


def test_evidence_is_bounded_and_copied():
    oversized = {"k": "x " * 4500}
    ctx_big = _record_with(oversized)
    assert _stored(ctx_big).evidence == {}
    many = {f"k{i}": i for i in range(40)}
    with pytest.raises(ValueError):
        _record_with(many)
    mutable: dict = {"run_index": 1}
    ctx = _record_with(mutable)
    mutable["run_index"] = 2  # caller mutation must not leak into storage
    assert _stored(ctx).evidence == {"run_index": 1}


def test_event_detail_uses_the_same_allowlist():
    ctx = FakeCtx()
    evidence.record_event(ctx, "consent.recorded", {"consent": "full text"})  # type: ignore[arg-type]
    assert evidence.list_events(ctx) == (("consent.recorded", {}),)  # type: ignore[arg-type]
    evidence.reset_stores()
    evidence.record_event(  # type: ignore[arg-type]
        ctx, "consent.recorded", {"reason": "from 10.0.0.8"}
    )
    ((name, detail),) = evidence.list_events(ctx)  # type: ignore[arg-type]
    assert (name, detail) == ("consent.recorded", {})
    evidence.reset_stores()
    evidence.record_event(ctx, "consent.recorded", {"stage": "locate"})  # type: ignore[arg-type]
    assert evidence.list_events(ctx) == (("consent.recorded", {"stage": "locate"}),)  # type: ignore[arg-type]
    evidence.reset_stores()
    evidence.record_event(  # type: ignore[arg-type]
        ctx, "readback.counted", {"stage": "locate", "count": 2}
    )
    assert evidence.list_events(ctx) == (  # type: ignore[arg-type]
        ("readback.counted", {"stage": "locate", "count": 2}),
    )
    # The receipt funnel event keeps its declared stage/status detail.
    call_ctx = _record_with({"run_index": 1})
    assert evidence.list_events(call_ctx) == (  # type: ignore[arg-type]
        ("model.call_recorded", {"stage": "extract", "status": "succeeded"}),
    )


def test_model_allowlist_is_closed():
    # Independently written: an accidental extra model id must fail here.
    # muse-spark-1.3 (the id spark.py/readback.py emit)
    # meta-files (the clip-upload receipt stage) and spark-open (the background
    # second check on Baseten) are deliberate.
    assert evidence.MODEL_ALLOWLIST == frozenset(
        {"spark-1.3", "muse-spark-1.3", "sam-3.1", "meta-files", "spark-open"}
    )


def test_name_shaped_model_id_rejected_on_write():
    # Isolated: every other field is valid, only the model is unknown.
    with pytest.raises(ValueError, match="model"):
        _record_with({"run_index": 1}, model="Jane-Doe")


def test_name_shaped_model_id_dropped_on_read(caplog):
    # Isolated: the adapter row is valid except for the model id.
    bad = _direct_record("job-own", model="Jane-Doe")
    good = _direct_record("job-own")
    ctx = _ctx_with_rows("job-own", [bad, good])
    with caplog.at_level("WARNING", logger="app.evidence"):
        calls = evidence.list_calls(ctx)  # type: ignore[arg-type]
    assert len(calls) == 1
    assert calls[0].model == "spark-1.3"
    assert "dropped 1" in caplog.text
    assert "Jane-Doe" not in caplog.text


def test_stage_model_endpoint_status_cannot_carry_private_data():
    for over in (
        {"stage": "10.0.0.8"},
        {"stage": "I, Jane Doe, agree to recording"},
        {"stage": "exfiltrate"},
        {"model": "Jane Doe"},
        {"model": "10.0.0.8"},
        {"model": "Bearer [REDACTED]"},
        {"endpoint": "/var/data/clip.mp4"},
        {"endpoint": "clip.mp4"},
        {"endpoint": "I, Jane Doe, agree to recording"},
        {"status": "Bearer [REDACTED]"},
        {"status": "ok-ish"},
    ):
        with pytest.raises(ValueError):
            _record_with({"phrase": "fan"}, **over)


def test_returned_records_cannot_mutate_the_store():
    ctx = _record_with({"run_index": 1})
    (found,) = evidence.list_calls(ctx)  # type: ignore[arg-type]
    found.evidence["run_index"] = 999  # type: ignore[index]
    assert evidence.list_calls(ctx)[0].evidence["run_index"] == 1  # type: ignore[arg-type]
    ectx = FakeCtx(JobId("job-evt-mut"))
    evidence.record_event(ectx, "job.queued", {"stage": "locate"})  # type: ignore[arg-type]
    ((_, detail),) = evidence.list_events(ectx)  # type: ignore[arg-type]
    detail["stage"] = "pwned"
    ((_, again),) = evidence.list_events(ectx)  # type: ignore[arg-type]
    assert again == {"stage": "locate"}
    store = evidence.InMemoryEvidenceStore()
    record = CallRecord(
        job_id=JobId("job-direct"),
        stage="locate",
        model="sam-3.1",
        endpoint="/locate",
        latency_ms=10,
        units_in=None,
        units_out=None,
        estimated_cost_usd=Decimal("0.01"),
        status="succeeded",
        evidence={"phrase": "fan", "score": 0.66},
    )
    store.save_call(record)
    record.evidence["phrase"] = "mutated"  # type: ignore[index]
    assert store.calls_for("job-direct")[0].evidence["phrase"] == "fan"


def _aliasing_ctx(job_id):
    """Exact owned store whose readers hand out live objects without copying.

    Instance patches only: the store keeps its exact type, so get_store
    accepts it, but the read path sees uncopied rows, as from a hostile
    adapter.
    """
    store = evidence.InMemoryEvidenceStore()
    job = str(job_id)
    store.calls_for = lambda _job: tuple(store._calls.get(job, {}).values())  # type: ignore[method-assign]
    store.events_for = lambda _job: tuple(store._events.get(job, ()))  # type: ignore[method-assign]
    return FakeCtx(JobId(job), evidence_store=store)


def test_attached_aliasing_store_cannot_leak_mutations():
    ctx = _aliasing_ctx("job-alias")
    evidence.record_call(  # type: ignore[arg-type]
        ctx,
        CallRecord(
            job_id=ctx.job_id,
            stage="locate",
            model="sam-3.1",
            endpoint="/locate",
            latency_ms=4300,
            units_in=1,
            units_out=1,
            estimated_cost_usd=Decimal("0.0025"),
            status="succeeded",
            evidence={"phrase": "fan", "score": 0.66},
        ),
    )
    (found,) = evidence.list_calls(ctx)  # type: ignore[arg-type]
    found.evidence["phrase"] = "Jane Doe PXL_001.MP4"  # type: ignore[index]
    assert evidence.list_calls(ctx)[0].evidence == {"phrase": "fan", "score": 0.66}  # type: ignore[arg-type]
    evidence.record_event(ctx, "job.queued", {"stage": "locate"})  # type: ignore[arg-type]
    ((_, detail),) = evidence.list_events(ctx)[-1:]  # type: ignore[arg-type]
    detail["stage"] = "pwned"
    assert evidence.list_events(ctx)[-1][1] == {"stage": "locate"}  # type: ignore[arg-type]


def test_unvalidated_row_with_junk_phrase_never_renders():
    # Every writable store entry point now enforces the allowlists, so a
    # corrupt row can only come from test-only code: the read-back gate
    # must still refuse its phrase.
    ctx = _ctx_with_rows(
        "job-oob",
        [
            _direct_record(
                "job-oob",
                evidence={"phrase": "Jane Doe", "decision": "yes", "score": 0.66},
            )
        ],
    )
    calls = evidence.list_calls(ctx)  # type: ignore[arg-type]
    assert evidence.select_sample_readback(calls) is None
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[resolve_job_context] = lambda: ctx
    body = TestClient(app).get(f"/calls/{ctx.job_id}").text
    assert "Jane Doe" not in body
    assert "No measured read-back for this run" in body


def _ctx_with_rows(job_id, rows):
    """Exact owned store whose reader returns the given rows for any job.

    Instance patch only: the store keeps its exact type, so get_store
    accepts it, but the read path sees unfiltered adapter rows.
    """
    store = evidence.InMemoryEvidenceStore()
    store.calls_for = lambda _job: tuple(rows)  # type: ignore[method-assign]
    return FakeCtx(JobId(job_id), evidence_store=store)


def _direct_record(job_id, **over):
    base = dict(
        job_id=JobId(job_id),
        stage="readback",
        model="spark-1.3",
        endpoint="/readback",
        latency_ms=5400,
        units_in=1,
        units_out=1,
        estimated_cost_usd=Decimal("0.0038"),
        status="succeeded",
        evidence={"phrase": "fan", "decision": "yes", "score": 0.66},
    )
    base.update(over)
    return CallRecord(**base)


def test_list_calls_drops_foreign_job_rows_and_logs_only_a_count(caplog):
    own = _direct_record("job-own")
    foreign = _direct_record(
        "job-foreign",
        latency_ms=9999,
        evidence={"phrase": "battery", "decision": "yes", "score": 0.9},
    )
    ctx = _ctx_with_rows("job-own", [own, foreign])
    with caplog.at_level("WARNING", logger="app.evidence"):
        (found,) = evidence.list_calls(ctx)  # type: ignore[arg-type]
    assert found.job_id == "job-own"
    assert "dropped 1" in caplog.text
    assert "job-foreign" not in caplog.text
    assert "9999" not in caplog.text
    assert "battery" not in caplog.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "Jane Doe"),
        ("endpoint", "/private/clip.mp4"),
        ("stage", "exfiltrate"),
        ("status", "leaked"),
    ],
)
def test_adapter_returned_unsafe_fields_are_dropped(field, value, caplog):
    bad = _direct_record("job-own", **{field: value})
    good = _direct_record("job-own")
    ctx = _ctx_with_rows("job-own", [bad, good])
    with caplog.at_level("WARNING", logger="app.evidence"):
        calls = evidence.list_calls(ctx)  # type: ignore[arg-type]
    assert len(calls) == 1
    assert calls[0].model == "spark-1.3"
    assert "dropped 1" in caplog.text
    assert value not in caplog.text


def test_read_path_resanitizes_out_of_band_evidence():
    # The corrupt row is built in test-only code: store writes sanitize
    # first, so only the read path can meet the extra key.
    ctx = _ctx_with_rows(
        "job-oob-ev",
        [
            _direct_record(
                "job-oob-ev",
                evidence={"phrase": "fan", "decision": "yes", "leak": "x"},
            )
        ],
    )
    (found,) = evidence.list_calls(ctx)  # type: ignore[arg-type]
    assert found.evidence == {"phrase": "fan", "decision": "yes"}


def test_huge_shapes_rejected_before_copy(monkeypatch):
    def _no_copy(*args, **kwargs):
        raise AssertionError("deepcopy must not run for oversized shapes")

    monkeypatch.setattr(evidence.copy, "deepcopy", _no_copy)
    with pytest.raises(ValueError, match="too many values"):
        evidence.sanitize_evidence({"items": list(range(100_000))})
    with pytest.raises(ValueError, match="text exceeds bound"):
        evidence.sanitize_evidence({"blob": "z" * 1_000_000})
