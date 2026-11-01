"""Spark extraction tests: reservation order, strict schema, settlement."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app import spark
from app.contracts import SPARK_3RUNS_RESERVATION_USD
from app.spark import (
    NotATeardown,
    SparkClientError,
    SparkRunResult,
    SparkSchemaError,
    extract_steps,
    validate_spark_payload,
)


class ReserveRefused(Exception):
    pass


class RecordingHold:
    def __init__(self, log, kind, amount):
        self._log = log
        self.kind = kind
        self.amount = amount
        self.settlements = []

    def settle(self, actual_usd, outcome):
        self.settlements.append((actual_usd, outcome))
        self._log.append(("settle", self.kind, actual_usd, outcome))


class RecordingArtifacts:
    def __init__(self, log):
        self._log = log
        self.deleted = []

    def register(self, kind, path):
        raise AssertionError("extract_steps registers nothing")

    def mark_kept(self, artifact_id):
        raise AssertionError("extract_steps keeps nothing")

    def mark_deleted(self, artifact_id):
        self.deleted.append(artifact_id)
        self._log.append(("mark_deleted", artifact_id))


class RecordingContext:
    """Fake JobContext; every reserve/settle lands in one shared event log."""

    def __init__(self, log, *, fail_reserve_at=None, job_id="job-l2-test"):
        self._log = log
        self._fail_at = fail_reserve_at
        self._job_id = job_id
        self._reserves = 0
        self.holds = []
        self.finishes = []
        self.artifacts = RecordingArtifacts(log)

    @property
    def job_id(self):
        return self._job_id  # type: ignore[return-value]

    def reserve(self, kind, worst_case_usd):
        self._reserves += 1
        self._log.append(("reserve", kind, worst_case_usd))
        if self._fail_at is not None and self._reserves >= self._fail_at:
            raise ReserveRefused(f"refused reserve #{self._reserves}")
        hold = RecordingHold(self._log, kind, worst_case_usd)
        self.holds.append(hold)
        return hold

    def finish(self, outcome, *, keep_artifact_ids=(), failed_output_names=None):
        self.finishes.append(outcome)
        self._log.append(("finish", outcome))


class ScriptedSparkClient:
    """Fake Spark step client; invocations append to the shared event log."""

    def __init__(self, log, results):
        self._log = log
        self._results = list(results)
        self.calls = []
        self.durations: list = []

    def run_steps(self, video_file_id, clip_duration_s=None, scene_cuts=None, clocked=False):
        self.calls.append(video_file_id)
        self.durations.append(clip_duration_s)
        self._log.append(("invoke", "run_steps", video_file_id))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_step(number=1, start=10.0, end=20.0, action="Remove the rear screws",
              parts=("battery",), tools=("Phillips #0",), hazards=(),
              kind="repair"):
    step = {
        "number": number,
        "start_s": start,
        "end_s": end,
        "action": action,
        "tools": list(tools),
        "parts": [{"phrase": p, "position_hint": f"{p} area"} for p in parts],
        "hazards": list(hazards),
    }
    if kind is not None:
        step["kind"] = kind
    return step


def make_payload(steps=(None,), device="Cordless hand vacuum", confidence=0.82):
    steps = [make_step() if s is None else s for s in steps]
    for i, step in enumerate(steps, start=1):
        step["number"] = i
    return {"device_guess": device, "confidence": confidence, "steps": steps}


def make_result(payload, cost="0.08", latency=1200):
    return SparkRunResult(payload=payload, cost_usd=Decimal(cost), latency_ms=latency)


def test_spark_reservation_happens_before_client_invocation():
    log: list = []
    ctx = RecordingContext(log)
    client = ScriptedSparkClient(log, [make_result(make_payload()) for _ in range(3)])
    bundle = extract_steps(ctx, "vid-1", client=client)
    assert bundle.source_runs == 3
    assert [e[0] for e in log] == [
        "reserve", "invoke", "settle",
        "reserve", "invoke", "settle",
        "reserve", "invoke", "settle",
    ]
    reserves = [e for e in log if e[0] == "reserve"]
    assert [r[1] for r in reserves] == ["spark_steps"] * 3
    assert sum((r[2] for r in reserves), Decimal("0")) == SPARK_3RUNS_RESERVATION_USD
    assert sum(spark.SPARK_RUN_SHARES, Decimal("0")) == SPARK_3RUNS_RESERVATION_USD
    assert client.calls == ["vid-1"] * 3  # client's own record is the awaited side effect


def test_spark_reserve_refusal_raises_before_client_invocation():
    log: list = []
    ctx = RecordingContext(log, fail_reserve_at=2)
    client = ScriptedSparkClient(log, [make_result(make_payload()) for _ in range(3)])
    with pytest.raises(ReserveRefused):
        extract_steps(ctx, "vid-1", client=client)
    assert client.calls == ["vid-1"]  # second run never invoked; run count never reduced
    assert [e[0] for e in log] == ["reserve", "invoke", "settle", "reserve"]
    assert ctx.holds[0].settlements == [(Decimal("0.08"), "succeeded")]


def test_extract_steps_client_exception_settles_hold_once_as_failed():
    log: list = []
    ctx = RecordingContext(log)
    boom = RuntimeError("spark exploded")
    client = ScriptedSparkClient(
        log, [make_result(make_payload()), boom, make_result(make_payload())]
    )
    with pytest.raises(RuntimeError, match="spark exploded"):
        extract_steps(ctx, "vid-1", client=client)
    assert client.calls == ["vid-1", "vid-1"]  # failed run invoked once, then stop
    assert ctx.holds[0].settlements == [(Decimal("0.08"), "succeeded")]
    assert ctx.holds[1].settlements == [(Decimal("0"), "failed")]  # exactly one failed settlement
    assert len(ctx.holds) == 2  # no third reserve after the failure


def test_extract_steps_bad_result_settles_failed_once():
    log: list = []
    ctx = RecordingContext(log)
    bad = SparkRunResult(payload={}, cost_usd=Decimal("-1"), latency_ms=5)
    client = ScriptedSparkClient(log, [bad])
    with pytest.raises(SparkClientError):
        extract_steps(ctx, "vid-1", client=client)
    assert ctx.holds[0].settlements == [(Decimal("0"), "failed")]


def test_schema_max_twelve_steps_three_parts():
    ok_steps = tuple(
        make_step(start=float(i * 10), end=float(i * 10 + 5), parts=("fan", "battery", "fan"))
        for i in range(12)
    )
    run = validate_spark_payload(make_payload(steps=ok_steps))
    assert len(run.steps) == 12
    assert len(run.steps[0].parts) == 3
    with pytest.raises(SparkSchemaError):
        validate_spark_payload(make_payload(steps=tuple(make_step() for _ in range(13))))
    with pytest.raises(SparkSchemaError):
        validate_spark_payload(make_payload(steps=(make_step(parts=("a", "b", "c", "d")),)))


def test_schema_rejects_unknown_keys_bad_numbers_and_gaps():
    base = make_payload()
    with pytest.raises(SparkSchemaError):  # unknown top-level key
        validate_spark_payload({**base, "summary": "nope"})
    with pytest.raises(SparkSchemaError):  # unknown step key
        validate_spark_payload(make_payload(steps=({**make_step(), "note": 1},)))
    with pytest.raises(SparkSchemaError):  # start >= end
        validate_spark_payload(make_payload(steps=(make_step(start=20.0, end=20.0),)))
    with pytest.raises(SparkSchemaError):  # confidence out of range
        validate_spark_payload(make_payload(confidence=1.5))
    with pytest.raises(SparkSchemaError):  # non-1..n numbering
        dup = dict(make_payload())
        dup["steps"] = [make_step(number=1), make_step(number=1)]
        validate_spark_payload(dup)
    with pytest.raises(SparkSchemaError):  # empty action
        validate_spark_payload(make_payload(steps=(make_step(action="  "),)))


def test_extract_steps_nonfinite_cost_settles_failed_once():
    for bad in ("NaN", "Infinity"):
        log: list = []
        ctx = RecordingContext(log)
        result = SparkRunResult(payload=make_payload(), cost_usd=Decimal(bad), latency_ms=5)
        with pytest.raises(SparkClientError):
            extract_steps(ctx, "vid-1", client=ScriptedSparkClient(log, [result]))
        assert ctx.holds[0].settlements == [(Decimal("0"), "failed")]  # exactly once


def test_zero_consensus_steps_refuses_not_a_teardown():
    log: list = []
    ctx = RecordingContext(log)
    empty = make_payload(steps=(), confidence=0.9)
    client = ScriptedSparkClient(log, [make_result(empty) for _ in range(3)])
    with pytest.raises(NotATeardown) as exc:
        extract_steps(ctx, "vid-1", client=client)
    assert exc.value.refusal.reason == "zero_steps"
    assert exc.value.refusal.verdict == "not_a_teardown"
