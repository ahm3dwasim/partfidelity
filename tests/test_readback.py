"""Read-back tests: provisional use, order, budget, cutoff config."""

from __future__ import annotations

import inspect
from decimal import Decimal
from pathlib import Path

import pytest

from app import readback
from app.contracts import (
    READBACK_CONFIDENCE_CUTOFF,
    READBACK_COST_PER_CALL_USD,
    ArtifactId,
    Detection,
    JobId,
    LabelSlot,
    PartRequest,
    ProvisionalRender,
)
from app.readback import (
    ReadbackError,
    ReadbackResult,
    eligible_for_readback,
    read_back_label,
    run_readbacks,
)

from test_spark_schema import RecordingContext


class ScriptedReadbackClient:
    """Fake Spark read-back client; invocations append to the shared event log."""

    def __init__(self, log, answers):
        self._log = log
        self._answers = list(answers)
        self.calls = []

    def read_back(self, image_path, phrase, position_hint):
        self.calls.append((image_path, phrase, position_hint))
        self._log.append(("invoke", "read_back", phrase))
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _yes(cost="0.004", latency=5000):
    return ReadbackResult(answer="yes", cost_usd=Decimal(cost), latency_ms=latency)


def _no(cost="0.004", latency=5000):
    return ReadbackResult(answer="no", cost_usd=Decimal(cost), latency_ms=latency)


def make_provisional(phrase="fan", score=0.65, hint="motor housing", tag="p1"):
    return ProvisionalRender(
        detection=Detection(
            phrase=phrase, mask_path=Path(f"{tag}.png"), score=score, box_xyxy=(1, 2, 3, 4)
        ),
        artifact_id=ArtifactId(f"art-{tag}"),
        rendered_image_path=Path(f"/jobs/j1/{tag}.png"),
        position_hint=hint,
    )


def make_slot(step_index=0, part_index=0, phrase="fan", hint="motor housing"):
    return LabelSlot(
        step_index=step_index,
        part_index=part_index,
        part=PartRequest(phrase=phrase, position_hint=hint),
    )


def slots_for(provisionals):
    """Parallel slots with distinct part_index values, one per provisional."""
    return [
        make_slot(0, i, p.detection.phrase, p.position_hint)
        for i, p in enumerate(provisionals)
    ]


def test_readback_uses_complete_provisional_render():
    sig = inspect.signature(read_back_label)
    params = list(sig.parameters)
    assert params[:2] == ["ctx", "provisional"]  # whole object in, path read internally
    assert params == ["ctx", "provisional", "slot", "client"]  # contract signature
    assert sig.parameters["slot"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["client"].kind is inspect.Parameter.KEYWORD_ONLY
    log: list = []
    ctx = RecordingContext(log)
    provisional = make_provisional()
    slot = make_slot()
    client = ScriptedReadbackClient(log, [_yes()])
    decision = read_back_label(ctx, provisional, slot=slot, client=client)
    assert client.calls == [
        (provisional.rendered_image_path, "fan", "motor housing")
    ]
    assert decision.provisional is provisional
    assert decision.slot is slot
    assert decision.readback == "yes"
    assert decision.status == "part_pixels_unchanged"
    with pytest.raises(ValueError):
        read_back_label(ctx, {"not": "a render"}, slot=slot, client=client)  # type: ignore[arg-type]


def test_readback_reservation_happens_before_client_invocation():
    log: list = []
    ctx = RecordingContext(log)
    client = ScriptedReadbackClient(log, [_yes()])
    read_back_label(ctx, make_provisional(), slot=make_slot(), client=client)
    assert [e[0] for e in log] == ["reserve", "invoke", "settle"]
    kind, amount = log[0][1], log[0][2]
    assert (kind, amount) == ("spark_readback", READBACK_COST_PER_CALL_USD)
    assert log[2][1:] == ("spark_readback", Decimal("0.004"), "succeeded")


def test_readback_no_tombstones_provisional_immediately():
    log: list = []
    ctx = RecordingContext(log)
    provisional = make_provisional()
    slot = make_slot(0, 1)
    client = ScriptedReadbackClient(log, [_no()])
    decision = read_back_label(ctx, provisional, slot=slot, client=client)
    assert decision.readback == "no"
    assert decision.status == "not_checked"
    assert decision.checked is False
    assert decision.slot is slot  # removed label keeps its slot
    assert decision.provisional is None  # label removed
    assert decision.detection is provisional.detection  # part named for the row
    assert ctx.artifacts.deleted == [provisional.artifact_id]  # tombstoned before return
    assert ("mark_deleted", provisional.artifact_id) in log


def test_readback_tombstone_failure_finishes_job_failed():
    log: list = []
    ctx = RecordingContext(log)

    def boom(artifact_id):
        raise RuntimeError("tombstone exploded")

    ctx.artifacts.mark_deleted = boom  # type: ignore[method-assign]
    client = ScriptedReadbackClient(log, [_no(cost="0.004")])
    with pytest.raises(RuntimeError, match="tombstone exploded"):
        read_back_label(ctx, make_provisional(), slot=make_slot(), client=client)
    assert ctx.holds[0].settlements == [(Decimal("0.004"), "succeeded")]  # paid call settled
    assert ctx.finishes == ["failed"]  # provisional never left live on an active job
    assert log[-1] == ("finish", "failed")


def test_readback_exception_finishes_job_failed():
    log: list = []
    ctx = RecordingContext(log)
    boom = RuntimeError("read-back exploded")
    client = ScriptedReadbackClient(log, [boom])
    with pytest.raises(RuntimeError, match="read-back exploded"):
        read_back_label(ctx, make_provisional(), slot=make_slot(), client=client)
    assert ctx.finishes == ["failed"]
    assert log[-1] == ("finish", "failed")


def test_readback_nonfinite_cost_settles_failed_once():
    for bad in ("NaN", "Infinity"):
        log: list = []
        ctx = RecordingContext(log)
        answer = ReadbackResult(answer="yes", cost_usd=Decimal(bad), latency_ms=5)
        with pytest.raises(ReadbackError):
            read_back_label(
                ctx, make_provisional(), slot=make_slot(),
                client=ScriptedReadbackClient(log, [answer]),
            )
        assert ctx.holds[0].settlements == [(Decimal("0"), "failed")]  # exactly once
        assert ctx.finishes == ["failed"]


def test_readback_client_exception_settles_hold_once_as_failed():
    log: list = []
    ctx = RecordingContext(log)
    client = ScriptedReadbackClient(log, [RuntimeError("no")])
    with pytest.raises(RuntimeError):
        read_back_label(ctx, make_provisional(), slot=make_slot(), client=client)
    assert len(ctx.holds) == 1
    assert ctx.holds[0].settlements == [(Decimal("0"), "failed")]  # exactly once
    assert ctx.finishes == ["failed"]


def test_readback_cap_and_reservation_failure_mean_not_checked():
    log: list = []
    ctx = RecordingContext(log)
    labels = [make_provisional(phrase="fan", score=s, tag=f"f{i}") for i, s in enumerate([0.61, 0.63, 0.66, 0.68])]
    slots = slots_for(labels)
    client = ScriptedReadbackClient(log, [_yes()] * 4)
    decisions = run_readbacks(ctx, labels, slots=slots, client=client)
    assert len(client.calls) == 3  # cap: fourth eligible label never fires
    assert [d.readback for d in decisions] == ["yes", "yes", "yes", "not_run"]
    assert decisions[3].status == "not_checked"
    assert decisions[3].detection is labels[3].detection
    assert [d.slot for d in decisions] == slots  # every path keeps its slot

    log2: list = []
    ctx2 = RecordingContext(log2, fail_reserve_at=1)
    client2 = ScriptedReadbackClient(log2, [_yes()])
    refused_slot = make_slot(1, 0)
    (refused,) = run_readbacks(
        ctx2, [make_provisional(tag="r")], slots=[refused_slot], client=client2
    )
    assert client2.calls == []  # refusal means no client invocation
    assert refused.readback == "not_run"
    assert refused.status == "not_checked"
    assert refused.slot is refused_slot


def test_readback_picks_lowest_c_first_and_keeps_above_cutoff():
    log: list = []
    ctx = RecordingContext(log)
    labels = [
        make_provisional(phrase="fan", score=0.69, tag="hi"),
        make_provisional(phrase="battery", score=0.80, tag="kept"),
        make_provisional(phrase="battery", score=0.55, tag="lo"),
        make_provisional(phrase="screen", score=0.65, tag="unknown"),
        make_provisional(phrase="fan", score=0.40, tag="below"),
    ]
    slots = slots_for(labels)
    client = ScriptedReadbackClient(log, [_yes(), _yes()])
    decisions = run_readbacks(ctx, labels, slots=slots, client=client)
    assert [c[1] for c in client.calls] == ["battery", "fan"]  # 0.55 before 0.69
    assert [d.slot for d in decisions] == slots  # firing order never moves a slot
    assert decisions[0].readback == "yes"
    kept = decisions[1]
    assert (kept.readback, kept.status, kept.checked) == ("not_run", "part_pixels_unchanged", True)
    assert decisions[2].readback == "yes"
    assert decisions[3].readback == "not_run"  # unknown part: never shown
    assert decisions[4].readback == "not_run"  # below part threshold


def test_cutoff_is_config_not_sample_mutation():
    assert READBACK_CONFIDENCE_CUTOFF == 0.70
    assert readback.READBACK_CONFIDENCE_CUTOFF is READBACK_CONFIDENCE_CUTOFF
    assert eligible_for_readback(make_provisional(phrase="fan", score=0.69)) is True
    assert eligible_for_readback(make_provisional(phrase="fan", score=0.70)) is False
    provisional = make_provisional(phrase="fan", score=0.75)
    assert eligible_for_readback(provisional) is False


def test_cutoff_moves_only_through_config(monkeypatch):
    provisional = make_provisional(phrase="fan", score=0.75)
    monkeypatch.setattr(readback, "READBACK_CONFIDENCE_CUTOFF", 0.99)
    assert eligible_for_readback(provisional) is True  # the band follows the config value
    monkeypatch.setattr(readback, "READBACK_CONFIDENCE_CUTOFF", 0.70)


def test_readback_logs_phrase_decision_latency_cost_badge_and_count():
    log: list = []
    ctx = RecordingContext(log)
    calls, events = [], []
    labels = [make_provisional(tag="first"), make_provisional(phrase="fan", score=0.80, tag="second")]
    slots = slots_for(labels)
    decisions = run_readbacks(
        ctx,
        labels,
        slots=slots,
        client=ScriptedReadbackClient(log, [_yes(cost="0.004", latency=5100)]),
        record_call=lambda c, r: calls.append(r),
        record_event=lambda c, e, d: events.append((e, d)),
    )
    assert decisions[0].readback == "yes"
    assert [d.slot for d in decisions] == slots
    (record,) = calls
    assert record.job_id == JobId(ctx.job_id)
    assert (record.stage, record.model, record.endpoint) == ("readback", "muse-spark-1.3", "read_back")
    assert (record.latency_ms, record.estimated_cost_usd) == (5100, Decimal("0.004"))
    assert record.evidence["readback_fired_n"] == 1
    fired, skipped = events
    assert fired[0] == "readback"
    assert fired[1]["phrase"] == "fan"
    assert fired[1]["decision"] == "yes"
    assert fired[1]["latency_ms"] == 5100
    assert fired[1]["cost_usd"] == "0.004"
    assert fired[1]["badge_changed"] is True
    assert fired[1]["readback_fired_n"] == 1
    assert skipped[1]["decision"] == "not_run"
    assert skipped[1]["badge_changed"] is False


def test_readback_preserves_slot_on_yes_no_and_not_run():
    log: list = []
    ctx = RecordingContext(log)
    yes_slot = make_slot(0, 0, "fan")
    yes_decision = read_back_label(
        ctx, make_provisional(), slot=yes_slot,
        client=ScriptedReadbackClient(log, [_yes()]),
    )
    assert yes_decision.readback == "yes"
    assert yes_decision.slot is yes_slot
    no_slot = make_slot(0, 1, "fan")  # same phrase, distinct slot
    no_decision = read_back_label(
        ctx, make_provisional(tag="p2"), slot=no_slot,
        client=ScriptedReadbackClient(log, [_no()]),
    )
    assert no_decision.readback == "no"
    assert no_decision.slot is no_slot
    assert yes_decision.slot != no_decision.slot

    refused_log: list = []
    refused_ctx = RecordingContext(refused_log, fail_reserve_at=1)
    refused_slot = make_slot(1, 0, "battery")
    (refused,) = run_readbacks(
        refused_ctx,
        [make_provisional(phrase="battery", score=0.55, tag="r")],
        slots=[refused_slot],
        client=ScriptedReadbackClient(refused_log, [_yes()]),
    )
    assert refused.readback == "not_run"  # reservation refusal fires nothing
    assert refused.slot is refused_slot


def test_run_readbacks_keeps_distinct_slots_for_same_phrase_parts():
    log: list = []
    ctx = RecordingContext(log)
    labels = [
        make_provisional(phrase="fan", score=0.61, tag="a"),
        make_provisional(phrase="fan", score=0.63, tag="b"),
    ]
    slots = [make_slot(0, 0, "fan"), make_slot(0, 1, "fan")]
    decisions = run_readbacks(
        ctx, labels, slots=slots,
        client=ScriptedReadbackClient(log, [_yes(), _yes()]),
    )
    assert [d.readback for d in decisions] == ["yes", "yes"]
    assert decisions[0].slot is slots[0]
    assert decisions[1].slot is slots[1]
    assert decisions[0].slot != decisions[1].slot


def test_run_readbacks_rejects_mismatched_slots():
    log: list = []
    with pytest.raises(ValueError, match="parallel"):
        run_readbacks(
            RecordingContext(log), [make_provisional()], slots=[],
            client=ScriptedReadbackClient(log, []),
        )


class _SettleFailingContext(RecordingContext):
    """RecordingContext whose holds raise on settle (settlement outage)."""

    def reserve(self, kind, worst_case_usd):
        hold = super().reserve(kind, worst_case_usd)

        def _boom(actual_usd, outcome):
            raise RuntimeError("settle exploded")

        hold.settle = _boom  # type: ignore[method-assign]
        return hold


def test_client_exception_with_settlement_error_still_finishes_failed():
    log: list = []
    ctx = _SettleFailingContext(log)
    boom = RuntimeError("provider exploded")
    with pytest.raises(RuntimeError) as caught:
        read_back_label(
            ctx, make_provisional(), slot=make_slot(),
            client=ScriptedReadbackClient(log, [boom]),
        )
    assert caught.value is boom  # original preserved, never mutated or chained
    assert ctx.finishes == ["failed"]
    assert log[-1] == ("finish", "failed")


def test_invalid_result_with_settlement_error_still_finishes_failed():
    log: list = []
    ctx = _SettleFailingContext(log)
    bad = ReadbackResult(answer="maybe", cost_usd=Decimal("0.004"), latency_ms=5)
    with pytest.raises(ReadbackError):
        read_back_label(
            ctx, make_provisional(), slot=make_slot(),
            client=ScriptedReadbackClient(log, [bad]),
        )
    assert ctx.finishes == ["failed"]
    assert log[-1] == ("finish", "failed")
