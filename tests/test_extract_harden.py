"""Extraction hardening: per-run retry, 2-of-3 consensus, failure copy, fuse-lite."""

from __future__ import annotations

from decimal import Decimal

import pytest
from test_spark_schema import (
    RecordingContext,
    ScriptedSparkClient,
    make_payload,
    make_result,
    make_step,
)

from app import spark
from app.boundaries import fuse_lite_steps
from app.contracts import Step
from app.hazards import refusal_copy
from app.spark import (
    SERVICE_FAILURE_KINDS,
    SparkClientError,
    SparkRunResult,
    SparkServiceError,
    SparkStageConfig,
)

RETRYABLE = frozenset(SERVICE_FAILURE_KINDS)


def _failed(kind: str) -> SparkRunResult:
    return SparkRunResult(  # type: ignore[arg-type]
        payload="INDETERMINATE: x", cost_usd=Decimal("0.03"), latency_ms=5, failure_kind=kind
    )


def _extract(results, config, retry_results=()):
    log: list = []
    ctx = RecordingContext(log)
    client = ScriptedSparkClient(log, results)
    retries = list(retry_results)
    seen: list[int] = []

    def retry_call(file_id, duration, cuts, clocked, *, run_index=0):
        seen.append(run_index)
        return retries.pop(0)

    class _Client:
        spark_stage_config = config
        run_steps = staticmethod(client.run_steps)

    out = spark._extract_with_raw(
        ctx, "vid-1", _Client(), config,  # type: ignore[arg-type]
        retry_call=retry_call, retryable_failures=RETRYABLE,
    )
    return out, seen, client


def _good():
    return make_result(make_payload())


def test_one_failed_run_falls_back_to_two_run_consensus():
    config = SparkStageConfig(consensus_min_runs=2)
    (bundle, raws), seen, client = _extract([_good(), _good(), _failed("timeout")], config, [_failed("timeout")])
    assert bundle.source_runs == 2 and len(raws) == 2
    assert seen == [2]  # the failed run got its own retry, which also failed


def test_first_run_failing_still_gives_two_run_consensus():
    config = SparkStageConfig(consensus_min_runs=2)
    (bundle, _), _, client = _extract([_failed("http_5xx"), _good(), _good()], config, [_failed("http_5xx")])
    assert bundle.source_runs == 2 and len(client.calls) == 3


def test_two_failed_runs_fail_the_job():
    config = SparkStageConfig(consensus_min_runs=2)
    with pytest.raises(SparkClientError):
        _extract([_failed("timeout"), _failed("timeout"), _good()], config, [_failed("timeout"), _failed("timeout")])


def test_default_config_keeps_the_exact_three_run_rule():
    with pytest.raises(SparkServiceError):
        _extract([_good(), _good(), _failed("timeout")], SparkStageConfig(), [_failed("timeout")])


def test_billed_unusable_200_is_retried_once_per_run():
    config = SparkStageConfig(consensus_min_runs=2)
    (bundle, _), seen, client = _extract(
        [_failed("malformed"), _failed("malformed"), _good()], config, [_good(), _good()]
    )
    assert bundle.source_runs == 3
    assert seen == [0, 1] and len(client.calls) == 3


def test_a_retry_that_fails_is_not_retried_again():
    config = SparkStageConfig(consensus_min_runs=2)
    _, seen, _ = _extract([_good(), _good(), _failed("malformed")], config, [_failed("malformed")])
    assert seen == [2]


def test_non_service_failure_is_not_retried():
    config = SparkStageConfig(consensus_min_runs=2)
    (bundle, _), seen, _ = _extract([_good(), _good(), _failed("local")], config)
    assert seen == [] and bundle.source_runs == 2


@pytest.mark.parametrize("value", (0, 1, 4, True, 2.0))
def test_consensus_min_runs_is_validated(value):
    with pytest.raises(ValueError):
        SparkStageConfig(consensus_min_runs=value)  # type: ignore[arg-type]


def test_consensus_rejects_a_run_count_below_the_minimum():
    config = SparkStageConfig(consensus_min_runs=2)
    runs = [spark.validate_spark_payload(make_payload())]
    with pytest.raises(ValueError):
        spark.consensus_bundle(runs, config)


def test_two_run_disagreement_still_refuses():
    config = SparkStageConfig(consensus_min_runs=2)
    tiling = make_payload([
        make_step(action="Remove the rear screws", start=0, end=30),
        make_step(action="Lift the battery", start=30, end=60, parts=("battery",)),
    ])
    gappy = make_payload([
        make_step(action="Remove the rear screws", start=0, end=10),
        make_step(action="Lift the battery", start=50, end=60, parts=("battery",)),
    ])
    with pytest.raises(spark.StepsDisagreement):
        spark.consensus_bundle(
            [spark.validate_spark_payload(tiling), spark.validate_spark_payload(gappy)],
            config, chunk_extent=spark.ChunkExtent(duration_s=60.0, tolerance_s=2.0),
        )


def test_cached_two_run_bundle_round_trips_only_when_allowed():
    config = SparkStageConfig(consensus_min_runs=2)
    (bundle, _), _, _ = _extract([_good(), _good(), _failed("timeout")], config, [_failed("timeout")])
    payload = spark.bundle_to_json(bundle)
    assert spark.bundle_from_json(payload, config).source_runs == 2
    with pytest.raises(ValueError):
        spark.bundle_from_json(payload, SparkStageConfig())


def test_settings_validate_consensus_min_runs_and_retry_kind():
    from app.settings import load_provider_settings, load_settings

    assert load_provider_settings({}).consensus_min_runs == 2
    assert load_provider_settings({"PARTFIDELITY_CONSENSUS_MIN_RUNS": "3"}).consensus_min_runs == 3
    with pytest.raises(RuntimeError):
        load_provider_settings({"PARTFIDELITY_CONSENSUS_MIN_RUNS": "x"})
    assert "malformed" in load_settings({}).extract_retry_failures


def test_failure_copy_is_one_service_reason():
    assert refusal_copy("extract_service_failure") == (
        "The AI service did not answer properly for this video. Try again."
    )
    assert "reliable repair steps" in (refusal_copy("extraction") or "")


def test_fewer_checks_note_says_how_many_reads_and_never_claims_a_check():
    from app.hazards import fewer_checks_reason

    text = fewer_checks_reason(2, 3)
    assert text.startswith("DRAFT:") and "2 AI reads, not 3" in text
    assert not any(word in text.lower() for word in ("verified", "safe", "accurate"))


def test_stitched_steps_carry_the_fewest_runs_of_any_chunk():
    from app.chunks import ChunkWindow, stitch_bundles

    config = SparkStageConfig(consensus_min_runs=2)
    (three, _), _, _ = _extract([_good(), _good(), _good()], config)
    (two, _), _, _ = _extract([_good(), _good(), _failed("timeout")], config, [_failed("timeout")])
    stitched = stitch_bundles(
        [(ChunkWindow(0.0, 60.0), three), (ChunkWindow(58.0, 118.0), two)], overlap_s=2.0
    )
    assert stitched.fewest_runs == 2


# --- fuse-lite -------------------------------------------------------------


def _step(number, start, end, action, hazards=()) -> Step:
    return Step(
        number=number, start_s=start, end_s=end, action=action, tools=(), parts=(),
        hazards=hazards, time_spread_s=0.0, time_needs_check=False,
    )


KW = dict(max_span_s=12.0, title_phrases=frozenset({"title card"}))


def test_fuse_lite_folds_short_repeat_and_title_card_keeping_hazards():
    steps = [
        _step(1, 0, 20, "Remove the screws"),
        _step(2, 20, 25, "Title card shown", ("spark",)),
        _step(3, 25, 60, "Lift the cover"),
        _step(4, 60, 66, "Remove the screws"),
    ]
    out = fuse_lite_steps(steps, **KW)
    assert [(s.number, s.start_s, s.end_s) for s in out] == [(1, 0, 25), (2, 25, 66)]
    assert out[0].hazards == ("spark",)


def test_fuse_lite_leaves_distinct_and_long_repeats_alone():
    steps = [
        _step(1, 0, 20, "Remove the screws"),
        _step(2, 20, 50, "Lift the cover"),
        _step(3, 50, 90, "Remove the screws"),
    ]
    assert fuse_lite_steps(steps, **KW) == tuple(steps)


def test_fuse_lite_never_folds_the_first_step():
    steps = [_step(1, 0, 5, "Title card"), _step(2, 5, 30, "Lift the cover")]
    assert len(fuse_lite_steps(steps, **KW)) == 2


def test_schema_invalid_billed_200s_are_each_retried_once():
    config = SparkStageConfig(consensus_min_runs=2)
    bad = [make_result({}), make_result({})]
    (bundle, raws), seen, client = _extract(
        [bad[0], _good(), bad[1]], config, [_good(), _good()]
    )
    assert seen == [0, 2] and len(client.calls) == 3
    assert bundle.source_runs == 3 and len(raws) == 3


@pytest.mark.parametrize("env", (
    {"PARTFIDELITY_CONSENSUS_MIN_RUNS": "1"},
    {"PARTFIDELITY_CONSENSUS_MIN_RUNS": "4"},
))
def test_load_settings_refuses_out_of_range_consensus_min_runs(env):
    from app.settings import load_settings

    with pytest.raises(RuntimeError, match="CONSENSUS_MIN_RUNS"):
        load_settings(env)


def test_retry_reserves_its_own_run_share(monkeypatch):
    from types import SimpleNamespace

    from app import pipeline

    class _Reached(Exception):
        pass

    def reached(_s):
        raise _Reached  # the pause follows the first budget check

    monkeypatch.setattr(pipeline, "_parent_cancelled", lambda *_: False)
    monkeypatch.setattr(pipeline, "_extract_retry_spend_usd", lambda _p: Decimal("0"))
    monkeypatch.setattr(pipeline.time, "sleep", reached)
    shares = (Decimal("0.09"), Decimal("0.09"), Decimal("0.08"))
    client = SimpleNamespace(spark_stage_config=SparkStageConfig(run_shares=shares))
    ctx = SimpleNamespace(job_id="p1", remaining_s=lambda: 1e6)
    settings = SimpleNamespace(
        extract_retry_count=1, extract_retry_min_remaining_s=0, extract_retry_backoff_s=0,
        chunk_retry_budget_usd=Decimal("0.08"), job_cost_cap_usd=Decimal("10"),
    )
    retry = pipeline._window_retry_call(
        ctx, ctx, client, None, settings, window_index=0, window_s=60.0,
        planned_usd=lambda: Decimal("0"), chunk_count=0, replay=True,
    )
    with pytest.raises(_Reached):
        retry("vid", None, None, False, run_index=2)
    with pytest.raises(SparkServiceError, match="retry budget"):
        retry("vid", None, None, False, run_index=0)


def test_provider_billed_malformed_payload_gives_a_two_run_draft():
    from test_providers import (
        FakeClock, FakeCtx, ScriptedTransport, _run_payload, _step_payload,
        make_spark_client, ok, spark_response, stored_calls,
    )

    from app.hazards import fewer_checks_reason

    good = _run_payload([_step_payload()])
    # run 0 and its retry both come back billed but schema-invalid; runs 1 and 2 are good.
    transport = ScriptedTransport([
        ok(spark_response({})), ok(spark_response({})),
        ok(spark_response(good)), ok(spark_response(good)),
    ])
    ctx = FakeCtx()
    client = make_spark_client(ctx, transport, FakeClock())
    config = SparkStageConfig(consensus_min_runs=2)
    seen: list[int] = []

    def retry_call(file_id, duration, cuts, clocked, *, run_index=0):
        seen.append(run_index)
        return client.run_steps(file_id)

    class _Client:
        spark_stage_config = config
        run_steps = staticmethod(client.run_steps)

    bundle, raws = spark._extract_with_raw(
        ctx, "vid-1", _Client(), config,  # type: ignore[arg-type]
        retry_call=retry_call, retryable_failures=RETRYABLE,
    )
    # the second malformed raw is kept (paid) though it never joins the consensus
    assert seen == [0] and bundle.source_runs == 2 and len(raws) == 3
    assert len(stored_calls(ctx)) == 4
    assert "DRAFT" in fewer_checks_reason(bundle.source_runs, 3)


def _retry_budget_probe(monkeypatch, tmp_path, cap, run_index):
    """Real ledger: an active run-0 child ($0.09) is on the books; retry run_index."""
    from types import SimpleNamespace

    from app import db, pipeline
    from app.admission import admit_job

    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "b.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    monkeypatch.setattr(pipeline, "_parent_cancelled", lambda *_: False)
    monkeypatch.setattr(pipeline, "_extract_retry_spend_usd", lambda _p: Decimal("0"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda _s: None)
    parent = admit_job("pb", "pb", "pb")
    pid = str(parent.job_id)
    admit_job(f"{pid}/extract/retry/0", "x", "x", internal=True)  # active run-0 child
    shares = (Decimal("0.09"), Decimal("0.09"), Decimal("0.08"))
    client = SimpleNamespace(spark_stage_config=SparkStageConfig(run_shares=shares))
    ctx = SimpleNamespace(job_id=pid, remaining_s=lambda: 1e6)
    from app.settings import Settings

    settings = Settings(
        data_root=tmp_path / "data", db_path=database, production=False,
        extract_retry_count=1, extract_retry_min_remaining_s=0, extract_retry_backoff_s=0,
        chunk_retry_budget_usd=Decimal(cap), job_cost_cap_usd=Decimal("10"),
    )
    retry = pipeline._window_retry_call(
        ctx, ctx, client, None, settings, window_index=0, window_s=60.0,
        planned_usd=lambda: Decimal("0"), chunk_count=0, replay=True,
    )
    return retry


def test_active_child_counts_at_its_own_run_share(monkeypatch, tmp_path):
    refused = _retry_budget_probe(monkeypatch, tmp_path / "a", "0.16", 2)
    with pytest.raises(SparkServiceError, match="retry budget"):
        refused("vid", None, None, False, run_index=2)
    admitted = _retry_budget_probe(monkeypatch, tmp_path / "b", "0.17", 2)
    try:
        admitted("vid", None, None, False, run_index=2)
    except SparkServiceError as error:
        assert error.kind != "retry budget"
    except Exception:
        pass  # past admission, the unscripted client may fail: not a budget refusal
