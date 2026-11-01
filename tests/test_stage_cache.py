"""Stage-cache tests: replay without model calls, atomic JSON storage."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from app import stage_cache
from app.spark import (
    SPARK_PROMPT_VERSION,
    SPARK_SCHEMA_VERSION,
    ChunkExtent,
    SparkRunResult,
    SparkStageConfig,
    bundle_from_json,
    bundle_to_json,
    chunk_bundle_digest,
    consensus_digest,
    extract_steps,
    extract_steps_cached,
    run_digest,
)
from app.stage_cache import (
    SAMPLE_REPLAY_LABEL,
    STAGE_SPARK_CONSENSUS,
    STAGE_SPARK_RUN,
    cache_get,
    cache_put,
    digest_for,
)

from test_spark_schema import RecordingContext, ScriptedSparkClient, make_payload, make_step


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    root = tmp_path / "stage_cache"
    monkeypatch.setattr(stage_cache, "_CACHE_ROOT", root)
    return root


def _results(payloads):
    return [
        SparkRunResult(payload=p, cost_usd=Decimal("0.08"), latency_ms=900) for p in payloads
    ]


def _payloads():
    steps = (make_step(start=10.0, end=20.0), make_step(start=20.0, end=44.0))
    return [make_payload(steps=steps) for _ in range(3)]


def test_cached_sample_is_replay_with_zero_model_calls(tmp_cache):
    first_log: list = []
    first = extract_steps_cached(
        RecordingContext(first_log),
        "sample-vid",
        client=ScriptedSparkClient(first_log, _results(_payloads())),
    )
    assert first.replay is False
    assert len([e for e in first_log if e[0] == "invoke"]) == 3
    assert (tmp_cache / STAGE_SPARK_CONSENSUS / f"{consensus_digest('job-l2-test', 'sample-vid')}.json").exists()
    assert (tmp_cache / STAGE_SPARK_RUN / f"{run_digest('job-l2-test', 'sample-vid', 0)}.json").exists()

    replay_log: list = []
    replay_ctx = RecordingContext(replay_log)

    class RefusingClient:
        def run_steps(self, video_file_id, clip_duration_s=None, scene_cuts=None, clocked=False):
            raise AssertionError("replay must not invoke the client")

    replay = extract_steps_cached(replay_ctx, "sample-vid", client=RefusingClient())
    assert replay.replay is True
    assert replay.bundle == first.bundle  # deterministic consensus round-trips
    assert replay_log == []  # zero reserves, zero settles, zero context calls
    assert SAMPLE_REPLAY_LABEL == "Replay of cached sample, not a live run"


def test_cache_miss_returns_none_and_put_round_trips(tmp_cache):
    digest = digest_for("stage", "input", "prompt", "schema")
    assert cache_get("sparkrun", digest) is None
    value = {"steps": [{"n": 1}], "cost": "0.08"}
    cache_put("sparkrun", digest, value)
    assert cache_get("sparkrun", digest) == value


def test_cache_rejects_bad_stage_digest_corrupt_and_non_json(tmp_cache):
    digest = digest_for("x")
    with pytest.raises(ValueError):
        cache_get("Bogus-Stage", digest)
    with pytest.raises(ValueError):
        cache_get("sparkrun", "not-a-digest")
    with pytest.raises(ValueError):
        cache_put("sparkrun", "../escape", {})
    with pytest.raises(TypeError):
        cache_put("sparkrun", digest, {"bad": object()})
    path = tmp_cache / "sparkrun" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        cache_get("sparkrun", digest)


def test_digests_distinguish_inputs_and_run_indexes():
    assert digest_for("a", "b") != digest_for("a", "c")
    assert digest_for("ab", "c") != digest_for("a", "bc")  # length-prefixed, no ambiguity
    assert run_digest("job", "vid", 0) != run_digest("job", "vid", 1)
    assert run_digest("job", "vid", 0) != consensus_digest("job", "vid")
    assert consensus_digest("job-a", "vid") != consensus_digest("job-b", "vid")
    assert run_digest("job-a", "vid", 0) != run_digest("job-b", "vid", 0)
    assert len(digest_for("a")) == 64


def test_digests_bind_the_chunk_extent():
    extent = ChunkExtent(duration_s=60.0, tolerance_s=2.0)
    assert consensus_digest("job", "vid", SparkStageConfig(), extent) != consensus_digest(
        "job", "vid"
    )
    assert run_digest("job", "vid", 0, SparkStageConfig(), extent) != run_digest(
        "job", "vid", 0
    )
    assert chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0, SparkStageConfig(),
        chunk_tolerance_s=2.0, cut_start_s=0.0, cut_duration_s=60.0,
    ) != chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    longer = ChunkExtent(duration_s=61.0, tolerance_s=2.0)
    looser = ChunkExtent(duration_s=60.0, tolerance_s=3.0)
    assert consensus_digest("job", "vid", SparkStageConfig(), longer) != consensus_digest(
        "job", "vid", SparkStageConfig(), extent
    )
    assert consensus_digest("job", "vid", SparkStageConfig(), looser) != consensus_digest(
        "job", "vid", SparkStageConfig(), extent
    )
    assert run_digest("job", "vid", 0, SparkStageConfig(), longer) != run_digest(
        "job", "vid", 0, SparkStageConfig(), extent
    )
    # The prompt carries the duration only: tolerance never re-keys a raw.
    assert run_digest("job", "vid", 0, SparkStageConfig(), looser) == run_digest(
        "job", "vid", 0, SparkStageConfig(), extent
    )
    assert chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0, SparkStageConfig(),
        chunk_tolerance_s=3.0, cut_start_s=0.0, cut_duration_s=60.0,
    ) != chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0, SparkStageConfig(),
        chunk_tolerance_s=2.0, cut_start_s=0.0, cut_duration_s=60.0,
    )


def test_legacy_and_exact_digests_ignore_the_extent() -> None:
    """Rollback-path cache identity: with an extent passed, legacy and
    exact digests equal the no-extent digest and the pre-extent
    component shape — no extent components on either path."""
    extent = ChunkExtent(duration_s=60.0, tolerance_s=2.0)
    for config in (
        SparkStageConfig(span_mode="legacy"),
        SparkStageConfig(alignment_mode="exact"),
    ):
        assert consensus_digest(
            "job", "vid", config, extent
        ) == consensus_digest("job", "vid", config)
        assert run_digest(
            "job", "vid", 0, config, extent
        ) == run_digest("job", "vid", 0, config)
        assert consensus_digest("job", "vid", config) == digest_for(
            STAGE_SPARK_CONSENSUS,
            "job",
            "vid",
            SPARK_PROMPT_VERSION,
            SPARK_SCHEMA_VERSION,
            config.model_id,
            str(config.runs),
            repr(config.spread_s),
            config.span_mode,
            config.alignment_mode,
            ",".join(sorted(config.generic_tokens)),
            ";".join(
                ",".join(sorted(group)) for group in config.verb_synonyms
            ),
            ",".join(sorted(config.transition_verbs)),
            repr(config.verb_less_overlap_fraction),
            ",".join(sorted(config.negation_words)),
            ",".join(sorted(config.qualifiers)),
            ";".join(
                ",".join(sorted(group)) for group in config.part_synonyms
            ),
            repr(config.boundary_tolerance_s),
        )
        assert run_digest("job", "vid", 0, config) == digest_for(
            STAGE_SPARK_RUN,
            "job",
            "vid",
            SPARK_PROMPT_VERSION,
            SPARK_SCHEMA_VERSION,
            config.model_id,
            str(config.runs),
            repr(config.spread_s),
            str(0),
        )


def _covering_payloads():
    steps = (make_step(start=0.0, end=20.0), make_step(start=20.0, end=60.0))
    return [make_payload(steps=steps) for _ in range(3)]


def test_cached_extract_with_extent_replays_and_isolates(tmp_cache):
    extent = ChunkExtent(duration_s=60.0, tolerance_s=2.0)
    first_log: list = []
    first_client = ScriptedSparkClient(first_log, _results(_covering_payloads()))
    first = extract_steps_cached(
        RecordingContext(first_log), "extent-vid",
        client=first_client, chunk_extent=extent,
    )
    assert first.replay is False
    assert first_client.durations == [60.0] * 3  # the prompt got the chunk
    assert (tmp_cache / STAGE_SPARK_CONSENSUS / f"{consensus_digest('job-l2-test', 'extent-vid', SparkStageConfig(), extent)}.json").exists()

    replay_log: list = []
    replay = extract_steps_cached(
        RecordingContext(replay_log), "extent-vid",
        client=ScriptedSparkClient(replay_log, []),
        chunk_extent=extent,
    )
    assert replay.replay is True
    assert replay.bundle == first.bundle
    assert replay_log == []

    other_log: list = []
    other = extract_steps_cached(
        RecordingContext(other_log), "extent-vid",
        client=ScriptedSparkClient(other_log, _results(_covering_payloads())),
        chunk_extent=ChunkExtent(duration_s=61.0, tolerance_s=2.0),
    )
    assert other.replay is False  # another duration never replays this chunk
    assert len([e for e in other_log if e[0] == "invoke"]) == 3


def test_cache_isolated_across_jobs(tmp_cache):
    job_a_log: list = []
    extract_steps_cached(
        RecordingContext(job_a_log, job_id="job-a"),
        "shared-vid",
        client=ScriptedSparkClient(job_a_log, _results(_payloads())),
    )
    job_b_log: list = []
    second = extract_steps_cached(
        RecordingContext(job_b_log, job_id="job-b"),
        "shared-vid",
        client=ScriptedSparkClient(job_b_log, _results(_payloads())),
    )
    assert second.replay is False  # job B never replays job A's consensus
    assert len([e for e in job_b_log if e[0] == "invoke"]) == 3
    assert consensus_digest("job-a", "shared-vid") != consensus_digest("job-b", "shared-vid")


def test_cache_invalidated_across_model_versions(tmp_cache):
    from app.spark import SparkStageConfig

    first_log: list = []
    extract_steps_cached(
        RecordingContext(first_log),
        "model-vid",
        client=ScriptedSparkClient(first_log, _results(_payloads())),
    )
    replay_log: list = []
    other_model = ScriptedSparkClient(replay_log, _results(_payloads()))
    # The configured model id rides on the client's own stage config.
    other_model.spark_stage_config = SparkStageConfig(model_id="muse-spark-9.9")
    second = extract_steps_cached(
        RecordingContext(replay_log),
        "model-vid",
        client=other_model,
    )
    assert second.replay is False  # a new model never reuses the old consensus
    assert len([e for e in replay_log if e[0] == "invoke"]) == 3


def _cached_step(number=1, **overrides):
    step = {
        **make_step(number=number),
        "time_spread_s": 0.0,
        "time_needs_check": False,
        "kind": "repair",
    }
    step.update(overrides)
    return step


def _cached_bundle(**overrides):
    bundle = {
        "device_guess": "Cordless hand vacuum",
        "confidence": 0.82,
        "source_runs": 3,
        "timestamps_approximate": True,
        "boundaries": [],
        "excluded_spans": [],
        "steps": [_cached_step()],
    }
    bundle.update(overrides)
    return bundle


def test_bundle_from_json_round_trips_a_valid_bundle():
    bundle = bundle_from_json(_cached_bundle())
    assert bundle.device_guess == "Cordless hand vacuum"
    assert bundle.confidence == 0.82
    assert bundle.source_runs == 3
    assert bundle.timestamps_approximate is True
    (step,) = bundle.steps
    assert (step.number, step.start_s, step.end_s) == (1, 10.0, 20.0)
    assert step.action == "Remove the rear screws"
    assert [p.phrase for p in step.parts] == ["battery"]
    assert (step.time_spread_s, step.time_needs_check) == (0.0, False)
    assert step.kind == "repair"
    assert bundle.excluded_spans == ()


@pytest.mark.parametrize(
    "bad",
    [
        _cached_bundle(confidence=float("nan")),
        _cached_bundle(confidence=1.5),
        _cached_bundle(steps=[_cached_step(start=20.0, end=10.0)]),
        _cached_bundle(steps=[_cached_step(start=10.0, end=10.0)]),
        _cached_bundle(steps=[_cached_step(action="  ")]),
        _cached_bundle(steps=[_cached_step(number=1), _cached_step(number=2)]),
        _cached_bundle(steps=[_cached_step(number=i) for i in range(1, 10)]),
        _cached_bundle(steps=[_cached_step(number=1), _cached_step(number=1, start=30.0, end=40.0)]),
        _cached_bundle(steps=[_cached_step(parts=[{"phrase": p, "position_hint": ""} for p in "abcd"])]),
        _cached_bundle(steps=[_cached_step(time_needs_check="yes")]),
        _cached_bundle(steps=[_cached_step(time_spread_s=6.0, time_needs_check=False)]),
        {**_cached_bundle(), "summary": "nope"},
    ],
    ids=[
        "nan-confidence", "confidence-over-1", "start-past-end", "zero-length-window",
        "empty-action", "duplicate-steps", "nine-steps", "duplicate-numbers",
        "four-parts", "string-flag", "spread-over-five-flag-false",
        "unknown-key",
    ],
)
def test_bundle_from_json_rejects_garbage(bad):
    with pytest.raises(ValueError):
        bundle_from_json(bad)


def test_bundle_from_json_accepts_boundary_spread_flag_pairs():
    bundle = bundle_from_json(
        _cached_bundle(
            steps=[
                _cached_step(
                    number=1, start_s=0.0, end_s=5.0,
                    time_spread_s=5.0, time_needs_check=False,
                ),
                _cached_step(
                    number=2, start_s=10.0, end_s=15.0,
                    time_spread_s=8.0, time_needs_check=True,
                ),
            ]
        )
    )
    assert [(s.time_spread_s, s.time_needs_check) for s in bundle.steps] == [
        (5.0, False),
        (8.0, True),
    ]


def test_honest_disagreement_bundle_round_trips_with_spread_zero_flags():
    pair = (
        make_step(start=0.0, end=10.0, action="Remove the rear screws", parts=("battery",)),
        make_step(start=2.0, end=8.0, action="Lift out the motor", parts=("fan",)),
    )
    log: list = []
    payloads = [make_payload(steps=steps) for steps in (pair, pair, (dict(pair[0]),))]
    # Legacy wording consensus: the rivals pin needs the sparse kept
    # pair (contiguous phases mode would separate them by time).
    client = ScriptedSparkClient(log, _results(payloads))
    client.spark_stage_config = SparkStageConfig(span_mode="legacy")
    bundle = extract_steps(RecordingContext(log), "rival-vid", client=client)
    assert [(s.time_spread_s, s.time_needs_check) for s in bundle.steps] == [
        (0.0, True),
        (0.0, True),
    ]
    revived = bundle_from_json(bundle_to_json(bundle))
    assert revived == bundle
    assert [s.time_needs_check for s in revived.steps] == [True, True]


def test_refusal_caches_no_bundle_but_keeps_paid_raws(tmp_cache):
    from app.spark import NotATeardown

    log: list = []
    empty = [make_payload(steps=(), confidence=0.9) for _ in range(3)]
    with pytest.raises(NotATeardown):
        extract_steps_cached(RecordingContext(log), "vid-empty", client=ScriptedSparkClient(log, _results(empty)))
    # The refusal caches no consensus bundle, but the three
    # paid raws are persisted for offline diagnosis.
    assert (
        tmp_cache / STAGE_SPARK_CONSENSUS
        / f"{consensus_digest('job-l2-test', 'vid-empty')}.json"
    ).exists() is False
    for index in range(3):
        assert (
            tmp_cache / STAGE_SPARK_RUN
            / f"{run_digest('job-l2-test', 'vid-empty', index)}.json"
        ).exists()
