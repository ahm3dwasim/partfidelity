"""boundary clustering, tolerance knob, and threading.

Unit-pins `cluster_time_boundaries` (median time, majority-run
support, one-run-only drops, tolerance edges), the
`boundary_tolerance_s` knob (None means the run's own flag line),
its settings/env plumbing, digest keying, and factory/client
threading. Uses the new API, so it is trivially red before the
change; the behavioural red lives in test_time_consensus.py, which
imports only pre-existing names.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from decimal import Decimal

import pytest

from app.providers import Providers, select_providers
from app.providers.meta_spark import MetaSparkClient
from app.settings import (
    CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR,
    ProviderSettings,
    load_provider_settings,
    load_settings,
)
from app.spark import (
    SparkStageConfig,
    chunk_bundle_digest,
    cluster_time_boundaries,
    consensus_bundle,
    consensus_digest,
    validate_spark_payload,
)

from test_spark_schema import make_payload, make_step


def _consensus(step_sets, config):
    runs = [validate_spark_payload(make_payload(steps=steps)) for steps in step_sets]
    return consensus_bundle(runs, config)


def test_cluster_keeps_majority_supported_median() -> None:
    kept = cluster_time_boundaries(
        [(10.0, 0), (9.0, 1), (10.0, 2), (15.0, 2)],
        tolerance_s=2.0,
        min_support=2,
    )
    assert kept == [(10.0, 1.0, 3)]  # the lone 15.0 edge drops


def test_cluster_even_members_average() -> None:
    assert cluster_time_boundaries(
        [(36.0, 0), (37.0, 1)], tolerance_s=2.0, min_support=2
    ) == [(36.5, 1.0, 2)]


def test_cluster_tolerance_edge_is_inclusive() -> None:
    pair = [(10.0, 0), (12.0, 1)]
    assert cluster_time_boundaries(
        pair, tolerance_s=2.0, min_support=2
    ) == [(11.0, 2.0, 2)]
    assert (
        cluster_time_boundaries(pair, tolerance_s=1.99, min_support=2) == []
    )


def test_cluster_does_not_chain_across_a_wider_gap() -> None:
    # 10.0 and 11.5 pair, but 13.0 sits past the tolerance from the
    # cluster's earliest edge, so it opens its own cluster instead
    # of chaining on.
    assert cluster_time_boundaries(
        [(10.0, 0), (11.5, 1), (13.0, 2)],
        tolerance_s=2.0,
        min_support=2,
    ) == [(10.75, 1.5, 2)]
    assert cluster_time_boundaries(
        [(10.0, 0), (11.5, 1), (13.0, 2)],
        tolerance_s=2.0,
        min_support=1,
    ) == [(10.75, 1.5, 2), (13.0, 0.0, 1)]


def test_cluster_support_counts_distinct_runs() -> None:
    assert (
        cluster_time_boundaries(
            [(10.0, 0), (10.5, 0)], tolerance_s=2.0, min_support=2
        )
        == []
    )  # one run twice is still one run


def test_cluster_empty_input_keeps_nothing() -> None:
    assert cluster_time_boundaries([], tolerance_s=2.0, min_support=2) == []


def test_cluster_keeps_one_boundary_per_run() -> None:
    """same-run edges never share a cluster."""
    assert cluster_time_boundaries(
        [(10.0, 0), (10.0, 1), (10.0, 2), (12.0, 0), (12.0, 1), (12.0, 2)],
        tolerance_s=5.0,
        min_support=2,
    ) == [(10.0, 0.0, 3), (12.0, 0.0, 3)]


def test_cluster_rejects_bad_inputs() -> None:
    for bad_tol in (0, 0.0, -1.0, True, "2", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cluster_time_boundaries(
                [(10.0, 0)], tolerance_s=bad_tol, min_support=2  # type: ignore[arg-type]
            )
    for bad_support in (0, -1, True, "2"):
        with pytest.raises(ValueError):
            cluster_time_boundaries(
                [(10.0, 0)], tolerance_s=2.0, min_support=bad_support  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError):
        cluster_time_boundaries(
            [(float("nan"), 0)], tolerance_s=2.0, min_support=1
        )
    with pytest.raises(ValueError):
        cluster_time_boundaries([(10.0, -1)], tolerance_s=2.0, min_support=1)


def test_config_rejects_bad_boundary_tolerance() -> None:
    assert SparkStageConfig().boundary_tolerance_s is None
    assert SparkStageConfig(boundary_tolerance_s=2).boundary_tolerance_s == 2.0
    for bad in (0, 0.0, -1.0, True, "2", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SparkStageConfig(boundary_tolerance_s=bad)  # type: ignore[arg-type]


def test_none_tolerance_follows_the_flag_line() -> None:
    """Unset tolerance clusters with the run's own spread_s."""
    runs = [
        (make_step(start=0.0, end=10.0), make_step(start=10.0, end=20.0)),
        (make_step(start=0.0, end=12.0), make_step(start=12.0, end=20.0)),
        (make_step(start=0.0, end=12.5), make_step(start=12.5, end=20.0)),
    ]
    wide = _consensus(runs, SparkStageConfig(spread_s=5.0))
    assert [(s.start_s, s.end_s) for s in wide.steps] == [(0.0, 12.0), (12.0, 20.0)]
    narrow = _consensus(runs, SparkStageConfig(spread_s=0.5))
    assert [(s.start_s, s.end_s) for s in narrow.steps] == [
        (0.0, 12.25),
        (12.25, 20.0),
    ]  # the lone 10.0 edge drops, the 12.0/12.5 pair holds


def test_explicit_tolerance_overrides_the_flag_line() -> None:
    runs = [
        (make_step(start=0.0, end=10.0), make_step(start=10.0, end=20.0)),
        (make_step(start=0.0, end=12.0), make_step(start=12.0, end=20.0)),
        (make_step(start=0.0, end=12.5), make_step(start=12.5, end=20.0)),
    ]
    bundle = _consensus(
        runs, SparkStageConfig(spread_s=5.0, boundary_tolerance_s=0.5)
    )
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 12.25),
        (12.25, 20.0),
    ]


def test_boundary_tolerance_setting() -> None:
    assert load_provider_settings({}).consensus_boundary_tolerance_s is None
    tuned = load_provider_settings({CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR: "2.5"})
    assert tuned.consensus_boundary_tolerance_s == 2.5
    for bad in ("0", "0.0", "-1", "bogus", "nan", "inf"):
        with pytest.raises(RuntimeError):
            load_provider_settings({CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR: bad})
    blank = load_provider_settings({CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR: "  "})
    assert blank.consensus_boundary_tolerance_s is None


def test_no_join_tolerance_knob() -> None:
    """joins are exact; no setting, env var, or client arg loosens them."""
    assert "join_tolerance_s" not in SparkStageConfig.__dataclass_fields__
    assert "consensus_join_tolerance_s" not in {
        f.name for f in fields(ProviderSettings)
    }
    assert "join_tolerance_s" not in inspect.signature(MetaSparkClient).parameters
    with pytest.raises(TypeError):
        SparkStageConfig(join_tolerance_s=0.1)  # type: ignore[call-arg]
    # A stale export of the removed env var is ignored, not honoured.
    tuned = load_provider_settings(
        {"PARTFIDELITY_CONSENSUS_JOIN_TOLERANCE_S": "60"}
    )
    assert not hasattr(tuned, "consensus_join_tolerance_s")


def test_consensus_digest_keys_boundary_tolerance() -> None:
    base = SparkStageConfig()
    tuned = SparkStageConfig(boundary_tolerance_s=2.0)
    assert consensus_digest("j", "v", base) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)


def test_chunk_bundle_digest_keys_boundary_tolerance() -> None:
    base = {
        "parent_job_id": "parent-1",
        "source_hash": "aaa",
        "index": 0,
        "window_start_s": 0.0,
        "window_end_s": 60.0,
        "cut_start_s": 0.0,
        "cut_duration_s": 60.0,
    }
    assert chunk_bundle_digest(**base) != chunk_bundle_digest(
        **base, config=SparkStageConfig(boundary_tolerance_s=2.0)
    )


def test_factory_carries_boundary_tolerance() -> None:
    settings = load_settings({CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR: "1.5"})
    trio = select_providers(settings, replay=Providers(spark=1, sam=2, readback=3))
    assert trio.spark.spark_stage_config.boundary_tolerance_s == 1.5
    defaulted = select_providers(
        load_settings({}), replay=Providers(spark=1, sam=2, readback=3)
    )
    assert defaulted.spark.spark_stage_config.boundary_tolerance_s is None


def test_live_client_carries_boundary_tolerance() -> None:
    client = MetaSparkClient(
        api_key="k-test",
        base_url="https://api.test/v1",
        model_id="muse-spark-1.3",
        input_price=Decimal("0.00000125"),
        output_price=Decimal("0.00000425"),
        timeout_s=30,
        ctx=None,
        boundary_tolerance_s=1.5,
    )
    assert client.spark_stage_config.boundary_tolerance_s == 1.5
    defaulted = MetaSparkClient(
        api_key="k-test",
        base_url="https://api.test/v1",
        model_id="muse-spark-1.3",
        input_price=Decimal("0.00000125"),
        output_price=Decimal("0.00000425"),
        timeout_s=30,
        ctx=None,
    )
    assert defaulted.spark_stage_config.boundary_tolerance_s is None
