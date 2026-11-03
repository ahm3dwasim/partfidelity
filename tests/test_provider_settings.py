"""Loader tests for the provider settings: defaults, overrides, loud failures."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app import sam as app_sam
from app import spark as app_spark
from app.contracts import (
    PART_CHECK_THRESHOLDS,
    SAM_COST_PER_CALL_USD,
    SAM_MAX_REQUEST_STARTS_PER_MINUTE,
    SPARK_RUNS_PER_GUIDE,
    TIME_SPREAD_FLAG_S,
)
from app.settings import load_provider_settings, load_settings


def test_extract_retry_settings_are_configurable() -> None:
    default = load_settings({})
    assert default.extract_retry_count == 1
    assert default.extract_retry_backoff_s == 1.0
    assert default.extract_retry_min_remaining_s == 60.0
    assert default.extract_retry_failures == frozenset({"timeout", "transport", "http_5xx", "empty", "malformed"})
    configured = load_settings({
        "PARTFIDELITY_EXTRACT_RETRY_COUNT": "0",
        "PARTFIDELITY_EXTRACT_RETRY_BACKOFF_S": "0",
        "PARTFIDELITY_EXTRACT_RETRY_MIN_REMAINING_S": "30",
        "PARTFIDELITY_EXTRACT_RETRY_FAILURES": "timeout,http_5xx",
    })
    assert (configured.extract_retry_count, configured.extract_retry_backoff_s,
            configured.extract_retry_min_remaining_s, configured.extract_retry_failures) == (
                0, 0.0, 30.0, frozenset({"timeout", "http_5xx"})
            )


@pytest.mark.parametrize("env", (
    {"PARTFIDELITY_EXTRACT_RETRY_COUNT": "2"},
    {"PARTFIDELITY_EXTRACT_RETRY_BACKOFF_S": "-1"},
    {"PARTFIDELITY_EXTRACT_RETRY_MIN_REMAINING_S": "nan"},
    {"PARTFIDELITY_EXTRACT_RETRY_FAILURES": "local"},
))
def test_extract_retry_settings_reject_invalid_values(env) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        load_settings(env)


def test_provider_defaults_match_contracts_and_measured_values():
    providers = load_settings({}).providers
    assert providers.mode == "replay"
    assert providers.meta_api_base_url == "https://api.meta.ai/v1"
    assert providers.spark_model_id == app_spark.SPARK_MODEL_ID
    assert providers.sam_model_id == app_sam.SAM_MODEL
    assert providers.spark_runs == SPARK_RUNS_PER_GUIDE
    assert providers.time_spread_flag_s == float(TIME_SPREAD_FLAG_S)
    assert providers.sam_max_starts_per_minute == SAM_MAX_REQUEST_STARTS_PER_MINUTE
    assert providers.sam_threshold_fan == PART_CHECK_THRESHOLDS["fan"]
    assert providers.sam_threshold_battery == PART_CHECK_THRESHOLDS["battery"]
    assert providers.sam_phrase_fan == "cooling fan"
    assert providers.sam_phrase_battery == "rectangular battery with label"
    assert providers.spark_timeout_s == 180.0
    assert providers.sam_timeout_s == 120.0
    assert providers.readback_timeout_s == 120.0
    assert providers.spark_input_usd_per_token == Decimal("0.00000125")
    assert providers.spark_output_usd_per_token == Decimal("0.00000425")
    assert providers.sam_usd_per_call == SAM_COST_PER_CALL_USD


def test_provider_overrides_parse():
    providers = load_provider_settings(
        {
            "PARTFIDELITY_PROVIDER_MODE": "live",
            "META_SPARK_MODEL_ID": " muse-spark-1.3 ",
            "META_SAM_MODEL_ID": "sam-3.1",
            "PARTFIDELITY_SPARK_RUNS": "5",
            "PARTFIDELITY_TIME_SPREAD_FLAG_S": "7.5",
            "PARTFIDELITY_SAM_RATE_PER_MIN": "30",
            "PARTFIDELITY_SAM_THRESHOLD_FAN": "0.1",
            "PARTFIDELITY_SAM_THRESHOLD_BATTERY": "0.9",
            "PARTFIDELITY_SAM_PHRASE_FAN": "fan",
            "PARTFIDELITY_SAM_PHRASE_BATTERY": "battery",
            "PARTFIDELITY_SPARK_TIMEOUT_S": "10",
            "PARTFIDELITY_SAM_TIMEOUT_S": "20",
            "PARTFIDELITY_READBACK_TIMEOUT_S": "30",
            "PARTFIDELITY_SPARK_INPUT_USD_PER_TOKEN": "0.5",
            "PARTFIDELITY_SPARK_OUTPUT_USD_PER_TOKEN": "1.5",
            "PARTFIDELITY_SAM_USD_PER_CALL": "0.5",
        }
    )
    assert providers.mode == "live"
    assert providers.spark_model_id == "muse-spark-1.3"
    assert providers.sam_model_id == "sam-3.1"
    assert providers.spark_runs == 5
    assert providers.time_spread_flag_s == 7.5
    assert providers.sam_max_starts_per_minute == 30
    assert providers.sam_threshold_fan == 0.1
    assert providers.sam_threshold_battery == 0.9
    assert providers.sam_phrase_fan == "fan"
    assert providers.sam_phrase_battery == "battery"
    assert providers.spark_timeout_s == 10.0
    assert providers.sam_timeout_s == 20.0
    assert providers.readback_timeout_s == 30.0
    assert providers.spark_input_usd_per_token == Decimal("0.5")
    assert providers.spark_output_usd_per_token == Decimal("1.5")
    assert providers.sam_usd_per_call == Decimal("0.5")


def test_provider_mode_is_case_insensitive_and_validated():
    assert load_provider_settings({"PARTFIDELITY_PROVIDER_MODE": "LIVE"}).mode == "live"
    with pytest.raises(RuntimeError, match="PARTFIDELITY_PROVIDER_MODE"):
        load_provider_settings({"PARTFIDELITY_PROVIDER_MODE": "sometimes"})


def test_provider_base_url_must_be_https():
    providers = load_provider_settings(
        {"META_API_BASE_URL": "https://model.example.test/v2/"}
    )
    assert providers.meta_api_base_url == "https://model.example.test/v2"
    for bad in ("http://model.example.test/v1", "not-a-url", "https://"):
        with pytest.raises(RuntimeError, match="META_API_BASE_URL"):
            load_provider_settings({"META_API_BASE_URL": bad})


@pytest.mark.parametrize(
    "var",
    [
        "PARTFIDELITY_SPARK_RUNS",
        "PARTFIDELITY_SAM_RATE_PER_MIN",
    ],
)
@pytest.mark.parametrize("bad", ["0", "-3", "2.5", "many"])
def test_provider_ints_must_be_positive(var, bad):
    with pytest.raises(RuntimeError, match=var):
        load_provider_settings({var: bad})


@pytest.mark.parametrize(
    "var",
    [
        "PARTFIDELITY_TIME_SPREAD_FLAG_S",
        "PARTFIDELITY_SPARK_TIMEOUT_S",
        "PARTFIDELITY_SAM_TIMEOUT_S",
        "PARTFIDELITY_READBACK_TIMEOUT_S",
    ],
)
@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "soon"])
def test_provider_floats_must_be_positive(var, bad):
    with pytest.raises(RuntimeError, match=var):
        load_provider_settings({var: bad})


@pytest.mark.parametrize(
    "var",
    ["PARTFIDELITY_SAM_THRESHOLD_FAN", "PARTFIDELITY_SAM_THRESHOLD_BATTERY"],
)
@pytest.mark.parametrize("bad", ["-0.1", "1.1", "high"])
def test_provider_thresholds_must_be_0_to_1(var, bad):
    with pytest.raises(RuntimeError, match=var):
        load_provider_settings({var: bad})


@pytest.mark.parametrize(
    "var",
    [
        "PARTFIDELITY_SPARK_INPUT_USD_PER_TOKEN",
        "PARTFIDELITY_SPARK_OUTPUT_USD_PER_TOKEN",
        "PARTFIDELITY_SAM_USD_PER_CALL",
    ],
)
@pytest.mark.parametrize("bad", ["0", "0.000", "-0.5", "nan", "inf", "pricey"])
def test_provider_prices_must_be_finite_positive(var, bad):
    # Zero is refused at load: the Spark shares and SAM holds derived from
    # these prices must stay positive end to end.
    with pytest.raises(RuntimeError, match=var):
        load_provider_settings({var: bad})


@pytest.mark.parametrize(
    "var,bad",
    [("META_SPARK_MODEL_ID", "muse-spark-9"), ("META_SAM_MODEL_ID", "sam-9")],
)
def test_provider_model_ids_must_map_to_a_receipt_model(var, bad):
    # Receipts may only name models in the evidence module's closed allowlist, so an id
    # without a receipt name is refused instead of mislabelled.
    with pytest.raises(RuntimeError, match=var):
        load_provider_settings({var: bad})


def test_live_mode_fails_loudly_without_key():
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        load_settings({"PARTFIDELITY_PROVIDER_MODE": "live"})
    live = load_settings(
        {"PARTFIDELITY_PROVIDER_MODE": "live", "META_API_KEY": "k"}
    )
    assert live.providers.mode == "live"
    assert live.meta_api_key == "k"
    # Replay stays keyless.
    assert load_settings({}).meta_api_key is None


def test_blank_overrides_keep_defaults():
    providers = load_provider_settings(
        {
            "PARTFIDELITY_SPARK_RUNS": "  ",
            "META_SPARK_MODEL_ID": "  ",
            "PARTFIDELITY_SAM_USD_PER_CALL": "",
        }
    )
    assert providers.spark_runs == SPARK_RUNS_PER_GUIDE
    assert providers.spark_model_id == app_spark.SPARK_MODEL_ID
    assert providers.sam_usd_per_call == SAM_COST_PER_CALL_USD
