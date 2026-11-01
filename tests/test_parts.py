"""Checked-part configuration: fan + battery only, thresholds from contracts."""

from app.contracts import PART_CHECK_THRESHOLDS
from app.parts import checked_parts, is_checked_part, normalize_phrase, threshold_for


def test_fan_06_battery_05():
    assert threshold_for("fan") == 0.6
    assert threshold_for("battery") == 0.5
    assert is_checked_part("fan") is True
    assert is_checked_part("battery") is True


def test_unsupported_parts_have_no_threshold():
    for phrase in ("ram", "ssd", "connector", "screws", "battery connector",
                   "fan screws", "cooling fan", "the battery", ""):
        assert threshold_for(phrase) is None, phrase
        assert is_checked_part(phrase) is False, phrase


def test_phrase_matching_is_case_and_space_insensitive():
    assert threshold_for("  Fan ") == 0.6
    assert threshold_for("BATTERY") == 0.5
    assert normalize_phrase("  Fan\t") == "fan"


def test_checked_parts_lists_configured_parts_in_order():
    assert checked_parts() == ("fan", "battery")
    assert dict(PART_CHECK_THRESHOLDS) == {"fan": 0.6, "battery": 0.5}


def test_thresholds_are_configurable_for_later_modal_parts():
    extended = {"fan": 0.6, "battery": 0.5, "ram": 0.7}
    assert threshold_for("ram", thresholds=extended) == 0.7
    assert is_checked_part("ram", thresholds=extended) is True
    assert checked_parts(thresholds=extended) == ("fan", "battery", "ram")
