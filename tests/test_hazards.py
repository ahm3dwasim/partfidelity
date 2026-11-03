"""Hazard refusal, warning-template, and battery-order tests (synthetic steps only)."""

import sys
import unicodedata

import pytest

from app.contracts import PartRequest, SAM_MAX_CALLS_PER_JOB, Step
from app.hazards import (
    CTA_SAMPLE,
    DRAFT_BANNER,
    DRAFT_ORDER_REASON,
    HAZARD_TEMPLATES,
    UNKNOWN_HAZARD_TAG,
    UNKNOWN_HAZARD_TEXT,
    assess_hazards,
    difficulty_for,
    normalize_tag,
    reader_stop,
)


def make_step(number, action="Do the thing.", hazards=(), parts=()):
    return Step(
        number=number,
        start_s=0.0,
        end_s=1.0,
        action=action,
        tools=(),
        parts=tuple(parts),
        hazards=tuple(hazards),
        time_spread_s=0.0,
        time_needs_check=False,
    )


def test_non_teardown_and_scope_refusal_authorize_zero_sam():
    cases = [
        # Non-teardown: blank, unknown, or ambiguous guesses, or no steps.
        ("unknown", (make_step(1),), "not_a_teardown"),
        ("  ", (make_step(1),), "not_a_teardown"),
        ("?", (make_step(1),), "not_a_teardown"),
        ("uncertain phone", (make_step(1),), "not_a_teardown"),
        ("maybe a laptop", (make_step(1),), "not_a_teardown"),
        ("phone?", (make_step(1),), "not_a_teardown"),
        ("ThinkPad T480", (), "not_a_teardown"),
        # Scope: refusal concepts in any field (tags included).
        ("ThinkPad T480", (make_step(1, hazards=("mains",)),), "scope"),
        ("ThinkPad T480", (make_step(1, hazards=("swollen",)),), "scope"),
        ("ThinkPad T480", (make_step(1, hazards=("fire",)),), "scope"),
        ("ThinkPad T480", (make_step(1, "Unplug the mains lead."),), "scope"),
        ("ThinkPad T480", (make_step(1, "The cell looks swollen."),), "scope"),
        ("e-bike", (make_step(1),), "scope"),
        ("laptop", (make_step(1, "Open the high-voltage pack."),), "scope"),
        ("phone", (make_step(1, "It was on fire yesterday."),), "scope"),
        ("microwave oven", (make_step(1),), "scope"),
        # Scope: out-of-scope devices fail the phone/laptop allowlist.
        ("tablet", (make_step(1),), "scope"),
        ("iPad Pro", (make_step(1),), "scope"),
        ("toaster", (make_step(1),), "scope"),
        ("toaster oven", (make_step(1),), "scope"),
        ("television", (make_step(1),), "scope"),
        ("tablet or phone", (make_step(1),), "scope"),
        ("phone case", (make_step(1),), "scope"),
        ("smartphone-case", (make_step(1),), "scope"),
        ("laptop charger", (make_step(1),), "scope"),
    ]
    for device_guess, steps, kind in cases:
        assessment = assess_hazards(steps, device_guess)
        assert assessment.refused is True, (device_guess, steps)
        assert assessment.refusal_kind == kind, (device_guess, steps)
        assert assessment.sam_authorized == 0, (device_guess, steps)
        assert CTA_SAMPLE in (assessment.refusal_text or ""), (device_guess, steps)
        assert assessment.warnings == (), (device_guess, steps)


def test_proceeding_guide_authorizes_the_frozen_sam_ceiling():
    assessment = assess_hazards((make_step(1),), "ThinkPad T480")
    assert assessment.refused is False
    assert assessment.refusal_kind == "none"
    assert assessment.refusal_text is None
    assert assessment.sam_authorized == SAM_MAX_CALLS_PER_JOB == 32


@pytest.mark.parametrize(
    "device_guess",
    [
        "phone",
        "  PHONE  ",
        "a phone",
        "smartphone",
        "iPhone",
        "laptop",
        "a laptop",
        "notebook",
        "ThinkPad T480",
        "Lenovo ThinkPad T480",
        "a Lenovo ThinkPad T480",
        "MacBook",
        "Lenovo ThinkPad laptop",
        "Apple iPhone 12",
        "Dell XPS 13 laptop",
        "Samsung Galaxy S21 phone",
        "ThinkPad T480 laptop",
        "MacBook Pro",
        "Lenovo ThinkPad T-480",
        "ThinkPad T-480 laptop",
        "Lenovo ThinkPad X-1 laptop",
        "Acer Aspire 3 laptop computer",
        "15.6 inch gaming laptop",
        "Windows laptop PC",
        "Dell Inspiron 15 3000 2-in-1 laptop",
        "Apple iPhone 12 mobile phone",
        "Samsung Galaxy S21 cell phone",
        "Google Pixel 7 smart phone",
        "Intel Core i5 11th gen laptop",
    ],
)
def test_guesses_of_allowlisted_tokens_with_one_family_word_are_admitted(device_guess):
    assessment = assess_hazards((make_step(1),), device_guess)
    assert assessment.refused is False, device_guess
    assert assessment.sam_authorized == SAM_MAX_CALLS_PER_JOB, device_guess


@pytest.mark.parametrize(
    "device_guess",
    [
        "not a phone",
        "definitely not a laptop",
        "phone or smartwatch",
        "phone or laptop",
        "phone/laptop",
        "either phone",
        "phone and smartwatch",
        "smartwatch phone",
        "phone repair",
        "Lenovo",
        "Dell XPS 13",
        "tablet",
        "Apple iPad tablet",
        "Apple Watch",
        "desktop computer",
        "swollen laptop battery",
        "laptop or phone",
        "phone toaster",
        "phone-toaster",
        "toaster-9 phone",
        "phone accessories",
        "laptop charger",
        "laptop case",
        "phone screen protector",
        "gaming computer",
        "Windows PC",
        "smart speaker",
        "mobile",
        "smart tv",
    ],
)
def test_negated_mixed_or_near_miss_device_guesses_refuse(device_guess):
    assessment = assess_hazards((make_step(1),), device_guess)
    assert assessment.refused is True, device_guess
    assert assessment.refusal_kind == "scope", device_guess
    assert assessment.sam_authorized == 0, device_guess


def test_scope_wins_over_non_teardown():
    assessment = assess_hazards((), "unknown swollen pack")
    assert assessment.refusal_kind == "scope"


@pytest.mark.parametrize(
    ("action", "hazards"),
    [
        ("Open the high‑voltage pack.", ()),
        ("Open the EV battery.", ()),
        ("Inspect the e–bike pack.", ()),
        ("Do the thing.", ("high—voltage",)),
        ("Do the thing.", ("EV",)),
        ("Do the thing.", ("e‑bike",)),
    ],
)
def test_unsupported_device_terms_in_actions_and_tags_refuse_before_sam(
    action, hazards
):
    assessment = assess_hazards((make_step(1, action, hazards=hazards),), "phone")
    assert assessment.refused is True
    assert assessment.refusal_kind == "scope"
    assert assessment.sam_authorized == 0


@pytest.mark.parametrize(
    "dash",
    ["\u2e3a", "\u00ad", "\u058a", "\u2e17", "\u301c", "\ufe31"],
)
def test_scope_normalizes_omitted_unicode_dashes_before_sam(dash):
    assessment = assess_hazards(
        (make_step(1, f"Open the high{dash}voltage pack."),), "phone"
    )
    assert assessment.refused is True
    assert assessment.refusal_kind == "scope"
    assert assessment.sam_authorized == 0


def test_every_unicode_dash_punctuation_is_normalized_before_sam():
    tested = 0
    for codepoint in range(sys.maxunicode + 1):
        dash = chr(codepoint)
        if unicodedata.category(dash) != "Pd":
            continue
        assessment = assess_hazards(
            (make_step(1, f"Open the high{dash}voltage pack."),), "phone"
        )
        assert assessment.refused is True, f"U+{codepoint:04X}"
        assert assessment.refusal_kind == "scope", f"U+{codepoint:04X}"
        assert assessment.sam_authorized == 0, f"U+{codepoint:04X}"
        tested += 1
    assert tested > 0


def test_refusal_text_is_fixed_with_sample_cta():
    scope = assess_hazards((make_step(1, hazards=("mains",)),), "phone")
    assert scope.refusal_text == (
        "This tool handles phones and laptops only. It will not make guides for "
        "mains-powered devices, e-bike or high-voltage packs, swollen batteries "
        f"or anything that was on fire. {CTA_SAMPLE} instead."
    )
    teardown = assess_hazards((), "phone")
    assert teardown.refusal_text == f"We could not find a phone or laptop repair in this video. {CTA_SAMPLE} instead."


def test_missing_battery_disconnect_marks_draft():
    omitted = assess_hazards(
        (make_step(1, "Pry the battery out.", hazards=("battery",)),), "phone"
    )
    assert omitted.draft is True
    assert omitted.draft_reasons == (DRAFT_ORDER_REASON,)
    assert omitted.battery_order.battery_first is False
    assert omitted.battery_order.disconnect_step is None

    late = assess_hazards(
        (
            make_step(1, "Pry the battery.", hazards=("battery",)),
            make_step(2, "Disconnect the battery connector.", hazards=("battery",)),
        ),
        "phone",
    )
    assert late.draft is True
    assert late.draft_reasons == (DRAFT_ORDER_REASON,)
    assert late.battery_order.disconnect_step == 2


def test_battery_first_order_is_not_draft():
    assessment = assess_hazards(
        (
            make_step(1, "Disconnect the battery connector.", hazards=("battery",)),
            make_step(2, "Lift the battery out.", hazards=("battery",)),
        ),
        "phone",
    )
    assert assessment.draft is False
    assert assessment.draft_reasons == ()
    assert assessment.battery_order.battery_first is True
    assert assessment.battery_order.disconnect_step == 1


def test_remove_without_disconnect_verbs_is_still_draft():
    # Removal is not a disconnect: the order rule refuses to infer it.
    assessment = assess_hazards(
        (make_step(1, "Remove the battery.", hazards=("battery",)),), "phone"
    )
    assert assessment.draft is True


@pytest.mark.parametrize(
    "action",
    [
        "Disconnect the screen cable near the battery.",
        "Maybe disconnect the battery.",
        "Do anything but disconnect the battery.",
        "Do not disconnect the battery; pry it out.",
        "Don't unplug the battery pack.",
        "Don’t disconnect the battery.",
        "Never disconnect the battery.",
        # Verb and battery word in different clauses: the verb is unbound.
        "Disconnect the cable; the battery stays put.",
        "Unplug the laptop, then remove the battery.",
        # A question is not affirmative evidence of a disconnect.
        "Should I disconnect the battery?",
        # Only the exact whole imperative is accepted in text fallback.
        "Disconnect the battery.",
        "Please disconnect the battery connector.",
        "Try to disconnect the battery connector.",
        "Disconnect the battery connector if needed.",
        "Disconnect the battery connector and screen cable.",
        "Disconnect the battery connector; lift it out.",
        "Unplug the battery connector.",
    ],
)
def test_negated_unbound_or_unsure_disconnect_is_still_draft(action):
    assessment = assess_hazards(
        (make_step(1, action, hazards=("battery",)),), "phone"
    )
    assert assessment.draft is True, action
    assert assessment.draft_reasons == (DRAFT_ORDER_REASON,), action
    assert assessment.battery_order.battery_first is False, action
    assert assessment.battery_order.disconnect_step is None, action


@pytest.mark.parametrize(
    "action",
    [
        "Disconnect the battery connector.",
        "disconnect battery connector",
        "Disconnect the battery cable.",
        "  DISCONNECT   THE   BATTERY   CABLE  ",
    ],
)
def test_exact_text_disconnect_imperative_counts(action):
    assessment = assess_hazards(
        (make_step(1, action, hazards=("battery",)),),
        "phone",
    )
    assert assessment.battery_order.battery_first is True
    assert assessment.battery_order.disconnect_step == 1
    assert assessment.draft is False


def test_structured_disconnect_requires_exact_action_and_battery_object():
    valid = assess_hazards(
        (
            make_step(
                1,
                "Disconnect",
                hazards=("battery",),
                parts=(PartRequest("battery connector", "lower edge"),),
            ),
        ),
        "phone",
    )
    assert valid.draft is False
    assert valid.battery_order.disconnect_step == 1

    invalid_steps = (
        make_step(
            1,
            "Disconnect",
            hazards=("battery",),
            parts=(PartRequest("screen cable", "near battery"),),
        ),
        make_step(
            1,
            "Maybe disconnect",
            hazards=("battery",),
            parts=(PartRequest("battery connector", "lower edge"),),
        ),
        make_step(
            1,
            "Disconnect",
            hazards=("battery",),
            parts=(
                PartRequest("battery connector", "lower edge"),
                PartRequest("screen cable", "upper edge"),
            ),
        ),
    )
    for step in invalid_steps:
        assessment = assess_hazards((step,), "phone")
        assert assessment.draft is True, step
        assert assessment.battery_order.disconnect_step is None, step


def test_guide_without_battery_needs_no_disconnect():
    assessment = assess_hazards((make_step(1, "Undo two screws."),), "ThinkPad T480")
    assert assessment.draft is False
    assert assessment.battery_order.battery_first is True


def test_battery_order_evidence_names_steps_for_the_judge_path():
    assessment = assess_hazards(
        (
            make_step(1, "Pry the battery.", hazards=("battery",)),
            make_step(2, "Disconnect the battery connector.", hazards=("battery",)),
        ),
        "phone",
    )
    detail = assessment.battery_order.detail
    assert "1" in detail and "2" in detail
    assert assessment.battery_order.battery_steps == (1, 2)


def test_every_warning_class_maps_to_its_fixed_template():
    steps = (
        make_step(1, hazards=("battery",)),
        make_step(2, hazards=("adhesive",)),
        make_step(3, hazards=("heat",)),
        make_step(4, hazards=("display",)),
        make_step(5, hazards=("esd",)),
        make_step(6, hazards=("data",)),
        make_step(7, hazards=("sharp",)),
    )
    assessment = assess_hazards(steps, "phone")
    by_tag = {w.tag: w for w in assessment.warnings}
    assert set(by_tag) == set(HAZARD_TEMPLATES)
    for tag, warning in by_tag.items():
        assert warning.text == HAZARD_TEMPLATES[tag]
    assert by_tag["adhesive/heat"].steps == (2, 3)
    assert by_tag["battery"].steps == (1,)
    # Fixed template order, regardless of step order.
    assert [w.tag for w in assessment.warnings] == [
        "battery",
        "adhesive/heat",
        "display",
        "esd",
        "data",
        "sharp",
    ]


def test_unknown_tags_produce_generic_warning_never_silence():
    steps = (
        make_step(1, hazards=("explosive",)),
        make_step(2, hazards=("battery?",)),
        make_step(
            3, "Disconnect the battery connector.", hazards=("Battery ",)
        ),
        make_step(4, hazards=("",)),  # blank tags are not hazards
    )
    assessment = assess_hazards(steps, "phone")
    assert assessment.refused is False
    by_tag = {w.tag: w for w in assessment.warnings}
    assert UNKNOWN_HAZARD_TAG in by_tag
    unknown = by_tag[UNKNOWN_HAZARD_TAG]
    assert unknown.text == UNKNOWN_HAZARD_TEXT
    assert unknown.steps == (1, 2)
    # Known tags still normalize; the generic warning sorts after them.
    assert by_tag["battery"].steps == (3,)
    assert [w.tag for w in assessment.warnings] == ["battery", UNKNOWN_HAZARD_TAG]
    # Unknown tags never mark DRAFT by themselves; DRAFT stays battery-order-only.
    assert assessment.draft is False
    assert assessment.draft_reasons == ()


def test_adhesive_heat_warning_dedupes_shared_steps():
    assessment = assess_hazards(
        (make_step(1, "Soften the glue.", hazards=("adhesive", "heat")),), "phone"
    )
    (warning,) = assessment.warnings
    assert warning.tag == "adhesive/heat"
    assert warning.steps == (1,)


def test_draft_banner_is_universal_fixed_text():
    assert DRAFT_BANNER == (
        "AI-assisted draft. Not reviewed by a qualified technician. "
        "Not endorsed by any device maker. Repairs can cause injury, fire, "
        "data loss and may void warranties. You follow these steps at your own risk."
    )


@pytest.mark.parametrize(
    "action",
    [
        "Watch for the hidden clips.",
        "Remove the screws near the monitor hinge.",
        "Unplug the external monitor port cover.",
        "Switch off tablet mode in the BIOS.",
        "Lift the desktop replacement laptop base.",
        "Open the tablet case.",
        "Remove the watch back.",
    ],
)
def test_device_type_words_in_steps_do_not_refuse_a_laptop(action):
    assessment = assess_hazards((make_step(1, action, hazards=("tablet",)),), "Acer Aspire 3 laptop")
    assert assessment.refused is False


@pytest.mark.parametrize(
    ("guess", "rule"),
    [
        ("Apple Watch", "device_type:watch"),
        ("Samsung tablet", "device_type:tablet"),
        ("desktop PC", "device_type:desktop"),
        ("LG monitor", "device_type:monitor"),
        ("microwave", "device_type:microwave"),
    ],
)
def test_device_type_guesses_refuse_and_log_the_rule(guess, rule, caplog):
    with caplog.at_level("INFO", logger="app.hazards"):
        assessment = assess_hazards((make_step(1),), guess)
    assert assessment.refusal_kind == "scope"
    assert f"scope refusal: {rule}" in caplog.text


def test_hazard_step_logs_hazard_rule_without_step_text(caplog):
    with caplog.at_level("INFO", logger="app.hazards"):
        assessment = assess_hazards((make_step(1, "Unplug the mains lead SN123."),), "ThinkPad T480")
    assert assessment.refused is True
    assert "scope refusal: hazard:mains" in caplog.text
    assert "SN123" not in caplog.text


@pytest.mark.parametrize("word", ["firmware", "campfire stories", "smokestack photo"])
def test_refusal_words_need_word_boundaries(word):
    assessment = assess_hazards((make_step(1, f"Show the {word}."),), "ThinkPad T480")
    assert assessment.refused is False


class _BlankStripStr(str):
    """str subclass whose strip() erases its content."""

    def strip(self, chars=None):
        return ""


class _PhoneStripStr(str):
    """str subclass of a blank guess whose strip() invents "phone"."""

    def strip(self, chars=None):
        return "phone"


class _ExplodingStr(str):
    """str subclass whose normalizers must never be invoked."""

    def strip(self, chars=None):
        raise AssertionError("strip must never be called")

    def lower(self):
        raise AssertionError("lower must never be called")


def test_r4_subclass_tags_and_guesses_never_bypass_safety():
    """str-subclass tags and guesses fail closed, methods uncalled."""
    assert normalize_tag(_BlankStripStr("battery")) == UNKNOWN_HAZARD_TAG
    assert normalize_tag(_ExplodingStr("battery")) == UNKNOWN_HAZARD_TAG
    assessment = assess_hazards(
        (make_step(1, hazards=(_BlankStripStr("battery"),)),), "phone"
    )
    assert assessment.refused is False
    assert [warning.tag for warning in assessment.warnings] == [UNKNOWN_HAZARD_TAG]
    assert (
        difficulty_for(
            (make_step(1, hazards=(_BlankStripStr("battery"),)),), assessment
        ).level
        == "Hard"
    )
    stop = reader_stop(make_step(1, hazards=(_BlankStripStr("battery"),)))
    assert stop is not None
    assert stop.reason == "unknown_hazard"
    blank_guess = assess_hazards((make_step(1),), _PhoneStripStr("   "))
    assert blank_guess.refused is True
    assert blank_guess.refusal_kind == "not_a_teardown"
    raising = assess_hazards(
        (make_step(1, hazards=(_ExplodingStr("battery"),)),), "phone"
    )
    assert [warning.tag for warning in raising.warnings] == [UNKNOWN_HAZARD_TAG]
    assert reader_stop(make_step(1, _ExplodingStr("Pry the glued screen."))) is None


def test_device_scope_config_fails_loudly(tmp_path):
    from app.hazards import load_device_scope

    good = load_device_scope()
    assert "laptop" in good["laptop_words"] and "phone" in good["phone_words"]
    bad = tmp_path / "bad.toml"
    bad.write_text('[device_scope]\nphone_words = ["phone"]\nlaptop_wordz = ["laptop"]\nreject_words = ["not"]\n')
    with pytest.raises(RuntimeError):
        load_device_scope(bad)
    bad.write_text('[device_scope]\nphone_words = []\nlaptop_words = ["laptop"]\nreject_words = ["not"]\n')
    with pytest.raises(RuntimeError):
        load_device_scope(bad)


def test_lenovo_thinkpad_laptop_from_live_run_is_admitted():
    assessment = assess_hazards((make_step(1),), "Lenovo ThinkPad laptop")
    assert assessment.refused is False
    assert assess_hazards((make_step(1),), "").refusal_kind == "not_a_teardown"


def test_device_scope_loader_fails_on_overlapping_word_sets(tmp_path):
    from app.hazards import load_device_scope

    base = {
        "phone_words": '["phone"]', "laptop_words": '["laptop"]',
        "reject_words": '["not"]', "brand_words": '["dell"]',
        "model_words": '["xps"]', "article_words": '["a"]',
    }
    good = tmp_path / "good.toml"
    good.write_text("[device_scope]\n" + "\n".join(f"{k} = {v}" for k, v in base.items()))
    assert load_device_scope(good)["phone_words"] == {"phone"}
    base["brand_words"] = '["dell", "phone"]'
    bad = tmp_path / "bad.toml"
    bad.write_text("[device_scope]\n" + "\n".join(f"{k} = {v}" for k, v in base.items()))
    with pytest.raises(RuntimeError, match="phone"):
        load_device_scope(bad)


@pytest.mark.parametrize(
    "word",
    ["mains", "swollen", "e-bike", "high voltage", "ev", "fire", "burning", "smoke", "flames"],
)
@pytest.mark.parametrize("field", ["device_guess", "action", "tag"])
def test_every_hazard_concept_refuses_from_every_field(word, field):
    guess, action, tags = "ThinkPad T480", "Do the thing.", ()
    if field == "device_guess":
        guess = f"ThinkPad {word}"
    elif field == "action":
        action = f"Careful with the {word} here."
    else:
        tags = (word,)
    assessment = assess_hazards((make_step(1, action, hazards=tags),), guess)
    assert assessment.refusal_kind == "scope", (word, field)
    assert assessment.sam_authorized == 0


def test_unlisted_device_guess_logs_no_input_token(caplog):
    with caplog.at_level("INFO", logger="app.hazards"):
        assessment = assess_hazards((make_step(1),), "Acer laptop AssetTag")
    assert assessment.refusal_kind == "scope"
    assert "scope refusal: device_guess:unlisted" in caplog.text
    assert "assettag" not in caplog.text.lower()
