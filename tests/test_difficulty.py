"""Ordered-difficulty tests."""

import pytest

from app.hazards import (
    CTA_HELP,
    CTA_PROFESSIONAL,
    CTA_SAMPLE,
    CTA_TRY,
    assess_hazards,
    difficulty_for,
    reader_stop,
)

from test_hazards import make_step


def safe_steps(n):
    return tuple(make_step(i, f"Do safe thing {i}.") for i in range(1, n + 1))


def classify(device_guess, steps):
    return difficulty_for(steps, assess_hazards(steps, device_guess))


def test_every_settled_difficulty_branch():
    # Refusal first: no level, only the sample CTA.
    refused = classify("microwave oven", (make_step(1),))
    assert (refused.refused, refused.level, refused.cta) == (True, None, CTA_SAMPLE)

    # Battery, adhesive, or display: Hard + See a professional.
    for tag in ("battery", "adhesive", "display"):
        result = classify("phone", (make_step(1, hazards=(tag,)),))
        assert (result.refused, result.level, result.cta) == (
            False,
            "Hard",
            CTA_PROFESSIONAL,
        ), tag

    # Another hazard on a short guide: Medium + Try it with help.
    for tag in ("heat", "esd", "data", "sharp"):
        result = classify("ThinkPad T480", (make_step(1, hazards=(tag,)),))
        assert (result.refused, result.level, result.cta) == (
            False,
            "Medium",
            CTA_HELP,
        ), tag

    # 4-7 safe steps: Medium + Try it with help.
    for n in (4, 5, 6, 7):
        result = classify("ThinkPad T480", safe_steps(n))
        assert (result.refused, result.level, result.cta) == (
            False,
            "Medium",
            CTA_HELP,
        ), n

    # 1-3 safe steps: Easy + Try it.
    for n in (1, 2, 3):
        result = classify("ThinkPad T480", safe_steps(n))
        assert (result.refused, result.level, result.cta) == (False, "Easy", CTA_TRY), n


def test_out_of_scope_and_ambiguous_devices_refuse_with_sample_cta():
    for guess in ("tablet", "toaster", "iPad Pro", "uncertain phone", "?", "phone?"):
        result = classify(guess, (make_step(1),))
        assert (result.refused, result.level, result.cta) == (
            True,
            None,
            CTA_SAMPLE,
        ), guess


def test_unknown_hazard_tags_resolve_hard():
    for tag in ("explosive", "battery?"):
        result = classify("phone", (make_step(1, hazards=(tag,)),))
        assert (result.refused, result.level, result.cta) == (
            False,
            "Hard",
            CTA_PROFESSIONAL,
        ), tag


def test_hazard_free_eight_steps_is_medium():
    result = classify("ThinkPad T480", safe_steps(8))
    assert (result.refused, result.level, result.cta) == (False, "Medium", CTA_HELP)


def test_safe_three_step_sample_is_easy():
    # The reader's safe-sample contract: exactly three steps, no hazard, battery,
    # adhesive, or display; resolves Easy and fires no stop.
    steps = safe_steps(3)
    result = classify("ThinkPad T480", steps)
    assert (result.refused, result.level, result.cta) == (False, "Easy", CTA_TRY)
    assert all(reader_stop(s) is None for s in steps)
    assessment = assess_hazards(steps, "ThinkPad T480")
    assert assessment.warnings == ()
    assert assessment.draft is False


def test_empty_guide_is_never_easy():
    assessment = assess_hazards((), "phone")
    assert assessment.refused is True
    assert difficulty_for((), assessment).refused is True


@pytest.mark.parametrize("tag", ["battery", "adhesive", "display", "heat", "esd", "data", "sharp"])
def test_single_hazard_step_length_does_not_change_the_row(tag):
    # One hazard step plus safe steps: Hard tags stay Hard, other hazards
    # stay Medium, at every settled length.
    for n in (1, 2, 3):
        steps = (make_step(1, hazards=(tag,)),) + tuple(
            make_step(i, f"Do safe thing {i}.") for i in range(2, n + 1)
        )
        result = classify("phone", steps)
        expected = "Hard" if tag in ("battery", "adhesive", "display") else "Medium"
        assert result.level == expected, (tag, n)
