"""Reader tests: reuse, safe sample, difficulty, brevity, hard stop."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import reader as R
from app.contracts import (
    Detection,
    Guide,
    JobId,
    LabelDecision,
    LabelSlot,
    PartRequest,
    Step,
)
from app.routes.site import create_site_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_site_app())


def _step(number, action="Do the thing.", **kw):
    base = dict(
        start_s=0.0,
        end_s=5.0,
        tools=(),
        parts=(),
        hazards=(),
        time_spread_s=1.0,
        time_needs_check=False,
    )
    base.update(kw)
    return Step(number=number, action=action, **base)


def _label(status="part_pixels_unchanged", readback="not_run", detection=None):
    phrase = detection.phrase if detection is not None else "fan"
    return LabelDecision(
        slot=LabelSlot(
            step_index=0,
            part_index=0,
            part=PartRequest(phrase, "center"),
        ),
        detection=detection,
        provisional=None,
        checked=detection is not None,
        status=status,
        readback=readback,
    )


def _guide(steps, difficulty="Easy", verdict="Try it", **kw):
    base = dict(
        job_id=JobId("t"),
        frame_paths=tuple(),
        labels=tuple(),
        draft_reasons=tuple(),
    )
    base.update(kw)
    return Guide(steps=tuple(steps), difficulty=difficulty, verdict=verdict, **base)


def test_reader_reuses_maker_frames_and_labels(client):
    detection = Detection(
        phrase="fan cable",
        mask_path=Path("jobs/abc/mask-01.png"),
        score=0.82,
        box_xyxy=(0, 0, 10, 10),
    )
    guide = _guide(
        (
            _step(
                1,
                "Reseat the fan cable.",
                parts=(PartRequest("fan cable", "above the fan"),),
            ),
        ),
        frame_paths=(Path("jobs/abc/frame-01.png"),),
        labels=(
            _label(detection=detection),
            _label(status="not_checked", readback="not_run"),
        ),
    )
    ctx = R.reader_context(guide, set())
    assert ctx["steps"][0]["frame_src"] == "/jobs/abc/frame-01.png"
    assert ctx["steps"][0]["labels"] == [
        {
            "phrase": "fan cable",
            "status": "No visual changes to this part",
            "readback": "not run",
        }
    ]
    assert [label["status"] for label in ctx["labels_unmatched"]] == ["Not checked"]
    # The served sample page inlines its fixture placeholders (there are
    # no static/frames files); real Guide paths render as-is, per above.
    # The server-side side plate re-shows the first frame, hence + 1.
    page = client.get("/reader/sample").text
    assert page.count('src="data:image/svg+xml,') == len(
        R.sample_guide().frame_paths
    ) + 1
    assert "/static/frames/" not in page


def test_safe_three_step_sample_is_easy(client):
    guide = R.sample_guide()
    assert len(guide.steps) == 3
    assert guide.steps[0].number == 1
    for step in guide.steps:
        assert step.hazards == ()
        lowered = step.action.lower()
        assert "batter" not in lowered
        assert "adhesive" not in lowered
        assert "display" not in lowered
    assert guide.difficulty == "Easy"
    assert guide.verdict == "Try it"
    ctx = R.reader_context(guide, set())
    assert ctx["difficulty"] == "Easy"
    assert ctx["verdict"] == "Try it"
    assert ctx["stop_step"] is None
    assert ctx["hidden_after_stop"] == 0
    page = client.get("/reader/sample").text
    assert '<span class="badge-easy">Easy</span>' in page
    assert "Should you try this? Try it" in page


def test_every_difficulty_branch_matches_l6():
    sharp = _step(1, "Mind the sharp edge.", hazards=("sharp",))
    battery = _step(1, "Disconnect the battery.", hazards=("battery",))
    safe = [_step(n, f"Do safe thing {n}.") for n in range(1, 9)]
    branches = [
        (_guide(safe[:1], "Easy", "Try it"), "Easy", "Try it"),
        (_guide(safe[:3], "Easy", "Try it"), "Easy", "Try it"),
        (_guide(safe[:4], "Medium", "Try it with help"), "Medium", "Try it with help"),
        (_guide(safe[:8], "Medium", "Try it with help"), "Medium", "Try it with help"),
        (
            _guide((sharp, _step(2)), "Medium", "Try it with help"),
            "Medium",
            "Try it with help",
        ),
        (_guide((battery,), "Hard", "See a professional"), "Hard", "See a professional"),
    ]
    for guide, difficulty, verdict in branches:
        ctx = R.reader_context(guide, set())
        # The reader surfaces the hazard module's settled values; it never recomputes them.
        assert ctx["difficulty"] == difficulty, difficulty
        assert ctx["verdict"] == verdict, verdict


def test_unsafe_shortening_keeps_intact_action_and_long_step(client):
    risky = (
        "Seat the drive and refit the screw. Press it gently until flat. "
        "Keep the battery connected while you use 2 metal tools."
    )
    short = R.shorten_action(risky, is_stop=False)
    assert short.display == " ".join(risky.split())
    assert short.long_step is True
    overflow = " ".join(f"word{n}" for n in range(25))
    short = R.shorten_action(overflow + ".", is_stop=False)
    assert short.display == short.full
    assert short.long_step is True
    safe = (
        "Seat the new fan in its recess and refit the screws. "
        "Press it gently until it sits flat. "
        "The recess is shaped so it only fits one way."
    )
    short = R.shorten_action(safe, is_stop=False)
    assert short.display != short.full
    assert short.long_step is True
    assert short.full in client.get("/reader/hero?ack=2").text


def test_hard_stop_never_truncated_and_blocks_later_steps(client):
    assert R.is_hard_stop(_step(1, hazards=("battery",))) is True
    assert R.is_hard_stop(_step(1, hazards=("glued screen",))) is True
    assert R.is_hard_stop(_step(1, hazards=("sharp",))) is False
    hero = R.hero_guide()
    assert hero.difficulty == "Hard"
    assert hero.verdict == "See a professional"
    ctx = R.reader_context(hero, set())
    stop = ctx["steps"][1]
    assert stop["stop"] is True
    assert stop["display"] == stop["full"] == " ".join(hero.steps[1].action.split())
    assert stop["long_step"] is True
    assert [s["visible"] for s in ctx["steps"]] == [True, True, False, False]
    assert ctx["hidden_after_stop"] == 2
    page = client.get("/reader/hero").text
    assert "see a professional" in page.lower()
    assert " ".join(hero.steps[1].action.split()) in page
    assert "Full maker sentence" in page
    assert 'href="/reader/hero?ack=2"' in page
    assert 'id="step-3"' not in page
    assert "Remove the fan screws" not in page
    assert "stay hidden until you acknowledge" in page
    opened = client.get("/reader/hero?ack=2").text
    assert 'id="step-3"' in opened
    assert "Remove the fan screws" in opened
    assert "Stop acknowledged." in opened
    assert "stay hidden until you acknowledge" not in opened
    junk = client.get("/reader/hero?ack=abc&ack=999&ack=-1&ack=2.5").text
    assert 'id="step-3"' not in junk
    assert "stay hidden until you acknowledge" in junk


def test_guide_sample_redirect_resolves_primary_cta(client):
    response = client.get("/guide/sample", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/reader/sample"
    followed = client.get("/guide/sample").text
    assert "Sample guide: fan cable reseat" in followed


def test_calls_and_battery_evidence_within_two_clicks(client):
    landing = client.get("/").text
    assert (
        '<a class="cta-primary" href="/guide/sample">'
        "Explore the sample</a>" in landing
    )
    # Click 1 opens the sample guide through the CTA target; click 2
    # leaves the same sample guide (reader render) for the calls page
    # and the battery-order evidence.
    sample = client.get("/guide/sample").text
    assert '<a href="/calls/sample">Calls</a>' in sample
    assert (
        '<a href="/calls/sample#battery-order">Battery order evidence</a>'
        in sample
    )
