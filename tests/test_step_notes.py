"""Step source notes: strict loader and the sample's step 5 and step 8 notes."""
from __future__ import annotations

import json

import pytest

from app.step_notes import load_step_notes
from test_real_sample import BUNDLE, client  # noqa: F401 (fixture)


def test_loader_is_strict(tmp_path):
    f = tmp_path / "n.json"
    assert load_step_notes(f) == {}  # no file, no notes
    for bad in ('[]', '{"x": {"label": "a", "text": "b", "stage": "c"}}', '{"5": {"label": "a", "text": "b"}}',
                '{"5": {"label": "", "text": "b", "stage": "c"}}', '{"5": {"label": "a", "text": "b", "stage": "c", "x": 1}}'):
        f.write_text(bad)
        with pytest.raises(ValueError):
            load_step_notes(f)
    f.write_text('{"5": {"label": " A ", "text": "b", "stage": "c"}}')
    assert load_step_notes(f)[5].label == "A"


def test_sample_notes_cover_the_missing_disconnect_and_the_windows_check():
    notes = load_step_notes(BUNDLE / "step_notes.json")
    guide = json.loads((BUNDLE / "guide.json").read_text())
    assert sorted(notes) == [5, 8]
    assert "disconnect" in guide["steps"][5]["action"].lower()
    assert "does not show the battery being disconnected" in notes[5].text
    assert "Part pixels unchanged" in notes[5].text and "does not mean the disconnect was filmed" in notes[5].text
    # Step 8 claims only what the frame shows: Device Manager open, Batteries closed.
    assert "does not show the new battery listed" in notes[8].text


def test_notes_render_on_the_sample_cards_only(client):
    page = client.get("/guide/sample", follow_redirects=True).text
    assert page.count('class="gd-step-note"') == 2
    six = page[page.index('id="step-5"'):page.index('id="step-6"')]
    assert "Not in the video" in six and six.index("gd-step-note") < six.index("pf-hazard")
    nine = page[page.index('id="step-8"'):]
    assert "Check this yourself" in nine and "verified" not in nine.lower()
    assert 'id="gd-stage-note"' in page


def test_home_sample_teaser(client):
    import re

    landing = client.get("/").text
    teaser = landing[landing.index('class="sample"'):]
    assert "exactly as your guide will" not in teaser
    assert "same layout" in teaser and "depends on your footage" in teaser
    # Separate badges for step count, difficulty and draft status.
    badges = re.findall(r'<li class="sample-badge">([^<]*)</li>', teaser)
    assert badges[:2] == ["8 steps", "Difficulty: Hard"] and badges[2] == "Draft"
    # Warnings sit in their own list, not on the step lines; three steps, link to the rest.
    assert re.findall(r'<li class="sample-warn">([^<]*)</li>', teaser) == ["Battery", "Data", "ESD", "Sharp"]
    steps = re.findall(r'<li value="(\d+)">([^<]*)</li>', teaser)
    assert len(steps) == 3 and steps[0][0] == "1" and steps[-1][0] == "8"
    assert "warning" not in "".join(t for _, t in steps).lower()
    assert 'href="/guide/sample">See all 8 steps in the full guide</a>' in teaser
    # The closing lines stay word for word.
    assert "Good work deserves to travel." in landing and "Make what you know easier to follow." in landing


def test_required_sample_notes_fail_if_missing_or_incomplete(tmp_path):
    f = tmp_path / "notes.json"
    with pytest.raises((FileNotFoundError, ValueError)):
        load_step_notes(f, required={5, 8})
    f.write_text('{"5": {"label": "a", "text": "b", "stage": "c"}}')
    with pytest.raises(ValueError):
        load_step_notes(f, required={5, 8})
