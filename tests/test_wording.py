"""The wording pass: each owner-approved or must-fix string is on its page."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.guide_text import TEXT
from app.hazards import HAZARD_TEMPLATES, refusal_copy
from app.routes import uploads as uploads_routes
from app.routes.jobs import STATE_COPY
from app.routes.site import create_site_app

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("seamless", "effortless", "revolutionary", "magic", "cutting-edge", "unlock", "empower")


def test_upload_consent_and_formats():
    page = uploads_routes._upload_page()
    assert "By uploading, you confirm you filmed this video or have the right to share it." in page
    assert "your own footage and you may share it" not in page
    assert "<strong>MP4, MOV or M4V</strong>" in page
    from app.uploads import ALLOWED_CONTAINERS
    import re
    pill = re.search(r"<strong>([^<]*)</strong>", page[page.index("<strong>MP4"):]).group(1)
    assert {t.lower() for t in re.split(r",\s*|\s+or\s+", pill)} == set(ALLOWED_CONTAINERS)
    assert "if the device has one" in TEXT["stop_risk_battery"]
    gallery = (ROOT / "static" / "ui" / "gallery" / "upload.html").read_text(encoding="utf-8")
    assert "you filmed this video or have the right to share it" in gallery
    assert "phones and laptops only" in gallery and "video/*" not in gallery and ".m4v" in gallery
    assert "video/quicktime" in page


def test_how_it_works_transfer_waits_for_confirmation_and_names_outline_limit():
    text = (ROOT / "config" / "how_it_works.toml").read_text(encoding="utf-8")
    assert "Only after you confirm do we send the clip to Meta" in text
    assert "who receives the clip and how long each copy is kept" in text
    assert "Today only the fan and the battery can be outlined" in text


def test_view_switch_names_and_correction_hint():
    for name in ("reader.html", "guide.html"):
        text = (ROOT / "templates" / name).read_text(encoding="utf-8")
        assert ">Edit guide<" in text and ">Follow guide<" in text
        assert "I filmed this" not in text
    assert "Make corrections in Edit guide." in TEXT["view_hint"]


def test_battery_wording_separates_external_and_internal():
    assert "external battery" in HAZARD_TEMPLATES["battery"] and "internal battery" in HAZARD_TEMPLATES["battery"]
    assert "external battery" in TEXT["stop_risk_battery"] and "internal battery" in TEXT["stop_risk_battery"]
    note = TEXT["draft_note_battery_reader"]
    assert "does not show the internal battery being disconnected" in note and "before you go on" in note


def test_stopped_messages_give_a_next_step():
    for kind in ("deadline", "extraction", "frames", "failure", "cancelled", "settlement"):
        assert refusal_copy(kind)
    for kind in ("deadline", "extraction", "frames", "failure", "processing", "readback"):
        text = refusal_copy(kind)
        assert any(w in text for w in ("try", "Try", "Upload", "Trim")), kind
    for state in ("failed", "expired", "pending_settlement"):
        assert any(w in STATE_COPY[state] for w in ("Upload", "Check this page")), state


def test_calls_page_opens_with_a_plain_line():
    template = (ROOT / "templates" / "calls.html").read_text(encoding="utf-8")
    assert "Each row below is one request our app sent to an AI model" in template


def test_no_hype_or_banned_claims_in_changed_copy():
    blob = " ".join(
        [*TEXT.values(), *STATE_COPY.values()]
        + [(ROOT / "config" / n).read_text(encoding="utf-8") for n in ("upload.toml", "how_it_works.toml", "film.toml")]
    ).lower()
    for word in BANNED:
        assert word not in blob, word
