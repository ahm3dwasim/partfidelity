"""Reader hard-stop tests: stop object, full text, and hidden-until-ack rule."""

import asyncio
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from app.hazards import (
    HARD_STOP_ACK_PATH,
    HARD_STOP_READER_COOKIE,
    HARD_STOP_TEXT,
    HARD_STOP_TITLE,
    HardStopAcknowledgements,
    UNKNOWN_STOP_TEXT,
    assess_hazards,
    deliverable_steps,
    difficulty_for,
    hard_stop_acknowledgement_router,
    hidden_step_numbers,
    reader_stop,
)

from test_hazards import make_step

ROOT = Path(__file__).resolve().parent.parent
HARD_STOP_PARTIAL = (ROOT / "templates" / "partials" / "hard_stop.html").read_text()
HAZARD_PARTIAL = (ROOT / "templates" / "partials" / "hazard_block.html").read_text()
HARD_STOP_JS = (ROOT / "static" / "hard_stop.js").read_text()

GUIDE_ID = "guide-A"
GUIDE_REVISION = "rev-2026-10-27-a"
STOP_STEP = 2
SEALED_MARKER = "ZZZ-SEALED-9"


def test_battery_and_glued_screen_hard_stop():
    battery = reader_stop(make_step(2, "Disconnect the battery.", hazards=("battery",)))
    assert battery is not None
    assert battery.step_number == 2
    assert battery.reason == "battery"

    glued_tags = reader_stop(
        make_step(3, "Pry the screen.", hazards=("display", "adhesive"))
    )
    assert glued_tags is not None
    assert glued_tags.reason == "glued_screen"

    glued_words = reader_stop(make_step(4, "Soften the glued screen edges."))
    assert glued_words is not None
    assert glued_words.reason == "glued_screen"

    glued_back = reader_stop(make_step(5, "Pry the glued back cover."))
    assert glued_back is not None
    assert glued_back.reason == "glued_screen"

    for stop in (battery, glued_tags, glued_words, glued_back):
        assert stop.title == HARD_STOP_TITLE
        assert stop.text == HARD_STOP_TEXT
        assert stop.cta == "See a professional"


def test_unknown_hazard_stop_uses_generic_fixed_text():
    for tag in ("explosive", "battery?", "swollen", "mains"):
        stop = reader_stop(make_step(2, "Do the thing.", hazards=(tag,)))
        assert stop is not None, tag
        assert stop.step_number == 2, tag
        assert stop.reason == "unknown_hazard", tag
        assert stop.title == HARD_STOP_TITLE, tag
        assert stop.text == UNKNOWN_STOP_TEXT, tag
        assert stop.cta == "See a professional", tag
    # A known battery stop keeps its specific reason and text.
    both = reader_stop(make_step(2, hazards=("battery", "explosive")))
    assert both is not None
    assert both.reason == "battery"
    assert both.text == HARD_STOP_TEXT


def test_no_stop_without_battery_or_glued_surface():
    assert reader_stop(make_step(1, "Undo two screws.")) is None
    assert reader_stop(make_step(1, "Warm the part.", hazards=("heat",))) is None
    # Display alone is Hard but not a glued-screen stop.
    assert reader_stop(make_step(1, "Lift the screen.", hazards=("display",))) is None
    # Glue words without a screen/back word are not a stop.
    assert reader_stop(make_step(1, "Reglue the rubber foot.")) is None


def test_stop_text_is_never_shortened():
    assert HARD_STOP_TEXT == (
        "Stop here. This step involves a battery or a glued screen. "
        "These can catch fire or shatter. See a professional."
    )
    stop = reader_stop(make_step(1, hazards=("battery",)))
    assert stop is not None and stop.text == HARD_STOP_TEXT
    # The partial renders the variable unfiltered: no truncate/ellipsis/slice.
    assert "{{ stop.text }}" in HARD_STOP_PARTIAL
    for banned in ("|truncat", "|ellipsis", "|slice", "|length", "..."):
        assert banned not in HARD_STOP_PARTIAL, banned
    assert "{{ warning.text }}" in HAZARD_PARTIAL


def test_later_steps_hidden_until_acknowledged():
    stops = (
        reader_stop(make_step(2, hazards=("battery",))),
        reader_stop(make_step(5, hazards=("display", "adhesive"))),
    )
    assert all(s is not None for s in stops)
    numbers = (1, 2, 3, 4, 5, 6)

    assert hidden_step_numbers(numbers, stops, frozenset()) == frozenset({3, 4, 5, 6})
    # Acknowledging the first stop never reveals past the second.
    assert hidden_step_numbers(numbers, stops, frozenset({2})) == frozenset({6})
    assert hidden_step_numbers(numbers, stops, frozenset({2, 5})) == frozenset()
    # The stop step itself is always visible.
    assert 2 not in hidden_step_numbers(numbers, stops, frozenset())
    # No stops: nothing hidden.
    assert hidden_step_numbers(numbers, (), frozenset()) == frozenset()


def test_server_withholds_later_steps_until_acknowledged():
    steps = (
        make_step(1, "Undo the bottom screws."),
        make_step(2, "Disconnect the battery connector.", hazards=("battery",)),
        make_step(3, f"Lift the battery out; serial {SEALED_MARKER}."),
        make_step(4, "Lift the fan out."),
    )
    stops = tuple(s for s in (reader_stop(s) for s in steps) if s is not None)
    assert [s.step_number for s in stops] == [STOP_STEP]
    sealed = deliverable_steps(steps, stops, frozenset())
    assert [s.number for s in sealed] == [1, 2]
    assert SEALED_MARKER not in " ".join(s.action for s in sealed)
    opened = deliverable_steps(steps, stops, frozenset({STOP_STEP}))
    assert [s.number for s in opened] == [1, 2, 3, 4]


def test_stop_steps_resolve_hard():
    steps = (
        make_step(1, "Undo two screws."),
        make_step(2, "Disconnect the battery.", hazards=("battery",)),
        make_step(3, "Soften the glued screen edges."),
    )
    assert [reader_stop(s) is not None for s in steps] == [False, True, True]
    result = difficulty_for(steps, assess_hazards(steps, "phone"))
    assert (result.level, result.cta) == ("Hard", "See a professional")


def test_stop_partial_carries_acknowledgement_contract():
    for hook in (
        'role="alert"',
        "data-hard-stop",
        "data-acknowledged",
        "data-guide-revision",
        "data-stop-id",
        '<form method="post"',
        'name="guide_revision"',
        'name="stop_id"',
        "data-hard-stop-ack",
        "data-hard-stop-acked",
    ):
        assert hook in HARD_STOP_PARTIAL, hook
    # The partial documents the server rule: render only deliverable steps.
    assert "deliverable_steps" in HARD_STOP_PARTIAL


def test_stop_partial_fails_closed_without_revision_or_stop_identity():
    env = Environment(
        loader=FileSystemLoader(str(ROOT / "templates" / "partials")),
        autoescape=True,
    )
    stop = reader_stop(make_step(2, hazards=("battery",)))
    assert stop is not None
    rendered = env.get_template("hard_stop.html").render(
        stop=stop,
        acknowledged=False,
    )
    assert "<form" not in rendered
    assert "Later steps remain locked" in rendered


def test_browser_script_only_tracks_pending_submission():
    # Client state is pending-only. It cannot acknowledge a stop or reveal a step.
    for hook in (
        "data-hard-stop-form",
        "guide_revision",
        "stop_id",
        "partfidelity-hard-stop-pending-",
        "aria-busy",
    ):
        assert hook in HARD_STOP_JS, hook
    for banned in (
        "data-acknowledged\", \"true",
        "removeAttribute(\"hidden\")",
        "aria-hidden",
        "hard-stop-acknowledged",
        "partfidelity-hard-stop-ack-",
        "fetch(",
        "XMLHttpRequest",
        "navigator.sendBeacon",
        "import(",
    ):
        assert banned not in HARD_STOP_JS, banned


def test_registering_new_revision_invalidates_old_acknowledgements():
    acknowledgements = HardStopAcknowledgements()
    acknowledgements.register_guide(
        GUIDE_ID, "old-revision", ("step-2-battery",), "/guides/guide-A"
    )
    assert acknowledgements.acknowledge(
        "reader-A", GUIDE_ID, "old-revision", "step-2-battery"
    ) == "/guides/guide-A"
    acknowledgements.register_guide(
        GUIDE_ID, GUIDE_REVISION, ("step-2-battery",), "/guides/guide-A"
    )
    assert not acknowledgements.is_acknowledged(
        "reader-A", GUIDE_ID, "old-revision", "step-2-battery"
    )
    assert acknowledgements.acknowledge(
        "reader-A", GUIDE_ID, "old-revision", "step-2-battery"
    ) is None
    with pytest.raises(ValueError):
        acknowledgements.register_guide(
            "guide-B", "revision-B", ("step-2-battery",), "/../outside"
        )


def test_acknowledgements_expire_and_stay_bounded():
    now = [100.0]
    acknowledgements = HardStopAcknowledgements(now=lambda: now[0])
    acknowledgements.register_guide(
        GUIDE_ID, GUIDE_REVISION, ("step-2-battery",), "/guides/guide-A"
    )

    for index in range(4097):
        assert acknowledgements.acknowledge(
            f"reader-{index}", GUIDE_ID, GUIDE_REVISION, "step-2-battery"
        ) == "/guides/guide-A"

    assert len(acknowledgements._acknowledged) <= 4096
    assert not acknowledgements.is_acknowledged(
        "reader-0", GUIDE_ID, GUIDE_REVISION, "step-2-battery"
    )
    assert acknowledgements.is_acknowledged(
        "reader-4096", GUIDE_ID, GUIDE_REVISION, "step-2-battery"
    )

    now[0] += 24 * 60 * 60
    assert not acknowledgements.is_acknowledged(
        "reader-4096", GUIDE_ID, GUIDE_REVISION, "step-2-battery"
    )


def test_guide_registry_stays_bounded_and_eviction_removes_acknowledgements():
    acknowledgements = HardStopAcknowledgements()
    stop_id = "step-2-battery"

    acknowledgements.register_guide(
        "guide-0", GUIDE_REVISION, (stop_id,), "/guides/guide-0"
    )
    assert acknowledgements.acknowledge(
        "reader-A", "guide-0", GUIDE_REVISION, stop_id
    ) == "/guides/guide-0"

    for index in range(1, 4097):
        acknowledgements.register_guide(
            f"guide-{index}",
            GUIDE_REVISION,
            (stop_id,),
            f"/guides/guide-{index}",
        )

    assert len(acknowledgements._guides) == 4096
    assert "guide-0" not in acknowledgements._guides
    assert not any(key[1] == "guide-0" for key in acknowledgements._acknowledged)
    assert acknowledgements.acknowledge(
        "reader-A", "guide-0", GUIDE_REVISION, stop_id
    ) is None


def test_unregister_guide_removes_registration_and_acknowledgements():
    acknowledgements = HardStopAcknowledgements()
    stop_id = "step-2-battery"
    acknowledgements.register_guide(
        GUIDE_ID, GUIDE_REVISION, (stop_id,), "/guides/guide-A"
    )
    assert acknowledgements.acknowledge(
        "reader-A", GUIDE_ID, GUIDE_REVISION, stop_id
    ) == "/guides/guide-A"

    acknowledgements.unregister_guide(GUIDE_ID)

    assert GUIDE_ID not in acknowledgements._guides
    assert not any(key[1] == GUIDE_ID for key in acknowledgements._acknowledged)
    assert acknowledgements.acknowledge(
        "reader-A", GUIDE_ID, GUIDE_REVISION, stop_id
    ) is None


def test_revision_replacement_only_removes_matching_guide_acknowledgements():
    acknowledgements = HardStopAcknowledgements()
    stop_id = "step-2-battery"
    for guide_id in ("guide-A", "guide-B"):
        acknowledgements.register_guide(
            guide_id, "same-revision", (stop_id,), f"/guides/{guide_id}"
        )
        assert acknowledgements.acknowledge(
            "reader-A", guide_id, "same-revision", stop_id
        ) == f"/guides/{guide_id}"

    acknowledgements.register_guide(
        "guide-A", "new-revision", (stop_id,), "/guides/guide-A"
    )

    assert not acknowledgements.is_acknowledged(
        "reader-A", "guide-A", "same-revision", stop_id
    )
    assert acknowledgements.is_acknowledged(
        "reader-A", "guide-B", "same-revision", stop_id
    )


def test_chunked_oversized_acknowledgement_stops_reading_at_limit():
    acknowledgements = HardStopAcknowledgements()
    router = hard_stop_acknowledgement_router(acknowledgements)
    endpoint = router.routes[0].endpoint
    received = 0

    async def receive():
        nonlocal received
        received += 1
        if received == 1:
            return {"type": "http.request", "body": b"a" * 1024, "more_body": True}
        if received == 2:
            return {"type": "http.request", "body": b"b" * 1025, "more_body": True}
        raise AssertionError("handler read past the chunk that crossed the form limit")

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": HARD_STOP_ACK_PATH,
            "raw_path": HARD_STOP_ACK_PATH.encode(),
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/x-www-form-urlencoded"),
                (b"cookie", f"{HARD_STOP_READER_COOKIE}=reader-A".encode()),
            ],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "root_path": "",
        },
        receive,
    )

    async def post():
        with pytest.raises(HTTPException) as caught:
            await endpoint(request)
        assert caught.value.status_code == 413

    asyncio.run(post())
    assert received == 2


# --- Real browser test: shipped script plus the production POST handler. ---


def _fixture_steps():
    return (
        make_step(1, "Undo the bottom screws."),
        make_step(2, "Disconnect the battery connector.", hazards=("battery",)),
        make_step(3, f"Lift the battery out; serial {SEALED_MARKER}."),
        make_step(4, "Lift the fan out."),
    )


def _render_guide(
    acknowledgements: HardStopAcknowledgements,
    reader_id: str,
) -> str:
    """Render only bytes the server may deliver for this reader."""
    env = Environment(
        loader=FileSystemLoader(str(ROOT / "templates" / "partials")),
        autoescape=True,
    )
    stop_tpl = env.get_template("hard_stop.html")
    steps = _fixture_steps()
    stops = {s.number: reader_stop(s) for s in steps}
    acknowledged_steps = frozenset(
        step_number
        for step_number, stop in stops.items()
        if stop is not None
        and acknowledgements.is_acknowledged(
            reader_id, GUIDE_ID, GUIDE_REVISION, stop.stop_id
        )
    )
    delivered = deliverable_steps(
        steps,
        tuple(stop for stop in stops.values() if stop is not None),
        acknowledged_steps,
    )

    def card(step):
        extra = ""
        if stops[step.number] is not None:
            stop = stops[step.number]
            extra = stop_tpl.render(
                stop=stop,
                acknowledged=step.number in acknowledged_steps,
                guide_id=GUIDE_ID,
                guide_revision=GUIDE_REVISION,
                acknowledge_url=HARD_STOP_ACK_PATH,
            )
        return (
            f'<section data-step="{step.number}">'
            f"<h2>Step {step.number}</h2><p>{step.action}</p>{extra}</section>"
        )

    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Reader hard stop</title></head>"
        f'<body><main data-guide="{GUIDE_ID}" '
        f'data-guide-revision="{GUIDE_REVISION}">'
        f'{"".join(card(s) for s in delivered)}</main>'
        f"<script>{HARD_STOP_JS}</script></body></html>"
    )


@pytest.fixture
def hard_stop_server():
    acknowledgements = HardStopAcknowledgements()
    stop = reader_stop(_fixture_steps()[1])
    assert stop is not None
    acknowledgements.register_guide(
        GUIDE_ID,
        GUIDE_REVISION,
        (stop.stop_id,),
        f"/guides/{GUIDE_ID}",
    )
    started = threading.Event()

    @asynccontextmanager
    async def lifespan(_app):
        started.set()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(hard_stop_acknowledgement_router(acknowledgements))

    @app.get("/guides/{guide_id}", response_class=HTMLResponse)
    def guide(guide_id: str, request: Request) -> HTMLResponse:
        assert guide_id == GUIDE_ID
        reader_id = request.cookies.get(HARD_STOP_READER_COOKIE) or "reader-test"
        response = HTMLResponse(_render_guide(acknowledgements, reader_id))
        if HARD_STOP_READER_COOKIE not in request.cookies:
            response.set_cookie(
                HARD_STOP_READER_COOKIE,
                reader_id,
                httponly=True,
                samesite="strict",
            )
        return response

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="critical", access_log=False)
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    assert started.wait(timeout=10), "ASGI server did not report startup"
    try:
        yield f"http://{host}:{port}", acknowledgements, stop
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "ASGI server did not stop"


def _launch_chrome(playwright):
    """Launch the installed Playwright Chromium, with system Chrome fallback."""
    try:
        return playwright.chromium.launch(headless=True)
    except Exception:
        return playwright.chromium.launch(
            headless=True,
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            args=["--no-sandbox"],
        )


def test_browser_hard_stop_requires_valid_server_acknowledgement(hard_stop_server):
    from playwright.sync_api import expect, sync_playwright

    pages_base, acknowledgements, stop = hard_stop_server
    pending_key = (
        "partfidelity-hard-stop-pending-"
        f"{GUIDE_REVISION}:{stop.stop_id}"
    )
    with sync_playwright() as playwright:
        browser = _launch_chrome(playwright)
        try:
            anonymous = browser.new_context()
            no_session = anonymous.request.post(
                f"{pages_base}{HARD_STOP_ACK_PATH}",
                form={
                    "guide_id": GUIDE_ID,
                    "guide_revision": GUIDE_REVISION,
                    "stop_id": stop.stop_id,
                },
            )
            assert no_session.status == 403
            anonymous.close()

            ctx = browser.new_context()
            ctx.add_init_script(
                f"window.localStorage.setItem({pending_key!r}, '1')"
            )
            page = ctx.new_page()
            initial = page.goto(f"{pages_base}/guides/{GUIDE_ID}")
            assert initial is not None
            initial_body = initial.text()
            expect(page.locator('[data-step="2"]')).to_be_visible()
            assert SEALED_MARKER not in initial_body
            assert page.locator('[data-step="3"]').count() == 0
            assert page.evaluate(
                "key => window.localStorage.getItem(key)", pending_key
            ) == "1"

            cookies = {cookie["name"]: cookie for cookie in ctx.cookies()}
            reader_cookie = cookies[HARD_STOP_READER_COOKIE]
            assert reader_cookie["httpOnly"] is True
            assert reader_cookie["sameSite"] == "Strict"
            reader_id = reader_cookie["value"]

            # Stale revisions and forged stop ids are refused and record nothing.
            stale = page.request.post(
                f"{pages_base}{HARD_STOP_ACK_PATH}",
                form={
                    "guide_id": GUIDE_ID,
                    "guide_revision": "stale-revision",
                    "stop_id": stop.stop_id,
                },
            )
            assert stale.status == 409
            forged = page.request.post(
                f"{pages_base}{HARD_STOP_ACK_PATH}",
                form={
                    "guide_id": GUIDE_ID,
                    "guide_revision": GUIDE_REVISION,
                    "stop_id": "step-999-forged",
                },
            )
            assert forged.status == 409
            missing = page.request.post(
                f"{pages_base}{HARD_STOP_ACK_PATH}",
                form={"guide_id": GUIDE_ID},
            )
            assert missing.status == 400
            extra = page.request.post(
                f"{pages_base}{HARD_STOP_ACK_PATH}",
                form={
                    "guide_id": GUIDE_ID,
                    "guide_revision": GUIDE_REVISION,
                    "stop_id": stop.stop_id,
                    "extra": "ignored-no-more",
                },
            )
            assert extra.status == 400
            assert not acknowledgements.is_acknowledged(
                reader_id, GUIDE_ID, GUIDE_REVISION, stop.stop_id
            )

            # The native form makes the real POST. Only the redirected response
            # carries the later-step bytes.
            with page.expect_navigation():
                page.locator("[data-hard-stop-ack]").click()
            expect(page.locator('[data-step="3"]')).to_be_visible()
            expect(page.locator('[data-step="4"]')).to_be_visible()
            assert SEALED_MARKER in page.content()
            assert acknowledgements.is_acknowledged(
                reader_id, GUIDE_ID, GUIDE_REVISION, stop.stop_id
            )
            assert page.evaluate(
                "key => window.localStorage.getItem(key)", pending_key
            ) is None
            ctx.close()
        finally:
            browser.close()


class _LyingIdStr(str):
    """str subclass whose equality and hash lie."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


class _StartswithLyingStr(str):
    """str subclass that claims any single-slash prefix."""

    def startswith(self, prefix, *args):
        return prefix == "/"


class _AliasStopStr(str):
    """str subclass aliasing a registered stop id's hash and equality."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash("step-2-battery")


def test_r4_subclass_hard_stop_ids_fail_closed():
    """str-subclass ids never register, match, or remove."""
    store = HardStopAcknowledgements()
    store.register_guide(
        GUIDE_ID, GUIDE_REVISION, ("step-2-battery",), "/guides/guide-A"
    )
    # A return path whose startswith() lies about its scheme is refused.
    with pytest.raises(ValueError, match="invalid hard-stop guide registration"):
        store.register_guide(
            GUIDE_ID,
            GUIDE_REVISION,
            ("step-2-battery",),
            _StartswithLyingStr("https://evil.example/x"),
        )
    # Subclass ids are refused at registration ...
    with pytest.raises(ValueError, match="invalid hard-stop guide registration"):
        store.register_guide(
            _LyingIdStr(GUIDE_ID),
            GUIDE_REVISION,
            ("step-2-battery",),
            "/guides/guide-A",
        )
    # ... fail closed on the request path without recording anything ...
    assert (
        store.acknowledge(
            "reader-A", GUIDE_ID, GUIDE_REVISION, _AliasStopStr("step-9-evil")
        )
        is None
    )
    assert (
        store.acknowledge("reader-A", GUIDE_ID, GUIDE_REVISION, "step-2-battery")
        == "/guides/guide-A"
    )
    assert (
        store.is_acknowledged(
            "reader-A", GUIDE_ID, GUIDE_REVISION, _AliasStopStr("step-9-evil")
        )
        is False
    )
    assert (
        store.is_acknowledged(
            "reader-A", GUIDE_ID, GUIDE_REVISION, "step-2-battery"
        )
        is True
    )
    # ... and cannot remove a registration.
    with pytest.raises(ValueError, match="invalid hard-stop guide id"):
        store.unregister_guide(_LyingIdStr(GUIDE_ID))
    assert (
        store.acknowledge("reader-A", GUIDE_ID, GUIDE_REVISION, "step-2-battery")
        == "/guides/guide-A"
    )
