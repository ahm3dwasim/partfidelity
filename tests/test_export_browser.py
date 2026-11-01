"""The correction control in a real browser: the control stays out of reader view, its
failure shows, and the offline export makes zero network requests, renders
every image and still blocks at the hard stop. Local server, local files."""

from __future__ import annotations

import io
import socket
import threading
import time
import zipfile

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.guides import router, set_corrections_store, set_guide_loader
from test_export import JOB, MemoryStore, guide, _export  # noqa: F401 (fixture)
from test_reader_browser import browser  # noqa: F401 (fixture)


@pytest.fixture
def server(guide):
    app = FastAPI()
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="static"))
    set_guide_loader(app, lambda j: guide if j == JOB else None)
    set_corrections_store(app, MemoryStore())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server never started"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.mark.parametrize("size", [(375, 800), (1280, 800)])
def test_control_hidden_in_reader_view_and_failure_is_shown(browser, server, size):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    try:
        page.goto(f"{server}/guides/{JOB}")
        controls = page.locator("details.fix-label summary")
        assert controls.first.is_visible()
        page.locator("label.pf-view-label").click()  # I'm fixing this (reader view)
        assert page.evaluate("document.body.classList.contains('simple')")
        assert not controls.first.is_visible()
        page.locator("label.pf-view-label").click()
        controls.first.click()
        page.locator("input[name=phrase]").first.fill("")
        page.get_by_role("button", name="Rename").first.click()
        page.wait_for_selector(".fix-note.is-error")
        assert "needs a name" in page.locator(".fix-note").inner_text()
    finally:
        page.close()


def test_offline_export_makes_no_request_and_blocks_at_the_stop(browser, guide, tmp_path):
    with zipfile.ZipFile(io.BytesIO(_export(guide))) as z:
        z.extractall(tmp_path)
    context = browser.new_context(offline=True, viewport={"width": 375, "height": 800})
    requests: list[str] = []
    context.on("request", lambda r: requests.append(r.url))
    try:
        page = context.new_page()
        page.goto((tmp_path / "guide.html").as_uri())
        page.wait_for_load_state("load")
        assert all(u.startswith("file://") for u in requests), requests
        sizes = page.evaluate("[...document.images].map(i => i.complete && i.naturalWidth)")
        assert sizes and all(sizes), sizes
        assert page.locator("details.fix-label").count() == 0
        assert not page.locator("#step-3 .action").is_visible()  # instructions fold ...
        assert page.locator("#step-3 h2").is_visible()  # ... the row and its warnings stay
        hazards = page.locator("#step-3 .pf-hazard")
        assert all(hazards.nth(i).is_visible() for i in range(hazards.count()))
        page.get_by_text("I understand. Show the next steps.").click()
        assert page.locator("#step-3 .action").is_visible()
    finally:
        context.close()
