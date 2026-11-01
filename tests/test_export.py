"""the one correction control, the numbered image zip, and the
offline follower export. Local fixtures only: no network, no model call."""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import re
import threading
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.contracts import Guide, JobId
from app import db
from app.deletion import delete_guide_files, sweep_expired
from app.ledger import LifecycleRepositoryImpl
from app.uploads_store import UploadStoreImpl
from app.render import badge_extent, read_png_rgb, read_mask, write_png_rgb
from app.corrections import (
    CorrectionError,
    CorrectionStore,
    apply_corrections,
    clean_phrase,
    validated_op,
)
from app.exports import (
    assert_self_contained,
    export_zip_bytes,
    follower_html,
    image_zip_bytes,
    ExportsConfigError,
    load_zip_date_time,
)
from app.guides import STATUS_UNCHANGED_HELP, assemble_guide, step_cards
from app.main import create_app
from app.pipeline import guide_path, save_guide
from app.routes.guides import router, set_corrections_store, set_guide_loader
from app.settings import Settings
from test_guide import (
    RecordingContext,
    _decision,
    _frame,
    _render,
    _slot,
    _step,
)

JOB = "job-guide"


@pytest.fixture
def guide(tmp_path) -> Guide:
    """Step 1: fan + cover, both labelled. Step 2: battery step (hard stop), unchecked."""
    ctx = RecordingContext()
    fan = _render(ctx, tmp_path, "fan", "fan")
    cover = _render(ctx, tmp_path, "cover", "cover", number=2)
    steps = [
        _step(1, ("fan", "cover")),
        _step(2, ("battery",), tools=("tweezers",)),
        _step(3, ("lid",)),
    ]
    steps[1] = type(steps[1])(
        **{**steps[1].__dict__, "hazards": ("battery",)}
    )
    frames = [
        _frame(tmp_path / f"frame{i}.png", color=(10 * i, 60, 90)) for i in (1, 2, 3)
    ]
    labels = [
        _decision(fan, slot=_slot(0, 0, "fan")),
        _decision(cover, slot=_slot(0, 1, "cover")),
    ]
    return assemble_guide(ctx, steps, frames, labels)


class MemoryStore(CorrectionStore):
    def __init__(self, **kw) -> None:
        super().__init__(lambda j: Path("/nonexistent"), "c.json", max_ops=kw.pop("max_ops", 5), **kw)
        self.ops: dict[str, list] = {}
        self.fail = False

    def locked(self, job_id):
        return contextlib.nullcontext()

    def read(self, job_id):
        return list(self.ops.get(job_id, []))

    def write(self, job_id, ops):
        if self.fail:
            raise CorrectionError("failed")
        self.ops[job_id] = list(ops)


def _client(guide: Guide, store=None) -> TestClient:
    app = FastAPI()
    root = guide.frame_paths[0].parent
    settings = Settings(data_root=root, db_path=root / "export-test.sqlite3", production=False)
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    app.state.settings = settings
    app.include_router(router)
    set_guide_loader(app, lambda j: guide if j == JOB else None)
    set_corrections_store(app, store)
    return TestClient(app, follow_redirects=False)


def _job_row(settings, job_id):
    """Main's guide loader hides guides without a jobs row; give the test guide one."""
    now = db.format_dt(db.utcnow())
    with db.connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO jobs (id, state, anon_id, ip_hash, video_hash, created_at,"
            " deadline_at, confirmed_at, completed_at, cost_cap_usd, readback_fired_n)"
            " VALUES (?, 'active', 'a', 'i', 'v', ?, ?, NULL, NULL, '0.50', 0)",
            (job_id, now, now),
        )


def _form(**fields) -> dict:
    return {"content": "&".join(f"{k}={v}" for k, v in fields.items()),
            "headers": {"content-type": "application/x-www-form-urlencoded"}}


def _post(client, **fields):
    return client.post(f"/guides/{JOB}/labels", **_form(**fields))


# --- correction logic --------------------------------------------------

def test_rename_changes_phrase_everywhere_and_keeps_the_label(guide):
    fixed = apply_corrections(
        guide, [{"op": "rename", "step": 0, "part": 1, "phrase": "top cover"}]
    )
    assert fixed.steps[0].parts[1].phrase == "top cover"
    card = step_cards(fixed)[0]
    assert [p.phrase for p in card.parts] == ["fan", "top cover"]
    assert [p.status for p in card.parts] == ["No visual changes to this part"] * 2
    assert card.parts[1].labelled_path == step_cards(guide)[0].parts[1].labelled_path
    assert fixed.labels[1].provisional.detection.phrase == "top cover"


def test_delete_removes_the_part_and_shifts_later_slots(guide):
    fixed = apply_corrections(guide, [{"op": "delete", "step": 0, "part": 0}])
    card = step_cards(fixed)[0]
    assert [p.phrase for p in card.parts] == ["cover"]
    assert card.parts[0].status == "No visual changes to this part"
    assert card.parts[0].labelled_path is not None
    assert len(fixed.labels) == 1 and fixed.labels[0].slot.part_index == 0


def test_ops_replay_in_order_against_the_corrected_guide(guide):
    ops = [
        {"op": "delete", "step": 0, "part": 0},
        {"op": "rename", "step": 0, "part": 0, "phrase": "lid"},
    ]
    assert [p.phrase for p in step_cards(apply_corrections(guide, ops))[0].parts] == ["lid"]
    with pytest.raises(CorrectionError, match="failed"):
        apply_corrections(guide, [{"op": "delete", "step": 9, "part": 0}, {"junk": 1}])


def test_clean_phrase_failures():
    assert clean_phrase("  big   fan ", 20) == "big fan"
    for raw, code in (("   ", "blank"), ("x" * 21, "long"), ("a\x07b", "bad"), (None, "bad")):
        with pytest.raises(CorrectionError) as err:
            clean_phrase(raw, 20)
        assert err.value.code == code


def test_validated_op_refuses_a_stale_or_out_of_range_target(guide):
    good = {"op": "rename", "step": 0, "part": 0, "expect": "fan", "phrase": "fan 2"}
    assert validated_op(guide, good, max_chars=20)["phrase"] == "fan 2"
    for bad, code in (
        ({**good, "expect": "other"}, "gone"),
        ({**good, "part": 5}, "gone"),
        ({**good, "op": "zap"}, "bad"),
        ({**good, "step": "0"}, "bad"),
    ):
        with pytest.raises(CorrectionError) as err:
            validated_op(guide, bad, max_chars=20)
        assert err.value.code == code


def test_store_roundtrip_failure_and_shared_sample(tmp_path):
    store = CorrectionStore(lambda j: tmp_path / j, "c.json", shared_job=lambda: "sample", max_ops=3)
    (tmp_path / "j").mkdir()
    assert store.read("j") == []
    store.write("j", [{"op": "delete", "step": 0, "part": 0}])
    assert store.read("j") == [{"op": "delete", "step": 0, "part": 0}]
    with pytest.raises(CorrectionError) as err:
        store.write("missing-dir", [])
    assert err.value.code == "failed"
    assert store.editable("j") and not store.editable("sample")


# --- the control on the page -------------------------------------------

def test_control_renders_once_per_part_inside_the_filmed_only_legend(guide):
    page = _client(guide, MemoryStore()).get(f"/guides/{JOB}").text
    assert page.count('class="fix-label"') == 4
    legend = re.search(r'<ul class="parts pf-legend pf-filmed-only">.*?</ul>', page, re.S).group(0)
    assert legend.count('class="fix-label"') >= 2
    assert 'name="action" value="rename"' in page and 'value="delete"' in page
    css = Path("static/ui/guide_step.css").read_text()
    assert "body.simple .pf-filmed-only { display: none; }" in css


def test_no_control_without_a_store_or_for_the_shared_sample(guide):
    assert "fix-label" not in _client(guide, None).get(f"/guides/{JOB}").text
    shared = MemoryStore(shared_job=lambda: JOB)
    client = _client(guide, shared)
    assert "fix-label" not in client.get(f"/guides/{JOB}").text
    assert _post(client, action="delete", step=0, part=0, expect="fan").headers["location"].startswith(
        f"/guides/{JOB}?fix=locked"
    )
    assert shared.ops == {}


def test_rename_and_delete_through_the_page(guide):
    store = MemoryStore()
    client = _client(guide, store)
    r = _post(client, action="rename", step=0, part=0, expect="fan", phrase="big+fan")
    assert r.status_code == 303 and "fix=renamed" in r.headers["location"]
    page = client.get(f"/guides/{JOB}?fix=renamed").text
    assert "big fan" in page and "Label renamed." in page and 'role="status"' in page
    r = _post(client, action="delete", step=0, part=1, expect="cover")
    assert "fix=deleted" in r.headers["location"]
    assert "cover" not in client.get(f"/guides/{JOB}").text.split('class="parts')[1].split("</ul>")[0]
    assert [op["op"] for op in store.ops[JOB]] == ["rename", "delete"]


@pytest.mark.parametrize(
    "fields, code",
    [
        (dict(action="rename", step=0, part=0, expect="fan", phrase="+++"), "blank"),
        (dict(action="rename", step=0, part=0, expect="fan", phrase=""), "blank"),
        (dict(action="rename", step=0, part=0, expect="fan", phrase="x" * 61), "long"),
        (dict(action="rename", step=0, part=0, expect="not+fan", phrase="a"), "gone"),
        (dict(action="rename", step=7, part=0, expect="fan", phrase="a"), "gone"),
        (dict(action="explode", step=0, part=0, expect="fan"), "bad"),
        (dict(action="rename", step="x", part=0, expect="fan", phrase="a"), "bad"),
    ],
)
def test_failure_states_change_nothing_and_say_why(guide, fields, code):
    store = MemoryStore()
    client = _client(guide, store)
    r = _post(client, **fields)
    assert r.status_code == 303 and f"fix={code}" in r.headers["location"]
    assert store.ops == {}
    page = client.get(f"/guides/{JOB}?fix={code}").text
    assert 'class="fix-note is-error" id="fix-note" role="alert"' in page


def test_failure_note_text_for_blank_and_long(guide):
    client = _client(guide, MemoryStore())
    assert "needs a name" in client.get(f"/guides/{JOB}?fix=blank").text
    assert "60 characters" in client.get(f"/guides/{JOB}?fix=long").text
    assert "fix-note" not in client.get(f"/guides/{JOB}?fix=nonsense").text


def test_save_failure_and_edit_cap(guide):
    store = MemoryStore(max_ops=1)
    client = _client(guide, store)
    store.fail = True
    assert "fix=failed" in _post(client, action="delete", step=0, part=0, expect="fan").headers["location"]
    store.fail = False
    assert "fix=deleted" in _post(client, action="delete", step=0, part=0, expect="fan").headers["location"]
    assert "fix=full" in _post(client, action="delete", step=0, part=0, expect="cover").headers["location"]
    assert len(store.ops[JOB]) == 1


def test_unknown_guide_404s_on_post(guide):
    assert _client(guide, MemoryStore()).post(
        "/guides/nope/labels", **_form(action="delete")
    ).status_code == 404


# --- numbered image zip -------------------------------------------------

def test_image_zip_is_numbered_in_step_order_with_labelled_images(guide):
    cards = step_cards(guide)
    data = image_zip_bytes(guide, entry_pattern="step-{number:02d}.png")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.namelist() == ["step-01.png", "step-02.png", "step-03.png"]
        assert z.read("step-02.png") == guide.frame_paths[1].read_bytes()
        assert z.read("step-03.png") == guide.frame_paths[2].read_bytes()
        composite = z.read("step-01.png")
        for removed in guide.labels:
            reduced = Guide(**{**guide.__dict__, "labels": tuple(
                label for label in guide.labels if label is not removed
            )})
            with zipfile.ZipFile(io.BytesIO(image_zip_bytes(
                reduced, entry_pattern="step-{number:02d}.png"
            ))) as smaller:
                assert smaller.read("step-01.png") != composite
        assert composite != cards[0].parts[0].labelled_path.read_bytes()
        assert composite != cards[0].parts[1].labelled_path.read_bytes()
        image = guide.frame_paths[0].parent / "composite.png"
        image.write_bytes(composite)
        width, height, pixels = read_png_rgb(image)
        _, _, original = read_png_rgb(cards[0].frame_path)
        for decision in guide.labels:
            mask = read_mask(decision.provisional.detection.mask_path, width, height)
            assert all(pixels[i * 3:i * 3 + 3] == original[i * 3:i * 3 + 3]
                       for i, marked in enumerate(mask) if marked)


@pytest.mark.parametrize("seed", [None, 3, 17, 43])
def test_composite_keeps_every_badge_outside_all_masks(guide, tmp_path, monkeypatch, seed):
    import app.exports as exports

    width, height, original = read_png_rgb(guide.frame_paths[0])
    rng = random.Random(seed)
    masks = []
    labels = []
    for index, decision in enumerate(guide.labels):
        if seed is None:
            mask = bytes([1] * (width * height))
        else:
            left = width // 3 + rng.randrange(-2, 3)
            top = height // 3 + rng.randrange(-2, 3)
            mask = bytes(
                left <= x < left + width // 4 and top <= y < top + height // 4
                for y in range(height) for x in range(width)
            )
        mask_path = tmp_path / f"mask-{seed}-{index}.png"
        write_png_rgb(mask_path, width, height,
                      bytes(channel for marked in mask for channel in (marked,) * 3))
        detection = replace(decision.provisional.detection, mask_path=mask_path)
        labels.append(replace(decision, provisional=replace(decision.provisional, detection=detection)))
        masks.append(mask)
    changed = replace(guide, labels=tuple(labels))
    drawn = []
    actual_draw = exports.draw_badge

    def record_badge(pixels, w, h, center, radius, number):
        drawn.append((w, h, center, radius, number))
        actual_draw(pixels, w, h, center, radius, number)

    monkeypatch.setattr(exports, "draw_badge", record_badge)
    with zipfile.ZipFile(io.BytesIO(image_zip_bytes(changed, entry_pattern="step-{number:02d}.png"))) as archive:
        image = tmp_path / "composite.png"
        image.write_bytes(archive.read("step-01.png"))
    out_w, out_h, pixels = read_png_rgb(image)
    assert len(drawn) == len(labels)
    assert [row[-1] for row in drawn] == list(range(1, len(labels) + 1))
    assert pixels != original
    union = bytes(any(mask[i] for mask in masks) for i in range(width * height))
    for w, h, (cx, cy), radius, number in drawn:
        assert (w, h) == (out_w, out_h)
        extent = badge_extent(radius)
        assert cy >= height or all(
            not union[y * width + x]
            for y in range(max(0, cy - extent), min(height, cy + extent + 1))
            for x in range(max(0, cx - extent), min(width, cx + extent + 1))
            if (x - cx) ** 2 + (y - cy) ** 2 <= extent ** 2
        )
        expected = bytearray(pixels)
        actual_draw(expected, out_w, out_h, (cx, cy), radius, number)
        assert expected == pixels  # each numbered badge survives the final composite
    if seed is None:
        assert out_h > height
        assert pixels[:width * height * 3] == original


def test_image_zip_route_and_link_at_guide_end(guide):
    client = _client(guide, MemoryStore())
    page = client.get(f"/guides/{JOB}").text
    token = re.search(r'/images.zip\?token=([^"&]+)', page).group(1)
    r = client.get(f"/guides/{JOB}/images.zip?token={token}")
    assert r.headers["content-type"] == "application/zip"
    assert "partfidelity-guide-images.zip" in r.headers["content-disposition"]
    assert zipfile.ZipFile(io.BytesIO(r.content)).namelist()[0] == "step-01.png"
    foot = page.split('<footer class="guide-foot">')[1]
    assert f"/guides/{JOB}/images.zip" in foot and f"/guides/{JOB}/export.zip" in foot


def test_zips_build_when_source_files_have_epoch_mtime(guide, tmp_path):
    sources = [p for p in tmp_path.rglob("*.png")]
    for path in sources:
        os.utime(path, (0, 0))
    frames = {c.frame_path: Path(c.frame_path).read_bytes() for c in step_cards(guide) if c.frame_path}
    images = image_zip_bytes(guide, entry_pattern="step-{number:02d}.png")
    export = export_zip_bytes(
        guide, html_name="guide.html", images_dir="images",
        frame_pattern="step-{number:02d}-original.png",
        label_pattern="step-{number:02d}-part{part}-labelled.png",
    )
    with zipfile.ZipFile(io.BytesIO(images)) as z:
        assert z.testzip() is None
        assert z.read("step-03.png") == frames[next(p for p in frames if p.name == "frame3.png")]
        assert {i.date_time for i in z.infolist()} == {load_zip_date_time()}
    with zipfile.ZipFile(io.BytesIO(export)) as z:
        assert z.testzip() is None
        assert z.read("images/step-03-original.png") == frames[next(p for p in frames if p.name == "frame3.png")]
        assert {i.date_time for i in z.infolist()} == {load_zip_date_time()}


# --- follower export ----------------------------------------------------

def _export(guide, **kw):
    return export_zip_bytes(
        guide, html_name="guide.html", images_dir="images",
        frame_pattern="step-{number:02d}-original.png",
        label_pattern="step-{number:02d}-part{part}-labelled.png", **kw,
    )


def test_export_is_one_html_plus_local_images_and_self_contained(guide):
    with zipfile.ZipFile(io.BytesIO(_export(guide))) as z:
        names = set(z.namelist())
        page = z.read("guide.html").decode()
        assert "guide.html" in names and all(n == "guide.html" or n.startswith("images/") for n in names)
        refs = set(re.findall(r'(?:src|data-original|data-labelled)="([^"]+)"', page))
        assert refs and all(r in names for r in refs)
    assert not re.search(r"https?://|<script[^>]*src|<link|<form|fetch\(|url\(", page)
    assert_self_contained(page, images_dir="images")


def test_export_defaults_to_reader_view_and_carries_the_guide_facts(guide):
    page = follower_html(guide, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}{part}.png")[0]
    assert '<body class="pf-theme-guide simple">' in page and 'id="simple-view" class="pf-view-box" checked' in page
    assert "tweezers" in page and "screwdriver" in page
    assert f'badge-{guide.difficulty.lower()}' in page and 'Hard: only if you can safely disconnect a laptop battery' in page and 'verdict-line">Should you try this? DRAFT' not in page
    assert page.count("No visual changes to this part") >= 2 and "Not checked" in page
    assert "<details" not in page and "Delete label" not in page


def test_export_hard_stop_blocks_later_steps_until_acknowledged(guide):
    page = follower_html(guide, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}{part}.png")[0]
    assert 'class="ack-box" id="ack-2"' in page and 'for="ack-2"' in page
    assert page.index('id="ack-2"') < page.index('class="after-stop"') < page.index('id="step-3"')
    # later steps fold to a row: number, title and the full warnings stay; the rest hides
    assert ".ack-box:not(:checked) ~ .after-stop .card > :not(header):not(.pf-hazard)" in page
    assert ".ack-box:not(:checked) ~ .after-stop { display: none; }" not in page


def test_export_uses_the_corrected_guide(guide):
    fixed = apply_corrections(guide, [{"op": "rename", "step": 0, "part": 0, "phrase": "big fan"}])
    with zipfile.ZipFile(io.BytesIO(_export(fixed))) as z:
        assert "big fan" in z.read("guide.html").decode()


@pytest.mark.parametrize(
    "snippet",
    [
        '<img src="https://x.test/a.png">',
        '<img src="//x.test/a.png">',
        '<a href="/guides/x">go</a>',
        "<div style=\"background:url(http://x.test/a.png)\"></div>",
        '<link rel="stylesheet" href="images/a.css">',
        "<script>fetch('images/a')</script>",
        "<style>@import 'images/a.css';</style>",
        '<form action="images/x"></form>',
    ],
)
def test_build_fails_on_an_external_reference(snippet):
    with pytest.raises(ValueError):
        assert_self_contained(snippet, images_dir="images")


def test_local_references_pass():
    assert_self_contained('<img src="images/a.png"><a href="#step-1">x</a>', images_dir="images")


def test_export_route_serves_the_zip(guide):
    client = _client(guide, MemoryStore())
    token = re.search(r'/export.zip\?token=([^"&]+)', client.get(f"/guides/{JOB}").text).group(1)
    r = client.get(f"/guides/{JOB}/export.zip?token={token}")
    assert r.status_code == 200 and "partfidelity-guide-offline.zip" in r.headers["content-disposition"]
    assert "guide.html" in zipfile.ZipFile(io.BytesIO(r.content)).namelist()


# --- settings -----------------------------------------------------------

def test_settings_defaults_and_validation(tmp_path):
    base = dict(data_root=tmp_path, db_path=tmp_path / "d.sqlite3", production=False)
    s = Settings(**base)
    assert s.correction_phrase_max_chars == 60 and s.export_images_dir == "images"
    for bad in (
        dict(correction_phrase_max_chars=0),
        dict(correction_max_ops=True),
        dict(image_zip_entry_pattern="a/{number}.png"),
        dict(export_label_pattern="{number}.png{bogus}"),
        dict(export_html_name="a/b.html"),
    ):
        with pytest.raises(ValueError):
            Settings(**base, **bad)


# --- wired app: sidecar on disk, sample stays shared ---------------------

def test_wired_app_persists_corrections_beside_the_guide(guide, tmp_path):
    root = tmp_path / "data"
    settings = Settings(data_root=root, db_path=root / "t.sqlite3", production=False)
    mine = Guide(**{**guide.__dict__, "job_id": JobId("mine")})
    app = create_app(settings)
    _job_row(settings, "mine")
    save_guide(mine, data_root=root, guide_filename=settings.guide_filename)
    before = guide_path("mine", data_root=root).read_text()
    client = TestClient(app, follow_redirects=False)
    r = client.post("/guides/mine/labels", **_form(action="rename", step=0, part=0, expect="fan", phrase="big+fan"))
    assert "fix=renamed" in r.headers["location"]
    sidecar = json.loads((guide_path("mine", data_root=root).parent / "corrections.json").read_text())
    assert sidecar["ops"][0]["phrase"] == "big fan"
    assert guide_path("mine", data_root=root).read_text() == before
    assert "big fan" in client.get("/guides/mine").text
    token = re.search(r'/images.zip\?token=([^"&]+)', client.get("/guides/mine").text).group(1)
    names = zipfile.ZipFile(io.BytesIO(client.get(f"/guides/mine/images.zip?token={token}").content)).namelist()
    assert names == ["step-01.png", "step-02.png", "step-03.png"]
    # the shared sample guide is one copy for every visitor: never editable
    (root / settings.sample_pointer_filename).write_text(json.dumps({"job_id": "mine"}))
    assert "fix=locked" in client.post(
        "/guides/mine/labels", **_form(action="delete", step=0, part=0, expect="big+fan")
    ).headers["location"]
    assert "fix-label" not in client.get("/guides/mine").text


def test_export_build_itself_fails_on_an_external_reference(guide, monkeypatch):
    monkeypatch.setattr("app.exports._css", lambda: "body{background:url(https://x.test/a.png)}")
    with pytest.raises(ValueError, match="external url"):
        _export(guide)


def test_concurrent_edits_serialise_on_the_store_lock(tmp_path):
    import threading

    (tmp_path / "j").mkdir()
    store = CorrectionStore(lambda j: tmp_path / j, "c.json", max_ops=50)

    def add(n):
        with store.locked("j"):
            ops = store.read("j")
            store.write("j", [*ops, {"op": "delete", "step": n, "part": 0}])

    threads = [threading.Thread(target=add, args=(n,)) for n in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(op["step"] for op in store.read("j")) == list(range(12))


def test_meta_refresh_is_refused():
    with pytest.raises(ValueError):
        assert_self_contained('<meta http-equiv="refresh" content="0">', images_dir="images")


def test_corrupt_corrections_are_not_overwritten(guide, tmp_path):
    path = tmp_path / 'c.json'
    path.write_text('{broken', encoding='utf-8')
    store = CorrectionStore(lambda _: tmp_path, 'c.json', max_ops=5)
    response = _post(_client(guide, store), action='rename', step='0', part='0', expect='fan', phrase='new')
    assert 'fix=failed' in response.headers['location']
    assert path.read_text(encoding='utf-8') == '{broken'


@pytest.mark.parametrize("operation", [
    {"op": "bogus", "step": 0, "part": 0},
    {"op": "delete", "step": True, "part": 0},
    {"op": "rename", "step": 0, "part": 0, "phrase": "  "},
])
def test_invalid_persisted_operation_is_not_overwritten(guide, tmp_path, operation):
    path = tmp_path / "c.json"
    original = json.dumps({"ops": [operation]}).encode()
    path.write_bytes(original)
    store = CorrectionStore(lambda _: tmp_path, "c.json", max_ops=5)
    response = _post(_client(guide, store), action="delete", step="0", part="0", expect="fan")
    assert "fix=failed" in response.headers["location"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("ops", [
    [{"op": "delete", "step": 9, "part": 0}],
    [{"op": "delete", "step": 0, "part": 9}],
    [{"op": "delete", "step": 0, "part": 0},
     {"op": "rename", "step": 0, "part": 1, "phrase": "cover"}],
])
def test_out_of_range_replay_refuses_post_without_rewriting(guide, tmp_path, ops):
    guide = replace(guide, steps=guide.steps[:1], frame_paths=guide.frame_paths[:1])
    path = tmp_path / "c.json"
    original = json.dumps({"ops": ops}, separators=(",", ":")).encode()
    path.write_bytes(original)
    store = CorrectionStore(lambda _: tmp_path, "c.json", max_ops=5)
    response = _post(_client(guide, store), action="delete", step="0", part="0", expect="fan")
    assert "fix=failed" in response.headers["location"]
    assert path.read_bytes() == original


def test_guide_get_reuses_one_live_export_link(guide):
    client = _client(guide, MemoryStore())
    for _ in range(3):
        assert client.get(f"/guides/{JOB}").status_code == 200
    with db.connect(guide.frame_paths[0].parent / "export-test.sqlite3") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM export_links WHERE guide_id = ?", (JOB,)
        ).fetchone()[0] == 1


def test_offline_page_uses_configured_retention(guide):
    page, _ = follower_html(
        guide, images_dir="images", frame_pattern="f{number}.png",
        label_pattern="l{number}{part}.png", retention_hours=7,
    )
    assert "expires after 7 hours" in page


@pytest.mark.parametrize('name,value', [
    ('image_zip_entry_pattern', '..'),
    ('image_zip_entry_pattern', 'C:{number}'),
    ('image_zip_entry_pattern', 'x\n{number}'),
    ('image_zip_entry_pattern', 'same.png'),
    ('export_frame_pattern', 'same.png'),
    ('export_label_pattern', 'step-{number}.png'),
    ('image_zip_name', 'bad"\r\nX-Test: injected'),
])
def test_export_settings_reject_unsafe_names(name, value):
    with pytest.raises(ValueError):
        Settings(data_root=Path('/tmp/data'), db_path=Path('/tmp/db'), production=False, **{name: value})


def test_export_settings_reject_collisions():
    with pytest.raises(ValueError):
        Settings(data_root=Path('/tmp/data'), db_path=Path('/tmp/db'), production=False, export_frame_pattern='{number}-1.png', export_label_pattern='{number}-{part}.png')


def test_export_settings_reject_collisions_at_later_steps():
    with pytest.raises(ValueError):
        Settings(data_root=Path('/tmp/data'), db_path=Path('/tmp/db'), production=False,
                 image_zip_entry_pattern='{number!s:.1}.png')


def test_corrupt_corrections_fail_closed_on_read(guide, tmp_path):
    path = tmp_path / 'c.json'
    path.write_text('{broken', encoding='utf-8')
    store = CorrectionStore(lambda _: tmp_path, 'c.json', max_ops=5)
    assert _client(guide, store).get(f'/guides/{JOB}').status_code == 503
    assert path.read_text(encoding='utf-8') == '{broken'


def test_unreadable_corrections_fail_without_overwriting(guide, tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_text('{"ops": [{"op": "delete", "step": 0, "part": 0}]}')
    store = CorrectionStore(lambda _: tmp_path, "c.json", max_ops=5)
    original = Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == path:
            raise PermissionError("unreadable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    response = _post(_client(guide, store), action="delete", step="0", part="0", expect="fan")
    assert "fix=failed" in response.headers["location"]
    assert original(path) == '{"ops": [{"op": "delete", "step": 0, "part": 0}]}'


def test_export_link_expires_but_published_guide_and_corrections_remain(guide, tmp_path, monkeypatch):
    root = tmp_path / "data"
    settings = Settings(data_root=root, db_path=root / "t.sqlite3", production=False)
    mine = Guide(**{**guide.__dict__, "job_id": JobId("mine")})
    app = create_app(settings)
    _job_row(settings, "mine")
    saved = save_guide(mine, data_root=root, guide_filename=settings.guide_filename)
    sidecar = saved.parent / settings.corrections_filename
    sidecar.write_text('{"ops": []}')
    now = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(db, "utcnow", lambda: now)
    client = TestClient(app)
    token = re.search(r'/images.zip\?token=([^"&]+)', client.get("/guides/mine").text).group(1)
    assert client.get(f"/guides/mine/images.zip?token={token}").status_code == 200
    assert client.get(f"/guides/mine/export.zip?token={token}").status_code == 200
    assert client.get("/guides/mine/images.zip").status_code == 404
    now += timedelta(hours=1)
    assert client.get(f"/guides/mine/images.zip?token={token}").status_code == 200
    now += timedelta(hours=24)
    assert client.get(f"/guides/mine/images.zip?token={token}").status_code == 404
    assert client.get(f"/guides/mine/export.zip?token={token}").status_code == 404
    result = sweep_expired(now, uploads=UploadStoreImpl(settings.db_path),
                           lifecycle=LifecycleRepositoryImpl(settings.db_path), client=object())
    assert result.export_links_purged == 1
    with db.connect(settings.db_path) as conn:
        assert conn.execute("SELECT 1 FROM export_links WHERE token = ?", (token,)).fetchone() is None
    assert saved.exists() and sidecar.exists()
    assert client.get("/guides/mine").status_code == 200


def test_deleting_guide_deletes_sidecar_and_links(guide, tmp_path):
    root = tmp_path / "data"
    settings = Settings(data_root=root, db_path=root / "t.sqlite3", production=False)
    app = create_app(settings)
    _job_row(settings, "mine")
    saved = save_guide(Guide(**{**guide.__dict__, "job_id": JobId("mine")}),
                       data_root=root, guide_filename=settings.guide_filename)
    sidecar = saved.parent / settings.corrections_filename
    sidecar.write_text('{"ops": []}')
    client = TestClient(app)
    token = re.search(r'/images.zip\?token=([^"&]+)', client.get("/guides/mine").text).group(1)
    delete_guide_files("mine", guide_filename=settings.guide_filename,
                       corrections_filename=settings.corrections_filename)
    assert not saved.exists() and not sidecar.exists()
    assert client.get(f"/guides/mine/images.zip?token={token}").status_code == 404
    with db.connect(settings.db_path) as conn:
        assert conn.execute("SELECT 1 FROM export_links WHERE guide_id = 'mine'").fetchone() is None


def test_correction_waiting_for_deletion_cannot_restore_sidecar(guide, tmp_path, monkeypatch):
    root = tmp_path / "data"
    settings = Settings(data_root=root, db_path=root / "t.sqlite3", production=False)
    app = create_app(settings)
    saved = save_guide(Guide(**{**guide.__dict__, "job_id": JobId("mine")}),
                       data_root=root, guide_filename=settings.guide_filename)
    sidecar = saved.parent / settings.corrections_filename
    sidecar.write_text('{"ops": []}')
    attempting_delete = threading.Event()
    store = app.state.corrections_store
    original_locked = CorrectionStore.locked

    @contextlib.contextmanager
    def observed_lock(self, job_id):
        if self is not store:
            attempting_delete.set()
        with original_locked(self, job_id):
            yield

    monkeypatch.setattr(CorrectionStore, "locked", observed_lock)
    deleter = threading.Thread(target=delete_guide_files, args=("mine",), kwargs={
        "guide_filename": settings.guide_filename,
        "corrections_filename": settings.corrections_filename,
    })
    with store.locked("mine"):
        deleter.start()
        assert attempting_delete.wait(5)
        assert saved.exists()
        store.write("mine", [{"op": "delete", "step": 0, "part": 0}])
    deleter.join(5)
    assert not deleter.is_alive()
    assert not sidecar.exists()
    with store.locked("mine"), pytest.raises(CorrectionError) as error:
        store.write("mine", [{"op": "delete", "step": 0, "part": 0}])
    assert error.value.code == "gone"
    assert not sidecar.exists()


def test_guide_text_cannot_inject_download_header(guide):
    hostile = Guide(**{**guide.__dict__, "verdict": 'title"\r\nX-Test: injected'})
    client = _client(hostile, MemoryStore())
    token = re.search(r'/export.zip\?token=([^"&]+)', client.get(f"/guides/{JOB}").text).group(1)
    response = client.get(f"/guides/{JOB}/export.zip?token={token}")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="partfidelity-guide-offline.zip"'
    assert "x-test" not in response.headers


@pytest.mark.parametrize("line", [
    "date_time = [2026, 13, 1, 0, 0, 0]",
    "date_time = [2026, 0, 1, 0, 0, 0]",
    "date_time = [2026, 1, 1, 24, 0, 0]",
    "date_time = [2026, 1, 1, 0, 0, 1]",
    "date_time = [1979, 1, 1, 0, 0, 0]",
    "date_time = [2108, 1, 1, 0, 0, 0]",
    "date_time = [2026, 2, 30, 0, 0, 0]",
    "date_time = [2026.0, 1, 1, 0, 0, 0]",
    "date_time = [2026, 1, 1, 0, 0]",
    "other = 1",
])
def test_invalid_zip_date_time_config_fails_loudly(tmp_path, line):
    path = tmp_path / "exports.toml"
    path.write_text(f"[zip]\n{line}\n")
    with pytest.raises(ExportsConfigError):
        load_zip_date_time(path)


def test_missing_zip_section_fails_loudly(tmp_path):
    path = tmp_path / "exports.toml"
    path.write_text("")
    with pytest.raises(ExportsConfigError):
        load_zip_date_time(path)


def test_unreadable_zip_config_fails_loudly(tmp_path):
    with pytest.raises(ExportsConfigError) as caught:
        load_zip_date_time(tmp_path / "missing.toml")
    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize("raw", [
    b"[zip]\ndate_time = [2026, 1, 1, 0, 0, 0]\n# \xff\xfe\n",
    b"[zip]\ndate_time = [2026, 1, 1, 0, 0, 0]\x00\n",
    b"[zip]\ndate_time = [2026, 99999999999999999999, 1, 0, 0, 0]\n",
    b"x = " + b"[" * 100000 + b"]" * 100000 + b"\n",
])
def test_odd_zip_config_bytes_raise_exports_config_error(tmp_path, raw):
    path = tmp_path / "exports.toml"
    path.write_bytes(raw)
    with pytest.raises(ExportsConfigError):
        load_zip_date_time(path)


def test_malformed_zip_config_fails_loudly(tmp_path):
    path = tmp_path / "exports.toml"
    path.write_text("[zip\n")
    with pytest.raises(ExportsConfigError) as caught:
        load_zip_date_time(path)
    assert caught.value.__cause__ is not None


def test_zip_config_that_is_not_a_table_fails_loudly(tmp_path):
    path = tmp_path / "exports.toml"
    path.write_text('zip = "no"\n')
    with pytest.raises(ExportsConfigError):
        load_zip_date_time(path)


def test_image_zip_draws_the_outline(guide, tmp_path):
    data = image_zip_bytes(guide, entry_pattern="step-{number:02d}.png")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in ("step-01.png", "step-02.png"):
            path = tmp_path / name
            path.write_bytes(archive.read(name))
    width, height, labelled = read_png_rgb(tmp_path / "step-01.png")
    amber = bytes((232, 112, 50))
    assert sum(
        1 for i in range(width * height)
        if labelled[3 * i : 3 * i + 3] == amber
    ) > 0
    _, _, unchecked = read_png_rgb(tmp_path / "step-02.png")
    assert amber not in (
        unchecked[3 * i : 3 * i + 3] for i in range(len(unchecked) // 3)
    )


def test_offline_intro_is_unnumbered_and_the_explanation_appears_once(guide):
    page, names = follower_html(guide, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}{part}.png")
    intro = next((c for c in step_cards(guide) if c.step.is_intro), None)
    if intro is not None and intro.frame_path is not None:
        assert names[intro.frame_path] == "intro-original.png"
    assert not re.search(r"step[- ]?0\b", page, re.I) and "step-00" not in "".join(names.values())
    assert page.count("No visual changes to this part") >= 2
    assert page.count(STATUS_UNCHANGED_HELP) == 1
