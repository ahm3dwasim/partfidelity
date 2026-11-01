"""Guide tests: assembly lifecycle, filtering, cards, toggle, routes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import db
from app.admission import admit_job
from app.contracts import (
    ArtifactId,
    Detection,
    Guide,
    GuideHazardAssessment,
    JobContext,
    JobId,
    LabelDecision,
    LabelSlot,
    PartRequest,
    ProvisionalRender,
    Step,
)
from app.guides import (
    STATUS_NOT_CHECKED,
    STATUS_UNCHANGED,
    assemble_guide,
    render_cards_html,
    status_text,
    step_cards,
)
from app.readback import run_readbacks
from app.render import render_label, write_png_rgb
from app.routes.guides import router, set_guide_loader
from test_readback import ScriptedReadbackClient, _yes


class RecordingStore:
    def __init__(self) -> None:
        self.registered: list[tuple[str, Path, ArtifactId]] = []
        self.kept: list[ArtifactId] = []
        self.deleted: list[ArtifactId] = []
        self._next = 0

    def register(self, kind: str, path: Path) -> ArtifactId:
        self._next += 1
        artifact_id = ArtifactId(f"guide-artifact-{self._next}")
        self.registered.append((kind, path, artifact_id))
        return artifact_id

    def mark_kept(self, artifact_id: ArtifactId) -> None:
        self.kept.append(artifact_id)

    def mark_deleted(self, artifact_id: ArtifactId) -> None:
        self.deleted.append(artifact_id)


class RecordingContext:
    def __init__(self, fail_on_finish: str | None = None) -> None:
        self._job_id = JobId("job-guide")
        self._store = RecordingStore()
        self._fail_on_finish = fail_on_finish
        self.finishes: list[str] = []
        self.finish_keeps: list[tuple[ArtifactId, ...]] = []
        self.recorded_calls: list[tuple] = []

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def artifacts(self) -> RecordingStore:
        return self._store

    def reserve(self, kind: str, worst_case_usd):  # pragma: no cover
        raise AssertionError("the guide reserves nothing")

    def record_call_with_event(self, key, call, event) -> None:
        self.recorded_calls.append((key, call, event))

    def finish(self, outcome, *, keep_artifact_ids=(), failed_output_names=None) -> None:
        self.finishes.append(outcome)
        self.finish_keeps.append(tuple(keep_artifact_ids))
        if outcome == self._fail_on_finish:
            raise RuntimeError("finish failed")


assert isinstance(RecordingContext(), JobContext)


def _frame(path: Path, color=(40, 60, 90)) -> Path:
    write_png_rgb(path, 200, 150, bytes(color) * (200 * 150))
    return path


def _mask(path: Path, box) -> Path:
    buf = bytearray(b"\x00\x00\x00" * (200 * 150))
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            buf[(y * 200 + x) * 3 : (y * 200 + x) * 3 + 3] = b"\xff\xff\xff"
    write_png_rgb(path, 200, 150, bytes(buf))
    return path


def _detection(mask_path: Path, phrase: str, box=(20, 30, 70, 80)) -> Detection:
    return Detection(phrase=phrase, mask_path=mask_path, score=0.8, box_xyxy=box)


def _step(number: int, phrases=("fan",), **over) -> Step:
    parts = tuple(PartRequest(phrase=p, position_hint="center") for p in phrases)
    return Step(
        number=number,
        start_s=10.0 * number,
        end_s=10.0 * number + 8.0,
        action=over.get("action", f"Do step {number}"),
        tools=over.get("tools", ("screwdriver",)),
        parts=parts,
        hazards=(),
        time_spread_s=over.get("time_spread_s", 1.0),
        time_needs_check=over.get("time_needs_check", False),
    )


def _render(ctx, tmp_path: Path, name: str, phrase: str, number: int = 1) -> ProvisionalRender:
    """``number`` is the part's position in its step: the badge must match its row."""
    frame = _frame(tmp_path / f"{name}-frame.png")
    mask_path = _mask(tmp_path / f"{name}-mask.png", (20, 30, 70, 80))
    return render_label(
        ctx, frame, _detection(mask_path, phrase), tmp_path / f"{name}-out.png", number=number
    )


def _slot(step_index: int = 0, part_index: int = 0, phrase: str = "fan") -> LabelSlot:
    return LabelSlot(
        step_index=step_index,
        part_index=part_index,
        part=PartRequest(phrase=phrase, position_hint="center"),
    )


def _decision(provisional=None, status="part_pixels_unchanged", readback="yes", slot=None):
    detection = provisional.detection if provisional is not None else None
    if slot is None:
        phrase = detection.phrase if detection is not None else "fan"
        slot = _slot(phrase=phrase)
    return LabelDecision(
        slot=slot,
        detection=detection,
        provisional=provisional,
        checked=status == "part_pixels_unchanged",
        status=status,
        readback=readback,
    )


def _client_for(guide: Guide) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    set_guide_loader(app, lambda job_id: guide if job_id == "job-guide" else None)
    return TestClient(app)


class _EqualToEverything:
    """Hostile equality probe: claims to equal every value it meets."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


class _LyingStr(str):
    """str subclass whose equality always claims True."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


class _LyingPartRequest(PartRequest):
    """PartRequest subclass whose equality always claims True."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


def test_provisional_render_is_registered_and_retained_until_final_assembly(tmp_path):
    ctx = RecordingContext()
    first = _render(ctx, tmp_path, "a", "fan")
    second = _render(ctx, tmp_path, "b", "battery")
    assert [kind for kind, _, _ in ctx.artifacts.registered] == [
        "provisional_render",
        "provisional_render",
    ]
    assert first.artifact_id != second.artifact_id

    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",)), _step(2, ("battery",))],
        [],
        [
            _decision(first, slot=_slot(0, 0, "fan")),
            _decision(second, slot=_slot(1, 0, "battery")),
        ],
    )
    assert ctx.artifacts.deleted == []
    assert ctx.artifacts.kept == []
    assert guide.labels == (
        _decision(first, slot=_slot(0, 0, "fan")),
        _decision(second, slot=_slot(1, 0, "battery")),
    )
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [(first.artifact_id, second.artifact_id)]


def test_successful_assembly_finishes_job_done_and_keeps_included_artifacts(tmp_path):
    ctx = RecordingContext()
    passed = _render(ctx, tmp_path, "a", "fan")
    skipped_readback = _render(ctx, tmp_path, "b", "battery")
    failed = _render(ctx, tmp_path, "c", "fan")
    labels = [
        _decision(passed, "part_pixels_unchanged", "yes", _slot(0, 0, "fan")),
        _decision(skipped_readback, "part_pixels_unchanged", "not_run", _slot(0, 1, "battery")),
        _decision(failed, "not_checked", "no", _slot(0, 0, "fan")),
        _decision(None, "not_checked", "not_run", _slot(0, 1, "battery")),
    ]
    guide = assemble_guide(ctx, [_step(1, ("fan", "battery"))], [], labels)
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [(passed.artifact_id, skipped_readback.artifact_id)]
    assert ctx.artifacts.kept == []
    assert [d.status for d in guide.labels] == [
        "part_pixels_unchanged",
        "part_pixels_unchanged",
    ]
    assert guide.job_id == JobId("job-guide")
    assert guide.draft_reasons != ()


def _hinted_step(number: int, phrase: str, hint: str) -> Step:
    return Step(
        number=number,
        start_s=10.0 * number,
        end_s=10.0 * number + 8.0,
        action=f"Do step {number}",
        tools=("screwdriver",),
        parts=(PartRequest(phrase=phrase, position_hint=hint),),
        hazards=(),
        time_spread_s=1.0,
        time_needs_check=False,
    )


def test_same_phrase_parts_flow_through_real_readback_into_real_assembly(tmp_path):
    """Real run_readbacks into real assemble_guide on the ledger's SQLite.

    Two same-phrase parts (two fans, different position hints) go through
    the real read-back into the real guide assembly on a real ledger job
    context, land on the right steps, and keep exactly their artifacts.
    """
    database = tmp_path / "e2e.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    ctx = admit_job("anon-e2e", "ip-e2e", "video-e2e")
    job_root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    steps = (_hinted_step(1, "fan", "top"), _hinted_step(2, "fan", "bottom"))
    provisionals: list[ProvisionalRender] = []
    frames: list[Path] = []
    for tag, hint in (("top", "top"), ("bottom", "bottom")):
        frame = _frame(job_root / f"{tag}-frame.png")
        mask = _mask(job_root / f"{tag}-mask.png", (20, 30, 70, 80))
        rendered = job_root / f"{tag}-label.png"
        write_png_rgb(rendered, 200, 150, bytes((10, 20, 30)) * (200 * 150))
        artifact_id = ctx.artifacts.register("provisional", rendered)
        provisionals.append(
            ProvisionalRender(
                detection=Detection(
                    phrase="fan", mask_path=mask, score=0.65,
                    box_xyxy=(20, 30, 70, 80),
                ),
                artifact_id=artifact_id,
                rendered_image_path=rendered,
                position_hint=hint,
            )
        )
        frames.append(frame)
    slots = [
        LabelSlot(
            step_index=index, part_index=0,
            part=PartRequest(phrase="fan", position_hint=hint),
        )
        for index, hint in enumerate(("top", "bottom"))
    ]
    decisions = run_readbacks(
        ctx, provisionals, slots=slots,
        client=ScriptedReadbackClient([], [_yes(), _yes()]),
    )
    assert [decision.readback for decision in decisions] == ["yes", "yes"]
    assert [decision.slot for decision in decisions] == slots
    guide = assemble_guide(ctx, steps, frames, decisions)
    assert guide.labels == decisions
    card_top, card_bottom = step_cards(guide)
    assert card_top.parts[0].status == STATUS_UNCHANGED
    assert card_top.parts[0].labelled_path == provisionals[0].rendered_image_path
    assert card_bottom.parts[0].status == STATUS_UNCHANGED
    assert card_bottom.parts[0].labelled_path == provisionals[1].rendered_image_path
    conn = db.connect(database)
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
        kept = {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM artifacts WHERE job_id = ? AND kept_at IS NOT NULL",
                (str(ctx.job_id),),
            ).fetchall()
        }
    finally:
        conn.close()
    assert job_state == "done"
    assert kept == {str(p.artifact_id) for p in provisionals}


def test_failed_real_assembly_keeps_nothing_on_real_context(tmp_path):
    """Lifecycle on the real ledger finish: a refused keep-set fails atomically."""
    from app.ledger import ArtifactNotFound

    database = tmp_path / "guide-fail.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    ctx = admit_job("anon-guide-fail", "ip-guide-fail", "video-guide-fail")
    job_root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    frame = _frame(job_root / "frame.png")
    mask = _mask(job_root / "mask.png", (20, 30, 70, 80))
    rendered = job_root / "label.png"
    write_png_rgb(rendered, 200, 150, bytes((10, 20, 30)) * (200 * 150))
    good_id = ctx.artifacts.register("provisional", rendered)
    bogus = ProvisionalRender(
        detection=Detection(
            phrase="fan", mask_path=mask, score=0.65, box_xyxy=(20, 30, 70, 80)
        ),
        artifact_id=ArtifactId("no-such-artifact"),
        rendered_image_path=rendered,
        position_hint="center",
    )
    decision = LabelDecision(
        slot=_slot(0, 0, "fan"),
        detection=bogus.detection,
        provisional=bogus,
        checked=True,
        status="part_pixels_unchanged",
        readback="yes",
    )
    with pytest.raises(ArtifactNotFound, match="unknown artifact"):
        assemble_guide(ctx, [_step(1)], [frame], [decision])
    conn = db.connect(database)
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
        kept = conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE job_id = ? AND kept_at IS NOT NULL",
            (str(ctx.job_id),),
        ).fetchone()[0]
        live = conn.execute(
            "SELECT delete_state FROM artifacts WHERE id = ?", (str(good_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert job_state == "failed"
    assert kept == 0
    assert live == "pending_delete"


def test_missing_detection_has_no_label(tmp_path):
    ctx = RecordingContext()
    missing = LabelDecision(
        slot=_slot(0, 0, "fan"),
        detection=None, provisional=None, checked=False,
        status="not_checked", readback="not_run",
    )
    assert status_text(missing) == STATUS_NOT_CHECKED
    assert status_text(None) == STATUS_NOT_CHECKED
    guide = assemble_guide(ctx, [_step(1, ("fan",))], [], [missing])
    assert guide.labels == ()
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.parts[0].labelled_path is None
    assert card.labelled_path is None


def test_failed_readback_becomes_not_checked(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    failed = _decision(provisional, "not_checked", "no", _slot(0, 0, "fan"))
    assert status_text(failed) == STATUS_NOT_CHECKED
    guide = assemble_guide(ctx, [_step(1, ("fan",))], [tmp_path / "a-frame.png"], [failed])
    assert guide.labels == ()
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.labelled_path is None
    assert ctx.finish_keeps == [()]
    assert ctx.artifacts.kept == []
    assert ctx.finishes == ["done"]


def test_final_guide_only_includes_passed_labels(tmp_path):
    ctx = RecordingContext()
    provisionals = {
        name: _render(ctx, tmp_path, name, name) for name in ("a", "b", "c", "d", "e")
    }
    labels = [
        _decision(provisionals["a"], "part_pixels_unchanged", "yes", _slot(0, 0, "a")),
        _decision(provisionals["b"], "part_pixels_unchanged", "not_run", _slot(0, 1, "b")),
        _decision(provisionals["c"], "part_pixels_unchanged", "no", _slot(0, 2, "c")),
        _decision(provisionals["d"], "not_checked", "not_run", _slot(0, 3, "d")),
        _decision(provisionals["e"], "not_checked", "no", _slot(0, 4, "e")),
    ]
    guide = assemble_guide(ctx, [_step(1, ("a", "b"))], [], labels)
    assert [d.provisional.detection.phrase for d in guide.labels] == ["a", "b"]


def test_guide_uses_real_frame_paths_only(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    frames = [tmp_path / "a-frame.png", tmp_path / "extra-frame.png"]
    _frame(frames[1])
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",)), _step(2, ("battery",))],
        frames,
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
    )
    allowed = set(frames) | {provisional.rendered_image_path}
    cards = step_cards(guide)
    for card in cards:
        assert card.frame_path in set(frames)
        assert card.labelled_path is None or card.labelled_path in allowed
    seen: list[Path] = []
    render_cards_html(cards, lambda path: seen.append(path) or f"/img/{len(seen)}")
    assert seen and set(seen) <= allowed


def test_original_labelled_toggle(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    frames = [tmp_path / "a-frame.png", tmp_path / "extra-frame.png"]
    _frame(frames[1])
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",)), _step(2, ("battery",))],
        frames,
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
    )
    body = _client_for(guide).get("/guides/job-guide").text
    # data-* feed the stage the display copies; the <img> src and data-*-png keep the stored PNG (fallback)
    labelled_url = "/guides/job-guide/image/1"
    original_url = "/guides/job-guide/image/0"
    assert f'data-original="{original_url}?w=800"' in body and f'data-original-png="{original_url}"' in body
    assert f'data-labelled="{labelled_url}?w=800"' in body and f'data-labelled-png="{labelled_url}"' in body
    assert f'<img id="shot-' in body and f'src="{labelled_url}"' in body
    assert 'data-original-srcset="' in body and 'data-labelled-srcset="' in body
    assert '<source type="image/webp" srcset="' in body
    assert 'class="toggle"' in body
    assert body.count('class="toggle"') == 1  # only the passed step toggles


def test_guide_page_and_images_serve_and_404(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [tmp_path / "a-frame.png"],
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
    )
    client = _client_for(guide)
    page = client.get("/guides/job-guide")
    assert page.status_code == 200
    assert STATUS_UNCHANGED in page.text
    assert "times are approximate" in page.text
    image = client.get("/guides/job-guide/image/1")
    assert image.status_code == 200
    assert image.content == provisional.rendered_image_path.read_bytes()
    assert client.get("/guides/job-guide/image/9").status_code == 404
    assert client.get("/guides/other-job").status_code == 404


def test_step_text_is_html_escaped(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    step = _step(1, ("fan",), action='Lift <script>alert("x")</script> & go')
    guide = assemble_guide(
        ctx, [step], [tmp_path / "a-frame.png"], [_decision(provisional, slot=_slot(0, 0, "fan"))]
    )
    body = _client_for(guide).get("/guides/job-guide").text
    assert '<script>alert("x")</script>' not in body
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in body
    assert "&amp; go" in body


def test_guide_page_without_loader_is_unavailable():
    app = FastAPI()
    app.include_router(router)
    assert TestClient(app).get("/guides/job-guide").status_code == 503


def test_two_app_instances_keep_their_own_loaders(tmp_path):
    ctx = RecordingContext()
    fan = _render(ctx, tmp_path, "a", "fan")
    battery = _render(ctx, tmp_path, "b", "battery")
    guide_a = assemble_guide(
        ctx,
        [_step(1, ("fan",), action="Alpha lift the fan")],
        [tmp_path / "a-frame.png"],
        [_decision(fan, slot=_slot(0, 0, "fan"))],
    )
    guide_b = assemble_guide(
        ctx,
        [_step(1, ("battery",), action="Beta lift the battery")],
        [tmp_path / "b-frame.png"],
        [_decision(battery, slot=_slot(0, 0, "battery"))],
    )
    app_a = FastAPI()
    app_a.include_router(router)
    set_guide_loader(app_a, lambda job_id: guide_a if job_id == "job-a" else None)
    app_b = FastAPI()
    app_b.include_router(router)
    set_guide_loader(app_b, lambda job_id: guide_b if job_id == "job-b" else None)

    body_a = TestClient(app_a).get("/guides/job-a").text
    body_b = TestClient(app_b).get("/guides/job-b").text
    assert "Alpha lift the fan" in body_a
    assert "Beta lift the battery" in body_b
    assert "Beta lift the battery" not in body_a
    assert "Alpha lift the fan" not in body_b
    assert TestClient(app_a).get("/guides/job-b").status_code == 404


def test_repeated_phrase_steps_keep_distinct_renders(tmp_path):
    ctx = RecordingContext()
    first = _render(ctx, tmp_path, "a", "fan")
    second = _render(ctx, tmp_path, "b", "fan")
    assert first.rendered_image_path != second.rendered_image_path
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",)), _step(2, ("fan",))],
        [],
        [
            _decision(first, slot=_slot(0, 0, "fan")),
            _decision(second, slot=_slot(1, 0, "fan")),
        ],
    )
    card_one, card_two = step_cards(guide)
    assert card_one.parts[0].labelled_path == first.rendered_image_path
    assert card_two.parts[0].labelled_path == second.rendered_image_path


def test_reversed_slot_delivery_still_lands_on_the_right_steps(tmp_path):
    """slot identity, never phrase order, places renders."""
    ctx = RecordingContext()
    first = _render(ctx, tmp_path, "a", "fan")
    second = _render(ctx, tmp_path, "b", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",)), _step(2, ("fan",))],
        [],
        [
            _decision(second, slot=_slot(1, 0, "fan")),
            _decision(first, slot=_slot(0, 0, "fan")),
        ],
    )
    card_one, card_two = step_cards(guide)
    assert card_one.parts[0].labelled_path == first.rendered_image_path
    assert card_two.parts[0].labelled_path == second.rendered_image_path


def test_slot_naming_another_part_stays_not_checked(tmp_path):
    ctx = RecordingContext()
    battery = _render(ctx, tmp_path, "b", "battery")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [],
        [_decision(battery, slot=_slot(0, 0, "battery"))],
    )
    assert guide.labels == ()
    assert ctx.finish_keeps == [()]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.parts[0].labelled_path is None
    assert card.labelled_path is None


def test_duplicate_slots_keep_first_decision_only(tmp_path):
    """a repeated slot keeps the first decision, drops the rest."""
    ctx = RecordingContext()
    first = _render(ctx, tmp_path, "a", "fan")
    second = _render(ctx, tmp_path, "b", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [],
        [
            _decision(first, slot=_slot(0, 0, "fan")),
            _decision(second, slot=_slot(0, 0, "fan")),
        ],
    )
    assert guide.labels == (_decision(first, slot=_slot(0, 0, "fan")),)
    assert ctx.finish_keeps == [(first.artifact_id,)]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_UNCHANGED
    assert card.parts[0].labelled_path == first.rendered_image_path


def test_out_of_range_slots_render_not_checked_and_keep_nothing(tmp_path):
    """out-of-range slots never land and never keep."""
    ctx = RecordingContext()
    first = _render(ctx, tmp_path, "a", "fan")
    second = _render(ctx, tmp_path, "b", "fan")
    third = _render(ctx, tmp_path, "c", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [],
        [
            _decision(first, slot=_slot(5, 0, "fan")),
            _decision(second, slot=_slot(0, 9, "fan")),
            _decision(third, slot=_slot(-1, 0, "fan")),
        ],
    )
    assert guide.labels == ()
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [()]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.parts[0].labelled_path is None


def test_slot_with_wrong_position_hint_renders_not_checked(tmp_path):
    """the full PartRequest must match, not just the phrase."""
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    step = _hinted_step(1, "fan", "top")
    wrong_hint = LabelSlot(
        step_index=0,
        part_index=0,
        part=PartRequest(phrase="fan", position_hint="bottom"),
    )
    guide = assemble_guide(ctx, [step], [], [_decision(provisional, slot=wrong_hint)])
    assert guide.labels == ()
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [()]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.parts[0].labelled_path is None


def test_lying_part_and_bool_index_slots_never_land_or_keep(tmp_path):
    """a hostile part and a bool index never land and never keep."""
    ctx = RecordingContext()
    evil = _render(ctx, tmp_path, "a", "fan")
    lying = _render(ctx, tmp_path, "b", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [],
        [
            _decision(
                evil,
                slot=LabelSlot(0, 0, _EqualToEverything()),  # type: ignore[arg-type]
            ),
            _decision(lying, slot=_slot(True, 0, "fan")),
        ],
    )
    assert guide.labels == ()
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [()]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.parts[0].labelled_path is None
    assert card.labelled_path is None


def test_lying_status_and_readback_never_count_as_passed(tmp_path):
    """lying status/read-back out."""
    ctx = RecordingContext()
    first = _render(ctx, tmp_path, "a", "fan")
    second = _render(ctx, tmp_path, "b", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [],
        [
            _decision(
                first, status=_EqualToEverything(), slot=_slot(0, 0, "fan")
            ),
            _decision(
                second, readback=_EqualToEverything(), slot=_slot(0, 0, "fan")
            ),
        ],
    )
    assert guide.labels == ()
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [()]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.parts[0].labelled_path is None


def test_r3_subclass_part_and_lying_fields_never_match(tmp_path):
    """a PartRequest subclass or lying str field never lands."""
    ctx = RecordingContext()
    evil = _render(ctx, tmp_path, "a", "fan")
    lying_phrase = _render(ctx, tmp_path, "b", "fan")
    lying_hint = _render(ctx, tmp_path, "c", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [],
        [
            _decision(
                evil,
                slot=LabelSlot(
                    0, 0, _LyingPartRequest(phrase="fan", position_hint="center")
                ),  # type: ignore[arg-type]
            ),
            _decision(
                lying_phrase,
                slot=LabelSlot(
                    0,
                    0,
                    PartRequest(
                        phrase=_LyingStr("other"), position_hint="center"
                    ),
                ),  # type: ignore[arg-type]
            ),
            _decision(
                lying_hint,
                slot=LabelSlot(
                    0,
                    0,
                    PartRequest(
                        phrase="fan", position_hint=_LyingStr("other")
                    ),
                ),  # type: ignore[arg-type]
            ),
        ],
    )
    assert guide.labels == ()
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [()]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED
    assert card.parts[0].labelled_path is None


def test_r3_subclass_status_readback_and_difficulty_fail_closed(tmp_path):
    """str-subclass status/read-back/difficulty never pass."""
    ctx = RecordingContext()
    bad_status = _render(ctx, tmp_path, "a", "fan")
    bad_readback = _render(ctx, tmp_path, "b", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [],
        [
            _decision(
                bad_status, status=_LyingStr("bogus"), slot=_slot(0, 0, "fan")
            ),
            _decision(
                bad_readback, readback=_LyingStr("bogus"), slot=_slot(0, 0, "fan")
            ),
        ],
        hazards=GuideHazardAssessment(
            difficulty=_LyingStr("bogus"),  # type: ignore[arg-type]
            verdict="READY",
            draft_reasons=("Hazard note.",),
        ),
    )
    assert guide.labels == ()
    assert (guide.difficulty, guide.verdict, guide.draft_reasons) == (
        "Hard",
        "DRAFT",
        ("Hazard assessment unavailable.",),
    )
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [()]
    (card,) = step_cards(guide)
    assert card.parts[0].status == STATUS_NOT_CHECKED


def test_r3_subclass_verdict_and_reasons_fall_back_to_default(tmp_path):
    """subclass verdict/reasons/tuple never pass through."""
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")

    class _LyingTuple(tuple):
        def __len__(self):
            return 1

    for hazards in (
        GuideHazardAssessment(
            difficulty="Easy",
            verdict=_LyingStr("READY"),  # type: ignore[arg-type]
            draft_reasons=("Hazard note.",),
        ),
        GuideHazardAssessment(
            difficulty="Easy",
            verdict="READY",
            draft_reasons=(_LyingStr("Hazard note."),),  # type: ignore[arg-type]
        ),
        GuideHazardAssessment(
            difficulty="Easy",
            verdict="READY",
            draft_reasons=_LyingTuple(("Hazard note.",)),  # type: ignore[arg-type]
        ),
    ):
        guide = assemble_guide(
            ctx,
            [_step(1)],
            [],
            [_decision(provisional, slot=_slot(0, 0, "fan"))],
            hazards=hazards,
        )
        assert (guide.difficulty, guide.verdict, guide.draft_reasons) == (
            "Hard",
            "DRAFT",
            ("Hazard assessment unavailable.",),
        )
    assert ctx.finishes == ["done", "done", "done"]


def test_every_passed_part_exposes_its_labelled_image(tmp_path):
    ctx = RecordingContext()
    fan = _render(ctx, tmp_path, "a", "fan")
    battery = _render(ctx, tmp_path, "b", "battery")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan", "battery"))],
        [tmp_path / "a-frame.png"],
        [
            _decision(fan, slot=_slot(0, 0, "fan")),
            _decision(battery, slot=_slot(0, 1, "battery")),
        ],
    )
    (card,) = step_cards(guide)
    assert card.parts[0].labelled_path == fan.rendered_image_path
    assert card.parts[1].labelled_path == battery.rendered_image_path
    client = _client_for(guide)
    first = client.get("/guides/job-guide/image/1")
    assert first.status_code == 200
    assert first.content == fan.rendered_image_path.read_bytes()
    second = client.get("/guides/job-guide/image/2")
    assert second.status_code == 200
    assert second.content == battery.rendered_image_path.read_bytes()
    assert client.get("/guides/job-guide").text.count('class="toggle"') == 2


def test_done_finish_carries_the_whole_keep_set_at_once(tmp_path):
    ctx = RecordingContext()
    first = _render(ctx, tmp_path, "a", "fan")
    second = _render(ctx, tmp_path, "b", "battery")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan", "battery"))],
        [],
        [
            _decision(first, slot=_slot(0, 0, "fan")),
            _decision(second, slot=_slot(0, 1, "battery")),
        ],
    )
    assert ctx.finishes == ["done"]
    assert ctx.finish_keeps == [(first.artifact_id, second.artifact_id)]
    assert ctx.artifacts.kept == []  # single atomic keep-set; no per-artifact keeps
    assert ctx.artifacts.deleted == []
    assert len(guide.labels) == 2


class _AlwaysFailingFinishContext(RecordingContext):
    """RecordingContext whose every finish raises (terminal outage)."""

    def finish(self, outcome, *, keep_artifact_ids=(), failed_output_names=None) -> None:
        self.finishes.append(outcome)
        self.finish_keeps.append(tuple(keep_artifact_ids))
        raise RuntimeError(f"{outcome} exploded")


def test_failed_finish_preserves_original_error(tmp_path):
    ctx = _AlwaysFailingFinishContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    with pytest.raises(RuntimeError, match="done exploded"):
        assemble_guide(
            ctx, [_step(1)], [], [_decision(provisional, slot=_slot(0, 0, "fan"))]
        )
    assert ctx.finishes == ["done", "failed"]


def test_failed_done_transition_attempts_failed_and_keeps_nothing(tmp_path):
    ctx = RecordingContext(fail_on_finish="done")
    provisional = _render(ctx, tmp_path, "a", "fan")
    with pytest.raises(RuntimeError, match="finish failed"):
        assemble_guide(
            ctx, [_step(1)], [], [_decision(provisional, slot=_slot(0, 0, "fan"))]
        )
    assert ctx.finishes == ["done", "failed"]
    assert ctx.finish_keeps == [(provisional.artifact_id,), ()]
    assert ctx.artifacts.kept == []
    assert ctx.artifacts.deleted == []


def test_difficulty_comes_from_provided_hazard_verbatim(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1)],
        [],
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
        hazards=GuideHazardAssessment(
            difficulty="Easy", verdict="READY", draft_reasons=("Hazard note.",)
        ),
    )
    assert guide.difficulty == "Easy"
    assert guide.verdict == "READY"
    assert guide.draft_reasons == ("Hazard note.",)
    assert ctx.finishes == ["done"]


def test_absent_hazard_fails_closed_as_draft(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    guide = assemble_guide(
        ctx, [_step(1)], [], [_decision(provisional, slot=_slot(0, 0, "fan"))]
    )
    assert ctx.finishes == ["done"]
    assert (guide.difficulty, guide.verdict, guide.draft_reasons) == (
        GuideHazardAssessment().difficulty,
        GuideHazardAssessment().verdict,
        GuideHazardAssessment().draft_reasons,
    )
    assert guide.difficulty == "Hard"
    assert guide.verdict == "DRAFT"
    assert guide.draft_reasons == ("Hazard assessment unavailable.",)
    assert ctx.finish_keeps == [(provisional.artifact_id,)]


@pytest.mark.parametrize(
    "hazards",
    [
        GuideHazardAssessment(difficulty="Bogus"),  # type: ignore[arg-type]
        GuideHazardAssessment(difficulty="Easy", verdict=""),
        GuideHazardAssessment(difficulty="Easy", verdict="READY", draft_reasons=()),
        GuideHazardAssessment(
            difficulty="Easy", verdict="", draft_reasons=("",)
        ),
        GuideHazardAssessment(
            difficulty="Easy", verdict="READY", draft_reasons=["Hazard note."]
        ),  # type: ignore[arg-type]
        "not-an-assessment",
    ],
    ids=["bad-difficulty", "empty-verdict", "empty-reasons", "empty-reason",
         "list-reasons", "wrong-type"],
)
def test_malformed_hazard_falls_back_to_closed_default(tmp_path, hazards):
    """a malformed projection never raises and never corrupts."""
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1)],
        [],
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
        hazards=hazards,  # type: ignore[arg-type]
    )
    assert ctx.finishes == ["done"]
    assert (guide.difficulty, guide.verdict, guide.draft_reasons) == (
        "Hard",
        "DRAFT",
        ("Hazard assessment unavailable.",),
    )
    assert ctx.finish_keeps == [(provisional.artifact_id,)]


def test_lying_difficulty_falls_back_to_closed_default(tmp_path):
    """a difficulty equal to everything still yields the default."""
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1)],
        [],
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
        hazards=GuideHazardAssessment(
            difficulty=_EqualToEverything(),  # type: ignore[arg-type]
            verdict="READY",
            draft_reasons=("Hazard note.",),
        ),
    )
    assert ctx.finishes == ["done"]
    assert (guide.difficulty, guide.verdict, guide.draft_reasons) == (
        "Hard",
        "DRAFT",
        ("Hazard assessment unavailable.",),
    )
    assert ctx.finish_keeps == [(provisional.artifact_id,)]


def test_guides_invents_no_difficulty_value():
    import app.guides

    source = Path(app.guides.__file__).read_text(encoding="utf-8")
    assert "app.hazards" not in source
    assert "GuideHazardAssessment" in source
    for literal in ("Easy", "Medium", "Hard"):
        assert f'"{literal}"' not in source, literal
        assert f"'{literal}'" not in source, literal


def test_placeholder_like_step_text_is_preserved(tmp_path):
    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    step = _step(1, ("fan",), action="Keep {{GUIDE_CSS}} and {{CARDS}} literally")
    guide = assemble_guide(
        ctx,
        [step],
        [tmp_path / "a-frame.png"],
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
    )
    body = _client_for(guide).get("/guides/job-guide").text
    assert "{{GUIDE_CSS}}" in body
    assert "{{CARDS}}" in body
    assert body.count("--paper:") == 1  # stylesheet inlined exactly once


def test_loader_cleared_between_check_and_call_still_serves(tmp_path):
    # Deterministic simulation of a concurrent set_guide_loader(app, None):
    # clear the app state after the loader's first statement runs (line
    # events fire before each line executes, so the second event lands
    # between the None check and the call). A snapshot serves; a double
    # read calls None.
    import app.routes.guides as guides_mod
    from app.routes.guides import guide_page, set_guide_loader
    from starlette.requests import Request

    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "a", "fan")
    guide = assemble_guide(
        ctx,
        [_step(1, ("fan",))],
        [tmp_path / "a-frame.png"],
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
    )
    app = FastAPI()
    app.include_router(router)
    set_guide_loader(app, lambda job_id: guide)
    request = Request({"type": "http", "app": app, "headers": []})
    state = {"events": 0, "flipped": False}

    def tracer(frame, event, arg):
        if frame.f_code is guides_mod._load_guide_or_raise.__code__ and event == "line":
            state["events"] += 1
            if state["events"] == 2:
                state["flipped"] = True
                set_guide_loader(app, None)
                sys.settrace(None)
        return tracer

    sys.settrace(tracer)
    try:
        response = guide_page("job-guide", request)
    finally:
        sys.settrace(None)
    assert state["flipped"]
    assert response.status_code == 200
