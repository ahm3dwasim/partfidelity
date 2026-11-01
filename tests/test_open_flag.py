"""Second-check flag, confirm/undo, label-number mapping and keyboard contract: fakes only, no network."""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app import render
from app.contracts import RECEIPT_MODEL_SPARK_OPEN
from app.corrections import (
    CorrectionError,
    CorrectionStore,
    apply_corrections,
    confirmations,
    validated_op,
)
from app.guide_text import GuideTextError, load_guide_text
from app.guides import STATUS_UNCHANGED, STATUS_UNCHANGED_HELP, step_cards
from app.providers import open_readback as ob
from tests.test_export import JOB, _client, _post, guide  # noqa: F401  (fixture)

CFG = ob.load_config()
FLAG = "second-check-flag"
FAN, COVER = "fan-out.png", "cover-out.png"  # label image names of the export fixture's two parts


def call(label, phrase, decision="no", status="succeeded", gen=0):
    evidence = {"label": label, "phrase_id": ob.phrase_id(phrase)}
    if gen:
        evidence["gen"] = gen
    if decision:
        evidence["decision"] = decision
    return SimpleNamespace(model=RECEIPT_MODEL_SPARK_OPEN, status=status, evidence=evidence)


@pytest.fixture
def calls(monkeypatch):
    rows: list = []
    monkeypatch.setattr("app.routes.guides.list_calls", lambda ctx: tuple(rows))
    return rows


def make(guide, tmp_path):
    store = CorrectionStore(lambda j: tmp_path, "c.json", max_ops=9)
    return _client(guide, store), store


def page(client):
    return client.get(f"/guides/{JOB}").text


def row(body, phrase):
    """The <li> of one part row."""
    return next(li for li in re.findall(r"<li.*?</li>", body, re.S) if f"{phrase} —" in li)


# --- the flag: neutral, attached to the right label ----------------------

def test_flag_only_on_disagree_and_only_that_label(guide, tmp_path, calls):
    calls.append(call(COVER, "cover"))
    body = page(make(guide, tmp_path)[0])
    assert body.count(f'class="{FLAG}"') == 1
    assert FLAG in row(body, "cover") and FLAG not in row(body, "fan")
    assert CFG.flag_text == "Another AI check disagrees with this label. Review the highlighted part."
    assert CFG.flag_text in body and "fan —" in body and "cover —" in body  # labels untouched


def test_flag_never_says_the_label_is_wrong_or_styles_it_as_danger(guide, tmp_path, calls):
    calls.append(call(COVER, "cover"))
    body = page(make(guide, tmp_path)[0])
    block = re.search(rf'<div class="{FLAG}".*?</div>', body, re.S).group(0)
    assert not re.search(r"wrong|error|incorrect|mistake|danger|invalid", block, re.I)
    css = (Path(__file__).resolve().parents[1] / "static/ui/guide_step.css").read_text()
    rules = "".join(re.findall(rf"\.guide \.{FLAG}[^{{]*\{{[^}}]*\}}", css))
    assert rules and "danger" not in rules and "#c2331f" not in rules.lower()


@pytest.mark.parametrize("rows", [
    [call(FAN, "fan", "yes")],
    [call(FAN, "fan", None, "failed")],
    [],
])
def test_agree_or_unavailable_leaves_the_guide_byte_identical(guide, tmp_path, calls, rows, monkeypatch):
    client, _ = make(guide, tmp_path)
    monkeypatch.setattr("app.routes.guides.list_calls", lambda ctx: ())
    baseline = page(client)
    calls.extend(rows)
    monkeypatch.setattr("app.routes.guides.list_calls", lambda ctx: tuple(calls))
    assert FLAG not in page(client) and page(client) == baseline


def test_latest_receipt_of_a_generation_wins_and_is_keyed_by_label():
    assert ob.outcomes([call(FAN, "fan"), call(FAN, "fan", "yes")])[FAN].state == "agree"
    assert ob.outcomes([call(FAN, "fan", "yes"), call(FAN, "fan")])[FAN] == ob.Outcome(0, ob.phrase_id("fan"), "disagree")
    same_text = ob.outcomes([call(FAN, "fan"), call(COVER, "fan")])
    assert set(same_text) == {FAN, COVER}  # two labels with one text stay separate items


def test_wording_comes_from_config(guide, tmp_path, calls, monkeypatch):
    calls.append(call(COVER, "cover"))
    monkeypatch.setattr(ob, "load_config", lambda: replace(
        CFG, flag_text="Look again at this one", flag_confirm_label="Checked it",
        flag_aria_label="Look again"))
    body = page(make(guide, tmp_path)[0])
    assert "Look again at this one" in body and CFG.flag_text not in body
    assert ">Checked it<" in body and 'aria-label="Look again"' in body


def test_disabled_config_shows_nothing(guide, tmp_path, calls, monkeypatch):
    calls.append(call(COVER, "cover"))
    monkeypatch.setattr(ob, "load_config", lambda: replace(CFG, enabled=False))
    assert FLAG not in page(make(guide, tmp_path)[0])


# --- confirm, undo, persistence -------------------------------------------

def confirm(client, **over):
    return _post(client, **{"action": "confirm", "step": "0", "part": "1", "expect": "cover", "gen": "0", **over})


def test_confirm_records_a_user_confirmation_and_shows_a_confirmed_state(guide, tmp_path, calls):
    calls.append(call(COVER, "cover"))
    client, store = make(guide, tmp_path)
    r = confirm(client)
    assert r.headers["location"].endswith(f"fix=confirmed#flag-{COVER.replace('.', '-')}")
    (op,) = store.read(JOB)
    assert op["op"] == "confirm" and op["by"] == "user" and op["label"] == COVER and op["at"]
    body = page(client)
    block = row(body, "cover")
    assert CFG.flag_text not in block and CFG.flag_confirm_label not in block
    assert CFG.flag_confirmed_text in block and f'datetime="{op["at"]}"' in block
    assert f">{CFG.flag_undo_label}<" in block and 'value="unconfirm"' in block
    assert not re.search(r"verif|independent|AI check (agrees|passed)", block, re.I)
    assert "fan —" in body and "cover —" in body  # the label itself is unchanged


def test_undo_restores_the_flag(guide, tmp_path, calls):
    calls.append(call(COVER, "cover"))
    client, store = make(guide, tmp_path)
    confirm(client)
    r = _post(client, action="unconfirm", step="0", part="1", expect="cover", gen="0")
    assert "fix=unconfirmed" in r.headers["location"]
    block = row(page(client), "cover")
    assert CFG.flag_text in block and CFG.flag_confirm_label in block and CFG.flag_confirmed_text not in block
    assert confirmations(store.read(JOB)) == {}


def test_confirmation_persists_across_reload_and_is_kept_with_the_guide(guide, tmp_path, calls):
    calls.append(call(COVER, "cover"))
    client, _ = make(guide, tmp_path)
    confirm(client)
    reloaded, fresh = make(guide, tmp_path)  # a new store and app over the same files
    assert CFG.flag_confirmed_text in row(page(reloaded), "cover")
    assert (tmp_path / "c.json").is_file() and confirmations(fresh.read(JOB)).get(COVER, (None,))[0] == 0


def test_confirm_op_never_changes_the_guide_and_is_validated(guide, tmp_path):
    op = validated_op(guide, {"op": "confirm", "step": 0, "part": 1, "expect": "cover", "gen": 0}, max_chars=40)
    assert op["by"] == "user" and op["label"] == COVER and op["gen"] == 0
    assert apply_corrections(guide, [op]) == guide
    with pytest.raises(CorrectionError):  # a part with no label image has nothing to confirm
        validated_op(guide, {"op": "confirm", "step": 1, "part": 0, "expect": "battery", "gen": 0}, max_chars=40)
    store = CorrectionStore(lambda j: tmp_path, "c.json", max_ops=9)
    (tmp_path / "c.json").write_text('{"ops": [%s]}' % str({**op, "by": "ai"}).replace("'", '"'))
    with pytest.raises(CorrectionError):
        store.read(JOB)


# --- editing a label: clean slate, recheck for that label only ------------

def test_rename_clears_flag_and_confirmation_and_queues_one_recheck(guide, tmp_path, calls, monkeypatch):
    calls.append(call(COVER, "cover"))
    queued = []
    monkeypatch.setattr("app.routes.guides._live_second_check", lambda request: True)
    monkeypatch.setattr(ob, "start", lambda ctx, decisions, **kw: queued.append((list(decisions), kw)) or object())
    client, store = make(guide, tmp_path)
    confirm(client)
    r = _post(client, action="rename", step="0", part="1", expect="cover", phrase="top cover")
    assert "fix=renamed#" in r.headers["location"]
    body = page(client)
    assert FLAG not in body and "top cover —" in body  # old receipt is about the old text
    ((one, kw),) = queued
    assert [(d.slot.step_index, d.slot.part_index) for d in one] == [(0, 1)]
    assert one[0].provisional.detection.phrase == "top cover"
    assert kw["gens"] == {COVER: 1} and kw["budget"] is store
    # the recheck lands: a new disagreement is shown fresh, not as the old confirmation
    calls.append(call(COVER, "top cover", gen=1))
    block = row(page(client), "top cover")
    assert CFG.flag_text in block and CFG.flag_confirmed_text not in block


def test_rename_back_to_the_confirmed_text_needs_no_new_ack_only_if_unchanged(guide, tmp_path, calls):
    """A confirmation is for the text the person saw: it never carries to a different text."""
    calls.append(call(COVER, "cover"))
    client, store = make(guide, tmp_path)
    confirm(client)
    _post(client, action="rename", step="0", part="1", expect="cover", phrase="lid")
    calls.append(call(COVER, "lid", gen=1))
    assert CFG.flag_confirm_label in row(page(client), "lid")


def test_no_recheck_when_second_check_is_not_live(guide, tmp_path, calls, monkeypatch):
    queued = []
    monkeypatch.setattr(ob, "start", lambda *a, **k: queued.append(1))
    client, _ = make(guide, tmp_path)
    _post(client, action="rename", step="0", part="1", expect="cover", phrase="lid")
    assert queued == []


def test_flag_stays_on_the_same_label_after_a_delete_shifts_rows(guide, tmp_path, calls):
    calls.append(call(COVER, "cover"))
    client, _ = make(guide, tmp_path)
    _post(client, action="delete", step="0", part="0", expect="fan")  # cover moves from row 2 to row 1
    body = page(client)
    assert "fan —" not in body
    block = row(body, "cover")
    assert FLAG in block and 'name="part" value="0"' in block
    confirm(client, part="0")
    assert CFG.flag_confirmed_text in row(page(client), "cover")


# --- keyboard and screen reader contract ----------------------------------

def test_flag_is_announced_in_tab_order_and_the_row_is_described_by_it(guide, tmp_path, calls):
    calls.append(call(COVER, "cover"))
    block_row = row(page(make(guide, tmp_path)[0]), "cover")
    flag_id = re.search(rf'<div class="{FLAG}" id="([^"]+)"', block_row).group(1)
    assert f'aria-describedby="{flag_id}"' in block_row.split(">", 1)[0]
    assert 'role="status"' in block_row and 'tabindex="0"' in block_row and 'tabindex="-1"' not in block_row
    assert f'aria-label="{CFG.flag_aria_label}"' in block_row
    assert re.search(r'<button type="submit" name="action" value="confirm">', block_row)


# --- visual-change wording --------------------------------------------------

def test_status_wording_is_visual_only_and_comes_from_config(guide):
    assert STATUS_UNCHANGED == "No visual changes to this part"
    assert "does not confirm the label" in STATUS_UNCHANGED_HELP
    assert not re.search(r"correct|right|verified|safe", STATUS_UNCHANGED, re.I)
    assert [p.status for p in step_cards(guide)[0].parts] == [STATUS_UNCHANGED] * 2


def test_status_help_text_shows_on_the_guide_page(guide, tmp_path):
    assert STATUS_UNCHANGED_HELP in page(make(guide, tmp_path)[0])


def test_guide_text_config_fails_loudly(tmp_path):
    bad = tmp_path / "g.toml"
    bad.write_text('[guide_text]\nstatus_unchanged = "x"\n')
    with pytest.raises(GuideTextError):
        load_guide_text(bad)
    bad.write_text('[guide_text]\nstatus_unchanged = " "\nstatus_unchanged_help = "y"\n')
    with pytest.raises(GuideTextError):
        load_guide_text(bad)


# --- label number = row number = pin number --------------------------------

def badge_digit(path):
    """Read the digit drawn in the badge of one rendered label image."""
    w, h, px = render.read_png_rgb(path)

    def where(color):
        return {(i % w, i // w) for i in range(w * h) if tuple(px[i * 3:i * 3 + 3]) == color}

    fill = where(render._BADGE_FILL)
    assert fill, "no badge drawn"
    bx0, bx1 = min(x for x, _ in fill), max(x for x, _ in fill)
    by0, by1 = min(y for _, y in fill), max(y for _, y in fill)
    pts = {(x, y) for x, y in where(render._BADGE_GLYPH) if bx0 <= x <= bx1 and by0 <= y <= by1}
    x0, y0 = min(x for x, _ in pts), min(y for _, y in pts)
    scale = max(1, render.badge_radius(w, h) // 7)
    got = tuple(
        "".join("1" if (x0 + c * scale + scale // 2, y0 + r * scale + scale // 2) in pts else "0"
                for c in range(3))
        for r in range(5)
    )
    return render._GLYPHS.index(got) + 1


@pytest.mark.parametrize("number", [1, 2, 3])
def test_renderer_draws_the_number_it_is_given(tmp_path, number):
    from tests.test_guide import RecordingContext, _render
    out = _render(RecordingContext(), tmp_path, "p", "fan", number=number)
    assert badge_digit(out.rendered_image_path) == number




def test_pipeline_numbers_every_badge_row_and_pin_for_one_two_and_three_part_steps(
    tmp_path, monkeypatch
):
    """a real replay pipeline run with 1-, 2- and 3-part steps; the
    badge digit read from each rendered label, its guide row and its pin agree."""
    from decimal import Decimal

    from fastapi.testclient import TestClient

    from app import db
    from app.ledger import LifecycleRepositoryImpl
    from app.main import create_app
    from app.pipeline import PipelineClients, ReplayMetaFilesClient, ReplaySamClient, ReplaySparkClient, run_job
    from app.settings import Settings
    from app.spark import SparkRunResult
    from app.uploads_store import UploadStoreImpl
    from tests.test_pipeline import _step as spark_step, make_clip

    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    layout = {1: ["fan"], 2: ["screw", "cable"], 3: ["hinge", "speaker", "lid"]}
    boxes = {
        "fan": (12, 30, 80, 90), "screw": (10, 20, 60, 70), "cable": (200, 120, 260, 180),
        "hinge": (10, 10, 60, 60), "speaker": (130, 90, 190, 150), "lid": (240, 160, 300, 220),
    }
    windows = [(0.4, 2.0), (2.0, 4.0), (4.0, 5.8)]
    runs = [
        SparkRunResult(
            payload={
                "device_guess": "laptop", "confidence": 0.82,
                "steps": [
                    spark_step(n, *windows[n - 1], f"Step action {n}", [(p, "center") for p in phrases])
                    for n, phrases in layout.items()
                ],
            },
            cost_usd=Decimal("0.05"), latency_ms=900,
        )
        for _ in range(3)
    ]
    from app.sam import SamStageConfig

    sam = ReplaySamClient({p: (b, [0.72, 0.60]) for p, b in boxes.items()})
    # Production checks only fan and battery; these parts join the check list the way a
    # Modal-proven part would (configuration), so steps can carry three labels.
    sam.sam_stage_config = SamStageConfig(thresholds={p: 0.6 for p in boxes})
    monkeypatch.setattr("app.pipeline.is_checked_part", lambda phrase: phrase in boxes)
    monkeypatch.setattr("app.readback.PART_CHECK_THRESHOLDS", {p: 0.6 for p in boxes})
    clients = PipelineClients(spark=ReplaySparkClient(runs), sam=sam, meta=ReplayMetaFilesClient())
    result = run_job(
        make_clip(tmp_path / "clip.mp4"), clients=clients, uploads=UploadStoreImpl(database),
        lifecycle=LifecycleRepositoryImpl(database), anon_id="a", ip_hash="i", replay=True,
    )
    assert result.state == "done", result.error
    by_slot = {(d.slot.step_index, d.slot.part_index): d for d in result.guide.labels if d.provisional}
    assert sorted(by_slot) == [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    for (si, pi), d in by_slot.items():
        assert badge_digit(d.provisional.rendered_image_path) == pi + 1, (si, pi)

    client = TestClient(create_app(Settings(data_root=tmp_path / "data", db_path=database, production=False)))
    body = client.get(f"/guides/{result.job_id}").text
    for number, phrases in layout.items():
        article = re.search(rf'<article[^>]*id="step-{number}".*?</article>', body, re.S).group(0)
        for pos, phrase in enumerate(phrases, start=1):
            item = row(article, phrase)
            assert re.search(rf'<span class="pf-n" aria-hidden="true">{pos}</span>', item), phrase
            assert f'data-pin="pin-{number}-{pos}"' in item, phrase
            pin = re.search(rf'<span class="pf-pin" id="pin-{number}-{pos}"[^>]*>(\d+)</span>', article)
            assert pin and pin.group(1) == str(pos), phrase


# --- generation, stale forms, out-of-order, enqueue, budget ---

def rename(client, phrase, expect, **over):
    return _post(client, **{"action": "rename", "step": "0", "part": "1", "expect": expect, "phrase": phrase, **over})


def test_generation_counts_every_rename_of_a_label(guide):
    from app.corrections import generations

    ops = [
        {"op": "rename", "step": 0, "part": 1, "phrase": "lid"},
        {"op": "rename", "step": 0, "part": 1, "phrase": "cover"},
        {"op": "delete", "step": 0, "part": 0},
        {"op": "rename", "step": 0, "part": 0, "phrase": "top"},  # cover now sits at row 0
    ]
    assert generations(guide, []) == {FAN: 0, COVER: 0}
    assert generations(guide, ops[:2]) == {FAN: 0, COVER: 2}
    assert generations(guide, ops) == {COVER: 3}


def test_rename_away_and_back_never_revives_a_confirmation(guide, tmp_path, calls, monkeypatch):
    """a confirmation holds only for the generation it was made in."""
    calls.append(call(COVER, "cover"))
    client, store = make(guide, tmp_path)
    confirm(client)
    rename(client, "lid", "cover")
    rename(client, "cover", "lid")
    calls.append(call(COVER, "cover", gen=2))  # the recheck of the renamed-back label disagrees again
    block = row(page(client), "cover")
    assert CFG.flag_text in block and CFG.flag_confirm_label in block
    assert CFG.flag_confirmed_text not in block
    assert 'name="gen" value="2"' in block


def test_stale_confirm_form_is_refused(guide, tmp_path, calls):
    """a form from an older generation, or for a check that no longer disagrees, records nothing."""
    calls.append(call(COVER, "cover"))
    client, store = make(guide, tmp_path)
    rename(client, "lid", "cover")
    rename(client, "cover", "lid")
    calls.append(call(COVER, "cover", "yes", gen=2))  # the newer check agrees
    r = confirm(client, gen="0")  # the page the person opened before the renames
    assert "fix=stale" in r.headers["location"]
    r = confirm(client, gen="2")  # current generation, but nothing disagrees now
    assert "fix=stale" in r.headers["location"]
    assert confirmations(store.read(JOB)) == {}


def test_confirm_needs_a_current_disagreement_and_undo_a_current_confirmation(guide, tmp_path, calls):
    calls.extend([call(COVER, "cover"), call(COVER, "cover", "yes")])  # latest of gen 0 agrees
    client, store = make(guide, tmp_path)
    assert "fix=stale" in confirm(client).headers["location"]
    r = _post(client, action="unconfirm", step="0", part="1", expect="cover", gen="0")
    assert "fix=stale" in r.headers["location"]
    assert store.read(JOB) == []
    calls.append(call(COVER, "cover"))  # disagrees again
    assert "fix=confirmed" in confirm(client).headers["location"]
    assert "fix=stale" in confirm(client).headers["location"]  # already confirmed: a second confirm is stale
    assert len(store.read(JOB)) == 1


def test_outcome_is_chosen_by_generation_not_completion_order(guide, tmp_path, calls):
    """the newer label's disagreement lands first, the older label's agree lands last."""
    client, _ = make(guide, tmp_path)
    rename(client, "lid", "cover")
    rename(client, "top", "lid")
    calls.extend([call(COVER, "top", gen=2), call(COVER, "lid", "yes", gen=1)])
    assert ob.outcomes(calls)[COVER] == ob.Outcome(2, ob.phrase_id("top"), "disagree")
    block = row(page(client), "top")
    assert CFG.flag_text in block


def test_recheck_is_queued_once_with_the_captured_version_while_the_lock_is_held(
    guide, tmp_path, calls, monkeypatch
):
    """each rename enqueues exactly its own decision and generation, inside the correction lock."""
    import fcntl

    monkeypatch.setattr("app.routes.guides._live_second_check", lambda request: True)
    client, store = make(guide, tmp_path)
    queued = []

    def fake_start(ctx, decisions, **kw):
        with open(tmp_path / "c.lock", "a+b") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        queued.append(([d.provisional.detection.phrase for d in decisions], kw["gens"]))
        return object()

    monkeypatch.setattr(ob, "start", fake_start)
    rename(client, "lid", "cover")
    rename(client, "top", "lid")
    assert queued == [(["lid"], {COVER: 1}), (["top"], {COVER: 2})]


@pytest.mark.parametrize("failure", ["none", "raise"])
def test_recheck_that_cannot_start_is_shown_not_reported_as_success(
    guide, tmp_path, calls, monkeypatch, failure
):
    """a recheck that could not be queued says so after the rename and on the label."""
    monkeypatch.setattr("app.routes.guides._live_second_check", lambda request: True)
    recorded = []
    monkeypatch.setattr(ob, "record_unavailable", lambda ctx, decision, gen: recorded.append(
        (decision.provisional.detection.phrase, gen)) or calls.append(call(COVER, "lid", None, "failed", gen=gen)))

    def broken(*a, **k):
        if failure == "raise":
            raise RuntimeError("runner stopped")
        return None

    monkeypatch.setattr(ob, "start", broken)
    client, _ = make(guide, tmp_path)
    r = rename(client, "lid", "cover")
    assert "fix=renamed_unchecked" in r.headers["location"]
    assert recorded == [("lid", 1)]
    body = client.get(r.headers["location"].split("#")[0]).text
    from app.guide_text import TEXT
    assert TEXT["fix_renamed_unchecked"] in body and f">{TEXT['fix_renamed']}</p>" not in body
    assert CFG.flag_unavailable_text in row(body, "lid")


def test_a_failed_initial_check_still_adds_nothing(guide, tmp_path, calls):
    calls.append(call(COVER, "cover", None, "failed"))
    assert "second-check" not in page(make(guide, tmp_path)[0])


def test_unavailable_note_records_a_failed_receipt_for_that_generation(tmp_path):
    from tests.test_open_readback import decision
    from tests.test_providers import FakeCtx

    ctx = FakeCtx()
    ob.record_unavailable(ctx, decision(tmp_path), 3)
    (c,) = ctx.store.calls_for(str(ctx.job_id))
    assert c.status == "failed" and c.evidence["gen"] == 3 and c.evidence["cost"] == "skipped"
    assert ob.outcomes([c])["p0.png"] == ob.Outcome(3, ob.phrase_id("cooling fan"), "unavailable")


# --- every post-action message and the status help come from config -------

@pytest.mark.parametrize("code", ["renamed", "deleted", "confirmed", "unconfirmed", "renamed_unchecked",
                                  "stale", "blank", "bad", "gone", "locked", "full", "failed"])
def test_every_result_message_comes_from_config(guide, tmp_path, monkeypatch, code):
    from app import guide_text

    monkeypatch.setitem(guide_text.TEXT, f"fix_{code}", f"Configured {code} line")
    body = make(guide, tmp_path)[0].get(f"/guides/{JOB}?fix={code}").text
    assert f"Configured {code} line" in body


def test_result_messages_live_only_in_config():
    from app.guide_text import TEXT

    source = (Path(__file__).resolve().parents[1] / "app/routes/guides.py").read_text()
    for key, text in TEXT.items():
        if key.startswith("fix_"):
            assert text.split("{")[0][:20] not in source, key


def test_guide_text_loader_requires_every_result_message(tmp_path):
    from app.guide_text import CONFIG

    text = CONFIG.read_text()
    bad = tmp_path / "g.toml"
    bad.write_text("\n".join(line for line in text.splitlines() if not line.startswith("fix_stale")))
    with pytest.raises(GuideTextError):
        load_guide_text(bad)


def test_partial_and_gallery_use_the_configured_status_wording():
    from jinja2 import Environment, FileSystemLoader

    from app.guide_text import TEXT

    root = Path(__file__).resolve().parents[1]
    partial = (root / "templates/partials/ui/guide_step.html").read_text()
    assert STATUS_UNCHANGED not in partial and STATUS_UNCHANGED_HELP not in partial
    env = Environment(loader=FileSystemLoader(str(root / "templates/partials/ui")), autoescape=True)
    text = {**TEXT, "status_unchanged": "Visual line X", "status_unchanged_help": "Help line Y"}
    out = env.get_template("guide_step.html").render(
        number=1, window="0:01", frame_src="/x", action="a", phrases="fan", pins=[],
        parts=[{"phrase": "fan", "ok": True}], text=text)
    assert "Visual line X" in out and "Help line Y" in out
    import subprocess, sys
    done = subprocess.run([sys.executable, "tools/gallery_text.py", "--check"], cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    gallery = (root / "static/ui/gallery/guide_step.html").read_text()
    assert gallery.count(STATUS_UNCHANGED_HELP) == 3  # one help line per card with a visual-change badge


def test_gallery_generator_rewrites_from_config(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gallery_text", root / "tools/gallery_text.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = (root / "static/ui/gallery/guide_step.html").read_text()
    out = mod.apply(src, {"status_unchanged": "Visual line X", "status_unchanged_help": "Help line Y"})
    assert STATUS_UNCHANGED not in out and out.count("Visual line X") == src.count(STATUS_UNCHANGED)
    assert out.count("Help line Y") == 3


def test_export_carries_the_configured_status_and_help_lines(guide):
    from app.exports import follower_html

    out, _ = follower_html(guide, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}{part}.png")
    assert STATUS_UNCHANGED in out and STATUS_UNCHANGED_HELP in out


# --- real browser keyboard path  ---------------------------------

@pytest.fixture
def flag_server(guide, tmp_path, calls):
    """The guide page with one disagreement, served by uvicorn with the real static files."""
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn
    from fastapi.staticfiles import StaticFiles

    calls.append(call(COVER, "cover"))
    client, store = make(guide, tmp_path)
    app = client.app
    app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parents[1] / "static"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn never started"
        time.sleep(0.05)
    try:
        yield base, store
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_browser_keyboard_reaches_confirms_and_undoes_the_flag(flag_server):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")
    base, store = flag_server
    flag_id = "flag-" + COVER.replace(".", "-")
    active = "() => [document.activeElement.id, document.activeElement.tagName, document.activeElement.value || '']"
    with sync_playwright() as pw:
        browser = None
        for kwargs in ({"headless": True}, {"headless": True, "channel": "chrome"}):
            try:
                browser = pw.chromium.launch(**kwargs)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no Chromium or Chrome binary")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(f"{base}/guides/{JOB}")
            for _ in range(80):  # sequential Tab order from the top of the page
                page.keyboard.press("Tab")
                if page.evaluate(active)[0] == flag_id:
                    break
            else:
                raise AssertionError("Tab never reached the flag")
            page.keyboard.press("Tab")
            assert page.evaluate(active)[1:] == ["BUTTON", "confirm"]
            with page.expect_navigation():
                page.keyboard.press("Enter")
            page.wait_for_load_state("load")
            assert page.evaluate(active)[0] == flag_id  # focus lands on the note it changed
            assert CFG.flag_confirmed_text in page.inner_text(f"#{flag_id}")
            assert len(store.read(JOB)) == 1
            page.keyboard.press("Tab")
            assert page.evaluate(active)[1:] == ["BUTTON", "unconfirm"]
            with page.expect_navigation():
                page.keyboard.press("Space")
            page.wait_for_load_state("load")
            assert page.evaluate(active)[0] == flag_id
            assert CFG.flag_text in page.inner_text(f"#{flag_id}")
            page.keyboard.press("Tab")
            with page.expect_navigation():
                page.keyboard.press("Space")  # Space confirms too
            page.wait_for_load_state("load")
            assert page.evaluate(active)[0] == flag_id
            assert CFG.flag_confirmed_text in page.inner_text(f"#{flag_id}")
        finally:
            browser.close()
