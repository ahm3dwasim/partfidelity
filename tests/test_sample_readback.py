"""The sample's one-shot Spark Open check: provider mocked, nothing leaves the process."""
from __future__ import annotations

import json
import threading
from dataclasses import replace
from decimal import Decimal

import pytest

from app import proof_links, sample_readback, service
from app.contracts import RECEIPT_MODEL_SPARK_OPEN
from app.evidence import list_calls
from app.ledger import JobContextImpl
from app.admission import job_dir
from app.contracts import JobId
from app.pipeline import load_guide
from app.providers import open_readback as ob
from app.providers._transport import HttpResponse
from test_real_sample import client  # noqa: F401 (fixture)

CFG = sample_readback.SampleReadbackConfig(True, ("fan", "battery"))
ENV = {"BASETEN_API_KEY": "k-test"}


class Fake:
    def __init__(self):
        self.posts = []

    def post(self, url, *, headers, body, timeout_s):
        self.posts.append(json.loads(body))
        out = {"choices": [{"message": {"content": "yes"}}], "usage": {"prompt_tokens": 470, "completion_tokens": 60}}
        return HttpResponse(200, json.dumps(out).encode(), {})


@pytest.fixture()
def sample(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ob, "_jpeg_data_url", lambda *a: "data:image/jpeg;base64,AAAA")
    settings = client.app.state.settings
    job_id = service.ensure_sample_job(settings)
    ctx = JobContextImpl(settings.db_path, JobId(job_id), job_dir(job_id, root=settings.data_root))
    return client, settings, job_id, ctx


def _rows(ctx):
    return [c for c in list_calls(ctx) if c.model == RECEIPT_MODEL_SPARK_OPEN]


def _run(settings, fake, cfg=CFG):
    batch = sample_readback.run_once(settings, cfg=cfg, transport=fake, environ=ENV)
    if batch:
        assert batch.join(10)
    return batch


def test_checks_only_fan_and_battery_labels_once_and_idempotently(sample):
    _, settings, job_id, ctx = sample
    guide = load_guide(job_id, data_root=settings.data_root, guide_filename=settings.guide_filename)
    before = [(d.detection.phrase, d.provisional.rendered_image_path.name) for d in guide.labels]
    fake = Fake()
    _run(settings, fake)
    rows = _rows(ctx)
    real = [r for r in rows if r.evidence.get("reason") != sample_readback.PENDING_REASON]
    assert len(real) == len(fake.posts) > 0
    assert {r.evidence["phrase"] for r in rows} <= {"fan", "battery"} and all(r.status == "succeeded" for r in real)
    sent, count = len(fake.posts), len(rows)
    assert _run(settings, fake) is None and len(fake.posts) == sent and len(_rows(ctx)) == count  # second run: nothing
    after = load_guide(job_id, data_root=settings.data_root, guide_filename=settings.guide_filename)
    assert [(d.detection.phrase, d.provisional.rendered_image_path.name) for d in after.labels] == before


def test_a_label_outside_the_configured_phrases_is_skipped(sample):
    _, settings, _, ctx = sample
    fake = Fake()
    _run(settings, fake, replace(CFG, labels=("fan",)))
    assert {r.evidence["phrase"] for r in _rows(ctx)} == {"fan"}


def test_spend_stays_within_the_job_caps(sample):
    _, settings, job_id, ctx = sample
    ocfg = ob.load_config()
    tight = replace(ocfg, max_calls_per_job=2)
    fake = Fake()
    batch = sample_readback.run_once(settings, cfg=CFG, ocfg=tight, transport=fake, environ=ENV)
    batch.join(10)
    rows = [r for r in _rows(ctx) if r.evidence.get("reason") != sample_readback.PENDING_REASON]
    assert len(fake.posts) == 2 and len(rows) == 4  # two sent, the rest recorded as budget reached
    assert sum(r.estimated_cost_usd for r in rows) <= ocfg.max_cost_usd_per_job
    assert sum(r.status == "failed" and r.evidence.get("reason") == "second-check budget reached" for r in rows) == 2


def test_disabled_config_sends_nothing(sample):
    _, settings, _, ctx = sample
    fake = Fake()
    assert _run(settings, fake, replace(CFG, enabled=False)) is None
    assert fake.posts == [] and _rows(ctx) == []


def test_plan_counts_calls_and_worst_case(sample):
    _, settings, job_id, _ = sample
    guide = load_guide(job_id, data_root=settings.data_root, guide_filename=settings.guide_filename)
    ocfg = ob.load_config()
    p = sample_readback.plan(guide.labels, CFG, ocfg)
    assert p["calls"] == len(p["labels"]) <= ocfg.max_calls_per_job
    assert p["worst_case_usd"] <= ocfg.max_cost_usd_per_job
    assert sample_readback.plan(guide.labels, CFG, ocfg, done_labels=[n for n, _ in p["labels"]])["calls"] == 0


def test_page_wording_switches_only_on_real_rows(sample):
    client, settings, job_id, ctx = sample
    links = proof_links.load()
    guide_page = lambda: client.get("/guide/sample", follow_redirects=True).text  # noqa: E731
    calls_page = lambda: client.get(links["sample_calls"]).text  # noqa: E731
    assert links["open_weights_note"] in guide_page() and links["open_weights_note"] in calls_page()
    _run(settings, Fake())
    assert _rows(ctx)
    page = guide_page()
    assert links["open_weights_note"] not in page and links["open_weights_sample_note"] in page
    assert links["open_weights_note"] not in calls_page() and "<td>spark-open</td>" in calls_page()
    assert "verified" not in links["open_weights_sample_note"].lower()


def test_failed_rows_do_not_switch_the_wording(sample):
    client, settings, job_id, ctx = sample
    sample_readback.run_once(settings, cfg=CFG, transport=Fake(), environ={}).join(10)  # no key: failed rows only
    assert _rows(ctx) and not proof_links.sample_has_open_rows(settings.db_path, job_id)
    assert proof_links.load()["open_weights_note"] in client.get("/guide/sample", follow_redirects=True).text


def test_config_is_strict(tmp_path):
    bad = tmp_path / "c.toml"
    for text in ('[sample_readback]\nenabled = true\n', '[sample_readback]\nenabled = true\nlabels = []\n',
                 '[sample_readback]\nenabled = true\nlabels = ["fan"]\nextra = 1\n'):
        bad.write_text(text)
        with pytest.raises(sample_readback.SampleReadbackConfigError):
            sample_readback.load_config(bad)
    assert sample_readback.load_config().enabled is False  # shipped default: off until the maintainer's one run


class Blocking(Fake):
    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def post(self, url, *, headers, body, timeout_s):
        self.started.set()
        assert self.release.wait(10)
        return super().post(url, headers=headers, body=body, timeout_s=timeout_s)


def test_shutdown_during_a_call_keeps_its_receipt_and_a_restart_sends_nothing(sample):
    _, settings, _, ctx = sample
    fake = Blocking()
    batch = sample_readback.run_once(settings, cfg=CFG, transport=fake, environ=ENV)
    assert fake.started.wait(10)
    threading.Timer(0.2, fake.release.set).start()
    assert ob.shutdown(10)  # drains the in-flight call before returning
    assert batch.join(10)
    assert any(r.status == "succeeded" for r in _rows(ctx))
    sent = len(fake.posts)
    assert sample_readback.run_once(settings, cfg=CFG, transport=fake, environ=ENV) is None
    assert len(fake.posts) == sent


def test_an_interrupted_call_leaves_a_marker_so_a_restart_never_resends(sample, monkeypatch):
    _, settings, _, ctx = sample
    monkeypatch.setattr(ob, "start", lambda *a, **k: None)  # the process died before any receipt
    sample_readback.run_once(settings, cfg=CFG, transport=Fake(), environ=ENV)
    assert _rows(ctx) and all(r.evidence.get("reason") == sample_readback.PENDING_REASON for r in _rows(ctx))
    monkeypatch.undo()
    fake = Fake()
    assert sample_readback.run_once(settings, cfg=CFG, transport=fake, environ=ENV) is None and not fake.posts


def test_markers_are_snapshotted_before_the_paid_call_starts(sample, monkeypatch):
    _, settings, _, ctx = sample
    seen = []
    monkeypatch.setattr(ob, "start", lambda *a, **k: seen.append("start"))
    sample_readback.run_once(settings, cfg=CFG, transport=Fake(), environ=ENV,
                             snapshot=lambda: seen.append(len(_rows(ctx))) or True)
    assert seen[0] > 0 and seen[1:] == ["start"]  # markers already written when the snapshot ran


def test_a_failed_marker_snapshot_sends_nothing(sample):
    _, settings, _, _ = sample
    fake = Fake()
    with pytest.raises(RuntimeError):
        sample_readback.run_once(settings, cfg=CFG, transport=fake, environ=ENV, snapshot=lambda: False)
    assert not fake.posts


def test_crash_after_the_marker_snapshot_restores_with_the_marker_and_never_resends(sample, monkeypatch, tmp_path):
    from app import persistence

    _, settings, _, ctx = sample
    snap_dir = tmp_path / "snaps"
    take = lambda: persistence.take_snapshot(settings.db_path, snap_dir) or True  # noqa: E731
    monkeypatch.setattr(ob, "start", lambda *a, **k: (_ for _ in ()).throw(SystemExit))  # crash mid-call
    with pytest.raises(SystemExit):
        sample_readback.run_once(settings, cfg=CFG, transport=Fake(), environ=ENV, snapshot=take)
    monkeypatch.undo()
    monkeypatch.setattr(ob, "_jpeg_data_url", lambda *a: "data:image/jpeg;base64,AAAA")
    settings.db_path.unlink()  # the live db is lost with the crash
    assert persistence.restore_latest(settings.db_path, snap_dir).source
    assert _rows(ctx) and all(r.evidence.get("reason") == sample_readback.PENDING_REASON for r in _rows(ctx))
    fake = Fake()
    assert sample_readback.run_once(settings, cfg=CFG, transport=fake, environ=ENV) is None and not fake.posts


def test_mark_interrupted_adds_a_row_per_label_still_in_flight_and_keeps_the_marker(sample, monkeypatch):
    _, settings, _, ctx = sample
    monkeypatch.setattr(ob, "start", lambda *a, **k: None)
    sample_readback.run_once(settings, cfg=CFG, transport=Fake(), environ=ENV)
    markers = len(_rows(ctx))
    assert sample_readback.mark_interrupted(settings, cfg=CFG) == markers
    reasons = [r.evidence.get("reason") for r in _rows(ctx)]
    assert reasons.count(sample_readback.PENDING_REASON) == markers
    assert reasons.count(sample_readback.INTERRUPTED_REASON) == markers
    assert sample_readback.mark_interrupted(settings, cfg=CFG) == 0  # idempotent


class _Snap:
    def __init__(self, now):
        self.now, self.requested = now, 0

    def snapshot_now(self):
        return self.now()

    def request(self):
        self.requested += 1


def _calls_text(client):
    return client.get(proof_links.load()["sample_calls"]).text


def test_a_receipt_replaces_its_marker_on_the_receipts_page_after_a_restore(sample, tmp_path):
    from app import persistence

    client, settings, _, ctx = sample
    snap_dir = tmp_path / "snaps"
    batch = sample_readback.run_once(settings, cfg=CFG, transport=Fake(), environ=ENV,
                                     snapshot=lambda: persistence.take_snapshot(settings.db_path, snap_dir) or True)
    # the hosting thread's end-of-run step: durable before it returns, no debounce
    assert sample_readback.persist_when_done(batch, _Snap(lambda: persistence.take_snapshot(settings.db_path, snap_dir) or True))
    settings.db_path.unlink()  # container replaced: only the volume snapshot survives
    assert persistence.restore_latest(settings.db_path, snap_dir).source
    real = [r for r in _rows(ctx) if r.status == "succeeded"]
    assert len(real) == 4
    page = _calls_text(client)
    assert page.count(": </span>") == 0 and "check started, receipt not yet recorded" not in page
    assert page.count(">agrees<") == 4


def test_a_marker_without_a_receipt_is_still_shown_as_not_available(sample, monkeypatch):
    client, settings, _, _ = sample
    monkeypatch.setattr(ob, "start", lambda *a, **k: None)
    sample_readback.run_once(settings, cfg=CFG, transport=Fake(), environ=ENV)
    page = _calls_text(client)
    assert page.count("check started, receipt not yet recorded") == 4 and ">agrees<" not in page


def test_a_failed_end_of_run_snapshot_falls_back_to_the_debounced_one():
    batch = type("B", (), {"join": lambda self, t=None: True})()
    for ok in (True, False):
        snap = _Snap(lambda ok=ok: ok)
        assert sample_readback.persist_when_done(batch, snap) is ok
        assert snap.requested == (0 if ok else 1)


def test_sample_readback_needs_the_single_container_deploy():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "deploy" / "modal_app.py").read_text()
    for key in ("min_containers", "max_containers"):
        m = re.search(rf"^\s*{key}=(\w+),", src, re.M)
        assert m and m.group(1) == "1", "read-back has no job lock: it needs exactly one container"


def test_calls_page_shows_the_latest_generation_for_a_label(sample):
    client, settings, job_id, ctx = sample
    guide = load_guide(job_id, data_root=settings.data_root, guide_filename=settings.guide_filename)
    d = next(x for x in guide.labels if ob.eligible(x) and x.detection is not None and x.detection.phrase in CFG.labels)
    ob.record_unavailable(ctx, d, 2, reason="request failed or timed out")
    ob.record_unavailable(ctx, d, 1, reason="answer not understood")  # older generation, written later
    page = _calls_text(client)
    assert "request failed or timed out" in page and "answer not understood" not in page
