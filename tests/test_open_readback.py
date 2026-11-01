"""Spark Open background second check: no network, fake transport only."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.contracts import RECEIPT_MODEL_SPARK_OPEN, ProvisionalRender
from app.providers import open_readback as ob
from app.providers._transport import HttpResponse
from app.routes.calls import render_calls_page
from tests.test_providers import FakeCtx

CFG = ob.load_config()
ENV = {"BASETEN_API_KEY": "k-test"}


def reply(text, ptok=470, ctok=60) -> HttpResponse:
    body = {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok}}
    return HttpResponse(200, json.dumps(body).encode(), {})


class Fake:
    def __init__(self, *script, gate=None):
        self.script, self.posts, self.gate = list(script), [], gate
        self.lock = threading.Lock()

    def post(self, url, *, headers, body, timeout_s):
        if self.gate:
            self.gate.wait(5)
        with self.lock:
            self.posts.append((url, headers, json.loads(body)))
            entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(entry, BaseException):
            raise entry
        return entry


def decision(tmp_path: Path, i=0, readback="yes", name="cooling fan"):
    img = tmp_path / f"p{i}.png"
    cv2.imwrite(str(img), np.full((720, 1280, 3), 100, np.uint8))
    prov = ProvisionalRender(detection=type("D", (), {"phrase": name, "score": 0.6})(),
                             artifact_id=f"a{i}", rendered_image_path=img, position_hint="")
    return type("L", (), {"readback": readback, "provisional": prov,
                          "checked": readback == "yes",
                          "status": "part_pixels_unchanged" if readback == "yes" else "not_checked"})()


def run(tmp_path, transport, n=1, cfg=CFG, environ=ENV):
    ctx = FakeCtx()
    t = ob.start(ctx, [decision(tmp_path, i=i) for i in range(n)], cfg=cfg,
                 transport=transport, environ=environ)
    if t:
        t.join(10)
    return ctx, [c for c in ctx.store.calls_for(str(ctx.job_id)) if c.model == RECEIPT_MODEL_SPARK_OPEN]


def test_agree_and_request_shape(tmp_path):
    fake = Fake(reply("yes"))
    _, calls = run(tmp_path, fake)
    assert calls[0].status == "succeeded" and calls[0].evidence["decision"] == "yes"
    assert calls[0].estimated_cost_usd > 0 and calls[0].endpoint == "/chat-completions"
    url, headers, body = fake.posts[0]
    assert url == CFG.endpoint and body["model"] == CFG.model_id and body["reasoning_effort"] == "low"
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_disagree_is_recorded_not_acted_on(tmp_path):
    ctx, calls = run(tmp_path, Fake(reply("no")))
    assert calls[0].status == "succeeded" and calls[0].evidence["decision"] == "no"
    assert ctx.finishes == [] and ctx.artifacts.deleted == []


@pytest.mark.parametrize("answer", [TimeoutError("slow"), reply("Maybe, it looks like one"),
                                    HttpResponse(200, b"not json", {}), HttpResponse(500, b"{}", {})])
def test_failures_are_unavailable(tmp_path, answer):
    ctx, calls = run(tmp_path, Fake(answer))
    assert calls[0].status == "failed" and "decision" not in calls[0].evidence
    assert ctx.finishes == []


def test_missing_key_is_unavailable_without_a_call(tmp_path):
    fake = Fake(reply("yes"))
    _, calls = run(tmp_path, fake, environ={})
    assert fake.posts == [] and calls[0].status == "failed"
    assert calls[0].evidence["reason"] == "provider key not configured"


def test_failed_request_says_it_failed_not_that_the_key_is_missing(tmp_path):
    _, calls = run(tmp_path, Fake(TimeoutError("slow")))
    assert calls[0].evidence["reason"] == "request failed or timed out"


def test_disabled_does_nothing(tmp_path):
    fake = Fake(reply("yes"))
    ctx = FakeCtx()
    assert ob.start(ctx, [decision(tmp_path)], cfg=replace(CFG, enabled=False),
                    transport=fake, environ=ENV) is None
    assert fake.posts == [] and ctx.store.calls_for(str(ctx.job_id)) == ()


def test_only_kept_labels_are_checked(tmp_path):
    fake = Fake(reply("yes"))
    ob.start(FakeCtx(), [decision(tmp_path, readback="no"), decision(tmp_path, i=1, readback="not_run")],
             cfg=CFG, transport=fake, environ=ENV)
    assert fake.posts == []


def test_call_cap(tmp_path):
    fake = Fake(reply("yes"))
    _, calls = run(tmp_path, fake, n=3, cfg=replace(CFG, max_calls_per_job=2))
    assert len(fake.posts) == 2
    assert sorted(c.status for c in calls) == ["failed", "succeeded", "succeeded"]


def test_cost_cap(tmp_path):
    # Calls run concurrently, so each holds its whole bounded reservation while in flight.
    gate = threading.Event()
    fake = Fake(reply("yes"), gate=gate)
    cfg = replace(CFG, max_cost_usd_per_job=2 * ob.reservation_usd(CFG), concurrency=3)
    ctx = FakeCtx()
    t = ob.start(ctx, [decision(tmp_path, i=i) for i in range(3)], cfg=cfg, transport=fake, environ=ENV)
    deadline = time.monotonic() + 10
    while not ctx.store.calls_for(str(ctx.job_id)) and time.monotonic() < deadline:
        time.sleep(0.01)  # the third check is skipped while the first two are held
    gate.set()
    assert t.join(10)
    calls = [c for c in ctx.store.calls_for(str(ctx.job_id)) if c.model == RECEIPT_MODEL_SPARK_OPEN]
    assert len(fake.posts) == 2 and sorted(c.status for c in calls) == ["failed", "succeeded", "succeeded"]
    assert sum(c.estimated_cost_usd for c in calls) <= cfg.max_cost_usd_per_job


def test_audit_overspend_case_cannot_be_configured(tmp_path):
    # a $0.001 cap/reservation let a 470-in/300-out call bill $0.0018625.
    worst = ob._cost(CFG, 470, 300)
    assert worst == Decimal("0.0018625") and worst <= ob.reservation_usd(CFG)
    path = tmp_path / "c.toml"
    path.write_text(Path(ob.CONFIG).read_text().replace(
        "max_cost_usd_per_job = 0.02", "max_cost_usd_per_job = 0.001"))
    with pytest.raises(ob.OpenReadbackConfigError, match="cannot cover one"):
        ob.load_config(path)
    fake = Fake(reply("yes"))
    assert ob.start(FakeCtx(), [decision(tmp_path)], cfg=replace(CFG, max_cost_usd_per_job=Decimal("0.001")),
                    transport=fake, environ=ENV) is None
    assert fake.posts == []


def test_reservation_covers_the_longest_allowed_request(tmp_path):
    phrase = "\U0001f527" * CFG.label_max_chars  # 4 UTF-8 bytes a character
    text_bytes = len(ob._question(phrase).encode("utf-8"))
    assert ob._cost(CFG, CFG.prompt_overhead_tokens + text_bytes + CFG.image_token_ceiling,
                    CFG.max_tokens) <= ob.reservation_usd(CFG)


def test_label_over_the_bound_is_not_sent(tmp_path):
    fake = Fake(reply("yes"))
    ctx = FakeCtx()
    t = ob.start(ctx, [decision(tmp_path, name="x" * (CFG.label_max_chars + 1))], cfg=CFG,
                 transport=fake, environ=ENV)
    assert t.join(10) and fake.posts == []
    (call,) = ctx.store.calls_for(str(ctx.job_id))
    assert call.estimated_cost_usd == 0 and call.evidence["cost"] == "skipped"


def test_bill_above_the_reservation_stops_the_job(tmp_path):
    over = ob.reservation_usd(CFG) * 1_000_000 / CFG.output_usd_per_mtok
    fake = Fake(reply("yes", ptok=0, ctok=int(over) + 1))
    _, calls = run(tmp_path, fake, n=3, cfg=replace(CFG, concurrency=1))
    assert len(fake.posts) == 1
    assert sorted(c.evidence.get("cost", "measured") for c in calls) == ["measured", "skipped", "skipped"]


def test_the_users_path_never_waits(tmp_path):
    gate = threading.Event()
    ctx = FakeCtx()
    t = ob.start(ctx, [decision(tmp_path)], cfg=CFG, transport=Fake(reply("yes"), gate=gate), environ=ENV)
    assert t is not None and t.is_alive()  # start() returned while the call is blocked
    assert ctx.store.calls_for(str(ctx.job_id)) == ()
    gate.set()
    t.join(10)
    assert len(ctx.store.calls_for(str(ctx.job_id))) == 1


def test_config_is_strict(tmp_path):
    text = Path(ob.CONFIG).read_text()
    for broken in (text.replace("concurrency", "concurency"), text.replace("timeout_s = 60.0", 'timeout_s = "x"')):
        p = tmp_path / "c.toml"
        p.write_text(broken)
        with pytest.raises(ob.OpenReadbackConfigError):
            ob.load_config(p)


def test_calls_page_lines(tmp_path):
    ctx, _ = run(tmp_path, Fake(reply("yes"), reply("no"), TimeoutError("x")), n=3,
                 cfg=replace(CFG, concurrency=1))
    page = render_calls_page(str(ctx.job_id), ctx)
    assert page.count("Second check (Meta open weights, Spark Open)") == 3
    for word in ("agrees", "disagrees", "not available"):
        assert word in page
    assert "spark-open" in page  # receipt row with latency and cost


def test_calls_page_without_the_check_has_no_section():
    ctx = FakeCtx()
    assert "Second check" not in render_calls_page(str(ctx.job_id), ctx)


def test_key_never_in_repr_or_receipts(tmp_path):
    _, calls = run(tmp_path, Fake(reply("yes")), environ={"BASETEN_API_KEY": "sk-secret-123"})
    assert "sk-secret-123" not in repr(calls) and "sk-secret-123" not in repr(CFG)


def test_checked_high_confidence_label_is_included(tmp_path):
    label = decision(tmp_path, readback="not_run")
    label.checked = True
    label.status = "part_pixels_unchanged"
    fake = Fake(reply("yes"))
    t = ob.start(FakeCtx(), [label], cfg=CFG, transport=fake, environ=ENV)
    assert t is not None
    t.join(10)
    assert len(fake.posts) == 1


def test_rate_limit_does_not_retry(tmp_path):
    fake = Fake(HttpResponse(429, b"{}", {}), reply("yes"))
    _, calls = run(tmp_path, fake, cfg=replace(CFG, max_calls_per_job=1))
    assert len(fake.posts) == 1 and calls[0].status == "failed"


@pytest.mark.parametrize("answer", [TimeoutError("slow"), reply("yes", ptok=None)])
def test_attempt_without_usage_records_the_reservation_as_estimate(tmp_path, answer):
    _, calls = run(tmp_path, Fake(answer))
    assert calls[0].estimated_cost_usd == ob.reservation_usd(CFG)
    assert calls[0].evidence["cost"] == "estimate"


def test_skips_record_zero_marked_as_skips(tmp_path):
    _, no_key = run(tmp_path, Fake(reply("yes")), environ={})
    _, capped = run(tmp_path, Fake(reply("yes")), n=2, cfg=replace(CFG, max_calls_per_job=1, concurrency=1))
    _, measured = run(tmp_path, Fake(reply("yes")))
    skipped = no_key + [c for c in capped if c.status == "failed"]
    assert len(skipped) == 2
    assert all(c.estimated_cost_usd == 0 and c.evidence["cost"] == "skipped" for c in skipped)
    assert "cost" not in measured[0].evidence


def test_temperature_zero_allowed_negative_rejected(tmp_path):
    assert CFG.temperature == 0
    path = tmp_path / "c.toml"
    path.write_text(Path(ob.CONFIG).read_text().replace("temperature = 0 ", "temperature = -0.1 "))
    with pytest.raises(ob.OpenReadbackConfigError, match="temperature"):
        ob.load_config(path)


def test_cancel_wait_shorter_than_timeout_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(Path(ob.CONFIG).read_text().replace("cancel_wait_s = 65.0", "cancel_wait_s = 1.0"))
    with pytest.raises(ob.OpenReadbackConfigError, match="cancel_wait_s"):
        ob.load_config(path)


@pytest.mark.parametrize("field", ["timeout_s", "max_cost_usd_per_job", "input_usd_per_mtok", "output_usd_per_mtok",
                                   "temperature", "cancel_wait_s"])
@pytest.mark.parametrize("value", ["inf", "nan"])
def test_nonfinite_numbers_rejected(tmp_path, field, value):
    text = Path(ob.CONFIG).read_text()
    lines = text.splitlines()
    lines = [f"{field} = {value}" if line.startswith(field + " =") else line for line in lines]
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines))
    with pytest.raises(ob.OpenReadbackConfigError):
        ob.load_config(path)


def test_other_credential_variable_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(Path(ob.CONFIG).read_text() + '\nkey_env = "OTHER_API_KEY"\n')
    with pytest.raises(ob.OpenReadbackConfigError):
        ob.load_config(path)


def test_invalid_config_surfaces_at_app_startup(monkeypatch):
    from app import main

    def invalid():
        raise ob.OpenReadbackConfigError("open_readback.timeout_s: bad value")

    monkeypatch.setattr(ob, "load_config", invalid)
    with pytest.raises(ob.OpenReadbackConfigError, match="timeout_s"):
        main.create_app()


def test_blocked_worker_does_not_block_process_shutdown(tmp_path):
    import subprocess
    import sys

    script = '''
import sys, threading, time
from pathlib import Path
from app.providers import open_readback as ob
from tests.test_open_readback import decision, reply
from tests.test_providers import FakeCtx
entered = threading.Event()
class Blocked:
    def post(self, *args, **kwargs):
        entered.set()
        threading.Event().wait(30)
        return reply("yes")
t = ob.start(FakeCtx(), [decision(Path(sys.argv[1]))], cfg=ob.load_config(), transport=Blocked(),
             environ={"BASETEN_API_KEY": "test"})
assert t is not None and entered.wait(10)
began = time.monotonic()
assert ob.shutdown() is False  # the blocked worker is still in its call; shutdown does not wait
assert time.monotonic() - began < 1
assert ob.start(FakeCtx(), [decision(Path(sys.argv[1]), i=1)], cfg=ob.load_config(), transport=Blocked(),
                environ={"BASETEN_API_KEY": "test"}) is None
'''
    began = __import__("time").monotonic()
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            env={}, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
    assert __import__("time").monotonic() - began < 10  # the 30 s blocked call never held up exit


class Blocked:
    """A transport held inside its POST until released."""

    def __init__(self):
        self.entered, self.release, self.posts = threading.Event(), threading.Event(), []

    def post(self, url, *, headers, body, timeout_s):
        self.posts.append(json.loads(body))
        self.entered.set()
        assert self.release.wait(10)
        return reply("yes")


def spark_open(ctx):
    return [c for c in ctx.store.calls_for(str(ctx.job_id)) if c.model == RECEIPT_MODEL_SPARK_OPEN]


def test_shutdown_drops_queued_checks(tmp_path):
    fake, ctx = Blocked(), FakeCtx()
    t = ob.start(ctx, [decision(tmp_path, i=i) for i in range(3)], cfg=replace(CFG, concurrency=1),
                 transport=fake, environ=ENV)
    assert fake.entered.wait(10)
    assert ob.shutdown(0) is False  # one worker still in its call; shutdown returned at once
    fake.release.set()
    assert t.join(10) and ob.shutdown(5) is True
    assert len(fake.posts) == 1 and len(spark_open(ctx)) == 1


def test_cancel_returns_at_once_while_the_call_is_in_flight(tmp_path, monkeypatch):
    reads = []
    real = ob._jpeg_data_url
    monkeypatch.setattr(ob, "_jpeg_data_url", lambda path, *a: reads.append(path) or real(path, *a))
    fake, ctx = Blocked(), FakeCtx()
    t = ob.start(ctx, [decision(tmp_path, i=i) for i in range(3)], cfg=replace(CFG, concurrency=1),
                 transport=fake, environ=ENV)
    assert fake.entered.wait(10)
    began = time.monotonic()
    assert ob.cancel(ctx.job_id) is False  # the provider is still held; cancel did not wait
    assert time.monotonic() - began < 1 and not fake.release.is_set()
    assert ob.wait_quiet(ctx.job_id, 0) is False
    fake.release.set()
    assert ob.wait_quiet(ctx.job_id) is True and t.join(10)
    # Only the in-flight call was sent or read; its receipt is kept, the queued two never ran.
    assert len(fake.posts) == 1 and len(reads) == 1 and len(spark_open(ctx)) == 1
    assert ob.start(ctx, [decision(tmp_path, i=9)], cfg=CFG, transport=fake, environ=ENV) is None


def test_queued_check_released_after_cancel_never_sends(tmp_path):
    """Mark/drop, then the durable commit: a worker freed in between finds nothing to send."""
    fake, ctx, other = Blocked(), FakeCtx(), FakeCtx()
    # One worker, held by another job's call; this job's check waits in the queue.
    t_other = ob.start(other, [decision(tmp_path, i=0)], cfg=replace(CFG, concurrency=1),
                       transport=fake, environ=ENV)
    assert fake.entered.wait(10)
    t = ob.start(ctx, [decision(tmp_path, i=1)], cfg=replace(CFG, concurrency=1),
                 transport=fake, environ=ENV)
    assert ob.cancel(ctx.job_id) is True  # step 1: queued check dropped, nothing active
    fake.release.set()  # the worker is released between the two steps
    assert t_other.join(10) and t.join(10)
    sent_at_commit = len(fake.posts)  # step 2: the durable cancellation point
    assert ob.wait_quiet(ctx.job_id, 1) is True
    assert len(fake.posts) == sent_at_commit == 1 and spark_open(ctx) == []


def test_finished_batches_leave_no_wait_entries(tmp_path):
    fake = Fake(reply("yes"))
    for _ in range(100):
        ctx = FakeCtx()
        t = ob.start(ctx, [decision(tmp_path)], cfg=CFG, transport=fake, environ=ENV)
        assert t.join(10)  # a batch ends only after its worker's bookkeeping
    with ob._RUNNER._cond:
        assert ob._RUNNER._wait_s == {} and not ob._RUNNER._active


def test_deletion_between_encoding_and_post_sends_nothing(tmp_path, monkeypatch):
    encoded, go = threading.Event(), threading.Event()
    real = ob._jpeg_data_url

    def slow_encode(path, *a):
        url = real(path, *a)
        encoded.set()
        assert go.wait(10)
        return url

    monkeypatch.setattr(ob, "_jpeg_data_url", slow_encode)
    fake, ctx = Blocked(), FakeCtx()
    fake.release.set()
    t = ob.start(ctx, [decision(tmp_path)], cfg=CFG, transport=fake, environ=ENV)
    assert encoded.wait(10)
    assert ob.cancel(ctx.job_id) is False  # the task holding the image is still running
    go.set()
    assert ob.wait_quiet(ctx.job_id) is True and t.join(10)
    assert fake.posts == [] and spark_open(ctx) == []


def test_deletion_before_start_refuses_the_job(tmp_path):
    fake, ctx = Blocked(), FakeCtx()
    assert ob.cancel(ctx.job_id) is True
    assert ob.start(ctx, [decision(tmp_path)], cfg=CFG, transport=fake, environ=ENV) is None
    assert fake.posts == []


def test_deletion_gives_up_after_cancel_wait(tmp_path):
    fake, ctx = Blocked(), FakeCtx()
    t = ob.start(ctx, [decision(tmp_path)], cfg=replace(CFG, timeout_s=0.1, cancel_wait_s=0.2),
                 transport=fake, environ=ENV)
    assert fake.entered.wait(10)
    assert ob.cancel(ctx.job_id) is False
    began = time.monotonic()
    assert ob.wait_quiet(ctx.job_id) is False  # gave up after cancel_wait_s
    assert 0.15 < time.monotonic() - began < 2
    fake.release.set()
    assert t.join(10)


def test_cancel_and_delete_stops_queued_checks_before_cancellation_commits(tmp_path, monkeypatch):
    from app import deletion

    fake, other = Blocked(), FakeCtx()
    # The one worker is held by another job; job-1's check waits in the queue.
    t_other = ob.start(other, [decision(tmp_path, i=0)], cfg=replace(CFG, concurrency=1),
                       transport=fake, environ=ENV)
    assert fake.entered.wait(10)
    job = FakeCtx()
    t = ob.start(job, [decision(tmp_path, i=1)], cfg=replace(CFG, concurrency=1),
                 transport=fake, environ=ENV)
    order, at_commit = [], []
    real_cancel = ob.cancel
    monkeypatch.setattr(ob, "cancel", lambda job_id: order.append("cancel") or real_cancel(job_id))

    class Artifacts:
        def mark_deleted(self, artifact_id):
            order.append(("delete", artifact_id))

    class Ctx:
        artifacts = Artifacts()

    class Record:
        preview_artifact_id = "preview"

    class Uploads:
        def request_cancellation(self, job_id, now, **kw):
            fake.release.set()  # the worker is freed between mark/drop and the commit
            assert t_other.join(10) and t.join(10)
            at_commit.append(len(fake.posts))
            order.append("commit")

        def load_job(self, job_id):
            return Ctx(), Record()

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(deletion.guide_clip, "delete_job_clips", stop)
    monkeypatch.setattr(deletion, "delete_guide_files", stop)
    with pytest.raises(Exception):
        deletion.cancel_and_delete(job.job_id, uploads=Uploads(), lifecycle=None, client=None)
    assert order[:3] == ["cancel", "commit", ("delete", "preview")]
    assert ob.wait_quiet(job.job_id, 1) is True
    assert at_commit == [1] and len(fake.posts) == 1 and spark_open(job) == []  # zero sends after the commit


def test_failed_mark_drop_aborts_cancellation_and_no_queued_send_follows(tmp_path, monkeypatch):
    from app import deletion

    fake, other = Blocked(), FakeCtx()
    # The one worker is held by another job; job-1's check waits in the queue.
    t_other = ob.start(other, [decision(tmp_path, i=0)], cfg=replace(CFG, concurrency=1),
                       transport=fake, environ=ENV)
    assert fake.entered.wait(10)
    job = FakeCtx()
    t = ob.start(job, [decision(tmp_path, i=1)], cfg=replace(CFG, concurrency=1),
                 transport=fake, environ=ENV)

    def broken_drop(keep):
        raise RuntimeError("injected drop failure")

    monkeypatch.setattr(ob._RUNNER, "_drop", broken_drop)
    commits = []

    class Uploads:
        def request_cancellation(self, job_id, now, **kw):
            commits.append(job_id)

    with pytest.raises(deletion.CancellationRetryError):
        deletion.cancel_and_delete(job.job_id, uploads=Uploads(), lifecycle=None, client=None)
    assert commits == []  # nothing committed: the user is asked to retry
    monkeypatch.undo()
    fake.release.set()  # the queued check now reaches a free worker
    assert t_other.join(10) and t.join(10)
    assert len(fake.posts) == 1 and spark_open(job) == []  # the queued check sent nothing


def test_recheck_of_one_renamed_label_records_one_receipt_keyed_by_label_and_text(tmp_path):
    """A renamed label (free text, not in the evidence vocabulary) is rechecked alone; its receipt
    names the label image and a digest of the new text, so the guide can tell it from the old one."""
    d = decision(tmp_path, i=3, name="blower")
    ctx = FakeCtx()
    batch = ob.start(ctx, [d], cfg=CFG, transport=Fake(reply("no")), environ=ENV)
    batch.join(10)
    (c,) = [c for c in ctx.store.calls_for(str(ctx.job_id)) if c.model == RECEIPT_MODEL_SPARK_OPEN]
    assert c.evidence["label"] == "p3.png" and c.evidence["phrase_id"] == ob.phrase_id("blower")
    assert "phrase" not in c.evidence  # free text never enters receipts
    assert ob.outcomes([c]) == {"p3.png": ob.Outcome(0, ob.phrase_id("blower"), "disagree")}
    assert ob.phrase_id("blower") != ob.phrase_id("fan")


# --- one durable spend budget per job --------------------------

def _batch(ctx, tmp_path, fake, start_i, n, cfg, **kw):
    t = ob.start(ctx, [decision(tmp_path, i=start_i + i) for i in range(n)], cfg=cfg,
                 transport=fake, environ=ENV, **kw)
    assert t is not None and t.join(10)


def test_call_cap_holds_across_batches_of_one_job(tmp_path):
    fake = Fake(reply("yes"))
    ctx = FakeCtx()
    cfg = replace(CFG, max_calls_per_job=3)
    for b in range(4):  # the initial batch plus three one-label rechecks
        _batch(ctx, tmp_path, fake, b * 2, 2 if b == 0 else 1, cfg)
    assert len(fake.posts) == 3
    other = FakeCtx()  # a different job has its own budget
    _batch(other, tmp_path, fake, 9, 1, cfg)
    assert len(fake.posts) == 4


def test_cost_cap_holds_across_batches_of_one_job(tmp_path):
    unbilled = HttpResponse(200, json.dumps({"choices": [{"message": {"content": "yes"}}]}).encode(), {})
    fake = Fake(unbilled)  # no usage back: each call is charged its whole reservation
    ctx = FakeCtx()
    cfg = replace(CFG, max_calls_per_job=50, max_cost_usd_per_job=2 * ob.reservation_usd(CFG))
    for b in range(5):
        _batch(ctx, tmp_path, fake, b, 1, cfg)
    spent = sum(c.estimated_cost_usd for c in ctx.store.calls_for(str(ctx.job_id))
                if c.model == RECEIPT_MODEL_SPARK_OPEN)
    assert spent <= cfg.max_cost_usd_per_job and len(fake.posts) == 2


def test_budget_is_persisted_in_the_corrections_sidecar(tmp_path):
    from app.corrections import CorrectionStore

    (tmp_path / "guide.json").write_text("{}")
    fake = Fake(reply("yes"))
    ctx = FakeCtx()
    cfg = replace(CFG, max_calls_per_job=2)

    def store():  # a fresh store each time: another worker process or a restart
        return CorrectionStore(lambda j: tmp_path, "c.json", max_ops=9, guide_filename="guide.json")

    _batch(ctx, tmp_path, fake, 0, 1, cfg, budget=store())
    calls, cost, breached = store().spend(str(ctx.job_id))
    assert calls == 1 and Decimal("0") < cost <= ob.reservation_usd(CFG) and not breached
    _batch(ctx, tmp_path, fake, 1, 1, cfg, budget=store())
    _batch(ctx, tmp_path, fake, 2, 1, cfg, budget=store())
    assert len(fake.posts) == 2 and store().spend(str(ctx.job_id))[0] == 2
    assert store().read(str(ctx.job_id)) == []  # spend never shows up as an edit


def test_overbilled_call_stops_later_batches_too(tmp_path):
    fake = Fake(reply("yes", ptok=10**7))
    ctx = FakeCtx()
    _batch(ctx, tmp_path, fake, 0, 1, CFG)
    _batch(ctx, tmp_path, fake, 1, 1, CFG)
    assert len(fake.posts) == 1


def test_receipts_carry_the_label_generation(tmp_path):
    fake = Fake(reply("no"))
    ctx = FakeCtx()
    d = decision(tmp_path)
    t = ob.start(ctx, [d], cfg=CFG, transport=fake, environ=ENV, gens={"p0.png": 4})
    assert t.join(10)
    (c,) = [c for c in ctx.store.calls_for(str(ctx.job_id)) if c.model == RECEIPT_MODEL_SPARK_OPEN]
    assert c.evidence["gen"] == 4 and ob.outcomes([c])["p0.png"].gen == 4
