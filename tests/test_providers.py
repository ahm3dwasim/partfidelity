"""Live-provider tests: fake transport per client, wrappers, records, factory.

Every test runs offline against a scripted transport and a fake job
context whose receipts land in the evidence module's own in-memory store, so the record
assertions prove the evidence vocabulary instead of assuming it. The one live
run lives in tests/test_providers_live.py.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import evidence as _evidence
from app import sam as app_sam
from app.contracts import JobId
from app.evidence import InMemoryEvidenceStore
from app.ledger import ReserveRefused
from app.providers import Providers, select_providers
from app.providers._transport import (
    HttpResponse,
    HttpTransport,
    ProviderConnectError,
    ProviderError,
    ProviderHttpError,
    ProviderMalformed,
    ProviderRateLimited,
    ProviderTimeout,
    UrllibTransport,
    output_text,
    post_with_retry,
    response_json,
    still_data_url,
    usage_counts,
)
from app.providers.meta_readback import MetaReadbackClient, readback_worst_case
from app.providers.meta_sam import MetaSamClient
from app.providers.meta_spark import SPARK_PROMPT, MetaSparkClient, spark_worst_case
from app.readback import ReadbackError, read_back_label
from app.sam import SamPacer
from app.settings import ProviderSettings, Settings, load_settings
from app.spark import SparkClientError

IN_PRICE = Decimal("0.00000125")
OUT_PRICE = Decimal("0.00000425")
SPARK_WORST = spark_worst_case(IN_PRICE, OUT_PRICE)
READBACK_WORST = readback_worst_case(IN_PRICE, OUT_PRICE)
assert SPARK_WORST == Decimal("0.0825")
assert READBACK_WORST == Decimal("0.0185")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedTransport:
    """Scripted HttpTransport; entries are HttpResponse or Exception."""

    def __init__(self, script) -> None:
        self._script = list(script)
        self.posts: list[dict] = []

    def post(self, url, *, headers, body, timeout_s):
        self.posts.append(
            {"url": url, "headers": dict(headers), "body": body, "timeout_s": timeout_s}
        )
        if not self._script:
            raise AssertionError("transport script exhausted")
        entry = self._script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry


def ok(payload, *, headers=None) -> HttpResponse:
    return HttpResponse(
        status=200, body=json.dumps(payload).encode(), headers=headers or {}
    )


def err(status, *, headers=None, body=b'{"error":{"code":"x"}}') -> HttpResponse:
    return HttpResponse(status=status, body=body, headers=headers or {})


def spark_response(text_obj, in_tok=16000, out_tok=2000) -> dict:
    return {
        "output": [
            {"content": [{"type": "output_text", "text": json.dumps(text_obj)}]}
        ],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def sam_response(text) -> dict:
    span = {"type": "output_text", "text": text}
    item = {"content": [span]}
    return {"output": [item]}


class FakeHold:
    def __init__(self) -> None:
        self.settlements: list[tuple] = []

    def settle(self, actual, outcome):
        self.settlements.append((actual, outcome))


class FakeArtifacts:
    def __init__(self) -> None:
        self.deleted: list = []

    def mark_deleted(self, artifact_id):
        self.deleted.append(artifact_id)


class FakeCtx:
    """Recording context whose receipts land in the evidence module's own in-memory store."""

    _seq = 0

    def __init__(self, *, refuse_reserve=False) -> None:
        FakeCtx._seq += 1
        self._job_id = JobId(f"live-test-{FakeCtx._seq}")
        self.store = InMemoryEvidenceStore()
        self.evidence_store = self.store
        self.reserves: list[tuple] = []
        self.holds: list[FakeHold] = []
        self.finishes: list = []
        self.artifacts = FakeArtifacts()
        self._refuse = refuse_reserve

    @property
    def job_id(self):
        return self._job_id

    def reserve(self, kind, worst_case_usd):
        self.reserves.append((kind, worst_case_usd))
        if self._refuse:
            raise ReserveRefused("test refusal")
        hold = FakeHold()
        self.holds.append(hold)
        return hold

    def record_call_with_event(self, key, call, event):
        self.store.record_call_with_event(key, call, event)

    def finish(self, outcome, *, keep_artifact_ids=(), failed_output_names=None):
        self.finishes.append(outcome)


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0
        self.sleeps: list[float] = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def make_spark_client(ctx, transport, clock) -> MetaSparkClient:
    return MetaSparkClient(
        api_key="k-test",
        base_url="https://api.test/v1",
        model_id="muse-spark-1.3",
        input_price=IN_PRICE,
        output_price=OUT_PRICE,
        timeout_s=30,
        ctx=ctx,
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.now,
    )


def make_sam_client(ctx, transport, clock) -> MetaSamClient:
    return MetaSamClient(
        api_key="k-test",
        base_url="https://api.test/v1",
        model_id="sam-3.1",
        price_per_call=Decimal("0.0025"),
        timeout_s=30,
        phrases={"fan": "cooling fan", "battery": "rectangular battery with label"},
        ctx=ctx,
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.now,
    )


def make_readback_client(ctx, transport, clock) -> MetaReadbackClient:
    return MetaReadbackClient(
        api_key="k-test",
        base_url="https://api.test/v1",
        model_id="muse-spark-1.3",
        input_price=IN_PRICE,
        output_price=OUT_PRICE,
        timeout_s=30,
        ctx=ctx,
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.now,
    )


def make_png(path: Path, width=320, height=240) -> Path:
    img = np.full((height, width, 3), 128, np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


def stored_calls(ctx: FakeCtx):
    return ctx.store.calls_for(str(ctx.job_id))


@pytest.fixture
def fast_sam():
    """Unpaced SAM wrapper calls; global pacer and counts restored after."""
    clock = FakeClock()
    app_sam.install_pacer(
        SamPacer(monotonic=clock.now, sleep=lambda seconds: None)
    )
    app_sam.reset_sam_job_counts()
    yield clock
    app_sam.reset_pacer()
    app_sam.reset_sam_job_counts()
    app_sam.reset_call_sink()


@pytest.fixture(autouse=True)
def _pristine_stage_config():
    """Factory installs never leak between tests."""
    yield
    app_sam.reset_pacer()
    app_sam.reset_sam_job_counts()


class FailingStore(InMemoryEvidenceStore):
    """Evidence store whose receipt writes fail like a wedged database."""

    def __init__(self, *, fail_times: int | None = None) -> None:
        super().__init__()
        self.fail_times = fail_times  # None means every write fails
        self.write_attempts = 0

    def record_call_with_event(self, key, call, event):
        import sqlite3

        self.write_attempts += 1
        if self.fail_times is None or self.write_attempts <= self.fail_times:
            raise sqlite3.OperationalError("database is locked")
        super().record_call_with_event(key, call, event)


class AdvancingTransport(ScriptedTransport):
    """Scripted transport whose slow responses advance the fake clock.

    Each entry is ``(response_or_error, elapsed_s)``; the clock advances
    by the attempt's own ``timeout_s`` when the scripted elapse would
    overrun it, exactly like a real per-attempt timeout firing.
    """

    def __init__(self, script, clock: FakeClock) -> None:
        super().__init__([])
        self._script = list(script)
        self._clock = clock

    def post(self, url, *, headers, body, timeout_s):
        self.posts.append(
            {"url": url, "headers": dict(headers), "body": body, "timeout_s": timeout_s}
        )
        if not self._script:
            raise AssertionError("transport script exhausted")
        entry, elapsed = self._script.pop(0)
        self._clock.t += min(elapsed, timeout_s)
        if isinstance(entry, Exception):
            raise entry
        return entry


# ---------------------------------------------------------------------------
# Transport: retries, mapping, real-socket round trip
# ---------------------------------------------------------------------------


def test_retry_success_first_try_sends_once_and_sleeps_never():
    clock = FakeClock()
    transport = ScriptedTransport([ok({"ok": True})])
    response = post_with_retry(
        transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
        sleep=clock.sleep,
    )
    assert response.status == 200
    assert len(transport.posts) == 1
    assert clock.sleeps == []


def test_retry_429_honors_retry_after_then_succeeds():
    clock = FakeClock()
    transport = ScriptedTransport(
        [err(429, headers={"retry-after": "30"}), ok({"ok": True})]
    )
    response = post_with_retry(
        transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
        sleep=clock.sleep,
    )
    assert response.status == 200
    assert len(transport.posts) == 2
    assert clock.sleeps == [30.0]


def test_retry_429_caps_a_huge_retry_after():
    clock = FakeClock()
    transport = ScriptedTransport(
        [err(429, headers={"retry-after": "600"}), ok({"ok": True})]
    )
    post_with_retry(
        transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
        sleep=clock.sleep,
    )
    assert clock.sleeps == [65.0]


def test_retry_429_with_garbage_header_falls_back_to_backoff():
    clock = FakeClock()
    transport = ScriptedTransport(
        [
            err(429, headers={"retry-after": "soon-ish"}),
            err(429, headers={"retry-after": "-5"}),
            ok({"ok": True}),
        ]
    )
    post_with_retry(
        transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
        sleep=clock.sleep,
    )
    assert clock.sleeps[0] == 2.0
    assert clock.sleeps[1] == 0.0


def test_retry_429_without_header_backs_off_exponentially():
    clock = FakeClock()
    transport = ScriptedTransport([err(429)] * 3 + [ok({"ok": True})])
    post_with_retry(
        transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
        sleep=clock.sleep,
    )
    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_retry_429_exhausted_raises_rate_limited():
    clock = FakeClock()
    transport = ScriptedTransport([err(429)] * 4)
    with pytest.raises(ProviderRateLimited):
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
            sleep=clock.sleep,
        )
    assert len(transport.posts) == 4
    assert clock.sleeps == [2.0, 4.0, 8.0]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_retry_5xx_raises_indeterminate_at_once(status):
    clock = FakeClock()
    transport = ScriptedTransport([err(status), ok({"ok": True})])
    with pytest.raises(ProviderHttpError) as caught:
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
            sleep=clock.sleep,
        )
    assert caught.value.status == status
    assert caught.value.indeterminate is True
    assert caught.value.retryable is False
    assert len(transport.posts) == 1  # never re-POST past the point of charge
    assert clock.sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_retry_non_retryable_status_fails_fast(status):
    clock = FakeClock()
    transport = ScriptedTransport([err(status)])
    with pytest.raises(ProviderHttpError) as caught:
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
            sleep=clock.sleep,
        )
    assert caught.value.status == status
    assert len(transport.posts) == 1
    assert clock.sleeps == []


def test_retry_timeout_raises_indeterminate_at_once():
    clock = FakeClock()
    transport = ScriptedTransport(
        [ProviderTimeout("slow"), ok({"ok": True})]
    )
    with pytest.raises(ProviderTimeout) as caught:
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
            sleep=clock.sleep,
        )
    assert caught.value.indeterminate is True
    assert caught.value.retryable is False
    assert len(transport.posts) == 1  # the timed-out POST may have been billed
    assert clock.sleeps == []


def test_retry_post_send_failure_raises_indeterminate_at_once():
    clock = FakeClock()
    transport = ScriptedTransport(
        [ProviderError("reset by peer", indeterminate=True), ok({"ok": True})]
    )
    with pytest.raises(ProviderError) as caught:
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
            sleep=clock.sleep,
        )
    assert caught.value.indeterminate is True
    assert len(transport.posts) == 1
    assert clock.sleeps == []


def test_retry_pre_send_connect_failure_retries_then_succeeds_and_exhausts():
    clock = FakeClock()
    transport = ScriptedTransport(
        [ProviderConnectError("refused"), ok({"ok": True})]
    )
    response = post_with_retry(
        transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
        sleep=clock.sleep,
    )
    assert response.status == 200
    assert len(transport.posts) == 2
    assert clock.sleeps == [2.0]
    clock2 = FakeClock()
    transport2 = ScriptedTransport([ProviderConnectError("refused")] * 4)
    with pytest.raises(ProviderConnectError) as caught:
        post_with_retry(
            transport2, "https://x.test/r", headers={}, body=b"{}",
            timeout_s=5, sleep=clock2.sleep,
        )
    assert caught.value.indeterminate is False  # never sent: determinate $0
    assert len(transport2.posts) == 4
    assert clock2.sleeps == [2.0, 4.0, 8.0]


def test_retry_non_retryable_error_raises_at_once():
    clock = FakeClock()
    transport = ScriptedTransport(
        [ProviderError("redirect refused", retryable=False)]
    )
    with pytest.raises(ProviderError):
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
            sleep=clock.sleep,
        )
    assert len(transport.posts) == 1
    assert clock.sleeps == []


def test_deadline_stops_attempts_and_shrinks_timeouts():
    # Slow 429s advance the fake clock by real per-attempt elapses; the
    # 60 s end-to-end deadline stops the call at 3 attempts with the
    # last attempt running under the 4 s that remain.
    clock = FakeClock()
    transport = AdvancingTransport([(err(429), 25.0)] * 4, clock)
    with pytest.raises(ProviderRateLimited):
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}",
            timeout_s=30, sleep=clock.sleep,
            deadline_s=60.0, monotonic=clock.now,
        )
    assert [post["timeout_s"] for post in transport.posts] == [30, 30, 4.0]
    assert clock.sleeps == [2.0, 4.0]
    assert clock.t - 100.0 <= 60.0


def test_deadline_truncates_backoff_and_keeps_the_last_error():
    clock = FakeClock()
    transport = ScriptedTransport(
        [err(429, headers={"retry-after": "60"}), ok({"ok": True})]
    )
    with pytest.raises(ProviderRateLimited):
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=30,
            sleep=clock.sleep, deadline_s=40.0, monotonic=clock.now,
        )
    assert clock.sleeps == [40.0]  # truncated to the remaining budget
    assert len(transport.posts) == 1


def test_deadline_allows_success_within_budget():
    clock = FakeClock()
    transport = ScriptedTransport(
        [err(429, headers={"retry-after": "3"}), ok({"ok": True})]
    )
    response = post_with_retry(
        transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=30,
        sleep=clock.sleep, deadline_s=60.0, monotonic=clock.now,
    )
    assert response.status == 200
    assert clock.sleeps == [3.0]


@pytest.mark.parametrize("bad", [0, -5, True, "60"])
def test_deadline_rejects_non_positive(bad):
    clock = FakeClock()
    transport = ScriptedTransport([ok({"ok": True})])
    with pytest.raises(ValueError, match="deadline_s"):
        post_with_retry(
            transport, "https://x.test/r", headers={}, body=b"{}", timeout_s=5,
            sleep=clock.sleep, deadline_s=bad, monotonic=clock.now,
        )
    assert transport.posts == []


def test_client_bounds_the_logical_call_by_its_timeout():
    # The client passes its per-call timeout as the end-to-end deadline:
    # slow 429s stop the call instead of running past it.
    ctx, clock = FakeCtx(), FakeClock()
    transport = AdvancingTransport([(err(429), 20.0)] * 4, clock)
    client = make_spark_client(ctx, transport, clock)
    with pytest.raises(ProviderRateLimited):
        client.run_steps("vid-1")
    assert [post["timeout_s"] for post in transport.posts] == [30, 8.0]
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == Decimal("0")  # refused, never charged


def test_response_helpers_reject_bad_shapes():
    with pytest.raises(ProviderMalformed):
        response_json(HttpResponse(status=200, body=b"not json", headers={}))
    with pytest.raises(ProviderMalformed):
        output_text({"output": []})
    with pytest.raises(ProviderMalformed):
        output_text({"output": [{"content": [{"type": "other"}]}]})
    assert (
        output_text(
            {"output": [{"content": [{"type": "output_text", "text": "ab"}]}]}
        )
        == "ab"
    )
    assert usage_counts({"usage": {"input_tokens": 3, "output_tokens": 4}}) == (3, 4)
    with pytest.raises(ProviderMalformed):
        usage_counts({"usage": {"input_tokens": -1, "output_tokens": 4}})
    with pytest.raises(ProviderMalformed):
        usage_counts({})


def test_still_data_url_sniffs_png_jpeg_and_refuses_other(tmp_path):
    png = make_png(tmp_path / "a.png")
    assert still_data_url(png).startswith("data:image/png;base64,")
    jpg = tmp_path / "b.jpg"
    assert cv2.imwrite(str(jpg), np.full((8, 8, 3), 10, np.uint8))
    assert still_data_url(jpg).startswith("data:image/jpeg;base64,")
    weird = tmp_path / "c.bin"
    weird.write_bytes(b"definitely not an image")
    with pytest.raises(ProviderError):
        still_data_url(weird)
    with pytest.raises(ProviderError):
        still_data_url(tmp_path / "missing.png")


class _StubHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/slow":
            time.sleep(2)
            self._send(200, b'{"ok":true}')
        elif self.path == "/missing":
            self._send(404, b"{}")
        elif self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
        else:
            self._send(200, b'{"ok":true}')

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_api():
    server = http.server.HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = UrllibTransport()
    try:
        # Wait on the observable: the first accepted connection.
        for _ in range(200):
            try:
                transport.post(
                    f"http://127.0.0.1:{server.server_port}/ok",
                    headers={}, body=b"{}", timeout_s=2,
                )
            except ProviderError:
                time.sleep(0.01)
            else:
                break
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_urllib_transport_round_trip_status_and_lowercased_headers(stub_api):
    transport = UrllibTransport()
    response = transport.post(
        stub_api + "/ok", headers={"X-T": "1"}, body=b"{}", timeout_s=5
    )
    assert response.status == 200
    assert json.loads(response.body) == {"ok": True}
    assert response.headers["content-type"] == "application/json"
    missing = transport.post(stub_api + "/missing", headers={}, body=b"{}", timeout_s=5)
    assert missing.status == 404


def test_urllib_transport_timeout_and_redirect(stub_api):
    transport = UrllibTransport()
    with pytest.raises(ProviderTimeout):
        transport.post(stub_api + "/slow", headers={}, body=b"{}", timeout_s=0.2)
    with pytest.raises(ProviderError) as caught:
        transport.post(stub_api + "/redir", headers={}, body=b"{}", timeout_s=5)
    assert caught.value.retryable is False


# ---------------------------------------------------------------------------
# Spark steps client
# ---------------------------------------------------------------------------


def _step_payload(number=1, start=10.0, end=20.0, action="Remove the rear screws",
                  parts=("battery",), tools=("Phillips #0",), hazards=(),
                  kind="repair"):
    return {
        "number": number, "start_s": start, "end_s": end, "action": action,
        "tools": list(tools),
        "parts": [{"phrase": p, "position_hint": f"{p} area"} for p in parts],
        "hazards": list(hazards),
        "kind": kind,
    }


def _run_payload(steps, device="Test laptop", confidence=0.8):
    fixed = []
    for index, step in enumerate(steps, start=1):
        step = dict(step)
        step["number"] = index
        fixed.append(step)
    return {"device_guess": device, "confidence": confidence, "steps": fixed}


def test_spark_success_returns_payload_cost_latency_and_record():
    ctx, clock = FakeCtx(), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport([ok(spark_response(payload))])
    client = make_spark_client(ctx, transport, clock)
    result = client.run_steps("vid-1")
    assert result.payload == payload
    assert result.cost_usd == Decimal(16000) * IN_PRICE + Decimal(2000) * OUT_PRICE
    assert result.latency_ms == 0
    assert len(transport.posts) == 1
    post = transport.posts[0]
    assert post["url"] == "https://api.test/v1/responses"
    assert post["headers"]["Authorization"] == "Bearer k-test"
    sent = json.loads(post["body"].decode())
    assert sent["model"] == "muse-spark-1.3"
    assert sent["text"]["format"]["strict"] is True
    assert sent["input"][0]["content"][1] == {
        "type": "input_video", "file_id": "vid-1",
    }
    (record,) = stored_calls(ctx)
    assert record.stage == "extract"
    assert record.model == "spark-1.3"
    assert record.endpoint == "/responses"
    assert record.status == "succeeded"
    assert (record.units_in, record.units_out) == (16000, 2000)
    assert record.estimated_cost_usd == result.cost_usd
    assert record.evidence == {"run_index": 0}
    assert "k-test" not in repr(client)
    # The evidence module's own read path accepts the receipt.
    assert _evidence.list_calls(ctx)[0].evidence == {"run_index": 0}


def test_spark_run_index_counts_calls_on_the_instance():
    ctx, clock = FakeCtx(), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport([ok(spark_response(payload))] * 3)
    client = make_spark_client(ctx, transport, clock)
    for _ in range(3):
        client.run_steps("vid-1")
    assert [c.evidence["run_index"] for c in stored_calls(ctx)] == [0, 1, 2]


def test_spark_429_then_success_records_latency_with_backoff():
    ctx, clock = FakeCtx(), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport(
        [err(429, headers={"retry-after": "3"}), ok(spark_response(payload))]
    )
    client = make_spark_client(ctx, transport, clock)
    result = client.run_steps("vid-1")
    assert result.payload == payload
    assert clock.sleeps == [3.0]
    assert result.latency_ms == 3000
    assert len(transport.posts) == 2
    (record,) = stored_calls(ctx)
    assert record.status == "succeeded"
    assert record.latency_ms == 3000


def test_spark_timeout_returns_poison_at_worst_case(caplog):
    ctx, clock = FakeCtx(), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport(
        [ProviderTimeout("slow"), ok(spark_response(payload))]
    )
    client = make_spark_client(ctx, transport, clock)
    result = client.run_steps("vid-1")  # no raise: the wrapper settles it
    assert len(transport.posts) == 1  # never re-POST a timed-out paid call
    assert result.cost_usd == SPARK_WORST
    assert result.failure_kind == "timeout"
    assert isinstance(result.payload, str) and "INDETERMINATE" in result.payload
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == SPARK_WORST
    assert record.units_in is None and record.units_out is None
    assert record.evidence == {"run_index": 0}
    assert "reconcile" in caplog.text


def test_spark_timeout_settles_worst_case_through_the_wrapper():
    from app.spark import extract_steps

    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([ProviderTimeout("slow")])
    with pytest.raises(SparkClientError):
        extract_steps(ctx, "vid-1", client=make_spark_client(ctx, transport, clock))
    assert ctx.reserves == [("spark_steps", SPARK_WORST)]
    assert ctx.holds[0].settlements == [(SPARK_WORST, "failed")]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", SPARK_WORST)


@pytest.mark.parametrize("status", (500, 520))
def test_spark_500_returns_poison_at_worst_case(status):
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([err(status), err(status)])
    client = make_spark_client(ctx, transport, clock)
    result = client.run_steps("vid-1")
    assert len(transport.posts) == 1
    assert result.cost_usd == SPARK_WORST
    assert result.failure_kind == "http_5xx"
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", SPARK_WORST)


def test_spark_429_exhausted_raises_and_settles_zero():
    from app.spark import extract_steps

    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([err(429)] * 4)
    with pytest.raises(ProviderRateLimited):
        extract_steps(ctx, "vid-1", client=make_spark_client(ctx, transport, clock))
    assert ctx.holds[0].settlements == [(Decimal("0"), "failed")]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0"))


@pytest.mark.parametrize(
    "response,measured,kind",
    [
        (HttpResponse(status=200, body=b"", headers={}), False, "empty"),
        (HttpResponse(status=200, body=b"nope", headers={}), False, "malformed"),
        (ok({"nope": True}), False, "malformed"),
        (ok({"output": []}), False, "empty"),
        (
            ok({"output": [{"content": [{"type": "output_text", "text": "nope"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}),
            True, "malformed",
        ),
        (
            ok({"output": [{"content": [{"type": "output_text", "text": "[1,2]"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}),
            True, "malformed",
        ),
        (ok({"output": [{"content": [{"type": "output_text", "text": "{}"}]}]}), False, "malformed"),
        (
            ok({"output": [{"content": [{"type": "output_text", "text": "{}"}]}],
                "usage": {"input_tokens": 1}}),
            False, "malformed",
        ),
    ],
)
def test_spark_malformed_200_poisons_with_measured_or_worst(response, measured, kind):
    # A 200 was paid for: settle the measured usage when the body carries
    # it, else the worst case. Never $0, never a retry.
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([response])
    client = make_spark_client(ctx, transport, clock)
    result = client.run_steps("vid-1")
    expected = IN_PRICE + OUT_PRICE if measured else SPARK_WORST
    assert result.cost_usd == expected
    assert result.failure_kind == kind
    assert isinstance(result.payload, str)
    assert len(transport.posts) == 1  # deterministic shape: no retry
    assert clock.sleeps == []
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == expected


def test_spark_empty_output_with_zero_tokens_is_service_failure():
    ctx, clock = FakeCtx(), FakeClock()
    client = make_spark_client(
        ctx,
        ScriptedTransport([ok({"output": [{"content": []}],
                               "usage": {"input_tokens": 0, "output_tokens": 0}})]),
        clock,
    )
    result = client.run_steps("vid-1")
    assert result.failure_kind == "empty"
    assert result.cost_usd == Decimal("0")
    assert stored_calls(ctx)[0].status == "failed"


def test_spark_measured_poison_settles_measured_through_the_wrapper():
    from app.spark import extract_steps

    ctx, clock = FakeCtx(), FakeClock()
    bad = ok(
        {
            "output": [{"content": [{"type": "output_text", "text": "nope"}]}],
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
    )
    transport = ScriptedTransport([bad])
    with pytest.raises(SparkClientError):
        extract_steps(ctx, "vid-1", client=make_spark_client(ctx, transport, clock))
    expected = Decimal(100) * IN_PRICE + Decimal(10) * OUT_PRICE
    assert ctx.holds[0].settlements == [(expected, "failed")]


def test_spark_rejects_bad_input_and_bad_construction():
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([])
    client = make_spark_client(ctx, transport, clock)
    with pytest.raises(ValueError):
        client.run_steps("  ")
    assert transport.posts == []
    assert stored_calls(ctx) == ()
    with pytest.raises(ValueError):
        MetaSparkClient(
            api_key=" ", base_url="https://api.test/v1", model_id="m",
            input_price=IN_PRICE, output_price=OUT_PRICE, timeout_s=30, ctx=ctx,
        )
    with pytest.raises(ValueError):
        MetaSparkClient(
            api_key="k", base_url="https://api.test/v1", model_id="m",
            input_price=IN_PRICE, output_price=OUT_PRICE, timeout_s=0, ctx=ctx,
        )


# ---------------------------------------------------------------------------
# SAM client
# ---------------------------------------------------------------------------


def _box(x1=10, y1=20, x2=100, y2=200, w=320, h=240, c=0.72):
    tail = f";c={c}" if c is not None else ""
    return f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}{tail}|>"


def test_sam_success_through_wrapper_writes_detection_mask_and_record(
    tmp_path, fast_sam
):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box()))])
    client = make_sam_client(ctx, transport, clock)
    detections = app_sam.segment(ctx, still, "fan", client=client)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.phrase == "fan"
    assert detection.score == 0.72
    assert detection.box_xyxy == (10, 20, 100, 200)
    mask = cv2.imread(str(detection.mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None and mask.shape == (240, 320)
    assert mask[30, 50] == 255 and mask[0, 0] == 0
    sent = json.loads(transport.posts[0]["body"].decode())
    assert sent["model"] == "sam-3.1"
    assert sent["metadata"] == {"include_confidence": "true"}
    assert sent["input"][0]["content"][0] == {
        "type": "input_text", "text": "cooling fan",
    }
    assert sent["input"][0]["content"][1]["image_url"].startswith(
        "data:image/png;base64,"
    )
    assert ctx.reserves == [("sam", Decimal("0.0025"))]
    assert ctx.holds[0].settlements == [(Decimal("0.0025"), "succeeded")]
    (record,) = stored_calls(ctx)
    assert record.stage == "segment"
    assert record.model == "sam-3.1"
    assert record.endpoint == "/responses"
    assert record.status == "succeeded"
    assert record.units_in is None and record.units_out is None
    assert record.estimated_cost_usd == Decimal("0.0025")
    assert record.evidence == {"phrase": "fan", "score": 0.72}
    assert "k-test" not in repr(client)
    assert _evidence.list_calls(ctx)[0].evidence == {"phrase": "fan", "score": 0.72}


def test_sam_battery_uses_its_winning_phrase(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box(c=0.61)))])
    client = make_sam_client(ctx, transport, clock)
    result = client.segment_still(still, "battery")
    assert len(result.masks) == 1
    sent = json.loads(transport.posts[0]["body"].decode())
    assert sent["input"][0]["content"][0]["text"] == "rectangular battery with label"


@pytest.mark.parametrize("phrase,score", [("fan", 0.55), ("battery", 0.49)])
def test_sam_below_threshold_box_is_not_found(tmp_path, fast_sam, phrase, score):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box(c=score)))])
    client = make_sam_client(ctx, transport, clock)
    assert app_sam.segment(ctx, still, phrase, client=client) == ()
    (record,) = stored_calls(ctx)
    assert record.status == "succeeded"
    assert record.evidence == {"phrase": phrase, "score": score}


def test_sam_empty_text_is_not_found(tmp_path, fast_sam):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response("nothing here"))])
    client = make_sam_client(ctx, transport, clock)
    assert app_sam.segment(ctx, still, "fan", client=client) == ()
    (record,) = stored_calls(ctx)
    assert record.evidence == {"phrase": "fan"}


def test_sam_boxes_rescale_clamp_and_drop_garbage(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png", width=640, height=480)
    text = " ".join(
        [
            _box(x1=10, y1=20, x2=50, y2=60, w=320, h=240, c=0.7),
            _box(x1=0, y1=0, x2=0, y2=5, w=320, h=240, c=0.9),  # inside-out
            _box(x1=5, y1=5, x2=9, y2=9, w=320, h=240, c=1.5),  # c out of range
            _box(x1=1, y1=1, x2=2, y2=2, w=0, h=240, c=0.8),  # no sent size
        ]
    )
    transport = ScriptedTransport([ok(sam_response(text))])
    client = make_sam_client(ctx, transport, clock)
    result = client.segment_still(still, "  Fan ")
    assert [m.box_xyxy for m in result.masks] == [(20, 40, 100, 120)]
    assert [m.c for m in result.masks] == [0.7]
    (record,) = stored_calls(ctx)
    assert record.evidence == {"phrase": "fan", "score": 0.7}


def test_sam_429_then_success(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport(
        [err(429, headers={"retry-after": "1"}), ok(sam_response(_box()))]
    )
    client = make_sam_client(ctx, transport, clock)
    assert len(client.segment_still(still, "fan").masks) == 1
    assert clock.sleeps == [1.0]
    assert len(transport.posts) == 2


def test_sam_timeout_returns_indeterminate_status(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ProviderTimeout("slow"), ok(sam_response(_box()))])
    result = make_sam_client(ctx, transport, clock).segment_still(still, "fan")
    assert len(transport.posts) == 1  # never re-POST a timed-out paid call
    assert result.masks == ()
    assert result.status == "indeterminate"
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == Decimal("0.0025")


def test_sam_indeterminate_settles_full_price_through_the_wrapper(tmp_path, fast_sam):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([err(500), ok(sam_response(_box()))])
    client = make_sam_client(ctx, transport, clock)
    assert app_sam.segment(ctx, still, "fan", client=client) == ()
    assert len(transport.posts) == 1
    assert ctx.holds[0].settlements == [(Decimal("0.0025"), "failed")]


@pytest.mark.parametrize(
    "bad",
    [
        HttpResponse(status=200, body=b"nope", headers={}),
        ok({"output": []}),
    ],
)
def test_sam_malformed_200_returns_indeterminate_status(tmp_path, bad):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([bad])
    result = make_sam_client(ctx, transport, clock).segment_still(still, "fan")
    assert result.masks == ()
    assert result.status == "indeterminate"
    assert len(transport.posts) == 1
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == Decimal("0.0025")


def test_sam_429_exhausted_raises_and_settles_zero(tmp_path, fast_sam):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([err(429)] * 4)
    with pytest.raises(ProviderRateLimited):
        app_sam.segment(
            ctx, still, "fan", client=make_sam_client(ctx, transport, clock)
        )
    assert ctx.holds[0].settlements == [(Decimal("0"), "failed")]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0"))


def test_sam_unreadable_still_and_unsupported_phrase(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([])
    client = make_sam_client(ctx, transport, clock)
    with pytest.raises(ProviderError):
        client.segment_still(tmp_path / "missing.png", "fan")
    assert transport.posts == []
    assert stored_calls(ctx)[0].status == "failed"
    with pytest.raises(ValueError, match="unsupported SAM phrase"):
        client.segment_still(make_png(tmp_path / "s.png"), "ram module")
    assert transport.posts == []
    # Validation errors attempt no call, so they record nothing further.
    assert len(stored_calls(ctx)) == 1


def test_sam_constructor_takes_configured_price_and_thresholds():
    ctx = FakeCtx()
    # A configured price is used, not refused: the factory proves it
    # fits the budget before the client is ever built.
    MetaSamClient(
        api_key="k", base_url="https://x.test", model_id="sam-3.1",
        price_per_call=Decimal("0.001"), timeout_s=30,
        phrases={"fan": "f", "battery": "b"},
        thresholds={"fan": 0.9, "battery": 0.1}, ctx=ctx,
    )
    with pytest.raises(ValueError, match="price_per_call"):
        MetaSamClient(
            api_key="k", base_url="https://x.test", model_id="sam-3.1",
            price_per_call=Decimal("-1"), timeout_s=30,
            phrases={"fan": "f", "battery": "b"}, ctx=ctx,
        )
    with pytest.raises(ValueError, match="no API phrasing"):
        MetaSamClient(
            api_key="k", base_url="https://x.test", model_id="sam-3.1",
            price_per_call=Decimal("0.0025"), timeout_s=30,
            phrases={"fan": "f"}, ctx=ctx,
        )
    with pytest.raises(ValueError, match="no usable threshold"):
        MetaSamClient(
            api_key="k", base_url="https://x.test", model_id="sam-3.1",
            price_per_call=Decimal("0.0025"), timeout_s=30,
            phrases={"fan": "f", "battery": "b"},
            thresholds={"fan": 0.9}, ctx=ctx,
        )
    with pytest.raises(ValueError, match="no usable threshold"):
        MetaSamClient(
            api_key="k", base_url="https://x.test", model_id="sam-3.1",
            price_per_call=Decimal("0.0025"), timeout_s=30,
            phrases={"fan": "f", "battery": "b"},
            thresholds={"fan": 1.5, "battery": 0.5}, ctx=ctx,
        )


def test_sam_configured_threshold_filters_returned_boxes_not_the_receipt(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box(c=0.72)))])
    client = MetaSamClient(
        api_key="k-test", base_url="https://api.test/v1", model_id="sam-3.1",
        price_per_call=Decimal("0.0025"), timeout_s=30,
        phrases={"fan": "cooling fan", "battery": "rectangular battery with label"},
        thresholds={"fan": 0.9, "battery": 0.5}, ctx=ctx,
        transport=transport, sleep=clock.sleep, monotonic=clock.now,
    )
    assert client.segment_still(still, "fan").masks == ()
    (record,) = stored_calls(ctx)
    assert record.status == "succeeded"
    assert record.evidence == {"phrase": "fan", "score": 0.72}  # best seen kept


def test_sam_include_confidence_flag_reaches_the_request(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box(c=None)))])
    client = make_sam_client(ctx, transport, clock)
    result = client.segment_still(still, "fan", include_confidence=False)
    assert [m.c for m in result.masks] == [None]
    sent = json.loads(transport.posts[0]["body"].decode())
    assert sent["metadata"] == {"include_confidence": "false"}


# ---------------------------------------------------------------------------
# Read-back client
# ---------------------------------------------------------------------------


def _readback_response(visible, in_tok=1500, out_tok=60):
    return spark_response({"visible": visible}, in_tok, out_tok)


@pytest.mark.parametrize("visible,answer", [(True, "yes"), (False, "no")])
def test_readback_success_returns_answer_cost_and_record(tmp_path, visible, answer):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport([ok(_readback_response(visible))])
    client = make_readback_client(ctx, transport, clock)
    result = client.read_back(still, "fan", "motor housing")
    assert result.answer == answer
    assert result.cost_usd == Decimal(1500) * IN_PRICE + Decimal(60) * OUT_PRICE
    sent = json.loads(transport.posts[0]["body"].decode())
    assert sent["model"] == "muse-spark-1.3"
    assert sent["text"]["format"]["strict"] is True
    question = sent["input"][0]["content"][0]["text"]
    assert "'fan'" in question and "motor housing" in question
    (record,) = stored_calls(ctx)
    assert record.stage == "readback"
    assert record.model == "spark-1.3"
    assert record.endpoint == "/responses"
    assert record.status == "succeeded"
    assert (record.units_in, record.units_out) == (1500, 60)
    assert record.evidence == {"phrase": "fan", "decision": answer}
    assert "k-test" not in repr(client)
    assert _evidence.list_calls(ctx)[0].evidence == {
        "phrase": "fan", "decision": answer,
    }


def test_readback_empty_hint_asks_without_it(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport([ok(_readback_response(True))])
    client = make_readback_client(ctx, transport, clock)
    assert client.read_back(still, "battery", "").answer == "yes"
    question = json.loads(transport.posts[0]["body"].decode())["input"][0][
        "content"
    ][0]["text"]
    assert "'battery'" in question and "labelled near" not in question


def test_readback_429_then_success(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport(
        [err(429, headers={"retry-after": "2"}), ok(_readback_response(True))]
    )
    client = make_readback_client(ctx, transport, clock)
    assert client.read_back(still, "fan", "hub").answer == "yes"
    assert clock.sleeps == [2.0]


def test_readback_timeout_returns_poison_at_reserved_amount(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport(
        [ProviderTimeout("slow"), ok(_readback_response(True))]
    )
    result = make_readback_client(ctx, transport, clock).read_back(still, "fan", "hub")
    assert len(transport.posts) == 1  # never re-POST a timed-out paid call
    assert result.cost_usd == Decimal("0.04")
    assert result.answer.startswith("indeterminate:")
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == Decimal("0.04")


def test_readback_timeout_settles_reserved_through_the_wrapper(tmp_path):
    from app.contracts import ArtifactId, Detection, LabelSlot, PartRequest, ProvisionalRender

    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport([ProviderTimeout("slow")])
    with pytest.raises(ReadbackError) as caught:
        read_back_label(
            ctx,
            ProvisionalRender(
                detection=Detection(
                    phrase="fan", mask_path=still, score=0.65, box_xyxy=(1, 2, 3, 4)
                ),
                artifact_id=ArtifactId("art-1"),
                rendered_image_path=still,
                position_hint="motor housing",
            ),
            slot=LabelSlot(
                step_index=0, part_index=0,
                part=PartRequest(phrase="fan", position_hint="motor housing"),
            ),
            client=make_readback_client(ctx, transport, clock),
        )
    assert "indeterminate" in str(caught.value)  # the note rides the rejection
    assert ctx.holds[0].settlements == [(Decimal("0.04"), "failed")]
    assert ctx.finishes == ["failed"]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0.04"))


@pytest.mark.parametrize("text", ["nope", "[1]", '{"visible": "yes"}', '{"other": true}'])
def test_readback_malformed_200_with_usage_poisons_measured(tmp_path, text):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    bad = ok(
        {
            "output": [{"content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    result = make_readback_client(ctx, ScriptedTransport([bad]), clock).read_back(
        still, "fan", "hub"
    )
    assert result.cost_usd == IN_PRICE + OUT_PRICE
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == IN_PRICE + OUT_PRICE


def test_readback_200_without_usage_poisons_reserved(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    no_usage = ok(
        {"output": [{"content": [{"type": "output_text", "text": '{"visible": true}'}]}]}
    )
    result = make_readback_client(ctx, ScriptedTransport([no_usage]), clock).read_back(
        still, "fan", "hub"
    )
    assert result.cost_usd == Decimal("0.04")
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0.04"))


def test_readback_429_exhausted_raises_and_settles_zero(tmp_path):
    from app.contracts import ArtifactId, Detection, LabelSlot, PartRequest, ProvisionalRender

    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport([err(429)] * 4)
    with pytest.raises(ProviderRateLimited):
        read_back_label(
            ctx,
            ProvisionalRender(
                detection=Detection(
                    phrase="fan", mask_path=still, score=0.65, box_xyxy=(1, 2, 3, 4)
                ),
                artifact_id=ArtifactId("art-1"),
                rendered_image_path=still,
                position_hint="motor housing",
            ),
            slot=LabelSlot(
                step_index=0, part_index=0,
                part=PartRequest(phrase="fan", position_hint="motor housing"),
            ),
            client=make_readback_client(ctx, transport, clock),
        )
    assert ctx.holds[0].settlements == [(Decimal("0"), "failed")]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0"))


def test_readback_rejects_bad_input_without_a_call(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([])
    client = make_readback_client(ctx, transport, clock)
    still = make_png(tmp_path / "r.png")
    with pytest.raises(ValueError):
        client.read_back(still, "  ", "hub")
    with pytest.raises(ValueError):
        client.read_back(still, "fan", None)  # type: ignore[arg-type]
    assert transport.posts == []
    assert stored_calls(ctx) == ()


def test_readback_yes_and_no_through_the_wrapper(tmp_path):
    from app.contracts import ArtifactId, Detection, LabelSlot, PartRequest, ProvisionalRender

    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")

    def provisional():
        return ProvisionalRender(
            detection=Detection(
                phrase="fan", mask_path=still, score=0.65, box_xyxy=(1, 2, 3, 4)
            ),
            artifact_id=ArtifactId("art-1"),
            rendered_image_path=still,
            position_hint="motor housing",
        )

    def slot():
        return LabelSlot(
            step_index=0, part_index=0,
            part=PartRequest(phrase="fan", position_hint="motor housing"),
        )

    yes_transport = ScriptedTransport([ok(_readback_response(True))])
    yes = read_back_label(
        ctx, provisional(), slot=slot(),
        client=make_readback_client(ctx, yes_transport, clock),
    )
    assert (yes.checked, yes.status, yes.readback) == (
        True, "part_pixels_unchanged", "yes",
    )
    no_transport = ScriptedTransport([ok(_readback_response(False))])
    no = read_back_label(
        ctx, provisional(), slot=slot(),
        client=make_readback_client(ctx, no_transport, clock),
    )
    assert (no.checked, no.status, no.readback) == (False, "not_checked", "no")
    assert ctx.artifacts.deleted == [ArtifactId("art-1")]
    assert [c.evidence["decision"] for c in stored_calls(ctx)] == ["yes", "no"]


# ---------------------------------------------------------------------------
# Cancel: a BaseException records its receipt, then re-raises
# ---------------------------------------------------------------------------


class Boom(BaseException):
    pass


def test_spark_cancel_during_backoff_records_worst_case_and_reraises():
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([err(429, headers={"retry-after": "5"})])

    def interrupt(seconds):
        raise KeyboardInterrupt

    client = MetaSparkClient(
        api_key="k-test", base_url="https://api.test/v1", model_id="muse-spark-1.3",
        input_price=IN_PRICE, output_price=OUT_PRICE, timeout_s=30, ctx=ctx,
        transport=transport, sleep=interrupt, monotonic=clock.now,
    )
    with pytest.raises(KeyboardInterrupt):
        client.run_steps("vid-1")
    assert len(transport.posts) == 1
    (record,) = stored_calls(ctx)
    assert record.status == "failed"
    assert record.estimated_cost_usd == SPARK_WORST  # bytes may have flown


def test_spark_cancel_during_request_records_and_reraises():
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([KeyboardInterrupt()])
    client = make_spark_client(ctx, transport, clock)
    with pytest.raises(KeyboardInterrupt):
        client.run_steps("vid-1")
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", SPARK_WORST)


def test_sam_cancel_during_backoff_records_price_and_reraises(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([err(429, headers={"retry-after": "5"})])

    def interrupt(seconds):
        raise Boom

    client = MetaSamClient(
        api_key="k-test", base_url="https://api.test/v1", model_id="sam-3.1",
        price_per_call=Decimal("0.0025"), timeout_s=30,
        phrases={"fan": "cooling fan", "battery": "rectangular battery with label"},
        ctx=ctx, transport=transport, sleep=interrupt, monotonic=clock.now,
    )
    with pytest.raises(Boom):
        client.segment_still(still, "fan")
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0.0025"))


def test_readback_cancel_during_request_records_reserved_and_reraises(tmp_path):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport([KeyboardInterrupt()])
    client = make_readback_client(ctx, transport, clock)
    with pytest.raises(KeyboardInterrupt):
        client.read_back(still, "fan", "hub")
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0.04"))


def test_cancel_before_any_send_records_zero(tmp_path, monkeypatch):
    # The SAM still read happens before the first request byte: an
    # interrupt there cannot have been charged.
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box()))])

    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "read_bytes", interrupt)
    with pytest.raises(KeyboardInterrupt):
        make_sam_client(ctx, transport, clock).segment_still(still, "fan")
    assert transport.posts == []
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0"))


def test_cancel_during_receipt_write_propagates():
    ctx, clock = FakeCtx(), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport([ok(spark_response(payload))])
    store = FailingStore()
    ctx.store = store
    ctx.evidence_store = store

    def interrupt(key, call, event):
        raise KeyboardInterrupt

    store.record_call_with_event = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        make_spark_client(ctx, transport, clock).run_steps("vid-1")
    assert stored_calls(ctx) == ()


# ---------------------------------------------------------------------------
# Receipt persistence failure: the measured cost settles first
# ---------------------------------------------------------------------------


def test_spark_receipt_failure_settles_actual_through_the_wrapper():
    from app.spark import extract_steps

    ctx, clock = FakeCtx(), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport([ok(spark_response(payload))])
    store = FailingStore()
    ctx.store = store
    ctx.evidence_store = store
    with pytest.raises(SparkClientError):
        extract_steps(ctx, "vid-1", client=make_spark_client(ctx, transport, clock))
    assert store.write_attempts == 2  # initial write plus one retry
    expected = Decimal(16000) * IN_PRICE + Decimal(2000) * OUT_PRICE
    assert ctx.holds[0].settlements == [(expected, "failed")]
    assert stored_calls(ctx) == ()  # the receipt is lost, the cost is not


def test_spark_flaky_receipt_recovers_on_retry():
    ctx, clock = FakeCtx(), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport([ok(spark_response(payload))])
    store = FailingStore(fail_times=1)
    ctx.store = store
    ctx.evidence_store = store
    result = make_spark_client(ctx, transport, clock).run_steps("vid-1")
    assert result.payload == payload
    assert store.write_attempts == 2
    (record,) = stored_calls(ctx)
    assert record.status == "succeeded"


def test_sam_receipt_failure_settles_full_price_through_the_wrapper(
    tmp_path, fast_sam
):
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box()))])
    store = FailingStore()
    ctx.store = store
    ctx.evidence_store = store
    client = make_sam_client(ctx, transport, clock)
    assert app_sam.segment(ctx, still, "fan", client=client) == ()
    assert ctx.holds[0].settlements == [(Decimal("0.0025"), "failed")]
    assert stored_calls(ctx) == ()


def test_readback_receipt_failure_settles_actual_through_the_wrapper(tmp_path):
    from app.contracts import ArtifactId, Detection, LabelSlot, PartRequest, ProvisionalRender

    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport([ok(_readback_response(True))])
    store = FailingStore()
    ctx.store = store
    ctx.evidence_store = store
    with pytest.raises(ReadbackError):
        read_back_label(
            ctx,
            ProvisionalRender(
                detection=Detection(
                    phrase="fan", mask_path=still, score=0.65, box_xyxy=(1, 2, 3, 4)
                ),
                artifact_id=ArtifactId("art-1"),
                rendered_image_path=still,
                position_hint="motor housing",
            ),
            slot=LabelSlot(
                step_index=0, part_index=0,
                part=PartRequest(phrase="fan", position_hint="motor housing"),
            ),
            client=make_readback_client(ctx, transport, clock),
        )
    expected = Decimal(1500) * IN_PRICE + Decimal(60) * OUT_PRICE
    assert ctx.holds[0].settlements == [(expected, "failed")]
    assert stored_calls(ctx) == ()


# ---------------------------------------------------------------------------
# Reservation refused: no request is ever sent
# ---------------------------------------------------------------------------


def test_refused_spark_reservation_sends_no_request():
    from app.spark import extract_steps

    ctx, clock = FakeCtx(refuse_reserve=True), FakeClock()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport([ok(spark_response(payload))] * 3)
    with pytest.raises(ReserveRefused):
        extract_steps(ctx, "vid-1", client=make_spark_client(ctx, transport, clock))
    assert transport.posts == []
    assert stored_calls(ctx) == ()
    assert ctx.holds == []


def test_refused_sam_reservation_sends_no_request(tmp_path, fast_sam):
    ctx, clock = FakeCtx(refuse_reserve=True), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box()))])
    with pytest.raises(ReserveRefused):
        app_sam.segment(ctx, still, "fan", client=make_sam_client(ctx, transport, clock))
    assert transport.posts == []
    assert stored_calls(ctx) == ()


def test_refused_readback_reservation_sends_no_request(tmp_path):
    from app.contracts import ArtifactId, Detection, LabelSlot, PartRequest, ProvisionalRender

    ctx, clock = FakeCtx(refuse_reserve=True), FakeClock()
    still = make_png(tmp_path / "r.png")
    transport = ScriptedTransport([ok(_readback_response(True))])
    decision = read_back_label(
        ctx,
        ProvisionalRender(
            detection=Detection(
                phrase="fan", mask_path=still, score=0.65, box_xyxy=(1, 2, 3, 4)
            ),
            artifact_id=ArtifactId("art-1"),
            rendered_image_path=still,
            position_hint="motor housing",
        ),
        slot=LabelSlot(
            step_index=0, part_index=0,
            part=PartRequest(phrase="fan", position_hint="motor housing"),
        ),
        client=make_readback_client(ctx, transport, clock),
    )
    assert (decision.checked, decision.status, decision.readback) == (
        False, "not_checked", "not_run",
    )
    assert transport.posts == []
    assert stored_calls(ctx) == ()


# ---------------------------------------------------------------------------
# Median consensus and disagreement flag on three synthetic runs
# ---------------------------------------------------------------------------


def test_three_synthetic_runs_yield_median_times_and_flagged_spread():
    from app.spark import extract_steps

    ctx, clock = FakeCtx(), FakeClock()
    runs = [
        _run_payload(
            [
                _step_payload(start=10.0, end=20.0),
                _step_payload(
                    start=20.0, end=30.0, action="Lift out the fan",
                    parts=("fan",),
                ),
            ]
        ),
        _run_payload(
            [
                _step_payload(start=11.0, end=21.0),
                _step_payload(
                    start=21.0, end=31.0, action="Lift out the fan",
                    parts=("fan",),
                ),
            ]
        ),
        _run_payload(
            [
                _step_payload(start=12.0, end=22.0),
                _step_payload(
                    start=22.0, end=42.0, action="Lift out the fan",
                    parts=("fan",),
                ),
            ]
        ),
    ]
    transport = ScriptedTransport([ok(spark_response(run)) for run in runs])
    bundle = extract_steps(
        ctx, "vid-1", client=make_spark_client(ctx, transport, clock)
    )
    assert bundle.source_runs == 3
    assert bundle.timestamps_approximate is True
    first, second = bundle.steps
    # Time consensus: the {20, 21, 22} edges agree on 21, and the two
    # segments tile the observed [10, 42] span with no gaps.
    assert (first.start_s, first.end_s) == (10.0, 21.0)
    assert first.time_spread_s == 2.0
    assert first.time_needs_check is False
    assert (second.start_s, second.end_s) == (21.0, 42.0)
    assert second.time_spread_s == 12.0
    assert second.time_needs_check is True
    assert [c.evidence["run_index"] for c in stored_calls(ctx)] == [0, 1, 2]
    assert len(transport.posts) == 3


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _live_settings(**overrides):
    base = dict(
        PARTFIDELITY_PROVIDER_MODE="live",
        META_API_KEY="k-test",
    )
    base.update(overrides)
    return load_settings(base)


def test_factory_replay_delegates_to_the_trio_and_carries_config():
    from app.sam import sam_config_of
    from app.spark import spark_config_of

    class Recorded:
        def __init__(self, name):
            self.name = name

        def run_steps(self, video_file_id, clip_duration_s=None, scene_cuts=None, clocked=False):
            return (self.name, video_file_id)

    settings = load_settings({})
    trio = Providers(spark=Recorded("s"), sam=Recorded("m"), readback=Recorded("r"))
    chosen = select_providers(settings, replay=trio)
    assert chosen.readback is trio.readback
    assert chosen.spark.run_steps("v") == ("s", "v")  # delegates untouched
    assert chosen.sam.name == "m"
    assert spark_config_of(chosen.spark).run_shares == (SPARK_WORST,) * 3
    assert sam_config_of(chosen.sam).price_per_call == Decimal("0.0025")
    assert not hasattr(trio.spark, "spark_stage_config")  # the trio is not mutated
    with pytest.raises(RuntimeError, match="recorded trio"):
        select_providers(settings)


def test_factory_live_needs_ctx_and_key():
    settings = _live_settings()
    with pytest.raises(RuntimeError, match="job context"):
        select_providers(settings)
    keyless = Settings(
        data_root=settings.data_root,
        db_path=settings.db_path,
        production=False,
        meta_api_key=None,
        providers=ProviderSettings(mode="live"),
    )
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        select_providers(keyless, FakeCtx())


def test_factory_live_builds_bound_protocol_shaped_clients():
    from app.contracts import SamClient, SparkClient

    settings = _live_settings()
    ctx = FakeCtx()
    trio = select_providers(settings, ctx)
    assert isinstance(trio.spark, MetaSparkClient)
    assert isinstance(trio.sam, MetaSamClient)
    assert isinstance(trio.readback, MetaReadbackClient)
    assert callable(trio.spark.run_steps)
    assert callable(trio.sam.segment_still)
    assert callable(trio.readback.read_back)
    assert isinstance(trio.spark, SparkClient)
    assert isinstance(trio.sam, SamClient)
    assert isinstance(trio.readback, SparkClient)


def test_factory_rejects_unknown_mode():
    settings = _live_settings()
    broken = Settings(
        data_root=settings.data_root,
        db_path=settings.db_path,
        production=False,
        meta_api_key="k",
        providers=ProviderSettings(mode="sometimes"),
    )
    with pytest.raises(ValueError, match="unknown provider mode"):
        select_providers(broken, FakeCtx())


def test_factory_uses_configured_run_count():
    from app.spark import extract_steps

    settings = _live_settings(PARTFIDELITY_SPARK_RUNS="2")
    ctx, clock = FakeCtx(), FakeClock()
    runs = [
        _run_payload([_step_payload(start=10.0, end=20.0)]),
        _run_payload([_step_payload(start=12.0, end=22.0)]),
    ]
    transport = ScriptedTransport([ok(spark_response(run)) for run in runs])
    trio = select_providers(settings, ctx, transport=transport)
    bundle = extract_steps(ctx, "vid-1", client=trio.spark)
    assert bundle.source_runs == 2
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [(10.0, 22.0)]
    assert ctx.reserves == [("spark_steps", SPARK_WORST)] * 2
    assert [c.evidence["run_index"] for c in stored_calls(ctx)] == [0, 1]


def test_factory_uses_configured_spread():
    from app.spark import extract_steps

    settings = _live_settings(PARTFIDELITY_TIME_SPREAD_FLAG_S="7.5")
    ctx = FakeCtx()
    runs = [
        _run_payload([_step_payload(start=10.0, end=20.0)]),
        _run_payload([_step_payload(start=13.0, end=23.0)]),
        _run_payload([_step_payload(start=16.0, end=26.0)]),
    ]
    transport = ScriptedTransport([ok(spark_response(run)) for run in runs])
    trio = select_providers(settings, ctx, transport=transport)
    (step,) = extract_steps(ctx, "vid-1", client=trio.spark).steps
    assert step.time_spread_s == 6.0
    assert step.time_needs_check is False  # flagged at the default 5 s line


def test_factory_uses_configured_sam_rate(tmp_path):
    settings = _live_settings(PARTFIDELITY_SAM_RATE_PER_MIN="30")
    ctx, clock = FakeCtx(), FakeClock()
    app_sam.install_pacer(SamPacer(monotonic=clock.now, sleep=clock.sleep))
    app_sam.reset_sam_job_counts()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box()))] * 2)
    trio = select_providers(settings, ctx, transport=transport)
    assert trio.sam.sam_stage_config.interval_s == pytest.approx(2.0)
    app_sam.segment(ctx, still, "fan", client=trio.sam)
    app_sam.segment(ctx, still, "fan", client=trio.sam)
    assert clock.sleeps == [pytest.approx(2.0)]  # spaced at 30/min, not 55/min
    # Nothing process-wide changed: the shared pacer keeps its own rate.
    assert app_sam.default_pacer()._interval == pytest.approx(60 / 55)


def test_factory_uses_configured_thresholds(tmp_path):
    settings = _live_settings(PARTFIDELITY_SAM_THRESHOLD_FAN="0.9")
    ctx = FakeCtx()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box(c=0.72)))])
    trio = select_providers(settings, ctx, transport=transport)
    assert trio.sam.segment_still(still, "fan").masks == ()
    (record,) = stored_calls(ctx)
    assert record.evidence == {"phrase": "fan", "score": 0.72}


def test_factory_uses_configured_sam_price(tmp_path):
    settings = _live_settings(PARTFIDELITY_SAM_USD_PER_CALL="0.001")
    ctx = FakeCtx()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box()))])
    trio = select_providers(settings, ctx, transport=transport)
    assert len(trio.sam.segment_still(still, "fan").masks) == 1
    (record,) = stored_calls(ctx)
    assert record.estimated_cost_usd == Decimal("0.001")


@pytest.mark.parametrize(
    "var,value,match",
    [
        ("PARTFIDELITY_SPARK_RUNS", "5", "count cap"),
        ("PARTFIDELITY_SPARK_RUNS", "1", "consensus"),
        (
            "PARTFIDELITY_SPARK_INPUT_USD_PER_TOKEN",
            "1",
            "Spark reservation",
        ),
        (
            "PARTFIDELITY_SPARK_OUTPUT_USD_PER_TOKEN",
            "1",
            "Spark reservation",
        ),
        ("PARTFIDELITY_SAM_USD_PER_CALL", "0.5", "SAM reservation"),
    ],
)
def test_factory_refuses_configs_that_break_the_budget(var, value, match):
    settings = _live_settings(**{var: value})
    transport = ScriptedTransport([])
    with pytest.raises(RuntimeError, match=match) as caught:
        select_providers(settings, FakeCtx(), transport=transport)
    assert "diverge" not in str(caught.value)
    assert transport.posts == []  # refused at startup, before any request
    replay = load_settings({var: value})
    with pytest.raises(RuntimeError, match=match):
        select_providers(replay, replay=Providers(spark=1, sam=2, readback=3))


def test_factory_derives_default_shares_from_prices():
    from app.spark import extract_steps

    ctx = FakeCtx()
    payload = _run_payload([_step_payload()])
    transport = ScriptedTransport([ok(spark_response(payload))] * 3)
    trio = select_providers(_live_settings(), ctx, transport=transport)
    extract_steps(ctx, "vid-1", client=trio.spark)
    assert ctx.reserves == [("spark_steps", SPARK_WORST)] * 3
    assert sum((share for _, share in ctx.reserves), Decimal("0")) == Decimal("0.2475")


def test_worst_case_math_follows_the_active_prices():
    assert spark_worst_case(Decimal("1"), Decimal("2")) == Decimal("52000")
    assert readback_worst_case(Decimal("1"), Decimal("2")) == Decimal("12000")


# ---------------------------------------------------------------------------
# job deadline, interrupts, per-provider config, post-send
# failures, zero prices, model ids
# ---------------------------------------------------------------------------


class DeadlineCtx(FakeCtx):
    """FakeCtx that reports the ledger's remaining seconds."""

    def __init__(self, remaining_s=240.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._remaining = remaining_s

    def remaining_s(self):
        return self._remaining


def test_two_spark_runs_cannot_exceed_the_job_deadline():
    from app.spark import extract_steps

    ctx, clock = DeadlineCtx(240.0), FakeClock()
    started = clock.t
    payload = _run_payload([_step_payload()])
    transport = AdvancingTransport(
        [
            (ok(spark_response(payload)), 179.0),  # run 1 lands at ~179 s
            (ProviderTimeout("stalled"), 500.0),  # run 2 would stall past 240 s
        ],
        clock,
    )
    trio = select_providers(
        _live_settings(), ctx, transport=transport, monotonic=clock.now
    )
    with pytest.raises(SparkClientError):
        extract_steps(ctx, "vid-1", client=trio.spark)
    assert transport.posts[0]["timeout_s"] == 180.0
    assert transport.posts[1]["timeout_s"] == pytest.approx(61.0)
    assert clock.t - started <= 240.0  # the job deadline, not 179 + 180
    # The expired run settles conservatively (indeterminate: worst case).
    (first,), (second,) = [h.settlements for h in ctx.holds]
    assert first[1] == "succeeded" and first[0] < SPARK_WORST
    assert second == (SPARK_WORST, "failed")


def test_spark_backoff_is_truncated_to_the_job_deadline():
    ctx, clock = DeadlineCtx(), FakeClock()
    transport = AdvancingTransport(
        [(err(429, headers={"retry-after": "60"}), 0.0)] * 4, clock
    )
    client = MetaSparkClient(
        api_key="k-test", base_url="https://api.test/v1", model_id="muse-spark-1.3",
        input_price=IN_PRICE, output_price=OUT_PRICE, timeout_s=180, ctx=ctx,
        deadline_at=clock.t + 30.0, transport=transport,
        sleep=clock.sleep, monotonic=clock.now,
    )
    with pytest.raises(ProviderRateLimited):
        client.run_steps("vid-1")
    assert clock.sleeps == [30.0]  # one wait, cut at the deadline
    assert len(transport.posts) == 1


@pytest.mark.parametrize("which", ["spark", "sam", "readback"])
def test_call_after_the_job_deadline_refuses_before_send(tmp_path, fast_sam, which):
    from app.providers._transport import ProviderDeadline

    ctx, clock = DeadlineCtx(0.0), FakeClock()
    transport = ScriptedTransport([])
    trio = select_providers(
        _live_settings(), ctx, transport=transport, monotonic=clock.now
    )
    still = make_png(tmp_path / "s.png")
    with pytest.raises(ProviderDeadline):
        if which == "spark":
            trio.spark.run_steps("vid-1")
        elif which == "sam":
            trio.sam.segment_still(still, "fan")
        else:
            trio.readback.read_back(still, "fan", "")
    assert transport.posts == []
    assert [c.estimated_cost_usd for c in stored_calls(ctx)] == [Decimal("0")]


class Interrupting:
    """A client whose every call raises KeyboardInterrupt mid-request."""

    def run_steps(self, video_file_id, clip_duration_s=None, scene_cuts=None, clocked=False):
        raise KeyboardInterrupt

    def segment_still(self, image_path, phrase, *, include_confidence=True):
        raise KeyboardInterrupt

    def read_back(self, image_path, phrase, position_hint):
        raise KeyboardInterrupt


def test_spark_wrapper_settles_its_share_on_keyboard_interrupt():
    from app.spark import SPARK_RUN_SHARES, extract_steps

    ctx = FakeCtx()
    with pytest.raises(KeyboardInterrupt):
        extract_steps(ctx, "vid-1", client=Interrupting())
    assert [h.settlements for h in ctx.holds] == [[(SPARK_RUN_SHARES[0], "failed")]]


def test_sam_wrapper_settles_the_price_on_keyboard_interrupt(tmp_path, fast_sam):
    ctx = FakeCtx()
    still = make_png(tmp_path / "s.png")
    with pytest.raises(KeyboardInterrupt):
        app_sam.segment(ctx, still, "fan", client=Interrupting())
    assert [h.settlements for h in ctx.holds] == [[(Decimal("0.0025"), "failed")]]


def test_readback_wrapper_settles_the_hold_on_keyboard_interrupt(tmp_path):
    from app.contracts import ArtifactId, Detection, LabelSlot, PartRequest, ProvisionalRender

    ctx = FakeCtx()
    still = make_png(tmp_path / "r.png")
    provisional = ProvisionalRender(
        detection=Detection(phrase="fan", mask_path=still, score=0.65, box_xyxy=(1, 2, 3, 4)),
        artifact_id=ArtifactId("art-1"),
        rendered_image_path=still,
        position_hint="",
    )
    slot = LabelSlot(step_index=0, part_index=0, part=PartRequest(phrase="fan", position_hint=""))
    with pytest.raises(KeyboardInterrupt):
        read_back_label(ctx, provisional, slot=slot, client=Interrupting())
    assert [h.settlements for h in ctx.holds] == [[(Decimal("0.04"), "failed")]]
    assert ctx.finishes == ["failed"]


def test_segment_applies_the_configured_threshold_and_price(tmp_path, fast_sam):
    settings = _live_settings(
        PARTFIDELITY_SAM_THRESHOLD_FAN="0.5", PARTFIDELITY_SAM_USD_PER_CALL="0.001"
    )
    ctx = FakeCtx()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box(c=0.55)))])
    trio = select_providers(settings, ctx, transport=transport)
    (detection,) = app_sam.segment(ctx, still, "fan", client=trio.sam)
    assert detection.score == 0.55  # survives the configured 0.5, not the frozen 0.6
    assert ctx.reserves == [("sam", Decimal("0.001"))]
    assert ctx.holds[0].settlements == [(Decimal("0.001"), "succeeded")]
    (record,) = stored_calls(ctx)
    assert record.estimated_cost_usd == Decimal("0.001")


@pytest.mark.parametrize("order", ["three_first", "two_first"])
def test_providers_with_different_run_counts_each_run_their_own(order):
    from app.spark import extract_steps

    def build(runs):
        ctx = FakeCtx()
        payloads = [_run_payload([_step_payload()])] * runs
        transport = ScriptedTransport([ok(spark_response(p)) for p in payloads])
        settings = _live_settings(PARTFIDELITY_SPARK_RUNS=str(runs))
        return ctx, transport, select_providers(settings, ctx, transport=transport)

    built = {}
    for runs in ((3, 2) if order == "three_first" else (2, 3)):
        built[runs] = build(runs)  # both selected before either runs
    for runs in (3, 2):
        ctx, transport, trio = built[runs]
        bundle = extract_steps(ctx, "vid-1", client=trio.spark)
        assert bundle.source_runs == runs
        assert len(transport.posts) == runs
        assert ctx.reserves == [("spark_steps", SPARK_WORST)] * runs


def test_spark_unexpected_failure_after_send_settles_worst_case(monkeypatch):
    from app.providers import meta_spark
    from app.spark import extract_steps

    def boom(payload):
        raise RuntimeError("local bug after a paid 200")

    monkeypatch.setattr(meta_spark, "usage_counts", boom)
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport([ok(spark_response(_run_payload([_step_payload()])))])
    client = make_spark_client(ctx, transport, clock)
    with pytest.raises(SparkClientError):
        extract_steps(ctx, "vid-1", client=client)
    assert ctx.holds[0].settlements == [(SPARK_WORST, "failed")]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", SPARK_WORST)


def test_sam_unexpected_failure_after_send_settles_the_price(
    tmp_path, fast_sam, monkeypatch
):
    from app.providers import meta_sam

    def boom(ext, image):
        raise RuntimeError("local mask failure after a paid 200")

    monkeypatch.setattr(meta_sam.cv2, "imencode", boom)
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "s09.png")
    transport = ScriptedTransport([ok(sam_response(_box()))])
    client = make_sam_client(ctx, transport, clock)
    assert app_sam.segment(ctx, still, "fan", client=client) == ()
    assert ctx.holds[0].settlements == [(Decimal("0.0025"), "failed")]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0.0025"))


def test_readback_unexpected_failure_after_send_settles_the_hold(tmp_path, monkeypatch):
    from app.contracts import ArtifactId, Detection, LabelSlot, PartRequest, ProvisionalRender
    from app.providers import meta_readback

    def boom(payload):
        raise RuntimeError("local bug after a paid 200")

    monkeypatch.setattr(meta_readback, "usage_counts", boom)
    ctx, clock = FakeCtx(), FakeClock()
    still = make_png(tmp_path / "r.png")
    provisional = ProvisionalRender(
        detection=Detection(phrase="fan", mask_path=still, score=0.65, box_xyxy=(1, 2, 3, 4)),
        artifact_id=ArtifactId("art-1"),
        rendered_image_path=still,
        position_hint="",
    )
    slot = LabelSlot(step_index=0, part_index=0, part=PartRequest(phrase="fan", position_hint=""))
    transport = ScriptedTransport([ok(_readback_response(True))])
    with pytest.raises(ReadbackError):
        read_back_label(
            ctx, provisional, slot=slot,
            client=make_readback_client(ctx, transport, clock),
        )
    assert ctx.holds[0].settlements == [(Decimal("0.04"), "failed")]
    (record,) = stored_calls(ctx)
    assert (record.status, record.estimated_cost_usd) == ("failed", Decimal("0.04"))


@pytest.mark.parametrize(
    "field", ["spark_input_usd_per_token", "spark_output_usd_per_token", "sam_usd_per_call"]
)
def test_zero_prices_are_refused_before_any_request(field):
    settings = _live_settings()
    zeroed = Settings(
        data_root=settings.data_root,
        db_path=settings.db_path,
        production=False,
        meta_api_key="k",
        providers=ProviderSettings(mode="live", **{field: Decimal("0")}),
    )
    transport = ScriptedTransport([])
    with pytest.raises(RuntimeError, match=f"{field}=.*positive"):
        select_providers(zeroed, FakeCtx(), transport=transport)
    assert transport.posts == []
    with pytest.raises(RuntimeError, match="positive"):
        load_settings({"PARTFIDELITY_PROVIDER_MODE": "live", "META_API_KEY": "k",
                       "PARTFIDELITY_SAM_USD_PER_CALL": "0"})


def test_configured_model_id_reaches_receipts_and_cache_keys():
    from app.spark import SparkStageConfig, consensus_digest, run_digest

    ctx = FakeCtx()
    transport = ScriptedTransport([ok(spark_response(_run_payload([_step_payload()])))])
    trio = select_providers(_live_settings(), ctx, transport=transport)
    config = trio.spark.spark_stage_config
    assert config.model_id == "muse-spark-1.3"
    trio.spark.run_steps("vid-1")
    assert json.loads(transport.posts[0]["body"])["model"] == config.model_id
    (record,) = stored_calls(ctx)
    assert record.model == "spark-1.3"  # the receipt name mapped from the id
    other = SparkStageConfig(model_id="muse-spark-2.0")
    assert consensus_digest("j", "v", config) != consensus_digest("j", "v", other)
    assert run_digest("j", "v", 0, config) != run_digest("j", "v", 0, other)
    with pytest.raises(ValueError, match="unknown Spark model id"):
        MetaSparkClient(
            api_key="k", base_url="https://api.test/v1", model_id="muse-spark-2.0",
            input_price=IN_PRICE, output_price=OUT_PRICE, timeout_s=30, ctx=ctx,
        )
    with pytest.raises(ValueError, match="unknown SAM model id"):
        MetaSamClient(
            api_key="k", base_url="https://api.test/v1", model_id="sam-9",
            price_per_call=Decimal("0.0025"), timeout_s=30,
            phrases={"fan": "f", "battery": "b"}, ctx=ctx,
        )


def test_spark_wrapper_settles_zero_when_the_deadline_has_passed():
    from app.providers._transport import ProviderDeadline
    from app.spark import extract_steps

    ctx, clock = DeadlineCtx(0.0), FakeClock()
    transport = ScriptedTransport([])
    trio = select_providers(_live_settings(), ctx, transport=transport, monotonic=clock.now)
    with pytest.raises((ProviderDeadline, SparkClientError)):
        extract_steps(ctx, "vid-1", client=trio.spark)
    assert transport.posts == []
    assert [h.settlements for h in ctx.holds] == [[(Decimal("0"), "failed")]]


# ---------------------------------------------------------------------------
# the job deadline bounds the in-flight and pacing waits
# ---------------------------------------------------------------------------


def _paced_sam(ctx, clock, transport):
    pacer = SamPacer(monotonic=clock.now, sleep=clock.sleep)
    app_sam.install_pacer(pacer)
    trio = select_providers(_live_settings(), ctx, transport=transport, monotonic=clock.now)
    return pacer, trio.sam


def test_sam_pacing_slot_past_the_deadline_refuses_without_sending(tmp_path):
    records = []
    app_sam.install_call_sink(lambda ctx, record: records.append(record))
    try:
        ctx, clock = DeadlineCtx(1.0), FakeClock()
        transport = ScriptedTransport([])
        pacer, client = _paced_sam(ctx, clock, transport)
        with pacer.paced_call(60.0):  # the next slot is now 60 s ahead
            pass
        slot = pacer._next_start
        with pytest.raises(app_sam.SamDeadlineExceeded):
            app_sam.segment(ctx, make_png(tmp_path / "s.png"), "fan", client=client)
    finally:
        app_sam.reset_call_sink()
    assert transport.posts == [] and ctx.reserves == [] and clock.sleeps == []
    assert pacer._next_start == slot  # the slot was not consumed
    (receipt,) = stored_calls(ctx)
    assert receipt.estimated_cost_usd == Decimal("0") and receipt.status == "failed"
    assert [(r.status, r.estimated_cost_usd) for r in records] == [("deadline", Decimal("0"))]


def test_sam_pacing_slot_inside_the_deadline_waits_and_sends(tmp_path):
    ctx, clock = DeadlineCtx(120.0), FakeClock()
    transport = ScriptedTransport([ok(sam_response(""))])
    pacer, client = _paced_sam(ctx, clock, transport)
    with pacer.paced_call(60.0):
        pass
    app_sam.segment(ctx, make_png(tmp_path / "s.png"), "fan", client=client)
    assert clock.sleeps == [60.0] and len(transport.posts) == 1


def test_sam_pacing_sleep_past_the_deadline_refuses_without_sending(tmp_path):
    records = []
    app_sam.install_call_sink(lambda ctx, record: records.append(record))
    try:
        ctx, clock = DeadlineCtx(120.0), FakeClock()
        transport = ScriptedTransport([])
        pacer, client = _paced_sam(ctx, clock, transport)
        with pacer.paced_call(60.0):  # the next slot is 60 s ahead, inside the deadline
            pass
        slot = pacer._next_start
        requested: list[float] = []

        def overshooting_sleep(seconds: float) -> None:
            requested.append(seconds)
            clock.sleep(seconds + 120.0)  # overshoot past the job deadline

        pacer._sleep = overshooting_sleep
        with pytest.raises(app_sam.SamDeadlineExceeded):
            app_sam.segment(ctx, make_png(tmp_path / "s.png"), "fan", client=client)
    finally:
        app_sam.reset_call_sink()
    assert requested == [60.0]
    assert transport.posts == [] and ctx.reserves == [] and clock.sleeps == [180.0]
    assert pacer._next_start == slot  # the slot was not consumed
    (receipt,) = stored_calls(ctx)
    assert receipt.estimated_cost_usd == Decimal("0") and receipt.status == "failed"
    assert [(r.status, r.estimated_cost_usd) for r in records] == [("deadline", Decimal("0"))]


def test_sam_ninth_call_gives_up_on_a_full_in_flight_pool_by_its_deadline(tmp_path):
    ctx, clock = DeadlineCtx(0.2), FakeClock()
    transport = ScriptedTransport([])
    _pacer, client = _paced_sam(ctx, clock, transport)
    still = make_png(tmp_path / "s.png")
    held = 0
    try:
        for _ in range(app_sam.SAM_MAX_IN_FLIGHT):  # eight calls in flight
            assert app_sam._sam_in_flight.acquire(timeout=1)
            held += 1
        began = time.monotonic()
        with pytest.raises(app_sam.SamDeadlineExceeded):
            app_sam.segment(ctx, still, "fan", client=client)
        waited = time.monotonic() - began
    finally:
        for _ in range(held):
            app_sam._sam_in_flight.release()
    assert 0.15 <= waited < 2.0
    assert transport.posts == [] and ctx.reserves == []
    assert [c.estimated_cost_usd for c in stored_calls(ctx)] == [Decimal("0")]


def test_spark_prompt_pins_the_checked_part_vocabulary():
    # The part-check gates (parts.is_checked_part, sam.segment, the SAM
    # client's phrase map) match the exact names "fan" and "battery";
    # free-text Spark output never emits them, so the prompt must pin
    # them or the SAM and read-back stages never run live.
    assert 'the phrase "fan"' in SPARK_PROMPT
    assert 'the phrase "battery"' in SPARK_PROMPT
