"""Hosted SAM stills: pacing, budget, thresholds, and honest refusal.

All hosted calls run through RecordingSamClient on a fake clock: no sleeps,
no key, no network.
"""

import fcntl
import inspect
import json
import os
import tempfile
import threading
import time
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import contracts
from app.contracts import SAM_COST_PER_CALL_USD, SAM_MAX_CALLS_PER_JOB
from app.sam import (
    PNG_MAGIC,
    SamCallLimitExceeded,
    SamClient,
    SamMask,
    SamPacer,
    SamStillResult,
    choose_detection,
    install_call_sink,
    install_pacer,
    publish_staged,
    reset_call_sink,
    reset_pacer,
    reset_sam_job_counts,
    segment,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "sam_responses.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def mask_png(seed: int = 0) -> bytes:
    pattern = np.zeros((8, 8), np.uint8)
    pattern[::2, ::2] = 255
    pattern[0, 0] = seed % 256
    ok, buf = cv2.imencode(".png", pattern)
    assert ok
    return bytes(buf)


def case_result(name: str) -> SamStillResult:
    masks = tuple(
        SamMask(
            box_xyxy=tuple(m["box_xyxy"]),
            c=m["c"],
            mask_png=mask_png(i),
        )
        for i, m in enumerate(CASES[name]["masks"])
    )
    return SamStillResult(masks=masks)


class RecordingReservation:
    def __init__(self) -> None:
        self.settles: list[tuple[Decimal, str]] = []

    def settle(self, actual_usd: Decimal, outcome) -> None:
        self.settles.append((actual_usd, outcome))


class RecordingJobContext:
    def __init__(self, job_id: str = "job-1", events=None) -> None:
        self._job_id = job_id
        self.events = events if events is not None else []
        self.reserves: list[tuple[str, Decimal]] = []
        self.reservations: list[RecordingReservation] = []
        self.finishes: list[str] = []

    @property
    def job_id(self):
        return self._job_id

    @property
    def artifacts(self):
        raise AssertionError("segment never touches artifacts")

    def reserve(self, kind: str, worst_case_usd: Decimal) -> RecordingReservation:
        self.events.append(("reserve", kind, worst_case_usd))
        self.reserves.append((kind, worst_case_usd))
        reservation = RecordingReservation()
        self.reservations.append(reservation)
        return reservation

    def finish(self, outcome, *, keep_artifact_ids=(), failed_output_names=None) -> None:
        self.finishes.append(outcome)


class RecordingSamClient:
    """Fake SamClient; script entries are SamStillResults or Exceptions."""

    def __init__(self, clock: FakeClock, script, events=None) -> None:
        self.clock = clock
        self.script = list(script)
        self.events = events if events is not None else []
        self.calls: list[dict] = []

    def segment_still(self, image_path, phrase, *, include_confidence=True):
        self.events.append(("invoke", str(image_path), phrase))
        self.calls.append(
            {
                "image_path": image_path,
                "phrase": phrase,
                "include_confidence": include_confidence,
                "start": self.clock.now(),
            }
        )
        entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(entry, Exception):
            raise entry
        return entry


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture(autouse=True)
def _fake_time_and_clean_state(clock):
    install_pacer(SamPacer(monotonic=clock.now, sleep=clock.sleep))
    reset_sam_job_counts()
    reset_call_sink()
    yield
    reset_pacer()
    reset_sam_job_counts()
    reset_call_sink()


@pytest.fixture
def records():
    return []


@pytest.fixture
def sink(records):
    install_call_sink(lambda ctx, record: records.append((ctx, record)))
    return records


def test_segment_and_choose_signatures_match_interfaces():
    assert list(inspect.signature(segment).parameters) == [
        "ctx",
        "image_path",
        "phrase",
        "client",
    ]
    assert (
        inspect.signature(segment).parameters["client"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert list(inspect.signature(choose_detection).parameters) == [
        "detections",
        "position_hint",
    ]


def test_sam_client_surface_extends_contracts(clock):
    assert issubclass(SamClient, contracts.SamClient)
    fake = RecordingSamClient(clock, [case_result("empty")])
    assert isinstance(fake, SamClient)
    assert isinstance(fake, contracts.SamClient)


def test_include_confidence_parses_c(tmp_path, clock, sink):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("fan_single_above")])
    (detection,) = segment(ctx, still, "fan", client=client)
    assert client.calls[0]["include_confidence"] is True
    assert detection.phrase == "fan"
    assert detection.score == pytest.approx(0.72)
    assert detection.box_xyxy == (12, 30, 140, 150)
    assert detection.mask_path.read_bytes().startswith(PNG_MAGIC)
    assert ctx.reservations[0].settles == [(SAM_COST_PER_CALL_USD, "succeeded")]
    (_, record), = sink
    assert record.status == "ok"
    assert record.model == "sam-3.1"
    assert record.evidence == {"phrase": "fan", "c": 0.72, "request_index": 1}


def test_unsupported_parts_are_not_checked(tmp_path, clock, sink):
    still = tmp_path / "still.png"
    still.touch()
    for phrase in ("ram", "ssd", "connector", "screws", "battery connector"):
        ctx = RecordingJobContext()
        client = RecordingSamClient(clock, [case_result("fan_single_above")])
        assert segment(ctx, still, phrase, client=client) == ()
        assert ctx.reserves == []
        assert client.calls == []
    assert sink == []


def test_empty_or_ambiguous_result_has_no_label(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    empty_client = RecordingSamClient(clock, [case_result("empty")])
    assert segment(ctx, still, "fan", client=empty_client) == ()
    pair_client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    pair = segment(ctx, still, "fan", client=pair_client)
    assert len(pair) == 2
    assert choose_detection(pair, "the fan") is None
    assert choose_detection((), "the fan") is None


def test_fan_below_threshold_returns_no_detection(tmp_path, clock, sink):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("fan_below")])
    assert segment(ctx, still, "fan", client=client) == ()
    assert ctx.reservations[0].settles == [(SAM_COST_PER_CALL_USD, "succeeded")]
    assert sink[0][1].status == "below_threshold"
    assert sink[0][1].evidence["c"] == pytest.approx(0.55)


def test_missing_confidence_returns_no_detection(tmp_path, clock, sink):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("missing_c")])
    assert segment(ctx, still, "battery", client=client) == ()
    assert sink[0][1].status == "missing_confidence"


def test_sam_reservation_happens_before_client_invocation(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    events: list = []
    ctx = RecordingJobContext(events=events)
    client = RecordingSamClient(clock, [case_result("battery_single_above")],
                                events=events)
    segment(ctx, still, "battery", client=client)
    assert [kind for kind, *_ in events] == ["reserve", "invoke"]
    assert ctx.reserves == [("sam", Decimal("0.0025"))]


def test_sam_33rd_call_is_refused_before_client_invocation(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext(job_id="job-1")
    client = RecordingSamClient(clock, [case_result("fan_single_above")])
    for _ in range(SAM_MAX_CALLS_PER_JOB):
        segment(ctx, still, "fan", client=client)
    assert len(client.calls) == 32
    with pytest.raises(SamCallLimitExceeded):
        segment(ctx, still, "fan", client=client)
    assert len(client.calls) == 32
    assert len(ctx.reserves) == 32


def test_per_job_bound_is_scoped_to_job(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    first = RecordingJobContext(job_id="job-1")
    client = RecordingSamClient(clock, [case_result("fan_single_above")])
    for _ in range(SAM_MAX_CALLS_PER_JOB):
        segment(first, still, "fan", client=client)
    second = RecordingJobContext(job_id="job-2")
    (detection,) = segment(second, still, "fan", client=client)
    assert detection.score == pytest.approx(0.72)


def test_segment_client_exception_settles_hold_once_as_failed(
    tmp_path, clock, sink
):
    still = tmp_path / "still.png"
    still.touch()
    boom = RuntimeError("hosted SAM exploded")
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [boom])
    with pytest.raises(RuntimeError) as caught:
        segment(ctx, still, "fan", client=client)
    assert caught.value is boom
    assert ctx.reservations[0].settles == [(Decimal("0"), "failed")]
    (_, record), = sink
    assert record.status == "error"
    assert record.estimated_cost_usd == Decimal("0")


def test_sampler_paces_at_55_per_minute(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("fan_single_above")])
    for _ in range(4):
        segment(ctx, still, "fan", client=client)
    starts = [call["start"] for call in client.calls]
    assert starts[0] == pytest.approx(1000.0)
    for first, second in zip(starts, starts[1:]):
        assert second - first == pytest.approx(60.0 / 55)


def test_sampler_paces_concurrent_request_starts(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("fan_single_above")])
    barrier = threading.Barrier(8)
    errors: list = []

    def run() -> None:
        try:
            barrier.wait(timeout=30)
            for _ in range(4):
                segment(ctx, still, "fan", client=client)
        except Exception as exc:  # noqa: BLE001 — re-raised on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    starts = sorted(call["start"] for call in client.calls)
    assert len(starts) == 32
    interval = 60.0 / 55
    for first, second in zip(starts, starts[1:]):
        assert second - first >= interval - 1e-9


class GatedReserveContext(RecordingJobContext):
    """Job context whose reserve blocks until the test opens the gate."""

    def __init__(self, gate: threading.Event, entered: list) -> None:
        super().__init__()
        self._gate = gate
        self._entered = entered
        self._lock = threading.Lock()

    def reserve(self, kind: str, worst_case_usd: Decimal):
        with self._lock:
            self._entered.append(kind)
        assert self._gate.wait(timeout=60)
        return super().reserve(kind, worst_case_usd)


def test_sampler_preempted_thread_cannot_start_with_later_slot(
    tmp_path, clock
):
    still = tmp_path / "still.png"
    still.touch()
    gate = threading.Event()
    entered: list = []
    ctx = GatedReserveContext(gate, entered)
    client = RecordingSamClient(clock, [case_result("fan_single_above")])
    errors: list = []

    def run() -> None:
        try:
            segment(ctx, still, "fan", client=client)
        except Exception as exc:  # noqa: BLE001 — re-raised on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        # Hold both threads pre-invocation: on unfixed code both take their
        # slots, then start nearly simultaneously once released.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(entered) < 2:
            time.sleep(0.01)
        gate.set()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        gate.set()
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    starts = sorted(call["start"] for call in client.calls)
    assert len(starts) == 2
    assert starts[1] - starts[0] >= 60.0 / 55 - 1e-9


class BlockingSamClient:
    """Fake client that blocks inside the call until released."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()

    def segment_still(self, image_path, phrase, *, include_confidence=True):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.entered.set()
        assert self.release.wait(timeout=60)
        with self.lock:
            self.in_flight -= 1
        return SamStillResult(
            masks=(
                SamMask(
                    box_xyxy=(12, 30, 140, 150),
                    c=0.72,
                    mask_png=mask_png(0),
                ),
            )
        )


def test_sam_in_flight_capped_at_eight_with_slow_calls(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = BlockingSamClient()
    errors: list = []

    def run() -> None:
        try:
            segment(ctx, still, "fan", client=client)
        except Exception as exc:  # noqa: BLE001 — re-raised on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(9)]
    for thread in threads:
        thread.start()
    try:
        assert client.entered.wait(timeout=60)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if client.max_in_flight > 8:
                break
            time.sleep(0.01)
        assert client.max_in_flight <= 8
    finally:
        client.release.set()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)


@pytest.mark.parametrize("status", ["error", "timeout"])
def test_sam_provider_error_status_is_not_checked(tmp_path, clock, sink, status):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    failed = SamStillResult(
        masks=(
            SamMask(
                box_xyxy=(12, 30, 140, 150),
                c=0.72,
                mask_png=mask_png(1),
            ),
        ),
        status=status,
    )
    client = RecordingSamClient(clock, [failed])
    assert segment(ctx, still, "fan", client=client) == ()
    assert ctx.reservations[0].settles == [(SAM_COST_PER_CALL_USD, "failed")]
    (_, record), = sink
    assert record.status == status
    assert record.estimated_cost_usd == SAM_COST_PER_CALL_USD
    assert record.evidence == {"phrase": "fan", "c": None, "request_index": 1}
    assert list(tmp_path.glob("*.mask.png")) == []


@pytest.mark.parametrize(
    ("box", "c", "png"),
    [
        ((-1, 0, 2, 2), 0.72, None),  # negative origin
        ((0, 0, 0, 2), 0.72, None),  # zero width
        ((0, 0, 2, 0), 0.72, None),  # zero height
        ((12, 30, 140, 150), True, None),  # boolean confidence
        ((12, 30, 140, 150), "0.72", None),  # non-numeric confidence
        ((12, 30, 140, 150), 0.72, PNG_MAGIC),  # signature-only bytes
        ((12, 30, 140, 150), 0.72, PNG_MAGIC + b"not-a-png"),
    ],
)
def test_sam_malformed_masks_are_rejected(tmp_path, clock, sink, box, c, png):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    bad = SamStillResult(
        masks=(
            SamMask(
                box_xyxy=box,
                c=c,
                mask_png=mask_png(2) if png is None else png,
            ),
        )
    )
    client = RecordingSamClient(clock, [bad])
    with pytest.raises(ValueError):
        segment(ctx, still, "fan", client=client)
    assert ctx.reservations[0].settles == [(Decimal("0"), "failed")]
    (_, record), = sink
    assert record.status == "error"


def test_choose_detection_clear_winner_stands(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("clear_winner")])
    pair = segment(ctx, still, "fan", client=client)
    winner = choose_detection(pair, "lower right")
    assert winner is not None
    assert winner.score == pytest.approx(0.84)
    assert winner.box_xyxy == (10, 10, 50, 50)


def test_choose_detection_hint_breaks_tie(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    pair = segment(ctx, still, "fan", client=client)
    lower_right = choose_detection(pair, "lower right")
    assert lower_right is not None
    assert lower_right.box_xyxy == (200, 180, 260, 240)
    upper_left = choose_detection(pair, "upper left")
    assert upper_left is not None
    assert upper_left.box_xyxy == (10, 10, 50, 50)
    assert choose_detection(pair, "left right") is None


def test_mask_files_land_next_to_still_as_png(tmp_path, clock):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    pair = segment(ctx, still, "fan", client=client)
    assert [d.mask_path for d in pair] == [
        tmp_path / "still_fan_0.mask.png",
        tmp_path / "still_fan_1.mask.png",
    ]
    for detection in pair:
        assert detection.mask_path.read_bytes().startswith(PNG_MAGIC)


def test_sam_mask_write_failure_cleans_staged_and_emits_record(
    tmp_path, clock, sink, monkeypatch
):
    still = tmp_path / "still.png"
    still.touch()
    prior0 = tmp_path / "still_fan_0.mask.png"
    prior1 = tmp_path / "still_fan_1.mask.png"
    prior0.write_bytes(b"prior-0")
    prior1.write_bytes(b"prior-1")
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    real_write_bytes = Path.write_bytes
    calls = 0

    def flaky_write_bytes(self, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky_write_bytes)
    with pytest.raises(OSError, match="disk full"):
        segment(ctx, still, "fan", client=client)
    assert calls == 2
    assert ctx.reservations[0].settles == [(SAM_COST_PER_CALL_USD, "succeeded")]
    (_, record), = sink
    assert record.status == "ok"
    assert record.estimated_cost_usd == SAM_COST_PER_CALL_USD
    assert record.evidence == {"phrase": "fan", "c": 0.75, "request_index": 1}
    assert prior0.read_bytes() == b"prior-0"
    assert prior1.read_bytes() == b"prior-1"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["still.png", prior0.name, prior1.name]
    )


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_sam_staged_masks_cleaned_on_signal(
    tmp_path, clock, sink, signal, monkeypatch
):
    still = tmp_path / "still.png"
    still.touch()
    prior0 = tmp_path / "still_fan_0.mask.png"
    prior1 = tmp_path / "still_fan_1.mask.png"
    prior0.write_bytes(b"prior-0")
    prior1.write_bytes(b"prior-1")
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    real_write_bytes = Path.write_bytes
    calls = 0

    def signal_write_bytes(self, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise signal()
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", signal_write_bytes)
    with pytest.raises(signal):
        segment(ctx, still, "fan", client=client)
    assert calls == 2
    assert prior0.read_bytes() == b"prior-0"
    assert prior1.read_bytes() == b"prior-1"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["still.png", prior0.name, prior1.name]
    )


def test_sam_staged_mask_cleaned_on_close_error(tmp_path, clock, monkeypatch):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("fan_single_above")])
    real_mkstemp = tempfile.mkstemp
    real_close = os.close
    staged_fds: list[int] = []

    def spy_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        staged_fds.append(fd)
        return fd, name

    def flaky_close(fd):
        if fd in staged_fds:
            real_close(fd)
            raise OSError("injected close failure")
        return real_close(fd)

    monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)
    monkeypatch.setattr(os, "close", flaky_close)
    with pytest.raises(OSError, match="injected close failure"):
        segment(ctx, still, "fan", client=client)
    assert staged_fds != []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["still.png"]


def test_sam_failed_publication_restores_prior_masks(
    tmp_path, clock, sink, monkeypatch
):
    still = tmp_path / "still.png"
    still.touch()
    prior0 = tmp_path / "still_fan_0.mask.png"
    prior1 = tmp_path / "still_fan_1.mask.png"
    prior0.write_bytes(b"prior-0")
    prior1.write_bytes(b"prior-1")
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    real_replace = os.replace
    calls = 0

    def flaky_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk fault")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk fault"):
        segment(ctx, still, "fan", client=client)
    assert ctx.reservations[0].settles == [(SAM_COST_PER_CALL_USD, "succeeded")]
    (_, record), = sink
    assert record.status == "ok"
    assert prior0.read_bytes() == b"prior-0"
    assert prior1.read_bytes() == b"prior-1"
    assert sorted(
        p.name for p in tmp_path.iterdir() if p.name != ".publish.lock"
    ) == sorted(["still.png", prior0.name, prior1.name])


def test_sam_failed_publication_removes_new_masks_without_priors(
    tmp_path, clock, monkeypatch
):
    still = tmp_path / "still.png"
    still.touch()
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    real_replace = os.replace
    calls = 0

    def flaky_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk fault")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk fault"):
        segment(ctx, still, "fan", client=client)
    assert sorted(
        p.name for p in tmp_path.iterdir() if p.name != ".publish.lock"
    ) == ["still.png"]


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_sam_interrupt_after_rename_restores_prior_masks(
    tmp_path, clock, signal, monkeypatch
):
    still = tmp_path / "still.png"
    still.touch()
    prior0 = tmp_path / "still_fan_0.mask.png"
    prior1 = tmp_path / "still_fan_1.mask.png"
    prior0.write_bytes(b"prior-0")
    prior1.write_bytes(b"prior-1")
    ctx = RecordingJobContext()
    client = RecordingSamClient(clock, [case_result("ambiguous_pair")])
    real_replace = os.replace
    calls = 0

    def rename_then_interrupt(src, dst):
        nonlocal calls
        calls += 1
        result = real_replace(src, dst)
        if calls == 1:
            raise signal()
        return result

    monkeypatch.setattr(os, "replace", rename_then_interrupt)
    with pytest.raises(signal):
        segment(ctx, still, "fan", client=client)
    assert calls == 2
    assert prior0.read_bytes() == b"prior-0"
    assert prior1.read_bytes() == b"prior-1"
    assert sorted(
        p.name for p in tmp_path.iterdir() if p.name != ".publish.lock"
    ) == sorted(["still.png", prior0.name, prior1.name])


def test_publish_staged_failed_call_keeps_completed_winner(
    tmp_path, monkeypatch
):
    dest_a = tmp_path / "first.mask.png"
    dest_b = tmp_path / "second.mask.png"
    dest_a.write_bytes(b"prior-0")
    dest_b.write_bytes(b"prior-1")

    def staged_tmp(content: bytes) -> Path:
        fd, name = tempfile.mkstemp(dir=tmp_path, suffix=".tmp")
        os.close(fd)
        path = Path(name)
        path.write_bytes(content)
        return path

    tmp_first = staged_tmp(b"first")
    tmp_second = staged_tmp(b"second")
    tmp_third = staged_tmp(b"third")
    real_replace = os.replace
    a_first_done = threading.Event()
    ops: list[str] = []
    ops_lock = threading.Lock()

    def spy_replace(src, dst):
        src_path = Path(src)
        if src_path == tmp_second:
            raise OSError("disk fault")
        result = real_replace(src, dst)
        with ops_lock:
            if src_path == tmp_first:
                ops.append("A-publish")
            elif src_path == tmp_third:
                ops.append("B-publish")
            else:
                ops.append("restore")
        if src_path == tmp_first:
            a_first_done.set()
        return result

    monkeypatch.setattr(os, "replace", spy_replace)
    outcomes: dict[str, BaseException | None] = {}

    def run(name: str, pairs: list[tuple[Path, Path]]) -> None:
        try:
            publish_staged(pairs)
        except BaseException as exc:
            outcomes[name] = exc
        else:
            outcomes[name] = None

    thread_a = threading.Thread(
        target=run, args=("A", [(tmp_first, dest_a), (tmp_second, dest_b)])
    )
    thread_a.start()
    assert a_first_done.wait(timeout=30)
    thread_b = threading.Thread(target=run, args=("B", [(tmp_third, dest_a)]))
    thread_b.start()
    thread_a.join(timeout=30)
    thread_b.join(timeout=30)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert isinstance(outcomes["A"], OSError)
    assert outcomes["B"] is None
    assert ops == ["A-publish", "restore", "B-publish"]
    assert dest_a.read_bytes() == b"third"
    assert dest_b.read_bytes() == b"prior-1"
    assert sorted(
        p.name for p in tmp_path.iterdir() if p.name != ".publish.lock"
    ) == sorted([dest_a.name, dest_b.name])


def test_publish_staged_three_workers_never_publish_concurrently(
    tmp_path, monkeypatch
):
    """A releases while B waits; C arrives after; B and C never overlap.

    Regression: the releasing worker unlinked ``.publish.lock``, so C
    created a fresh lock file and published concurrently with B, which
    still held the old inode. The lock file is now never unlinked, so B
    and C serialize on the same file.
    """
    dest = tmp_path / "d.mask.png"

    def staged_tmp(content: bytes) -> Path:
        fd, name = tempfile.mkstemp(dir=tmp_path, suffix=".tmp")
        os.close(fd)
        path = Path(name)
        path.write_bytes(content)
        return path

    tmpA = staged_tmp(b"A")
    tmpB = staged_tmp(b"B")
    tmpC = staged_tmp(b"C")
    real_replace = os.replace
    real_flock = fcntl.flock
    state_lock = threading.Lock()
    in_publish = 0
    max_in_publish = 0
    lock_inodes: list[int] = []
    a_inside = threading.Event()
    b_in_flock = threading.Event()
    b_inside = threading.Event()
    c_in_flock = threading.Event()
    let_a_finish = threading.Event()
    let_b_finish = threading.Event()

    def spy_replace(src, dst):
        gated = Path(src) in (tmpA, tmpB)
        if gated:
            gate = let_a_finish if Path(src) == tmpA else let_b_finish
            (a_inside if Path(src) == tmpA else b_inside).set()
            assert gate.wait(timeout=30)
        nonlocal in_publish, max_in_publish
        with state_lock:
            in_publish += 1
            max_in_publish = max(max_in_publish, in_publish)
        try:
            return real_replace(src, dst)
        finally:
            with state_lock:
                in_publish -= 1

    def spy_flock(fd, op):
        if op == fcntl.LOCK_EX:
            with state_lock:
                lock_inodes.append(os.fstat(fd).st_ino)
                n = len(lock_inodes)
            if n == 2:
                b_in_flock.set()
            elif n == 3:
                c_in_flock.set()
        return real_flock(fd, op)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(fcntl, "flock", spy_flock)
    outcomes: dict[str, BaseException | None] = {}

    def run(name: str, pair: tuple[Path, Path]) -> None:
        try:
            publish_staged([pair])
        except BaseException as exc:
            outcomes[name] = exc
        else:
            outcomes[name] = None

    thread_a = threading.Thread(target=run, args=("A", (tmpA, dest)))
    thread_a.start()
    assert a_inside.wait(timeout=30)
    thread_b = threading.Thread(target=run, args=("B", (tmpB, dest)))
    thread_b.start()
    assert b_in_flock.wait(timeout=30)  # B waits while A holds the lock
    let_a_finish.set()  # A releases; B already holds its open file handle
    thread_a.join(timeout=30)
    assert not thread_a.is_alive()
    assert b_inside.wait(timeout=30)  # B now holds the lock
    thread_c = threading.Thread(target=run, args=("C", (tmpC, dest)))
    thread_c.start()
    assert c_in_flock.wait(timeout=30)  # C arrives after A released
    let_b_finish.set()
    thread_b.join(timeout=30)
    thread_c.join(timeout=30)
    assert not thread_b.is_alive()
    assert not thread_c.is_alive()
    assert outcomes == {"A": None, "B": None, "C": None}
    assert max_in_publish == 1
    assert lock_inodes[1] == lock_inodes[2]  # same file, never unlinked
    assert dest.read_bytes() in (b"B", b"C")
    assert (tmp_path / ".publish.lock").exists()


def test_feature_fragment_states_pacing_as_upper_bound():
    # The generated feature document ships in every copy of the repository; its source notes may not.
    features = (Path(__file__).resolve().parent.parent / "docs" / "FEATURES.md").read_text(encoding="utf-8")
    section = features.split("\n## SAM stills and frame scoring\n", 1)[1].split("\n## ", 1)[0]
    assert "at most 55 starts/minute" in section
    assert features.count("55 starts/minute") == 1


def test_no_crop_or_homography_path():
    root = Path(__file__).resolve().parent.parent / "app"
    sources = (root / "sam.py").read_text(
        encoding="utf-8"
    ) + (root / "frame_scoring.py").read_text(encoding="utf-8")
    for token in ("crop", "homography", "warp", "upscale", "affine",
                  "perspective"):
        assert token not in sources.lower(), token
