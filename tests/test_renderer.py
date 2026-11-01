"""Renderer tests: badge placement, part-pixel restore, registration."""

from __future__ import annotations

import errno
import os
import struct
import zlib
from pathlib import Path

import pytest

from app.contracts import ArtifactId, Detection, JobContext, JobId
from app.render import (
    PROVISIONAL_KIND,
    changed_part_pixel_count,
    choose_badge_center,
    draw_badge,
    draw_label_pixels,
    draw_outline,
    badge_radius,
    read_mask,
    read_png_rgb,
    render_label,
    restore_part_pixels,
    write_png_rgb,
)

FRAME_W, FRAME_H = 200, 150
FRAME_COLOR = (40, 60, 90)


class RecordingStore:
    def __init__(self) -> None:
        self.registered: list[tuple[str, Path, ArtifactId]] = []
        self.kept: list[ArtifactId] = []
        self.deleted: list[ArtifactId] = []
        self._next = 0

    def register(self, kind: str, path: Path) -> ArtifactId:
        self._next += 1
        artifact_id = ArtifactId(f"artifact-{self._next}")
        self.registered.append((kind, path, artifact_id))
        return artifact_id

    def mark_kept(self, artifact_id: ArtifactId) -> None:
        self.kept.append(artifact_id)

    def mark_deleted(self, artifact_id: ArtifactId) -> None:
        self.deleted.append(artifact_id)


class RecordingContext:
    def __init__(self) -> None:
        self._job_id = JobId("job-render")
        self._store = RecordingStore()
        self.finishes: list[str] = []
        self.finish_keeps: list[tuple] = []
        self.recorded_calls: list[tuple] = []

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def artifacts(self) -> RecordingStore:
        return self._store

    def reserve(self, kind: str, worst_case_usd):  # pragma: no cover
        raise AssertionError("the renderer reserves nothing")

    def record_call_with_event(self, key, call, event) -> None:
        self.recorded_calls.append((key, call, event))

    def finish(self, outcome, *, keep_artifact_ids=(), failed_output_names=None) -> None:
        self.finishes.append(outcome)
        self.finish_keeps.append(tuple(keep_artifact_ids))


assert isinstance(RecordingContext(), JobContext)


def _solid_frame(path: Path, width=FRAME_W, height=FRAME_H) -> Path:
    pixels = bytes(FRAME_COLOR) * (width * height)
    write_png_rgb(path, width, height, pixels)
    return path


def _mask(path: Path, rects, width=FRAME_W, height=FRAME_H) -> Path:
    buf = bytearray(b"\x00\x00\x00" * (width * height))
    for x0, y0, x1, y1 in rects:
        for y in range(y0, y1):
            for x in range(x0, x1):
                buf[(y * width + x) * 3 : (y * width + x) * 3 + 3] = b"\xff\xff\xff"
    write_png_rgb(path, width, height, bytes(buf))
    return path


def _detection(mask_path: Path, box, phrase="fan") -> Detection:
    return Detection(phrase=phrase, mask_path=mask_path, score=0.8, box_xyxy=box)


def _changed_in_memory(drawn: bytes, original: bytes, mask: bytes) -> int:
    return sum(
        1
        for i, marked in enumerate(mask)
        if marked and drawn[3 * i : 3 * i + 3] != original[3 * i : 3 * i + 3]
    )


def test_mask_pixels_are_byte_identical(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    mask_path = _mask(tmp_path / "mask.png", [(20, 30, 70, 80)])
    detection = _detection(mask_path, (20, 30, 70, 80))
    ctx = RecordingContext()

    provisional = render_label(ctx, frame, detection, tmp_path / "out.png")

    assert changed_part_pixel_count(frame, provisional.rendered_image_path, mask_path) == 0
    assert provisional.rendered_image_path.read_bytes() != frame.read_bytes()

    # Full-mask render through the real entry point: the badge necessarily
    # crosses the mask, so this assertion fails if the restore line is gone.
    full_path = _mask(tmp_path / "full.png", [(0, 0, FRAME_W, FRAME_H)])
    full_out = render_label(
        ctx, frame, _detection(full_path, (0, 0, FRAME_W, FRAME_H), "fan"),
        tmp_path / "full-out.png",
    )
    assert changed_part_pixel_count(frame, full_out.rendered_image_path, full_path) == 0

    # Mutation control: the same draw WITHOUT the restore line must change
    # mask pixels, so deleting the restore line fails this test.
    width, height, original = read_png_rgb(frame)
    full_mask = bytes((1,)) * (width * height)
    drawn = draw_label_pixels(width, height, original, detection.box_xyxy, full_mask)
    assert _changed_in_memory(bytes(drawn), original, full_mask) > 0


def test_label_stays_outside_when_space_exists(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    box = (20, 30, 70, 80)
    mask_path = _mask(tmp_path / "mask.png", [box])
    provisional = render_label(
        RecordingContext(), frame, _detection(mask_path, box), tmp_path / "out.png"
    )

    width, height, original = read_png_rgb(frame)
    _, _, rendered = read_png_rgb(provisional.rendered_image_path)
    changed = [
        (i % width, i // width)
        for i in range(width * height)
        if rendered[3 * i : 3 * i + 3] != original[3 * i : 3 * i + 3]
    ]
    assert changed, "a label must be drawn"
    x0, y0, x1, y1 = box
    assert all(not (x0 <= x < x1 and y0 <= y < y1) for x, y in changed)


def test_crossing_ring_restores_part(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    box = (90, 65, 110, 85)
    mask_path = _mask(tmp_path / "mask.png", [box])
    width, height, original = read_png_rgb(frame)
    mask = read_mask(mask_path, width, height)

    # Force the crossing case: badge centered on the part, then restore.
    drawn = bytearray(original)
    draw_badge(drawn, width, height, (100, 75), badge_radius(width, height))
    assert _changed_in_memory(bytes(drawn), original, mask) > 0
    restore_part_pixels(drawn, original, mask)
    assert _changed_in_memory(bytes(drawn), original, mask) == 0
    assert bytes(drawn) != original  # badge ring still visible off the part

    # Integration: a full-frame mask restores to a byte-identical image.
    full_mask_path = _mask(tmp_path / "full.png", [(0, 0, width, height)])
    ctx = RecordingContext()
    provisional = render_label(
        ctx, frame, _detection(full_mask_path, (0, 0, width, height)),
        tmp_path / "full-out.png",
    )
    _, _, rendered = read_png_rgb(provisional.rendered_image_path)
    assert rendered == original
    assert changed_part_pixel_count(frame, provisional.rendered_image_path, full_mask_path) == 0
    assert ctx.artifacts.registered[0][0] == PROVISIONAL_KIND


def test_badge_center_prefers_outside_and_falls_back_to_box(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    width, height, _ = read_png_rgb(frame)
    box = (20, 30, 70, 80)
    mask = read_mask(_mask(tmp_path / "m.png", [box]), width, height)
    cx, cy = choose_badge_center(width, height, box, mask)
    assert not (box[0] <= cx < box[2] and box[1] <= cy < box[3])

    full = read_mask(_mask(tmp_path / "f.png", [(0, 0, width, height)]), width, height)
    assert choose_badge_center(width, height, box, full) == (45, 55)


def test_mask_size_mismatch_and_non_png_raise(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    small = tmp_path / "small.png"
    write_png_rgb(small, 10, 10, bytes((255, 255, 255)) * 100)
    ctx = RecordingContext()
    try:
        render_label(ctx, frame, _detection(small, (0, 0, 10, 10)), tmp_path / "o.png")
    except ValueError:
        pass
    else:
        raise AssertionError("mask/frame size mismatch must raise")
    assert ctx.artifacts.registered == []

    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a png")
    try:
        render_label(ctx, junk, _detection(small, (0, 0, 10, 10)), tmp_path / "o2.png")
    except ValueError:
        pass
    else:
        raise AssertionError("non-PNG frame must raise")


def test_oversized_png_header_is_rejected_without_decoding(tmp_path):
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    huge = tmp_path / "huge.png"
    huge.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 9000, 9000, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" * 100))
        + chunk(b"IEND", b"")
    )
    try:
        read_png_rgb(huge)
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("oversized PNG must be rejected")


def _png_chunk(tag: bytes, data: bytes, bad_crc: bool = False) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    if bad_crc:
        crc ^= 0xFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _png_ihdr(
    width, height, color_type=2, depth=8, interlace=0, bad_crc=False, comp=0, filt=0
) -> bytes:
    return _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, depth, color_type, comp, filt, interlace),
        bad_crc,
    )


def _png_idat(raw: bytes) -> bytes:
    return _png_chunk(b"IDAT", zlib.compress(raw))


def _solid_raw(width, height, channels) -> bytes:
    pixel = bytes((10, 20, 30, 40)[:channels])
    return b"".join(b"\x00" + pixel * width for _ in range(height))


def _write(parts, path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(parts))
    return path


def test_output_path_equal_to_frame_is_refused(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    before = frame.read_bytes()
    mask_path = _mask(tmp_path / "mask.png", [(20, 30, 70, 80)])
    ctx = RecordingContext()
    with pytest.raises(ValueError, match="frame_path"):
        render_label(ctx, frame, _detection(mask_path, (20, 30, 70, 80)), frame)
    # Same file through a different spelling is the same refusal.
    sneaky = tmp_path / "sub" / ".." / "frame.png"
    with pytest.raises(ValueError, match="frame_path"):
        render_label(ctx, frame, _detection(mask_path, (20, 30, 70, 80)), sneaky)
    assert frame.read_bytes() == before
    assert ctx.artifacts.registered == []


def test_output_path_equal_to_mask_is_refused(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    mask_path = _mask(tmp_path / "mask.png", [(20, 30, 70, 80)])
    before = mask_path.read_bytes()
    ctx = RecordingContext()
    with pytest.raises(ValueError, match="mask_path"):
        render_label(ctx, frame, _detection(mask_path, (20, 30, 70, 80)), mask_path)
    assert mask_path.read_bytes() == before
    assert ctx.artifacts.registered == []


def test_rejecting_store_leaves_no_file(tmp_path):
    class RejectingStore(RecordingStore):
        def register(self, kind: str, path: Path) -> ArtifactId:
            raise ValueError("outside job directory")

    class RejectingContext(RecordingContext):
        def __init__(self) -> None:
            super().__init__()
            self._store = RejectingStore()

    frame = _solid_frame(tmp_path / "frame.png")
    mask_path = _mask(tmp_path / "mask.png", [(20, 30, 70, 80)])
    out = tmp_path / "out.png"
    with pytest.raises(ValueError, match="outside job directory"):
        render_label(
            RejectingContext(), frame, _detection(mask_path, (20, 30, 70, 80)), out
        )
    assert not out.exists()


def test_write_failure_tombstones_registered_row(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    frame = _solid_frame(tmp_path / "frame.png")
    mask_path = _mask(tmp_path / "mask.png", [(20, 30, 70, 80)])
    ctx = RecordingContext()
    with pytest.raises(OSError):
        render_label(
            ctx,
            frame,
            _detection(mask_path, (20, 30, 70, 80)),
            blocker / "out.png",
        )
    assert len(ctx.artifacts.registered) == 1
    assert ctx.artifacts.deleted == [ctx.artifacts.registered[0][2]]


def test_rgba_frame_is_rejected_at_the_door(tmp_path):
    frame = _write(
        (
            _png_ihdr(4, 3, color_type=6),
            _png_idat(_solid_raw(4, 3, 4)),
            _png_chunk(b"IEND", b""),
        ),
        tmp_path / "rgba.png",
    )
    gray_mask = tmp_path / "mask.png"
    write_png_rgb(gray_mask, 4, 3, bytes((255, 255, 255)) * 12)
    ctx = RecordingContext()
    out = tmp_path / "out.png"
    with pytest.raises(ValueError, match="RGBA"):
        render_label(ctx, frame, _detection(gray_mask, (0, 0, 4, 3)), out)
    assert ctx.artifacts.registered == []
    assert not out.exists()


def test_huge_declared_size_rejected_before_payload(tmp_path):
    # Declared 9000x9000 but the file ends right after the chunk header:
    # the size check must fire before the missing payload matters.
    truncated = tmp_path / "truncated-huge.png"
    truncated.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_ihdr(9000, 9000)
        + struct.pack(">I", 0xFFFFFF) + b"IDAT" + b"\x00" * 10
    )
    with pytest.raises(ValueError, match="too large"):
        read_png_rgb(truncated)

    # Tiny declared image with an enormous declared payload: rejected by
    # the declared-size budget, not by reading the payload.
    tiny = tmp_path / "tiny-huge.png"
    tiny.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_ihdr(2, 2)
        + struct.pack(">I", 50_000_000) + b"IDAT" + b"\x00" * 100
    )
    with pytest.raises(ValueError, match="budget"):
        read_png_rgb(tiny)


def test_corrupt_crc_is_rejected(tmp_path):
    bad = _write(
        (
            _png_ihdr(2, 2, bad_crc=True),
            _png_idat(_solid_raw(2, 2, 3)),
            _png_chunk(b"IEND", b""),
        ),
        tmp_path / "bad-crc.png",
    )
    with pytest.raises(ValueError, match="CRC"):
        read_png_rgb(bad)


def test_chunk_order_is_validated(tmp_path):
    valid_idat = _png_idat(_solid_raw(2, 2, 3))
    iend = _png_chunk(b"IEND", b"")

    idat_first = tmp_path / "idat-first.png"
    idat_first.write_bytes(
        b"\x89PNG\r\n\x1a\n" + valid_idat + _png_ihdr(2, 2) + iend
    )
    with pytest.raises(ValueError, match="IHDR"):
        read_png_rgb(idat_first)

    no_iend = _write((_png_ihdr(2, 2), valid_idat), tmp_path / "no-iend.png")
    with pytest.raises(ValueError, match="IEND"):
        read_png_rgb(no_iend)

    split_idat = _write(
        (
            _png_ihdr(2, 2),
            valid_idat,
            _png_chunk(b"tEXt", b"Comment\x00hi"),
            valid_idat,
            iend,
        ),
        tmp_path / "split.png",
    )
    with pytest.raises(ValueError, match="IDAT"):
        read_png_rgb(split_idat)

    interlaced = _write(
        (_png_ihdr(2, 2, interlace=1), valid_idat, iend),
        tmp_path / "interlaced.png",
    )
    with pytest.raises(ValueError, match="interlaced"):
        read_png_rgb(interlaced)

    corrupt = _write(
        (_png_ihdr(2, 2), _png_chunk(b"IDAT", b"not zlib data"), iend),
        tmp_path / "corrupt-idat.png",
    )
    with pytest.raises(ValueError, match="corrupt"):
        read_png_rgb(corrupt)

    two_streams = _write(
        (
            _png_ihdr(2, 2),
            _png_chunk(
                b"IDAT",
                zlib.compress(_solid_raw(2, 2, 3)) + zlib.compress(_solid_raw(2, 2, 3)),
            ),
            iend,
        ),
        tmp_path / "extra-stream.png",
    )
    with pytest.raises(ValueError, match="trailing"):
        read_png_rgb(two_streams)


def test_ihdr_compression_and_filter_methods_must_be_zero(tmp_path):
    valid_idat = _png_idat(_solid_raw(2, 2, 3))
    iend = _png_chunk(b"IEND", b"")

    bad_comp = _write(
        (_png_ihdr(2, 2, comp=1), valid_idat, iend),
        tmp_path / "bad-method-a.png",
    )
    with pytest.raises(ValueError, match="compression"):
        read_png_rgb(bad_comp)

    bad_filt = _write(
        (_png_ihdr(2, 2, filt=1), valid_idat, iend),
        tmp_path / "bad-method-b.png",
    )
    with pytest.raises(ValueError, match="filter"):
        read_png_rgb(bad_filt)


def test_double_render_to_same_output_fails_without_replacing(tmp_path, monkeypatch):
    frame = _solid_frame(tmp_path / "frame.png")
    box = (20, 30, 70, 80)
    mask_path = _mask(tmp_path / "mask.png", [box])
    ctx = RecordingContext()
    out = tmp_path / "out.png"

    def no_hard_links(*args, **kwargs):
        raise PermissionError(errno.EPERM, "hard links unavailable")

    monkeypatch.setattr(os, "link", no_hard_links)
    first = render_label(ctx, frame, _detection(mask_path, box), out)
    before = out.read_bytes()

    other_box = (100, 100, 150, 130)
    with pytest.raises(FileExistsError):
        render_label(ctx, frame, _detection(mask_path, other_box, "battery"), out)

    assert out.read_bytes() == before
    assert changed_part_pixel_count(frame, out, mask_path) == 0
    assert len(ctx.artifacts.registered) == 2
    assert ctx.artifacts.deleted == [ctx.artifacts.registered[1][2]]
    assert first.rendered_image_path == out


def test_failed_publish_removes_claim(tmp_path, monkeypatch):
    frame = _solid_frame(tmp_path / "frame.png")
    box = (20, 30, 70, 80)
    mask_path = _mask(tmp_path / "mask.png", [box])
    out = tmp_path / "out.png"
    ctx = RecordingContext()

    def fail_replace(src, dst):
        assert not dst.exists()
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        render_label(ctx, frame, _detection(mask_path, box), out)
    assert not out.exists()
    assert not list(tmp_path.glob(".render-*.png"))
    assert not list(tmp_path.glob(".*.claim"))
    assert ctx.artifacts.deleted == [ctx.artifacts.registered[0][2]]


def test_grid_search_finds_clear_spot_when_fixed_points_are_blocked(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    box = (20, 30, 70, 80)
    # Mask everything except one clear patch far from the part.
    buf = bytearray(b"\xff\xff\xff" * (FRAME_W * FRAME_H))
    for y in range(100, 140):
        for x in range(150, 190):
            buf[(y * FRAME_W + x) * 3 : (y * FRAME_W + x) * 3 + 3] = b"\x00\x00\x00"
    mask_path = tmp_path / "sparse.png"
    write_png_rgb(mask_path, FRAME_W, FRAME_H, bytes(buf))

    ctx = RecordingContext()
    provisional = render_label(
        ctx, frame, _detection(mask_path, box), tmp_path / "out.png"
    )
    assert changed_part_pixel_count(frame, provisional.rendered_image_path, mask_path) == 0

    width, height, original = read_png_rgb(frame)
    _, _, rendered = read_png_rgb(provisional.rendered_image_path)
    changed = [
        (i % width, i // width)
        for i in range(width * height)
        if rendered[3 * i : 3 * i + 3] != original[3 * i : 3 * i + 3]
    ]
    assert changed, "a label must be drawn"
    assert all(150 <= x < 190 and 100 <= y < 140 for x, y in changed)


def _publish_args():
    return 4, 3, bytes(range(36))


def test_public_path_absent_until_replace(tmp_path, monkeypatch):
    from app.render import _write_png_atomic

    out = tmp_path / "out.png"
    seen = []
    real = os.replace

    def checking(src, dst):
        seen.append(os.path.lexists(dst))
        real(src, dst)

    monkeypatch.setattr(os, "replace", checking)
    _write_png_atomic(out, *_publish_args())
    assert seen == [False]
    assert read_png_rgb(out)[0] == 4
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_signal_after_replace_keeps_published_png(tmp_path, monkeypatch):
    from app.render import _write_png_atomic

    out = tmp_path / "out.png"
    real = os.replace

    def replace_then_interrupt(src, dst):
        real(src, dst)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        _write_png_atomic(out, *_publish_args())
    assert read_png_rgb(out)[0] == 4
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_interrupt_after_claim_open_leaves_no_claim(tmp_path, monkeypatch):
    from app.render import _write_png_atomic

    out = tmp_path / "out.png"
    real_open, real_close = os.open, os.close
    claimed = []

    def spy_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if str(path).endswith(".claim"):
            claimed.append(fd)
        return fd

    def interrupt_close(fd):
        if fd in claimed:
            real_close(fd)
            raise KeyboardInterrupt
        real_close(fd)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "close", interrupt_close)
    with pytest.raises(KeyboardInterrupt):
        _write_png_atomic(out, 2, 2, bytes(12))
    monkeypatch.undo()
    assert not out.exists()
    assert not list(tmp_path.glob(".*.claim"))
    _write_png_atomic(out, 2, 2, bytes(12))
    assert out.is_file()


_AMBER = (232, 112, 50)  # config/outline.toml: the How-it-works amber


def _outline_count(pixels: bytes, width: int, height: int) -> int:
    return sum(
        1
        for i in range(width * height)
        if pixels[3 * i : 3 * i + 3] == bytes(_AMBER)
    )


def test_outline_follows_the_mask_contour_with_fixed_pixels():
    width, height = 40, 30
    original = bytes((10, 20, 30)) * (width * height)
    mask = bytearray(width * height)
    for y in range(10, 20):
        for x in range(10, 20):
            mask[y * width + x] = 1
    drawn = bytearray(original)
    draw_outline(drawn, width, height, (10, 10, 20, 20), bytes(mask))
    # A 4 px halo outside the 10x10 part: (10 + 8) ** 2 - 10 ** 2.
    assert _outline_count(bytes(drawn), width, height) == 224
    assert bytes(drawn[(15 * width + 9) * 3 : (15 * width + 9) * 3 + 3]) == bytes(_AMBER)
    assert bytes(drawn[(15 * width + 5) * 3 : (15 * width + 5) * 3 + 3]) == bytes((10, 20, 30))
    for i, marked in enumerate(mask):
        if marked:
            assert bytes(drawn[3 * i : 3 * i + 3]) == bytes((10, 20, 30))


def test_outline_falls_back_to_the_box_edge_with_fixed_pixels():
    width, height = 40, 30
    original = bytes((10, 20, 30)) * (width * height)
    drawn = bytearray(original)
    draw_outline(drawn, width, height, (10, 10, 20, 20), bytes(width * height))
    assert _outline_count(bytes(drawn), width, height) == 224
    assert bytes(drawn[(9 * width + 15) * 3 : (9 * width + 15) * 3 + 3]) == bytes(_AMBER)
    # A mask too small to trace takes the same box edge.
    speck = bytearray(width * height)
    speck[15 * width + 15] = 1
    drawn = bytearray(original)
    draw_outline(drawn, width, height, (10, 10, 20, 20), bytes(speck))
    assert _outline_count(bytes(drawn), width, height) == 224
    # A box with nothing visible draws nothing.
    drawn = bytearray(original)
    draw_outline(drawn, width, height, (50, 50, 60, 60), bytes(width * height))
    assert bytes(drawn) == original


def test_rendered_label_carries_the_outline_and_keeps_part_pixels(tmp_path):
    frame = _solid_frame(tmp_path / "frame.png")
    box = (20, 30, 70, 80)
    mask_path = _mask(tmp_path / "mask.png", [box])
    provisional = render_label(
        RecordingContext(), frame, _detection(mask_path, box), tmp_path / "out.png"
    )
    assert changed_part_pixel_count(frame, provisional.rendered_image_path, mask_path) == 0
    width, height, rendered = read_png_rgb(provisional.rendered_image_path)
    assert _outline_count(rendered, width, height) > 0
    # Below the part, clear of the badge above it: the halo hugs the edge.
    assert rendered[(83 * width + 45) * 3 : (83 * width + 45) * 3 + 3] == bytes(_AMBER)
    assert rendered[(84 * width + 45) * 3 : (84 * width + 45) * 3 + 3] != bytes(_AMBER)
    assert rendered[(60 * width + 45) * 3 : (60 * width + 45) * 3 + 3] != bytes(_AMBER)


def test_outline_follows_an_l_shaped_mask_not_its_bounding_box():
    width, height = 120, 120
    # L: vertical arm x 30..50, y 30..90; foot x 30..90, y 70..90.
    # The bbox (30,30,90,90) has an empty top-right corner.
    mask = bytearray(width * height)
    for y in range(30, 90):
        for x in range(30, 90):
            if x < 50 or y >= 70:
                mask[y * width + x] = 1
    pixels = bytearray(bytes(FRAME_COLOR) * (width * height))
    draw_outline(pixels, width, height, (30, 30, 90, 90), bytes(mask))

    def drawn(x, y):
        i = (y * width + x) * 3
        return bytes(pixels[i : i + 3]) != bytes(FRAME_COLOR)

    # Bounding-box edge pixels in the L's empty corner stay untouched.
    assert not any(drawn(x, 30) for x in range(60, 88))
    assert not any(drawn(88, y) for y in range(32, 58))
    # The halo hugs the real edge: just right of the arm and above the foot.
    assert drawn(51, 50) and drawn(70, 68)
