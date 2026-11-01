"""Real-frame label renderer, part of the guide module.

Draws the part outline along the SAM mask edge, then one code-drawn
numbered badge per frame, then pastes the original photographed part
pixels back through the SAM mask. The part is never redrawn: the
renderer invariant is that changed pixels inside the source part mask
are zero (a renderer property only, never proof a repair is
correct). The outline sits outside the mask, so the restore line never
touches it; a mask too small to trace falls back to the box edge.

PNG is read and written with the standard library only (``struct`` +
``zlib``), so ``render_label`` needs no new dependency. Masks are
same-size PNG files where any nonzero channel marks a part pixel.
8-bit gray and RGB are accepted; RGBA is rejected at the door because
dropping alpha would silently change part pixels. Decode streams with a
budget derived from the declared size right after IHDR, so a corrupt
file fails fast instead of ballooning memory.
"""

from __future__ import annotations

import contextlib
import errno
import math
import os
import struct
import tempfile
import zlib
from pathlib import Path

from app.contracts import Detection, JobContext, ProvisionalRender
from app.outline_config import cached as outline_cached

PROVISIONAL_KIND = "provisional_render"

#: Default badge number. A render carries one badge, and the pipeline passes the
#: part's own position in its step (1 to MAX_PARTS_PER_STEP) so the thumbnail's
#: number matches the row and the pin on the main image.
LABEL_NUMBER = 1

_BADGE_FILL = (242, 177, 52)
_BADGE_RING = (255, 255, 255)
_BADGE_GLYPH = (26, 26, 26)
_RING_WIDTH_PX = 2

# 3x5 bitmap for the badge digit "1".
_GLYPH_1 = (
    "010",
    "110",
    "010",
    "010",
    "111",
)
_GLYPHS = (_GLYPH_1, ("111", "001", "111", "100", "111"),
           ("111", "001", "111", "001", "111"))

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Decode guards: dimension caps plus slack over the header-declared pixel
#: data for compressed bytes and ancillary chunks. The total-bytes budget
#: is fixed immediately after IHDR, before any IDAT payload is read.
_MAX_PNG_SIDE_PX = 8192
_MAX_PNG_PIXELS = 64_000_000
_PNG_READ_SLACK_BYTES = 2_097_152
_CHUNK_BLOCK_BYTES = 65_536

#: Badge search evaluates at most this many grid centers over the frame.
_MAX_PLACEMENT_CENTERS = 512

__all__ = (
    "PROVISIONAL_KIND",
    "LABEL_NUMBER",
    "render_label",
    "draw_label_pixels",
    "draw_outline",
    "draw_badge",
    "restore_part_pixels",
    "changed_part_pixel_count",
    "badge_radius",
    "choose_badge_center",
    "read_png_rgb",
    "write_png_rgb",
    "read_mask",
)


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def png_rgb_bytes(width: int, height: int, pixels: bytes) -> bytes:
    """Encode row-major RGB bytes as an 8-bit PNG."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad PNG size: {width}x{height}")
    if len(pixels) != width * height * 3:
        raise ValueError("pixel buffer does not match size")
    stride = width * 3
    raw = bytearray((stride + 1) * height)
    for y in range(height):
        base = y * (stride + 1)
        raw[base] = 0  # filter type 0: none
        raw[base + 1 : base + 1 + stride] = pixels[y * stride : (y + 1) * stride]
    out = bytearray(_PNG_MAGIC)
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    out += _chunk(b"IDAT", zlib.compress(bytes(raw)))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def write_png_rgb(path: Path, width: int, height: int, pixels: bytes) -> None:
    path.write_bytes(png_rgb_bytes(width, height, pixels))


def _unfilter(width: int, height: int, bpp: int, raw: bytes) -> bytearray:
    stride = width * bpp
    if len(raw) != (stride + 1) * height:
        raise ValueError("PNG data size does not match header")
    out = bytearray(height * stride)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        kind = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        if kind == 0:
            pass
        elif kind == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                up = prev[i]
                upper_left = prev[i - bpp] if i >= bpp else 0
                base = left + up - upper_left
                dist = (abs(base - left), abs(base - up), abs(base - upper_left))
                pick = (left, up, upper_left)[dist.index(min(dist))]
                line[i] = (line[i] + pick) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter: {kind}")
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return out


def _read_exact(stream, size: int, path: Path, what: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"truncated PNG {what}: {path}")
    return data


def read_png_rgb(path: Path) -> tuple[int, int, bytes]:
    """Read an 8-bit gray/RGB PNG as row-major RGB bytes.

    RGBA is rejected: silently dropping alpha would change part pixels.
    Chunks stream under a budget fixed from the declared size right after
    IHDR; every chunk CRC is validated, IHDR must be first and unique with
    zero compression/filter methods, IDATs must run consecutively up to
    IEND, and the compressed stream must decode to exactly the declared
    size with no trailing data after the zlib stream. Trailing bytes after
    IEND are ignored.
    """
    path = Path(path)
    with open(path, "rb") as stream:
        if stream.read(8) != _PNG_MAGIC:
            raise ValueError(f"not a PNG file: {path}")
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated PNG header: {path}")
        (first_length,) = struct.unpack(">I", header[:4])
        if header[4:] != b"IHDR" or first_length != 13:
            raise ValueError(f"PNG must start with a 13-byte IHDR: {path}")
        ihdr = _read_exact(stream, 13, path, "IHDR")
        crc = _read_exact(stream, 4, path, "IHDR CRC")
        if crc != struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF):
            raise ValueError(f"bad IHDR CRC: {path}")
        width, height, depth, color_type, comp_method, filter_method, interlace = (
            struct.unpack(">IIBBBBB", ihdr)
        )
        if depth != 8 or color_type not in (0, 2):
            if depth == 8 and color_type == 6:
                raise ValueError(f"RGBA PNG not supported (alpha must survive): {path}")
            raise ValueError(f"need 8-bit gray/RGB PNG: {path}")
        if comp_method != 0:
            raise ValueError(f"unsupported PNG compression method: {path}")
        if filter_method != 0:
            raise ValueError(f"unsupported PNG filter method: {path}")
        if interlace != 0:
            raise ValueError(f"interlaced PNG not supported: {path}")
        if width <= 0 or height <= 0:
            raise ValueError(f"bad PNG size: {path}")
        if (
            width > _MAX_PNG_SIDE_PX
            or height > _MAX_PNG_SIDE_PX
            or width * height > _MAX_PNG_PIXELS
        ):
            raise ValueError(f"PNG too large: {path}")
        channels = 1 if color_type == 0 else 3
        expected = (width * channels + 1) * height
        remaining = expected + _PNG_READ_SLACK_BYTES - 37  # magic+IHDR already read
        compressed = bytearray()
        seen_idat = False
        while True:
            header = stream.read(8)
            if not header:
                raise ValueError(f"PNG missing IEND: {path}")
            if len(header) != 8:
                raise ValueError(f"truncated PNG chunk header: {path}")
            remaining -= 8
            (length,) = struct.unpack(">I", header[:4])
            tag = header[4:]
            if length > remaining:
                raise ValueError(f"PNG chunk exceeds declared-size budget: {path}")
            if tag == b"IHDR":
                raise ValueError(f"duplicate IHDR: {path}")
            if tag == b"IDAT":
                seen_idat = True
            elif tag == b"IEND":
                if length != 0:
                    raise ValueError(f"bad IEND chunk: {path}")
                if not seen_idat:
                    raise ValueError(f"PNG has no image data: {path}")
            elif seen_idat:
                raise ValueError(f"chunk {tag!r} between IDAT and IEND: {path}")
            running = zlib.crc32(tag)
            left = length
            while left:
                block = stream.read(min(left, _CHUNK_BLOCK_BYTES))
                if not block:
                    raise ValueError(f"truncated PNG chunk {tag!r}: {path}")
                running = zlib.crc32(block, running)
                if tag == b"IDAT":
                    compressed += block
                left -= len(block)
            remaining -= length + 4
            if remaining < 0:
                raise ValueError(f"PNG exceeds declared-size budget: {path}")
            stored = _read_exact(stream, 4, path, f"{tag!r} CRC")
            if stored != struct.pack(">I", running & 0xFFFFFFFF):
                raise ValueError(f"bad {tag!r} CRC: {path}")
            if tag == b"IEND":
                break
    if not compressed:
        raise ValueError(f"PNG has no image data: {path}")
    worker = zlib.decompressobj()
    try:
        flat_raw = worker.decompress(bytes(compressed), expected + 1)
    except zlib.error as exc:
        raise ValueError(f"PNG data is corrupt: {path}") from exc
    if worker.unconsumed_tail or len(flat_raw) > expected:
        raise ValueError(f"PNG data exceeds header size: {path}")
    flat_raw += worker.flush()
    if worker.unused_data:
        raise ValueError(f"PNG has trailing data after zlib stream: {path}")
    if len(flat_raw) != expected or not worker.eof:
        raise ValueError(f"PNG data does not match header size: {path}")
    flat = _unfilter(width, height, channels, flat_raw)
    if channels == 3:
        return width, height, bytes(flat)
    rgb = bytearray(width * height * 3)
    for i in range(width * height):
        rgb[3 * i : 3 * i + 3] = bytes((flat[i],) * 3)
    return width, height, bytes(rgb)


def read_mask(path: Path, width: int, height: int) -> bytes:
    """Read a mask PNG as one byte per pixel: 1 where any channel is nonzero."""
    mask_w, mask_h, pixels = read_png_rgb(path)
    if (mask_w, mask_h) != (width, height):
        raise ValueError(f"mask size {mask_w}x{mask_h} != frame {width}x{height}")
    mask = bytearray(width * height)
    for i in range(width * height):
        r, g, b = pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]
        mask[i] = 1 if (r or g or b) else 0
    return bytes(mask)


def badge_radius(width: int, height: int) -> int:
    return max(10, min(width, height) // 30)


def badge_extent(radius: int) -> int:
    """Distance from a badge center through its contrast ring."""
    return radius + _RING_WIDTH_PX


def _disc_clear(
    width: int,
    height: int,
    mask: bytes,
    center: tuple[int, int],
    radius: int,
) -> bool:
    cx, cy = center
    outer = radius + _RING_WIDTH_PX
    if cx - outer < 0 or cy - outer < 0 or cx + outer >= width or cy + outer >= height:
        return False
    limit = outer * outer
    for dy in range(-outer, outer + 1):
        for dx in range(-outer, outer + 1):
            if dx * dx + dy * dy <= limit and mask[(cy + dy) * width + cx + dx]:
                return False
    return True


def choose_badge_center(
    width: int,
    height: int,
    box_xyxy: tuple[int, int, int, int],
    mask: bytes,
    *,
    fallback: bool = True,
) -> tuple[int, int] | None:
    """Pick a badge center fully outside the mask where possible.

    Tries the eight near-part spots first, then a bounded grid over the
    whole frame. With ``fallback=False``, return None when none fits;
    otherwise use the box center for the single-label renderer.
    """
    x0, y0, x1, y1 = box_xyxy
    radius = badge_radius(width, height)
    pad = radius + 3
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    candidates = (
        (cx, y0 - pad),
        (x0 - pad, cy),
        (x1 + pad, cy),
        (cx, y1 + pad),
        (pad, pad),
        (width - 1 - pad, pad),
        (pad, height - 1 - pad),
        (width - 1 - pad, height - 1 - pad),
    )
    for candidate in candidates:
        if _disc_clear(width, height, mask, candidate, radius):
            return candidate
    stride = max(1, math.isqrt(width * height // _MAX_PLACEMENT_CENTERS))
    for gy in range(stride // 2, height, stride):
        for gx in range(stride // 2, width, stride):
            if _disc_clear(width, height, mask, (gx, gy), radius):
                return (gx, gy)
    return (min(max(cx, 0), width - 1), min(max(cy, 0), height - 1)) if fallback else None


def draw_badge(
    pixels: bytearray,
    width: int,
    height: int,
    center: tuple[int, int],
    radius: int,
    number: int = LABEL_NUMBER,
) -> None:
    """Paint the filled badge disc, its contrast ring, and the digit."""
    cx, cy = center
    outer = radius + _RING_WIDTH_PX
    for dy in range(-outer, outer + 1):
        y = cy + dy
        if y < 0 or y >= height:
            continue
        for dx in range(-outer, outer + 1):
            x = cx + dx
            if x < 0 or x >= width:
                continue
            dist2 = dx * dx + dy * dy
            if dist2 > outer * outer:
                continue
            color = _BADGE_FILL if dist2 <= radius * radius else _BADGE_RING
            pixels[(y * width + x) * 3 : (y * width + x) * 3 + 3] = bytes(color)
    scale = max(1, radius // 7)
    glyph_w, glyph_h = 3 * scale, 5 * scale
    gx, gy = cx - glyph_w // 2, cy - glyph_h // 2
    glyph = _GLYPHS[number - 1]
    for row, line in enumerate(glyph):
        for col, bit in enumerate(line):
            if bit != "1":
                continue
            for sy in range(scale):
                y = gy + row * scale + sy
                if y < 0 or y >= height:
                    continue
                for sx in range(scale):
                    x = gx + col * scale + sx
                    if 0 <= x < width:
                        pixels[(y * width + x) * 3 : (y * width + x) * 3 + 3] = bytes(
                            _BADGE_GLYPH
                        )


def _is_edge_pixel(mask: bytes, width: int, height: int, x: int, y: int) -> bool:
    """True when a mask pixel touches a non-mask pixel (8-neighbourhood)."""
    for dy in (-1, 0, 1):
        ny = y + dy
        if ny < 0 or ny >= height:
            continue
        for dx in (-1, 0, 1):
            nx = x + dx
            if 0 <= nx < width and not mask[ny * width + nx]:
                return True
    return False


def draw_outline(
    pixels: bytearray,
    width: int,
    height: int,
    box_xyxy: tuple[int, int, int, int],
    mask: bytes,
) -> None:
    """Paint the part outline: a halo outside the mask edge, or the box edge.

    Colour, stroke width and the smallest traceable mask come from
    config/outline.toml. Every outline pixel sits outside the part
    pixels, so the restore line never erases it and the renderer
    invariant holds. A mask with fewer part pixels than the configured
    minimum (a blank mask included) falls back to the box edge, clamped
    to the frame; a box with nothing visible draws nothing.
    """
    if len(pixels) != width * height * 3 or len(mask) != width * height:
        raise ValueError("pixel or mask buffer does not match size")
    cfg = outline_cached()["outline"]
    color = bytes(cfg["color"])
    stroke = cfg["width_px"]
    if sum(mask) < cfg["min_mask_px"]:
        x0 = max(0, box_xyxy[0])
        y0 = max(0, box_xyxy[1])
        x1 = min(width, box_xyxy[2])
        y1 = min(height, box_xyxy[3])
        if x0 >= x1 or y0 >= y1:
            return
        fill = bytearray(width * height)
        for y in range(y0, y1):
            for x in range(x0, x1):
                fill[y * width + x] = 1
        mask = bytes(fill)
    edge = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if mask[y * width + x] and _is_edge_pixel(mask, width, height, x, y)
    ]
    for cx, cy in edge:
        for y in range(max(0, cy - stroke), min(height, cy + stroke + 1)):
            for x in range(max(0, cx - stroke), min(width, cx + stroke + 1)):
                if not mask[y * width + x]:
                    pixels[(y * width + x) * 3 : (y * width + x) * 3 + 3] = color


def draw_label_pixels(
    width: int,
    height: int,
    original: bytes,
    box_xyxy: tuple[int, int, int, int],
    mask: bytes,
    number: int = LABEL_NUMBER,
) -> bytearray:
    """Draw the outline and badge without restoring part pixels (pre-restore image)."""
    if len(original) != width * height * 3 or len(mask) != width * height:
        raise ValueError("pixel or mask buffer does not match size")
    drawn = bytearray(original)
    draw_outline(drawn, width, height, box_xyxy, mask)
    center = choose_badge_center(width, height, box_xyxy, mask)
    draw_badge(drawn, width, height, center, badge_radius(width, height), number)
    return drawn


def restore_part_pixels(rendered: bytearray, original: bytes, mask: bytes) -> None:
    """Paste original part pixels back through the mask (the restore line)."""
    for i, marked in enumerate(mask):
        if marked:
            rendered[3 * i : 3 * i + 3] = original[3 * i : 3 * i + 3]


def changed_part_pixel_count(
    frame_path: Path, rendered_path: Path, mask_path: Path
) -> int:
    """Count rendered pixels that differ from the frame inside the mask.

    The renderer invariant: 0 for every ``render_label`` output. This says
    only that the renderer kept the photographed pixels, never that a
    repair step is correct.
    """
    width, height, original = read_png_rgb(frame_path)
    mask = read_mask(mask_path, width, height)
    rendered_w, rendered_h, rendered = read_png_rgb(rendered_path)
    if (rendered_w, rendered_h) != (width, height):
        raise ValueError("rendered size does not match frame")
    changed = 0
    for i, marked in enumerate(mask):
        if marked and rendered[3 * i : 3 * i + 3] != original[3 * i : 3 * i + 3]:
            changed += 1
    return changed


def _position_hint(
    width: int, height: int, box_xyxy: tuple[int, int, int, int]
) -> str:
    x0, y0, x1, y1 = box_xyxy
    cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    if (row, col) == ("middle", "center"):
        return "center"
    if col == "center":
        return row
    if row == "middle":
        return col
    return f"{row} {col}"


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return os.path.abspath(first) == os.path.abspath(second)


def _write_png_atomic(path: Path, width: int, height: int, pixels: bytes) -> None:
    """Publish the PNG under ``path``, failing if ``path`` already exists.

    The bytes are written to a hidden temp sibling. A hidden exclusive claim
    (``.<name>.claim``) serialises publishes to one name; under it, an existing
    ``path`` raises FileExistsError, otherwise ``os.replace`` publishes the PNG.
    A reader sees either no file or the complete PNG at ``path``; only hidden
    files (temp, claim) are ever partial, and reconciliation removes those
    after a crash. Once published, the file is never unlinked here.
    """
    tmp: Path | None = None
    claim = path.with_name(f".{path.name}.claim")
    claimed = published = False
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=".render-", suffix=".png"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        write_png_rgb(tmp, width, height, pixels)
        claim_fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        claimed = True
        os.close(claim_fd)
        if os.path.lexists(path):
            raise FileExistsError(errno.EEXIST, "output already exists", str(path))
        os.replace(tmp, path)
        published = True
    finally:
        if claimed:
            with contextlib.suppress(OSError):
                claim.unlink()
        if not published and tmp is not None:
            with contextlib.suppress(OSError):
                tmp.unlink()


def render_label(
    ctx: JobContext,
    frame_path: Path,
    detection: Detection,
    output_path: Path,
    number: int = LABEL_NUMBER,
) -> ProvisionalRender:
    """Draw the outline and one numbered badge, restore part pixels, register the artifact.

    The output must differ from the frame and mask paths. Registration
    runs before any filesystem write, so the store's containment check
    rejects a bad path before a byte lands; a later write failure
    tombstones the registered row instead of leaving it dangling.
    Output names are unique: publishing fails atomically with
    ``FileExistsError`` when the target already exists.
    """
    frame_path = Path(frame_path)
    output_path = Path(output_path)
    if _same_path(output_path, frame_path):
        raise ValueError(f"output_path must differ from frame_path: {output_path}")
    if _same_path(output_path, Path(detection.mask_path)):
        raise ValueError(f"output_path must differ from mask_path: {output_path}")
    width, height, original = read_png_rgb(frame_path)
    mask = read_mask(detection.mask_path, width, height)
    drawn = draw_label_pixels(width, height, original, detection.box_xyxy, mask, number)
    restore_part_pixels(drawn, original, mask)
    artifact_id = ctx.artifacts.register(PROVISIONAL_KIND, output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_png_atomic(output_path, width, height, bytes(drawn))
    except Exception:
        with contextlib.suppress(Exception):
            ctx.artifacts.mark_deleted(artifact_id)
        raise
    return ProvisionalRender(
        detection=detection,
        artifact_id=artifact_id,
        rendered_image_path=output_path,
        position_hint=_position_hint(width, height, detection.box_xyxy),
    )
