"""Display copies of guide stills: config, copies, page markup, originals."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import db, display_images, service
from app.main import create_app
from app.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
GOOD = ROOT / "config" / "display_images.toml"


def _redacted_still(path: Path, size=(1280, 720)) -> Path:
    """A still whose redaction (a flat dark block) must survive into every copy."""
    im = Image.new("RGB", size)
    px = im.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (x * 255 // size[0], y * 255 // size[1], 150)
    for y in range(300, 420):
        for x in range(500, 800):
            px[x, y] = (52, 56, 62)
    im.save(path, "PNG")
    return path


def _bad(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "c.toml"
    path.write_text(GOOD.read_text().replace(old, new, 1))
    return path


def test_config_loads_and_fails_loudly(tmp_path: Path) -> None:
    cfg = display_images.load(GOOD)
    assert cfg["stage_width"] in cfg["widths"] and cfg["poster_width"] in cfg["widths"]
    for old, new in (("quality = 78", "quality = 0"), ("stage_width = 800", "stage_width = 801"),
                     ("format = \"webp\"", "format = \"gif\""), ("cache_dir = \"display\"", "cache_dir = \"../x\""),
                     ("quality = 78", "qualty = 78"), ("widths = [480, 800, 1280]", "widths = [800, 480]")):
        with pytest.raises(display_images.DisplayImagesError):
            display_images.load(_bad(tmp_path, old, new))


def test_copies_never_pass_the_source_width() -> None:
    cfg = display_images.load(GOOD)
    d = display_images.describe(lambda w: f"/i?w={w}", (1280, 720), cfg)
    assert d.srcset == "/i?w=480 480w, /i?w=800 800w, /i?w=1280 1280w" and (d.width, d.height) == (800, 450)
    small = display_images.describe(lambda w: f"/i?w={w}", (300, 200), cfg)
    assert small.srcset == "/i?w=480 300w" and (small.width, small.height) == (300, 200)  # one entry, not upscaled
    mid = display_images.describe(lambda w: f"/i?w={w}", (1000, 500), cfg)
    assert mid.srcset == "/i?w=480 480w, /i?w=800 800w, /i?w=1280 1000w"


def test_copy_pixels_come_from_the_redacted_source_and_follow_a_rebuild(tmp_path: Path) -> None:
    src = _redacted_still(tmp_path / "step-06.png")
    body, media = display_images.display_bytes(src, 480)
    assert media == "image/webp" and isinstance(body, Path) and body.parent == tmp_path / "display"
    copy = Image.open(body).convert("RGB")
    assert copy.size == (480, 270)
    reference = Image.open(src).convert("RGB").resize((480, 270), Image.LANCZOS)
    # the redaction block is still a flat panel in the copy, and the copy tracks the redacted source
    assert all(abs(a - b) <= 8 for a, b in zip(copy.getpixel((240, 135)), (52, 56, 62)))
    diffs = [abs(a - b) for x in range(0, 480, 16) for y in range(0, 270, 16)
             for a, b in zip(copy.getpixel((x, y)), reference.getpixel((x, y)))]
    assert sum(diffs) / len(diffs) < 6
    again, _ = display_images.display_bytes(src, 480)
    assert again == body  # second ask is a cache hit
    # a re-redacted still (new bytes, same path): a new copy, and the old copy is gone
    other = tmp_path / "step-06-part-1.png"
    _redacted_still(other)
    display_images.display_bytes(other, 480)
    im = Image.open(src)
    px = im.load()
    for y in range(100, 200):
        for x in range(100, 300):
            px[x, y] = (0, 0, 0)
    im.save(src, "PNG")
    fresh, _ = display_images.display_bytes(src, 480)
    assert fresh != body and not body.exists()
    assert Image.open(fresh).convert("RGB").getpixel((60, 50))[0] < 8  # the new redaction shows
    assert len(list((tmp_path / "display").iterdir())) == 2  # the sibling still's copy was not touched


def test_unreadable_source_has_no_copy(tmp_path: Path) -> None:
    bad = tmp_path / "x.png"
    bad.write_bytes(b"not an image")
    assert display_images.display_bytes(bad, 480) is None


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    database = tmp_path / "db.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    settings = Settings(data_root=tmp_path / "data", db_path=database, production=False, pipeline_mode="replay")
    return TestClient(create_app(settings))


def test_sample_page_serves_responsive_stills_and_a_poster(client: TestClient) -> None:
    job_id = service.ensure_sample_job(client.app.state.settings)
    page = client.get(f"/guides/{job_id}").text
    pictures = re.findall(r"<picture>(<source [^>]*>)(<img [^>]*>)</picture>", page)
    assert len(pictures) >= 13
    for source, tag in pictures:
        assert 'type="image/webp"' in source and 'srcset="' in source and "?w=" in source and 'sizes="' in source
        assert re.search(r'width="\d+" height="\d+"', tag) and 'decoding="async"' in tag and 'loading="lazy"' in tag
        assert re.search(r'src="[^"]*/image/\d+"', tag) and "srcset" not in tag.replace("data-original-srcset", "").replace("data-labelled-srcset", "")  # PNG fallback
    thumbs = [t for t in re.findall(r"<img [^>]*>", page) if 'class="time-preview"' in t]
    assert thumbs and all("?w=480" in t for t in thumbs)  # the maker view's set-start thumbnail is the smallest copy
    poster = re.search(r'<video[^>]*poster="([^"]+)"', page).group(1)
    assert re.fullmatch(r"/guides/[^/]+/image/(\d+)\?w=800&poster=1", poster)  # a display copy of a redacted still
    got = client.get(poster)
    assert got.status_code == 200 and got.headers["content-type"] == "image/jpeg" and got.content[:2] == b"\xff\xd8"
    source = Image.open(io.BytesIO(client.get(poster.split("?")[0]).content))  # the bare, redacted PNG it derives from
    assert Image.open(io.BytesIO(got.content)).size == (min(800, source.width), round(source.height * min(800, source.width) / source.width))
    assert "resized copies" in page  # the page says what these pictures are


def test_display_copy_route_serves_only_configured_widths(client: TestClient) -> None:
    job_id = service.ensure_sample_job(client.app.state.settings)
    original = client.get(f"/guides/{job_id}/image/1")
    assert original.headers["content-type"] == "image/png"
    small = client.get(f"/guides/{job_id}/image/1?w=480")
    assert small.headers["content-type"] == "image/webp" and 0 < len(small.content) <= 2 * len(original.content) + 2048
    got = Image.open(io.BytesIO(small.content))
    assert got.format == "WEBP" and got.width == min(480, Image.open(io.BytesIO(original.content)).width)
    top = Image.open(io.BytesIO(client.get(f"/guides/{job_id}/image/1?w=1280").content))
    assert top.width == min(1280, Image.open(io.BytesIO(original.content)).width)  # never upscaled
    assert client.get(f"/guides/{job_id}/image/1?w=481").status_code == 404
    assert client.get(f"/guides/{job_id}/image/99?w=480").status_code == 404
    assert client.get(f"/guides/{job_id}/image/1?w=foo").status_code == 422  # a clean refusal, never a 500


def test_originals_stay_authoritative_in_route_zip_and_export(client: TestClient) -> None:
    job_id = service.ensure_sample_job(client.app.state.settings)
    page = client.get(f"/guides/{job_id}").text
    token = re.search(r"/images.zip\?token=([^\"&]+)", page).group(1)
    stored = {f"step-{n:02}.png": None for n in range(1, 9)}
    zipped = zipfile.ZipFile(io.BytesIO(client.get(f"/guides/{job_id}/images.zip?token={token}").content))
    assert zipped.namelist() == list(stored)
    routes = set()  # every stored image as the bare route URL (no ?w=) serves it: the authoritative PNGs
    for i in range(30):
        r = client.get(f"/guides/{job_id}/image/{i}")
        if r.status_code == 200:
            assert r.headers["content-type"] == "image/png" and r.content[:8] == b"\x89PNG\r\n\x1a\n"
            routes.add(r.content)
    # steps without passed labels ship the stored file byte for byte; labelled steps are drawn from the original frame
    assert all(zipped.read(n) in routes for n in ("step-01.png", "step-02.png", "step-03.png", "step-04.png", "step-08.png"))
    assert all(zipped.read(n)[:8] == b"\x89PNG\r\n\x1a\n" for n in stored)
    exported = zipfile.ZipFile(io.BytesIO(client.get(f"/guides/{job_id}/export.zip?token={token}").content))
    pngs = [exported.read(n) for n in exported.namelist() if n.endswith(".png")]
    assert pngs and all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)
    assert all(p in routes for p in pngs)  # the offline export carries the same original bytes
    assert not any(n.endswith(".webp") for n in exported.namelist())


def test_copy_is_much_smaller_for_a_representative_still(tmp_path: Path) -> None:
    import random
    rnd = random.Random(7)  # smooth, photo-like content (a gradient alone makes a tiny PNG)
    small = Image.frombytes("RGB", (64, 36), rnd.randbytes(64 * 36 * 3))
    src = tmp_path / "step-01.png"
    big = small.resize((1280, 720), Image.BICUBIC)
    big = Image.blend(big, Image.frombytes("RGB", (1280, 720), rnd.randbytes(1280 * 720 * 3)), 0.08)
    big.save(src)
    assert src.stat().st_size > 500_000
    body, _ = display_images.display_bytes(src, 480)
    data = body.read_bytes() if isinstance(body, Path) else body
    assert len(data) < src.stat().st_size / 10


def test_concurrent_cold_requests_write_one_intact_copy(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    src = _redacted_still(tmp_path / "step-01.png")
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: display_images.display_bytes(src, 480), range(16)))
    copies = [p for p in (tmp_path / "display").iterdir()]
    assert len(copies) == 1 and not any(p.name.endswith(".tmp") for p in copies)
    for body, media in results:
        data = body.read_bytes() if isinstance(body, Path) else body
        assert media == "image/webp" and Image.open(io.BytesIO(data)).width == 480
