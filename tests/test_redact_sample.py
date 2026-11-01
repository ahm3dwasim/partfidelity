"""The public sample hides identifiers and screens, repeatably."""
from __future__ import annotations

import hashlib
import os
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db, service
from app.main import create_app
from app.render import read_png_rgb, write_png_rgb
from app.settings import Settings
from tools import redact_sample, redactions, sample_media

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "app" / "sample_real"
W, H = 320, 240

FIXTURE_TOML = """
[style]
blur_sigma = 6.0
panel_rgb = [52, 56, 62]
caption_rgb = [235, 238, 242]
caption = "Screen hidden for privacy"
caption_width_fraction = 0.5
caption_margin_px = 4
panel_noise_tolerance = 6
webp_quality = 90

[[still]]
file = "frames/step-01.png"
hide_screen = false
blur = [[40, 40, 80, 60]]

[[still]]
file = "frames/step-02.png"
hide_screen = true
blur = []

[[how_still]]
file = "k.webp"
blur = [[10, 10, 60, 40]]

[[clip_hide]]
start_s = 3.0
end_s = 3.9

[[clip_box]]
keyframes = [[0.5, 0, 0, 40, 0, 40, 40, 0, 40], [1.5, 100, 0, 140, 0, 140, 40, 100, 40]]

[[clip_check]]
start_s = 0.6
end_s = 1.4
min_boxes = 1

[[clip_check]]
start_s = 3.1
end_s = 3.8
min_boxes = 1
"""
FIX_FILES = ("sample.mp4", "frames/step-01.png", "frames/step-02.png",
             "labels/step-01-part-1.png", "labels/step-02-part-1.png")


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def _fixture(tmp: Path, monkeypatch, toml: str = FIXTURE_TOML) -> Path:
    root = tmp / "repo"
    sample = root / "app/sample_real"
    for sub in ("frames", "labels"):
        (sample / sub).mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=10:duration=4",
                    "-pix_fmt", "yuv420p", str(sample / "sample.mp4")], check=True)
    frames = {}
    for i in (1, 2):
        frames[i] = _noise(i)
        write_png_rgb(sample / f"frames/step-0{i}.png", W, H, frames[i].tobytes())
    labels = []
    for i in (1, 2):  # a thumbnail is its frame plus a badge at the corner
        shown = frames[i].copy()
        shown[200:230, 280:310] = (255, 140, 0)
        write_png_rgb(sample / f"labels/step-0{i}-part-1.png", W, H, shown.tobytes())
        labels.append({"slot": {"step_index": i - 1, "part_index": 0},
                       "provisional": {"rendered_image_path": f"labels/step-0{i}-part-1.png"}})
    (sample / "guide.json").write_text(json.dumps(
        {"frame_paths": ["frames/step-01.png", "frames/step-02.png"], "labels": labels}))
    how = root / "static/ui/media/how"
    how.mkdir(parents=True)
    cv2.imwrite(str(how / "k.webp"), _noise(9), [cv2.IMWRITE_WEBP_QUALITY, 100])
    cv2.imwrite(str(how / "other.webp"), _noise(10), [cv2.IMWRITE_WEBP_QUALITY, 100])
    (root / "config").mkdir()
    (root / "config/how_media.json").write_text(json.dumps({
        "dir": "static/ui/media/how", "bundle": "assets/how/how-frames.tar",
        "files": [{"name": n, "size": (how / n).stat().st_size,
                   "sha256": hashlib.sha256((how / n).read_bytes()).hexdigest()} for n in ("k.webp", "other.webp")]}))
    (root / "config/redactions.toml").write_text(toml)
    monkeypatch.setattr(sample_media, "FILES", FIX_FILES)
    return root


def _snapshot(root: Path) -> dict:
    return {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _png(path: Path) -> np.ndarray:
    width, height, data = read_png_rgb(path)
    return np.frombuffer(data, np.uint8).reshape(height, width, 3)


def _clip_frame(path: Path, t: float) -> np.ndarray:
    video = cv2.VideoCapture(str(path))
    video.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = video.read()
    video.release()
    assert ok
    return frame[..., ::-1]


def test_real_config_loads_and_covers_the_screen_frame_and_the_hidden_clip_span() -> None:
    config = redactions.load()
    assert any(row["file"] == "frames/step-09.png" and row["hide_screen"] for row in config["still"])
    assert any(hide["start_s"] <= 180 and hide["end_s"] >= 189.2 for hide in config["clip_hide"])


@pytest.mark.parametrize("edit", [
    lambda t: t.replace("blur_sigma", "blur_sigmaa"),                       # misspelled key
    lambda t: t.replace("blur_sigma = 6.0\n", ""),                          # missing key
    lambda t: t.replace("[[still]]\nfile = \"frames/step-01.png\"\nhide_screen = false\n",
                        "[[still]]\nfile = \"frames/step-01.png\"\nhide_screen = \"no\"\n"),  # mistyped
    lambda t: t.replace("[[clip_box]]", "[[clip_boxes]]"),                  # misspelled table
    lambda t: t.replace("[[0.5, 0, 0,", "[[1.5, 0, 0,"),                   # time runs backwards
    lambda t: t.replace("[1.5, 100, 0, 140", "[1.5, 100, 0, 140, 5"),      # keyframe with the wrong number of values
    lambda t: t.replace("[[40, 40, 80, 60]]", "[[40, 40, 80, 40, 80, 60]]"),  # a region that is neither rect nor quad
    lambda t: t.replace("[[40, 40, 80, 60]]", "[[40, 40, 0, 60]]"),        # empty region
])
def test_loader_rejects_a_bad_config(tmp_path: Path, edit) -> None:
    path = tmp_path / "redactions.toml"
    path.write_text(edit(FIXTURE_TOML))
    with pytest.raises(redactions.RedactionConfigError):
        redactions.load(path)


def test_the_tool_changes_only_the_listed_pixels_and_leaves_originals_alone(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    before = _snapshot(root)
    out = tmp_path / "out"
    duration = redact_sample.run(out, root)
    assert _snapshot(root) == before
    assert not list(tmp_path.glob(".redact-*"))
    assert abs(duration - 4.0) <= 0.15

    original, done = _png(root / "app/sample_real/frames/step-01.png"), _png(out / "app/sample_real/frames/step-01.png")
    region = (slice(40, 100), slice(40, 120))
    mask = np.zeros((H, W), bool)
    mask[region] = True
    assert (done[mask] != original[mask]).any(axis=1).mean() > 0.9          # blurred noise differs almost everywhere
    assert np.array_equal(done[~mask], original[~mask])                      # nothing else moved

    screen = _png(out / "app/sample_real/frames/step-02.png")                # fully obscured with a plain panel
    assert tuple(screen[2, 2]) == (52, 56, 62) and (screen != (52, 56, 62)).any()  # panel colour plus caption pixels
    assert not np.array_equal(screen, _noise(2))

    badge = _png(out / "app/sample_real/labels/step-01-part-1.png")          # thumbnail = redacted frame + its badge
    assert np.array_equal(badge[200:230, 280:310], np.full((30, 30, 3), (255, 140, 0), np.uint8))
    assert np.array_equal(badge[:200], done[:200])                           # the rest of the thumbnail is the redacted frame

    how = cv2.imread(str(out / "static/ui/media/how/k.webp"))
    assert not np.array_equal(how[10:50, 10:70], cv2.imread(str(root / "static/ui/media/how/k.webp"))[10:50, 10:70])
    assert not (out / "static/ui/media/how/other.webp").exists()             # only listed stills are rewritten


def test_the_clip_boxes_follow_their_keyframes_and_the_hidden_span_is_the_panel(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    redact_sample.run(out, root)
    source, done = root / "app/sample_real/sample.mp4", out / "app/sample_real/sample.mp4"
    def moved(t, box):  # mean difference of a region against the source frame
        x, y = box
        a, b = _clip_frame(source, t).astype(int), _clip_frame(done, t).astype(int)
        return np.abs(a[y:y + 40, x:x + 40] - b[y:y + 40, x:x + 40]).mean()
    assert moved(1.0, (50, 0)) > 20       # the box covers x=50..90 at t=1.0 (halfway along)
    assert moved(1.0, (200, 150)) < 12    # far from the box: only codec noise
    assert moved(0.2, (10, 10)) < 12      # before the box starts
    panel = _clip_frame(done, 3.5)
    assert np.abs(panel[5, 5].astype(int) - (52, 56, 62)).max() <= 3
    assert np.abs(_clip_frame(done, 2.5).astype(int) - _clip_frame(source, 2.5).astype(int)).mean() < 12


def test_a_quad_region_blurs_only_inside_its_corners() -> None:
    image = _noise(3)
    quad = [100, 100, 160, 100, 160, 110, 100, 140]          # a wedge: the bounding box is 60x40, the wedge is smaller
    done = redact_sample.blur_regions(image, [quad], 6.0, "t")
    inside = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(inside, np.array(quad, np.int32).reshape(4, 2), 1)
    assert np.array_equal(done[inside == 0], image[inside == 0])
    assert (done[inside == 1] != image[inside == 1]).any(axis=1).mean() > 0.9


def test_box_at_interpolates_between_keyframes_and_is_none_outside() -> None:
    box = {"keyframes": [[1.0, 0, 0, 10, 0, 10, 10, 0, 10], [2.0, 20, 0, 30, 0, 30, 10, 20, 10]]}
    assert redact_sample.box_at(box, 0.9) is None and redact_sample.box_at(box, 2.1) is None
    assert redact_sample.box_at(box, 1.5).tolist() == [[10, 0], [20, 0], [20, 10], [10, 10]]


def test_a_live_box_is_filled_with_its_own_mean_colour_in_each_frame() -> None:
    config = {"clip_hide": [], "clip_box": [{"keyframes": [[0.0, 10, 10, 30, 10, 30, 30, 10, 30],
                                                           [1.0, 50, 10, 70, 10, 70, 30, 50, 30]]}]}
    frame = _noise(4)
    at_start = redact_sample.redact_frame(frame.copy(), 0.0, config, np.zeros_like(frame))
    assert len({tuple(p) for p in at_start[12:28, 12:28].reshape(-1, 3)}) == 1     # one flat colour
    assert np.array_equal(at_start[:, 40:], frame[:, 40:])                          # the box has not reached x=40 yet
    assert redact_sample.redact_frame(frame.copy(), 1.5, config, np.zeros_like(frame)).tolist() == frame.tolist()


def test_every_real_region_hugs_its_label() -> None:
    """Precision guard: no region of the real config is a big blur (an over-wide blur hides the content)."""
    config = redactions.load()
    def box_area(region):
        xs, ys = (region[::2], region[1::2]) if len(region) == 8 else ((region[0], region[0] + region[2]),
                                                                       (region[1], region[1] + region[3]))
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    for row in (*config["still"], *config["how_still"]):
        assert all(box_area(r) <= 0.06 * 1280 * 720 for r in row["blur"]), row["file"]
    for box in config["clip_box"]:
        assert all(box_area(k[1:]) <= 0.06 * 1280 * 720 for k in box["keyframes"])


def test_the_tool_rewrites_the_manifest_zip_and_bundle_to_match(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    redact_sample.run(out, root)
    rows = sample_media.manifest(out / "config/sample_media.sha256")
    for name in FIX_FILES:
        shipped = out / "app/sample_real" / name
        data = (shipped if shipped.is_file() else root / "app/sample_real" / name).read_bytes()
        assert rows[name] == (hashlib.sha256(data).hexdigest(), len(data))
    with zipfile.ZipFile(out / "var/sample-media.zip") as archive:
        assert set(archive.namelist()) == set(FIX_FILES)
    how = json.loads((out / "config/how_media.json").read_text())
    with tarfile.open(out / "assets/how/how-frames.tar") as tar:
        for entry in how["files"]:
            data = tar.extractfile(entry["name"]).read()
            assert (entry["size"], entry["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())
    changed = {e["name"]: e["sha256"] for e in how["files"]}
    assert changed["k.webp"] != hashlib.sha256((root / "static/ui/media/how/k.webp").read_bytes()).hexdigest()
    assert changed["other.webp"] == hashlib.sha256((root / "static/ui/media/how/other.webp").read_bytes()).hexdigest()


def test_the_tool_refuses_an_existing_out_dir_and_a_region_off_the_image(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch, FIXTURE_TOML.replace("[[40, 40, 80, 60]]", "[[300, 40, 80, 60]]"))
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        redact_sample.run(tmp_path / "taken", root)
    with pytest.raises(ValueError, match="leaves the"):
        redact_sample.run(tmp_path / "out", root)
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob(".redact-*"))


# --- the served images: same URLs, new bytes, never cached stale ---

def _client(tmp_path: Path) -> TestClient:
    database = tmp_path / "db.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return TestClient(create_app(Settings(data_root=tmp_path / "data", db_path=database, production=False,
                                          pipeline_mode="replay")))


def _bundle(tmp_path: Path, monkeypatch, shade: int) -> Path:
    """A stand-in sample bundle: the real guide.json, tiny stills of one shade, a synthetic clip."""
    bundle = tmp_path / "bundle"
    guide = json.loads((SAMPLE / "guide.json").read_text())
    names = [*guide["frame_paths"], *(label["provisional"]["rendered_image_path"] for label in guide["labels"])]
    for number, name in enumerate(names):
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        write_png_rgb(bundle / name, 4, 4, bytes([shade, number, 7]) * 16)
    for name in ("guide.json", "receipts.json", "step_notes.json"):
        shutil.copy(SAMPLE / name, bundle / name)
    service.write_sample_clip(bundle / "sample.mp4")
    monkeypatch.setattr(service, "_sample_bundle", lambda settings: bundle)
    return bundle


def _restore_old_install(tmp_path: Path, bundle: Path, version: int = 2) -> str:
    """What the previous release left on the volume: old stills, a guide with no excluded spans, a version-2 pointer."""
    data = tmp_path / "data"
    guide = json.loads((bundle / "guide.json").read_text())
    job_id = guide["job_id"]
    directory = data / "jobs" / job_id
    names = [*guide["frame_paths"], *(label["provisional"]["rendered_image_path"] for label in guide["labels"])]
    for number, name in enumerate(names):
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        write_png_rgb(directory / name, 4, 4, bytes([99, number, 7]) * 16)   # old, unredacted bytes
    old = json.loads((SAMPLE / "guide.json").read_text())
    old["excluded_spans"] = []
    old["frame_paths"] = [str(directory / p) for p in old["frame_paths"]]
    for label in old["labels"]:
        label["detection"]["mask_path"] = str(directory / label["detection"]["mask_path"])
        label["provisional"]["rendered_image_path"] = str(directory / label["provisional"]["rendered_image_path"])
    (directory / "guide.json").write_text(json.dumps(old))
    (data / "sample_job.json").write_text(json.dumps({"job_id": job_id, "bundle_version": version}))
    return job_id


def test_guide_images_are_no_cache(tmp_path, monkeypatch) -> None:
    _bundle(tmp_path, monkeypatch, shade=10)
    client = _client(tmp_path)
    job_id = service.ensure_sample_job(client.app.state.settings)
    first = client.get(f"/guides/{job_id}/image/12")
    assert first.status_code == 200 and first.headers["cache-control"] == "no-cache"


def test_a_restored_old_install_serves_redacted_bytes_on_the_very_first_request(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path, monkeypatch, shade=10)
    job_id = _restore_old_install(tmp_path, bundle)
    client = _client(tmp_path)                    # startup refreshes before traffic; nothing else has run
    response = client.get(f"/guides/{job_id}/image/12")   # a direct hit on the known URL, not via /guide/sample
    assert response.status_code == 200
    assert response.content in {path.read_bytes() for path in bundle.rglob("*.png")}   # the bundle's bytes, not the old install's
    pointer = json.loads((tmp_path / "data" / "sample_job.json").read_text())
    assert pointer["bundle_version"] == client.app.state.settings.sample_bundle_version


def test_an_old_install_gets_the_new_guide_with_its_excluded_spans(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path, monkeypatch, shade=10)
    job_id = _restore_old_install(tmp_path, bundle)
    installed = tmp_path / "data" / "jobs" / job_id / "guide.json"
    assert json.loads(installed.read_text())["excluded_spans"] == []
    _client(tmp_path)
    assert json.loads(installed.read_text())["excluded_spans"] == json.loads((bundle / "guide.json").read_text())["excluded_spans"] != []


def test_a_current_install_whose_stills_differ_is_refreshed_at_startup(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path, monkeypatch, shade=10)
    job_id = _restore_old_install(tmp_path, bundle)
    _client(tmp_path)                             # now a current (version 4) install
    for path in (*(bundle / "frames").glob("*.png"), *(bundle / "labels").glob("*.png")):
        write_png_rgb(path, 4, 4, bytes([200, 1, 7]) * 16)   # a later redaction rebuild, same version
    client = _client(tmp_path)
    after = client.get(f"/guides/{job_id}/image/12")
    assert after.status_code == 200 and after.content in {path.read_bytes() for path in bundle.rglob("*.png")}
    assert (tmp_path / "data/jobs" / job_id / "frames/step-01.png").read_bytes() == (bundle / "frames/step-01.png").read_bytes()


def test_when_the_refresh_fails_the_old_media_is_removed_and_the_sample_shows_the_error_page(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path, monkeypatch, shade=10)
    job_id = _restore_old_install(tmp_path, bundle)
    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(service, "_install_media", broken)
    client = _client(tmp_path)
    assert client.get(f"/guides/{job_id}/image/12").status_code == 404
    assert not list((tmp_path / "data/jobs" / job_id).rglob("*.png"))
    page = client.get("/guide/sample", follow_redirects=False)
    assert page.status_code == 200 and "sample is unavailable" in page.text


def test_startup_refuses_to_run_when_the_old_media_can_be_neither_refreshed_nor_removed(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path, monkeypatch, shade=10)
    _restore_old_install(tmp_path, bundle)
    def broken(*args, **kwargs):
        raise OSError("read-only volume")
    monkeypatch.setattr(service, "_install_media", broken)
    monkeypatch.setattr(service.shutil, "rmtree", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="refusing to start"):
        _client(tmp_path)


@pytest.mark.skipif(any(not (SAMPLE / n).is_file() for n in sample_media.FILES),
                    reason="real sample media absent; run python tools/sample_media.py unpack")
def test_image_12_of_the_real_sample_is_the_plain_panel_not_the_device_manager(tmp_path) -> None:
    client = _client(tmp_path)
    job_id = service.ensure_sample_job(client.app.state.settings)
    response = client.get(f"/guides/{job_id}/image/12")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-cache"
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    panel = np.array(redactions.load()["style"]["panel_rgb"])
    assert abs(image.astype(int) - panel).max(axis=2).mean() < 40  # almost every pixel is the panel; only the caption differs
    assert json.loads((SAMPLE / "guide.json").read_text())["excluded_spans"] == [
        {"start_s": 192.0, "end_s": 200.0, "kind": "outro_or_unrelated"}]


# --- file names stay inside their media dirs, clip frames are all covered, archives repeat ---

@pytest.mark.parametrize("name", ["/etc/hosts", "../frames/step-01.png", "frames/../step-01.png", "./frames/step-01.png",
                                  "frames//step-01.png", "frames\\step-01.png", "frames/", ""])
@pytest.mark.parametrize("table", ["still", "how_still"])
def test_loader_rejects_absolute_traversal_and_non_normalized_file_names(tmp_path, table, name) -> None:
    old = 'file = "frames/step-01.png"' if table == "still" else 'file = "k.webp"'
    path = tmp_path / "redactions.toml"
    path.write_text(FIXTURE_TOML.replace(old, f"file = {json.dumps(name)}"))
    with pytest.raises(redactions.RedactionConfigError):
        redactions.load(path)


def test_the_tool_refuses_a_path_that_leaves_its_media_dir_even_through_a_symlink(tmp_path) -> None:
    base = tmp_path / "frames"
    base.mkdir()
    (tmp_path / "outside.png").write_bytes(b"x")
    (base / "link.png").symlink_to(tmp_path / "outside.png")
    assert redact_sample.under(base, "inside.png") == base / "inside.png"
    with pytest.raises(ValueError, match="leaves"):
        redact_sample.under(base, "link.png")
    with pytest.raises(ValueError, match="leaves"):
        redact_sample.under(base, "../outside.png")


def test_the_real_config_has_a_live_box_in_every_30fps_frame_of_each_check_window() -> None:
    config = redactions.load()
    assert any(w["start_s"] <= 110.25 and w["end_s"] >= 112.5 for w in config["clip_check"])
    for window in config["clip_check"]:
        for frame in range(int(window["start_s"] * 30), int(window["end_s"] * 30) + 2):
            t = frame / 30
            if window["start_s"] <= t <= window["end_s"]:
                live = len(redact_sample.live_boxes(config, t))
                assert live >= window["min_boxes"], f"frame {frame} ({t:.3f}s) has {live} boxes"


def test_the_clip_check_fails_the_build_on_a_frame_gap(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch, FIXTURE_TOML.replace("min_boxes = 1\n\n[[clip_check]]\nstart_s = 3.1",
                                                                "min_boxes = 2\n\n[[clip_check]]\nstart_s = 3.1"))
    with pytest.raises(ValueError, match="boxes live"):
        redact_sample.run(tmp_path / "out", root)
    assert not (tmp_path / "out").exists()


def test_the_clip_check_reads_the_built_pixels_not_just_the_config(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    redact_sample.run(out, root)
    source, built = root / "app/sample_real/sample.mp4", out / "app/sample_real/sample.mp4"
    config = redactions.load(root / "config/redactions.toml")
    assert redact_sample.verify_clip(source, built, config) > 10
    with pytest.raises(ValueError, match="not filled"):
        redact_sample.verify_clip(source, source, config)             # an unredacted clip fails the same check


def test_two_runs_give_identical_bytes_whatever_the_file_times(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    redact_sample.run(tmp_path / "one", root)
    for path in root.rglob("*"):
        os.utime(path, (1_000_000, 1_000_000))
    redact_sample.run(tmp_path / "two", root)
    for name in ("var/sample-media.zip", "assets/how/how-frames.tar", "config/sample_media.sha256", "config/how_media.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name
    for member in zipfile.ZipFile(tmp_path / "one/var/sample-media.zip").infolist():
        assert member.date_time == (1980, 1, 1, 0, 0, 0)
    for member in tarfile.open(tmp_path / "one/assets/how/how-frames.tar"):
        assert (member.mtime, member.uid, member.gid, member.uname, member.gname) == (0, 0, 0, "", "")


def test_a_failed_writer_launch_stops_the_decoder(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    started = []
    real = subprocess.Popen
    def flaky(cmd, *args, **kwargs):
        if len(started) == 1:
            raise OSError("cannot start the encoder")
        started.append(real(cmd, *args, **kwargs))
        return started[-1]
    monkeypatch.setattr(redact_sample.subprocess, "Popen", flaky)
    with pytest.raises(OSError, match="encoder"):
        redact_sample.run(tmp_path / "out", root)
    assert len(started) == 1 and started[0].poll() is not None      # killed and reaped, not leaked
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob(".redact-*"))


def _guide_with_label_path(root: Path, name: str) -> None:
    guide = root / "app/sample_real/guide.json"
    data = json.loads(guide.read_text())
    data["labels"][0]["provisional"]["rendered_image_path"] = name
    guide.write_text(json.dumps(data))


@pytest.mark.parametrize("kind", ["absolute", "dotdot"])
def test_a_guide_label_path_that_leaves_the_sample_is_refused_before_any_read_or_write(tmp_path, monkeypatch, kind) -> None:
    root = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside.png"
    write_png_rgb(outside, W, H, _noise(5).tobytes())
    before = outside.read_bytes()
    _guide_with_label_path(root, str(outside) if kind == "absolute" else "../../../../outside.png")
    with pytest.raises(ValueError, match="leaves"):
        redact_sample.run(tmp_path / "out", root)
    assert outside.read_bytes() == before and not (tmp_path / "out").exists()


def test_a_guide_label_output_path_through_a_symlink_out_of_the_media_dir_is_refused(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    sample, media, elsewhere = root / "app/sample_real", tmp_path / "media", tmp_path / "elsewhere"
    media.mkdir()
    elsewhere.mkdir()
    (media / "labels").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="leaves"):
        redact_sample.redact_guide_stills(sample, media, redactions.load(root / "config/redactions.toml"))
    assert not list(elsewhere.iterdir())


def test_a_clip_box_that_is_off_the_frame_cannot_satisfy_min_boxes(tmp_path, monkeypatch) -> None:
    off = FIXTURE_TOML.replace("[[clip_check]]\nstart_s = 0.6", """[[clip_box]]
keyframes = [[0.5, 900, 900, 940, 900, 940, 940, 900, 940], [1.5, 900, 900, 940, 900, 940, 940, 900, 940]]

[[clip_check]]
start_s = 0.6""", 1)
    root = _fixture(tmp_path, monkeypatch, off)
    with pytest.raises(ValueError, match="off the 320x240 frame"):
        redact_sample.run(tmp_path / "out", root)
    assert not (tmp_path / "out").exists()


def test_both_clip_urls_are_no_cache_and_replacement_bytes_change_the_etag(tmp_path, monkeypatch) -> None:
    bundle = _bundle(tmp_path, monkeypatch, shade=10)
    client = _client(tmp_path)
    job_id = service.ensure_sample_job(client.app.state.settings)
    for url in (f"/guides/{job_id}/clip", f"/guide/{job_id}/clip"):
        first = client.get(url)
        assert first.status_code == 200 and first.headers["cache-control"] == "no-cache"
        clip = bundle / "sample.mp4"
        clip.write_bytes(clip.read_bytes() + bytes([len(url)]) * 64)   # the same URL now serves different bytes
        second = client.get(url)
        assert second.headers["cache-control"] == "no-cache" and second.headers["etag"] != first.headers["etag"]
        assert client.get(url, headers={"If-None-Match": first.headers["etag"]}).status_code == 200


def _write_clip(path: Path, frames: list[np.ndarray]) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "10", "-i", "-",
                           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
                          input=b"".join(f.tobytes() for f in frames), check=True)
    assert proc.returncode == 0


def _hide_config(start_s: float) -> dict:
    config = redactions.load(ROOT / "config/redactions.toml")
    return {"style": config["style"], "clip_hide": [{"start_s": start_s, "end_s": 9.9}], "clip_box": [],
            "clip_check": [{"start_s": 3.0, "end_s": 3.5, "min_boxes": 1}]}  # inside the span: the panel check governs


def _cut_clips(tmp_path: Path, start_s: float) -> tuple[Path, Path, dict]:
    """A 4 s clip at 10 fps: a flat grey lid, then a bright screen from frame 20 (2.0 s); the built clip shows the panel from start_s."""
    config = _hide_config(start_s)
    lid, screen = np.full((H, W, 3), 100, np.uint8), _noise(3) // 2 + 100
    original = [lid] * 20 + [screen] * 20
    shown = redact_sample.panel(W, H, config["style"])
    built = [frame if n / 10 < start_s else shown for n, frame in enumerate(original)]
    _write_clip(tmp_path / "source.mp4", original)
    _write_clip(tmp_path / "built.mp4", built)
    return tmp_path / "source.mp4", tmp_path / "built.mp4", config


def test_the_clip_check_fails_when_the_screen_shows_for_three_frames_before_the_panel(tmp_path) -> None:
    source, built, config = _cut_clips(tmp_path, 2.3)   # frames 20, 21, 22 show the screen
    with pytest.raises(ValueError, match="scene cut is not covered"):
        redact_sample.verify_clip(source, built, config)


def test_the_clip_check_passes_when_the_panel_starts_before_the_cut(tmp_path) -> None:
    source, built, config = _cut_clips(tmp_path, 1.7)   # three frames of margin
    assert redact_sample.verify_clip(source, built, config) > 10


def test_the_clip_check_needs_two_frames_of_margin_before_the_cut(tmp_path) -> None:
    source, built, config = _cut_clips(tmp_path, 1.9)   # one frame of margin
    with pytest.raises(ValueError, match="scene cut is not covered"):
        redact_sample.verify_clip(source, built, config)


def test_the_clip_check_fails_a_frame_inside_a_hidden_span_that_is_not_the_panel(tmp_path) -> None:
    source, built, config = _cut_clips(tmp_path, 1.7)
    config["clip_hide"][0]["start_s"] = 1.7
    shown = redact_sample.panel(W, H, config["style"])
    leaky = [_noise(3) // 2 + 100 if n == 30 else shown for n in range(40)]  # the span starts at 1.7; frame 30 is a screen again
    _write_clip(tmp_path / "leaky.mp4", [np.full((H, W, 3), 100, np.uint8)] * 17 + leaky[17:])
    with pytest.raises(ValueError, match="panel is not applied"):
        redact_sample.verify_clip(source, tmp_path / "leaky.mp4", config)


def test_the_clip_check_fails_a_hidden_span_that_no_frame_falls_in(tmp_path) -> None:
    source, built, config = _cut_clips(tmp_path, 1.7)
    config["clip_hide"], config["clip_check"] = [{"start_s": 50.0, "end_s": 60.0}], []
    with pytest.raises(ValueError, match="no frame"):
        redact_sample.verify_clip(source, built, config)


def _panel_clip_with(tmp_path: Path, edit) -> tuple[Path, Path, dict]:
    """The 4 s clip with every frame from 1.7 s the panel; ``edit`` changes the panel frames it draws in (RGB, in place)."""
    source, built, config = _cut_clips(tmp_path, 1.7)
    shown = redact_sample.panel(W, H, config["style"])
    marked = shown.copy()
    edit(marked)
    _write_clip(tmp_path / "marked.mp4", [np.full((H, W, 3), 100, np.uint8)] * 17 + [marked] * 23)
    return source, tmp_path / "marked.mp4", config


def test_the_clip_check_fails_a_one_percent_taskbar_strip_on_a_panel_frame(tmp_path) -> None:
    def strip(frame):
        frame[H - 3:, :] = 255  # 3 of 240 rows: 1.25% of the frame, white on grey
    source, built, config = _panel_clip_with(tmp_path, strip)
    with pytest.raises(ValueError, match="panel is not applied"):
        redact_sample.verify_clip(source, built, config)


def test_the_clip_check_fails_one_small_bright_patch_on_a_panel_frame(tmp_path) -> None:
    def patch(frame):
        frame[5:15, 5:15] = 255
    source, built, config = _panel_clip_with(tmp_path, patch)
    with pytest.raises(ValueError, match="panel is not applied"):
        redact_sample.verify_clip(source, built, config)


def test_the_clip_check_passes_a_panel_with_real_codec_noise(tmp_path) -> None:
    source, built, config = _panel_clip_with(tmp_path, lambda frame: None)   # crf 12 encode of the exact panel
    assert redact_sample.verify_clip(source, built, config) > 10
