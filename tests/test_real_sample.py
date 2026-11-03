"""The sample guide is a real guide from a real repair video."""
from __future__ import annotations

import io
import json
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from app import db, service
from app.deploy_config import load_deploy_config
from app.main import create_app
from app.pipeline import guide_path
from app.settings import Settings
from tools.sample_media import FILES

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "app" / Settings.__dataclass_fields__["sample_bundle_dirname"].default


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    database = tmp_path / "db.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    settings = Settings(data_root=tmp_path / "data", db_path=database, production=False, pipeline_mode="replay")
    return TestClient(create_app(settings))


def test_sample_guide_renders_the_real_steps(client: TestClient) -> None:
    page = client.get("/guide/sample", follow_redirects=True)
    assert page.status_code == 200
    assert "8 steps" in page.text and "9 steps" not in page.text
    for title in ("Remove bottom cover", "Not shown in the video: the battery disconnect", "Power on and verify in Windows"):
        assert title in page.text
    assert "Warning: battery" in page.text and "Warning: esd" in page.text
    assert "Edit guide" in page.text and "Follow guide" in page.text  # both modes stay
    assert 'class="pf-pin"' in page.text  # the checked labels keep their pins
    assert "Step 9" not in page.text  # the intro left the numbered steps
    assert "Made from a real repair video" in page.text
    lowered = page.text.lower()
    assert "fixture" not in lowered and "demo data" not in lowered and "test clip" not in lowered


@pytest.mark.skipif(any(not (BUNDLE / name).is_file() for name in FILES), reason="real sample media absent; run python tools/sample_media.py unpack")
def test_sample_clip_url_serves_the_real_clip(client: TestClient) -> None:
    job_id = service.ensure_sample_job(client.app.state.settings)
    page = client.get(f"/guides/{job_id}")
    assert f'src="/guides/{job_id}/clip"' in page.text
    clip = client.get(f"/guides/{job_id}/clip")
    assert clip.status_code == 200
    assert clip.headers["content-type"] == "video/mp4"
    assert clip.content[4:8] == b"ftyp"
    assert client.get(f"/guides/{job_id}/image/0").status_code == 200


@pytest.mark.skipif(any(not (BUNDLE / name).is_file() for name in FILES), reason="real sample media absent; run python tools/sample_media.py unpack")
def test_step_times_fit_inside_the_shipped_clip() -> None:
    guide = json.loads((BUNDLE / "guide.json").read_text())
    capture = cv2.VideoCapture(str(BUNDLE / "sample.mp4"))
    try:
        length_s = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        capture.release()
    steps = guide["steps"]
    assert steps[0]["start_s"] == 0 and steps[-1]["end_s"] <= length_s + 0.5
    assert all(a["end_s"] == b["start_s"] for a, b in zip(steps, steps[1:]))
    assert height == 720


def test_every_sample_file_ships_in_the_image() -> None:
    deploy = load_deploy_config(ROOT / "deploy" / "modal_deploy.toml")
    assert "app" in deploy.modal.image_folders  # the bundle sits under app/, copied whole
    guide = json.loads((BUNDLE / "guide.json").read_text())
    needed = {Path("guide.json"), Path("sample.mp4"), Path("step_notes.json"), Path("receipts.json"), Path("open_weights_receipts.json"), *map(Path, guide["frame_paths"])}
    needed |= {Path(label["provisional"]["rendered_image_path"]) for label in guide["labels"]}
    needed |= {Path(label["detection"]["mask_path"]) for label in guide["labels"]}
    on_disk = {p.relative_to(BUNDLE) for p in BUNDLE.rglob("*") if p.is_file()}
    assert on_disk <= needed  # local private media may be absent
    ignored = subprocess.run(
        ["git", "check-ignore", *[str((BUNDLE / n).relative_to(ROOT)) for n in needed]],
        cwd=ROOT, capture_output=True,
    )
    ignored_paths = set(ignored.stdout.decode().splitlines())
    media = {str((BUNDLE / n).relative_to(ROOT)) for n in needed if n == Path("sample.mp4") or n.parts[0] in {"frames", "labels"}}
    assert ignored_paths == media
    assert media <= set(deploy.modal.image_files)


def test_sample_gap_is_honest_on_all_surfaces_even_for_an_installed_old_guide(client: TestClient) -> None:
    job_id = service.ensure_sample_job(client.app.state.settings)
    settings = client.app.state.settings
    path = guide_path(job_id, data_root=settings.data_root, guide_filename=settings.guide_filename)
    old = json.loads(path.read_text())
    old["steps"][5]["action"] = "Disconnect and remove internal battery"
    path.write_text(json.dumps(old))
    expected = "Not shown in the video: the battery disconnect"
    page = client.get(f"/guides/{job_id}")
    assert expected in page.text and "Disconnect and remove internal battery" not in page.text
    assert "DRAFT" in page.text
    home = client.get("/")
    assert "Disconnect and remove internal battery" not in home.text
    assert "Draft" in home.text
    reader = client.get("/reader/sample", follow_redirects=True)
    assert expected in reader.text and "Disconnect and remove internal battery" not in reader.text
    token = re.search(r'/export.zip\?token=([^"&]+)', page.text).group(1)
    export = client.get(f"/guides/{job_id}/export.zip?token={token}")
    assert export.status_code == 200
    with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
        offline = archive.read("guide.html").decode()
    assert expected in offline and "Disconnect and remove internal battery" not in offline
    assert "DRAFT" in offline


def test_sample_fails_closed_if_shipped_gap_wording_is_missing(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    job_id = service.ensure_sample_job(client.app.state.settings)
    monkeypatch.setattr(service, "_sample_bundle", lambda settings: tmp_path / "missing")
    assert client.get(f"/guides/{job_id}").status_code == 404


def test_old_sample_pointer_is_upgraded(client: TestClient, tmp_path: Path) -> None:
    settings = client.app.state.settings
    clip = service.write_sample_clip(tmp_path / "old-sample.mp4")
    result = service.run_job(
        clip, clients=service.default_replay_clients(),
        uploads=service.UploadStoreImpl(settings.db_path),
        lifecycle=service.LifecycleRepositoryImpl(settings.db_path),
        anon_id="sample-anon", ip_hash="sample-ip", replay=True,
        settings=settings, retain_clip=False,
    )
    assert result.state == "done", result.error
    old_id = result.job_id
    pointer = Path(settings.data_root) / settings.sample_pointer_filename
    pointer.write_text(json.dumps({"job_id": str(old_id)}))
    new_id = service.ensure_sample_job(settings)
    assert new_id != old_id
    page = client.get(f"/guides/{new_id}")
    assert "Not shown in the video: the battery disconnect" in page.text
    assert json.loads(pointer.read_text())["bundle_version"] == settings.sample_bundle_version


def test_no_footage_is_tracked_under_app_or_static() -> None:
    files = subprocess.check_output(["git", "ls-files", "app", "static"], cwd=ROOT, text=True).splitlines()
    footage = [name for name in files if Path(name).suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}
               or name.startswith(("app/sample_real/frames/", "app/sample_real/labels/", "static/ui/media/local/"))]
    assert footage == []


def test_missing_sample_shows_readable_fallback_and_logs(client, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(service, "_sample_bundle", lambda settings: tmp_path / "absent")
    page = client.get("/guide/sample")
    assert page.status_code == 200
    assert "Sample guide: fan cable reseat" in page.text
    assert "sample guide unavailable" in caplog.text
    assert "illustrated example is not a filmed repair" in page.text


def _fixture_manifest(sample_media):
    import hashlib
    return {n: (hashlib.sha256(n.encode()).hexdigest(), len(n.encode())) for n in sample_media.FILES}


def test_sample_media_manifest_lists_the_18_approved_files():
    from tools import sample_media
    rows = sample_media.manifest()
    assert len(rows) == 18 and set(rows) == set(sample_media.FILES)
    assert all(len(d) == 64 and size > 0 for d, size in rows.values())
    masks = [n for n in sample_media.FILES if n.endswith(".mask.png")]
    assert len(masks) == 4  # the true SAM part masks, one per sample label


def test_sample_media_rejects_wrong_bytes_everywhere(tmp_path):
    from tools import sample_media
    expected = _fixture_manifest(sample_media)
    source = tmp_path / "source"
    for name in sample_media.FILES:
        path = source / "app/sample_real" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    sample_media.check(source, expected)
    bundle = tmp_path / "ok.zip"
    sample_media.pack(source, bundle, expected)
    (source / "app/sample_real/sample.mp4").write_bytes(b"someone else's clip")
    with pytest.raises(ValueError, match=r"sample\.mp4 does not match.*expected sha256 " + expected["sample.mp4"][0]):
        sample_media.check(source, expected)
    with pytest.raises(ValueError, match="sample.mp4 does not match"):
        sample_media.pack(source, tmp_path / "bad.zip", expected)
    import zipfile
    bad = tmp_path / "wrong.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        for name in sample_media.FILES:
            archive.writestr(name, b"wrong" if name == "frames/step-03.png" else name.encode())
    target = tmp_path / "target-bad"
    with pytest.raises(ValueError, match=r"frames/step-03\.png does not match"):
        sample_media.unpack(target, bad, expected)
    assert not target.exists()  # nothing written before a mismatch


def test_installed_real_sample_matches_the_manifest():
    from tools import sample_media
    if any(not (sample_media.ROOT / "app/sample_real" / n).is_file() for n in sample_media.FILES):
        pytest.skip("real sample media absent; run python tools/sample_media.py unpack")
    sample_media.check()


def test_sample_media_bundle_roundtrip_and_missing_command(tmp_path):
    from tools import sample_media
    expected = _fixture_manifest(sample_media)
    source, target = tmp_path / "source", tmp_path / "target"
    for name in sample_media.FILES:
        path = source / "app/sample_real" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    bundle = tmp_path / "private.zip"
    sample_media.pack(source, bundle, expected)
    sample_media.unpack(target, bundle, expected)
    assert all((target / "app/sample_real" / name).read_bytes() == name.encode() for name in sample_media.FILES)
    with pytest.raises(RuntimeError, match="python tools/sample_media.py unpack"):
        sample_media.unpack(tmp_path / "empty", tmp_path / "missing.zip", expected)
    outside = tmp_path / "outside"
    outside.mkdir()
    unsafe_root = tmp_path / "unsafe" / "app/sample_real"
    unsafe_root.mkdir(parents=True)
    (unsafe_root / "frames").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe sample media path"):
        sample_media.unpack(tmp_path / "unsafe", bundle, expected)
    assert not list(outside.iterdir())


def test_missing_clip_invalidates_installed_sample(client, synthetic_sample_bundle, tmp_path, monkeypatch, caplog):
    import shutil
    bundle = tmp_path / "bundle"
    shutil.copytree(synthetic_sample_bundle, bundle)
    monkeypatch.setattr(service, "_sample_bundle", lambda settings: bundle)
    assert client.get("/guide/sample", follow_redirects=False).status_code == 303
    (bundle / "sample.mp4").rename(bundle / "clip-missing")
    response = client.get("/guide/sample")
    assert response.status_code == 200 and "Sample guide: fan cable reseat" in response.text
    assert "sample media missing: sample.mp4" in caplog.text


def _manifest_rows():
    from tools import sample_media
    return [f"{'a' * 64}  10  {name}" for name in sample_media.FILES]


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r + [r[0]], r"line 20: duplicate name sample\.mp4"),
    (lambda r: [r[0].replace("a" * 64, "not-a-hash")] + r[1:], r"line 2: digest is not 64 hex"),
    (lambda r: [r[0].replace("a" * 64, "a" * 63)] + r[1:], r"line 2: digest is not 64 hex"),
    (lambda r: [r[0].replace("  10  ", "  0  ")] + r[1:], r"line 2: size is not a positive integer"),
    (lambda r: [r[0].replace("  10  ", "  -5  ")] + r[1:], r"line 2: size is not a positive integer"),
    (lambda r: [r[0].replace("  10  ", "  1.5  ")] + r[1:], r"line 2: size is not a positive integer"),
    (lambda r: r[:1] + [""] + r[1:], r"line 3: expected"),
    (lambda r: r[:1] + ["garbage"] + r[1:], r"line 3: expected"),
    (lambda r: r[:-1], r"missing \['labels/step-08-part-2\.mask\.png'\]"),
    (lambda r: r + [f"{'b' * 64}  10  extra.bin"], r"unexpected \['extra\.bin'\]"),
    (lambda r: r[:-1] + [f"{'b' * 64}  10  unapproved.bin"],
     r"missing \['labels/step-08-part-2\.mask\.png'\].*unexpected \['unapproved\.bin'\]"),
])
def test_sample_media_manifest_rejects_malformed_rows(tmp_path, mutate, match):
    from tools import sample_media
    path = tmp_path / "m.sha256"
    path.write_text("# header\n" + "\n".join(mutate(_manifest_rows())) + "\n")
    with pytest.raises(ValueError, match=match):
        sample_media.manifest(path)


def test_sample_media_manifest_accepts_a_well_formed_file(tmp_path):
    from tools import sample_media
    path = tmp_path / "m.sha256"
    path.write_text("# header\n" + "\n".join(_manifest_rows()) + "\n")
    assert len(sample_media.manifest(path)) == 18


def _cards(html: str) -> dict[str, str]:
    """Card id -> its markup, in page order."""
    return {m.group(1): m.group(0) for m in re.finditer(r'<article class="card[^"]*" id="([^"]+)">.*?</article>', html, re.S)}


def test_intro_is_context_not_a_numbered_step(client: TestClient) -> None:
    page = client.get("/guide/sample", follow_redirects=True).text
    cards = _cards(page)
    assert list(cards) == ["intro", *(f"step-{n}" for n in range(1, 9))]  # intro first, steps 1-8 after it
    intro = cards["intro"]
    assert "Intro showcase laptop" in intro and "Step 0" not in intro and 'class="gd-no"' not in intro
    assert 'class="tools"' not in intro and "screwdriver" not in intro  # the footage shows no tool in use
    assert 'data-start="0.0"' in intro  # the clip still plays from 0
    assert page.count('class="gd-no"') == 8 == int(re.search(r"(\d+) steps ·", page).group(1))
    assert "Step 1</h2>" in cards["step-1"] and "Flip over and remove external battery" in cards["step-1"]
    assert "Intro showcase" not in "".join(v for k, v in cards.items() if k != "intro")


def test_old_step_links_still_resolve(client: TestClient) -> None:
    for old in range(1, 10):  # fragments never reach the server: the old URL is the page itself
        assert client.get(f"/guide/sample#step-{old}", follow_redirects=True).status_code == 200
    page = client.get("/guide/sample", follow_redirects=True).text
    cards = _cards(page)
    hard_stop = "/guide/sample#step-5"  # was #step-6 before the intro moved out
    assert tomllib.loads((ROOT / "config" / "proof_links.toml").read_text())["hard_stop"] == hard_stop
    assert "Not shown in the video: the battery disconnect" in cards["step-5"]
    assert 'href="#step-5"' in page  # the "Before you start" list points at the renumbered step
    notes = json.loads((BUNDLE / "step_notes.json").read_text())
    assert sorted(notes) == ["5", "8"]  # the two footage notes moved with their steps


def test_visual_change_label_is_explained_once_where_it_first_appears(client: TestClient) -> None:
    page = client.get("/guide/sample", follow_redirects=True).text
    cards = _cards(page)
    note = "image pixels for this part were not changed. It does not confirm the label is correct."
    assert page.count(note) == 1 and note in cards["step-5"]  # step 5 holds the first label
    assert all("No visual changes to this part" not in cards[k] for k in ("intro", "step-1", "step-2", "step-3", "step-4"))
    assert "Checked" not in page.replace("Not checked", "")  # no new "Checked" badge
    assert 'class="pf-hazard"' in cards["step-5"] and "Not shown in the video" in cards["step-5"]  # the stop stays prominent
    assert "check time" in cards["step-1"]  # approximate-time warnings stay


def test_guide_json_numbers_follow_the_intro() -> None:
    steps = json.loads((BUNDLE / "guide.json").read_text())["steps"]
    assert [(s["number"], s["kind"]) for s in steps[:2]] == [(0, "intro"), (1, "repair")]
    assert [s["number"] for s in steps[1:]] == list(range(1, 9)) and steps[0]["tools"] == []
    assert steps[0]["start_s"] == 0 and steps[0]["end_s"] == steps[1]["start_s"] == 9.4  # timing untouched


def test_images_zip_names_start_at_step_1(client: TestClient) -> None:
    job_id = service.ensure_sample_job(client.app.state.settings)
    token = re.search(r'/images.zip\?token=([^"&]+)', client.get(f"/guides/{job_id}").text).group(1)
    r = client.get(f"/guides/{job_id}/images.zip?token={token}")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names == [f"step-{n:02}.png" for n in range(1, 9)]  # the intro frame is not a numbered image
