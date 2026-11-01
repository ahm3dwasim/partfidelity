"""config/film.toml loads strictly: a missing or misspelled key, or a bad value, fails loudly."""
import pathlib
import re

import pytest

from app import film_config

SRC = film_config.CONFIG.read_text()


def _load(tmp_path, text):
    path = tmp_path / "film.toml"
    path.write_text(text)
    return film_config.load(path)


def test_shipped_config_loads_and_chapters_follow_the_rise_order():
    cfg = film_config.load()
    assert len(cfg["chapters"]) == 3
    # Parts start rising 10 frames apart from frame 44 (config/teardown.toml).
    assert [p["frame"] for p in cfg["parts"]][:2] == [44, 52]
    public = film_config.public(cfg)
    assert public["poster_url"].endswith("poster.webp?v=" + cfg["film"]["version"])


def test_labels_leave_as_each_part_starts_back_in():
    """a part's label goes on the frame the render starts to put that part back (teardown.toml [reassemble])."""
    import sys
    sys.path[:0] = ["tools", "assets/3d/teardown"]
    import pose
    from teardown_config import load
    tc = load()
    for p in film_config.load()["parts"]:
        f = p["leave"]
        assert pose.part_progress(tc, f)[p["id"]] == (1.0 if p["id"] in tc["explode"]["rise"] else 0.0) or p["id"] == "bottom_cover"
        if p["id"] in tc["explode"]["rise"]:
            assert pose.part_progress(tc, f - 1)[p["id"]] == 1.0 and pose.part_progress(tc, f + 3)[p["id"]] < 1.0, p["id"]


def test_every_labelled_part_is_a_part_of_the_teardown():
    """labels name real parts; the anchors the render writes are keyed by the same ids."""
    import tomllib

    teardown = tomllib.loads((film_config.CONFIG.parent / "teardown.toml").read_text())["explode"]
    ids = [p["id"] for p in film_config.load()["parts"]]
    assert set(ids) <= set(teardown["order"])
    # Rising parts carry the frame they start to rise (8 apart from 44, in rise order); the bottom case comes last.
    assert ids[: len(teardown["rise"])] == teardown["rise"]
    assert ids[len(teardown["rise"]):] == ["bottom_cover"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("frames = 250", "framez = 250"),
        ("smoothing = 0.14", "smoothing = 2"),
        ("passes = [8, 4, 1]", "passes = [8, 4]"),
        ("start = 60\nend = 84", "start = 66\nend = 84"),
        ('max_css_width = 0\nmax_device_width = 0', 'max_css_width = 900\nmax_device_width = 0'),
        ('name = "Lid back"', 'name = ""'),
        ('name = "Keyboard"\nchapter = 2', 'name = "Keyboard"\nchapter = 4'),
        ('name = "Lid back"\nchapter = 1', 'name = "Lid back"\nchapter = 3'),
        ("phone_art_width_vw = 108", "phone_art_width_vw = 20"),
        # frame pattern: only the grammar film.js reads
        ('frame_pattern = "{n:04d}.webp"', 'frame_pattern = "{nope}.webp"'),
        ('frame_pattern = "{n:04d}.webp"', 'frame_pattern = "{n:4d}.webp"'),
        ('frame_pattern = "{n:04d}.webp"', 'frame_pattern = "{n}-{n}.webp"'),
        ('frame_pattern = "{n:04d}.webp"', 'frame_pattern = "a/{n}.webp"'),
        ('frame_pattern = "{n:04d}.webp"', 'frame_pattern = "frame.webp"'),
        # passes strictly falling, ending at 1, first one below frames
        ("passes = [8, 4, 1]", "passes = [4, 8, 1]"),
        ("passes = [8, 4, 1]", "passes = [8, 8, 1]"),
        ("passes = [8, 4, 1]", "passes = [400, 1]"),
        # frames inside 1..frames
        ("opening_fade = [20, 40]", "opening_fade = [20, 251]"),
        ("frame = 108", "frame = 251"),
        ("leave = 150\n", "leave = 40\n"),
        ('name = "Speakers"\nchapter = 3', 'name = "Speakers"\nchapter = 4'),
        # chapters inside bounds and ordered
        ("start = 84\nend = 250", "start = 84\nend = 251"),
        ("start = 30\nend = 60", "start = 60\nend = 30"),
        # captions tile the film: no gap between chapters, the last runs to the end, chapter 1 starts by the
        # middle of the opening fade and covers the closed laptop
        ("start = 30\nend = 60", "start = 30\nend = 56"),
        ("start = 84\nend = 250", "start = 84\nend = 200"),
        ("start = 30\nend = 60", "start = 31\nend = 60"),
        ("opening_fade = [20, 40]", "opening_fade = [10, 40]"),
        # captions change with the stage: chapter 2 and 3 start on the frame their first part rises
        ("end = 84\nname", "end = 86\nname"),
        ("start = 60\nend = 84", "start = 62\nend = 84"),
        ('name = "Keyboard"\nchapter = 2', 'name = "Keyboard"\nchapter = 1'),
        # part labels: safe unique ids, a served anchors name, sane spacing, a known side rule
        ('id = "keyboard"', 'id = "lid_back"'),
        ('id = "keyboard"', 'id = "../keyboard"'),
        ('anchors = "anchors.json"', 'anchors = "../anchors.json"'),
        ("label_land = 0.75", "label_land = 0"),
        ("label_spacing_px = 22", "label_spacing_px = 0"),
        ("label_gap_px = 14", "label_gap_px = -1"),
        ('labels = "alternate"', 'labels = "left"'),
        ("caption_fade_ms = 260", "caption_fade_ms = -1"),
        ('cta_text = "Explore the sample"', 'cta_text = ""'),
        ('cta_url = "/guide/sample"', 'cta_url = "https://example.com/"'),
        ('name = "The screen"', 'name = ""'),
        # set breakpoints ordered, layout known, earlier sets limited
        ('max_css_width = 820\nmax_device_width = 1000', 'max_css_width = 0\nmax_device_width = 1000'),
        ('layout = "phone"', 'layout = "tablet"'),
        ('dir = "webp-900"', 'dir = "../x"'),
        ('dir = "webp-900"', 'dir = "webp-1600"'),
        # bundles: size at least 1, one {i} in a path-safe name, frame numbers fit the record's uint16
        ("bundle_frames = 40", "bundle_frames = 0"),
        ('bundle_name = "bundle-{i}.bin"', 'bundle_name = "bundle.bin"'),
        ('bundle_name = "bundle-{i}.bin"', 'bundle_name = "../bundle-{i}.bin"'),
        ('bundle_name = "bundle-{i}.bin"', 'bundle_name = "bundle-{n}.bin"'),
        ("frames = 250", "frames = 65536"),
        ("bitmap_window = 12", "bitmap_window = -1"),
        ("max_pending_decodes = 4", "max_pending_decodes = 0"),
        ("max_frame_bytes = 4194304", "max_frame_bytes = 11"),
    ],
)
def test_bad_config_fails(tmp_path, old, new):
    assert old in SRC
    with pytest.raises(film_config.FilmConfigError):
        _load(tmp_path, SRC.replace(old, new, 1))


def test_set_breakpoints_must_rise(tmp_path):
    third = '[[sets]]\nname = "mid"\ndir = "webp-1200"\nmax_css_width = 700\nmax_device_width = 1400\nlayout = "wide"\nlabels = "right"\n\n'
    text = SRC.replace('[[sets]]\nname = "wide"', third + '[[sets]]\nname = "wide"', 1)
    with pytest.raises(film_config.FilmConfigError, match="max_css_width must rise"):
        _load(tmp_path, text)
    assert len(_load(tmp_path, text.replace("max_css_width = 700", "max_css_width = 1200"))["sets"]) == 3


def test_chapters_render_from_config(tmp_path):
    from app.routes.site import _chapters_html

    cfg = _load(tmp_path, SRC.replace('name = "The keyboard"', 'name = "The keys & caps"').replace(
        'title = "Find the right part."', 'title = "A <new> title"'))
    html = _chapters_html(cfg["chapters"])
    assert html.count('<div class="film-chap" data-chap>') == 3
    assert '<p class="film-n">02 / The keys &amp; caps</p>' in html
    assert "<h2>A &lt;new&gt; title</h2>" in html
    assert "Meta&#8217;s SAM 3.1" in html
    assert html.count("cta-row") == 1 and 'href="/guide/sample">Explore the sample</a>' in html


def test_public_chapters_carry_ranges_only():
    assert film_config.public(film_config.load())["chapters"][0] == {"start": 30, "end": 60}


def test_unknown_key_fails(tmp_path):
    with pytest.raises(film_config.FilmConfigError):
        _load(tmp_path, SRC.replace("[film]\n", "[film]\nextra = 1\n", 1))


def test_no_literal_colours_in_the_home_template():
    html = (film_config.CONFIG.parents[1] / "templates/index.html").read_text()
    assert not re.search(r"(?<!&)#[0-9a-fA-F]{3,6}\b", html.replace("%23", "").split("<body")[1])


def test_frame_archives_hold_exactly_the_frames_and_unpack_is_idempotent(tmp_path):
    import tarfile

    from tools import film_frames

    cfg = film_config.load()
    want = sorted(cfg["film"]["frame_pattern"].format(n=n) for n in range(1, cfg["film"]["frames"] + 1))
    assert want[0] == "0001.webp" and want[-1] == "0250.webp" and len(want) == 250
    for s in cfg["sets"]:
        with tarfile.open(film_frames.ROOT / cfg["archives"]["source"].format(dir=s["dir"])) as tar:
            assert sorted(tar.getnames()) == [f"{s['dir']}/{n}" for n in want]
    (tmp_path / "assets").symlink_to(film_frames.ROOT / "assets")
    (tmp_path / "config").mkdir()
    # A distinct config at the alternate root: unpack must follow its target, not the tool checkout's.
    (tmp_path / "config/film.toml").write_text(SRC.replace('target = "static/ui/media/teardown"', 'target = "static/alt/frames"', 1))
    plan = film_config.bundles(cfg["film"])
    # A single frame left by an older build is removed: the folder holds exactly what the page fetches.
    stale = tmp_path / "static/alt/frames/webp-900/0001.webp"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    assert film_frames.unpack(tmp_path) == 2 * len(plan) + 1  # both sets' bundles and the anchors
    assert not stale.exists()
    served = __import__("json").loads((tmp_path / "static/alt/frames" / cfg["film"]["anchors"]).read_text())
    assert sorted(served, key=int) == [str(n) for n in range(1, 251)]
    assert all(sorted(row) == sorted(p["id"] for p in cfg["parts"]) for row in served.values())
    assert not (tmp_path / "static/ui").exists()
    assert film_frames.unpack(tmp_path) == 0
    # A same-size corrupt bundle is replaced; no temp sibling is left behind.
    bundle = tmp_path / "static/alt/frames/webp-900" / plan[2][0]
    data = bytearray(bundle.read_bytes())
    data[-1] ^= 0xFF
    bundle.write_bytes(bytes(data))
    assert film_frames.unpack(tmp_path) == 1
    assert bundle.read_bytes()[-1] == data[-1] ^ 0xFF
    assert not list((tmp_path / "static/alt/frames").rglob("*.tmp"))


def test_bundle_plan_keeps_pass_order_and_size(tmp_path):
    film = _load(tmp_path, SRC.replace("bundle_frames = 40", "bundle_frames = 30", 1))["film"]
    plan = film_config.bundles(film)
    flat = [n for _, nums in plan for n in nums]
    assert sorted(flat) == list(range(1, 251))
    assert plan[0] == ("bundle-0.bin", [*range(1, 248, 8), 250][:30])
    assert all(0 < len(nums) <= 30 for _, nums in plan)
    # A bundle never mixes passes: the coarse pass is whole before any finer frame arrives.
    pass_of = lambda n: 0 if n == 250 or n % 8 == 1 else 1 if n % 4 == 1 else 2
    assert all(len({pass_of(n) for n in nums}) == 1 for _, nums in plan)
    assert [pass_of(n) for n in flat] == sorted(pass_of(n) for n in flat)
    assert film_config.public({**_load(tmp_path, SRC), "film": film})["bundles"] == [b for b, _ in plan]


def test_unpack_refuses_an_archive_with_a_missing_or_extra_frame(tmp_path):
    import tarfile

    from tools import film_frames

    film = film_config.load()["film"]
    tar = tmp_path / "t.tar"
    with tarfile.open(tar, "w") as out:
        info = tarfile.TarInfo("webp-900/0001.webp")
        info.size = 3
        out.addfile(info, __import__("io").BytesIO(b"abc"))
    with pytest.raises(ValueError, match="249 frames missing"):
        film_frames.read_frames(tar, film, "webp-900")
    with tarfile.open(tar, "a") as out:
        info = tarfile.TarInfo("webp-900/../x.webp")
        info.size = 0
        out.addfile(info)
    with pytest.raises(ValueError, match="unexpected archive member"):
        film_frames.read_frames(tar, film, "webp-900")


def test_unpack_refuses_anchors_missing_a_frame_or_a_labelled_part(tmp_path):
    """the labels read the render's anchors for every frame and every [[parts]] id; a file short of either fails
    the build instead of shipping a film whose labels point at nothing."""
    import json

    from tools import film_frames

    cfg = film_config.load()
    src = json.loads((film_frames.ROOT / cfg["archives"]["anchors"]).read_text())
    ids = [p["id"] for p in cfg["parts"]]
    path = tmp_path / "a.json"
    ok = film_frames.read_anchors(film_frames.ROOT / cfg["archives"]["anchors"], cfg["film"], ids)
    assert ok["200"]["battery"] == [round(x, 3) for x in src["200"]["battery"]] and "feet" not in ok["200"]
    for broken, match in [
        ({k: v for k, v in src.items() if k != "131"}, "frames 1..250"),
        ({**src, "131": {k: v for k, v in src["131"].items() if k != "bottom_cover"}}, "lacks"),
        ({**src, "131": {**src["131"], "battery": [0.5, 0.5, "x", 0.5]}}, "bad anchor"),
    ]:
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=match):
            film_frames.read_anchors(path, cfg["film"], ids)


@pytest.mark.parametrize("key,bad", [
    ("target", "/tmp/unrelated"), ("target", "static/../../unrelated"), ("target", "static/./x"), ("target", "elsewhere/frames"),
    ("target", "static//x"), ("target", "static"), ("anchors", "/etc/passwd"), ("anchors", "assets/../../x.json"),
    ("source", "/abs/teardown-{dir}.tar"), ("source", "assets/../x/{dir}.tar"), ("source", "static/{dir}.tar"),
])
def test_archive_paths_must_be_normalized_relative_and_under_the_approved_roots(tmp_path, key, bad):
    text = re.sub(rf'^{key} = "[^"]*"', f'{key} = "{bad}"', SRC, count=1, flags=re.M)
    assert text != SRC
    with pytest.raises(film_config.FilmConfigError):
        _load(tmp_path, text)


def test_unpack_rechecks_paths_and_never_follows_a_symlink_out(tmp_path):
    from tools import film_frames

    outside = tmp_path / "outside"
    outside.mkdir()
    keep = outside / "precious.txt"
    keep.write_text("x")
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    (root / "assets").symlink_to(film_frames.ROOT / "assets")
    (root / "static").symlink_to(outside)             # the approved root itself leads out of the repo
    (root / "config/film.toml").write_text(SRC)
    with pytest.raises(ValueError):
        film_frames.unpack(root)
    assert keep.read_text() == "x" and sorted(p.name for p in outside.iterdir()) == ["precious.txt"]
    with pytest.raises(ValueError):
        film_frames._safe(root, "static/../x", "static")
    with pytest.raises(ValueError):
        film_frames._safe(root, "/tmp/x", "static")


def test_concurrent_unpackers_do_not_trip_over_each_others_files(tmp_path):
    import subprocess
    import sys

    from tools import film_frames

    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    (root / "assets").symlink_to(film_frames.ROOT / "assets")
    (root / "config/film.toml").write_text(SRC.replace('target = "static/ui/media/teardown"', 'target = "static/stress"', 1))
    cmd = [sys.executable, "-I", str(film_frames.ROOT / "tools/film_frames.py"), "unpack", "--root", str(root)]
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(16)]
    results = [(p.wait(timeout=120), p.stderr.read()) for p in procs]
    assert [r for r in results if r[0]] == [], results
    out = root / "static/stress"
    plan = film_config.bundles(film_config.load()["film"])
    assert sorted(f.name for f in (out / "webp-900").iterdir()) == sorted(n for n, _ in plan)
    assert not list(out.rglob("*.tmp"))


def test_guide_tunables_are_validated(tmp_path):
    source = film_config.CONFIG.read_text()
    for old, new in [('toast_ms = 2200', 'toast_ms = 0'),
                     ('teaser_steps = 3', 'teaser_steps = 0'),
                     ('sample_cta = "Explore the sample"', 'sample_cta = ""'),
                     ('observer_bottom_pct = 55', 'observer_bottom_pct = 70'),
                     ('observer_hold_ms = 300', 'observer_hold_ms = -1')]:
        candidate = tmp_path / 'film.toml'
        candidate.write_text(source.replace(old, new))
        with pytest.raises(film_config.FilmConfigError):
            film_config.load(candidate)
