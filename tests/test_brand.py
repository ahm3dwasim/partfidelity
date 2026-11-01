"""The PF mark: flat when small, real (satin steel wrench) from real_min_px up."""
from pathlib import Path
import ast
import re
import tomllib
import xml.etree.ElementTree as ET

import pytest

from app import brand_config
from app.brand_config import BrandConfigError

ROOT = Path(__file__).resolve().parents[1]
USE = re.compile(r'<svg[^>]*?\bwidth="(\d+)"[^>]*>\s*<use href="([^"]*icon[^"]*\.svg#[^"]+)"')


def test_config_loads_and_both_targets_exist():
    cfg = brand_config.load()
    for key in ("flat", "real"):
        path, frag = cfg[key].split("#")
        root = ET.parse(ROOT / path.lstrip("/")).getroot()
        assert any(el.attrib.get("id") == frag for el in root.iter()), cfg[key]


@pytest.mark.parametrize("bad", [
    {"real_min_px": 48, "flat": "/static/a.svg#a"},
    {"real_min_px": "48", "flat": "/static/a.svg#a", "real": "/static/b.svg#b"},
    {"real_min_px": 48, "flat": "a.svg#a", "real": "/static/b.svg#b"},
    {"real_min_px": 48, "flat": "/static/a.svg#a", "real": "/static/b.svg#b", "x": 1},
])
def test_loader_fails_loudly(bad):
    with pytest.raises(BrandConfigError):
        brand_config.validate(bad)


def filled(page: Path) -> str:
    return brand_config.fill(page.read_text(encoding="utf-8"))


def test_every_template_mark_follows_the_size_rule():
    cfg = brand_config.load()
    seen = 0
    for page in (ROOT / "templates").rglob("*.html"):
        for width, href in USE.findall(filled(page)):
            want = cfg["real"] if int(width) >= cfg["real_min_px"] else cfg["flat"]
            assert href == want, f"{page.name}: {width}px uses {href}"
            seen += 1
    assert seen >= 5


@pytest.mark.parametrize("real_min_px", [16, 24, 25, 64, 100])
def test_placement_follows_the_configured_threshold_on_both_sides(real_min_px, monkeypatch):
    cfg = dict(brand_config.load(), real_min_px=real_min_px)
    monkeypatch.setattr(brand_config, "load", lambda: cfg)
    assert brand_config.fill("@@mark:%d@@" % (real_min_px - 1), cfg) == cfg["flat"]
    assert brand_config.fill("@@mark:%d@@" % real_min_px, cfg) == cfg["real"]
    assert brand_config.fill("@@mark_file:%d@@" % real_min_px, cfg) == cfg["real"].split("#")[0]
    from app.brand_templates import make_env

    header = make_env(ROOT / "templates").loader.get_source(make_env(ROOT / "templates"), "partials/ui/shell-header.html")[0]
    assert cfg["real" if 24 >= real_min_px else "flat"] in header          # the 24px header mark moves with the threshold
    index = make_env(ROOT / "templates").loader.get_source(make_env(ROOT / "templates"), "index.html")[0]
    assert cfg["real" if 96 >= real_min_px else "flat"] in index           # and so does the 96px closing mark


def test_templates_do_not_pick_flat_or_real_by_hand():
    for page in [*(ROOT / "templates").rglob("*.html"), *(ROOT / "static" / "ui" / "gallery").glob("*.html")]:
        raw = page.read_text(encoding="utf-8")
        assert not re.search(r"@@(flat|real)(_file)?@@", raw), page.name


def test_size_slots_reject_bad_forms():
    for bad in ("@@mark@@", "@@favicon:24@@", "@@flat@@", "@@mark_file@@"):
        with pytest.raises(BrandConfigError):
            brand_config.fill(bad)


def test_pillow_is_a_declared_test_dependency():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]["test"]
    assert any(re.match(r"pillow\b", d, re.I) for d in extras)


def test_touch_icon_appearance_comes_from_the_config():
    good = brand_config.load()
    assert 0 <= good["touch_icon_pad"] < 0.5 and good["touch_icon_bg"].startswith("#")
    src = (ROOT / "tools" / "brand_touch_icon.py").read_text(encoding="utf-8")
    assert "0.14" not in src and "#0f0f0f" not in src
    for bad in ({"touch_icon_pad": 0.5}, {"touch_icon_pad": -0.1}, {"touch_icon_pad": True}, {"touch_icon_bg": "black"}, {"touch_icon_bg": "#fff"}):
        with pytest.raises(BrandConfigError):
            brand_config.validate({**good, **bad})


def test_real_mark_is_the_render_between_the_letters():
    svg = (ROOT / "static" / "ui" / "icon-real.svg").read_text(encoding="utf-8")
    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 32 32"
    assert svg.count('id="pf-mark-real"') == 1
    assert "var(--pf-mark-gap" in svg                      # the cut-out gap follows the page
    ns = "{http://www.w3.org/2000/svg}"
    images = list(root.iter(ns + "image"))
    assert len(images) == 1 and images[0].attrib["href"].startswith("data:image/webp;base64,")
    kids = [k.tag for k in root.find(f".//{ns}g[@id='pf-mark-real']")]
    assert kids == [ns + "g", ns + "image", ns + "g"], "order: P, wrench, F"
    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "").replace("http://www.w3.org/1999/xlink", "")


def test_wrench_exports_exist_in_three_sizes_with_alpha():
    from PIL import Image

    base = ROOT / "static" / "ui" / "wrench-real"
    heights = []
    for k in (1, 2, 3):
        for ext in ("png", "webp"):
            im = Image.open(base / f"wrench-{k}x.{ext}")
            assert im.mode == "RGBA" and im.getchannel("A").getextrema()[0] == 0, f"{k}x.{ext} needs a transparent background"
        heights.append(Image.open(base / f"wrench-{k}x.png").height)
    assert all(abs(h - k * heights[0]) <= 3 for k, h in enumerate(heights, 1)), heights   # 1x, 2x, 3x of one size


def test_every_brand_file_ships_in_the_modal_image():
    deploy = tomllib.loads((ROOT / "deploy" / "modal_deploy.toml").read_text(encoding="utf-8"))["modal"]
    cfg = brand_config.load()
    brand = {"config/brand.toml", "app/brand_config.py", "app/brand_templates.py"}
    brand |= {cfg[k].split("#")[0].lstrip("/") for k in ("flat", "real", "touch_icon", "favicon")}
    brand |= {str(p.relative_to(ROOT)) for p in (ROOT / "static" / "ui" / "wrench-real").iterdir()}
    for path in sorted(brand):
        assert path in deploy["image_files"] or any(path.startswith(d + "/") for d in deploy["image_folders"]), f"{path} is not in the image"
    assert cfg["touch_icon"].lstrip("/") in deploy["image_files"], "the touch icon is listed by name in image_files"


PAGES = sorted([*(ROOT / "templates").rglob("*.html"), *(ROOT / "static" / "ui" / "gallery").glob("*.html")])
def test_heads_link_the_configured_icon_and_touch_icon():
    cfg = brand_config.load()
    for page in PAGES:
        text = filled(page)
        if not re.search(r"<head[ >]", text):
            continue                                       # a partial, not a page
        icons = {rel: href for rel, href in re.findall(r'<link rel="([^"]*icon[^"]*)" href="([^"]+)"', text)}
        assert icons == {"icon": cfg["favicon"], "apple-touch-icon": cfg["touch_icon"]}, f"{page.name}: {icons}"


def test_touch_icon_is_the_approved_mark_at_the_configured_size():
    from PIL import Image

    cfg = brand_config.load()
    im = Image.open(ROOT / cfg["touch_icon"].lstrip("/"))
    assert im.size == (cfg["touch_icon_px"],) * 2 and im.mode == "RGB"
    assert len({im.getpixel((x, y)) for x in range(0, im.width, 6) for y in range(0, im.height, 6)}) > 20   # a drawn mark, not a blank tile


def test_changing_brand_toml_changes_every_render_path(monkeypatch):
    cfg = dict(brand_config.load(), flat="/static/ui/x.svg#x-flat", favicon="/static/ui/x-fav.svg", touch_icon="/static/ui/x-touch.png")
    monkeypatch.setattr(brand_config, "load", lambda: cfg)
    from app.brand_templates import make_env
    from app.routes import site

    env = make_env(ROOT / "templates")
    for name in ("upload.html", "partials/ui/shell-header.html"):
        src = env.loader.get_source(env, name)[0]
        assert "x-flat" in src or "x-fav" in src, name
    assert "x-touch.png" in env.loader.get_source(env, "upload.html")[0]
    assert "x-fav.svg" in site._template("index.html").template
    assert "x-flat" in site._template("reader.html").template


def test_gallery_pages_are_served_from_brand_toml(monkeypatch):
    from fastapi.testclient import TestClient
    from app.routes import site

    cfg = dict(brand_config.load(), touch_icon="/static/ui/x-touch.png")
    monkeypatch.setattr(brand_config, "load", lambda: cfg)
    r = TestClient(site.create_site_app()).get("/static/ui/gallery/shell.html")
    assert r.status_code == 200 and "x-touch.png" in r.text and "@@" not in r.text


def test_unknown_token_and_bad_touch_icon_fail_loudly():
    with pytest.raises(BrandConfigError):
        brand_config.fill("@@favicn@@")
    for token in ("@@mark:abc@@", "@@mark_file:abc@@", "@@mark:24x@@", "@@mark:0@@", "@@mark_file:0@@", "@@mark:-3@@", "@@mark@@", "@@mark:@@", "@@mark: 24@@", "@@mark:\t24@@", "@@mark:\n24@@", "@@@@", "@@mark:@@mark:24@@@@", "@@"):
        with pytest.raises(BrandConfigError):
            brand_config.fill(token)
    good = brand_config.load()
    for bad in ({"touch_icon": "/static/a.svg"}, {"touch_icon": "/other/a.png"}, {"touch_icon_px": 0}, {"touch_icon_px": 180.5},
                {"touch_icon_px": True}, {"real_min_px": True}, {"inline_margin_px": -1}, {"header_class": "x\" bad"}, {"gallery_sizes": [16, 24]}, {"flat": "/static/../x.svg#a"}, {"favicon": "/static/a.png"}):
        with pytest.raises(BrandConfigError):
            brand_config.validate({**good, **bad})
    with pytest.raises(BrandConfigError):
        brand_config.validate("not a dict")


# The source rule covers 15 known bypass shapes as well as real pages.
BYPASS_CORPUS = ['<img class="pf-brand-icon" src="/static/legacy-brand.png">', '<svg viewBox="0 0 32 32" width="24"><use href="/other/icon.svg#wrong-mark"></use></svg>', '<span class="pf-brand-icon"><img src="/static/legacy.png"></span>', '<svg class="pf-brand-icon" viewBox="0 0 24 24"><path d="M0 0h8v8z"/></svg>', '<svg viewBox="0 0 32 32"><use href="@@mark:32@@"></use><text>PF</text></svg>', '<svg viewBox="0 0 32 32"><desc><use href="@@mark:32@@"></use></desc></svg>', '<svg viewBox="0 0 32 32"><use href="/static/ui/icon.svg"></use></svg>', '<link rel="icon" href="@@mark_file:32@@">', '<svg viewBox="0 0 32 32"><use href="@@mark:32@@" style="display:none"></use></svg>', '<svg viewBox="0 0 32 32"><use href="@@mark:32@@" visibility="hidden"></use></svg>', '<svg viewBox="0 0 32 32"><use href="@@mark:32@@" opacity="0"></use></svg>', '<svg viewBox="0 0 32 32"><use href="@@mark:32@@" transform="scale(0)"></use></svg>', '<svg viewBox="0 0 32 32" style="display:none"><use href="@@mark:32@@"></use></svg>', '<svg class="pf-brand-icon" viewBox="0 0 32 32"><use class="invisible" href="@@mark:32@@"></use></svg>', '<svg class="end-mark" viewBox="0 0 64 64"><circle cx="8" cy="8" r="4"/></svg>']


def source_mark_markup(text):
    # The emitter owns use elements, mark classes, icon links and brand asset URLs.
    patterns = (
        r"<use\b", r"<[^>]+(?:pf-brand-icon|end-mark)",
        r"<link\b[^>]*\brel\s*=\s*(?:[\"'][^\"']*icon[^\"']*[\"']|[^\s>]*icon)",
        r"<(?:img|svg|link)\b[^>]*(?:icon|favicon|logo|mark|brand)[^>]*\.(?:svg|png|webp)",
        r"url\s*\([^)]*(?:icon|favicon|logo|mark|brand)[^)]*\)",
        r"@@(?:mark|mark_file|favicon|touch_icon)(?::|@@)",
    )
    if any(re.search(pattern, text, re.I | re.S) for pattern in patterns):
        return True
    cfg = brand_config.load()
    assets = [Path(cfg[key].split("#")[0]).name for key in ("flat", "real", "favicon", "touch_icon")]
    if any(re.search(r"<[^>]+" + re.escape(cfg[key]), text, re.I | re.S) for key in ("header_class", "footer_class")):
        return True
    return any(re.search(r"<(?:img|svg|link)\b[^>]*" + re.escape(name), text, re.I | re.S)
               or re.search(r"url\s*\([^)]*" + re.escape(name), text, re.I | re.S) for name in assets)


@pytest.mark.parametrize("snippet", BYPASS_CORPUS)
def test_source_rule_rejects_reviewed_bypasses(snippet):
    assert source_mark_markup(snippet)


def test_marks_are_emitted_as_complete_markup():
    for page in PAGES:
        assert not source_mark_markup(page.read_text(encoding="utf-8")), page
    assert '<use ' in brand_config.mark_badge()
    assert '<link ' in brand_config.icon_links()
    for module in (ROOT / "app").rglob("*.py"):
        if module.name == "brand_config.py":
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not source_mark_markup(node.value), (module, node.lineno)
    for css in (ROOT / "static").rglob("*.css"):
        assert not source_mark_markup(css.read_text(encoding="utf-8")), css
    for snippet in ('<link href="/x.svg" rel=icon>', '<img src="/static/ui/icon.svg">',
                    '<svg class="end-mark"></svg>', 'background:url(/static/ui/favicon.svg)'):
        assert source_mark_markup(snippet), snippet


def test_rendered_routes_have_emitter_markup():
    from fastapi.testclient import TestClient
    from app.routes import site, steps, uploads, guides, calls, how
    from app import reader
    from types import SimpleNamespace
    app = site.create_site_app()
    app.include_router(steps.router)
    app.include_router(uploads.router)
    app.include_router(guides.router)
    app.include_router(calls.router)
    app.include_router(how.router)
    app.dependency_overrides[calls.resolve_job_context] = lambda: SimpleNamespace(job_id="job-brand")
    guides.set_guide_loader(app, lambda job_id: reader.sample_guide())
    client = TestClient(app)
    urls = ['/', '/reader/sample', '/reader/hero', '/upload', '/delete-request', '/how-it-works', '/steps/preview', '/guides/job-guide', '/guide/job-guide', '/calls/job-brand'] + [
        '/static/ui/gallery/' + p.name for p in (ROOT / 'static/ui/gallery').glob('*.html')
    ]
    count = 0
    for url in urls:
        response = client.get(url)
        assert response.status_code == 200, url
        assert brand_config.icon_links() in response.text, url
        remainder = response.text
        emitters = [brand_config.icon_links(), brand_config.mark_badge(), brand_config.footer_mark(), brand_config.inline_mark()]
        emitters += [brand_config.gallery_mark(i) for i in range(4)]
        for markup in emitters:
            count += remainder.count(markup) * (2 if markup == brand_config.icon_links() else 1)
            remainder = remainder.replace(markup, "")
        assert not source_mark_markup(remainder), url
    assert count >= 15


def test_config_file_change_reaches_rendered_pages(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.routes import site, steps, uploads, guides, how
    from app import reader

    app = site.create_site_app()
    app.include_router(steps.router)
    app.include_router(uploads.router)
    app.include_router(guides.router)
    app.include_router(how.router)
    guides.set_guide_loader(app, lambda job_id: reader.sample_guide())
    client = TestClient(app)
    urls = ['/', '/reader/sample', '/reader/hero', '/upload', '/delete-request', '/how-it-works',
            '/steps/preview', '/guides/job-guide', '/guide/job-guide'] + [
        '/static/ui/gallery/' + p.name for p in (ROOT / 'static/ui/gallery').glob('*.html')]
    before = {url: client.get(url).text for url in urls}
    original = brand_config.CONFIG.read_text(encoding="utf-8")
    config = tmp_path / "brand.toml"
    config.write_text(original.replace('/static/ui/favicon.svg', '/static/ui/changed-favicon.svg'), encoding="utf-8")
    monkeypatch.setattr(brand_config, "CONFIG", config)
    brand_config.load.cache_clear()
    uploads._env.cache.clear()
    try:
        for url in urls:
            response = client.get(url)
            assert response.status_code == 200 and response.text != before[url], url
            assert '/static/ui/changed-favicon.svg' in response.text, url
        config.write_text(original.replace('favicon =', 'favicn ='), encoding="utf-8")
        brand_config.load.cache_clear()
        with pytest.raises(BrandConfigError):
            brand_config.load()
    finally:
        brand_config.load.cache_clear()
        uploads._env.cache.clear()


def test_gallery_16px_uses_flat_mark():
    cfg = brand_config.load()
    assert cfg["gallery_sizes"][0] == 16 < cfg["real_min_px"]
    assert f'src="{cfg["flat"].split("#")[0]}"' in brand_config.gallery_mark(0)
    assert 'width="16" height="16"' in brand_config.gallery_mark(0)


def test_stateful_routes_render_and_follow_config(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routes import calls, jobs, uploads

    app = FastAPI()
    app.include_router(calls.router)
    app.include_router(jobs.router)
    app.include_router(uploads.router)
    app.dependency_overrides[calls.resolve_job_context] = lambda: SimpleNamespace(job_id="job-brand")
    record = SimpleNamespace(state="confirmed", hidden=False, meta_file_id="attached",
                             preview=SimpleNamespace(duration_s=10, size_bytes=123))
    monkeypatch.setattr(uploads, "_uploads", lambda request: object())
    monkeypatch.setattr(uploads, "_expire_preview_during_lookup", lambda store, job: (None, record))
    monkeypatch.setattr(jobs, "_job_state", lambda database, job: "waiting")
    monkeypatch.setattr(jobs, "_sending", lambda database, job: False)
    monkeypatch.setattr(jobs, "read_pending_settlement", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "stage_for_job", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "_latest_stage", lambda *a, **k: None)
    monkeypatch.setattr(jobs.wait_view, "build", lambda *a, **k: None)
    client = TestClient(app, headers={"Accept": "text/html"})
    urls = ["/calls/job-brand", "/jobs/job-brand", "/preview/job-brand"]
    before = {}
    for url in urls:
        response = client.get(url)
        assert response.status_code == 200, url
        assert brand_config.icon_links() in response.text, url
        remainder = response.text
        for markup in [brand_config.icon_links(), brand_config.mark_badge(), brand_config.footer_mark(), brand_config.inline_mark()] + [brand_config.gallery_mark(i) for i in range(4)]:
            remainder = remainder.replace(markup, "")
        assert not source_mark_markup(remainder), url
        before[url] = response.text
    original = brand_config.CONFIG.read_text(encoding="utf-8")
    config = tmp_path / "brand.toml"
    config.write_text(original.replace('/static/ui/favicon.svg', '/static/ui/changed-favicon.svg'), encoding="utf-8")
    monkeypatch.setattr(brand_config, "CONFIG", config)
    brand_config.load.cache_clear()
    uploads._env.cache.clear()
    jobs._env.cache.clear()
    try:
        for url in urls:
            response = client.get(url)
            assert response.status_code == 200 and response.text != before[url], url
            assert brand_config.icon_links() in response.text, url
    finally:
        brand_config.load.cache_clear()
        uploads._env.cache.clear()
        jobs._env.cache.clear()
