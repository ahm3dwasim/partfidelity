"""tokens, shell CSS, partials, template wiring, gallery."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import brand_config

ROOT = Path(__file__).resolve().parents[1]


def _filled(path: Path) -> str:
    """A page as the app serves it: brand tokens filled from config/brand.toml."""
    return brand_config.fill(path.read_text(encoding="utf-8"))

TOKENS = (ROOT / "static" / "ui" / "tokens.css").read_text(encoding="utf-8")
SHELL = (ROOT / "static" / "ui" / "shell.css").read_text(encoding="utf-8")
HEADER_PARTIAL = _filled(ROOT / "templates" / "partials" / "ui" / "shell-header.html")
FOOTER_PARTIAL = _filled(ROOT / "templates" / "partials" / "ui" / "shell-footer.html")
ICON = (ROOT / "static" / "ui" / "icon.svg").read_text(encoding="utf-8")
GALLERY = _filled(ROOT / "static" / "ui" / "gallery" / "shell.html")

# The shell's tokens, exact.
EXPECTED_TOKENS = {
    "--pf-ink": "#0f0f0f",
    "--pf-ink-2": "#171717",
    "--pf-ink-line": "#2c2c2c",
    "--pf-on-ink": "#ecece9",
    "--pf-on-ink-muted": "#b9b9b4",
    "--pf-on-ink-dim": "#9a9a96",
    "--pf-fog": "#f4f4f2",
    "--pf-on-fog": "#161616",
    "--pf-on-fog-muted": "#55554f",
    "--pf-accent": "#e87032",
    "--pf-accent-ink": "#111111",
    "--pf-danger": "#ff8a70",
    "--pf-ok": "#7ed39a",
    # The older names follow D.
    "--pf-paper": "var(--pf-ink)",
    "--pf-amber": "var(--pf-accent)",
    "--pf-space-4": "4px",
    "--pf-space-8": "8px",
    "--pf-space-12": "12px",
    "--pf-space-16": "16px",
    "--pf-space-24": "24px",
    "--pf-space-32": "32px",
    "--pf-space-48": "48px",
    "--pf-space-64": "64px",
    "--pf-space-96": "96px",
    "--pf-bar": "64px",
    "--pf-radius-workshop": "2px",
    "--pf-radius-manual": "10px",
    "--pf-motion-ui": "180ms",
    "--pf-motion-reveal": "600ms",
}

EXPECTED_FONTS = {
    "--pf-font-display": '"David Libre"',
    "--pf-font-body": '"Roboto"',
    "--pf-font-mono": "ui-monospace",
}

FONTS = (
    ("David Libre", "400", "fonts/DavidLibre-400-normal.woff2"),
    ("David Libre", "500", "fonts/DavidLibre-500-normal.woff2"),
    ("Roboto", "100 900", "fonts/Roboto-variable-normal.woff2"),
)

START_PAGES = ("index.html", "upload.html", "preview.html")
GUIDE_PAGES = ("reader.html", "guide.html", "calls.html")


def _tokens(css: str) -> dict[str, str]:
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", css))


def test_tokens_are_exact_common_values():
    tokens = _tokens(TOKENS)
    for name, value in EXPECTED_TOKENS.items():
        assert tokens.get(name) == value, name
    for name, family in EXPECTED_FONTS.items():
        assert family in tokens.get(name, ""), name
    # Shared focus token (also used by the loading set): the one accent.
    assert tokens.get("--pf-focus") == "var(--pf-accent)"


def test_vendored_fonts_exist_and_are_woff2():
    seen_hashes = set()
    for family, weight, rel in FONTS:
        assert f'font-family: "{family}"; font-weight: {weight}' in TOKENS.replace(
            "\n", " "
        ).replace("  ", " "), (family, weight)
        assert rel in TOKENS, rel
        path = ROOT / "static" / "ui" / rel
        data = path.read_bytes()
        assert data[:4] == b"wOF2", rel  # real woff2, not a placeholder
        assert len(data) > 4000, rel
        seen_hashes.add(hash(data))
    # Each weight ships its own file: no weight reuses another's bytes.
    assert len(seen_hashes) == len(FONTS)
    license_text = (ROOT / "static" / "ui" / "fonts" / "OFL.txt").read_text(
        encoding="utf-8"
    )
    assert "SIL Open Font License" in license_text


def test_shell_css_contract():
    for required in (
        ".pf-theme-start",
        ".pf-theme-guide",
        ".pf-display",
        ".pf-display-sm",
        ".pf-lede",
        ".pf-container",
        ".pf-grid",
        "repeat(4,",
        "repeat(12,",
        ".pf-span-full",
        ".pf-header",
        ".pf-wordmark",
        ".pf-nav",
        ".pf-footer",
        ".pf-seam",
        ":focus-visible",
        "prefers-reduced-motion",
        "--pf-bar",
        ".pf-toast",
        ".pf-theme-start h1",
        ".pf-theme-guide h1",
        ".pf-header .skip",
    ):
        assert required in SHELL, required
    # Shell consumes tokens; it hardcodes no palette hex of its own.
    assert not re.findall(r"#[0-9a-fA-F]{6}", SHELL)
    assert "http" not in SHELL
    # Start-theme adaptation: light-mode page vars remap to dark tokens.
    for remap in (
        "--paper: var(--pf-ink)",
        "--card: var(--pf-ink-2)",
        "--field: var(--pf-ink-line)",
        "--ink: var(--pf-on-ink)",
        "--muted: var(--pf-on-ink-muted)",
        "--accent: var(--pf-amber)",
        "--accent-ink: var(--pf-amber-ink)",
        "--focus: var(--pf-focus)",
        "--pf-focus: var(--pf-amber)",
        "--replay-bg: var(--pf-ink-line)",
    ):
        assert remap in SHELL, remap
    # Guide theme: both themes sit on ink in D, so the guide's focus ring is
    # the accent too (6.2:1 on ink).
    # (The theme's own block, not the shared ".pf-theme-start, .pf-theme-guide" one.)
    guide_block = SHELL.split("}\n.pf-theme-guide {", 1)[1].split("}", 1)[0]
    for remap in (
        "--accent: var(--pf-amber-deep)",
        "--accent-ink: var(--pf-amber-ink)",
        "--focus: var(--pf-focus)",
        "--pf-focus: var(--pf-accent)",
    ):
        assert remap in guide_block, remap
    # Bright amber stays for guide fills only: CTA button + Easy chip keep
    # bright backgrounds with dark text.
    assert ".pf-theme-guide .cta-primary" in SHELL
    assert ".pf-theme-start .badge-easy" in SHELL
    assert ".pf-theme-guide .badge-easy" in SHELL
    # Focus override beats guide.css a/button:focus-visible:
    # plain theme classes, not :where (which carries zero specificity).
    assert ".pf-theme-guide a:focus-visible" in SHELL
    assert ".pf-theme-guide button:focus-visible" in SHELL
    assert ":where(.pf-theme-start, .pf-theme-guide) :focus-visible" not in SHELL
    # D focus: one accent outline set off the element on both themes.
    assert "outline: 2px solid var(--pf-focus)" in SHELL
    assert "outline-offset: 3px" in SHELL
    # The header is D's fixed bottom bar, with room left for it.
    assert "bottom: 0" in SHELL.split(".pf-header {", 1)[1].split("}", 1)[0]
    assert "padding-bottom: var(--pf-bar)" in SHELL


def test_header_partial_has_skip_brand_nav_and_icon():
    assert '<a class="skip" href="#main">Skip to content</a>' in HEADER_PARTIAL
    assert '<span class="pf-wordmark">PartFidelity</span>' in HEADER_PARTIAL
    assert 'aria-label="PartFidelity home"' in HEADER_PARTIAL
    assert '<nav class="pf-nav" aria-label="Site">' in HEADER_PARTIAL
    assert '<a href="/guide/sample" data-copy="nav.sample">Sample guide</a>' in HEADER_PARTIAL
    assert '<a href="/upload" data-copy="nav.upload">Make a guide</a>' in HEADER_PARTIAL
    assert '<a href="/how-it-works" data-copy="nav.how">How it works</a>' in HEADER_PARTIAL
    assert "<svg " in HEADER_PARTIAL and 'viewBox="0 0 32 32"' in HEADER_PARTIAL
    # One shared icon source: the header references icon.svg, it does not
    # embed a second copy of the shapes.
    assert '<use href="/static/ui/icon.svg#pf-mark"></use>' in HEADER_PARTIAL
    assert 'id="pf-mark"' in ICON
    for shape in ("<rect", "<circle", "<path"):
        assert shape not in HEADER_PARTIAL, shape
    scrubbed = re.sub(r"xmlns='[^']+'", "", HEADER_PARTIAL)
    for banned in ("<script", "onclick", "http://", "https://", "{{", "$"):
        assert banned not in scrubbed, banned


def test_footer_partial_has_note_slot_and_shell_line():
    assert "{{ pf_footer_note }}" in FOOTER_PARTIAL
    assert 'data-copy="footer.note"' in FOOTER_PARTIAL
    assert "PartFidelity · guides from your own footage" in FOOTER_PARTIAL
    assert 'data-copy="footer.shell"' in FOOTER_PARTIAL
    for banned in ("<script", "onclick", "http://", "https://"):
        assert banned not in FOOTER_PARTIAL, banned


def test_icon_is_crisp_svg():
    root = ET.fromstring(ICON)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["viewBox"] == "0 0 32 32"
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root) == 1 and root[0].tag == f"{ns}g", "one #pf-mark group"
    assert root[0].attrib["id"] == "pf-mark"
    kinds = [el.tag.split("}", 1)[1] for el in root[0]]
    # P, the wrench, then F drawn over a gap cut so the wrench passes between
    # the letters. One accent: only the wrench is orange.
    assert kinds == ["path", "g", "path", "path"], kinds
    assert root[0][1][0].attrib["fill"] == "#e87032"
    fills = re.findall(r'fill="(#[0-9a-f]{6})"', ICON)
    assert sorted(set(fills)) == ["#e87032"], fills
    assert ICON.count('stroke="#ecece9"') == 2


def _template(name: str) -> str:
    return _filled(ROOT / "templates" / name)


@pytest.mark.parametrize("name", START_PAGES)
def test_start_pages_use_start_theme(name):
    assert 'class="pf-theme-start"' in _template(name), name


@pytest.mark.parametrize("name", GUIDE_PAGES)
def test_guide_pages_use_guide_theme(name):
    assert 'class="pf-theme-guide"' in _template(name), name


def test_every_page_links_shell_after_page_css():
    for name in START_PAGES + GUIDE_PAGES:
        body = _template(name)
        tokens_link = '<link rel="stylesheet" href="/static/ui/tokens.css">'
        shell_link = '<link rel="stylesheet" href="/static/ui/shell.css">'
        assert tokens_link in body, name
        assert shell_link in body, name
        assert body.index(tokens_link) < body.index(shell_link), name
        for page_css in ("/static/site.css", "/static/calls.css", "{{GUIDE_CSS}}"):
            if page_css in body:
                assert body.index(page_css) < body.index(tokens_link), (name, page_css)
        # The D mark favicon (one orange), not the old green P.
        assert 'rel="icon" href="/static/ui/favicon.svg"' in body, name
        assert "%231e5b3c" not in body, name


def test_favicon_file_is_the_orange_mark():
    fav = (ROOT / "static" / "ui" / "favicon.svg").read_text(encoding="utf-8")
    assert "#e87032" in fav and "#1e5b3c" not in fav


def _normalize(html: str) -> str:
    return re.sub(r"\s+", " ", html).strip()


def test_inline_copies_match_partials():
    header = _normalize(HEADER_PARTIAL)
    before_slot, after_slot = _normalize(FOOTER_PARTIAL).split("{{ pf_footer_note }}")
    for name in ("index.html", "reader.html", "guide.html", "calls.html"):
        body = _normalize(_template(name))
        assert header in body, name
        assert before_slot in body, name
        assert after_slot in body, name
        assert body.index(before_slot) < body.index(after_slot), name


def test_jinja_pages_include_partials_live():
    for name in ("upload.html", "preview.html"):
        body = _template(name)
        assert '{% include "partials/ui/shell-header.html" %}' in body, name
        assert "{% set pf_footer_note" in body, name
        assert '{% include "partials/ui/shell-footer.html" %}' in body, name
        # No layout CSS of their own: main uses the shell container.
        assert '<main id="main" class="pf-container">' in body, name


def test_pages_with_layout_keep_their_own_main():
    assert '<main id="main">' in _template("index.html")
    assert '<main id="main"$main_attr>' in _template("reader.html")
    assert '<main id="main" class="guide">' in _template("guide.html")
    # Merged contract: shell id + page class, plus main's loading fade hook
    # (loading.js binds `main[data-pf-fade]`; dropping it would unwire the fade).
    assert '<main id="main" class="page" data-pf-fade>' in _template("calls.html")


def test_every_page_has_one_skip_target_main():
    for name in START_PAGES + GUIDE_PAGES:
        assert _template(name).count('id="main"') == 1, name


def test_landing_scripts_are_the_film_only():
    html = _template("index.html")
    assert html.count("<script") == 2
    assert 'id="film-config"' in html and '/static/ui/film.js' in html


def test_rendered_site_pages_carry_shell():
    from app.routes.site import create_site_app

    client = TestClient(create_site_app())
    landing = client.get("/")
    assert landing.status_code == 200
    assert 'class="pf-theme-start"' in landing.text
    assert "pf-header" in landing.text and "pf-footer" in landing.text
    assert landing.text.count("<script") == 2
    assert '<a class="skip" href="#main">Skip to content</a>' in landing.text
    assert '<main id="main">' in landing.text
    reader = client.get("/reader/sample")
    assert reader.status_code == 200
    assert 'class="pf-theme-guide"' in reader.text
    assert '<a class="skip" href="#main">Skip to content</a>' in reader.text
    assert reader.text.count('class="skip"') == 1
    assert '<main id="main">' in reader.text
    for asset in (
        "/static/ui/tokens.css",
        "/static/ui/shell.css",
        "/static/ui/icon.svg",
        "/static/ui/gallery/shell.html",
        "/static/ui/fonts/DavidLibre-400-normal.woff2",
        "/static/ui/fonts/Roboto-variable-normal.woff2",
    ):
        response = client.get(asset)
        assert response.status_code == 200, asset


def test_rendered_jinja_pages_carry_shell(tmp_path):
    from app.brand_templates import make_env

    env = make_env(ROOT / "templates")
    from app.routes.uploads import _copy

    upload = env.get_template("upload.html").render(
        c=_copy(300.0), error="", blocked=False, blocked_copy="", limits_json="{}"
    )
    assert 'class="pf-theme-start"' in upload
    assert "nothing goes to Meta until you confirm" in upload
    preview = env.get_template("preview.html").render(
        c=_copy(300.0),
        job_id="j",
        duration="1.0",
        size="8",
        video_url="/preview/j/video",
        show_video=False,
        state="consent",
        notice="",
        cancelled_copy="",
        hidden_copy="",
        meta_recipient="Meta",
        meta_hours="24",
        local_hours="24",
        verified_copy="",
        acknowledged_copy="",
    )
    assert "confirm to send · cancel to delete now" in preview
    assert preview.count("pf-header") >= 1


def test_rendered_guide_and_calls_pages_carry_shell():
    from test_calls_page import FakeCtx

    from app import evidence
    from app.contracts import Guide, JobId
    from app.routes.calls import render_calls_page
    from app.routes.guides import _render_page

    evidence.reset_stores()
    try:
        calls = render_calls_page("job-shell", FakeCtx(JobId("job-shell")))
    finally:
        evidence.reset_stores()
    assert 'class="pf-theme-guide"' in calls
    assert "Observed numbers only." in calls
    guide = Guide(
        steps=(),
        difficulty="Easy",
        verdict="Try it",
        job_id=JobId("job-shell"),
        frame_paths=(),
        labels=(),
        draft_reasons=(),
    )
    page = _render_page("job-shell", guide)
    assert 'class="pf-theme-guide"' in page
    assert "pf-header" in page and "pf-footer" in page


def _luminance(hex_color: str) -> float:
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def channel(value: float) -> float:
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(v) for v in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _ratio(first: str, second: str) -> float:
    light = _luminance(first)
    dark = _luminance(second)
    if dark > light:
        light, dark = dark, light
    return (light + 0.05) / (dark + 0.05)


def test_token_pairs_meet_aa():
    tokens = _tokens(TOKENS)
    pairs = [
        ("--pf-on-ink", "--pf-ink"),
        ("--pf-on-ink-muted", "--pf-ink"),
        ("--pf-on-ink-muted", "--pf-ink-2"),
        ("--pf-on-ink-dim", "--pf-ink"),
        ("--pf-on-ink-dim", "--pf-ink-2"),
        ("--pf-accent-ink", "--pf-accent"),
        ("--pf-accent", "--pf-ink"),
        ("--pf-accent", "--pf-ink-2"),
        ("--pf-danger", "--pf-ink"),
        ("--pf-danger", "--pf-warn-bg"),
        ("--pf-ok", "--pf-ink"),
        ("--pf-warn-text", "--pf-warn-bg"),
        ("--pf-warn-title", "--pf-warn-bg"),
        ("--pf-nudge-text", "--pf-nudge-bg"),
        ("--pf-on-fog", "--pf-fog"),
        ("--pf-on-fog-muted", "--pf-fog"),
    ]
    for fg, bg in pairs:
        assert _ratio(tokens[fg], tokens[bg]) >= 4.5, (fg, bg)


def test_focus_outline_clears_3_to_1_on_both_themes():
    tokens = _tokens(TOKENS)
    # Both themes sit on ink: the accent outline on ink and on the panel ink.
    assert _ratio(tokens["--pf-accent"], tokens["--pf-ink"]) >= 3
    assert _ratio(tokens["--pf-accent"], tokens["--pf-ink-2"]) >= 3


def test_gallery_shows_sizes_type_and_both_themes():
    assert '<link rel="stylesheet" href="../tokens.css">' in GALLERY
    assert '<link rel="stylesheet" href="../shell.css">' in GALLERY
    for size in ("16", "24", "32", "48"):
        assert f'<img src="/static/ui/icon.svg" width="{size}" height="{size}"' in GALLERY
    assert "pf-theme-start" in GALLERY
    assert "pf-theme-guide" in GALLERY
    assert "pf-seam" in GALLERY
    assert "pf-display" in GALLERY and "pf-lede" in GALLERY
    assert "pf-grid" in GALLERY
    assert "Fake data" in GALLERY
    # One shared icon source: the demo headers reference icon.svg, they do
    # not embed a second copy of the shapes.
    assert GALLERY.count('<use href="/static/ui/icon.svg#pf-mark"></use>') == 2
    for shape in ("<rect", "<circle", "<path"):
        assert shape not in GALLERY, shape
    scrubbed = re.sub(r"xmlns='[^']+'", "", GALLERY)
    for banned in ("<script", "http://", "https://", "@import"):
        assert banned not in scrubbed, banned
