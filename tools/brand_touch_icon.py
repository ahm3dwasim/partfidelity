"""Write the apple-touch icon: the approved real mark (static/ui/icon-real.svg, the existing render) on the dark tile.

Reads config/brand.toml for the file, size, padding and background. No Blender run; needs playwright with chromium.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import brand_config  # noqa: E402


def main() -> None:
    from playwright.sync_api import sync_playwright

    cfg = brand_config.load()
    px, out = cfg["touch_icon_px"], ROOT / cfg["touch_icon"].lstrip("/")
    svg = (ROOT / cfg["real"].split("#")[0].lstrip("/")).read_text(encoding="utf-8")
    pad, bg = round(px * cfg["touch_icon_pad"]), cfg["touch_icon_bg"]
    html = (f'<body style="margin:0;background:{bg}"><div style="width:{px}px;height:{px}px;padding:{pad}px;box-sizing:border-box">'
            f'<div style="width:100%;height:100%">{svg.replace("<svg ", "<svg width=%s height=%s " % (chr(34) + "100%" + chr(34), chr(34) + "100%" + chr(34)), 1)}'
            "</div></div></body>")
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": px, "height": px})
        page.set_content(html)
        page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": px, "height": px})
        b.close()


if __name__ == "__main__":
    main()
