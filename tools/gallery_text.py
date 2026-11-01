"""Write the configured status wording into the guide-step gallery fixture.

The gallery (static/ui/gallery/guide_step.html) is a static copy of the
guide_step partial. Its visual-change badge and the help line under each card
that shows one come from config/guide_text.toml, the same strict loader the
live page uses. Run with no arguments to rewrite the fixture; ``--check``
exits 1 when the fixture is out of date.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GALLERY = ROOT / "static" / "ui" / "gallery" / "guide_step.html"

_OK = re.compile(r'(<span class="pf-proof is-ok" data-copy="proof\.ok">)[^<]*(</span>)')
_HELP = re.compile(r'\n[ \t]*<p class="parts-note pf-filmed-only" data-copy="proof\.help">[^<]*</p>')
_LEGEND = re.compile(r'(?P<indent>[ \t]*)<ol class="pf-legend[^"]*".*?</ol>', re.S)


def apply(src: str, text: dict[str, str]) -> str:
    """The fixture with every badge and one help line per card that has a badge."""
    ok = html.escape(text["status_unchanged"])
    help_line = html.escape(text["status_unchanged_help"])
    out = _HELP.sub("", src)
    out = _OK.sub(lambda m: m.group(1) + ok + m.group(2), out)

    def legend(m: re.Match) -> str:
        block = m.group(0)
        if 'data-copy="proof.ok"' not in block:
            return block
        return (
            f'{block}\n{m.group("indent")}'
            f'<p class="parts-note pf-filmed-only" data-copy="proof.help">{help_line}</p>'
        )

    return _LEGEND.sub(legend, out)


def main(argv: list[str]) -> int:
    from app.guide_text import TEXT

    src = GALLERY.read_text(encoding="utf-8")
    out = apply(src, TEXT)
    if "--check" in argv:
        if out != src:
            print(f"{GALLERY.relative_to(ROOT)} is out of date: run python tools/gallery_text.py")
            return 1
        return 0
    if out != src:
        GALLERY.write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
