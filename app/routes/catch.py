"""The /catch page: a recorded comparison of what the image-match gate rejected and accepted.

Every number and row comes from config/catch_table.json (generated from the frozen evaluate.json by
tools/catch_page_data.py); nothing here runs the gate, calls a model or takes an upload. The small
images are crops installed by tools/catch_media.py under MEDIA.
"""
from __future__ import annotations

from functools import lru_cache
from html import escape
import json
from pathlib import Path
from string import Template

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.brand_config import fill

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "config" / "catch_table.json"
TEMPLATE = ROOT / "templates" / "catch.html"
# Recipe value of the hand-made redraws; "owner-edit" is its former stored name and is still read.
REDRAWN = "redrawn-edit"
REDRAWN_ALIASES = frozenset({REDRAWN, "owner-edit"})
MEDIA = "/static/ui/media/catch/"  # config-ok: the served folder tools/catch_media.py installs into

router = APIRouter()


@lru_cache(maxsize=1)
def load() -> dict:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for key in ("caption", "eval_limits", "ssim_cutoff", "summary", "rows", "cases"):
        if key not in data:
            raise ValueError(f"{DATA.name}: missing {key}")
    return data


def _verdict(row: dict) -> str:
    word, cls = ("Accepted", "ok") if row["accepted"] else ("Rejected", "no")
    return f'<span class="catch-verdict is-{cls}">{word}</span>'


def _num(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _table(rows: list[dict], first: str) -> str:
    body = "".join(
        f'<tr id="row-{escape(r["name"])}"><td>{escape(r["name"] if r["number"] is None else f"#{r["number"]} {r["name"]}")}</td>'
        f'<td>{escape(r["label"])}</td><td>{escape(REDRAWN if r["recipe"] in REDRAWN_ALIASES else r["recipe"])}</td><td>{_num(r["iou"])}</td><td>{_num(r["ssim"])}</td>'
        f'<td>{_verdict(r)}</td><td>{escape(r["reason"])}</td></tr>'
        for r in rows
    )
    return (f'<div class="catch-scroll"><table class="catch-table"><thead><tr><th>{escape(first)}</th><th>Label</th><th>Recipe</th>'
            f'<th>Overlap</th><th>Similarity</th><th>Result</th><th>Why</th></tr></thead><tbody>{body}</tbody></table></div>')


def _case(case: dict, title: str, note: str) -> str:
    row = case["row"]
    return (
        f'<figure class="catch-case" id="case-{escape(case["key"])}"><div class="catch-pair">'
        f'<div><img src="{MEDIA}{escape(case["original"])}" alt="Original frame, fan area" width="320" loading="lazy"><figcaption>Original</figcaption></div>'
        f'<div><img src="{MEDIA}{escape(case["edited"])}" alt="Edited frame, fan area" width="320" loading="lazy"><figcaption>Edited</figcaption></div>'
        f'</div><h3>{escape(title)} {_verdict(row)}</h3><p>{escape(note)} Edit <code>{escape(row["name"])}</code>; '
        f'overlap {_num(row["iou"])}, similarity {_num(row["ssim"])}. {escape(row["reason"])}.</p></figure>'
    )


@router.get("/catch", response_class=HTMLResponse)
def catch() -> str:
    d = load()
    s = d["summary"]
    cases = {c["key"]: c for c in d["cases"]}
    redrawn_rows = [r for r in d["rows"] if r["clip"] == "H1" and r["recipe"] in REDRAWN_ALIASES]
    v3 = [r for r in d["rows"] if r["split"] == "heldout"]
    header = fill((ROOT / "templates/partials/ui/shell-header.html").read_text(encoding="utf-8"))
    return Template(fill(TEMPLATE.read_text(encoding="utf-8"))).substitute(
        header=header,
        altered_case=_case(cases["altered_edit"], "The altered edit", "The one real edit that changed the part."),
        accepted_case=_case(cases["accepted_edit"], "What passes", "A faithful edit the gate accepted."),
        redrawn_table=_table(redrawn_rows, "Redrawn edit"), v3_table=_table(v3, "Recipe case"),
        altered_rejected=s["altered_edit"]["rejected"], altered_total=s["altered_edit"]["total"],
        faithful_rejected=s["faithful_edits"]["rejected"], faithful_total=s["faithful_edits"]["total"],
        recipes_rejected=s["heldout_recipes"]["rejected"], recipes_total=s["heldout_recipes"]["total"],
        blur_accepted=s["heldout_blur"]["accepted"], blur_total=s["heldout_blur"]["total"],
        caption=escape(d["caption"]), limits=escape(d["eval_limits"]),
    )
