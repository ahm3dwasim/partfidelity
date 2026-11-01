"""The /how-it-works page: the pipeline from a phone video to a guide, in plain words. Six stage cards, each
showing what goes in, what happens and what comes out, scroll past a sticky concept-phone figure (built by
tools/how_figure/build.mjs) whose labelled layer lights for the stage being read; a table says which model does what.

Copy and the six stages come from config/how_it_works.toml through the strict loader; every number and model
name is read from the app itself (app/how_config.py). The page makes no model call and reads no job data.
"""

from __future__ import annotations

from html import escape
from contextvars import ContextVar
import json
import logging
from pathlib import Path
from string import Template

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import how_config, how_media
from app.brand_config import fill
from app.settings import DEFAULT_MAX_VIDEO_S, DEFAULT_CHUNK_S, DEFAULT_CHUNK_OVERLAP_S

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "templates" / "how.html"
FIGURE = ROOT / "templates" / "partials" / "ui" / "how-figure.svg"

router = APIRouter()


def _e(text: str) -> str:
    return escape(text, quote=True)


MEDIA = "/static/ui/media/how/"


log = logging.getLogger(__name__)
_frames = ContextVar("how_frames", default=True)


def _img(file: str, alt: str, w: int, h: int, cls: str = "") -> str:
    if not _frames.get():
        return ""
    return (f'<img class="{cls}" src="{MEDIA}{_e(file)}" alt="{_e(alt)}" width="{w}" height="{h}" '
            f'loading="lazy" decoding="async">')


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}".replace(".0", "") if t % 1 else f"{int(t // 60)}:{int(t % 60):02d}"


def _strip(ex: dict, held: str = "") -> str:
    cells = "".join(
        f'<li>{_img(r["file"], "Frame from the illustrative clip at " + _clock(r["t"]), 96, 54)}<span>{_clock(r["t"])}</span></li>'
        for r in ex["strip"]
    )
    return f'<ul class="how-strip{" is-held" if held else ""}">{cells}</ul>' + (f'<p class="how-chip">{_e(held)}</p>' if held else "")


def _sam_overlay(ex: dict) -> str:
    sam = ex["sam"]
    w, h = sam["size"]
    x0, y0, x1, y1 = sam["box"]
    return (f'<svg class="how-sam" viewBox="0 0 {w:g} {h:g}" aria-hidden="true">'
            f'<rect class="how-box" x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" height="{y1 - y0:g}" rx="10" pathLength="1"/>'
            f'<g class="how-badge"><circle cx="{x0 + 8:g}" cy="{y0 + 8:g}" r="34"/><text x="{x0 + 8:g}" y="{y0 + 20:g}" text-anchor="middle">1</text></g></svg>')


def _framed(ex: dict, file: str, alt: str, outlined: bool = False) -> str:
    sam = ex["sam"]
    w, h = sam["size"]
    ov = _sam_overlay(ex) if outlined else ""
    if not _frames.get():
        return ""
    return f'<div class="how-shot">{_img(file, alt, 960, round(960 * h / w))}{ov}</div>'


def _runs(ex: dict) -> str:
    lo = min(r["start"] for r in ex["runs"]) - 1.0
    hi = max(r["end"] for r in ex["runs"]) + 1.0
    rows = "".join(
        f'<li><span class="how-run-t">{r["start"]:g}–{r["end"]:g} s</span><span class="how-track"><span class="how-run" style="left:{(r["start"] - lo) / (hi - lo) * 100:.1f}%;width:{(r["end"] - r["start"]) / (hi - lo) * 100:.1f}%"></span></span></li>'
        for r in ex["runs"]
    )
    return f'<ul class="how-runs" aria-label="{_e(ex["runs_head"])}">{rows}</ul>'


def _steps_list(ex: dict, mark: bool = True) -> str:
    rows = "".join(
        f'<li{" class=is-pick" if mark and i == ex["step_index"] else ""}><span class="how-st-n">{i + 1}</span>'
        f'<span class="how-st-t">{_e(st["title"])}</span><span class="how-st-time">{st["start"]:g}–{st["end"]:g} s</span></li>'
        for i, st in enumerate(ex["steps"])
    )
    return f'<ol class="how-steps">{rows}</ol>'


def _scored(ex: dict) -> str:
    top = ex["best_score"]
    bars = []
    for f in ex["scored"]:
        k = f.timestamp_s in ex["kept"]
        bars.append(
            f'<li{" class=is-kept" if k else ""}>{_img(f"c{f.timestamp_s:g}.webp", "Candidate frame at " + f"{f.timestamp_s:g} s", 64, 36)}'
            f'<span class="how-bar"><i style="height:{max(4.0, f.score / top * 100):.0f}%"></i></span><b>{f.timestamp_s:g}</b></li>'
        )
    return f'<ul class="how-cands" aria-label="{_e(ex["frames_head"])}">{"".join(bars)}</ul>'


def _kept(ex: dict) -> str:
    cells = []
    w, h = ex["sam"]["size"]
    for t in ex["kept"]:
        f = next(x for x in ex["scored"] if x.timestamp_s == t)
        cells.append(f'<figure class="how-kept">{_img(f"k{t:g}.webp", "Kept frame at " + f"{t:g} s", 960, round(960 * h / w))}'
                     f'<figcaption>{t:g} s · {_e(ex["kept_text"])} · {_e(ex["score_text"])} {f.score:.0f}</figcaption></figure>')
    return f'<div class="how-kept-row">{"".join(cells)}</div>'


def _guide_card(ex: dict, word: str) -> str:
    st = ex["steps"][ex["step_index"]]
    sam = ex["sam"]
    warns = "".join(f"<li>{_e(t)}</li>" for t in ex["warnings"])
    return (f'<div class="how-guide"><p class="how-guide-n">{_e(ex["guide_step_word"])} {ex["step_index"] + 1}</p>'
            f'<p class="how-guide-t">{_e(st["title"])}</p>'
            f'{_framed(ex, sam["file"], "The finished guide picture: the battery outlined and numbered 1", True)}'
            f'<ul class="how-warn">{warns}</ul>'
            f'<p class="how-chip">{_e(ex["difficulty_label"])}: {_e(ex["difficulty"])}</p></div>')


def _io(word: str, label: str, note: str, visual: str) -> str:
    return (f'<div class="how-io"><p class="how-io-l">{_e(word)}</p><p class="how-io-t">{_e(label)}</p>'
            f'<p class="how-io-n">{_e(note)}</p>{visual}</div>')


def _visuals(c: dict) -> list[tuple[str, str]]:
    ex = c["example"]
    sam = ex["sam"]
    conf = f"{sam['confidence']:.2f}"
    names = {"sam": c["stages"][3]["model"]}
    pick = ex["steps"][ex["step_index"]]
    tag = f'<p class="how-tag">{_e(c["page"]["illustrative"])}</p>'
    frame_alt = "Frame from the illustrative clip at " + f"{sam['frame_t']:g} s"
    return [
        (_strip(ex), _strip(ex, f'{c["facts"]["ttl_hours"]} h')),
        (_strip(ex), tag + f'<p class="how-io-n">{_e(ex["provenance"])}</p>' + _steps_list(ex)),
        (_scored(ex), _kept(ex)),
        (_framed(ex, sam["file"], frame_alt) + f'<p class="how-chip">{_e(sam["part"])}</p>',
         _framed(ex, sam["file"], frame_alt + ", the " + sam["part"] + " outlined", True)
         + f'<p class="how-chip">{_e(names["sam"])} · {conf}</p>'),
        (f'<div class="how-shot how-shot-s">{_framed(ex, sam["file"], frame_alt, True)}</div>'
         f'<p class="how-chip">{conf} {_e(ex["below_text"])} {c["facts"]["cutoff"]}</p>',
         tag + f'<p class="how-q">{_e(ex["q_text"])}</p><div class="how-fork"><span>{_e(ex["yes_text"])}</span><span>{_e(ex["no_text"])}</span></div>'
         f'<p class="how-io-n">{_e(ex["readback_note"])}</p>'),
        (f'<p class="how-chip">{_e(pick["title"])}</p>' + _framed(ex, sam["file"], frame_alt, True),
         tag + _guide_card(ex, ex["guide_step_word"]) + f'<p class="how-io-n">{_e(ex["guide_note"])}</p>'),
    ]


def _hood_extra(c: dict, i: int) -> str:
    """Secondary detail shown only under "Under the hood": the three readings of one step (stage 2)."""
    ex = c["example"]
    if c["stages"][i]["id"] != "steps":
        return ""
    return (f'<p class="how-io-l">{_e(ex["runs_head"])}</p>{_runs(ex)}'
            f'<p class="how-chip">{_e(ex["median_text"])} · {_e(ex["spread_text"])} {ex["spread"]:g} s</p>')


def _stepper_html(c: dict) -> str:
    return "".join(
        f'<button type="button" class="how-step" data-stage="{i}"{" aria-current=step" if i == 0 else ""} '
        f'aria-label="{_e(c["page"]["stage_word"])} {i + 1}: {_e(s["title"])}">'
        f'<span class="how-n">{i + 1:02d}</span><span class="how-step-t">{_e(s["title"])}</span></button>'
        for i, s in enumerate(c["stages"])
    )


def _stages_html(c: dict) -> str:
    page = c["page"]
    rows = []
    for i, (s, (vin, vout)) in enumerate(zip(c["stages"], _visuals(c))):
        rows.append(
            f'<li class="how-card" data-card="{i}" id="stage-{_e(s["id"])}"{" data-on" if i == 0 else ""}>'
            f'<h3 class="how-stage"><span class="how-n">{_e(page["stage_word"])} {i + 1:02d}</span>'
            f'<span class="how-stage-title">{_e(s["title"])}</span></h3>'
            f'<p class="how-kind" data-kind="{_e(s["kind"])}">{_e(c["kinds"][s["kind"]])}</p>'
            f'<p class="how-plain">{_e(s["plain"])}</p>'
            f'<div class="how-flow">{_io(page["in_word"], s["in_label"], s["in_note"], vin)}'
            f'<div class="how-op"><span class="how-arrow" aria-hidden="true"></span><p class="how-op-t">{_e(s["op"])}</p></div>'
            f'{_io(page["out_word"], s["out_label"], s["out_note"], vout)}</div>'
            f'<details class="how-detail"><summary>{_e(page["detail_label"])}</summary>'
            f'<p class="how-model">{_e(s["model"])}</p><p>{_e(s["detail"])}</p>{_hood_extra(c, i)}</details></li>'
        )
    return "\n".join(rows)


def _figure(c: dict) -> str:
    svg = FIGURE.read_text(encoding="utf-8")
    for i, s in enumerate(c["stages"]):
        svg = svg.replace(f"%%L{i}%%", _e(s["part"]))
    return svg


def _tech_html(tech: list[dict]) -> str:
    rows = [
        f'<div class="how-tech-row"><dt>{_e(t["name"])}</dt><dd>{_e(t["does"])} <span class="how-facts">{_e(t["facts"])}</span></dd></div>'
        for t in tech
    ]
    return "\n".join(rows)


def render(*, spark_model_id: str | None = None, sam_model_id: str | None = None,
           max_video_s: float = DEFAULT_MAX_VIDEO_S, chunk_s: float = DEFAULT_CHUNK_S,
           chunk_overlap_s: float = DEFAULT_CHUNK_OVERLAP_S) -> str:
    cfg = how_config.cached()
    missing = how_media.problems(deep=False)
    if missing:
        log.error("how-it-works: frames unavailable, rendering stage text only (%s); run: %s", "; ".join(missing), how_media.fix_hint())
    _frames.set(not missing)
    settings = {"max_video_s": max_video_s, "chunk_s": chunk_s, "chunk_overlap_s": chunk_overlap_s}
    if spark_model_id:
        settings["spark_model_id"] = spark_model_id
    if sam_model_id:
        settings["sam_model_id"] = sam_model_id
    c = how_config.context(cfg, **settings)
    c["facts"] = how_config.facts(cfg, **settings)
    page = c["page"]
    public = {
        "stages": [{"readout": s["readout"], "title": s["title"]} for s in c["stages"]],
        **cfg["motion"],
        "play": page["play"],
        "pause": page["pause"],
    }
    first = c["stages"][0]
    return fill(Template(TEMPLATE.read_text(encoding="utf-8")).substitute(
        title=_e(page["title"]),
        meta_description=_e(page["meta_description"]),
        eyebrow=_e(page["eyebrow"]),
        headline=_e(page["headline"]),
        intro=_e(page["intro"]),
        bench_heading=_e(page["bench_heading"]),
        bench_hint=_e(page["bench_hint"]),
        figure_label=_e(page["figure_label"]),
        fig_title=_e(page["fig_title"]),
        fig_hint=_e(page["fig_hint"]),
        concept_note=_e(page["concept_note"]),
        first_readout=_e(first["readout"]),
        first_title=_e(first["title"]),
        play=_e(page["play"]),
        stages_html=_stages_html(c),
        hood_heading=_e(page["hood_heading"]),
        hood_intro=_e(page["hood_intro"]),
        tech_html=_tech_html(c["tech"]),
        limits_heading=_e(page["limits_heading"]),
        limits_body=_e(page["limits_body"]),
        calls_link=_e(page["calls_link"]),
        cta_heading=_e(page["cta_heading"]),
        cta_sample=_e(page["cta_sample"]),
        cta_upload=_e(page["cta_upload"]),
        stepper_html=_stepper_html(c),
        stepper_label=_e(page["stepper_label"]),
        provenance=_e(c["example"]["provenance"]),
        figure_svg=_figure(c),
        config_json=json.dumps(public).replace("</", "<\\/"),
    ))


@router.get("/how-it-works", response_class=HTMLResponse)
def how_it_works(request: Request) -> str:
    settings = getattr(request.app.state, "settings", None)
    return render(
        spark_model_id=getattr(settings, "spark_model_id", None),
        sam_model_id=getattr(settings, "sam_model_id", None),
        max_video_s=getattr(settings, "max_video_s", DEFAULT_MAX_VIDEO_S),
        chunk_s=getattr(settings, "chunk_s", DEFAULT_CHUNK_S),
        chunk_overlap_s=getattr(settings, "chunk_overlap_s", DEFAULT_CHUNK_OVERLAP_S),
    )
