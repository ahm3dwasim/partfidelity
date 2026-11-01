"""Turn one saved live guide page into the shipped sample bundle.

Reads the guide page HTML and its numbered images (``/guides/<id>/image/N``)
saved from a live run, plus a web copy of the clip, and writes ``app/sample_real/``:
``guide.json`` (relative paths, the same shape ``app.pipeline.guide_to_json``
writes), ``frames/``, ``labels/`` and the clip. The page carries every value
the guide shows; the three the page does not show are fixed below and noted
in the bundle's ``README.md``.

    python3 tools/build_real_sample.py <guide.html> <image-dir> <clip.mp4> [out-dir]

The tool refuses (and touches nothing) when the output already holds real SAM masks, or when the
guide has the unnumbered intro card (number 0, kind intro): a rebuild would replace the real outlines
with placeholder masks, and this parser does not know the intro card. To rebuild on purpose, point it
at a fresh out-dir.
"""
from __future__ import annotations

import html
import json
import re
import secrets
import shutil
import sys
from pathlib import Path

FRAME_W, FRAME_H = 1280, 720
PIN_HALF = 24  # shipped pins keep only the box centre; the box is centre +- this (px)
UNSEEN_SPREAD_S = 6.0  # the page shows only the "check time" flag, not the spread
DIFFICULTY, VERDICT = "Hard", "DRAFT"  # battery steps; the page's draft banner


def _steps(page: str) -> list[dict]:
    steps = []
    for number, body in re.findall(r'<article class="card" id="step-(\d+)">(.*?)</article>', page, re.S):
        tools = re.search(r'<ul class="tools">(.*?)</ul>', body, re.S)
        start, end = re.search(r'data-start="([\d.]+)" data-end="([\d.]+)"', body).groups()
        parts = []
        for li in re.findall(r'<li(?: data-pin="[^"]*")?[^>]*><span class="pf-n".*?</li>', body, re.S):
            phrase = html.unescape(re.search(r"</span> (.*?) — <span", li).group(1))
            status = re.search(r'<span class="status-(\w+)"', li).group(1)
            labelled = re.search(r'data-labelled="[^"]*/image/(\d+)"', li)
            parts.append((phrase, status == "ok", int(labelled.group(1)) if labelled else None))
        steps.append({
            "number": int(number),
            "start_s": float(start),
            "end_s": float(end),
            "action": html.unescape(re.search(r'<p class="action">(.*?)</p>', body).group(1)),
            "tools": re.findall(r"<li>(.*?)</li>", tools.group(1)) if tools else [],
            "hazards": re.findall(r"Warning: (\w+)", body),
            "check": 'class="check-time"' in body,
            "frame": int(re.search(r'<img src="[^"]*/image/(\d+)"', body).group(1)),
            "parts": parts,
            "pins": {
                int(n): (float(x), float(y))
                for n, x, y in re.findall(rf'id="pin-{number}-(\d+)" style="--x:([\d.]+)%;--y:([\d.]+)%', body)
            },
        })
    return steps


def _refuse_if_lossy(page: str, out: Path) -> None:
    """Fail before anything is deleted: real masks and the intro card are not rebuilt by this tool."""
    if re.search(r'id="step-0"|data-kind="intro"|class="card[^"]*\bintro\b', page):
        raise SystemExit("refusing: the page has the unnumbered intro card (step 0, kind intro), "
                         "which build_real_sample.py does not handle")
    labels = out / "labels"
    stray = sorted(m for m in labels.rglob("*mask*.png") if m != labels / "no-mask.png") if labels.is_dir() else []
    if stray:
        raise SystemExit(f"refusing: {out / 'labels'} holds {len(stray)} real mask file(s) (e.g. {stray[0].relative_to(labels)}); "
                         "a rebuild would delete them. Use a fresh out-dir.")
    try:
        old = json.loads((out / "guide.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if any(s.get("number") == 0 or s.get("kind") == "intro" for s in old.get("steps", [])):
        raise SystemExit(f"refusing: {out / 'guide.json'} has the intro card, which a rebuild would drop")
    real = [lb["detection"]["mask_path"] for lb in old.get("labels", [])
            if lb.get("detection", {}).get("mask_path") != "labels/no-mask.png"]
    if real:
        raise SystemExit(f"refusing: {out} already holds {len(real)} real mask(s); a rebuild would delete "
                         "labels/ and point every label at no-mask.png. Use a fresh out-dir.")


def build(page_path: Path, images: Path, clip: Path, out: Path) -> Path:
    page = page_path.read_text(encoding="utf-8")
    _refuse_if_lossy(page, out)
    reasons = [html.unescape(r) for r in re.findall(r"<li>(DRAFT: [^<]*)</li>", page)]
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("frames", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir()
    steps, labels, frame_paths = [], [], []
    for index, step in enumerate(_steps(page)):
        frame = f"frames/step-{step['number']:02d}.png"
        shutil.copyfile(images / f"{step['frame']}.png", out / frame)
        frame_paths.append(frame)
        steps.append({
            "number": step["number"], "start_s": step["start_s"], "end_s": step["end_s"],
            "action": step["action"], "tools": step["tools"],
            "parts": [{"phrase": p, "position_hint": ""} for p, _, _ in step["parts"]],
            "hazards": step["hazards"],
            "time_spread_s": UNSEEN_SPREAD_S if step["check"] else 1.0,
            "time_needs_check": step["check"], "kind": "repair",
        })
        for part_index, (phrase, ok, labelled) in enumerate(step["parts"]):
            if labelled is None:
                continue
            x, y = step["pins"][part_index + 1]
            cx, cy = round(x / 100 * FRAME_W), round(y / 100 * FRAME_H)
            rendered = f"labels/step-{step['number']:02d}-part-{part_index + 1}.png"
            shutil.copyfile(images / f"{labelled}.png", out / rendered)
            labels.append({
                "slot": {"step_index": index, "part_index": part_index,
                         "part": {"phrase": phrase, "position_hint": ""}},
                "detection": {"phrase": phrase, "mask_path": "labels/no-mask.png", "score": 1.0,
                              "box_xyxy": [cx - PIN_HALF, cy - PIN_HALF, cx + PIN_HALF, cy + PIN_HALF]},
                "provisional": {"artifact_id": secrets.token_hex(16), "rendered_image_path": rendered,
                                "position_hint": ""},
                "checked": ok, "status": "part_pixels_unchanged" if ok else "not_checked",
                "readback": "yes" if ok else "not_run",
            })
    guide = {
        "job_id": secrets.token_hex(16), "steps": steps, "excluded_spans": [],
        "frame_paths": frame_paths, "labels": labels,
        "difficulty": DIFFICULTY, "verdict": VERDICT, "draft_reasons": reasons,
    }
    (out / "guide.json").write_text(json.dumps(guide, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(clip, out / "sample.mp4")
    return out


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    if len(args) not in (3, 4):
        sys.exit(__doc__)
    build(*args[:3], args[3] if len(args) == 4 else Path(__file__).resolve().parents[1] / "app" / "sample_real")
