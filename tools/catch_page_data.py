"""Generate config/catch_table.json (the /catch page's numbers and rows) from the frozen evaluate.json.

    python tools/catch_page_data.py path/to/evaluate.json

A row is ACCEPTED only when the gate's reason is "faithful edit"; every other reason is a rejection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPT_REASON = "faithful edit"
REDRAWN = "redrawn-edit"
LEGACY_REDRAWN = frozenset({REDRAWN, "owner-edit"})  # "owner-edit" is the former name in older evaluate.json files


def build(evaluate: dict, media: dict) -> dict:
    rows, number = [], 0
    for r in evaluate["rows"]:
        recipe = REDRAWN if r["recipe"] in LEGACY_REDRAWN else r["recipe"]
        if recipe == REDRAWN and r["clip"] == "H1":
            number += 1
        rows.append({
            "number": number if recipe == REDRAWN and r["clip"] == "H1" else None,
            "clip": r["clip"], "split": r["split"], "candidate": r["candidate"],
            "name": Path(r["candidate"]).stem, "recipe": recipe, "label": r["label"],
            "iou": r["iou"], "ssim": r["ssim"], "reason": r["reason"],
            "accepted": r["reason"] == ACCEPT_REASON,
        })

    def count(pick):
        sel = [r for r in rows if pick(r)]
        return {"total": len(sel), "accepted": sum(r["accepted"] for r in sel), "rejected": sum(not r["accepted"] for r in sel)}

    by_candidate = {r["candidate"]: r for r in rows}
    cases = []
    for c in media["cases"]:
        row = by_candidate[c["candidate"]]
        cases.append({"key": c["key"], "row": row, "original": c["original"] + ".jpg", "edited": c["edited"] + ".jpg"})
    return {
        "caption": evaluate["caption"], "eval_limits": evaluate["eval_limits"], "ssim_cutoff": evaluate["ssim_cutoff"],
        "summary": {
            "altered_edit": count(lambda r: r["recipe"] == REDRAWN and r["label"] == "altered"),
            "faithful_edits": count(lambda r: r["recipe"] == REDRAWN and r["label"] == "faithful"),
            "heldout_recipes": count(lambda r: r["split"] == "heldout"),
            "heldout_blur": count(lambda r: r["split"] == "heldout" and r["recipe"] == "blur"),
        },
        "rows": rows, "cases": cases,
    }


if __name__ == "__main__":
    evaluate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    media = json.loads((ROOT / "config/catch_media.json").read_text(encoding="utf-8"))
    (ROOT / "config/catch_table.json").write_text(json.dumps(build(evaluate, media), indent=1) + "\n", encoding="utf-8")
