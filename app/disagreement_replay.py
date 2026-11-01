"""The simulated disagreement replay page content (lane mh-disagreement, W-2816).

Copy comes from ``config/disagreement_replay.toml`` and the shared flag wording from
``config/open_readback.toml``, both through strict loaders. Nothing here calls a model.
The two recorded probe runs (paths in ``config/disagreement_probe.toml``) are shown above the replay
as "Recorded probe"; the replay is never presented as an observed disagreement.
"""
from __future__ import annotations

import json
import tomllib
from html import escape
from pathlib import Path

from app.providers import open_readback

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "disagreement_replay.toml"
PROBE_CONFIG = ROOT / "config" / "disagreement_probe.toml"
KEYS = frozenset("""title kicker banner replay_heading recorded_model_line intro separate_text separate_link_text separate_link card_heading label_caption
shown_label rename_input_label rename_button rename_empty state_renamed persist_note persist_failed_note static_heading static_steps
recorded_heading recorded_intro recorded_columns recorded_runs recorded_receipt_text recorded_limit storage_key""".split())
PROBE_URL = "/disagreement-replay/probe.json"          # run 2
EARLIER_URL = "/disagreement-replay/probe-run-1.json"  # run 1


def load(path: Path = CONFIG) -> dict:
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Invalid replay config: {error}") from error
    page = cfg.get("replay")
    if set(cfg) != {"replay"} or not isinstance(page, dict) or set(page) != KEYS:
        raise ValueError("Invalid replay config: unexpected or missing keys")
    steps = page["static_steps"]
    if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s.strip() for s in steps):
        raise ValueError("Invalid replay config: static_steps must be a non-empty list of text")
    for key in ("recorded_columns", "recorded_runs"):
        if not isinstance(page[key], list) or not all(isinstance(s, str) and s.strip() for s in page[key]):
            raise ValueError(f"Invalid replay config: {key} must be a list of text")
    if len(page["recorded_columns"]) != 4 or len(page["recorded_runs"]) != 2:
        raise ValueError("Invalid replay config: recorded_columns needs 4 names and recorded_runs 2")
    if "not an accuracy measure" not in page["recorded_limit"].lower():
        raise ValueError("Invalid replay config: recorded_limit must say two runs are not an accuracy measure")
    for key, value in page.items():
        if key not in ("static_steps", "recorded_columns", "recorded_runs") and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Invalid replay config: {key} must be non-empty text")
    if "{model}" not in page["recorded_model_line"] or "{date}" not in page["recorded_model_line"]:
        raise ValueError("Invalid replay config: recorded_model_line needs {model} and {date}")
    if "{label}" not in page["state_renamed"] or not page["separate_link"].startswith("/"):
        raise ValueError("Invalid replay config: state_renamed needs {label}; separate_link must be a site path")
    if "simulated" not in page["banner"].lower() or "no model was called" not in page["banner"].lower():
        raise ValueError("Invalid replay config: the banner must say the replay is simulated and no model was called")
    return page


def _receipt(key: str) -> tuple[Path, dict] | None:
    try:
        rel = tomllib.loads(PROBE_CONFIG.read_text(encoding="utf-8"))["probe"][key]
        path = ROOT / rel
        data = json.loads(path.read_text(encoding="utf-8"))
        ok = data.get("schema") == "disagreement-probe/1" and isinstance(data.get("cases"), list) and data["cases"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return (path, data) if ok else None


def probe_receipt_path(key: str = "receipt_path") -> Path | None:
    """A recorded probe receipt (run 2 by default; "earlier_receipt_path" is run 1) when it looks like one; else None."""
    found = _receipt(key)
    return found[0] if found else None


def _recorded(cfg: dict, e) -> str:
    rows, models = "", []
    cols = ["Run", *cfg["recorded_columns"]]
    for key, url in (("earlier_receipt_path", EARLIER_URL), ("receipt_path", PROBE_URL)):
        found = _receipt(key)
        if found is None:
            continue
        data = found[1]
        case = next((c for c in data["cases"] if c.get("case") != "control"), data["cases"][0])
        label = cfg["recorded_runs"][0 if key == "earlier_receipt_path" else 1]
        for c in (case, *[c for c in data["cases"] if c is not case]):
            model = cfg["recorded_model_line"].format(model=data.get("model_id"), date=str(c.get("started_at", ""))[:10])
            if model not in models:
                models.append(model)
            cells = (label, str(c.get("request_label")), str(c.get("model_answer")), str(c.get("agreement")))
            rows += ("<tr>" + "".join(f'<td data-label="{e(n)}">{e(v)}</td>' for n, v in zip(cols, cells))
                     + f'<td data-label="{e(cols[4])}"><a href="{url}">{e(cfg["recorded_receipt_text"])}</a></td></tr>')
    if not rows:
        return ""
    head = "".join(f"<th>{e(c)}</th>" for c in cols)
    return (f'<section class="built-section replay-recorded"><h2>{e(cfg["recorded_heading"])}</h2><p>{e(cfg["recorded_intro"])}</p>'
            f'<p class="replay-model">{e("; ".join(models))}</p>'
            f'<div class="replay-table-wrap"><table class="replay-table"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
            f'<p>{e(cfg["recorded_limit"])}</p></section>')


def render_body(cfg: dict | None = None, flags: open_readback.OpenReadbackConfig | None = None) -> dict:
    """Escaped pieces for templates/replay.html."""
    cfg = cfg or load()
    flags = flags or open_readback.load_config()
    e = lambda v: escape(v, quote=True)  # noqa: E731
    steps = "".join(f"<li>{e(s)}</li>" for s in cfg["static_steps"])
    recorded = _recorded(cfg, e)
    data = {
        "storage": cfg["storage_key"], "label": cfg["shown_label"], "confirmed": flags.flag_confirmed_text,
        "flagged": flags.flag_text, "renamed": cfg["state_renamed"], "empty": cfg["rename_empty"],
        "persist-failed": cfg["persist_failed_note"],
    }
    attrs = " ".join(f'data-{k}="{e(v)}"' for k, v in data.items())
    return {
        **{k: e(cfg[k]) for k in ("title", "kicker", "banner", "replay_heading", "intro", "separate_text", "separate_link_text",
                                  "card_heading", "label_caption", "rename_input_label", "rename_button",
                                  "persist_note", "static_heading")},
        "separate_link": e(cfg["separate_link"]), "shown_label": e(cfg["shown_label"]), "steps": steps,
        "recorded": recorded, "attrs": attrs, "flagged_text": e(flags.flag_text),
        "confirm_label": e(flags.flag_confirm_label), "undo_label": e(flags.flag_undo_label),
        "aria_label": e(flags.flag_aria_label),
    }
