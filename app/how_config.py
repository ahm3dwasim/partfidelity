"""Strict loader for config/how_it_works.toml (stdlib only): the /how-it-works page copy.

A missing, misspelled or mistyped key raises HowConfigError, and so does a ``{placeholder}`` the facts below
do not know. ``facts`` reads every number from the app itself (contracts, frame scoring, settings), so the
page cannot drift from the code; ``context`` returns the filled-in copy.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib
import math

from app import frame_scoring, hazards, how_media
from app.contracts import (
    READBACK_CONFIDENCE_CUTOFF,
    MAX_READBACKS_PER_JOB,
    MAX_STEPS,
    PART_CHECK_THRESHOLDS,
    SAM_MAX_CALLS_PER_JOB,
    SAM_MAX_REQUEST_STARTS_PER_MINUTE,
    SPARK_RUNS_PER_GUIDE,
    TIME_SPREAD_FLAG_S,
    UPLOAD_PREVIEW_TTL_HOURS,
)
from app.settings import (
    DEFAULT_CHUNK_OVERLAP_S,
    DEFAULT_CHUNK_S,
    DEFAULT_MAX_VIDEO_S,
    DEFAULT_SAM_MODEL_ID,
    DEFAULT_SPARK_MODEL_ID,
)

_EXAMPLE_TEXT = (
    "clip_name provenance readback_note q_text yes_text no_text kept_text score_text below_text steps_head runs_head "
    "median_text spread_text badge_text difficulty_label difficulty frames_head guide_step_word guide_note"
).split()
_EXAMPLE_KEYS = set(_EXAMPLE_TEXT) | {"step_index", "sam", "strip", "steps", "runs", "guide", "candidates"}

CONFIG = Path(__file__).resolve().parents[1] / "config" / "how_it_works.toml"


class HowConfigError(ValueError):
    pass


def _str(v):
    return isinstance(v, str) and bool(v)


_PAGE = (
    "title meta_description eyebrow headline intro bench_heading bench_hint figure_label fig_title fig_hint concept_note "
    "play pause detail_label hood_heading hood_intro limits_heading limits_body calls_link cta_heading "
    "cta_sample cta_upload in_word out_word stage_word stepper_label illustrative"
).split()
_STAGE = ("id", "part", "kind", "title", "plain", "readout", "model", "detail", "in_label", "in_note", "out_label", "out_note", "op")
_TECH = ("name", "does", "facts")
_KINDS = ("code", "ai", "rules")
STAGE_COUNT = 6


def _check(table, keys, where):
    if not isinstance(table, dict):
        raise HowConfigError(f"{where}: expected a table")
    if set(table) != set(keys):
        raise HowConfigError(
            f"{where}: keys differ (missing {sorted(set(keys) - set(table))}, unknown {sorted(set(table) - set(keys))})"
        )
    for key in keys:
        if not _str(table[key]):
            raise HowConfigError(f"{where}.{key}: expected a non-empty string")


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _nums(v, n, where, lo=0.0):
    if not (isinstance(v, list) and len(v) == n and all(_number(x) and x >= lo for x in v)):
        raise HowConfigError(f"{where}: expected {n} numbers")


def _rows(table, key, keys, where, minimum=1):
    rows = table[key]
    if not isinstance(rows, list) or len(rows) < minimum:
        raise HowConfigError(f"{where}.{key}: expected a list of at least {minimum} tables")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(keys):
            raise HowConfigError(f"{where}.{key}[{i}]: keys must be {sorted(keys)}")
    return rows


def _check_example(ex):
    if not isinstance(ex, dict) or set(ex) != _EXAMPLE_KEYS:
        raise HowConfigError(f"example: keys differ (missing {sorted(_EXAMPLE_KEYS - set(ex or {}))}, unknown {sorted(set(ex or {}) - _EXAMPLE_KEYS)})")
    for key in _EXAMPLE_TEXT:
        if not _str(ex[key]):
            raise HowConfigError(f"example.{key}: expected a non-empty string")
    steps = _rows(ex, "steps", ("start", "end", "title"), "example")
    for i, st in enumerate(steps):
        if not _str(st["title"]) or not all(_number(st[k]) for k in ("start", "end")) or not st["end"] > st["start"] >= 0:
            raise HowConfigError(f"example.steps[{i}]: needs a title and 0 <= start < end")
    idx = ex["step_index"]
    if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(steps):
        raise HowConfigError("example.step_index: must point at one of the steps")
    runs = _rows(ex, "runs", ("start", "end"), "example", minimum=SPARK_RUNS_PER_GUIDE)
    if len(runs) != SPARK_RUNS_PER_GUIDE:
        raise HowConfigError(f"example.runs: the app asks {SPARK_RUNS_PER_GUIDE} times, found {len(runs)}")
    for i, r in enumerate(runs):
        if not all(_number(r[k]) for k in ("start", "end")) or not r["end"] > r["start"] >= 0:
            raise HowConfigError(f"example.runs[{i}]: needs 0 <= start < end")
    for i, row in enumerate(_rows(ex, "strip", ("t", "file"), "example")):
        if not _number(row["t"]) or row["t"] < 0 or not _str(row["file"]) or Path(row["file"]).name != row["file"]:
            raise HowConfigError(f"example.strip[{i}].file: a bare file name in static/ui/media/how/")
    cands = _rows(ex, "candidates", ("t", "sharpness", "motion"), "example", minimum=frame_scoring.FRAMES_KEPT_PER_STEP + 1)
    for i, c in enumerate(cands):
        _nums([c["t"], c["sharpness"], c["motion"]], 3, f"example.candidates[{i}]")
    if not any(c["sharpness"] > 0 for c in cands):
        raise HowConfigError("example.candidates: at least one score must be positive")
    sam = ex["sam"]
    if not isinstance(sam, dict) or set(sam) != {"file", "frame_t", "size", "box", "confidence", "part"}:
        raise HowConfigError("example.sam: keys differ")
    _nums(sam["size"], 2, "example.sam.size", 1)
    _nums(sam["box"], 4, "example.sam.box")
    x0, y0, x1, y1 = sam["box"]
    if not (x0 < x1 <= sam["size"][0] and y0 < y1 <= sam["size"][1]):
        raise HowConfigError("example.sam.box: must sit inside the frame")
    if sam["part"] not in PART_CHECK_THRESHOLDS or not _str(sam["file"]) or Path(sam["file"]).name != sam["file"]:
        raise HowConfigError("example.sam: part must be one the app checks, file a bare name")
    conf = sam["confidence"]
    if not _number(conf) or not PART_CHECK_THRESHOLDS[sam["part"]] <= conf < READBACK_CONFIDENCE_CUTOFF:
        raise HowConfigError("example.sam.confidence: must sit between the part minimum and the read-back cutoff, or stage 5 has nothing to show")
    if ex["sam"]["frame_t"] not in {c["t"] for c in cands}:
        raise HowConfigError("example.sam.frame_t: must be one of the scored candidate times")
    scored = [frame_scoring.ScoredFrame(c["t"], c["sharpness"], c["motion"]) for c in cands]
    files = [row["file"] for row in ex["strip"]] + [sam["file"]]
    files += [f"c{c['t']:g}.webp" for c in cands]
    files += [f"k{f.timestamp_s:g}.webp" for f in frame_scoring.pick_best_two(scored)]
    unknown = sorted(set(files) - how_media.names(CONFIG.parents[1]))
    if unknown:  # the frames themselves are provisioned (tools/how_media.py); the page degrades without them
        raise HowConfigError(f"example: image files not in config/how_media.json: {unknown}")
    g = ex["guide"]
    if not isinstance(g, dict) or set(g) != {"hazards"} or not g["hazards"] or any(h not in hazards.HAZARD_TEMPLATES for h in g["hazards"]):
        raise HowConfigError("example.guide.hazards: names must be hazard classes the app has wording for")


def load(path: Path = CONFIG) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    if set(cfg) != {"page", "kinds", "stages", "tech", "names", "motion", "example"}:
        raise HowConfigError(f"top level keys differ: {sorted(cfg)}")
    page = cfg["page"]
    if isinstance(page, dict) and "h1" in page and "headline" not in page:  # `h1` is the former name of `headline`
        cfg = {**cfg, "page": {("headline" if k == "h1" else k): v for k, v in page.items()}}
    _check(cfg["page"], _PAGE, "page")
    _check(cfg["kinds"], _KINDS, "kinds")
    for name, keys in (("stages", _STAGE), ("tech", _TECH)):
        rows = cfg[name]
        if not isinstance(rows, list) or not rows:
            raise HowConfigError(f"{name}: expected a non-empty list of tables")
        for i, row in enumerate(rows):
            _check(row, keys, f"{name}[{i}]")
    if len(cfg["stages"]) != STAGE_COUNT:
        raise HowConfigError(f"stages: the figure has {STAGE_COUNT} stations, found {len(cfg['stages'])}")
    if len({s["id"] for s in cfg["stages"]}) != STAGE_COUNT:
        raise HowConfigError("stages: duplicate id")
    for i, st in enumerate(cfg["stages"]):
        if st["kind"] not in _KINDS:
            raise HowConfigError(f"stages[{i}].kind: one of {list(_KINDS)}")
    motion = cfg["motion"]
    limits = {"dwell_s": (0, None), "follow_s": (0, None), "reading_position": (0, 1),
              "lift": (0, None), "brightness_inactive": (0, 1), "brightness_active": (0, 1), "clearance": (0, None)}
    if not isinstance(motion, dict) or set(motion) != set(limits) or any(
        not _number(motion[k]) or motion[k] < lo or (hi is not None and motion[k] > hi)
        for k, (lo, hi) in limits.items()
    ) or motion["dwell_s"] == 0 or motion["follow_s"] == 0:
        raise HowConfigError("motion: invalid numeric settings")
    if not isinstance(cfg["names"], dict) or not all(_str(k) and _str(v) for k, v in cfg["names"].items()):
        raise HowConfigError("names: expected id = display name strings")
    _check_example(cfg["example"])
    context(cfg)  # every placeholder must resolve against the app's own facts
    return cfg


@lru_cache(maxsize=1)
def cached() -> dict:
    return load()


def _clock(seconds: float) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        n = int(seconds // 60)
        return f"{n} minute" + ("" if n == 1 else "s")
    return f"{int(seconds)} seconds"


def _join(words) -> str:
    words = list(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def facts(cfg: dict, *, spark_model_id: str = DEFAULT_SPARK_MODEL_ID, sam_model_id: str = DEFAULT_SAM_MODEL_ID,
          open_model_id: str | None = None,
          max_video_s: float = DEFAULT_MAX_VIDEO_S, chunk_s: float = DEFAULT_CHUNK_S,
          chunk_overlap_s: float = DEFAULT_CHUNK_OVERLAP_S) -> dict:
    """Every number the copy may quote, read from the code that enforces it."""
    names = cfg["names"]
    if open_model_id is None:  # the open-weights checker's own model config; lazy so this module stays stdlib-only
        from app.providers.open_readback import load_config as _load_open

        open_model_id = _load_open().model_id
    return {
        "spark": names.get(spark_model_id, spark_model_id),
        "sam": names.get(sam_model_id, sam_model_id),
        "spark_open": names.get(open_model_id, open_model_id),
        "runs": SPARK_RUNS_PER_GUIDE,
        "spread_s": TIME_SPREAD_FLAG_S,
        "max_steps": MAX_STEPS,
        "max_video": _clock(max_video_s),
        "ttl_hours": UPLOAD_PREVIEW_TTL_HOURS,
        "chunk_s": f"{chunk_s:g}",
        "overlap_s": f"{chunk_overlap_s:g}",
        "fps": f"{frame_scoring.SAMPLE_FPS:g}",
        "best_frames": frame_scoring.FRAMES_KEPT_PER_STEP,
        "sam_parts": _join(PART_CHECK_THRESHOLDS),
        "sam_thresholds": ", ".join(f"{p} {t:g}" for p, t in PART_CHECK_THRESHOLDS.items()),
        "sam_calls": SAM_MAX_CALLS_PER_JOB,
        "sam_pace": SAM_MAX_REQUEST_STARTS_PER_MINUTE,
        "cutoff": f"{READBACK_CONFIDENCE_CUTOFF:.2f}",
        "max_readbacks": MAX_READBACKS_PER_JOB,
    }


def _fill(row: dict, values: dict, where: str) -> dict:
    out = {}
    for key, text in row.items():
        try:
            out[key] = text.format_map(values)
        except (KeyError, IndexError, ValueError) as error:
            raise HowConfigError(f"{where}.{key}: bad placeholder {error}") from None
    return out


def context(cfg: dict, **settings) -> dict:
    """The page's copy with the app's real numbers filled in."""
    values = facts(cfg, **settings)
    return {
        "page": _fill(cfg["page"], values, "page"),
        "kinds": dict(cfg["kinds"]),
        "stages": [_fill(s, values, f"stages[{i}]") for i, s in enumerate(cfg["stages"])],
        "tech": [_fill(t, values, f"tech[{i}]") for i, t in enumerate(cfg["tech"])],
        "example": example(cfg),
    }


def first_sentences(text: str, n: int = 2) -> str:
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    return ". ".join(parts[:n]).rstrip(".") + "."


def example(cfg: dict) -> dict:
    """The recorded example with the facts derived from the app: kept frames are chosen by the app's own ranking."""
    ex = cfg["example"]
    scored = [frame_scoring.ScoredFrame(c["t"], c["sharpness"], c["motion"]) for c in ex["candidates"]]
    kept = frame_scoring.pick_best_two(scored)
    spread = max(r["end"] for r in ex["runs"]) - min(r["end"] for r in ex["runs"])
    out = dict(ex)
    out.update(
        scored=scored,
        kept=[f.timestamp_s for f in kept],
        best_score=max(f.score for f in scored),
        spread=spread,
        warnings=[first_sentences(hazards.HAZARD_TEMPLATES[h], 2 if h == "battery" else 1) for h in ex["guide"]["hazards"]],
    )
    return out
