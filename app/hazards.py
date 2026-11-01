"""Deterministic repair guardrails: refusal, warnings, battery order, difficulty, stops.

Stage 6 runs on Spark's fallible tags and part words, but every
safety sentence comes from a fixed template: models propose tags
and never author safety text. One ordered rule drives the maker view, the
reader view, and the later offline export.

The reader module consumes ``HazardAssessment``, ``DifficultyResult`` and ``ReaderStop``.
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Callable, Literal, Sequence
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.contracts import SAM_MAX_CALLS_PER_JOB, Step
from app.guide_text import DRAFT_ORDER_REASON

__all__ = (
    "DRAFT_BANNER",
    "DRAFT_ORDER_REASON",
    "HAZARD_TEMPLATES",
    "HARD_STOP_TITLE",
    "HARD_STOP_TEXT",
    "HARD_STOP_ACK_PATH",
    "HARD_STOP_READER_COOKIE",
    "UNKNOWN_HAZARD_TAG",
    "UNKNOWN_HAZARD_TEXT",
    "UNKNOWN_STOP_TEXT",
    "CTA_PROFESSIONAL",
    "CTA_HELP",
    "CTA_TRY",
    "CTA_SAMPLE",
    "refusal_copy",
    "HazardWarning",
    "BatteryOrderEvidence",
    "HazardAssessment",
    "DifficultyResult",
    "ReaderStop",
    "HardStopAcknowledgements",
    "assess_hazards",
    "step_hazard_blocks",
    "difficulty_for",
    "reader_stop",
    "stop_reasons",
    "hidden_step_numbers",
    "deliverable_steps",
    "hard_stop_acknowledgement_router",
    "normalize_tag",
)

# Universal banner, shown on every guide and export (verbatim).
DRAFT_BANNER = (
    "AI-assisted draft. Not reviewed by a qualified technician. "
    "Not endorsed by any device maker. Repairs can cause injury, fire, "
    "data loss and may void warranties. You follow these steps at your own risk."
)


# Fixed warning text per hazard class. The taxonomy names `sharp` but
# gives it no fixed text, so its warning below is our own wording.
_ADHESIVE_HEAT_TEXT = "Heat and adhesive can burn. Let parts cool."
HAZARD_TEMPLATES: dict[str, str] = {
    "battery": (
        "Power off and unplug. Disconnect the battery before touching "
        "anything else. Do not pierce, bend or use metal tools on a cell. "
        "If a battery is swollen, stop; do not pry it. Take it to e-waste."
    ),
    "adhesive/heat": _ADHESIVE_HEAT_TEXT,
    "display": "Glass and sharp edges. Wear eye protection when prying.",
    "esd": "Touch grounded metal or use a wrist strap.",
    "data": "Back up your data first; repairs can erase it.",
    "sharp": "Sharp edges and points. Cut away from yourself and wear gloves.",
}

# Reader hard stop (verbatim; the markdown emphasis is the
# renderer's job). The stop itself is never shortened.
HARD_STOP_TITLE = "Stop here."
HARD_STOP_TEXT = (
    "Stop here. This step involves a battery or a glued screen. "
    "These can catch fire or shatter. See a professional."
)
HARD_STOP_ACK_PATH = "/hard-stops/acknowledge"
HARD_STOP_READER_COOKIE = "partfidelity_reader"
HARD_STOP_ACK_TTL_SECONDS = 24 * 60 * 60
HARD_STOP_ACK_MAX_ENTRIES = 4096
_HARD_STOP_GUIDE_MAX_ENTRIES = 4096
_HARD_STOP_FORM_MAX_BYTES = 2048

# Generic fallback for unrecognized or ambiguous hazard tags ("explosive",
# "battery?") and out-of-scope tag concepts at step level. Fixed text, like
# every other safety sentence: never silence, always a warning plus a stop.
UNKNOWN_HAZARD_TAG = "unknown"
UNKNOWN_HAZARD_TEXT = (
    "Unrecognized or out-of-scope hazard. "
    "Stop and see a professional before continuing."
)
UNKNOWN_STOP_TEXT = (
    "Stop here. This step has an unrecognized or out-of-scope hazard. "
    "Do not continue. See a professional."
)

# Ordered-difficulty CTAs.
CTA_PROFESSIONAL = "See a professional"
CTA_HELP = "Try it with help"
CTA_TRY = "Try it"
CTA_SAMPLE = "Try the sample guide"

# Scope refusal (verbatim) plus the sample CTA from the operational
# refusal line, so a refused page always offers the sample.
_SCOPE_REFUSAL_TEXT = (
    "This tool handles phones and laptops only. It will not make guides for "
    "mains-powered devices, e-bike or high-voltage packs, swollen batteries "
    "or anything that was on fire."
)
_NOT_A_TEARDOWN_TEXT = "We could not find a teardown."
_DURATION_REFUSAL_TEXT = "This video is longer than the tool accepts. Trim it and try again."
_BUDGET_REFUSAL_TEXT = "This video is too long to process within the cost limit. Trim it and try again."

# Spark hazard taxonomy. `mains` is not a warning: it refuses.
# `heat` shares the adhesive/heat template but stays an "other hazard" for
# difficulty, per the ordered rule's literal Hard row (battery/adhesive/display).
BATTERY_TAG = "battery"
_HARD_TAGS = frozenset({"battery", "adhesive", "display"})
_WARNING_TAGS = ("battery", "adhesive", "heat", "display", "esd", "data", "sharp")
_ADHESIVE_CLASS = {"adhesive", "heat"}

# Scope refusal patterns over device_guess + step actions + hazard tags
# (brief: mains, swollen cells, e-bike/high-voltage packs, fire,
# out-of-scope devices). Word-boundaried so "firmware" never refuses.
# Refusal concepts count in every input field, tags included.
_SCOPE_PATTERNS = (
    re.compile(r"\bmains\b"),
    re.compile(r"\bswollen\b"),
    re.compile(r"\be[\s-]?bike\b"),
    re.compile(r"\bhigh[\s-]?voltage\b"),
    re.compile(r"\btablets?\b"),
    re.compile(r"\bsmart[\s-]?watch(?:es)?\b"),
    re.compile(r"\bwatch(?:es)?\b"),
    re.compile(r"\bev\b"),
    re.compile(r"\bfire\b"),
    re.compile(r"\bburning\b"),
    re.compile(r"\bsmoke\b"),
    re.compile(r"\bflames?\b"),
    re.compile(
        r"\b(television|microwave|dishwasher|washing machine|dryer|"
        r"refrigerator|fridge|freezer|air conditioner|oven|heater|"
        r"desktop|monitor|server)\b"
    ),
)

_SCOPE_DASH_EXTRAS = frozenset({"\u00ad", "\u2212", "\ufe63", "\uff0d"})

_DEVICE_SCOPE_FILE = Path(__file__).with_name("device_scope.toml")
_DEVICE_SCOPE_KEYS = (
    "phone_words", "laptop_words", "reject_words",
    "brand_words", "model_words", "article_words",
)
_LETTER_RUN = re.compile(r"[a-z]+")
# A short letter-hyphen-digit run ("t-480") is one model-number token; longer
# words keep splitting so "phone-toaster" still lands on an unlisted word.
_GUESS_TOKEN = re.compile(r"[a-z]{1,2}-[0-9][a-z0-9]*|[a-z0-9]+")


def load_device_scope(path: Path | str = _DEVICE_SCOPE_FILE) -> dict[str, frozenset[str]]:
    """Read the device-class word lists; a missing or misspelled key fails loudly."""
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    section = raw.get("device_scope")
    if not isinstance(section, dict) or set(section) != set(_DEVICE_SCOPE_KEYS):
        raise RuntimeError(
            "device scope config needs [device_scope] with exactly "
            + ", ".join(_DEVICE_SCOPE_KEYS)
        )
    words: dict[str, frozenset[str]] = {}
    for key in _DEVICE_SCOPE_KEYS:
        value = section[key]
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(w, str) and _LETTER_RUN.fullmatch(w) for w in value)
        ):
            raise RuntimeError(f"device scope {key} must be a non-empty list of lower-case words")
        words[key] = frozenset(value)
    seen: dict[str, str] = {}
    for key in _DEVICE_SCOPE_KEYS:
        for word in words[key]:
            if word in seen:
                raise RuntimeError(
                    f"device scope word {word!r} is in both {seen[word]} and {key}"
                )
            seen[word] = key
    return words


_DEVICE_SCOPE = load_device_scope()


def _names_one_supported_device(guess: str) -> bool:
    """True when every token is allowlisted and exactly one family is named."""
    tokens = _GUESS_TOKEN.findall(guess)
    scope = _DEVICE_SCOPE
    known = (
        scope["brand_words"] | scope["model_words"] | scope["article_words"]
        | scope["phone_words"] | scope["laptop_words"]
    )
    for token in tokens:
        if not any(c.isdigit() for c in token) and token not in known:
            return False  # reject words and any unlisted word land here
    names = set(tokens)
    return bool(names & scope["phone_words"]) != bool(names & scope["laptop_words"])


# Hedge words and marks that make a guess ambiguous, never allowlisted.
_AMBIGUOUS_GUESS = re.compile(
    r"\?|\b(uncertain|unsure|unknown|maybe|possibly|probably|unclear|ambiguous)\b"
)

# Battery disconnect must be shown, not inferred. Structured steps accept the
# exact action `disconnect` plus one exact battery object. Text-only steps
# accept one whole imperative and nothing around it. Everything else is DRAFT.
_TEXT_DISCONNECT = re.compile(
    r"disconnect (?:the )?battery (?:connector|cable)\.?"
)
_STRUCTURED_DISCONNECT_ACTIONS = frozenset({"disconnect", "disconnect."})
_STRUCTURED_BATTERY_OBJECTS = frozenset(
    {
        "battery connector",
        "the battery connector",
        "battery cable",
        "the battery cable",
    }
)
_FORM_IDENTIFIER = re.compile(r"[A-Za-z0-9._:-]{1,128}")

# Glued-screen wording: glue words plus a screen/back word. A glued phone back
# counts as a hard-stop beat.
_GLUE_WORD = re.compile(r"\bglu")
_SCREEN_WORDS = ("screen", "display", "glass", "back", "panel")


@dataclass(frozen=True)
class HazardWarning:
    """One fixed warning block: template text plus the steps carrying it."""

    tag: str  # canonical class: "adhesive/heat" merges those two Spark tags
    text: str
    steps: tuple[int, ...]


@dataclass(frozen=True)
class BatteryOrderEvidence:
    """Battery-first order check, exposed for the two-click walk-through."""

    battery_steps: tuple[int, ...]
    disconnect_step: int | None
    battery_first: bool
    detail: str


@dataclass(frozen=True)
class HazardAssessment:
    """Whole-guide safety verdict. Refused guides authorize zero SAM calls."""

    refused: bool
    refusal_kind: Literal["none", "not_a_teardown", "scope"]
    refusal_text: str | None
    sam_authorized: int
    warnings: tuple[HazardWarning, ...]
    draft: bool
    draft_reasons: tuple[str, ...]
    battery_order: BatteryOrderEvidence


@dataclass(frozen=True)
class DifficultyResult:
    """Ordered difficulty row. A refused guide has no level, only the sample CTA."""

    refused: bool
    level: Literal["Easy", "Medium", "Hard"] | None
    cta: str


@dataclass(frozen=True)
class ReaderStop:
    """Hard stop for one step. `text` is fixed and never shortened."""

    step_number: int
    stop_id: str
    title: str
    text: str
    cta: str
    reason: Literal["battery", "glued_screen", "unknown_hazard"]


class HardStopAcknowledgements:
    """Server-owned reader acknowledgements for registered guide revisions.

    Entries expire after 24 hours. Acknowledgements and guides each have a
    hard cap of 4096, with oldest-first eviction.

    Registration replaces an older revision of the same guide, so stale
    revision POSTs fail. A reader id comes from an HttpOnly same-site cookie;
    acknowledgement is never shared between readers or inferred from browser
    storage. The integration layer may replace this adapter with its durable
    store without changing the POST contract.
    """

    def __init__(self, *, now: Callable[[], float] = monotonic) -> None:
        self._guides: dict[str, tuple[str, frozenset[str], str]] = {}
        self._acknowledged: dict[tuple[str, str, str, str], float] = {}
        self._now = now
        self._lock = Lock()

    @staticmethod
    def _valid_identifier(value: str) -> bool:
        return bool(_FORM_IDENTIFIER.fullmatch(value))

    def _purge_expired(self, now: float) -> None:
        self._acknowledged = {
            key: expires_at
            for key, expires_at in self._acknowledged.items()
            if expires_at > now
        }

    def _remove_guide_acknowledgements(self, guide_id: str) -> None:
        self._acknowledged = {
            key: expires_at
            for key, expires_at in self._acknowledged.items()
            if key[1] != guide_id
        }

    def register_guide(
        self,
        guide_id: str,
        guide_revision: str,
        stop_ids: Sequence[str],
        return_path: str,
    ) -> None:
        """Register the only current revision and stop ids for one guide."""
        stop_list = tuple(stop_ids)
        if (
            type(guide_id) is not str
            or type(guide_revision) is not str
            or type(return_path) is not str
            or any(type(stop_id) is not str for stop_id in stop_list)
        ):
            raise ValueError("invalid hard-stop guide registration")
        stop_set = frozenset(stop_list)
        if (
            not self._valid_identifier(guide_id)
            or not self._valid_identifier(guide_revision)
            or not stop_set
            or any(not self._valid_identifier(stop_id) for stop_id in stop_set)
            or not return_path.startswith("/")
            or return_path.startswith("//")
            or ".." in return_path
            or "\\" in return_path
            or "\r" in return_path
            or "\n" in return_path
        ):
            raise ValueError("invalid hard-stop guide registration")
        with self._lock:
            previous = self._guides.pop(guide_id, None)
            self._guides[guide_id] = (guide_revision, stop_set, return_path)
            if previous is not None and previous[0] != guide_revision:
                old_revision = previous[0]
                self._acknowledged = {
                    key: expires_at
                    for key, expires_at in self._acknowledged.items()
                    if not (key[1] == guide_id and key[2] == old_revision)
                }
            while len(self._guides) > _HARD_STOP_GUIDE_MAX_ENTRIES:
                evicted_guide_id = next(iter(self._guides))
                del self._guides[evicted_guide_id]
                self._remove_guide_acknowledgements(evicted_guide_id)

    def unregister_guide(self, guide_id: str) -> None:
        """Remove a guide and every acknowledgement bound to it."""
        if type(guide_id) is not str:
            raise ValueError("invalid hard-stop guide id")
        with self._lock:
            self._guides.pop(guide_id, None)
            self._remove_guide_acknowledgements(guide_id)

    def acknowledge(
        self,
        reader_id: str,
        guide_id: str,
        guide_revision: str,
        stop_id: str,
    ) -> str | None:
        """Record one valid acknowledgement and return its server-owned URL."""
        if (
            type(reader_id) is not str
            or type(guide_id) is not str
            or type(guide_revision) is not str
            or type(stop_id) is not str
        ):
            return None
        if not reader_id or len(reader_id) > 256:
            return None
        with self._lock:
            registered = self._guides.get(guide_id)
            if registered is None:
                return None
            current_revision, stop_ids, return_path = registered
            if guide_revision != current_revision or stop_id not in stop_ids:
                return None
            now = self._now()
            self._purge_expired(now)
            key = (reader_id, guide_id, guide_revision, stop_id)
            self._acknowledged.pop(key, None)
            self._acknowledged[key] = now + HARD_STOP_ACK_TTL_SECONDS
            while len(self._acknowledged) > HARD_STOP_ACK_MAX_ENTRIES:
                del self._acknowledged[next(iter(self._acknowledged))]
            return return_path

    def is_acknowledged(
        self,
        reader_id: str,
        guide_id: str,
        guide_revision: str,
        stop_id: str,
    ) -> bool:
        """Read server state; missing identifiers always fail closed."""
        if (
            type(reader_id) is not str
            or type(guide_id) is not str
            or type(guide_revision) is not str
            or type(stop_id) is not str
        ):
            return False
        if not reader_id or not guide_id or not guide_revision or not stop_id:
            return False
        with self._lock:
            self._purge_expired(self._now())
            registered = self._guides.get(guide_id)
            if registered is None:
                return False
            current_revision, stop_ids, _return_path = registered
            return (
                guide_revision == current_revision
                and stop_id in stop_ids
                and (reader_id, guide_id, guide_revision, stop_id)
                in self._acknowledged
            )


def _normalise_spaces(value: str) -> str:
    """Collapse whitespace; anything but an exact ``str`` reads as blank."""
    if type(value) is not str:
        return ""
    return " ".join(value.strip().lower().split())


def normalize_tag(tag: str) -> str:
    """Normalize one Spark hazard tag for taxonomy lookup.

    Trust boundary: tags arrive with Spark output, so anything but an
    exact ``str`` maps to the unknown-hazard tag — never silenced, and
    no overridable method of the caller's object is ever invoked.
    """
    if type(tag) is not str:
        return UNKNOWN_HAZARD_TAG
    return tag.strip().lower()


def _step_tags(step: Step) -> frozenset[str]:
    return frozenset(normalize_tag(t) for t in step.hazards)


def _unrecognized_tags(tags: frozenset[str]) -> frozenset[str]:
    """Non-empty tags outside the warning taxonomy: never silenced."""
    return frozenset(t for t in tags if t and t not in _WARNING_TAGS)


def _normalise_scope_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        "-"
        if unicodedata.category(char) == "Pd" or char in _SCOPE_DASH_EXTRAS
        else char
        for char in normalized
    ).lower()


def _scope_hit(device_guess: str, steps: Sequence[Step]) -> bool:
    tags = [t for s in steps for t in _step_tags(s)]
    haystack = _normalise_scope_text(
        " ".join([device_guess, *(s.action for s in steps), *tags])
    )
    return any(p.search(haystack) for p in _SCOPE_PATTERNS)


def _affirmative_disconnect(step: Step) -> bool:
    """Accept only an exact structured or whole-text battery disconnect."""
    action = _normalise_spaces(step.action)
    if step.parts:
        objects = tuple(_normalise_spaces(part.phrase) for part in step.parts)
        return (
            action in _STRUCTURED_DISCONNECT_ACTIONS
            and len(objects) == 1
            and objects[0] in _STRUCTURED_BATTERY_OBJECTS
        )
    return bool(_TEXT_DISCONNECT.fullmatch(action))


def _disconnect_step(steps: Sequence[Step]) -> int | None:
    for step in steps:
        if _affirmative_disconnect(step):
            return step.number
    return None


def _battery_order(steps: Sequence[Step]) -> BatteryOrderEvidence:
    battery_steps = tuple(s.number for s in steps if BATTERY_TAG in _step_tags(s))
    if not battery_steps:
        return BatteryOrderEvidence((), None, True, "no battery step; order rule vacuous")
    disconnect = _disconnect_step(steps)
    if disconnect is None:
        return BatteryOrderEvidence(
            battery_steps,
            None,
            False,
            f"battery steps {list(battery_steps)}; no disconnect shown",
        )
    first = min(battery_steps)
    if disconnect > first:
        return BatteryOrderEvidence(
            battery_steps,
            disconnect,
            False,
            f"battery steps {list(battery_steps)}; disconnect at step {disconnect}, "
            f"after first battery step {first}",
        )
    return BatteryOrderEvidence(
        battery_steps,
        disconnect,
        True,
        f"battery steps {list(battery_steps)}; disconnect at step {disconnect}",
    )


def _warnings(steps: Sequence[Step]) -> tuple[HazardWarning, ...]:
    by_tag: dict[str, list[int]] = {}
    for step in steps:
        for tag in _step_tags(step):
            if tag in _WARNING_TAGS:
                by_tag.setdefault(tag, []).append(step.number)
    merged: dict[str, list[int]] = {}
    for tag in _WARNING_TAGS:  # fixed template order, deterministic
        if tag in _ADHESIVE_CLASS:
            if tag in by_tag:
                merged.setdefault("adhesive/heat", []).extend(by_tag[tag])
        elif tag in by_tag:
            merged.setdefault(tag, []).extend(by_tag[tag])
    unknown_nums = sorted(
        {s.number for s in steps if _unrecognized_tags(_step_tags(s))}
    )
    if unknown_nums:
        merged[UNKNOWN_HAZARD_TAG] = unknown_nums
    return tuple(
        HazardWarning(
            tag,
            UNKNOWN_HAZARD_TEXT if tag == UNKNOWN_HAZARD_TAG else HAZARD_TEMPLATES[tag],
            tuple(sorted(set(nums))),
        )
        for tag, nums in merged.items()
    )


def step_hazard_blocks(step: Step) -> tuple[tuple[str, str], ...]:
    """Canonical ``(tag, text)`` warnings for one step, in step tag order.

    The ``_warnings`` taxonomy over a single step: adhesive/heat share one
    template, anything outside the taxonomy (but non-blank) gets the fixed
    unknown-hazard text. Never silenced, never paraphrased. Card renderers
    take this as their hazard-text lookup so the fixed text lives in
    exactly one place.
    """
    blocks: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tag in step.hazards:
        normalized = normalize_tag(tag)
        if not normalized:
            continue
        if normalized in _ADHESIVE_CLASS:
            key = "adhesive/heat"
        elif normalized in HAZARD_TEMPLATES:
            key = normalized
        else:
            key = UNKNOWN_HAZARD_TAG
        if key in seen:
            continue
        seen.add(key)
        if key == UNKNOWN_HAZARD_TAG:
            blocks.append((key, UNKNOWN_HAZARD_TEXT))
        else:
            blocks.append((key, HAZARD_TEMPLATES[key]))
    return tuple(blocks)


def refusal_copy(kind: str) -> str | None:
    """The plain-words refusal shown for a refusal kind, or None for other kinds."""
    if kind == "scope":
        return f"{_SCOPE_REFUSAL_TEXT} {CTA_SAMPLE} instead."
    if kind == "not_a_teardown":
        return f"{_NOT_A_TEARDOWN_TEXT} {CTA_SAMPLE} instead."
    if kind == "duration":
        return f"{_DURATION_REFUSAL_TEXT} {CTA_SAMPLE} instead."
    if kind == "budget":
        return f"{_BUDGET_REFUSAL_TEXT} {CTA_SAMPLE} instead."
    failures = {
        "deadline": "This clip took longer than the time limit for one guide. Try a shorter clip.",
        "part_check": "We could not finish checking the parts in this clip. Try uploading it again.",
        "extraction": "We could not identify reliable repair steps in this clip. Try another clip.",
        "extract_service_failure": "The AI service did not answer for part of your video. Nothing was wrong with your clip. Please try again.",
        "upload_failed": "The clip could not be sent for processing. Try uploading it again.",
        "upload_expired": "The sent video expired before processing started. Please upload it again.",
        "queue_unavailable": "The job could not start. Please try uploading your clip again.",
        "interrupted": "Sending stopped when the service restarted. Please upload your clip again.",
        "processing": "We could not process the video frames. Try another clip.",
        "frames": "We could not select pictures for this guide. Try another clip.",
        "readback": "We could not finish checking the guide labels. Try uploading it again.",
        "finalize": "We could not save the completed guide. Try uploading the clip again.",
        "settlement": "This guide stopped, and its final accounting is still settling.",
        "cancelled": "This upload was cancelled and its guide was stopped.",
        "service_unavailable": "The guide service was unavailable before processing started. Try again later.",
        "failure": "This guide could not be built because processing stopped. Try uploading the clip again.",
    }
    if kind in failures:
        return failures[kind]
    return None


def assess_hazards(steps: Sequence[Step], device_guess: str) -> HazardAssessment:
    """Assess one guide: refusal first, then warnings, order, and DRAFT state.

    A scope hit wins over a non-teardown: a swollen cell in an unrecognized
    device is still a scope refusal. Device scope is a class-word match after
    that (app/device_scope.toml): a blank, unknown, or ambiguous guess is not
    a teardown, and a guess that is not unambiguously a supported phone or
    laptop is a scope refusal.
    Refused guides authorize zero SAM calls; anything else
    authorizes up to the frozen per-job ceiling.
    """
    steps = tuple(steps)
    scope = _scope_hit(device_guess, steps)
    guess = _normalise_spaces(device_guess)
    unknown_guess = not steps or not guess or bool(_AMBIGUOUS_GUESS.search(guess))
    if scope:
        refusal_kind: Literal["none", "not_a_teardown", "scope"] = "scope"
        refusal_text: str | None = f"{_SCOPE_REFUSAL_TEXT} {CTA_SAMPLE} instead."
    elif unknown_guess:
        refusal_kind = "not_a_teardown"
        refusal_text = f"{_NOT_A_TEARDOWN_TEXT} {CTA_SAMPLE} instead."
    elif not _names_one_supported_device(guess):
        refusal_kind = "scope"
        refusal_text = f"{_SCOPE_REFUSAL_TEXT} {CTA_SAMPLE} instead."
    else:
        refusal_kind = "none"
        refusal_text = None
    refused = refusal_kind != "none"
    order = _battery_order(steps)
    draft_reasons = () if refused or order.battery_first else (DRAFT_ORDER_REASON,)
    return HazardAssessment(
        refused=refused,
        refusal_kind=refusal_kind,
        refusal_text=refusal_text,
        sam_authorized=0 if refused else SAM_MAX_CALLS_PER_JOB,
        warnings=() if refused else _warnings(steps),
        draft=bool(draft_reasons),
        draft_reasons=draft_reasons,
        battery_order=order,
    )


def difficulty_for(steps: Sequence[Step], assessment: HazardAssessment) -> DifficultyResult:
    """Apply the shared ordered difficulty rule.

    Refusal first; battery/adhesive/display (or any unrecognized tag) →
    Hard; another hazard or 4+ steps → Medium; 1–3 safe steps → Easy. A
    hazard-free twelve-step guide is the upper edge of Medium, not a separate
    branch.
    """
    steps = tuple(steps)
    if assessment.refused:
        return DifficultyResult(True, None, CTA_SAMPLE)
    tags = frozenset(t for s in steps for t in _step_tags(s)) & frozenset(_WARNING_TAGS)
    unknown = any(_unrecognized_tags(_step_tags(s)) for s in steps)
    if tags & _HARD_TAGS or unknown:
        return DifficultyResult(False, "Hard", CTA_PROFESSIONAL)
    if tags or len(steps) >= 4 or not steps:
        # len >= 4 covers 4-12 and, defensively, anything longer; an empty
        # guide is never Easy (assess_hazards refuses it as not a teardown).
        return DifficultyResult(False, "Medium", CTA_HELP)
    return DifficultyResult(False, "Easy", CTA_TRY)


def stop_reasons(step: Step) -> tuple[str, ...]:
    """Every stop hazard on one step, in fixed order: battery, glued_screen, unknown_hazard.

    The one rule behind ``reader_stop`` and the live/export gates: glued means the
    display+adhesive tag pair across tags, or glued screen/back wording in the action.
    """
    tags = _step_tags(step)
    reasons: list[str] = []
    if BATTERY_TAG in tags:
        reasons.append("battery")
    raw_action = step.action
    action = raw_action.lower() if type(raw_action) is str else ""
    if ("display" in tags and "adhesive" in tags) or (
        _GLUE_WORD.search(action) and any(w in action for w in _SCREEN_WORDS)
    ):
        reasons.append("glued_screen")
    if _unrecognized_tags(tags):
        reasons.append("unknown_hazard")
    return tuple(reasons)


def reader_stop(step: Step) -> ReaderStop | None:
    """Return the hard-stop object for a battery, glued-screen, or
    unrecognized-hazard step, else None.

    Glued means the display+adhesive tag pair or glued screen/back wording in
    the action. Any other non-empty tag fires the generic unknown stop: never
    silence. Later steps stay hidden until acknowledgement (see
    ``hidden_step_numbers`` and ``deliverable_steps``).
    """
    reasons = stop_reasons(step)
    if not reasons:
        return None
    reason: Literal["battery", "glued_screen", "unknown_hazard"] = reasons[0]  # type: ignore[assignment]
    text = UNKNOWN_STOP_TEXT if reason == "unknown_hazard" else HARD_STOP_TEXT
    stop_id = f"step-{step.number}-{reason.replace('_', '-')}"
    return ReaderStop(
        step.number,
        stop_id,
        HARD_STOP_TITLE,
        text,
        CTA_PROFESSIONAL,
        reason,
    )


def hidden_step_numbers(
    step_numbers: Sequence[int],
    stops: Sequence[ReaderStop],
    acknowledged: frozenset[int] | set[int],
) -> frozenset[int]:
    """Step numbers hidden behind unacknowledged stops (pure render rule).

    Every step after an unacknowledged stop stays hidden; acknowledging an
    earlier stop never reveals steps past a later unacknowledged one. Server
    and offline renderers both apply this pure rule before writing bytes.
    """
    acked = frozenset(acknowledged)
    hidden: set[int] = set()
    for stop in stops:
        if stop.step_number in acked:
            continue
        hidden.update(n for n in step_numbers if n > stop.step_number)
    return frozenset(hidden)


def deliverable_steps(
    steps: Sequence[Step],
    stops: Sequence[ReaderStop],
    acknowledged: frozenset[int] | set[int],
) -> tuple[Step, ...]:
    """Steps the server may deliver: everything not hidden behind a stop.

    The server renders ONLY these, so later-step bytes never leave the
    server until the stop is acknowledged (POST, then re-render with the
    stop in `acknowledged`). Browser code never expands this result.
    """
    numbers = tuple(s.number for s in steps)
    hidden = hidden_step_numbers(numbers, tuple(stops), acknowledged)
    return tuple(s for s in steps if s.number not in hidden)


def _one_form_value(form: dict[str, list[str]], name: str) -> str:
    values = form.get(name, [])
    if len(values) != 1 or not _FORM_IDENTIFIER.fullmatch(values[0]):
        raise HTTPException(status_code=400, detail=f"invalid {name}")
    return values[0]


def hard_stop_acknowledgement_router(
    acknowledgements: HardStopAcknowledgements,
    *,
    reader_cookie: str = HARD_STOP_READER_COOKIE,
) -> APIRouter:
    """Build the real POST handler around injected server-owned state."""
    router = APIRouter()

    @router.post(HARD_STOP_ACK_PATH)
    async def acknowledge_hard_stop(request: Request) -> RedirectResponse:
        reader_id = request.cookies.get(reader_cookie, "")
        if not reader_id:
            raise HTTPException(status_code=403, detail="reader session required")
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type.lower() != "application/x-www-form-urlencoded":
            raise HTTPException(status_code=415, detail="form encoding required")
        body = bytearray()
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > _HARD_STOP_FORM_MAX_BYTES:
                raise HTTPException(status_code=413, detail="form too large")
            body.extend(chunk)
        try:
            form = parse_qs(
                body.decode("utf-8"),
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=3,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="invalid form") from exc
        if set(form) != {"guide_id", "guide_revision", "stop_id"}:
            raise HTTPException(status_code=400, detail="invalid form fields")
        guide_id = _one_form_value(form, "guide_id")
        guide_revision = _one_form_value(form, "guide_revision")
        stop_id = _one_form_value(form, "stop_id")
        return_path = acknowledgements.acknowledge(
            reader_id,
            guide_id,
            guide_revision,
            stop_id,
        )
        if return_path is None:
            raise HTTPException(status_code=409, detail="stale or unknown hard stop")
        return RedirectResponse(return_path, status_code=303)

    return router
