"""Per-step source notes shipped with a guide bundle.

A note says what the footage does not show for one step, so the card and the
stage never let a still frame imply an action was filmed. The file is strict:
a bad shape raises, never silently drops a note.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepNote:
    label: str
    text: str
    stage: str  # one short line for under the stage


def load_step_notes(path: Path, *, required: set[int] | None = None) -> dict[int, StepNote]:
    """Read ``{"<step number>": {"label": ..., "text": ...}}``; a missing file means no notes."""
    if not path.is_file() and required is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: expected an object keyed by step number")
    notes: dict[int, StepNote] = {}
    for key, value in raw.items():
        if not key.isdigit() or not isinstance(value, dict) or set(value) != {"label", "text", "stage"}:
            raise ValueError(f"{path.name}: bad entry {key!r}; need a step number with label, text and stage")
        fields = [value["label"], value["text"], value["stage"]]
        if not all(isinstance(f, str) and f.strip() for f in fields):
            raise ValueError(f"{path.name}: step {key} needs non-empty label, text and stage")
        notes[int(key)] = StepNote(*(f.strip() for f in fields))
    if required is not None and not required.issubset(notes):
        raise ValueError(f"{path.name}: missing required steps {sorted(required - notes.keys())}")
    return notes
