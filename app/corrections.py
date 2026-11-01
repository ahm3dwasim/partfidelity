"""The one correction control: rename or delete a part label before sharing.

Nothing else about a guide is editable (timestamps, re-pick and the rest
are cut). A correction is an ordered log of operations stored next to the
guide as a sidecar file; ``apply_corrections`` replays it over the stored
guide, so ``guide.json`` is never rewritten and every view (guide page,
image zip, follower export) sees the same corrected guide. Indices in an
operation are positions in the guide *as corrected so far*, which is what
the page shows when the owner clicks.

A rename changes the phrase everywhere it is carried (the step part, the
slot that claims it, the detection and its provisional render) and keeps
the pixels untouched; a delete removes the part and its label and shifts
later parts of that step down by one.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.contracts import Guide, LabelDecision, LabelSlot
from app.settings import DEFAULT_CORRECTION_PHRASE_MAX_CHARS

__all__ = (
    "CorrectionError",
    "CorrectionStore",
    "apply_corrections",
    "clean_phrase",
    "confirmations",
    "generations",
    "validated_op",
)

RENAME = "rename"
DELETE = "delete"
#: The person's own "I checked this label" (never an independent check) and its undo.
#: Neither changes the guide; ``confirmations`` replays them. ``label`` is the label
#: image's file name, which survives later renames, deletes and reordering; ``gen`` is
#: that label's generation when it was confirmed (see ``generations``).
CONFIRM = "confirm"
UNCONFIRM = "unconfirm"
CONFIRM_BY = "user"
_KINDS = (RENAME, DELETE, CONFIRM, UNCONFIRM)
_KEYS = {
    RENAME: {"op", "step", "part", "phrase"},
    DELETE: {"op", "step", "part"},
    CONFIRM: {"op", "step", "part", "phrase", "label", "gen", "at", "by"},
    UNCONFIRM: {"op", "step", "part", "phrase", "label", "gen"},
}


class CorrectionError(Exception):
    """A refused correction; ``code`` names the page's failure message."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def clean_phrase(raw: object, max_chars: int) -> str:
    """Normalise a typed phrase, or raise ``CorrectionError`` (blank/long/bad)."""
    if not isinstance(raw, str):
        raise CorrectionError("bad")
    text = " ".join(raw.split())
    if not text:
        raise CorrectionError("blank")
    if len(text) > max_chars:
        raise CorrectionError("long")
    if any(not ch.isprintable() for ch in text):
        raise CorrectionError("bad")
    return text


def _rename(guide: Guide, si: int, pi: int, phrase: str) -> Guide:
    old = guide.steps[si].parts[pi]
    new = replace(old, phrase=phrase)
    steps = list(guide.steps)
    parts = list(steps[si].parts)
    parts[pi] = new
    steps[si] = replace(steps[si], parts=tuple(parts))
    labels: list[LabelDecision] = []
    for decision in guide.labels:
        slot = decision.slot
        if (slot.step_index, slot.part_index) == (si, pi) and slot.part == old:
            detection = decision.detection
            provisional = decision.provisional
            labels.append(
                replace(
                    decision,
                    slot=replace(slot, part=new),
                    detection=(
                        replace(detection, phrase=phrase)
                        if detection is not None
                        else None
                    ),
                    provisional=(
                        replace(
                            provisional,
                            detection=replace(provisional.detection, phrase=phrase),
                        )
                        if provisional is not None
                        else None
                    ),
                )
            )
        else:
            labels.append(decision)
    return replace(guide, steps=tuple(steps), labels=tuple(labels))


def _delete(guide: Guide, si: int, pi: int) -> Guide:
    steps = list(guide.steps)
    parts = list(steps[si].parts)
    del parts[pi]
    steps[si] = replace(steps[si], parts=tuple(parts))
    labels: list[LabelDecision] = []
    for decision in guide.labels:
        slot = decision.slot
        if slot.step_index != si:
            labels.append(decision)
        elif slot.part_index == pi:
            continue
        elif slot.part_index > pi:
            labels.append(
                replace(
                    decision,
                    slot=LabelSlot(si, slot.part_index - 1, slot.part),
                )
            )
        else:
            labels.append(decision)
    return replace(guide, steps=tuple(steps), labels=tuple(labels))


def validated_op(
    guide: Guide, op: object, *, max_chars: int, gens: Mapping[str, int] | None = None
) -> dict:
    """Return a clean operation for ``guide`` as it stands, or raise.

    ``expect`` is the phrase the owner saw: if the part changed under them
    (another tab, a stale page) the operation is refused (``gone``) rather
    than hitting a different part. A confirm or undo also names the label
    generation the page showed; any other generation is ``stale``.
    """
    if not isinstance(op, Mapping):
        raise CorrectionError("bad")
    kind, si, pi = op.get("op"), op.get("step"), op.get("part")
    if kind not in _KINDS or type(si) is not int or type(pi) is not int:
        raise CorrectionError("bad")
    if not (0 <= si < len(guide.steps) and 0 <= pi < len(guide.steps[si].parts)):
        raise CorrectionError("gone")
    if op.get("expect") != guide.steps[si].parts[pi].phrase:
        raise CorrectionError("gone")
    clean: dict = {"op": kind, "step": si, "part": pi}
    if kind == RENAME:
        clean["phrase"] = clean_phrase(op.get("phrase"), max_chars)
    elif kind in (CONFIRM, UNCONFIRM):
        clean["phrase"] = guide.steps[si].parts[pi].phrase
        clean["label"] = _label_name(guide, si, pi)
        clean["gen"] = (gens or {}).get(clean["label"], 0)
        if op.get("gen") != clean["gen"]:
            raise CorrectionError("stale")
        if kind == CONFIRM:
            clean["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            clean["by"] = CONFIRM_BY
    return clean


def _label_name(guide: Guide, si: int, pi: int) -> str:
    """File name of the label image of one part; a part with none cannot be confirmed."""
    for decision in guide.labels:
        slot = decision.slot
        if (slot.step_index, slot.part_index) == (si, pi) and decision.provisional is not None:
            return decision.provisional.rendered_image_path.name
    raise CorrectionError("bad")


def confirmations(ops: Sequence[Mapping]) -> dict[str, tuple[int, str]]:
    """{label image name: (generation, when)} for labels the person confirmed and not undone.

    A confirmation counts only while its generation is the label's current one.
    """
    held: dict[str, tuple[int, str]] = {}
    for op in ops:
        if op.get("op") == CONFIRM:
            held[op["label"]] = (op["gen"], op["at"])
        elif op.get("op") == UNCONFIRM:
            held.pop(op["label"], None)
    return held


def generations(
    guide: Guide, ops: Sequence[Mapping], *, max_chars: int = DEFAULT_CORRECTION_PHRASE_MAX_CHARS
) -> dict[str, int]:
    """{label image name: generation} for every label of the corrected guide.

    A label's generation is how many renames it has had in the log. The log
    is only ever appended under the correction lock and written atomically,
    so this integer is durable and grows by exactly one per rename; second-check
    receipts and confirmations carry it, and only the current one counts.
    ``guide`` is the stored guide before any correction.
    """
    gens = {name: 0 for name in _label_names(guide)}
    for op in ops:
        if op.get("op") == RENAME:
            try:
                name = _label_name(guide, op["step"], op["part"])
            except CorrectionError:
                name = None
            if name is not None:
                gens[name] = gens.get(name, 0) + 1
        guide = apply_corrections(guide, [op], max_chars=max_chars)
    live = _label_names(guide)
    return {name: gen for name, gen in gens.items() if name in live}


def _label_names(guide: Guide) -> set[str]:
    return {d.provisional.rendered_image_path.name for d in guide.labels if d.provisional is not None}


def apply_corrections(
    guide: Guide, ops: Sequence[Mapping], *, max_chars: int = DEFAULT_CORRECTION_PHRASE_MAX_CHARS
) -> Guide:
    """Replay every operation in order; refuse an invalid log."""
    for op in ops:
        if not isinstance(op, Mapping):
            raise CorrectionError("failed")
        kind, si, pi = op.get("op"), op.get("step"), op.get("part")
        if (kind not in _KINDS or type(si) is not int or type(pi) is not int
                or not (0 <= si < len(guide.steps) and 0 <= pi < len(guide.steps[si].parts))
                or set(op) != _KEYS[kind]):
            raise CorrectionError("failed")
        if kind == RENAME:
            phrase = op["phrase"]
            try:
                if clean_phrase(phrase, max_chars) != phrase:
                    raise CorrectionError("failed")
            except CorrectionError as error:
                raise CorrectionError("failed") from error
            guide = _rename(guide, si, pi, phrase)
        elif kind in (CONFIRM, UNCONFIRM):
            continue
        else:
            guide = _delete(guide, si, pi)
    return guide


class CorrectionStore:
    """File-backed operation log, one sidecar per job directory.

    ``shared_job`` names the public sample job: its guide is one copy every
    visitor sees, so it is never editable.
    """

    def __init__(
        self,
        job_dir_for,
        filename: str,
        *,
        shared_job=lambda: None,
        max_ops: int,
        max_chars: int = DEFAULT_CORRECTION_PHRASE_MAX_CHARS,
        guide_filename: str | None = None,
    ) -> None:
        self._job_dir_for = job_dir_for
        self._filename = filename
        self._shared_job = shared_job
        self.max_ops = max_ops
        self.max_chars = max_chars
        self.guide_filename = guide_filename

    def _path(self, job_id: str) -> Path:
        return Path(self._job_dir_for(job_id)) / self._filename

    def editable(self, job_id: str) -> bool:
        return self._shared_job() != job_id

    @contextlib.contextmanager
    def locked(self, job_id: str) -> Iterator[None]:
        """Serialise one job's read-validate-write, so two edits never clobber."""
        try:
            handle = open(self._path(job_id).with_suffix(".lock"), "a+b")
        except OSError as error:
            raise CorrectionError("failed") from error
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            handle.close()  # closing drops the flock

    def _load(self, job_id: str) -> dict:
        """The whole sidecar: ``ops`` (the edit log) and ``spend`` (second-check budget)."""
        try:
            value = json.loads(self._path(job_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            raise CorrectionError("failed") from error
        if not isinstance(value, dict):
            raise CorrectionError("failed")
        return value

    def read(self, job_id: str) -> list[dict]:
        value = self._load(job_id)
        if not value:
            return []
        ops = value.get("ops")
        if not isinstance(ops, list) or any(
            not isinstance(op, dict)
            or op.get("op") not in _KINDS
            or set(op) != _KEYS[op["op"]]
            or type(op.get("step")) is not int
            or type(op.get("part")) is not int
            or op["step"] < 0 or op["part"] < 0
            or ("phrase" in op and (
                not isinstance(op["phrase"], str)
                or self._invalid_phrase(op["phrase"])
            ))
            or ("label" in op and not isinstance(op["label"], str))
            or ("gen" in op and (type(op["gen"]) is not int or op["gen"] < 0))
            or (op["op"] == CONFIRM and (not isinstance(op["at"], str) or op["by"] != CONFIRM_BY))
            for op in ops
        ):
            raise CorrectionError("failed")
        return ops

    def _invalid_phrase(self, phrase: str) -> bool:
        try:
            return clean_phrase(phrase, self.max_chars) != phrase
        except CorrectionError:
            return True

    def write(self, job_id: str, ops: Sequence[Mapping]) -> None:
        """Atomically replace the log, keeping the spend; any OS failure surfaces as ``failed``.
        Call it under ``locked``."""
        value = self._load(job_id)
        self._dump(job_id, {**value, "ops": list(ops)})

    def spend(self, job_id: str) -> tuple[int, Decimal, bool]:
        """(calls, cost USD, breached) of this job's second check, across every batch."""
        raw = self._load(job_id).get("spend")
        if raw is None:
            return 0, Decimal("0"), False
        try:
            calls, cost, breached = raw["calls"], Decimal(raw["cost"]), raw["breached"]
        except (TypeError, KeyError, InvalidOperation) as error:
            raise CorrectionError("failed") from error
        if type(calls) is not int or calls < 0 or not cost.is_finite() or type(breached) is not bool:
            raise CorrectionError("failed")
        return calls, cost, breached

    def take_spend(self, job_id: str, reserve: Decimal, max_calls: int, max_cost: Decimal) -> bool:
        """Reserve one call's worst case under the job caps; False (nothing sent) when it does not fit
        or the budget cannot be read or saved."""
        try:
            with self.locked(job_id):
                calls, cost, breached = self.spend(job_id)
                if breached or calls >= max_calls or cost + reserve > max_cost:
                    return False
                self._save_spend(job_id, calls + 1, cost + reserve, False)
                return True
        except CorrectionError:
            return False

    def settle_spend(self, job_id: str, delta: Decimal, breach: bool) -> None:
        """Swap a reservation for the measured cost; a breach stops every later call of the job."""
        with self.locked(job_id):
            calls, cost, breached = self.spend(job_id)
            self._save_spend(job_id, calls, cost + delta, breached or breach)

    def _save_spend(self, job_id: str, calls: int, cost: Decimal, breached: bool) -> None:
        value = self._load(job_id)
        self._dump(job_id, {"ops": value.get("ops", []), **value,
                            "spend": {"calls": calls, "cost": str(cost), "breached": breached}})

    def _dump(self, job_id: str, value: Mapping) -> None:
        path = self._path(job_id)
        if self.guide_filename is not None and not (path.parent / self.guide_filename).is_file():
            raise CorrectionError("gone")
        try:
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".corr-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(dict(value), handle, sort_keys=True)
                Path(tmp).replace(path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as error:
            raise CorrectionError("failed") from error
