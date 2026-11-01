"""End-to-end job pipeline: admission to rendered guide.

Runs one job through the existing lane modules in order — admission and
budget hold, preview and Meta file, Spark step extraction, frame scoring,
SAM checks for fan and battery only, provisional renders, read-backs,
hazards, evidence, guide assembly, and the deletion schedule. Every paid
call is authorized through the job's own ledger context by the module
that makes it (reserve before invoke, settle after); this module never
re-implements a stage.

REPLAY mode injects recorded provider responses (see
:class:`ReplaySparkClient`, :class:`ReplaySamClient`,
:class:`ReplayMetaFilesClient`) through the same code path as live
clients, so the sample clip produces a full guide with zero network
calls. Replay is enforced, not a label: with ``replay=True`` the
orchestrator selects the replay adapters itself (or validates injected
ones) and rejects any live client; live mode requires an injected
client set and rejects any replay client. Never mixed per run.

Every paid stage persists its receipts through ``evidence.record_call``.
Replay mode records through this seam (recording client wrappers for
Meta upload and Spark extraction, the SAM call sink, and the
read-back hooks), translated into the evidence module's vocabulary here; live mode
leaves those wrapper sinks unwired and each live client persists its
own single receipt per call instead, so a call is never receipted
twice. A job that reaches ``done`` always has its full receipt rows,
because a receipt that cannot persist fails the job instead of
vanishing.

Fail-closed: an admission or budget refusal stops the job before any
provider call; any provider error finishes the job failed (settling its
holds) and no partial guide is stored or served. The guide file is
stored before the success transition commits (the ``done`` finish is
deferred until the bytes are durable); a terminal transition lost to
ledger contention retries bounded by config, then records a recoverable
pending-settlement marker and reports ``pending_settlement`` — never
``failed`` with reservations still held.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from app import db
from app import guide_clip, persistence
from app.providers import open_readback
from app import sam as sam_module
from app.admission import admit_job, job_dir
from app.boundaries import (
    drop_edge_outro_phases,
    merge_same_action_phases,
    refine_consensus_bundle,
)
from app.chunks import (
    CHUNK_ARTIFACT_KIND,
    DEFAULT_SPAN_MODE,
    ChunkCutError,
    ChunkWindow,
    ContiguityError,
    StitchedSteps,
    chunk_filename,
    cut_chunk,
    derive_contiguous_ends,
    plan_chunks,
    probe_cut,
    real_cut_window,
    require_contiguous_cover,
    stitch_bundles,
)
from app.contracts import (
    JSONValue,
    JOB_DEADLINE_S,
    META_UPLOAD_MAX_HOURS,
    RECEIPT_MODEL_META_FILES,
    RECEIPT_STAGE_EXTRACT,
    RECEIPT_STAGE_READBACK,
    RECEIPT_STAGE_SEGMENT,
    RECEIPT_STAGE_UPLOAD,
    SPARK_RUNS_PER_GUIDE,
    ArtifactId,
    CallRecord,
    Detection,
    ExcludedSpan,
    Guide,
    GuideHazardAssessment,
    JobContext,
    JobId,
    LabelDecision,
    LabelSlot,
    LifecycleRepository,
    MetaFileId,
    MetaFilesClient,
    MetaFilesError,
    MetaUpload,
    PartRequest,
    ProvisionalRender,
    SamClient,
    SparkClient,
    Step,
    StepBundle,
    UploadStore,
)
from app.clock import ClockBurnError, burn_clock_in_place
from app.deletion import cleanup_failed_guide, request_meta_deletion
from app.evidence import record_call, record_event
from app.frame_scoring import score_frames
from app.guides import assemble_guide
from app.hazards import assess_hazards, difficulty_for, refusal_copy
from app.ledger import (
    active_job_publication,
    BudgetExhausted,
    ContentionError,
    JobAlreadyFinished,
    LedgerError,
    extend_job_deadline,
    job_reservation_spend_usd,
)
from app.parts import is_checked_part
from app.providers import (
    select_providers,
    spark_window_cap_usd,
    spark_window_estimate_usd,
    whole_video_fits,
)
from app.readback import READBACK_EVENT, ReadbackResult, run_readbacks
from app.render import render_label
from app.sam import (
    SAM_ENDPOINT,
    SamCallLimitExceeded,
    SamDeadlineExceeded,
    SamMask,
    SamStillResult,
    choose_detection,
    install_call_sink,
    segment,
)
from app.scene_cuts import SceneCutError, detect_scene_cuts, write_window_cuts
from app.settings import (
    DEFAULT_CHUNK_BUNDLE_TTL_S,
    DEFAULT_CHUNK_CUT_TIMEOUT_S,
    DEFAULT_CLOCK_BURN_IN,
    DEFAULT_CONTIGUITY_TOLERANCE_S,
    DEFAULT_CHUNK_MAX_ATTEMPTS,
    DEFAULT_CHUNK_OVERLAP_S,
    DEFAULT_CHUNK_RETRY_BUDGET_USD,
    DEFAULT_CHUNK_S,
    DEFAULT_CORRECTIONS_FILENAME,
    DEFAULT_DEADLINE_EXTENSION_MAX_S,
    DEFAULT_GUIDE_CLIP_FILENAME,
    DEFAULT_GUIDE_FILENAME,
    DEFAULT_EXCLUDE_OUTRO_PHASES,
    DEFAULT_JOB_COST_CAP_USD,
    DEFAULT_LABEL_IMAGE_PATTERN,
    DEFAULT_MAX_VIDEO_S,
    DEFAULT_MERGE_SAME_ACTION,
    DEFAULT_MIN_PHASE_LENGTH_S,
    DEFAULT_PENDING_SETTLEMENT_FILENAME,
    DEFAULT_SCENE_CUTS_IN_PROMPT,
    DEFAULT_SCENE_CUT_THRESHOLD,
    DEFAULT_SETTLE_MAX_ATTEMPTS,
    DEFAULT_SNAP_WINDOW_S,
    DEFAULT_SPARK_INPUT_USD_PER_TOKEN,
    DEFAULT_SPARK_INPUT_TOKENS_PER_VIDEO_S,
    DEFAULT_SPARK_OUTPUT_USD_PER_TOKEN,
    DEFAULT_SPARK_RESERVATION_USD,
    DEFAULT_SPARK_TIMEOUT_S,
    DEFAULT_WHOLE_VIDEO_MAX_S,
    DEFAULT_STEP_ALIGNMENT_GENERIC_TOKENS,
    DEFAULT_STEP_ALIGNMENT_NEGATION_WORDS,
    DEFAULT_STEP_ALIGNMENT_PART_SYNONYMS,
    DEFAULT_STEP_ALIGNMENT_QUALIFIERS,
    DEFAULT_STEP_ALIGNMENT_TRANSITION_VERBS,
    DEFAULT_STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION,
    DEFAULT_STEP_ALIGNMENT_VERB_SYNONYMS,
    Settings,
)
from app.step_alignment import DEFAULT_ALIGNMENT_MODE
from app.spark import (
    DEFAULT_PHASE_KIND,
    PHASE_KINDS,
    SPARK_MODEL_ID,
    ChunkExtent,
    NotATeardown,
    SparkRunResult,
    SparkClientError,
    SparkSchemaError,
    SparkServiceError,
    SparkStageConfig,
    StepsDisagreement,
    chunk_bundle_digest,
    extract_steps_cached,
    load_chunk_bundle,
    load_chunk_cut_window,
    save_chunk_bundle,
    spark_config_of,
    validate_spark_payload,
    _failed_actual,
    _require_valid_run_result,
)
from app.uploads import (
    META_UPLOAD_REQUEST_HOURS,
    OverLimitError,
    confirm_and_send,
    hash_file,
    over_limit_message,
    prepare_preview,
)

logger = logging.getLogger("partfidelity.pipeline")

PipelineState = Literal[
    "done", "failed", "refused", "pending_settlement", "contention",
]
# ``contention`` is non-terminal: another worker holds a chunk claim, so
# this run paid nothing, settled nothing, and left the shared parent
# active for the holder to finish; the caller retries later.

GUIDE_FILENAME = DEFAULT_GUIDE_FILENAME

# Extraction failed for money reasons (up-front chunk budget, a reserve
# refusal, or a settle refusal): the message names the budget limit so it
# never reads as a model failure.
STEP_EXTRACTION_OVER_BUDGET = "step extraction over the budget limit"

# Receipt shapes this seam translates lane records into (the evidence vocabulary).
# Stages and models come from the contracts-owned receipt vocabulary, so
# the seam and the evidence gate cannot drift apart; endpoints each have
# this file as their single home.
SPARK_EXTRACT_ENDPOINT = "/runs"
SAM_SEGMENT_STAGE = RECEIPT_STAGE_SEGMENT
SAM_SEGMENT_ENDPOINT = f"/{SAM_ENDPOINT}"
READBACK_RECEIPT_ENDPOINT = "/read_back"
READBACK_COUNTED_EVENT = "readback.counted"
META_UPLOAD_STAGE = RECEIPT_STAGE_UPLOAD
META_UPLOAD_MODEL = RECEIPT_MODEL_META_FILES
META_UPLOAD_ENDPOINT = "/files"


@dataclass(frozen=True)
class PipelineClients:
    """Injected provider adapters. Replay or live, never mixed per run."""

    spark: SparkClient
    sam: SamClient
    meta: MetaFilesClient


@dataclass(frozen=True)
class PipelineResult:
    """Plain terminal state of one pipeline run."""

    state: PipelineState
    job_id: JobId | None
    guide: Guide | None
    error: str | None
    replay: bool


@dataclass(frozen=True)
class DeletionSchedule:
    """Rows the hourly sweeper will collect for one finished job (read-only)."""

    job_id: JobId
    artifact_ids: tuple[str, ...]
    meta_file_ids: tuple[str, ...]


class ReplayExhausted(RuntimeError):
    """A replay client ran out of recorded responses (test fixture bug)."""


class ReplaySparkClient:
    """Recorded Spark responses: scripted step runs plus read-back answers.

    ``runs`` replays in order, one per extraction call; ``readbacks`` maps
    a part phrase to its recorded answer. Pure data: zero network calls.
    """

    def __init__(
        self,
        runs: list[SparkRunResult],
        *,
        readbacks: dict[str, ReadbackResult] | None = None,
    ) -> None:
        self._runs = list(runs)
        self._readbacks = dict(readbacks or {})
        self.run_calls: list[str] = []
        self.readback_calls: list[tuple[str, str]] = []

    def run_steps(
        self,
        video_file_id: str,
        *,
        clip_duration_s: float | None = None,
        scene_cuts: tuple[float, ...] | None = None,
        clocked: bool = False,
    ) -> SparkRunResult:
        # Replay answers from recordings, never from a prompt.
        _ = (clip_duration_s, scene_cuts, clocked)
        self.run_calls.append(video_file_id)
        if not self._runs:
            raise ReplayExhausted("replay has no more Spark step runs")
        return self._runs.pop(0)

    def read_back(
        self, image_path: Path, phrase: str, position_hint: str
    ) -> ReadbackResult:
        _ = (image_path, position_hint)
        self.readback_calls.append((phrase, position_hint))
        try:
            return self._readbacks[phrase]
        except KeyError:
            raise ReplayExhausted(
                f"replay has no read-back answer for {phrase!r}"
            ) from None


class ReplaySamClient:
    """Recorded SAM responses: scripted boxes and confidences per part.

    ``cases`` maps a part phrase to one ``(box, confidences)`` recording;
    the mask PNG is generated at the calling frame's own size with the box
    filled, because the renderer requires mask dims to equal frame dims.
    Successive calls for one phrase consume successive confidences, so a
    two-frame part replays a decisive winner and runner-up. Pure data:
    zero network calls.
    """

    def __init__(
        self, cases: dict[str, tuple[tuple[int, int, int, int], list[float]]]
    ) -> None:
        self._cases = {
            phrase: (box, list(scores)) for phrase, (box, scores) in cases.items()
        }
        self.calls: list[tuple[str, str]] = []

    def segment_still(
        self, image_path: Path, phrase: str, *, include_confidence: bool = True
    ) -> SamStillResult:
        _ = include_confidence
        self.calls.append((str(image_path), phrase))
        try:
            box, scores = self._cases[phrase]
        except KeyError:
            raise ReplayExhausted(
                f"replay has no SAM recording for {phrase!r}"
            ) from None
        if not scores:
            raise ReplayExhausted(f"replay exhausted SAM scores for {phrase!r}")
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ReplayExhausted(f"replay cannot read frame {image_path}")
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = (max(0, min(v, limit)) for v, limit in zip(box, (width, height) * 2))
        mask = np.zeros((height, width), np.uint8)
        mask[y0:y1, x0:x1] = 255
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise ReplayExhausted("replay cannot encode a SAM mask PNG")
        return SamStillResult(
            masks=(
                SamMask(
                    box_xyxy=(box[0], box[1], box[2], box[3]),
                    c=scores.pop(0),
                    mask_png=bytes(encoded),
                ),
            )
        )


class ReplayMetaFilesClient:
    """Recorded Meta files responses: one canned id per idempotency key.

    Expiry honors the requested hours within the frozen 24-hour cap, so
    the confirmation bounds always hold. Pure data: zero network calls.
    """

    def __init__(self) -> None:
        self.files: dict[str, MetaUpload] = {}
        self.by_key: dict[str, MetaUpload] = {}
        self.upload_calls: list[tuple[str, int, str]] = []
        self.delete_calls: list[str] = []

    def upload_file(
        self, path: Path, *, expires_after_hours: int, idempotency_key: str
    ) -> MetaUpload:
        self.upload_calls.append(
            (str(path), expires_after_hours, idempotency_key)
        )
        if idempotency_key in self.by_key:
            return self.by_key[idempotency_key]
        hours = max(1, min(int(expires_after_hours), META_UPLOAD_MAX_HOURS))
        found = MetaUpload(
            file_id=MetaFileId(f"replay-meta-{uuid.uuid4().hex}"),
            expires_at=db.utcnow() + timedelta(hours=hours),
        )
        self.files[str(found.file_id)] = found
        self.by_key[idempotency_key] = found
        return found

    def find_upload(self, idempotency_key: str) -> MetaUpload | None:
        return self.by_key.get(idempotency_key)

    def delete_file(self, file_id: MetaFileId) -> None:
        self.delete_calls.append(str(file_id))
        self.files.pop(str(file_id), None)

    def get_file(self, file_id: MetaFileId) -> MetaUpload | None:
        return self.files.get(str(file_id))


def _sample_step_payload(
    number: int, start: float, end: float, action: str, parts: list[tuple[str, str]],
    tools: tuple[str, ...] = (), hazards: tuple[str, ...] = (),
    kind: str = DEFAULT_PHASE_KIND,
) -> dict:
    return {
        "number": number,
        "start_s": start,
        "end_s": end,
        "action": action,
        "tools": list(tools),
        "parts": [
            {"phrase": phrase, "position_hint": hint} for phrase, hint in parts
        ],
        "hazards": list(hazards),
        "kind": kind,
    }


def default_replay_clients() -> PipelineClients:
    """Return the canonical recorded client set for replay mode.

    Three Spark runs (fan step plus battery step on a laptop), SAM boxes
    for fan and battery, and one battery read-back answer: the same
    fixture the sample route and replay tests run. Pure data, zero
    network calls.
    """
    windows = [(0.5, 2.5), (0.6, 2.6), (0.4, 2.4)]
    # Contiguous partitions: time consensus excludes a run whose
    # phases do not tile (edges {2.5, 2.6, 2.4} agree on 2.5).
    battery_windows = [(2.5, 5.5), (2.6, 5.6), (2.4, 5.4)]
    costs = ["0.08", "0.08", "0.07"]
    runs = []
    for index in range(3):
        start, end = windows[index]
        bstart, bend = battery_windows[index]
        runs.append(
            SparkRunResult(
                payload={
                    "device_guess": "laptop",
                    "confidence": 0.82,
                    "steps": [
                        _sample_step_payload(
                            1, start, end, "Remove the rear screws",
                            [("fan", "left side")], tools=("Phillips #0",),
                        ),
                        _sample_step_payload(
                            2, bstart, bend, "Disconnect the battery cable",
                            [("battery", "lower right")], hazards=("battery",),
                        ),
                    ],
                },
                cost_usd=Decimal(costs[index]),
                latency_ms=900,
            )
        )
    spark = ReplaySparkClient(
        runs,
        readbacks={
            "battery": ReadbackResult(
                answer="yes", cost_usd=Decimal("0.04"), latency_ms=120
            )
        },
    )
    sam = ReplaySamClient(
        {
            "fan": ((12, 30, 140, 150), [0.72, 0.60]),
            "battery": ((200, 90, 300, 210), [0.61, 0.55]),
        }
    )
    return PipelineClients(spark=spark, sam=sam, meta=ReplayMetaFilesClient())


def resolve_run_clients(
    clients: PipelineClients | None, *, replay: bool
) -> PipelineClients:
    """Enforce the replay/live client contract; raise on any violation.

    Replay mode selects the replay adapters itself (the canonical set
    when none are injected) and rejects any non-replay client; live mode
    requires an injected set from the provider factory and rejects any
    replay client. Never mixed per run. Raises before any side effect.
    """
    if replay:
        if clients is None:
            return default_replay_clients()
        for name, client, replay_type in (
            ("spark", clients.spark, ReplaySparkClient),
            ("sam", clients.sam, ReplaySamClient),
            ("meta", clients.meta, ReplayMetaFilesClient),
        ):
            if not isinstance(client, replay_type):
                raise ValueError(
                    f"replay mode rejects the injected {name} client:"
                    f" {type(client).__name__} is not a replay adapter"
                )
        return clients
    if clients is None:
        raise ValueError(
            "live mode requires an injected client set from the provider factory"
        )
    for name, client, replay_type in (
        ("spark", clients.spark, ReplaySparkClient),
        ("sam", clients.sam, ReplaySamClient),
        ("meta", clients.meta, ReplayMetaFilesClient),
    ):
        if isinstance(client, replay_type):
            raise ValueError(
                f"live mode rejects the injected {name} client:"
                " replay adapters never run outside replay mode"
            )
    return clients


def _require_live_binding(clients: PipelineClients, ctx: JobContext) -> None:
    """Refuse a live bundle bound to any job but this one.

    Live clients persist receipts through the context they were built
    with; running them under another job's context would receipt paid
    calls against the wrong job. Call before any reservation, request
    or receipt. Clients with no bound job (unbound test doubles) carry
    nothing to mismatch and are allowed; the Meta Files client is
    unbound by design and is not checked.
    """
    expected = str(ctx.job_id)
    for name in ("spark", "sam"):
        bound = getattr(getattr(clients, name), "bound_job_id", None)
        if bound is not None and str(bound) != expected:
            raise ValueError(
                f"live {name} client is bound to job {bound},"
                f" cannot run under job {expected}"
            )


def guide_hazard_assessment(steps: Sequence[Step], assessment) -> GuideHazardAssessment:
    """Realize the contracts-declared mapping from the hazard module's whole-guide verdict.

    The verdict line is the ordered difficulty CTA ("Try it" for Easy);
    draft reasons ride verbatim. A refused assessment fails closed on the
    contract default; the caller stops before assembly on refusal, so a
    refused guide is never built. Note: a clean guide carries zero draft
    reasons, which the guide module's assembly gate reads back as the DRAFT default —
    a cross-module gap the sample (battery DRAFT order) does not hit.
    """
    result = difficulty_for(tuple(steps), assessment)
    if result.refused or result.level is None:
        return GuideHazardAssessment()
    return GuideHazardAssessment(
        difficulty=result.level,
        verdict=result.cta,
        draft_reasons=tuple(assessment.draft_reasons),
    )


def _guide_filename(settings: Settings | None) -> str:
    return settings.guide_filename if settings is not None else DEFAULT_GUIDE_FILENAME


def _output_names(settings: Settings | None) -> dict[str, str]:
    """The guide and corrections names a new job row records."""
    return {
        "guide_filename": _guide_filename(settings),
        "corrections_filename": (
            settings.corrections_filename if settings is not None
            else DEFAULT_CORRECTIONS_FILENAME
        ),
    }


def _guide_clip_filename(settings: Settings | None) -> str:
    if settings is not None:
        return settings.guide_clip_filename
    return DEFAULT_GUIDE_CLIP_FILENAME


def _label_image_name(settings: Settings | None, *, step: int, part: int) -> str:
    if settings is not None:
        return settings.label_image_name(step=step, part=part)
    return DEFAULT_LABEL_IMAGE_PATTERN.format(step=step, part=part)


def _settle_max_attempts(settings: Settings | None) -> int:
    if settings is not None:
        return settings.settle_max_attempts
    return DEFAULT_SETTLE_MAX_ATTEMPTS


def _pending_filename(settings: Settings | None) -> str:
    if settings is not None:
        return settings.pending_settlement_filename
    return DEFAULT_PENDING_SETTLEMENT_FILENAME


def _max_video_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.max_video_s
    return DEFAULT_MAX_VIDEO_S


def _chunk_s(settings: Settings | None) -> float:
    return settings.chunk_s if settings is not None else DEFAULT_CHUNK_S


def _chunk_overlap_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.chunk_overlap_s
    return DEFAULT_CHUNK_OVERLAP_S


def _step_alignment_mode(settings: Settings | None) -> str:
    if settings is not None:
        return settings.providers.step_alignment_mode
    return DEFAULT_ALIGNMENT_MODE


def _step_alignment_generic_tokens(settings: Settings | None) -> frozenset[str]:
    if settings is not None:
        return settings.providers.step_alignment_generic_tokens
    return DEFAULT_STEP_ALIGNMENT_GENERIC_TOKENS


def _step_alignment_verb_synonyms(
    settings: Settings | None,
) -> tuple[frozenset[str], ...]:
    if settings is not None:
        return settings.providers.step_alignment_verb_synonyms
    return DEFAULT_STEP_ALIGNMENT_VERB_SYNONYMS


def _step_alignment_negation_words(settings: Settings | None) -> frozenset[str]:
    if settings is not None:
        return settings.providers.step_alignment_negation_words
    return DEFAULT_STEP_ALIGNMENT_NEGATION_WORDS


def _step_alignment_qualifiers(settings: Settings | None) -> frozenset[str]:
    if settings is not None:
        return settings.providers.step_alignment_qualifiers
    return DEFAULT_STEP_ALIGNMENT_QUALIFIERS


def _step_alignment_part_synonyms(
    settings: Settings | None,
) -> tuple[frozenset[str], ...]:
    if settings is not None:
        return settings.providers.step_alignment_part_synonyms
    return DEFAULT_STEP_ALIGNMENT_PART_SYNONYMS


def _step_alignment_transition_verbs(
    settings: Settings | None,
) -> frozenset[str]:
    if settings is not None:
        return settings.providers.step_alignment_transition_verbs
    return DEFAULT_STEP_ALIGNMENT_TRANSITION_VERBS


def _step_alignment_verb_less_overlap_fraction(
    settings: Settings | None,
) -> float:
    if settings is not None:
        return settings.providers.step_alignment_verb_less_overlap_fraction
    return DEFAULT_STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION


def _chunk_cut_timeout_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.chunk_cut_timeout_s
    return DEFAULT_CHUNK_CUT_TIMEOUT_S


def _chunk_deadline_extra_s(settings: Settings | None) -> float:
    """Extra deadline seconds covering one child's worst-case extraction.

    A child runs up to the configured Spark run count, each bounded by
    the configured Spark timeout; the envelope covers all of them back
    to back, so a worst-case child cannot expire mid-extraction.
    """
    if settings is not None:
        runs = settings.providers.spark_runs
        timeout_s = settings.providers.spark_timeout_s
    else:
        runs = SPARK_RUNS_PER_GUIDE
        timeout_s = DEFAULT_SPARK_TIMEOUT_S
    return float(runs) * float(timeout_s)


def _deadline_extension_max_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.deadline_extension_max_s
    return DEFAULT_DEADLINE_EXTENSION_MAX_S


def _job_cost_cap_usd(settings: Settings | None) -> Decimal:
    if settings is not None:
        return settings.job_cost_cap_usd
    return DEFAULT_JOB_COST_CAP_USD


def _chunk_max_attempts(settings: Settings | None) -> int:
    if settings is not None:
        return settings.chunk_max_attempts
    return DEFAULT_CHUNK_MAX_ATTEMPTS


def _chunk_retry_budget_usd(settings: Settings | None) -> Decimal:
    if settings is not None:
        return settings.chunk_retry_budget_usd
    return DEFAULT_CHUNK_RETRY_BUDGET_USD


def _spark_reservation_usd(settings: Settings | None) -> Decimal:
    if settings is not None:
        return settings.spark_reservation_usd
    return DEFAULT_SPARK_RESERVATION_USD


def _chunk_bundle_ttl_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.chunk_bundle_ttl_s
    return DEFAULT_CHUNK_BUNDLE_TTL_S


def _step_span_mode(settings: Settings | None) -> str:
    if settings is not None:
        return settings.step_span_mode
    return DEFAULT_SPAN_MODE


def _contiguity_tolerance_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.contiguity_tolerance_s
    return DEFAULT_CONTIGUITY_TOLERANCE_S


def _whole_video_max_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.whole_video_max_s
    return DEFAULT_WHOLE_VIDEO_MAX_S


def _spark_input_tokens_per_video_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.spark_input_tokens_per_video_s
    return DEFAULT_SPARK_INPUT_TOKENS_PER_VIDEO_S


def _clock_burn_in(settings: Settings | None) -> bool:
    if settings is not None:
        return settings.clock_burn_in
    return DEFAULT_CLOCK_BURN_IN


def _scene_cut_threshold(settings: Settings | None) -> float:
    if settings is not None:
        return settings.scene_cut_threshold
    return DEFAULT_SCENE_CUT_THRESHOLD


def _scene_cuts_in_prompt(settings: Settings | None) -> bool:
    if settings is not None:
        return settings.scene_cuts_in_prompt
    return DEFAULT_SCENE_CUTS_IN_PROMPT


def _snap_window_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.snap_window_s
    return DEFAULT_SNAP_WINDOW_S


def _min_phase_length_s(settings: Settings | None) -> float:
    if settings is not None:
        return settings.min_phase_length_s
    return DEFAULT_MIN_PHASE_LENGTH_S


def _merge_same_action(settings: Settings | None) -> bool:
    if settings is not None:
        return settings.merge_same_action
    return DEFAULT_MERGE_SAME_ACTION


def _exclude_outro_phases(settings: Settings | None) -> bool:
    if settings is not None:
        return settings.exclude_outro_phases
    return DEFAULT_EXCLUDE_OUTRO_PHASES


def _trim_edge_outro(
    steps: tuple[Step, ...], settings: Settings | None
) -> tuple[tuple[Step, ...], tuple[ExcludedSpan, ...]]:
    """Drop whole-video edge outros after deriving ends (contiguous+align).

    Legacy span mode and ``exact`` alignment keep every step exactly
    as before; otherwise the configured switch decides. Runs after
    stitching and deriving ends, so middle chunks and the contiguity
    gates never see a dropped edge.
    """
    if (
        not _exclude_outro_phases(settings)
        or _step_span_mode(settings) != "contiguous"
        or _step_alignment_mode(settings) != DEFAULT_ALIGNMENT_MODE
    ):
        return steps, ()
    return drop_edge_outro_phases(steps)


def plan_extraction_windows(
    duration_s: float, settings: Settings | None
) -> tuple[ChunkWindow, ...]:
    """Plan extraction windows, whole-video first when it fits.

    A video within the chunk length always yields its single window;
    a longer video within the whole-video ceiling yields one window
    too, but only when the honest single-window reservation fits the
    Spark cap at the active prices — otherwise it falls back to 60 s
    chunks, so a planned single window never refuses at client build
    time. Past the ceiling, plain 60 s chunks.
    """
    chunk_s = _chunk_s(settings)
    overlap_s = _chunk_overlap_s(settings)
    whole_max_s = _whole_video_max_s(settings)
    if duration_s <= whole_max_s and _whole_video_window_fits(
        duration_s, settings
    ):
        single_window_s = whole_max_s
    else:
        single_window_s = chunk_s
    return plan_chunks(
        duration_s, chunk_s, overlap_s, single_window_s=single_window_s
    )


def _window_pricing(
    settings: Settings | None,
) -> tuple[Decimal, Decimal, int, float]:
    """The (input price, output price, runs, rate) one window prices from."""
    if settings is not None:
        tuned = settings.providers
        return (
            tuned.spark_input_usd_per_token,
            tuned.spark_output_usd_per_token,
            tuned.spark_runs,
            settings.spark_input_tokens_per_video_s,
        )
    return (
        DEFAULT_SPARK_INPUT_USD_PER_TOKEN,
        DEFAULT_SPARK_OUTPUT_USD_PER_TOKEN,
        SPARK_RUNS_PER_GUIDE,
        DEFAULT_SPARK_INPUT_TOKENS_PER_VIDEO_S,
    )


def _window_estimate_usd(
    window_s: float, settings: Settings | None
) -> Decimal:
    """The shared reservation estimate for one window of ``window_s``.

    The same call child selection, preflight, retry accounting and the
    factory price through; the admitted child cap is
    :func:`_window_spark_cap_usd` on the same window.
    """
    input_price, output_price, runs, rate = _window_pricing(settings)
    return spark_window_estimate_usd(
        window_s,
        input_price,
        output_price,
        runs,
        tokens_per_video_s=rate,
    )


def _window_spark_cap_usd(
    window_s: float, settings: Settings | None
) -> Decimal:
    """The Spark job-context cap for one window of ``window_s``.

    The same call the factory installs on the window's clients: the
    ledger default, raised to the window's shared estimate when the
    window needs more.
    """
    input_price, output_price, runs, rate = _window_pricing(settings)
    return spark_window_cap_usd(
        window_s,
        input_price,
        output_price,
        runs,
        tokens_per_video_s=rate,
    )


def _whole_video_window_fits(
    duration_s: float, settings: Settings | None
) -> bool:
    """True when one window of ``duration_s`` reserves inside the cap."""
    input_price, output_price, runs, rate = _window_pricing(settings)
    return whole_video_fits(
        duration_s,
        input_price,
        output_price,
        runs,
        tokens_per_video_s=rate,
        spark_cap_usd=_spark_reservation_usd(settings),
    )


def chunk_cost_estimate(
    duration_s: float, spark: object, settings: Settings | None
) -> tuple[int, Decimal, Decimal]:
    """Return ``(chunk count, required $, job cap $)`` for one duration.

    The estimate behind the single budget preflight
    (:func:`chunk_budget_refused`): the upload route refuses over-cap
    confirmations before anything is sent to Meta, and the pipeline
    refuses over-cap runs before any cut, upload, or provider call.
    Each planned window prices through the shared window estimate —
    the same call child selection uses — so a window-sized single
    window can never pass preflight and then over-reserve. ``spark``
    is retained so route, pipeline and test call sites are unchanged;
    pricing comes from settings, not the client's shares.
    """
    windows = plan_extraction_windows(duration_s, settings)
    required = sum(
        (
            _window_estimate_usd(window.end_s - window.start_s, settings)
            for window in windows
        ),
        Decimal("0"),
    )
    return len(windows), required, _job_cost_cap_usd(settings)


def chunk_budget_refused(
    duration_s: float, spark: object, settings: Settings | None
) -> tuple[bool, int, Decimal, Decimal]:
    """Return ``(refused, chunk count, required $, job cap $)``.

    The ONE chunk-count cost preflight: the upload route, ``run_job``,
    and the pipeline stages all refuse through this helper, so the
    ``required > cap`` comparison lives in exactly one place. Every
    caller settles with :data:`STEP_EXTRACTION_OVER_BUDGET`.
    """
    chunk_count, required, cap = chunk_cost_estimate(duration_s, spark, settings)
    return required > cap, chunk_count, required, cap


def chunk_retry_fits(
    retry_spend_usd: Decimal,
    per_chunk_usd: Decimal,
    base_planned_usd: Decimal,
    cap_usd: Decimal,
    retry_budget_usd: Decimal,
) -> bool:
    """True when one more chunk retry fits both budget bounds.

    The per-video retry budget caps total retry spend, and the parent
    planned total plus retry spend must stay within the job cap. The
    caller checks this before admitting a retry attempt, so a retry
    that would cross the cap is refused before any upload or call.
    """
    one_more = retry_spend_usd + per_chunk_usd
    return (
        one_more <= retry_budget_usd
        and base_planned_usd + one_more <= cap_usd
    )


def guide_path(
    job_id: JobId | str,
    *,
    data_root: Path | None = None,
    guide_filename: str | None = None,
) -> Path:
    """Filesystem location of one job's stored pipeline guide."""
    text = str(job_id)
    if (
        not text
        or text in (".", "..")
        or "/" in text
        or "\\" in text
        or len(text) > 64
    ):
        raise ValueError(f"invalid job id for a guide path: {text!r}")
    name = guide_filename if guide_filename is not None else DEFAULT_GUIDE_FILENAME
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"invalid guide filename: {name!r}")
    root = Path(data_root) if data_root is not None else db.data_root()
    return job_dir(text, root=root) / name


REFUSAL_MARKER_FILENAME = "refusal.json"
STAGE_MARKER_FILENAME = "stage.txt"


def refusal_text_for_job(job_id: str, *, data_root: Path | None = None) -> str | None:
    """Plain-words reason a refused or failed job stopped, if recorded."""
    try:
        path = job_dir(job_id, root=data_root) / REFUSAL_MARKER_FILENAME
        kind = json.loads(path.read_text(encoding="utf-8")).get("kind")
    except (OSError, ValueError, AttributeError):
        return None
    return refusal_copy(kind) if isinstance(kind, str) else None


def stage_for_job(job_id: str, *, data_root: Path | None = None) -> str | None:
    """Last started parent stage, including work done by chunk children."""
    try:
        stage = (job_dir(job_id, root=data_root) / STAGE_MARKER_FILENAME).read_text(
            encoding="utf-8"
        )
    except (OSError, ValueError):
        return None
    return stage if stage in (
        RECEIPT_STAGE_EXTRACT, RECEIPT_STAGE_SEGMENT, RECEIPT_STAGE_READBACK
    ) else None


def _write_stage(ctx: JobContext, stage: str) -> None:
    try:
        persistence.atomic_write_text(
            job_dir(str(ctx.job_id)) / STAGE_MARKER_FILENAME, stage
        )
    except OSError as error:
        logger.warning(
            "stage marker job=%s unwritten: %s", ctx.job_id, type(error).__name__
        )


def pending_settlement_path(
    job_id: JobId | str,
    *,
    data_root: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Filesystem location of one job's pending-settlement marker, if any."""
    text = str(job_id)
    if (
        not text
        or text in (".", "..")
        or "/" in text
        or "\\" in text
        or len(text) > 64
    ):
        raise ValueError(f"invalid job id for a settlement path: {text!r}")
    name = filename if filename is not None else DEFAULT_PENDING_SETTLEMENT_FILENAME
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"invalid settlement filename: {name!r}")
    root = Path(data_root) if data_root is not None else db.data_root()
    return job_dir(text, root=root) / name


def read_pending_settlement(
    job_id: JobId | str,
    *,
    data_root: Path | None = None,
    filename: str | None = None,
) -> dict | None:
    """Return one job's pending-settlement marker, or None when settled."""
    try:
        raw = pending_settlement_path(
            job_id, data_root=data_root, filename=filename
        ).read_text(encoding="utf-8")
    except (FileNotFoundError, ValueError, OSError):
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def guide_to_json(guide: Guide) -> dict:
    """Serialize one assembled guide to JSON-safe mappings (paths as text)."""
    return {
        "job_id": str(guide.job_id),
        "steps": [
            {
                "number": step.number,
                "start_s": step.start_s,
                "end_s": step.end_s,
                "action": step.action,
                "tools": list(step.tools),
                "parts": [
                    {"phrase": part.phrase, "position_hint": part.position_hint}
                    for part in step.parts
                ],
                "hazards": list(step.hazards),
                "time_spread_s": step.time_spread_s,
                "time_needs_check": step.time_needs_check,
                "kind": step.kind,
            }
            for step in guide.steps
        ],
        "excluded_spans": [
            {
                "start_s": span.start_s,
                "end_s": span.end_s,
                "kind": span.kind,
            }
            for span in guide.excluded_spans
        ],
        "frame_paths": [str(path) for path in guide.frame_paths],
        "labels": [
            {
                "slot": {
                    "step_index": decision.slot.step_index,
                    "part_index": decision.slot.part_index,
                    "part": {
                        "phrase": decision.slot.part.phrase,
                        "position_hint": decision.slot.part.position_hint,
                    },
                },
                "detection": (
                    None
                    if decision.detection is None
                    else {
                        "phrase": decision.detection.phrase,
                        "mask_path": str(decision.detection.mask_path),
                        "score": decision.detection.score,
                        "box_xyxy": list(decision.detection.box_xyxy),
                    }
                ),
                "provisional": (
                    None
                    if decision.provisional is None
                    else {
                        "artifact_id": str(decision.provisional.artifact_id),
                        "rendered_image_path": str(
                            decision.provisional.rendered_image_path
                        ),
                        "position_hint": decision.provisional.position_hint,
                    }
                ),
                "checked": decision.checked,
                "status": decision.status,
                "readback": decision.readback,
            }
            for decision in guide.labels
        ],
        "difficulty": guide.difficulty,
        "verdict": guide.verdict,
        "draft_reasons": list(guide.draft_reasons),
    }


def _require_keys(mapping: object, keys: set[str], where: str) -> dict:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where}: expected a mapping")
    if set(mapping) != keys:
        raise ValueError(f"{where}: expected keys {sorted(keys)}")
    return mapping


def guide_from_json(value: object) -> Guide:
    """Rebuild one stored guide (strict; anything misshaped raises)."""
    keys = {
        "job_id",
        "steps",
        "frame_paths",
        "labels",
        "difficulty",
        "verdict",
        "draft_reasons",
    }
    if isinstance(value, dict) and "excluded_spans" in value:
        keys.add("excluded_spans")
    top = _require_keys(value, keys, "guide")
    if not isinstance(top["job_id"], str) or not top["job_id"]:
        raise ValueError("guide.job_id: expected a non-empty string")
    if top["difficulty"] not in ("Easy", "Medium", "Hard"):
        raise ValueError("guide.difficulty: expected Easy, Medium, or Hard")
    if not isinstance(top["verdict"], str) or not top["verdict"]:
        raise ValueError("guide.verdict: expected a non-empty string")
    if not isinstance(top["draft_reasons"], list) or not all(
        isinstance(reason, str) for reason in top["draft_reasons"]
    ):
        raise ValueError("guide.draft_reasons: expected a list of strings")
    if not isinstance(top["frame_paths"], list) or not all(
        isinstance(path, str) for path in top["frame_paths"]
    ):
        raise ValueError("guide.frame_paths: expected a list of strings")
    if not isinstance(top["steps"], list) or not top["steps"]:
        raise ValueError("guide.steps: expected a non-empty list")
    steps: list[Step] = []
    for index, raw in enumerate(top["steps"]):
        want = {
            "number",
            "start_s",
            "end_s",
            "action",
            "tools",
            "parts",
            "hazards",
            "time_spread_s",
            "time_needs_check",
        }
        if isinstance(raw, dict) and "kind" in raw:
            want = want | {"kind"}
        item = _require_keys(raw, want, f"guide.steps[{index}]")
        # Compatibility: guide files stored before kinds carry no
        # "kind" key and load as repair; a present kind must be exact.
        kind = item.get("kind", DEFAULT_PHASE_KIND)
        if kind not in PHASE_KINDS:
            raise ValueError(
                "guide.steps"
                f"[{index}].kind: expected one of {', '.join(PHASE_KINDS)},"
                f" got {kind!r}"
            )
        parts = tuple(
            PartRequest(
                phrase=_require_keys(part, {"phrase", "position_hint"}, "part")[
                    "phrase"
                ],
                position_hint=part["position_hint"],
            )
            for part in item["parts"]
        )
        steps.append(
            Step(
                number=item["number"],
                start_s=item["start_s"],
                end_s=item["end_s"],
                action=item["action"],
                tools=tuple(item["tools"]),
                parts=parts,
                hazards=tuple(item["hazards"]),
                time_spread_s=item["time_spread_s"],
                time_needs_check=item["time_needs_check"],
                kind=kind,
            )
        )
    # Compatibility default for guides stored before excluded spans existed.
    legacy_excluded_spans: tuple[ExcludedSpan, ...] = ()
    raw_excluded = top.get("excluded_spans", legacy_excluded_spans)
    if "excluded_spans" in top and not isinstance(raw_excluded, list):
        raise ValueError("guide.excluded_spans: expected a list")
    excluded: list[ExcludedSpan] = []
    for index, raw in enumerate(raw_excluded):
        span = _require_keys(
            raw,
            {"start_s", "end_s", "kind"},
            f"guide.excluded_spans[{index}]",
        )
        if span["kind"] not in PHASE_KINDS:
            raise ValueError(
                "guide.excluded_spans"
                f"[{index}].kind: expected one of {', '.join(PHASE_KINDS)},"
                f" got {span['kind']!r}"
            )
        excluded.append(
            ExcludedSpan(
                start_s=span["start_s"],
                end_s=span["end_s"],
                kind=span["kind"],
            )
        )
    labels: list[LabelDecision] = []
    for index, raw in enumerate(top["labels"]):
        item = _require_keys(
            raw,
            {"slot", "detection", "provisional", "checked", "status", "readback"},
            f"guide.labels[{index}]",
        )
        slot_raw = _require_keys(
            item["slot"], {"step_index", "part_index", "part"}, "slot"
        )
        part_raw = _require_keys(
            slot_raw["part"], {"phrase", "position_hint"}, "slot part"
        )
        slot = LabelSlot(
            step_index=slot_raw["step_index"],
            part_index=slot_raw["part_index"],
            part=PartRequest(
                phrase=part_raw["phrase"],
                position_hint=part_raw["position_hint"],
            ),
        )
        detection = None
        if item["detection"] is not None:
            det = _require_keys(
                item["detection"],
                {"phrase", "mask_path", "score", "box_xyxy"},
                "detection",
            )
            detection = Detection(
                phrase=det["phrase"],
                mask_path=Path(det["mask_path"]),
                score=det["score"],
                box_xyxy=tuple(det["box_xyxy"]),
            )
        provisional = None
        if item["provisional"] is not None:
            prov = _require_keys(
                item["provisional"],
                {"artifact_id", "rendered_image_path", "position_hint"},
                "provisional",
            )
            if detection is None:
                raise ValueError("guide label: provisional without a detection")
            provisional = ProvisionalRender(
                detection=detection,
                artifact_id=ArtifactId(prov["artifact_id"]),
                rendered_image_path=Path(prov["rendered_image_path"]),
                position_hint=prov["position_hint"],
            )
        labels.append(
            LabelDecision(
                slot=slot,
                detection=detection,
                provisional=provisional,
                checked=item["checked"],
                status=item["status"],
                readback=item["readback"],
            )
        )
    return Guide(
        job_id=JobId(top["job_id"]),
        steps=tuple(steps),
        frame_paths=tuple(Path(path) for path in top["frame_paths"]),
        labels=tuple(labels),
        difficulty=top["difficulty"],
        verdict=top["verdict"],
        draft_reasons=tuple(top["draft_reasons"]),
        excluded_spans=tuple(excluded),
    )


def save_guide(
    guide: Guide,
    *,
    data_root: Path | None = None,
    guide_filename: str | None = None,
) -> Path:
    """Store one assembled guide atomically under its job directory."""
    path = guide_path(guide.job_id, data_root=data_root, guide_filename=guide_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(guide_to_json(guide), sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".guide-", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(tmp_name).replace(path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def load_guide(
    job_id: JobId | str,
    *,
    data_root: Path | None = None,
    guide_filename: str | None = None,
) -> Guide | None:
    """Return one stored pipeline guide, or None when missing or corrupt."""
    try:
        raw = guide_path(
            job_id, data_root=data_root, guide_filename=guide_filename
        ).read_text(encoding="utf-8")
    except (FileNotFoundError, ValueError):
        return None
    except OSError as error:
        logger.warning("guide read job=%s failed: %s", job_id, type(error).__name__)
        return None
    try:
        guide = guide_from_json(json.loads(raw))
    except (ValueError, TypeError) as error:
        logger.warning("guide parse job=%s failed: %s", job_id, error)
        return None
    if str(guide.job_id) != str(job_id):
        logger.warning("guide job mismatch for %s", job_id)
        return None
    return guide


def deletion_schedule(
    job_id: JobId,
    *,
    lifecycle: LifecycleRepository,
    now: datetime | None = None,
) -> DeletionSchedule:
    """List this job's rows the hourly sweeper will collect (read-only)."""
    moment = db.utcnow() if now is None else now
    artifacts = tuple(
        str(record.id)
        for record in lifecycle.list_due_artifacts(moment)
        if str(record.job_id) == str(job_id)
    )
    meta_files = tuple(
        str(record.id)
        for record in lifecycle.list_due_meta_files(moment)
        if record.job_id is not None and str(record.job_id) == str(job_id)
    )
    return DeletionSchedule(
        job_id=job_id, artifact_ids=artifacts, meta_file_ids=meta_files
    )


def _job_row_state(job_id: JobId | str) -> str | None:
    """Return the jobs-table state for one id, or None when unreadable."""
    try:
        database = db.db_path()
    except Exception:
        return None
    try:
        conn = db.connect(database)
    except Exception:
        return None
    try:
        try:
            row = conn.execute(
                "SELECT state FROM jobs WHERE id = ?", (str(job_id),)
            ).fetchone()
        except Exception:
            return None
        return None if row is None else str(row["state"])
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _write_pending_marker(
    job_id: JobId | str,
    error: str,
    *,
    settings: Settings | None,
    attempts: int,
    guide_saved: bool,
) -> None:
    """Best-effort durable pending-settlement marker under the job directory.

    The job row stays active with its holds, so the sweeper's expiry pass
    can still recover it; the marker tells readers and the next wake why
    no terminal state was reported.
    """
    try:
        path = pending_settlement_path(
            job_id, filename=_pending_filename(settings)
        )
    except ValueError as path_error:
        logger.warning(
            "settlement marker job=%s refused: %s", job_id, path_error
        )
        return
    body = {
        "job_id": str(job_id),
        "error": error,
        "attempts": attempts,
        "guide_saved": guide_saved,
        "at": db.utcnow().isoformat(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        persistence.atomic_write_text(path, json.dumps(body, sort_keys=True))
    except OSError as write_error:
        logger.warning(
            "settlement marker job=%s unwritten: %s",
            job_id, type(write_error).__name__,
        )


def _run_terminal_transition(
    operation: Callable[[], None], *, attempts: int
) -> LedgerError | None:
    """Run one finish() with bounded ContentionError retries.

    Returns None when the transition is confirmed, else the last ledger
    error. Non-contention errors are not retried; anything outside the
    ledger family propagates loudly.
    """
    last: LedgerError | None = None
    for _ in range(max(1, attempts)):
        try:
            operation()
            return None
        except ContentionError as error:
            last = error
        except LedgerError as error:
            return error
    return last


def _settle_failed(
    ctx: JobContext, error: str, *, settings: Settings | None
) -> Literal["failed", "pending_settlement"]:
    """Confirm the failed transition; pending marker when it cannot be."""
    attempts = _settle_max_attempts(settings)
    failure = _run_terminal_transition(
        lambda: ctx.finish(
            "failed",
            failed_output_names=(
                _guide_filename(settings),
                settings.corrections_filename if settings else DEFAULT_CORRECTIONS_FILENAME,
            ),
        ),
        attempts=attempts,
    )
    if failure is None:
        return "failed"
    if isinstance(failure, JobAlreadyFinished):
        # A racing finish may have settled the job first: only a row that
        # reads back failed is reported failed; anything else is pending.
        if _job_row_state(ctx.job_id) == "failed":
            return "failed"
    logger.warning(
        "terminal settlement job=%s unconfirmed after %d attempt(s): %s",
        ctx.job_id, attempts, type(failure).__name__,
    )
    _write_pending_marker(
        ctx.job_id, error, settings=settings, attempts=attempts,
        guide_saved=False,
    )
    return "pending_settlement"


def _write_refusal_marker(
    ctx: JobContext, job_id: JobId | None, refusal_kind: str
) -> None:
    # The job row ends "failed"; this marker lets the status page say why.
    try:
        marker = job_dir(str(ctx.job_id)) / REFUSAL_MARKER_FILENAME
        marker.parent.mkdir(parents=True, exist_ok=True)
        fd, staging_name = tempfile.mkstemp(
            dir=marker.parent, prefix=f".{REFUSAL_MARKER_FILENAME}."
        )
        staging = Path(staging_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": refusal_kind}))
                handle.flush()
                os.fsync(handle.fileno())
            os.link(staging, marker)
        finally:
            staging.unlink(missing_ok=True)
    except FileExistsError:
        pass
    except OSError as write_error:
        logger.warning(
            "refusal marker job=%s unwritten: %s", job_id, type(write_error).__name__
        )


def _failure_kind(error: str) -> str:
    if error == "job deadline expired before processing":
        return "deadline"
    if error == "part check failed":
        return "part_check"
    if error == "step extraction failed":
        return "extraction"
    if error in ("could not upload video chunks", "upload not confirmed"):
        return "upload_failed"
    if error == "upload expired before processing":
        return "upload_expired"
    if error == "background job could not start":
        return "queue_unavailable"
    if error == "interrupted by a restart":
        return "interrupted"
    if error == "job cancelled":
        return "cancelled"
    if error == "pipeline unavailable":
        return "service_unavailable"
    if error.startswith(("could not cut", "could not burn", "could not detect")):
        return "processing"
    if error.startswith("frame scoring"):
        return "frames"
    if error.startswith("read-back"):
        return "readback"
    if error.startswith(("could not store", "could not keep", "guide assembly")):
        return "finalize"
    return "failure"


def _fail(
    ctx: JobContext | None,
    job_id: JobId | None,
    error: str,
    *,
    replay: bool,
    settings: Settings | None = None,
    refusal_kind: str | None = None,
) -> PipelineResult:
    state: PipelineState = "failed"
    if ctx is not None:
        _write_refusal_marker(ctx, job_id, refusal_kind or _failure_kind(error))
    if ctx is not None:
        state = _settle_failed(ctx, error, settings=settings)
        if state == "failed":
            try:
                cleanup_failed_guide(
                    job_id, artifacts=ctx.artifacts,
                    guide_filename=_guide_filename(settings),
                    corrections_filename=(settings.corrections_filename if settings else DEFAULT_CORRECTIONS_FILENAME),
                )
            except Exception as cleanup_error:
                # The failed transition already queued outputs and images.
                logger.warning(
                    "failed job=%s cleanup left to the sweep: %s",
                    job_id, type(cleanup_error).__name__,
                )
            try:
                record_event(ctx, "job.failed", {})
            except Exception as event_error:
                logger.warning(
                    "failure event job=%s failed: %s",
                    job_id, type(event_error).__name__,
                )
    if state == "pending_settlement":
        error = f"{error} (terminal settlement pending)"
    return PipelineResult(
        state=state, job_id=job_id, guide=None, error=error, replay=replay
    )


def _abandon_publication(
    ctx: JobContext, job_id: JobId, error: str, *,
    replay: bool, settings: Settings | None,
) -> PipelineResult:
    """Remove partial canonical output after cancel, expiry, or failed finish."""
    state = _job_row_state(job_id)
    if state == "done":
        return PipelineResult(
            state="done", job_id=job_id,
            guide=load_guide(job_id, guide_filename=_guide_filename(settings)),
            error=None, replay=replay,
        )
    if state != "active":
        _write_refusal_marker(
            ctx, job_id, "deadline" if state == "expired" else "finalize"
        )
    result = (
        _fail(ctx, job_id, error, replay=replay, settings=settings)
        if state == "active"
        else PipelineResult(
            state="failed", job_id=job_id, guide=None,
            error=error, replay=replay,
        )
    )
    try:
        cleanup_failed_guide(
            job_id, artifacts=ctx.artifacts,
            guide_filename=_guide_filename(settings),
            corrections_filename=(settings.corrections_filename if settings else DEFAULT_CORRECTIONS_FILENAME),
        )
    except Exception as cleanup_error:
        # The failed transition queued the guide; the sweep rescans the rest.
        logger.warning(
            "abandoned job=%s cleanup left to the sweep: %s",
            job_id, type(cleanup_error).__name__,
        )
    return result


def fail_job(
    ctx: JobContext, error: str, *, replay: bool,
    settings: Settings | None = None,
) -> PipelineResult:
    """Fail one admitted job closed: it could not start or was cut off."""
    return _fail(ctx, ctx.job_id, error, replay=replay, settings=settings)


def _fail_unsettled(
    ctx: JobContext | None,
    job_id: JobId | None,
    error: str,
    *,
    replay: bool,
    settings: Settings | None = None,
) -> PipelineResult:
    """Settle the parent failed, but always report pending settlement.

    A failed child is still active with held reservations, so even a
    confirmed parent transition leaves terminal work unconfirmed:
    the parent says so explicitly instead of reporting clean failed.
    """
    if ctx is not None:
        _write_refusal_marker(ctx, job_id, "settlement")
        _settle_failed(ctx, error, settings=settings)
    return PipelineResult(
        state="pending_settlement",
        job_id=job_id,
        guide=None,
        error=f"{error} (terminal settlement pending)",
        replay=replay,
    )


def _refuse(
    ctx: JobContext | None,
    job_id: JobId | None,
    error: str,
    *,
    replay: bool,
    settings: Settings | None = None,
    refusal_kind: str,
) -> PipelineResult:
    state: PipelineState = "refused"
    if ctx is not None:
        _write_refusal_marker(ctx, job_id, refusal_kind)
    if ctx is not None:
        if _settle_failed(ctx, error, settings=settings) == "pending_settlement":
            state = "pending_settlement"
            error = f"{error} (terminal settlement pending)"
    return PipelineResult(
        state=state, job_id=job_id, guide=None, error=error, replay=replay
    )


class _DeferredFinish:
    """A JobContext proxy that withholds the success transition.

    ``assemble_guide`` finishes ``done`` before returning, which would
    commit success before the guide bytes are durable. This proxy stashes
    a ``done`` finish instead of forwarding it; the caller stores the
    guide first and then replays the stashed transition with
    :meth:`commit_done`. A ``failed`` finish forwards immediately, since
    failure needs no durability gate. Everything else delegates.
    """

    def __init__(self, inner: JobContext) -> None:
        self._inner = inner
        self._stashed_done: tuple[Sequence[ArtifactId], tuple[str, str] | None] | None = None
        self._settled = False

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def finish(
        self,
        outcome: Literal["done", "failed"],
        *,
        keep_artifact_ids: Sequence[ArtifactId] = (),
        failed_output_names: tuple[str, str] | None = None,
    ) -> None:
        if outcome == "done":
            self._stashed_done = (tuple(keep_artifact_ids), failed_output_names)
            return
        self._inner.finish(
            outcome, keep_artifact_ids=keep_artifact_ids, failed_output_names=failed_output_names
        )
        self._settled = True

    def commit_done(self) -> None:
        """Replay the stashed success transition, once; no-op when settled."""
        if self._settled or self._stashed_done is None:
            return
        keeps, failed_output_names = self._stashed_done
        self._inner.finish(
            "done", keep_artifact_ids=keeps, failed_output_names=failed_output_names
        )
        self._stashed_done = None
        self._settled = True


def _commit_done(
    deferred: _DeferredFinish, *, settings: Settings | None
) -> Literal["done", "pending_settlement"]:
    """Confirm the deferred success transition; pending marker when not."""
    attempts = _settle_max_attempts(settings)
    failure = _run_terminal_transition(deferred.commit_done, attempts=attempts)
    if failure is None:
        return "done"
    logger.warning(
        "success settlement job=%s unconfirmed after %d attempt(s): %s",
        deferred.job_id, attempts, type(failure).__name__,
    )
    _write_pending_marker(
        deferred.job_id, "success transition unconfirmed",
        settings=settings, attempts=attempts, guide_saved=True,
    )
    return "pending_settlement"


# ---------------------------------------------------------------------------
# Receipt translation: client records become CallRecords the evidence module accepts, at this seam.
# The module shapes are pinned by their own modules' tests, so the mapping lives
# here (and only here) instead of in the emitting modules.
# ---------------------------------------------------------------------------


def translate_sam_record(record: CallRecord) -> CallRecord:
    """Map one SAM sink record onto a valid segment receipt.

    Only a provider ``ok`` is success; every other SAM status (empty,
    below-threshold, missing confidence, provider error/timeout) is a
    failed call. Evidence keeps the phrase and confidence for the evidence module's
    segment schema to allowlist.
    """
    evidence = record.evidence if isinstance(record.evidence, dict) else {}
    return CallRecord(
        job_id=record.job_id,
        stage=SAM_SEGMENT_STAGE,
        model=record.model,
        endpoint=SAM_SEGMENT_ENDPOINT,
        latency_ms=record.latency_ms,
        units_in=record.units_in,
        units_out=record.units_out,
        estimated_cost_usd=record.estimated_cost_usd,
        status="succeeded" if record.status == "ok" else "failed",
        evidence={
            "phrase": evidence.get("phrase"),
            "score": evidence.get("c"),
        },
    )


def sam_receipt_sink(ctx: JobContext, record: CallRecord) -> None:
    """SAM call sink that persists every SAM call as the evidence module segment receipt."""
    record_call(ctx, translate_sam_record(record))


def translate_readback_record(record: CallRecord) -> CallRecord:
    """Map one read-back hook record onto a valid readback receipt.

    A fired read-back is a succeeded call whatever it judged: the yes/no
    answer rides the evidence decision, not the call status.
    """
    evidence = record.evidence if isinstance(record.evidence, dict) else {}
    return CallRecord(
        job_id=record.job_id,
        stage=RECEIPT_STAGE_READBACK,
        model=record.model,
        endpoint=READBACK_RECEIPT_ENDPOINT,
        latency_ms=record.latency_ms,
        units_in=record.units_in,
        units_out=record.units_out,
        estimated_cost_usd=record.estimated_cost_usd,
        status="succeeded",
        evidence={
            "phrase": evidence.get("phrase"),
            "decision": record.status,
            "score": evidence.get("score"),
        },
    )


def readback_record_call(ctx: JobContext, record: CallRecord) -> None:
    """Read-back record_call hook that persists readback receipts the evidence module accepts."""
    record_call(ctx, translate_readback_record(record))


def readback_record_event(
    ctx: JobContext, event: str, detail: Mapping[str, JSONValue]
) -> None:
    """Read-back record_event hook mapped onto the counted-read-backs event."""
    if event == READBACK_EVENT:
        fired = detail.get("readback_fired_n", 0)
        record_event(
            ctx,
            READBACK_COUNTED_EVENT,
            {"stage": RECEIPT_STAGE_READBACK, "count": fired},
        )
        return
    record_event(ctx, event, detail)


class RecordingSparkClient:
    """Spark client wrapper that persists one extract receipt per run.

    Delegates every call; a ``run_steps`` result with a well-formed cost
    and latency is recorded before it is returned, so a receipt that
    cannot persist fails the job instead of vanishing. A malformed
    result is left for the extractor to refuse (logged, unrecorded):
    the wrapper never invents a cost. ``read_back`` delegates without
    recording — read-backs record through their own hooks.
    """

    def __init__(self, client: SparkClient, ctx: JobContext) -> None:
        self._client = client
        self._ctx = ctx
        self._runs = 0

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    def run_steps(
        self,
        video_file_id: str,
        *,
        clip_duration_s: float | None = None,
        scene_cuts: tuple[float, ...] | None = None,
        clocked: bool = False,
    ):
        result = self._client.run_steps(  # type: ignore[attr-defined]
            video_file_id,
            clip_duration_s=clip_duration_s,
            scene_cuts=scene_cuts,
            clocked=clocked,
        )
        index = self._runs
        self._runs += 1
        cost = getattr(result, "cost_usd", None)
        latency = getattr(result, "latency_ms", None)
        if (
            not isinstance(cost, Decimal)
            or not cost.is_finite()
            or cost < 0
            or not isinstance(latency, int)
            or isinstance(latency, bool)
            or latency < 0
        ):
            logger.warning(
                "spark receipt job=%s run %d skipped: malformed cost/latency",
                self._ctx.job_id, index,
            )
            return result
        record_call(
            self._ctx,
            CallRecord(
                job_id=self._ctx.job_id,
                stage=RECEIPT_STAGE_EXTRACT,
                model=SPARK_MODEL_ID,
                endpoint=SPARK_EXTRACT_ENDPOINT,
                latency_ms=latency,
                units_in=None,
                units_out=None,
                estimated_cost_usd=cost,
                status="failed" if getattr(result, "failure_kind", None) else "succeeded",
                evidence={"run_index": index},
            ),
        )
        return result

    def read_back(self, *args, **kwargs):
        return self._client.read_back(*args, **kwargs)  # type: ignore[attr-defined]


class RecordingMetaClient:
    """Meta files wrapper that persists one upload receipt per upload.

    Delegates every call; a successful ``upload_file`` is recorded with
    its measured latency before it is returned.
    """

    # shortcut: the upload carries no reservation, so its cost is untracked
    # and recorded as $0; upgrade to a metered cost when metering exists.
    def __init__(self, client: MetaFilesClient, ctx: JobContext) -> None:
        self._client = client
        self._ctx = ctx

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    def upload_file(
        self, path: Path, *, expires_after_hours: int, idempotency_key: str
    ):
        started = time.monotonic()
        uploaded = self._client.upload_file(
            path,
            expires_after_hours=expires_after_hours,
            idempotency_key=idempotency_key,
        )
        record_call(
            self._ctx,
            CallRecord(
                job_id=self._ctx.job_id,
                stage=META_UPLOAD_STAGE,
                model=META_UPLOAD_MODEL,
                endpoint=META_UPLOAD_ENDPOINT,
                latency_ms=int((time.monotonic() - started) * 1000),
                units_in=None,
                units_out=None,
                estimated_cost_usd=Decimal("0"),
                status="succeeded",
                evidence={},
            ),
        )
        return uploaded

    def find_upload(self, idempotency_key: str):
        return self._client.find_upload(idempotency_key)  # type: ignore[attr-defined]

    def delete_file(self, file_id: MetaFileId) -> None:
        self._client.delete_file(file_id)  # type: ignore[attr-defined]

    def get_file(self, file_id: MetaFileId):
        return self._client.get_file(file_id)  # type: ignore[attr-defined]


def _checked_slots(bundle) -> list[tuple[Step, int, PartRequest]]:
    """(step, part_index, part) for exact fan/battery parts, in step order."""
    found: list[tuple[Step, int, PartRequest]] = []
    for step in bundle.steps:
        for part_index, part in enumerate(step.parts):
            if is_checked_part(part.phrase):
                found.append((step, part_index, part))
    return found


def _second_check_budget(settings: Settings):
    """The per-job second-check spend book: the guide's corrections sidecar."""
    from app.corrections import CorrectionStore

    return CorrectionStore(
        lambda job_id: job_dir(job_id, root=settings.data_root),
        settings.corrections_filename,
        max_ops=settings.correction_max_ops,
        max_chars=settings.correction_phrase_max_chars,
        guide_filename=settings.guide_filename,
    )


def run_job(
    clip: Path,
    *,
    clients: PipelineClients | None,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    anon_id: str,
    ip_hash: str,
    replay: bool = False,
    settings: Settings | None = None,
    retain_clip: bool = True,
) -> PipelineResult:
    """Run one clip end to end; see the module docstring for the stage order.

    Client misuse (live clients in replay mode, replay clients or no
    clients in live mode) raises before any side effect. Live mode is
    refused outright: this entry point admits its own job after the
    bundle was built, so any pre-bound live bundle would receipt
    against the wrong job; live jobs run only through the
    confirmed-job path (:func:`run_confirmed_job`, via
    ``service.run_upload_job``), which validates the bundle against
    the loaded job context. ``retain_clip=False`` skips guide-clip
    retention at publish; the sample build passes it so the fixture
    guide keeps its sourceless player.
    """
    resolved = resolve_run_clients(clients, replay=replay)
    if not replay:
        raise ValueError(
            "live mode cannot run through run_job: it admits a new job"
            " after the bundle was built; use the confirmed-job path"
            " (service.run_upload_job)"
        )
    try:
        video_hash = hash_file(Path(clip))
    except OSError:
        return PipelineResult(
            state="failed",
            job_id=None,
            guide=None,
            error="could not read clip",
            replay=replay,
        )
    try:
        ctx = admit_job(anon_id, ip_hash, video_hash, **_output_names(settings))
    except LedgerError as error:
        return PipelineResult(
            state="refused",
            job_id=None,
            guide=None,
            error=f"admission refused: {type(error).__name__}",
            replay=replay,
        )
    job_id = ctx.job_id
    try:
        with open(clip, "rb") as handle:
            preview = prepare_preview(
                ctx, handle, Path(clip).name, uploads=uploads,
                max_duration_s=_max_video_s(settings),
            )
    except OverLimitError as error:
        logger.info("pipeline job=%s refused: %s", job_id, error.reason)
        return _refuse(ctx, job_id, error.message, replay=replay,
                       settings=settings, refusal_kind="duration")
    except Exception as error:
        logger.info("pipeline job=%s preview failed: %s", job_id, type(error).__name__)
        return _fail(ctx, job_id, "could not process clip", replay=replay,
                     settings=settings)
    refused, chunk_count, required, cap = chunk_budget_refused(
        preview.duration_s, resolved.spark, settings
    )
    if refused:
        logger.info(
            "pipeline job=%s refused: %d chunk(s) need $%s over the $%s job cap",
            job_id, chunk_count, required, cap,
        )
        return _refuse(
            ctx, job_id, STEP_EXTRACTION_OVER_BUDGET,
            replay=replay, settings=settings, refusal_kind="budget",
        )
    try:
        meta_file_id = confirm_and_send(
            job_id, uploads=uploads,
            client=RecordingMetaClient(resolved.meta, ctx),
        )
    except Exception as error:
        logger.info("pipeline job=%s confirm failed: %s", job_id, type(error).__name__)
        return _fail(ctx, job_id, "confirmation failed", replay=replay,
                     settings=settings)
    return run_stages(
        ctx, meta_file_id, preview.local_video,
        clients=resolved, lifecycle=lifecycle, uploads=uploads,
        replay=replay, settings=settings,
        duration_s=preview.duration_s,
        retain_clip=retain_clip,
    )


def run_confirmed_job(
    job_id: JobId | str,
    *,
    clients: PipelineClients | None,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    replay: bool = False,
    settings: Settings | None = None,
) -> PipelineResult:
    """Run the pipeline stages for one already-confirmed upload job.

    The upload route confirms first (preview stored, Meta file attached);
    this entry point loads that durable state and runs extraction onward
    through the same :func:`run_stages` as :func:`run_job`. An unknown,
    unconfirmed, or hidden upload fails the job closed.
    """
    resolved = resolve_run_clients(clients, replay=replay)
    loaded = uploads.load_job(JobId(str(job_id)))
    if loaded is None:
        return PipelineResult(
            state="failed", job_id=JobId(str(job_id)), guide=None,
            error="unknown job", replay=replay,
        )
    ctx, record = loaded
    meta_file_id = record.meta_file_id
    if record.state != "confirmed" or meta_file_id is None or record.hidden:
        return _fail(
            ctx, ctx.job_id, "upload not confirmed",
            replay=replay, settings=settings,
        )
    if record.meta_expires_at is None or record.meta_expires_at <= db.utcnow():
        return _fail(
            ctx, ctx.job_id, "upload expired before processing",
            replay=replay, settings=settings,
        )
    return run_stages(
        ctx, meta_file_id, record.preview.local_video,
        clients=resolved, lifecycle=lifecycle, uploads=uploads,
        replay=replay, settings=settings,
        duration_s=record.preview.duration_s,
    )


def run_stages(
    ctx: JobContext,
    meta_file_id: MetaFileId | str,
    local_video: Path,
    *,
    clients: PipelineClients,
    lifecycle: LifecycleRepository,
    uploads: UploadStore,
    replay: bool,
    settings: Settings | None = None,
    duration_s: float,
    retain_clip: bool = True,
) -> PipelineResult:
    """Run extraction through guide storage for one admitted job.

    Replay mode installs the SAM receipt sink for this thread's run
    (restoring the previous sink afterward) and records Meta, Spark,
    SAM, and read-back receipts through ``evidence.record_call``. The
    sink is context-local, so overlapping jobs keep their own
    receipts. Live mode leaves the wrapper sinks unwired: the live
    clients persist their own single receipt per call, and the SAM
    wrapper records stay on the ambient (log) sink. The read-back
    counted events stay wired in both modes; clients emit no events,
    so nothing duplicates. The callers guarantee validated, unmixed
    clients. Live mode additionally refuses a bundle bound to any job
    but this one, before any reservation, request or receipt.
    ``duration_s`` is the admitted preview length; past one chunk the
    extraction fans out per chunk (see :func:`_extract_chunked`). A
    ``duration_s`` over the configured ``max_video_s`` cap is refused
    before any planning, upload, or provider call.
    """
    if not replay:
        _require_live_binding(clients, ctx)
        return _run_stages_inner(
            ctx, meta_file_id, local_video,
            clients=clients, lifecycle=lifecycle, uploads=uploads,
            replay=replay, settings=settings, duration_s=duration_s,
            retain_clip=retain_clip,
        )
    previous_sink = sam_module._call_sink.get()
    install_call_sink(sam_receipt_sink)
    try:
        return _run_stages_inner(
            ctx, meta_file_id, local_video,
            clients=clients, lifecycle=lifecycle, uploads=uploads,
            replay=replay, settings=settings, duration_s=duration_s,
            retain_clip=retain_clip,
        )
    finally:
        install_call_sink(previous_sink)


class _ChunkRetryRefused(Exception):
    """A chunk retry was refused up front: attempts left, budget gone.

    Carries the honest parent-facing message naming the chunk, the
    disagreement, and the budget bound it would cross.
    """


class _ChunkBudgetRefused(Exception):
    """A chunk's decoded cost crosses the job cap: refused pre-admission.

    Raised after the probe and before child admission when the running
    decoded-duration total exceeds the job cap: no child, upload, or
    provider call follows. Carries the honest parent-facing message.
    """


class _ParentCancelled(Exception):
    """The parent upload was cancelled before a child's upload/extraction."""


class _ChunkAttemptsExhausted(Exception):
    """A chunk used all its configured attempts, across runs.

    Raised before admission when the durable next attempt number
    already reaches the per-chunk maximum: no new child, upload, or
    provider call follows. The parent fails closed exactly like an
    in-run exhaustion.
    """


class _ChunkSettlementPending(Exception):
    """A failed child's settlement could not confirm: no retry follows.

    Raised instead of starting the next attempt, so the old child
    keeps its holds untouched for the sweeper and the parent reports
    pending settlement rather than papering over it with a retry.
    """


class _ChunkClaimHeld(Exception):
    """Another worker holds this chunk's extraction claim.

    Raised when the per-chunk claim is held and no finished bundle
    appeared on recheck: this run paid nothing for the chunk, and
    the parent is left untouched for the holder to finish — the run
    reports contention with an honest retry-later message instead of
    failing the shared parent.
    """


def _chunk_first_free_attempt(
    uploads: UploadStore, parent_id: str, index: int
) -> int:
    """First unclaimed attempt number for one chunk (0 when untouched).

    Attempt 0 owns the base key ``{parent}:chunk:{index}``; attempt N
    owns ``{parent}:chunk:{index}:retry:{N}``. A re-entry after
    interruption resumes numbering where the earlier run stopped, so
    a resumed chunk never reissues a key the first claim still owns.
    A store read failure propagates: the chunk fails closed before
    any side effect rather than risking a duplicate claim.
    """
    base = f"{parent_id}:chunk:{index}"
    if not uploads.chunk_claim_taken(base):
        return 0
    attempt = 1
    while uploads.chunk_claim_taken(f"{base}:retry:{attempt}"):
        attempt += 1
    return attempt


def _load_resumed_bundle(
    digest: str, config: SparkStageConfig, ttl_s: float
) -> StepBundle | None:
    """Return the finished bundle for one digest, or None to rerun.

    An unreadable store fails closed to None — the chunk reruns —
    exactly like a missing, corrupt, or expired bundle.
    """
    try:
        return load_chunk_bundle(digest, config, ttl_s=ttl_s)
    except (OSError, RecursionError):
        return None


def _resumed_cut_matches(
    digest: str, real_window: ChunkWindow, ttl_s: float
) -> bool:
    """True when the record's cut matches the bytes just probed.

    The digest already binds the fresh probe, so an honest record
    always matches; a mismatch — or a record without usable cut
    facts — means the bundle is not for this cut, so the chunk
    reruns instead of reusing it. An unreadable store fails closed
    to False.
    """
    try:
        loaded = load_chunk_cut_window(digest, ttl_s=ttl_s)
    except (OSError, RecursionError):
        return False
    return loaded is not None and loaded == real_window


@contextlib.contextmanager
def _chunk_extraction_claim(parent_id: str, index: int) -> Iterator[None]:
    """Hold one chunk's exclusive extraction claim across processes.

    Same shape as the mask-publish locks: the lock file is created
    once and never unlinked, so a later worker can never claim beside
    a worker still holding the old inode. Non-blocking: raises
    :class:`_ChunkClaimHeld` when another worker holds the chunk, so
    the caller rechecks the bundle instead of paying twice for it.
    The OS releases the lock on process death, so a crashed holder
    never wedges the resume: no TTL, no steal, no second payer. Any
    other lock failure propagates, failing closed before any paid
    call rather than extracting without exclusion.
    """
    path = job_dir(parent_id) / f"chunk-{index}.lock"
    handle = open(path, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise _ChunkClaimHeld(
                    f"chunk {index} claim is held"
                ) from None
            raise
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            handle.close()


def _chunk_retry_spend_usd(
    uploads: UploadStore, parent_id: str, chunk_count: int
) -> Decimal:
    """Durable retry spend: ledger actuals plus holds of claimed retries.

    Every claimed retry attempt prices from its own child job's
    reservation rows — settled actuals for what was paid, held
    estimates for what is still held — never as retry-count times
    today's Spark shares, so a repricing restart cannot re-spend a
    budget the earlier runs already consumed. An attempt that never
    reserved (crashed before its first call) prices at zero: nothing
    was paid and nothing is held.
    """
    spend = Decimal("0")
    for index in range(chunk_count):
        base = f"{parent_id}:chunk:{index}"
        first_free = _chunk_first_free_attempt(uploads, parent_id, index)
        for attempt in range(1, first_free):
            found = uploads.load_chunk_claim(f"{base}:retry:{attempt}")
            if found is None:
                # Defensive: claim rows are never deleted, so a taken
                # key always reads back; skip rather than wedge.
                continue
            job_id, _ = found
            actual_usd, held_usd = job_reservation_spend_usd(job_id)
            spend += actual_usd + held_usd
    return spend


def _extract_retry_spend_usd(parent_id: str) -> Decimal:
    """Durable spend of extra window calls, including any outstanding holds."""
    with contextlib.closing(db.connect()) as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE anon_id LIKE ?",
            (f"{parent_id}/extract/retry/%",),
        ).fetchall()
    return sum(
        (sum(job_reservation_spend_usd(JobId(row["id"])), Decimal("0")) for row in rows),
        Decimal("0"),
    )


def _job_remaining_s(ctx: JobContext) -> float:
    remaining = getattr(ctx, "remaining_s", None)
    return float(remaining()) if callable(remaining) else float(JOB_DEADLINE_S)


def _window_retry_call(
    parent_ctx: JobContext,
    window_ctx: JobContext,
    original_client: SparkClient,
    uploads: UploadStore,
    settings: Settings | None,
    *,
    window_index: int,
    window_s: float,
    planned_usd: Callable[[], Decimal],
    chunk_count: int,
    replay: bool,
) -> Callable[[str, float | None, tuple[float, ...] | None, bool], SparkRunResult] | None:
    if settings is None or settings.extract_retry_count == 0:
        return None
    parent_id = str(parent_ctx.job_id)
    share = spark_config_of(original_client).run_shares[0]

    def retry(
        file_id: str, duration_s: float | None,
        cuts: tuple[float, ...] | None, clocked: bool,
    ) -> SparkRunResult:
        def check(backoff_pending: bool) -> None:
            if _parent_cancelled(uploads, parent_id):
                raise _ParentCancelled(f"parent job {parent_id} was cancelled")
            need = settings.extract_retry_min_remaining_s + (
                settings.extract_retry_backoff_s if backoff_pending else 0
            )
            if min(_job_remaining_s(parent_ctx), _job_remaining_s(window_ctx)) < need:
                raise SparkServiceError("deadline")
            spent = _extract_retry_spend_usd(parent_id)
            if chunk_count:
                spent += _chunk_retry_spend_usd(uploads, parent_id, chunk_count)
            if spent + share > settings.chunk_retry_budget_usd or planned_usd() + spent + share > settings.job_cost_cap_usd:
                raise SparkServiceError("retry budget")

        check(True)
        time.sleep(settings.extract_retry_backoff_s)
        check(False)
        label = f"{parent_id}/extract/retry/{window_index}"

        def claim_budget(conn: sqlite3.Connection) -> None:
            # Admission holds the write lock until this check and the child
            # insert commit. An active child counts its share even before its
            # first per-call reservation is written.
            rows = conn.execute(
                "SELECT id, state, anon_id FROM jobs WHERE anon_id LIKE ? OR anon_id LIKE ?",
                (f"{parent_id}/extract/retry/%", f"{parent_id}/chunk/%/retry/%"),
            ).fetchall()
            spent = Decimal("0")
            for row in rows:
                reservations = conn.execute(
                    "SELECT state, estimated_usd, actual_usd FROM reservations"
                    " WHERE job_id = ? AND state IN ('held', 'settled', 'failed')",
                    (row["id"],),
                ).fetchall()
                amount = sum((
                    Decimal(item["estimated_usd"] if item["state"] == "held" else item["actual_usd"])
                    for item in reservations
                ), Decimal("0"))
                if row["state"] == "active" and row["anon_id"].startswith(f"{parent_id}/extract/retry/"):
                    amount = max(amount, share)
                spent += amount
            if spent + share > settings.chunk_retry_budget_usd or planned_usd() + spent + share > settings.job_cost_cap_usd:
                raise SparkServiceError("retry budget")

        try:
            child = admit_job(
                label, label, label, internal=True,
                spark_cap_usd=_window_spark_cap_usd(window_s, settings),
                admission_check=claim_budget, **_output_names(settings),
            )
        except sqlite3.IntegrityError as error:
            if "idx_jobs_extract_retry_claim" not in str(error) and "jobs.anon_id" not in str(error):
                raise
            raise SparkServiceError("retry already used") from error
        try:
            if child.remaining_s() < settings.extract_retry_min_remaining_s:
                raise SparkServiceError("child deadline")
            client = (
                RecordingSparkClient(
                    original_client._client if isinstance(original_client, RecordingSparkClient) else original_client,
                    child,
                )
                if replay else select_providers(settings, child, window_s=window_s).spark
            )
            hold = child.reserve("spark_steps", share)
            try:
                if _parent_cancelled(uploads, parent_id) or min(_job_remaining_s(parent_ctx), _job_remaining_s(window_ctx), child.remaining_s()) < settings.extract_retry_min_remaining_s:
                    raise SparkServiceError("cancelled or deadline")
                result = client.run_steps(
                    file_id, clip_duration_s=duration_s,
                    scene_cuts=cuts, clocked=clocked,
                )
            except Exception:
                hold.settle(Decimal("0"), "failed")
                raise
            except BaseException:
                hold.settle(share, "failed")
                raise
            failure = getattr(result, "failure_kind", None)
            if failure:
                hold.settle(_failed_actual(result), "failed")
                raise SparkServiceError(failure) if failure in settings.extract_retry_failures else SparkClientError(failure)
            try:
                checked = _require_valid_run_result(result)
                validate_spark_payload(checked.payload)
            except (SparkClientError, SparkSchemaError) as error:
                hold.settle(_failed_actual(result), "failed")
                raise SparkClientError(str(error)) from error
            hold.settle(checked.cost_usd, "succeeded")
            child.finish("done")
            return checked
        except BaseException as error:
            if _settle_failed(child, str(error) or "extract retry failed", settings=settings) == "pending_settlement":
                raise _ChunkSettlementPending("extract retry settlement is pending")
            raise

    return retry


def _drain_prior_chunk_uploads(
    uploads: UploadStore,
    parent_id: str,
    index: int,
    *,
    client: MetaFilesClient,
    lifecycle: LifecycleRepository,
) -> None:
    """Best-effort delete of earlier runs' Meta ids for one chunk.

    Called under the chunk claim, so no other worker can be using
    this chunk's uploads: every id found predates this run and is
    superseded — a finished chunk needs none of them, and an
    unfinished chunk re-uploads under a fresh key. A failed delete
    records through the shared lifecycle, so the next sweep retries
    it; an unattached claim has nothing to delete.
    """
    base = f"{parent_id}:chunk:{index}"
    first_free = _chunk_first_free_attempt(uploads, parent_id, index)
    for attempt in range(first_free):
        key = base if attempt == 0 else f"{base}:retry:{attempt}"
        found = uploads.load_chunk_claim(key)
        if found is None:
            continue
        _, meta_file_id = found
        if meta_file_id is None:
            continue
        _delete_chunk_file(meta_file_id, client=client, lifecycle=lifecycle)


def _drain_pending_upload(
    pending_ids: list[MetaFileId],
    file_id: MetaFileId,
    *,
    client: MetaFilesClient,
    lifecycle: LifecycleRepository,
) -> None:
    """Delete one finished attempt's upload now; drop it from the sweep.

    The attempt's file served its purpose — its paid raws are already
    persisted — so it goes as soon as the attempt ends instead of at
    the run's sweep: a crash between attempts then strands no id the
    resume drain cannot already name. Best-effort: a failed delete
    records through the lifecycle for the next sweep.
    """
    _delete_chunk_file(file_id, client=client, lifecycle=lifecycle)
    if file_id in pending_ids:
        pending_ids.remove(file_id)


def _parent_cancelled(uploads: UploadStore, parent_id: str) -> bool:
    """True when the parent upload row shows a requested cancellation.

    A missing row (direct stage runs with no upload) is not a
    cancellation. An unreadable store fails closed: the caller stops
    before any further provider call rather than risk Meta/Spark work
    a transient read failure hid a cancellation from.
    """
    try:
        loaded = uploads.load_job(JobId(parent_id))
    except Exception:
        return True
    if loaded is None:
        return False
    _, record = loaded
    return (
        record.cancel_requested_at is not None
        or record.state in ("cancelling", "cancelled")
    )


def _record_chunk_failure(
    uploads: UploadStore,
    sub_id: JobId,
    failure_key: str,
    error: str,
    *,
    meta_file_id: MetaFileId | None = None,
    meta_expires_at: datetime | None = None,
) -> None:
    """Queue one chunk upload failure; the persisted claim stands on error."""
    try:
        uploads.record_failure(
            sub_id, failure_key, "meta_upload", error, db.utcnow(),
            meta_file_id=meta_file_id, meta_expires_at=meta_expires_at,
        )
    except Exception as write_error:
        logger.warning(
            "chunk job=%s key=%s failure write failed: %s",
            sub_id, failure_key, type(write_error).__name__,
        )


def _upload_chunk_file(
    uploads: UploadStore,
    sub_ctx: JobContext,
    client: MetaFilesClient,
    chunk_path: Path,
    *,
    artifact_id: ArtifactId,
    idempotency_key: str,
    duration_s: float,
) -> MetaFileId:
    """Claim, upload, and attach one chunk file; queue any known id.

    The claim is persisted before the upload call, so a crash between
    upload and attach still leaves the key listed for sweeper recovery,
    exactly like a crashed preview confirmation. A failed upload keeps
    the claim as the lookup handle; a known id that cannot attach is
    queued pending and deleted best-effort. Every failure raises
    :class:`MetaFilesError`, reported upstream as an unusable upload.
    """
    sub_id = sub_ctx.job_id
    try:
        uploads.claim_chunk_upload(
            sub_id, idempotency_key, artifact_id, duration_s=duration_s,
            size_bytes=chunk_path.stat().st_size, now=db.utcnow(),
        )
    except Exception as error:
        raise MetaFilesError("could not upload video chunks") from error
    recorder = RecordingMetaClient(client, sub_ctx)
    request_time_utc = db.utcnow()
    try:
        uploaded = recorder.upload_file(
            chunk_path,
            expires_after_hours=META_UPLOAD_REQUEST_HOURS,
            idempotency_key=idempotency_key,
        )
    except MetaFilesError as error:
        _record_chunk_failure(
            uploads, sub_id, f"chunk-upload:{idempotency_key}",
            "meta upload failed",
        )
        raise MetaFilesError("could not upload video chunks") from error
    if not isinstance(uploaded, MetaUpload):
        _record_chunk_failure(
            uploads, sub_id, f"chunk-upload:{idempotency_key}",
            "meta upload returned an unusable result",
        )
        raise MetaFilesError("could not upload video chunks")
    now = db.utcnow()
    try:
        completed = uploads.complete_confirmation(
            idempotency_key, uploaded.file_id, uploaded.expires_at,
            request_time_utc, now,
        )
    except Exception as error:
        _record_chunk_failure(
            uploads, sub_id,
            f"chunk-persist:{idempotency_key}:{uploaded.file_id}",
            "chunk confirmation persist failed after meta upload",
            meta_file_id=uploaded.file_id,
            meta_expires_at=uploaded.expires_at
            if isinstance(uploaded.expires_at, datetime) else None,
        )
        try:
            client.delete_file(uploaded.file_id)
        except Exception as delete_error:
            logger.warning(
                "chunk job=%s compensating delete of %s failed: %s",
                sub_id, uploaded.file_id, type(delete_error).__name__,
            )
        raise MetaFilesError("could not upload video chunks") from error
    if completed.state != "confirmed" or completed.meta_file_id is None:
        try:
            client.delete_file(uploaded.file_id)
        except Exception as delete_error:
            logger.warning(
                "chunk job=%s cancelling delete of %s failed: %s",
                sub_id, uploaded.file_id, type(delete_error).__name__,
            )
        raise MetaFilesError("could not upload video chunks")
    return completed.meta_file_id


def _delete_chunk_file(
    file_id: MetaFileId,
    *,
    client: MetaFilesClient,
    lifecycle: LifecycleRepository,
) -> None:
    """Delete one chunk file through the shared Meta deletion lifecycle.

    Success records the verified outcome; a failed delete records
    ``retry``, due on the next sweep instead of at provider expiry.
    """
    attempt_started_at = db.utcnow()
    try:
        request_meta_deletion(file_id, client=client, lifecycle=lifecycle)
    except Exception as delete_error:
        logger.warning(
            "chunk file %s delete failed: %s",
            file_id, type(delete_error).__name__,
        )
        try:
            lifecycle.record_meta_file_deletion(
                file_id, "retry",
                error=f"{type(delete_error).__name__}: {delete_error}"[:200],
                attempt_started_at=attempt_started_at,
            )
        except Exception as record_error:
            logger.warning(
                "chunk file %s retry record failed: %s",
                file_id, type(record_error).__name__,
            )


def _extract_chunked(
    ctx: JobContext,
    windows: Sequence[ChunkWindow],
    local_video: Path,
    *,
    clients: PipelineClients,
    lifecycle: LifecycleRepository,
    uploads: UploadStore,
    replay: bool,
    settings: Settings | None,
) -> StitchedSteps:
    """Extract one bundle per window and stitch them on the parent timeline.

    Each chunk is cut locally, registered as a provisional artifact (so
    the finish gate sweeps the bytes on every outcome), clocked in
    place when enabled, measured for scene cuts (stored per window),
    claimed under its own idempotency key before upload, attached
    through the same confirmation the preview flow uses, and extracted
    under its own internal active sub-job: every chunk reserves and
    settles its own window-sized envelope, and a refused chunk fails
    the parent job closed. Each chunk Meta id is deleted through the
    shared Meta deletion lifecycle once extracted, so a failed delete
    is due on the next sweep. A failed child settles through the
    bounded terminal-transition recovery, like the parent. A chunk
    that is not a teardown refuses the whole job, exactly like a
    single-chunk video. A parent cancellation noticed before a child's
    upload or extraction settles that child without any further
    provider call and stops the run.

    A finished chunk's bundle is persisted atomically under a
    parent-scoped digest, so a re-entry after interruption reuses it
    with zero cut, upload, or Spark calls for that chunk; a corrupt
    or expired bundle fails closed and only that chunk reruns. One
    chunk's exclusive extraction claim spans its bundle lookup
    through its bundle save, and the bundle is rechecked after the
    claim and before any paid call, so two workers never pay Spark
    for the same chunk. One chunk's StepsDisagreement is retried as
    a new internal child attempt with its own ledger cap and
    idempotency key, bounded by the configured per-chunk attempts
    against the durable next attempt number, and preflighted against
    the per-video retry budget and the job cap from durable ledger
    spend; a retry that would cross either is refused up front with
    an honest message, and finished chunks are never rerun. A failed
    child whose settlement stays pending stops the retry and reports
    pending settlement to the parent. Each finished attempt's upload
    drains as soon as the attempt ends, and a re-entry drains earlier
    runs' uploads for every chunk it touches.

    In the default ``contiguous`` span mode each chunk consensus is
    refined first (snap to the window's cuts, fold short phases) and
    must then tile its window (a missing phase retries once, then
    fails closed), and the stitched guide tiles the video with ends
    derived from the next start; ``legacy`` mode skips both and keeps
    model spans.
    """
    parent_id = str(ctx.job_id)
    directory = job_dir(parent_id)
    timeout_s = _chunk_cut_timeout_s(settings)
    try:
        source_hash = hash_file(local_video)
    except OSError as error:
        raise ChunkCutError(
            f"could not read source video {local_video}"
        ) from error
    # One digest config for every chunk save and resume lookup: the
    # chunk clients carry this same config (replay delegates to the
    # inner client; live builds from the same settings), and any
    # divergence fails safe to a resume miss, never a wrong hit.
    # Pricing never comes from these shares: every window prices
    # through the shared window estimate — the nominal span until its
    # chunk is probed, then the decoded duration — so the running
    # planned total always prices probed bytes honestly.
    digest_config = spark_config_of(clients.spark)
    nominal_usd = [
        _window_estimate_usd(window.end_s - window.start_s, settings)
        for window in windows
    ]
    decoded_usd: list[Decimal | None] = [None] * len(windows)

    def _planned_total_usd() -> Decimal:
        return sum(
            (
                decoded if decoded is not None else nominal
                for decoded, nominal in zip(decoded_usd, nominal_usd)
            ),
            Decimal("0"),
        )
    cap_usd = _job_cost_cap_usd(settings)
    retry_budget_usd = _chunk_retry_budget_usd(settings)
    max_attempts = _chunk_max_attempts(settings)
    bundle_ttl_s = _chunk_bundle_ttl_s(settings)
    span_mode = _step_span_mode(settings)
    tolerance_s = _contiguity_tolerance_s(settings)
    # Extraction tune, fixed for the run: the clocked upload, the cut
    # pass and prompt listing, and the snap/fold/merge refine pass.
    clock_burn_in = _clock_burn_in(settings)
    cut_threshold = _scene_cut_threshold(settings)
    cuts_in_prompt = _scene_cuts_in_prompt(settings)
    snap_window_s = _snap_window_s(settings)
    min_phase_length_s = _min_phase_length_s(settings)
    merge_same_action = _merge_same_action(settings)
    # Durable attempt numbering: an earlier run's claimed attempts
    # stay claimed, so a resumed run reissues no key and admits no
    # attempt past the per-chunk maximum. Retry spend is derived from
    # the ledger at each preflight, never counted from these numbers.
    first_attempts = [
        _chunk_first_free_attempt(uploads, parent_id, index)
        for index in range(len(windows))
    ]
    chunk_bundles: list[tuple[ChunkWindow, StepBundle]] = []
    chunk_file_ids: list[MetaFileId] = []

    def _settle_child_failed(
        sub_ctx: JobContext, index: int
    ) -> Literal["failed", "pending_settlement"]:
        try:
            return _settle_failed(
                sub_ctx, f"chunk {index} failed", settings=settings
            )
        except Exception as settle_error:
            logger.warning(
                "chunk sub-job job=%s settlement failed: %s",
                sub_ctx.job_id, type(settle_error).__name__,
            )
            # Unknown outcome: the child may still be active with
            # holds, so the caller treats it as pending, never as
            # settled, and starts no retry on top of it.
            return "pending_settlement"

    try:
        for index, window in enumerate(windows):
            if _parent_cancelled(uploads, parent_id):
                # Before any registration or admission for this chunk:
                # earlier children already settled, so just stop.
                raise _ParentCancelled(
                    f"parent job {parent_id} was cancelled"
                )
            try:
                with _chunk_extraction_claim(parent_id, index):
                    # Earlier runs' uploads for this chunk are
                    # superseded — a rerun re-uploads under a fresh
                    # key — so drain them before any new side effect.
                    _drain_prior_chunk_uploads(
                        uploads, parent_id, index,
                        client=clients.meta, lifecycle=lifecycle,
                    )
                    chunk_path = directory / chunk_filename(index)
                    # Register before cutting: a crash between cut and
                    # register would orphan footage no sweep knows,
                    # while a registered row with no bytes yet
                    # tombstones cleanly.
                    artifact_id = ctx.artifacts.register(
                        CHUNK_ARTIFACT_KIND, chunk_path
                    )
                    # The cut is local and free, and the resume digest
                    # binds its decoded facts — so every run cuts and
                    # probes before the lookup, and a resumed chunk
                    # still skips the upload and Spark calls.
                    cut_chunk(
                        local_video, window, chunk_path,
                        timeout_s=timeout_s,
                    )
                    # Decoded, not nominal: a stream-copy cut can play a
                    # different duration than the window asked for (a
                    # frame straddling the cut end, container rounding),
                    # and can open on a keyframe before the planned
                    # start. The prompt, the extent, upload metadata,
                    # validation, and digests all take the probed facts;
                    # stitching maps back with the cut's real start.
                    # The probe runs on the stream-copy cut, before the
                    # clock burn: the burn keeps every frame at the same
                    # rate, so it preserves the decoded duration, and a
                    # resumed chunk skips the burn entirely.
                    cut_info = probe_cut(chunk_path, timeout_s=timeout_s)
                    # Priced from the bytes, not the plan: a stream-copy
                    # cut can decode longer than requested, and every
                    # later budget check prices this chunk decoded.
                    decoded_usd[index] = _window_estimate_usd(
                        cut_info.duration_s, settings
                    )
                    real_window = real_cut_window(window, cut_info)
                    digest = chunk_bundle_digest(
                        parent_id, source_hash, index,
                        window.start_s, window.end_s, digest_config,
                        chunk_tolerance_s=tolerance_s,
                        cut_start_s=real_window.start_s,
                        cut_duration_s=cut_info.duration_s,
                        scene_cut_threshold=cut_threshold,
                        scene_cuts_in_prompt=cuts_in_prompt,
                        snap_window_s=snap_window_s,
                        min_phase_length_s=min_phase_length_s,
                        clock_burn_in=clock_burn_in,
                        merge_same_action=merge_same_action,
                    )
                    # Rechecked under the claim and before any paid
                    # call: a bundle the previous holder saved is
                    # reused, never re-paid — but only when the
                    # record's cut still matches the bytes just
                    # probed, stitched at the fresh real window.
                    resumed = _load_resumed_bundle(
                        digest, digest_config, bundle_ttl_s
                    )
                    if resumed is not None and _resumed_cut_matches(
                        digest, real_window, bundle_ttl_s
                    ):
                        logger.info(
                            "pipeline job=%s reusing finished chunk %d",
                            parent_id, index,
                        )
                        chunk_bundles.append((real_window, resumed))
                        continue
                    # Decoded preflight: the nominal preflight admitted
                    # this run before any cut, but the bytes just probed
                    # can cost more than the window asked for. Refuse
                    # before any admission, upload, or hold when the
                    # running decoded total crosses the job cap.
                    planned_usd = _planned_total_usd()
                    if planned_usd > cap_usd:
                        message = (
                            "step extraction failed: chunk"
                            f" {index} decoded to"
                            f" {cut_info.duration_s} s and the"
                            f" ${planned_usd} planned total would exceed"
                            f" the ${cap_usd} job budget"
                        )
                        logger.info(
                            "pipeline job=%s refused chunk %d"
                            " on decoded cost: %s",
                            parent_id, index, message,
                        )
                        raise _ChunkBudgetRefused(message) from None
                    extent = ChunkExtent(
                        duration_s=cut_info.duration_s,
                        tolerance_s=tolerance_s,
                    )
                    # The uploaded bytes carry the overlaid clock when
                    # enabled (burned in place, so the artifact row and
                    # upload path stay put); the cut pass then indexes
                    # those same bytes and stores the window's cuts.
                    # Stitching maps the chunk back with the cut's real
                    # start, so the stored cuts file does too.
                    clocked = False
                    if clock_burn_in:
                        burn_clock_in_place(chunk_path, timeout_s=timeout_s)
                        clocked = True
                    cuts = detect_scene_cuts(
                        chunk_path, cut_threshold, timeout_s=timeout_s
                    )
                    write_window_cuts(
                        directory, index, real_window.start_s,
                        cut_threshold, cuts,
                    )
                    attempt = first_attempts[index]
                    made_this_run = 0
                    while True:
                        if attempt >= max_attempts:
                            raise _ChunkAttemptsExhausted(
                                f"chunk {index} already used all"
                                f" {max_attempts} attempts"
                            )
                        if attempt == 0:
                            label = f"{parent_id}/chunk/{index}"
                            idempotency_key = f"{parent_id}:chunk:{index}"
                        else:
                            retry_spend_usd = _chunk_retry_spend_usd(
                                uploads, parent_id, len(windows)
                            )
                            retry_spend_usd += _extract_retry_spend_usd(parent_id)
                            window_usd = _window_estimate_usd(
                                cut_info.duration_s, settings
                            )
                            if not chunk_retry_fits(
                                retry_spend_usd, window_usd,
                                _planned_total_usd(),
                                cap_usd, retry_budget_usd,
                            ):
                                # In-run retries follow a disagreement
                                # this run saw; a resumed chunk's
                                # earlier attempt may instead have
                                # crashed, so the message says only
                                # what this run knows.
                                trigger = (
                                    "disagreed" if made_this_run > 0
                                    else "needs another attempt"
                                )
                                if (
                                    retry_spend_usd + window_usd
                                    > retry_budget_usd
                                ):
                                    message = (
                                        "step extraction failed: chunk"
                                        f" {index} {trigger} and the"
                                        f" ${retry_budget_usd} per-video"
                                        " retry budget is spent"
                                    )
                                else:
                                    message = (
                                        "step extraction failed: chunk"
                                        f" {index} {trigger} and a"
                                        f" ${window_usd} retry would"
                                        f" exceed the ${cap_usd} job budget"
                                    )
                                logger.info(
                                    "pipeline job=%s refused chunk %d"
                                    " retry: %s",
                                    parent_id, index, message,
                                )
                                raise _ChunkRetryRefused(message) from None
                            label = (
                                f"{parent_id}/chunk/{index}/retry/{attempt}"
                            )
                            idempotency_key = (
                                f"{parent_id}:chunk:{index}:retry:{attempt}"
                            )
                            # A retry claims under its own artifact row:
                            # one uploads row owns each artifact, so the
                            # initial attempt's row cannot back a second
                            # claim. The finish gate tombstones every row
                            # and file removal is idempotent, so the
                            # shared bytes sweep cleanly.
                            artifact_id = ctx.artifacts.register(
                                CHUNK_ARTIFACT_KIND, chunk_path
                            )
                        # The child's Spark cap is the window cap for the
                        # decoded duration — the same call the factory
                        # installs on the window's clients below, so the
                        # holds always fit the enforcing cap.
                        sub_ctx = admit_job(
                            label, label, hash_file(chunk_path), internal=True,
                            spark_cap_usd=_window_spark_cap_usd(
                                cut_info.duration_s, settings
                            ),
                            **_output_names(settings),
                        )
                        try:
                            if not replay:
                                assert settings is not None
                                extra_s = _chunk_deadline_extra_s(settings)
                                with contextlib.closing(db.connect(settings.db_path)) as conn:
                                    before = conn.execute(
                                        "SELECT created_at, deadline_at FROM jobs WHERE id = ?",
                                        (str(sub_ctx.job_id),),
                                    ).fetchone()
                                verified = False
                                for attempt in range(settings.deadline_extension_max_attempts):
                                    try:
                                        extended = extend_job_deadline(
                                            sub_ctx.job_id, extra_s, path=settings.db_path,
                                            max_extra_s=_deadline_extension_max_s(settings),
                                        )
                                    except ContentionError:
                                        logger.warning("chunk %d deadline extension lost a race (%d/%d)",
                                                       index, attempt + 1, settings.deadline_extension_max_attempts)
                                        continue
                                    with contextlib.closing(db.connect(settings.db_path)) as conn:
                                        after = conn.execute(
                                            "SELECT state, deadline_at FROM jobs WHERE id = ?",
                                            (str(sub_ctx.job_id),),
                                        ).fetchone()
                                    if before is not None and after is not None:
                                        required = db.parse_dt(before["deadline_at"] if extended else before["created_at"])
                                        required += timedelta(seconds=extra_s + (0 if extended else JOB_DEADLINE_S))
                                        verified = (after["state"] == "active"
                                                    and db.parse_dt(after["deadline_at"]) >= required
                                                    and after["deadline_at"] > db.format_dt(db.utcnow()))
                                    break
                                if not verified:
                                    raise RuntimeError("child deadline extension was not verified")
                        except BaseException as error:
                            _write_refusal_marker(sub_ctx, sub_ctx.job_id, "service_unavailable")
                            status = _settle_child_failed(sub_ctx, index)
                            if status == "pending_settlement" and isinstance(error, Exception):
                                raise _ChunkSettlementPending(
                                    f"step extraction failed: chunk {index} settlement is pending"
                                ) from error
                            raise
                        if _parent_cancelled(uploads, parent_id):
                            # Cancellation raced the cut: settle this
                            # child without any upload or provider call
                            # and stop the parent. A settlement that
                            # stays pending propagates, so the parent
                            # reports it instead of clean failed while
                            # the child still holds its reservations.
                            if (
                                _settle_child_failed(sub_ctx, index)
                                == "pending_settlement"
                            ):
                                raise _ChunkSettlementPending(
                                    f"step extraction failed: chunk {index}"
                                    " settlement is pending; cancellation"
                                    " left it unsettled"
                                ) from None
                            raise _ParentCancelled(
                                f"parent job {parent_id} was cancelled"
                            )
                        try:
                            if replay:
                                chunk_spark = RecordingSparkClient(
                                    clients.spark, sub_ctx
                                )
                            else:
                                assert settings is not None  # checked before any cut
                                # Sized from the probed decoded duration,
                                # not the nominal window: a stream-copy cut
                                # can decode longer than requested, and a
                                # nominal-sized client would under-reserve
                                # for the bytes actually sent. Built before
                                # the upload so an unfundable window refuses
                                # before any bytes move.
                                chunk_spark = select_providers(
                                    settings, sub_ctx,
                                    window_s=cut_info.duration_s,
                                ).spark
                            chunk_file_id = _upload_chunk_file(
                                uploads, sub_ctx, clients.meta, chunk_path,
                                artifact_id=artifact_id,
                                idempotency_key=idempotency_key,
                                duration_s=cut_info.duration_s,
                            )
                            chunk_file_ids.append(chunk_file_id)
                            if _parent_cancelled(uploads, parent_id):
                                # Cancellation raced the upload: settle
                                # this child without any Spark call. The
                                # id is already queued above, so the
                                # sweep below still deletes the file. A
                                # settlement that stays pending
                                # propagates, like the pre-upload path.
                                if (
                                    _settle_child_failed(sub_ctx, index)
                                    == "pending_settlement"
                                ):
                                    raise _ChunkSettlementPending(
                                        f"step extraction failed: chunk"
                                        f" {index} settlement is pending;"
                                        " cancellation left it unsettled"
                                    ) from None
                                raise _ParentCancelled(
                                    f"parent job {parent_id} was cancelled"
                                )
                            # An empty cut list prompts identically to an
                            # unlisted one, so it digests identically too.
                            prompt_cuts = (
                                cuts if cuts_in_prompt and cuts else None
                            )
                            bundle = extract_steps_cached(
                                sub_ctx,
                                str(chunk_file_id),
                                client=chunk_spark,  # type: ignore[arg-type]
                                chunk_extent=extent,
                                scene_cuts=prompt_cuts,
                                clocked=clocked,
                                retry_call=_window_retry_call(
                                    ctx, sub_ctx, chunk_spark, uploads, settings,
                                    window_index=index, window_s=cut_info.duration_s,
                                    planned_usd=_planned_total_usd,
                                    chunk_count=len(windows), replay=replay,
                                ),
                                retryable_failures=(
                                    settings.extract_retry_failures if settings else frozenset()
                                ),
                            ).bundle
                            if span_mode == "contiguous":
                                # Snap to the window's cuts, fold short
                                # phases, merge same-action neighbours,
                                # then enforce the tile: a missing phase
                                # retries once like any disagreement, then
                                # fails closed. Legacy mode skips the refine
                                # pass and keeps model spans exactly.
                                bundle = refine_consensus_bundle(
                                    bundle,
                                    cuts=cuts,
                                    snap_window_s=snap_window_s,
                                    min_phase_length_s=min_phase_length_s,
                                    generic_tokens=(
                                        _step_alignment_generic_tokens(settings)
                                    ),
                                    verb_synonyms=(
                                        _step_alignment_verb_synonyms(settings)
                                    ),
                                    negation_words=(
                                        _step_alignment_negation_words(settings)
                                    ),
                                    qualifiers=_step_alignment_qualifiers(settings),
                                    part_synonyms=(
                                        _step_alignment_part_synonyms(settings)
                                    ),
                                    transition_verbs=(
                                        _step_alignment_transition_verbs(settings)
                                    ),
                                    merge_same_action=merge_same_action,
                                )
                                require_contiguous_cover(
                                    bundle.steps,
                                    cut_info.duration_s,
                                    tolerance_s=tolerance_s,
                                )
                            save_chunk_bundle(
                                digest, bundle,
                                cut_start_s=real_window.start_s,
                                cut_duration_s=cut_info.duration_s,
                            )
                            chunk_bundles.append((real_window, bundle))
                            sub_ctx.finish("done")
                            _drain_pending_upload(
                                chunk_file_ids, chunk_file_id,
                                client=clients.meta, lifecycle=lifecycle,
                            )
                            break
                        except (StepsDisagreement, ContiguityError):
                            if (
                                _settle_child_failed(sub_ctx, index)
                                == "pending_settlement"
                            ):
                                raise _ChunkSettlementPending(
                                    f"step extraction failed: chunk {index}"
                                    " settlement is pending;"
                                    " retry not started"
                                ) from None
                            _drain_pending_upload(
                                chunk_file_ids, chunk_file_id,
                                client=clients.meta, lifecycle=lifecycle,
                            )
                            made_this_run += 1
                            attempt += 1
                            logger.info(
                                "pipeline job=%s retrying chunk %d"
                                " (attempt %d)",
                                parent_id, index, attempt,
                            )
                        except BaseException as error:
                            if isinstance(error, _ChunkSettlementPending):
                                # The post-upload cancel path already
                                # settled (or found it pending): re-raise
                                # without settling twice or rewriting
                                # the message.
                                raise
                            status = _settle_child_failed(sub_ctx, index)
                            if (
                                status == "pending_settlement"
                                and isinstance(error, Exception)
                            ):
                                raise _ChunkSettlementPending(
                                    f"step extraction failed: chunk {index}"
                                    " settlement is pending"
                                ) from error
                            raise
            except _ChunkClaimHeld:
                # Another worker owns this chunk: only the claim above
                # raises this. No recheck here — the resume digest
                # needs this run's own probed cut, and cutting beside
                # the holder would race its bytes — so step aside; the
                # caller retries after the holder finishes.
                raise _ChunkClaimHeld(
                    f"chunk {index} is being extracted by another"
                    " worker; retry later"
                ) from None
    finally:
        for file_id in chunk_file_ids:
            _delete_chunk_file(
                file_id, client=clients.meta, lifecycle=lifecycle
            )
    stitched = stitch_bundles(
        chunk_bundles,
        overlap_s=_chunk_overlap_s(settings),
        alignment_mode=_step_alignment_mode(settings),
        generic_tokens=_step_alignment_generic_tokens(settings),
        verb_synonyms=_step_alignment_verb_synonyms(settings),
        negation_words=_step_alignment_negation_words(settings),
        qualifiers=_step_alignment_qualifiers(settings),
        part_synonyms=_step_alignment_part_synonyms(settings),
        transition_verbs=_step_alignment_transition_verbs(settings),
        span_mode=_step_span_mode(settings),
        verb_less_overlap_fraction=(
            _step_alignment_verb_less_overlap_fraction(settings)
        ),
    )
    if span_mode != "contiguous":
        return stitched
    # Global contiguity: coverage enforced before ends derive, so a
    # missing phase fails the job closed instead of stretching a
    # neighbour (the caller's generic handler reports it as failed).
    # Edge outros drop only after deriving, so the gates still see
    # the full cover and only whole-video edges can go.
    duration_s = windows[len(windows) - 1].end_s - windows[0].start_s
    derived = derive_contiguous_ends(
        stitched.steps, duration_s, tolerance_s=tolerance_s
    )
    kept, excluded = _trim_edge_outro(derived, settings)
    return StitchedSteps(
        steps=kept,
        device_guess=stitched.device_guess,
        confidence=stitched.confidence,
        excluded_spans=excluded,
    )


def _takes_chunk_path(
    windows: Sequence[ChunkWindow], settings: Settings | None
) -> bool:
    """True when the windows extract through the chunk path.

    Only a single window within the chunk length takes the direct
    path; anything else — several windows, or one longer
    whole-video window — goes through the chunk path.
    """
    return not (
        len(windows) == 1
        and (windows[0].end_s - windows[0].start_s) <= _chunk_s(settings)
    )


def _extract_bundle(
    ctx: JobContext,
    meta_file_id: str,
    local_video: Path,
    windows: Sequence[ChunkWindow],
    *,
    spark: SparkClient,
    clients: PipelineClients,
    lifecycle: LifecycleRepository,
    uploads: UploadStore,
    replay: bool,
    settings: Settings | None,
) -> StepBundle | StitchedSteps:
    """One extraction bundle for short videos, stitched steps past one chunk.

    Only a window within the chunk length takes the direct path (the
    confirmed upload, no clock, no cuts, byte-identical to before); a
    longer single window — whole-video mode — goes through the chunk
    path with one chunk, so it gets the clocked upload, the cut pass,
    and the refine pass like every other chunk.
    """
    if not _takes_chunk_path(windows, settings):
        window = windows[0]
        bundle = extract_steps_cached(
            ctx, meta_file_id, client=spark,  # type: ignore[arg-type]
            chunk_extent=ChunkExtent(
                duration_s=window.end_s - window.start_s,
                tolerance_s=_contiguity_tolerance_s(settings),
            ),
            retry_call=_window_retry_call(
                ctx, ctx, spark, uploads, settings,
                window_index=0, window_s=window.end_s - window.start_s,
                planned_usd=lambda: _window_estimate_usd(window.end_s - window.start_s, settings),
                chunk_count=0, replay=replay,
            ),
            retryable_failures=(settings.extract_retry_failures if settings else frozenset()),
        ).bundle
        if _step_span_mode(settings) != "contiguous":
            return bundle
        if (
            _merge_same_action(settings)
            and _step_alignment_mode(settings) == DEFAULT_ALIGNMENT_MODE
        ):
            # Short videos skip the chunk refine pass (no cuts to
            # snap to), but same-action neighbours still fuse here,
            # before ends derive; legacy and exact modes keep every
            # step exactly as before.
            bundle = merge_same_action_phases(bundle)
        derived = derive_contiguous_ends(
            bundle.steps,
            window.end_s - window.start_s,
            tolerance_s=_contiguity_tolerance_s(settings),
        )
        kept, excluded = _trim_edge_outro(derived, settings)
        return StepBundle(
            device_guess=bundle.device_guess,
            confidence=bundle.confidence,
            steps=kept,
            source_runs=bundle.source_runs,
            timestamps_approximate=bundle.timestamps_approximate,
            excluded_spans=excluded,
        )
    return _extract_chunked(
        ctx, windows, local_video,
        clients=clients, lifecycle=lifecycle, uploads=uploads,
        replay=replay, settings=settings,
    )


def _run_stages_inner(
    ctx: JobContext,
    meta_file_id: MetaFileId | str,
    local_video: Path,
    *,
    clients: PipelineClients,
    lifecycle: LifecycleRepository,
    uploads: UploadStore,
    replay: bool,
    settings: Settings | None,
    duration_s: float,
    retain_clip: bool = True,
) -> PipelineResult:
    job_id = ctx.job_id
    # Live mode leaves the recording wrapper off: the live Spark
    # client persists its own extract receipt per run.
    spark = RecordingSparkClient(clients.spark, ctx) if replay else clients.spark
    cap = _max_video_s(settings)
    if duration_s > cap:
        logger.info(
            "pipeline job=%s refused: %.3f s over the %.3f s duration cap",
            job_id, duration_s, cap,
        )
        return _refuse(
            ctx, job_id, over_limit_message(cap),
            replay=replay, settings=settings, refusal_kind="duration",
        )
    windows = plan_extraction_windows(duration_s, settings)
    if not replay and settings is None and _takes_chunk_path(windows, settings):
        _settle_failed(
            ctx, "chunked extraction in live mode needs settings",
            settings=settings,
        )
        raise ValueError(
            "chunked extraction in live mode needs settings to bind chunk clients"
        )
    # Up-front budget: the job's total is the per-chunk share times the
    # chunk count, refused before any cut, upload, or provider call when
    # it exceeds the configured job cap.
    refused, chunk_count, required, cap = chunk_budget_refused(
        duration_s, spark, settings
    )
    if refused:
        logger.info(
            "pipeline job=%s refused: %d chunk(s) need $%s over the $%s job cap",
            job_id, chunk_count, required, cap,
        )
        return _refuse(
            ctx, job_id, STEP_EXTRACTION_OVER_BUDGET,
            replay=replay, settings=settings, refusal_kind="budget",
        )
    _write_stage(ctx, RECEIPT_STAGE_EXTRACT)
    try:
        bundle = _extract_bundle(
            ctx, str(meta_file_id), local_video, windows,
            spark=spark,  # type: ignore[arg-type]
            clients=clients, lifecycle=lifecycle, uploads=uploads,
            replay=replay, settings=settings,
        )
    except NotATeardown as refusal:
        logger.info("pipeline job=%s refused: %s", job_id, refusal.refusal.reason)
        return _refuse(ctx, job_id, "not a teardown", replay=replay,
                       settings=settings, refusal_kind="not_a_teardown")
    except BudgetExhausted:
        logger.info("pipeline job=%s extraction over the budget limit", job_id)
        return _fail(
            ctx, job_id, STEP_EXTRACTION_OVER_BUDGET,
            replay=replay, settings=settings, refusal_kind="budget",
        )
    except ChunkCutError:
        logger.info("pipeline job=%s chunk cut failed", job_id)
        return _fail(
            ctx, job_id, "could not cut video chunks",
            replay=replay, settings=settings,
        )
    except ClockBurnError:
        logger.info("pipeline job=%s clock burn failed", job_id)
        return _fail(
            ctx, job_id, "could not burn the extraction clock",
            replay=replay, settings=settings,
        )
    except SceneCutError:
        logger.info("pipeline job=%s scene-cut pass failed", job_id)
        return _fail(
            ctx, job_id, "could not detect scene cuts",
            replay=replay, settings=settings,
        )
    except MetaFilesError:
        logger.info("pipeline job=%s chunk upload failed", job_id)
        return _fail(
            ctx, job_id, "could not upload video chunks",
            replay=replay, settings=settings,
        )
    except _ChunkRetryRefused as refused:
        logger.info("pipeline job=%s %s", job_id, refused)
        return _fail(ctx, job_id, str(refused), replay=replay,
                     settings=settings, refusal_kind="budget")
    except _ChunkBudgetRefused as refused:
        logger.info("pipeline job=%s %s", job_id, refused)
        return _fail(ctx, job_id, str(refused), replay=replay,
                     settings=settings, refusal_kind="budget")
    except _ParentCancelled:
        logger.info("pipeline job=%s stopped: parent cancelled", job_id)
        return _fail(
            ctx, job_id, "job cancelled",
            replay=replay, settings=settings,
        )
    except SparkServiceError:
        return _fail(
            ctx, job_id, "step extraction failed", replay=replay,
            settings=settings, refusal_kind="extract_service_failure",
        )
    except _ChunkAttemptsExhausted as exhausted:
        logger.info("pipeline job=%s %s", job_id, exhausted)
        return _fail(ctx, job_id, "step extraction failed", replay=replay,
                     settings=settings)
    except _ChunkSettlementPending as pending:
        logger.info("pipeline job=%s %s", job_id, pending)
        return _fail_unsettled(
            ctx, job_id, str(pending), replay=replay, settings=settings
        )
    except _ChunkClaimHeld as held:
        # Contention is not failure: the holder is still extracting, so
        # settling the shared parent here would fail a job another
        # worker is about to complete. This run paid nothing and settles
        # nothing; the caller retries after the holder finishes.
        logger.info("pipeline job=%s %s", job_id, held)
        return PipelineResult(
            state="contention", job_id=job_id, guide=None,
            error=str(held), replay=replay,
        )
    except Exception as error:
        logger.info(
            "pipeline job=%s extraction failed: %s", job_id, type(error).__name__
        )
        return _fail(ctx, job_id, "step extraction failed", replay=replay,
                     settings=settings)
    except BaseException:
        # KeyboardInterrupt/SystemExit from extraction: settle the parent
        # (failed, or a pending marker when the transition cannot
        # confirm) so no active job with held reservations survives,
        # then re-raise for the caller.
        _write_refusal_marker(ctx, job_id, "extraction")
        _settle_failed(ctx, "step extraction failed", settings=settings)
        raise
    assessment = assess_hazards(tuple(bundle.steps), bundle.device_guess)
    if assessment.refused:
        logger.info(
            "pipeline job=%s refused: %s", job_id, assessment.refusal_kind
        )
        return _refuse(ctx, job_id, "not a teardown", replay=replay,
                       settings=settings, refusal_kind=assessment.refusal_kind)
    _write_stage(ctx, RECEIPT_STAGE_SEGMENT)
    try:
        frames = score_frames(local_video, tuple(bundle.steps))
    except Exception as error:
        logger.info(
            "pipeline job=%s frame scoring failed: %s", job_id, type(error).__name__
        )
        return _fail(ctx, job_id, "frame scoring failed", replay=replay,
                     settings=settings)
    slots: list[LabelSlot] = []
    provisionals: list[ProvisionalRender] = []
    sam_exhausted = False
    try:
        for step, part_index, part in _checked_slots(bundle):
            step_frames = tuple(frames.get(step.number, ()))
            if not step_frames:
                continue
            candidates: list[tuple[Detection, Path]] = []
            for frame_path in step_frames:
                try:
                    detections = segment(
                        ctx, frame_path, part.phrase, client=clients.sam
                    )
                except (BudgetExhausted, SamCallLimitExceeded):
                    # A stitched guide can hold more slots than the SAM
                    # budget fits: keep the finished slots and leave the
                    # rest unchecked, as read-back already does on caps.
                    sam_exhausted = True
                    break
                for detection in detections:
                    candidates.append((detection, frame_path))
            if sam_exhausted:
                break
            chosen = choose_detection(
                [detection for detection, _ in candidates], part.position_hint
            )
            if chosen is None:
                continue
            origin = next(
                frame for detection, frame in candidates if detection is chosen
            )
            slot = LabelSlot(
                step_index=step.number - 1, part_index=part_index, part=part
            )
            output = job_dir(str(job_id)) / _label_image_name(
                settings, step=step.number, part=part_index
            )
            provisionals.append(
                render_label(ctx, origin, chosen, output, number=part_index + 1)
            )
            slots.append(slot)
    except Exception as error:
        if isinstance(error, OSError) and error.errno is not None:
            roots = (Path(os.path.abspath(db.data_root())), Path(__file__).resolve().parents[1])
            paths = []
            for name in (error.filename, error.filename2):
                if isinstance(name, (str, bytes)):
                    candidate = Path(os.path.abspath(os.fsdecode(name)))
                    if any(candidate.is_relative_to(root) for root in roots):
                        paths.append(str(candidate))
            logger.info(
                "pipeline job=%s part check filesystem error: %s paths=%s",
                job_id, os.strerror(error.errno), paths,
            )
        logger.info(
            "pipeline job=%s part check failed: %s", job_id, type(error).__name__
        )
        return _fail(ctx, job_id, "part check failed", replay=replay,
                     settings=settings,
                     refusal_kind=(
                         "deadline" if isinstance(error, SamDeadlineExceeded) else None
                     ))
    if sam_exhausted:
        logger.info(
            "pipeline job=%s SAM budget exhausted: %d slot(s) kept, rest unchecked",
            job_id, len(slots),
        )
    _write_stage(ctx, RECEIPT_STAGE_READBACK)
    try:
        decisions = run_readbacks(
            ctx, provisionals, slots=slots, client=spark,  # type: ignore[arg-type]
            # Live mode leaves the receipt hook off: the live
            # read-back client persists its own receipt per call.
            record_call=readback_record_call if replay else None,
            record_event=readback_record_event,
        )
    except Exception as error:
        logger.info(
            "pipeline job=%s read-back failed: %s", job_id, type(error).__name__
        )
        return _fail(ctx, job_id, "read-back failed", replay=replay,
                     settings=settings)
    hazards = guide_hazard_assessment(tuple(bundle.steps), assessment)
    step_frames_ordered = [
        tuple(frames.get(step.number, ()))[:1] for step in bundle.steps
    ]
    ordered_frames = [paths[0] for paths in step_frames_ordered if paths]
    deferred = _DeferredFinish(ctx)
    try:
        guide = assemble_guide(
            deferred,  # type: ignore[arg-type]
            tuple(bundle.steps), ordered_frames, decisions, hazards=hazards,
            excluded_spans=tuple(bundle.excluded_spans),
        )
    except Exception as error:
        logger.info(
            "pipeline job=%s assembly failed: %s", job_id, type(error).__name__
        )
        return _fail(ctx, job_id, "guide assembly failed", replay=replay,
                     settings=settings)
    try:
        record_event(ctx, "job.completed", {})
    except Exception as error:
        logger.warning(
            "completion event job=%s failed: %s", job_id, type(error).__name__
    )
    active = False
    try:
        with active_job_publication(job_id) as active:
            if active:
                save_guide(guide, guide_filename=_guide_filename(settings))
    except OSError as error:
        # The success transition is still stashed, never committed: fail
        # the job so the failed finish tombstones every provisional output
        # and no guide file is left behind (the store is atomic).
        logger.warning(
            "guide store job=%s failed: %s", job_id, type(error).__name__
        )
        return _abandon_publication(
            ctx, job_id, "could not store guide", replay=replay, settings=settings,
        )
    if not active:
        return _abandon_publication(
            ctx, job_id, "job ended before guide publication",
            replay=replay, settings=settings,
        )
    if retain_clip:
        try:
            guide_clip.retain_guide_clip(
                ctx, uploads, filename=_guide_clip_filename(settings)
            )
        except guide_clip.GuideClipError as error:
            # The publish promised a playable guide: without its clip the
            # guide is half-published, so remove it and fail closed.
            logger.warning("guide clip job=%s failed: %s", job_id, type(error).__name__)
            return _abandon_publication(
                ctx, job_id, "could not keep guide clip",
                replay=replay, settings=settings,
            )
    if _commit_done(deferred, settings=settings) == "pending_settlement":
        if _job_row_state(job_id) in ("failed", "expired"):
            return _abandon_publication(
                ctx, job_id, "job ended before guide publication",
                replay=replay, settings=settings,
            )
        return PipelineResult(
            state="pending_settlement",
            job_id=job_id,
            guide=None,
            error="guide stored but the success transition is unconfirmed"
                  " (terminal settlement pending)",
            replay=replay,
        )
    schedule = deletion_schedule(job_id, lifecycle=lifecycle)
    logger.info(
        "pipeline job=%s done: %d artifact(s), %d meta file(s) due for the sweeper",
        job_id,
        len(schedule.artifact_ids),
        len(schedule.meta_file_ids),
    )
    persistence.request_snapshot()
    if not replay and settings is not None and settings.providers.mode == "live":
        # Background second check; never delays or alters the finished guide. Its spend
        # caps hold per job across this batch and every later recheck (the sidecar keeps them).
        open_readback.start(ctx, decisions, budget=_second_check_budget(settings))
    return PipelineResult(
        state="done", job_id=job_id, guide=guide, error=None, replay=replay
    )


__all__ = (
    "GUIDE_FILENAME",
    "META_UPLOAD_ENDPOINT",
    "META_UPLOAD_MODEL",
    "META_UPLOAD_STAGE",
    "READBACK_COUNTED_EVENT",
    "READBACK_RECEIPT_ENDPOINT",
    "SAM_SEGMENT_ENDPOINT",
    "SAM_SEGMENT_STAGE",
    "SPARK_EXTRACT_ENDPOINT",
    "STEP_EXTRACTION_OVER_BUDGET",
    "DeletionSchedule",
    "PipelineClients",
    "PipelineResult",
    "PipelineState",
    "RecordingMetaClient",
    "RecordingSparkClient",
    "ReplayExhausted",
    "ReplayMetaFilesClient",
    "ReplaySamClient",
    "ReplaySparkClient",
    "default_replay_clients",
    "deletion_schedule",
    "fail_job",
    "guide_from_json",
    "guide_hazard_assessment",
    "guide_path",
    "guide_to_json",
    "load_guide",
    "pending_settlement_path",
    "read_pending_settlement",
    "readback_record_call",
    "readback_record_event",
    "resolve_run_clients",
    "run_confirmed_job",
    "run_job",
    "run_stages",
    "sam_receipt_sink",
    "save_guide",
    "translate_readback_record",
    "translate_sam_record",
)
