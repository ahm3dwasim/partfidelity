"""Frozen cross-module contracts; matches the frozen declarations exactly.

Shapes are immutable dataclasses. Clients are injected protocols: production
adapters read META_API_KEY from the process environment (see app.settings),
tests inject fakes from tests/conftest.py. No raw bytes, secrets, consent
messages, IPs, or personal names enter these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import (
    BinaryIO,
    Literal,
    Mapping,
    NewType,
    Protocol,
    Sequence,
    runtime_checkable,
)

type JSONValue = bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None

JobId = NewType("JobId", str)
ArtifactId = NewType("ArtifactId", str)
MetaFileId = NewType("MetaFileId", str)
DeleteRequestId = NewType("DeleteRequestId", str)


@dataclass(frozen=True)
class PartRequest:
    phrase: str
    position_hint: str


@dataclass(frozen=True)
class LabelSlot:
    step_index: int
    part_index: int
    part: PartRequest


@dataclass(frozen=True)
class Step:
    number: int
    start_s: float
    end_s: float
    action: str
    tools: tuple[str, ...]
    parts: tuple[PartRequest, ...]  # at most 3
    hazards: tuple[str, ...]
    time_spread_s: float
    time_needs_check: bool  # spread > 5 s, or kept overlapping rivals
    # Phase kind from app.spark.PHASE_KINDS (the consensus run majority;
    # "repair" for pre-kind payloads and wording-grouped fallbacks).
    kind: str = "repair"

    @property
    def is_intro(self) -> bool:
        """The unnumbered intro block above step 1 (number 0): context, not a repair step."""
        return self.kind == "intro" and self.number == 0


@dataclass(frozen=True)
class BoundaryInfo:
    """One internal consensus boundary: support, cut evidence, snap record.

    ``time_s`` is the boundary's final position (post-snap). ``support_runs``
    of ``total_runs`` runs placed an edge here (time consensus only).
    ``nearest_cut_s``/``cut_distance_s`` name the closest scene cut, or None
    when no cut pass ran or no cut exists. ``snapped`` is True exactly when
    the boundary moved onto a cut, by the signed ``snap_delta_s`` seconds.
    ``needs_check`` flags a boundary for review (minority support, or no
    cut nearby); the adjacent steps carry the same flag.
    """

    time_s: float
    support_runs: int
    total_runs: int
    nearest_cut_s: float | None
    cut_distance_s: float | None
    snapped: bool
    snap_delta_s: float
    needs_check: bool


@dataclass(frozen=True)
class ExcludedSpan:
    """A consensus span left out of the guide (non-repair edge phases).

    ``kind`` is the consensus kind that excluded it (always
    ``"outro_or_unrelated"``: only edge phases with a run majority for
    that kind are dropped, never a middle phase and never the intro).
    Kept steps stay contiguous around it; the UI can report it as a
    skipped range ("not part of the repair").
    """

    start_s: float
    end_s: float
    kind: str


@dataclass(frozen=True)
class StepBundle:
    device_guess: str
    confidence: float
    steps: tuple[Step, ...]  # 1..12
    source_runs: int  # exactly 3 for live extraction
    timestamps_approximate: bool  # always true for Spark output
    # One record per internal boundary (time-consensus bundles only;
    # wording-grouped bundles carry none).
    boundaries: tuple[BoundaryInfo, ...] = ()
    # Spans dropped from the guide after consensus (edge outros only).
    excluded_spans: tuple[ExcludedSpan, ...] = ()


@dataclass(frozen=True)
class Detection:
    phrase: str
    mask_path: Path
    score: float  # SAM `c`
    box_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class ProvisionalRender:
    detection: Detection
    artifact_id: ArtifactId
    rendered_image_path: Path
    position_hint: str


@dataclass(frozen=True)
class ArtifactRecord:
    id: ArtifactId
    job_id: JobId
    kind: str
    relative_path: Path
    expires_at: datetime


@runtime_checkable
class ArtifactStore(Protocol):
    """Job-scoped artifact adapter. The ledger module owns the implementation; tests inject fakes."""

    def register(self, kind: str, path: Path) -> ArtifactId: ...
    def mark_kept(self, artifact_id: ArtifactId) -> None: ...
    def mark_deleted(self, artifact_id: ArtifactId) -> None: ...


@runtime_checkable
class CallReservation(Protocol):
    """Per-call cost hold. The ledger module owns the implementation; settled after each paid call."""

    def settle(
        self, actual_usd: Decimal, outcome: Literal["succeeded", "failed"]
    ) -> None: ...


@runtime_checkable
class JobContext(Protocol):
    """Job-scoped lane context. The ledger module owns the implementation; tests inject fakes."""

    @property
    def job_id(self) -> JobId: ...
    @property
    def artifacts(self) -> ArtifactStore: ...
    def reserve(self, kind: str, worst_case_usd: Decimal) -> CallReservation: ...
    def record_call_with_event(
        self, key: str, call: CallRecord, event: tuple[str, Mapping[str, JSONValue]]
    ) -> None: ...
    def finish(
        self,
        outcome: Literal["done", "failed"],
        *,
        keep_artifact_ids: Sequence[ArtifactId] = (),
        failed_output_names: tuple[str, str] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class MetaFileRecord:
    id: MetaFileId
    job_id: JobId
    expires_at: datetime
    delete_state: str


@runtime_checkable
class LifecycleRepository(Protocol):
    """Terminal-job deletion queue. The ledger module owns the implementation; tests inject fakes."""

    def list_due_artifacts(self, now: datetime) -> tuple[ArtifactRecord, ...]: ...
    def list_due_meta_files(self, now: datetime) -> tuple[MetaFileRecord, ...]: ...
    def record_artifact_deletion(
        self,
        artifact_id: ArtifactId,
        outcome: Literal["tombstoned", "retry"],
        error: str | None = None,
    ) -> None: ...
    def record_meta_file_deletion(
        self,
        meta_file_id: MetaFileId,
        outcome: Literal["verified_absent", "acknowledged_only", "retry"],
        error: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class LabelDecision:
    slot: LabelSlot
    detection: Detection | None
    provisional: ProvisionalRender | None
    checked: bool
    status: Literal["part_pixels_unchanged", "not_checked"]
    readback: Literal["yes", "no", "not_run"]


@dataclass(frozen=True)
class GuideHazardAssessment:
    difficulty: Literal["Easy", "Medium", "Hard"] = "Hard"
    verdict: str = "DRAFT"
    draft_reasons: tuple[str, ...] = ("Hazard assessment unavailable.",)


@dataclass(frozen=True)
class Guide:
    job_id: JobId
    steps: tuple[Step, ...]
    frame_paths: tuple[Path, ...]
    labels: tuple[LabelDecision, ...]
    difficulty: Literal["Easy", "Medium", "Hard"]
    verdict: str
    draft_reasons: tuple[str, ...]
    # Spans trimmed from the guide after consensus (edge outros only).
    excluded_spans: tuple[ExcludedSpan, ...] = ()


@dataclass(frozen=True)
class CallRecord:
    job_id: JobId
    stage: str
    model: str
    endpoint: str
    latency_ms: int
    units_in: int | None
    units_out: int | None
    estimated_cost_usd: Decimal
    status: str
    evidence: Mapping[str, JSONValue]


@dataclass(frozen=True)
class UploadPreview:
    job_id: JobId
    local_video: Path
    duration_s: float
    size_bytes: int
    audio_removed: bool
    confirmed_at: datetime | None


@dataclass(frozen=True)
class UploadRecord:
    preview: UploadPreview
    preview_artifact_id: ArtifactId
    preview_expires_at: datetime
    state: Literal["preview", "confirming", "confirmed", "cancelling", "cancelled", "failed"]
    consented_at: datetime | None
    claim_id: str | None
    claim_started_at: datetime | None
    guard_deadline_utc: datetime | None
    meta_file_id: MetaFileId | None
    meta_expires_at: datetime | None
    confirmed_at: datetime | None
    cancel_requested_at: datetime | None
    hidden: bool
    failure_count: int
    last_error: str | None


@dataclass(frozen=True)
class ConfirmationClaim:
    record: UploadRecord
    acquired: bool


@dataclass(frozen=True)
class DeleteRequestRecord:
    id: DeleteRequestId
    job_id: JobId | None
    contact: str
    requested_at: datetime
    expires_at: datetime


@runtime_checkable
class UploadStore(Protocol):
    """Durable upload lifecycle. The ledger module owns the implementation; tests inject fakes."""

    def register_preview(
        self,
        ctx: JobContext,
        preview: UploadPreview,
        artifact_id: ArtifactId,
        expires_at: datetime,
    ) -> UploadRecord: ...
    def load_job(self, job_id: JobId) -> tuple[JobContext, UploadRecord] | None: ...
    def claim_confirmation(
        self, job_id: JobId, consented_at: datetime
    ) -> ConfirmationClaim: ...
    def claim_chunk_upload(
        self,
        job_id: JobId,
        claim_id: str,
        artifact_id: ArtifactId,
        duration_s: float,
        size_bytes: int,
        now: datetime,
    ) -> UploadRecord: ...
    def chunk_claim_taken(self, claim_id: str) -> bool: ...
    def load_chunk_claim(
        self, claim_id: str
    ) -> tuple[JobId, MetaFileId | None] | None: ...
    def complete_confirmation(
        self,
        claim_id: str,
        meta_file_id: MetaFileId,
        expires_at: datetime,
        request_time_utc: datetime,
        now: datetime,
    ) -> UploadRecord: ...
    def request_cancellation(self, job_id: JobId, now: datetime) -> UploadRecord: ...
    def record_failure(
        self,
        job_id: JobId,
        failure_key: str,
        stage: Literal["meta_upload", "meta_persist", "local_delete", "meta_delete", "recovery"],
        error: str,
        now: datetime,
        *,
        meta_file_id: MetaFileId | None = None,
        meta_expires_at: datetime | None = None,
    ) -> UploadRecord: ...
    def list_recoverable(self, now: datetime) -> tuple[UploadRecord, ...]: ...
    def recover_job(
        self, job_id: JobId, now: datetime, *, no_match_claim_id: str | None = None
    ) -> UploadRecord: ...
    def public_uploads_blocked(self) -> bool: ...
    def submit_delete_request(
        self, job_id: JobId | None, contact: str, now: datetime
    ) -> DeleteRequestId: ...
    def list_delete_requests(self, limit: int) -> tuple[DeleteRequestRecord, ...]: ...
    def resolve_delete_request(self, request_id: DeleteRequestId) -> None: ...
    def purge_delete_requests(self, now: datetime) -> int: ...


@dataclass(frozen=True)
class GlimmerProof:
    model_id: str
    licence: str
    quantisation: str
    peak_ram_bytes: int
    latency_ms: int
    platform: Literal["mac", "vm_cpu"]
    graded_image: bool


@runtime_checkable
class SparkClient(Protocol):
    """Injected Spark adapter. The readback module owns the method surface; tests inject fakes."""

    ...


@runtime_checkable
class SamClient(Protocol):
    """Injected hosted-SAM adapter. The SAM module owns the method surface; tests inject fakes."""

    ...


@dataclass(frozen=True)
class MetaUpload:
    file_id: MetaFileId
    expires_at: datetime


class MetaFilesError(Exception): ...
class MetaReadUnsupported(MetaFilesError): ...


class MetaUploadExpiryError(MetaFilesError):
    file_id: MetaFileId
    expires_at: datetime | None
    def __init__(self, file_id: MetaFileId, expires_at: datetime | None) -> None: ...


@runtime_checkable
class MetaFilesClient(Protocol):
    """Injected Meta files adapter. The upload module owns the method surface; tests inject fakes."""

    def upload_file(
        self, path: Path, *, expires_after_hours: int, idempotency_key: str
    ) -> MetaUpload: ...
    def find_upload(self, idempotency_key: str) -> MetaUpload | None: ...
    def delete_file(self, file_id: MetaFileId) -> None: ...
    def get_file(self, file_id: MetaFileId) -> MetaUpload | None: ...


# Module-facing function signatures (function boundaries). Declarations
# only: each owning module implements its functions; the settings module never implements them.
def read_back_label(
    ctx: JobContext, provisional: ProvisionalRender, *, slot: LabelSlot, client: SparkClient
) -> LabelDecision: ...
def assemble_guide(
    ctx: JobContext,
    steps: Sequence[Step],
    frames: Sequence[Path],
    labels: Sequence[LabelDecision],
    *,
    hazards: GuideHazardAssessment | None = None,
    excluded_spans: Sequence[ExcludedSpan] = (),
) -> Guide: ...
def guide_hazard_assessment(
    steps: Sequence[Step], assessment: HazardAssessment
) -> GuideHazardAssessment: ...
def prepare_preview(
    ctx: JobContext, upload: BinaryIO, declared_name: str, *, uploads: UploadStore
) -> UploadPreview: ...
def confirm_and_send(
    job_id: JobId, *, uploads: UploadStore, client: MetaFilesClient
) -> MetaFileId: ...
def cancel_and_delete(
    job_id: JobId,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    client: MetaFilesClient,
) -> DeletionResult: ...
def sweep_expired(
    now: datetime,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    client: MetaFilesClient,
) -> SweepResult: ...


# Fixed configuration supplied to modules (facts fixed after the timing evaluation).
SPARK_RUNS_PER_GUIDE = 3
TIME_SPREAD_FLAG_S = 5
MAX_STEPS = 12
MAX_PARTS_PER_STEP = 3
MAX_READBACKS_PER_JOB = 3
JOB_DEADLINE_S = 240
SAM_MAX_REQUEST_STARTS_PER_MINUTE = 55
PART_CHECK_THRESHOLDS: Mapping[str, float] = MappingProxyType({"fan": 0.6, "battery": 0.5})
# Step-timing scoring law (fixed evaluation law, not a tunable knob:
# the rule keeps the bar at 7 of 9 within ±3 s while the
# scorer aligns by time overlap). Read only through
# app.scorer_time's loader, which fails loudly on a missing key; never restated
# elsewhere. strict_2_s/strict_1_s are report-only lines, never the bar.
TIMING_SCORE_LAW: Mapping[str, float] = MappingProxyType({
    "tolerance_s": 3.0,
    "strict_2_s": 2.0,
    "strict_1_s": 1.0,
    "pass_count": 7,
    "pass_of": 9,
})
D1B_SCORE_LAW = TIMING_SCORE_LAW  # former name, kept for existing imports
JOB_COST_CAP_USD = Decimal("0.50")
AGGREGATE_SPEND_CEILING_USD = Decimal("100")
AGGREGATE_SPEND_MIN_UNSPENT_USD = Decimal("50")
# Artifact kind for the clip kept with a published guide. One home for the string: the
# pipeline retains it, the sweeper exempts it, the guide route serves it,
# and cancellation deletes it.
GUIDE_CLIP_ARTIFACT_KIND = "guide_clip"
# Durable-upload and delete-request bounds (fixed configuration).
UPLOAD_PREVIEW_TTL_HOURS = 24
META_UPLOAD_MAX_HOURS = 24
META_EXPIRY_GUARD_MARGIN_HOURS = 1
CONFIRMATION_CLAIM_TIMEOUT_S = 120
UPLOAD_FAILURE_BLOCK_THRESHOLD = 3
UPLOAD_ERROR_MAX_CHARS = 512
DELETE_REQUEST_MAX_PENDING = 1000
DELETE_REQUEST_RETENTION_DAYS = 30
DELETE_REQUEST_CONTACT_MAX_CHARS = 254
DELETE_REQUEST_PAGE_SIZE = 100
# Per-job reservation: three 1x Spark runs ($0.26) + at most 32 SAM calls at
# $0.0025 ($0.08) + at most three read-backs at $0.04 ($0.12) = $0.46, leaving
# $0.04 inside JOB_COST_CAP_USD. The 32-call SAM bound no longer covers one
# call per slot on a 12-step guide (12 steps x 2 frames x 2 checked parts
# would be 48): the earliest slots win and the rest stay unchecked, as the
# pipeline already does when the budget runs out.
SPARK_3RUNS_RESERVATION_USD = Decimal("0.26")
SAM_MAX_CALLS_PER_JOB = 32
SAM_COST_PER_CALL_USD = Decimal("0.0025")
SAM_MAX_RESERVATION_USD = Decimal("0.08")
READBACK_COST_PER_CALL_USD = Decimal("0.04")
READBACK_MAX_RESERVATION_USD = Decimal("0.12")
JOB_RESERVATION_TOTAL_USD = Decimal("0.46")
JOB_RESERVATION_HEADROOM_USD = Decimal("0.04")
# Frozen with the other fixed configuration (cutoff 0.70, measured on the reference clips). The readback module reads it as config and never moves it
# to make a row fire.
READBACK_CONFIDENCE_CUTOFF = 0.70
C1_READBACK_CUTOFF = READBACK_CONFIDENCE_CUTOFF  # former name, kept for existing imports
# Receipt vocabulary (single home). These are the
# stage and model strings a persisted model-call receipt carries. The evidence module's gate
# derives its allowlists from them, and every receipt emitter (the
# pipeline translation seam, the live-client record builders, the settings
# record-names maps) imports them: no other module restates these strings.
# Module-side values live apart on purpose — sam.py's "sam" stage and
# "segment_stills" endpoint, readback.py's "read_back" endpoint, and the
# yes/no/not_run decisions are provider/lane namespaces the seam
# translates, and the API model ids sent on the wire (settings,
# spark.SPARK_MODEL_ID, sam.SAM_MODEL) may differ from the receipt names
# exactly as "muse-spark-1.3" already differs from "spark-1.3".
RECEIPT_STAGE_EXTRACT = "extract"
RECEIPT_STAGE_LOCATE = "locate"
RECEIPT_STAGE_SEGMENT = "segment"
RECEIPT_STAGE_READBACK = "readback"
RECEIPT_STAGE_UPLOAD = "upload"
RECEIPT_STAGES = frozenset(
    {
        RECEIPT_STAGE_EXTRACT,
        RECEIPT_STAGE_LOCATE,
        RECEIPT_STAGE_SEGMENT,
        RECEIPT_STAGE_READBACK,
        RECEIPT_STAGE_UPLOAD,
    }
)
# Live clients record Spark calls (extract and read-back) under
# RECEIPT_MODEL_SPARK via the settings record-names map; the replay seam
# copies the steps API id verbatim (RECEIPT_MODEL_MUSE_SPARK). Both stay
# allowlisted: old rows and replay rows carry the latter.
RECEIPT_MODEL_SPARK = "spark-1.3"
RECEIPT_MODEL_MUSE_SPARK = "muse-spark-1.3"
RECEIPT_MODEL_SAM = "sam-3.1"
RECEIPT_MODEL_META_FILES = "meta-files"
# The background second check on Meta's open-weights Spark Open (Baseten).
RECEIPT_MODEL_SPARK_OPEN = "spark-open"
RECEIPT_MODELS = frozenset(
    {
        RECEIPT_MODEL_SPARK,
        RECEIPT_MODEL_MUSE_SPARK,
        RECEIPT_MODEL_SAM,
        RECEIPT_MODEL_META_FILES,
        RECEIPT_MODEL_SPARK_OPEN,
    }
)

__all__ = (
    "JobId",
    "ArtifactId",
    "MetaFileId",
    "DeleteRequestId",
    "PartRequest",
    "LabelSlot",
    "Step",
    "StepBundle",
    "BoundaryInfo",
    "Detection",
    "ProvisionalRender",
    "ArtifactRecord",
    "ArtifactStore",
    "CallReservation",
    "JobContext",
    "MetaFileRecord",
    "LifecycleRepository",
    "LabelDecision",
    "GuideHazardAssessment",
    "Guide",
    "CallRecord",
    "UploadPreview",
    "UploadRecord",
    "ConfirmationClaim",
    "DeleteRequestRecord",
    "UploadStore",
    "GlimmerProof",
    "SparkClient",
    "SamClient",
    "MetaUpload",
    "MetaFilesError",
    "MetaReadUnsupported",
    "MetaUploadExpiryError",
    "MetaFilesClient",
    "read_back_label",
    "assemble_guide",
    "guide_hazard_assessment",
    "prepare_preview",
    "confirm_and_send",
    "cancel_and_delete",
    "sweep_expired",
    "JSONValue",
    "SPARK_RUNS_PER_GUIDE",
    "TIME_SPREAD_FLAG_S",
    "MAX_STEPS",
    "MAX_PARTS_PER_STEP",
    "MAX_READBACKS_PER_JOB",
    "JOB_DEADLINE_S",
    "SAM_MAX_REQUEST_STARTS_PER_MINUTE",
    "PART_CHECK_THRESHOLDS",
    "JOB_COST_CAP_USD",
    "AGGREGATE_SPEND_CEILING_USD",
    "AGGREGATE_SPEND_MIN_UNSPENT_USD",
    "UPLOAD_PREVIEW_TTL_HOURS",
    "META_UPLOAD_MAX_HOURS",
    "META_EXPIRY_GUARD_MARGIN_HOURS",
    "CONFIRMATION_CLAIM_TIMEOUT_S",
    "UPLOAD_FAILURE_BLOCK_THRESHOLD",
    "UPLOAD_ERROR_MAX_CHARS",
    "DELETE_REQUEST_MAX_PENDING",
    "DELETE_REQUEST_RETENTION_DAYS",
    "DELETE_REQUEST_CONTACT_MAX_CHARS",
    "DELETE_REQUEST_PAGE_SIZE",
    "SPARK_3RUNS_RESERVATION_USD",
    "SAM_MAX_CALLS_PER_JOB",
    "SAM_COST_PER_CALL_USD",
    "SAM_MAX_RESERVATION_USD",
    "READBACK_COST_PER_CALL_USD",
    "READBACK_MAX_RESERVATION_USD",
    "JOB_RESERVATION_TOTAL_USD",
    "JOB_RESERVATION_HEADROOM_USD",
    "READBACK_CONFIDENCE_CUTOFF",
    "C1_READBACK_CUTOFF",
    "RECEIPT_STAGE_EXTRACT",
    "RECEIPT_STAGE_LOCATE",
    "RECEIPT_STAGE_SEGMENT",
    "RECEIPT_STAGE_READBACK",
    "RECEIPT_STAGE_UPLOAD",
    "RECEIPT_STAGES",
    "RECEIPT_MODEL_SPARK",
    "RECEIPT_MODEL_MUSE_SPARK",
    "RECEIPT_MODEL_SAM",
    "RECEIPT_MODEL_META_FILES",
    "RECEIPT_MODEL_SPARK_OPEN",
    "RECEIPT_MODELS",
)
