"""Settings adapter. META_API_KEY comes from the process environment only.

The production Meta files endpoint likewise comes from the environment
(``META_FILES_BASE_URL``) and must be an ``https`` URL; production
refuses to start without it, so a keyed client can never silently fall
back to the non-resolving placeholder.

Provider knobs (``ProviderSettings``) carry every live-client tunable:
mode, model ids, endpoints, phrases, timeouts and prices default to the
measured values; run count, spread, rate and thresholds mirror the
frozen contracts and the provider factory refuses to build live
clients when a mirror diverges, so an override can never silently do
nothing. Live mode refuses to start without ``META_API_KEY``.
"""

from __future__ import annotations

import math
import ntpath
import os
import re
import string
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

from app import step_alignment
from app.chunks import DEFAULT_SPAN_MODE, SPAN_MODES
from app.contracts import (
    JOB_DEADLINE_S,
    MAX_PARTS_PER_STEP,
    MAX_STEPS,
    UPLOAD_PREVIEW_TTL_HOURS,
    PART_CHECK_THRESHOLDS,
    RECEIPT_MODEL_SAM,
    RECEIPT_MODEL_SPARK,
    SAM_COST_PER_CALL_USD,
    SAM_MAX_REQUEST_STARTS_PER_MINUTE,
    SPARK_RUNS_PER_GUIDE,
    TIME_SPREAD_FLAG_S,
)

PRODUCTION_ENV_VAR = "PARTFIDELITY_PRODUCTION"
DATA_ROOT_ENV_VAR = "PARTFIDELITY_DATA_ROOT"
DB_PATH_ENV_VAR = "PARTFIDELITY_DB_PATH"
API_KEY_ENV_VAR = "META_API_KEY"
META_FILES_BASE_URL_ENV_VAR = "META_FILES_BASE_URL"
PROVIDER_MODE_ENV_VAR = "PARTFIDELITY_PROVIDER_MODE"
META_API_BASE_URL_ENV_VAR = "META_API_BASE_URL"
SPARK_MODEL_ENV_VAR = "META_SPARK_MODEL_ID"
SAM_MODEL_ENV_VAR = "META_SAM_MODEL_ID"
SPARK_RUNS_ENV_VAR = "PARTFIDELITY_SPARK_RUNS"
TIME_SPREAD_ENV_VAR = "PARTFIDELITY_TIME_SPREAD_FLAG_S"
SAM_RATE_ENV_VAR = "PARTFIDELITY_SAM_RATE_PER_MIN"
SAM_THRESHOLD_FAN_ENV_VAR = "PARTFIDELITY_SAM_THRESHOLD_FAN"
SAM_THRESHOLD_BATTERY_ENV_VAR = "PARTFIDELITY_SAM_THRESHOLD_BATTERY"
SAM_PHRASE_FAN_ENV_VAR = "PARTFIDELITY_SAM_PHRASE_FAN"
SAM_PHRASE_BATTERY_ENV_VAR = "PARTFIDELITY_SAM_PHRASE_BATTERY"
SPARK_TIMEOUT_ENV_VAR = "PARTFIDELITY_SPARK_TIMEOUT_S"
SAM_TIMEOUT_ENV_VAR = "PARTFIDELITY_SAM_TIMEOUT_S"
READBACK_TIMEOUT_ENV_VAR = "PARTFIDELITY_READBACK_TIMEOUT_S"
SPARK_INPUT_PRICE_ENV_VAR = "PARTFIDELITY_SPARK_INPUT_USD_PER_TOKEN"
SPARK_OUTPUT_PRICE_ENV_VAR = "PARTFIDELITY_SPARK_OUTPUT_USD_PER_TOKEN"
SAM_PRICE_ENV_VAR = "PARTFIDELITY_SAM_USD_PER_CALL"
STEP_ALIGNMENT_MODE_ENV_VAR = "PARTFIDELITY_STEP_ALIGNMENT_MODE"
STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR = "PARTFIDELITY_STEP_ALIGNMENT_GENERIC_TOKENS"
STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR = "PARTFIDELITY_STEP_ALIGNMENT_VERB_SYNONYMS"
STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR = "PARTFIDELITY_STEP_ALIGNMENT_NEGATION_WORDS"
STEP_ALIGNMENT_QUALIFIERS_ENV_VAR = "PARTFIDELITY_STEP_ALIGNMENT_QUALIFIERS"
STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR = "PARTFIDELITY_STEP_ALIGNMENT_PART_SYNONYMS"
STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR = "PARTFIDELITY_STEP_ALIGNMENT_TRANSITION_VERBS"
STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR = (
    "PARTFIDELITY_STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION"
)
CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR = (
    "PARTFIDELITY_CONSENSUS_BOUNDARY_TOLERANCE_S"
)

PROVIDER_MODES = ("replay", "live")

# Measured defaults (from the first measurement runs): the Model API base, the standard
# (never contributor) model ids, the winning SAM phrasings, per-call
# timeouts that ride out the observed ~78 s stall episode while still
# bounding every call, and the docs prices the spike settled with.
DEFAULT_META_API_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_SPARK_MODEL_ID = "muse-spark-1.3"
DEFAULT_SAM_MODEL_ID = "sam-3.1"
DEFAULT_SAM_PHRASE_FAN = "cooling fan"
DEFAULT_SAM_PHRASE_BATTERY = "rectangular battery with label"
DEFAULT_SPARK_TIMEOUT_S = 180.0
DEFAULT_SAM_TIMEOUT_S = 120.0
DEFAULT_READBACK_TIMEOUT_S = 120.0
DEFAULT_SPARK_INPUT_USD_PER_TOKEN = Decimal("0.00000125")
DEFAULT_SPARK_OUTPUT_USD_PER_TOKEN = Decimal("0.00000425")

# Configurable model ids, each mapped to the model name its receipts
# carry in the evidence module's closed evidence allowlist. An id outside these maps is
# refused at load: a receipt could not name it honestly. Adding a model
# is one line here (plus its receipt name in the contracts vocabulary).
# Default generic subject nouns for step alignment and timing-scorer matching; the
# canonical set lives in app.step_alignment next to the link predicate
# that reads it (as with the alignment mode default above).
DEFAULT_STEP_ALIGNMENT_GENERIC_TOKENS = step_alignment.GENERIC_TOKENS
# Default verb synonym groups; canonical in app.step_alignment likewise.
DEFAULT_STEP_ALIGNMENT_VERB_SYNONYMS = step_alignment.VERB_SYNONYM_GROUPS
# Default negation words, qualifiers, and part synonym groups; canonical
# in app.step_alignment likewise.
DEFAULT_STEP_ALIGNMENT_NEGATION_WORDS = step_alignment.NEGATION_WORDS
DEFAULT_STEP_ALIGNMENT_QUALIFIERS = step_alignment.QUALIFIERS
DEFAULT_STEP_ALIGNMENT_PART_SYNONYMS = step_alignment.PART_SYNONYM_GROUPS
# Default transition verbs and verb-less overlap fraction; canonical in
# app.step_alignment likewise.
DEFAULT_STEP_ALIGNMENT_TRANSITION_VERBS = step_alignment.TRANSITION_VERBS
DEFAULT_STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION = (
    step_alignment.DEFAULT_VERB_LESS_OVERLAP_FRACTION
)
SPARK_MODEL_RECORD_NAMES: Mapping[str, str] = MappingProxyType(
    {DEFAULT_SPARK_MODEL_ID: RECEIPT_MODEL_SPARK}
)
SAM_MODEL_RECORD_NAMES: Mapping[str, str] = MappingProxyType(
    {DEFAULT_SAM_MODEL_ID: RECEIPT_MODEL_SAM}
)


@dataclass(frozen=True)
class ProviderSettings:
    """Live-client tunables. Mirrors of frozen contracts are checked, not read."""

    mode: str = "replay"
    meta_api_base_url: str = DEFAULT_META_API_BASE_URL
    spark_model_id: str = DEFAULT_SPARK_MODEL_ID
    sam_model_id: str = DEFAULT_SAM_MODEL_ID
    spark_runs: int = SPARK_RUNS_PER_GUIDE
    time_spread_flag_s: float = float(TIME_SPREAD_FLAG_S)
    sam_max_starts_per_minute: int = SAM_MAX_REQUEST_STARTS_PER_MINUTE
    sam_threshold_fan: float = PART_CHECK_THRESHOLDS["fan"]
    sam_threshold_battery: float = PART_CHECK_THRESHOLDS["battery"]
    sam_phrase_fan: str = DEFAULT_SAM_PHRASE_FAN
    sam_phrase_battery: str = DEFAULT_SAM_PHRASE_BATTERY
    spark_timeout_s: float = DEFAULT_SPARK_TIMEOUT_S
    sam_timeout_s: float = DEFAULT_SAM_TIMEOUT_S
    readback_timeout_s: float = DEFAULT_READBACK_TIMEOUT_S
    spark_input_usd_per_token: Decimal = DEFAULT_SPARK_INPUT_USD_PER_TOKEN
    spark_output_usd_per_token: Decimal = DEFAULT_SPARK_OUTPUT_USD_PER_TOKEN
    sam_usd_per_call: Decimal = SAM_COST_PER_CALL_USD
    step_alignment_mode: str = step_alignment.DEFAULT_ALIGNMENT_MODE
    step_alignment_generic_tokens: frozenset[str] = (
        DEFAULT_STEP_ALIGNMENT_GENERIC_TOKENS
    )
    step_alignment_verb_synonyms: tuple[frozenset[str], ...] = (
        DEFAULT_STEP_ALIGNMENT_VERB_SYNONYMS
    )
    step_alignment_negation_words: frozenset[str] = (
        DEFAULT_STEP_ALIGNMENT_NEGATION_WORDS
    )
    step_alignment_qualifiers: frozenset[str] = (
        DEFAULT_STEP_ALIGNMENT_QUALIFIERS
    )
    step_alignment_part_synonyms: tuple[frozenset[str], ...] = (
        DEFAULT_STEP_ALIGNMENT_PART_SYNONYMS
    )
    step_alignment_transition_verbs: frozenset[str] = (
        DEFAULT_STEP_ALIGNMENT_TRANSITION_VERBS
    )
    step_alignment_verb_less_overlap_fraction: float = (
        DEFAULT_STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION
    )
    # Time-consensus boundary tolerance; None means the run's own flag
    # line (the existing alignment tolerance).
    consensus_boundary_tolerance_s: float | None = None


MATCH_IOU_ENV_VAR = "PARTFIDELITY_MATCH_IOU"
DEFAULT_MATCH_IOU = 0.95
MATCH_SSIM_ENV_VAR = "PARTFIDELITY_MATCH_SSIM"
DEFAULT_MATCH_SSIM = None
DEFAULT_MATCH_SSIM_PARSE = 0.0
MATCH_NOOP_ENV_VAR = "PARTFIDELITY_MATCH_NOOP"
DEFAULT_MATCH_NOOP = 0.001
MATCH_ECC_ITERATIONS_ENV_VAR = "PARTFIDELITY_MATCH_ECC_ITERATIONS"
DEFAULT_MATCH_ECC_ITERATIONS = 100
MATCH_ECC_EPSILON_ENV_VAR = "PARTFIDELITY_MATCH_ECC_EPSILON"
DEFAULT_MATCH_ECC_EPSILON = 1e-5
MATCH_SSIM_WINDOW_ENV_VAR = "PARTFIDELITY_MATCH_SSIM_WINDOW"
DEFAULT_MATCH_SSIM_WINDOW = 11
MATCH_SSIM_SIGMA_ENV_VAR = "PARTFIDELITY_MATCH_SSIM_SIGMA"
DEFAULT_MATCH_SSIM_SIGMA = 1.5
MATCH_SMALL_SAMPLE_ENV_VAR = "PARTFIDELITY_MATCH_SMALL_SAMPLE"
DEFAULT_MATCH_SMALL_SAMPLE = 10
MATCH_CORRUPTION_BLUR_ENV_VAR = "PARTFIDELITY_MATCH_CORRUPTION_BLUR"
DEFAULT_MATCH_CORRUPTION_BLUR = 11
MATCH_CORRUPTION_SHIFT_ENV_VAR = "PARTFIDELITY_MATCH_CORRUPTION_SHIFT"
DEFAULT_MATCH_CORRUPTION_SHIFT = 20
MATCH_CASES_FILENAME_ENV_VAR = "PARTFIDELITY_MATCH_CASES_FILENAME"
DEFAULT_MATCH_CASES_FILENAME = "cases.csv"


@dataclass(frozen=True)
class ImageMatchSettings:
    """Offline image matching; SSIM stays unavailable until calibrated."""

    iou_cutoff: float = DEFAULT_MATCH_IOU
    ssim_cutoff: float | None = DEFAULT_MATCH_SSIM
    noop_floor: float = DEFAULT_MATCH_NOOP
    ecc_iterations: int = DEFAULT_MATCH_ECC_ITERATIONS
    ecc_epsilon: float = DEFAULT_MATCH_ECC_EPSILON
    ssim_window: int = DEFAULT_MATCH_SSIM_WINDOW
    ssim_sigma: float = DEFAULT_MATCH_SSIM_SIGMA
    small_sample_count: int = DEFAULT_MATCH_SMALL_SAMPLE
    corruption_blur: int = DEFAULT_MATCH_CORRUPTION_BLUR
    corruption_shift: int = DEFAULT_MATCH_CORRUPTION_SHIFT
    cases_filename: str = DEFAULT_MATCH_CASES_FILENAME

    def __post_init__(self) -> None:
        for name in ("iou_cutoff", "noop_floor", "ssim_cutoff"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be in 0..1")
        if (self.ecc_iterations <= 0 or not math.isfinite(self.ecc_epsilon)
                or self.ecc_epsilon <= 0 or self.ssim_window < 3
                or self.ssim_window % 2 != 1 or not math.isfinite(self.ssim_sigma)
                or self.ssim_sigma <= 0 or self.small_sample_count <= 0
                or self.corruption_blur < 3 or self.corruption_blur % 2 != 1
                or self.corruption_shift <= 0):
            raise ValueError("invalid image-match settings")
        if not self.cases_filename or Path(self.cases_filename).name != self.cases_filename:
            raise ValueError("image-match cases filename must be a filename")


PIPELINE_MODE_ENV_VAR = "PARTFIDELITY_PIPELINE_MODE"
SETTLE_MAX_ATTEMPTS_ENV_VAR = "PARTFIDELITY_SETTLE_MAX_ATTEMPTS"
MAX_CONCURRENT_JOBS_ENV_VAR = "PARTFIDELITY_MAX_CONCURRENT_JOBS"
JOB_STATUS_REFRESH_S_ENV_VAR = "PARTFIDELITY_JOB_STATUS_REFRESH_S"
MAX_VIDEO_S_ENV_VAR = "PARTFIDELITY_MAX_VIDEO_S"
CHUNK_S_ENV_VAR = "PARTFIDELITY_CHUNK_S"
CHUNK_OVERLAP_S_ENV_VAR = "PARTFIDELITY_CHUNK_OVERLAP_S"
CHUNK_CUT_TIMEOUT_S_ENV_VAR = "PARTFIDELITY_CHUNK_CUT_TIMEOUT_S"
CHUNK_DEADLINE_ALLOWANCE_S_ENV_VAR = "PARTFIDELITY_CHUNK_DEADLINE_ALLOWANCE_S"
DEADLINE_EXTENSION_MAX_ATTEMPTS_ENV_VAR = "PARTFIDELITY_DEADLINE_EXTENSION_MAX_ATTEMPTS"
DEADLINE_EXTENSION_MAX_S_ENV_VAR = "PARTFIDELITY_DEADLINE_EXTENSION_MAX_S"
JOB_COST_CAP_ENV_VAR = "PARTFIDELITY_JOB_COST_CAP_USD"
CHUNK_MAX_ATTEMPTS_ENV_VAR = "PARTFIDELITY_CHUNK_MAX_ATTEMPTS"
CHUNK_RETRY_BUDGET_ENV_VAR = "PARTFIDELITY_CHUNK_RETRY_BUDGET_USD"
EXTRACT_RETRY_COUNT_ENV_VAR = "PARTFIDELITY_EXTRACT_RETRY_COUNT"
EXTRACT_RETRY_BACKOFF_ENV_VAR = "PARTFIDELITY_EXTRACT_RETRY_BACKOFF_S"
EXTRACT_RETRY_MIN_REMAINING_ENV_VAR = "PARTFIDELITY_EXTRACT_RETRY_MIN_REMAINING_S"
EXTRACT_RETRY_FAILURES_ENV_VAR = "PARTFIDELITY_EXTRACT_RETRY_FAILURES"
CHUNK_BUNDLE_TTL_ENV_VAR = "PARTFIDELITY_CHUNK_BUNDLE_TTL_S"
STEP_SPAN_MODE_ENV_VAR = "PARTFIDELITY_STEP_SPAN_MODE"
CONTIGUITY_TOLERANCE_ENV_VAR = "PARTFIDELITY_CONTIGUITY_TOLERANCE_S"
WHOLE_VIDEO_MAX_S_ENV_VAR = "PARTFIDELITY_WHOLE_VIDEO_MAX_S"
SPARK_INPUT_TOKENS_PER_VIDEO_S_ENV_VAR = (
    "PARTFIDELITY_SPARK_INPUT_TOKENS_PER_VIDEO_S"
)
CLOCK_BURN_IN_ENV_VAR = "PARTFIDELITY_CLOCK_BURN_IN"
SCENE_CUT_THRESHOLD_ENV_VAR = "PARTFIDELITY_SCENE_CUT_THRESHOLD"
SCENE_CUTS_IN_PROMPT_ENV_VAR = "PARTFIDELITY_SCENE_CUTS_IN_PROMPT"
SNAP_WINDOW_S_ENV_VAR = "PARTFIDELITY_SNAP_WINDOW_S"
MIN_PHASE_LENGTH_S_ENV_VAR = "PARTFIDELITY_MIN_PHASE_LENGTH_S"
SPARK_RESERVATION_ENV_VAR = "PARTFIDELITY_SPARK_RESERVATION_USD"
MERGE_SAME_ACTION_ENV_VAR = "PARTFIDELITY_MERGE_SAME_ACTION"
EXCLUDE_OUTRO_PHASES_ENV_VAR = "PARTFIDELITY_EXCLUDE_OUTRO_PHASES"

# Integration-owned pipeline config defaults. The orchestrator reads output
# naming, settlement bounds, and replay/live selection from here — never from
# literals at the call site. Every value is validated on construction.
DEFAULT_GUIDE_FILENAME = "guide.json"
DEFAULT_GUIDE_CLIP_FILENAME = "guide_clip.mp4"
DEFAULT_LABEL_IMAGE_PATTERN = "label_step{step}_part{part}.png"
DEFAULT_PENDING_SETTLEMENT_FILENAME = "pending_settlement.json"
DEFAULT_SAMPLE_POINTER_FILENAME = "sample_job.json"
# The real sample guide ships as a folder under app/ (guide, frames, labels, web clip).
DEFAULT_SAMPLE_BUNDLE_DIRNAME = "sample_real"
DEFAULT_SAMPLE_BUNDLE_CLIP_FILENAME = "sample.mp4"
# Bump when the shipped sample guide or its media change: an install with an older pointer is reinstalled at startup.
DEFAULT_SAMPLE_BUNDLE_VERSION = 5
DEFAULT_SAMPLE_BUNDLE_NOTES_FILENAME = "step_notes.json"
# Correction control + exports: one rename/delete label control,
# the numbered image zip, and the offline follower export.
DEFAULT_CORRECTIONS_FILENAME = "corrections.json"
DEFAULT_CORRECTION_PHRASE_MAX_CHARS = 60
DEFAULT_CORRECTION_MAX_OPS = 60
DEFAULT_IMAGE_ZIP_NAME = "partfidelity-guide-images.zip"
DEFAULT_IMAGE_ZIP_ENTRY_PATTERN = "step-{number:02d}.png"
DEFAULT_EXPORT_FRAME_PATTERN = "step-{number:02d}-original.png"
DEFAULT_EXPORT_LABEL_PATTERN = "step-{number:02d}-part{part}-labelled.png"
DEFAULT_EXPORT_HTML_NAME = "guide.html"
DEFAULT_EXPORT_ZIP_NAME = "partfidelity-guide-offline.zip"
DEFAULT_EXPORT_IMAGES_DIR = "images"
DEFAULT_SETTLE_MAX_ATTEMPTS = 3
# Confirmed uploads run off the request thread, at most this many at once.
DEFAULT_MAX_CONCURRENT_JOBS = 1
# The job-status page reloads itself this often while the job runs.
DEFAULT_JOB_STATUS_REFRESH_S = 5
DEFAULT_PIPELINE_MODE = "live"
# Long-footage chunking: videos up to 300 s are accepted and
# cut into 60 s chunks with a 2 s overlap; each chunk runs its own Spark
# extraction under its own ledger share, then stitches into one guide.
DEFAULT_MAX_VIDEO_S = 300.0
DEFAULT_CHUNK_S = 60.0
DEFAULT_CHUNK_OVERLAP_S = 2.0
DEFAULT_CHUNK_CUT_TIMEOUT_S = 300.0
# The measured 60 s guide completed in 133 s. A job envelope plus one
# SAM-call timeout per extraction window leaves time after extraction
# for the parent part check. The initial envelope covers preview and
# confirmation, which also consume the job clock.
DEFAULT_CHUNK_DEADLINE_ALLOWANCE_S = float(JOB_DEADLINE_S) + DEFAULT_SAM_TIMEOUT_S
# Six windows cover the configured five-minute cap with the overlap stride.
# The ledger still allows only one extension, bounded by this setting.
DEFAULT_DEADLINE_EXTENSION_MAX_S = (
    math.ceil(DEFAULT_MAX_VIDEO_S / (DEFAULT_CHUNK_S - DEFAULT_CHUNK_OVERLAP_S))
    * DEFAULT_CHUNK_DEADLINE_ALLOWANCE_S
)
DEFAULT_DEADLINE_EXTENSION_MAX_ATTEMPTS = 3
# Per-video planned-total bound, enforced up front by the pipeline: a job
# whose chunks need more than this is refused before any provider call.
# Grown from the $0.50 single-job envelope: 300 s needs 6 chunks and each
# chunk reserves the $0.26 extraction share, so the default admits
# 6 x $0.26 = $1.56. The per-job ledger caps ($0.50 job, $0.26 Spark,
# 3 Spark calls) are unchanged; every chunk enforces them separately.
DEFAULT_JOB_COST_CAP_USD = Decimal("1.60")
# Per-chunk evidence and retry: each chunk gets one
# initial attempt plus one bounded StepsDisagreement retry by default;
# retries across the video share one $0.26 share (one retry per video),
# and a finished chunk's persisted bundle is reusable for 7 days.
DEFAULT_CHUNK_MAX_ATTEMPTS = 2
DEFAULT_CHUNK_RETRY_BUDGET_USD = Decimal("0.26")
DEFAULT_EXTRACT_RETRY_COUNT = 1
DEFAULT_EXTRACT_RETRY_BACKOFF_S = 1.0
DEFAULT_EXTRACT_RETRY_MIN_REMAINING_S = 60.0
DEFAULT_EXTRACT_RETRY_FAILURES = frozenset({"timeout", "transport", "http_5xx", "empty"})
DEFAULT_CHUNK_BUNDLE_TTL_S = 7.0 * 24.0 * 3600.0
# Contiguous repair phases: guides tile [0, duration) with
# ends derived from the next start; ``legacy`` keeps model spans as the
# configured rollback. Seams may miss by this tolerance (cut keyframe
# slop plus approximate model boundaries); anything wider fails closed.
DEFAULT_STEP_SPAN_MODE = DEFAULT_SPAN_MODE
DEFAULT_CONTIGUITY_TOLERANCE_S = 2.0
# Whole-video extraction: videos at most this long go
# out as ONE window when the honest reservation fits, instead of 60 s
# chunks. The live probe sent 214 s / 57 MB in one call and got an
# answer in ~30 s; 240 s covers that clip with headroom under the
# 300 s duration cap.
DEFAULT_WHOLE_VIDEO_MAX_S = 240.0
# Worst-case Spark input tokens per video second for windows past the
# 60 s chunk bound (measured ~273/s at 720p; 350 carries headroom for
# 1080p-capped previews). Sizes the per-run reservation for a window.
DEFAULT_SPARK_INPUT_TOKENS_PER_VIDEO_S = 350.0
# Burned-in m:ss.s clock overlaid on every extraction chunk before
# upload (the probe's clock variant placed 4/8 true boundaries within
# 1 s where the plain variant placed none).
DEFAULT_CLOCK_BURN_IN = True
# ffmpeg scene-cut threshold for boundary candidates (0.15 measured
# best: 7/8 true edges within 3 s). Strictly inside (0, 1]: 0 would
# cut at every frame.
DEFAULT_SCENE_CUT_THRESHOLD = 0.15
# List the cut candidates in the extraction prompt (wording validated:
# one run 7/9, consensus 6/9, every boundary on a listed
# cut — so on by the replay result).
DEFAULT_SCENE_CUTS_IN_PROMPT = True
# Snap each consensus boundary to the nearest cut within this window.
DEFAULT_SNAP_WINDOW_S = 2.0
# Consensus phases shorter than this fold into a neighbour.
DEFAULT_MIN_PHASE_LENGTH_S = 5.0
# Fuse adjacent phases with the same normalised action after fold
# (the live consensus split one verify phase in two).
DEFAULT_MERGE_SAME_ACTION = True
# Drop leading/trailing phases the run majority tagged
# outro_or_unrelated, recording their spans (the guide leaves out the
# non-repair teaser).
DEFAULT_EXCLUDE_OUTRO_PHASES = True
# The live Spark request output cap: one shared
# constant for the request body, the worst-case estimate, and the
# reservation default below. It lives here because the provider package
# already imports settings, so sharing the other way would cycle.
SPARK_MAX_OUTPUT_TOKENS = 10000
# Spark reservation for one extraction window: the
# default covers 3 whole-video runs of a 240 s clip at the default
# per-second input-token rate and docs prices, plus a 10% margin for
# decoded-duration overrun past the nominal window. Computed from those
# inputs, not guessed: 3 x (84000 x $0.00000125 + 10000 x $0.00000425)
# x 1.10 = $0.48675.
SPARK_RESERVATION_MARGIN = Decimal("1.10")
_SPARK_RESERVATION_INPUT_TOKENS = math.ceil(
    DEFAULT_WHOLE_VIDEO_MAX_S * DEFAULT_SPARK_INPUT_TOKENS_PER_VIDEO_S
)
DEFAULT_SPARK_RESERVATION_USD = (
    Decimal(_SPARK_RESERVATION_INPUT_TOKENS)
    * DEFAULT_SPARK_INPUT_USD_PER_TOKEN
    + Decimal(SPARK_MAX_OUTPUT_TOKENS)
    * DEFAULT_SPARK_OUTPUT_USD_PER_TOKEN
) * SPARK_RUNS_PER_GUIDE * SPARK_RESERVATION_MARGIN
META_FILES_TIMEOUT_ENV_VAR = "META_FILES_TIMEOUT_S"
META_FILES_UPLOAD_TIMEOUT_ENV_VAR = "META_FILES_UPLOAD_TIMEOUT_S"
META_FILES_MAX_UPLOAD_BYTES_ENV_VAR = "META_FILES_MAX_UPLOAD_BYTES"
META_FILES_UPLOAD_PURPOSE_ENV_VAR = "META_FILES_UPLOAD_PURPOSE"
META_FILES_PURPOSE_FIELD_ENV_VAR = "META_FILES_PURPOSE_FIELD"
META_FILES_FILE_FIELD_ENV_VAR = "META_FILES_FILE_FIELD"
META_FILES_FILE_CONTENT_TYPE_ENV_VAR = "META_FILES_FILE_CONTENT_TYPE"
META_FILES_EXPIRES_ANCHOR_FIELD_ENV_VAR = "META_FILES_EXPIRES_ANCHOR_FIELD"
META_FILES_EXPIRES_SECONDS_FIELD_ENV_VAR = "META_FILES_EXPIRES_SECONDS_FIELD"
META_FILES_EXPIRES_ANCHOR_ENV_VAR = "META_FILES_EXPIRES_ANCHOR"
META_FILES_EXPIRY_SKEW_ENV_VAR = "META_FILES_EXPIRY_SKEW_S"
META_FILES_MAX_ERROR_BYTES_ENV_VAR = "META_FILES_MAX_ERROR_BYTES"

# Multipart upload knobs: every field name, the purpose
# value, the expiry anchor, the timeouts, and the client-side size bound.
# Single owner: app.meta_files imports these for its upload config.
DEFAULT_META_FILES_TIMEOUT_S = 10.0
DEFAULT_META_FILES_UPLOAD_TIMEOUT_S = 600.0
DEFAULT_META_FILES_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_META_FILES_UPLOAD_PURPOSE = "user_data"
DEFAULT_META_FILES_PURPOSE_FIELD = "purpose"
DEFAULT_META_FILES_FILE_FIELD = "file"
DEFAULT_META_FILES_FILE_CONTENT_TYPE = "video/mp4"
DEFAULT_META_FILES_EXPIRES_ANCHOR_FIELD = "expires_after[anchor]"
DEFAULT_META_FILES_EXPIRES_SECONDS_FIELD = "expires_after[seconds]"
DEFAULT_META_FILES_EXPIRES_ANCHOR = "created_at"
DEFAULT_META_FILES_EXPIRY_SKEW_S = 300.0
DEFAULT_META_FILES_MAX_ERROR_BYTES = 64 * 1024

# Multipart field names are interpolated into Content-Disposition
# parameters, so only the supported token/bracket grammar is accepted.
_MULTIPART_FIELD_NAME_RE = re.compile(r"[A-Za-z0-9_\[\]]+")


def is_multipart_field_name(text: str) -> bool:
    """True when a field name uses only the token/bracket grammar."""
    return (
        isinstance(text, str)
        and _MULTIPART_FIELD_NAME_RE.fullmatch(text) is not None
    )


@dataclass(frozen=True)
class Settings:
    data_root: Path
    db_path: Path
    production: bool
    meta_api_key: str | None = field(default=None, repr=False)
    meta_files_base_url: str | None = None
    providers: ProviderSettings = field(default_factory=ProviderSettings)
    image_match: ImageMatchSettings = field(default_factory=ImageMatchSettings)
    pipeline_mode: str = DEFAULT_PIPELINE_MODE
    guide_filename: str = DEFAULT_GUIDE_FILENAME
    guide_clip_filename: str = DEFAULT_GUIDE_CLIP_FILENAME
    label_image_pattern: str = DEFAULT_LABEL_IMAGE_PATTERN
    settle_max_attempts: int = DEFAULT_SETTLE_MAX_ATTEMPTS
    max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT_JOBS
    job_status_refresh_s: int = DEFAULT_JOB_STATUS_REFRESH_S
    pending_settlement_filename: str = DEFAULT_PENDING_SETTLEMENT_FILENAME
    sample_pointer_filename: str = DEFAULT_SAMPLE_POINTER_FILENAME
    sample_bundle_dirname: str = DEFAULT_SAMPLE_BUNDLE_DIRNAME
    sample_bundle_clip_filename: str = DEFAULT_SAMPLE_BUNDLE_CLIP_FILENAME
    sample_bundle_version: int = DEFAULT_SAMPLE_BUNDLE_VERSION
    sample_bundle_notes_filename: str = DEFAULT_SAMPLE_BUNDLE_NOTES_FILENAME
    corrections_filename: str = DEFAULT_CORRECTIONS_FILENAME
    correction_phrase_max_chars: int = DEFAULT_CORRECTION_PHRASE_MAX_CHARS
    correction_max_ops: int = DEFAULT_CORRECTION_MAX_OPS
    image_zip_name: str = DEFAULT_IMAGE_ZIP_NAME
    image_zip_entry_pattern: str = DEFAULT_IMAGE_ZIP_ENTRY_PATTERN
    export_frame_pattern: str = DEFAULT_EXPORT_FRAME_PATTERN
    export_label_pattern: str = DEFAULT_EXPORT_LABEL_PATTERN
    export_html_name: str = DEFAULT_EXPORT_HTML_NAME
    export_zip_name: str = DEFAULT_EXPORT_ZIP_NAME
    export_images_dir: str = DEFAULT_EXPORT_IMAGES_DIR
    export_link_retention_hours: int = UPLOAD_PREVIEW_TTL_HOURS
    meta_files_timeout_s: float = DEFAULT_META_FILES_TIMEOUT_S
    meta_files_upload_timeout_s: float = DEFAULT_META_FILES_UPLOAD_TIMEOUT_S
    meta_files_max_upload_bytes: int = DEFAULT_META_FILES_MAX_UPLOAD_BYTES
    meta_files_upload_purpose: str = DEFAULT_META_FILES_UPLOAD_PURPOSE
    meta_files_purpose_field: str = DEFAULT_META_FILES_PURPOSE_FIELD
    meta_files_file_field: str = DEFAULT_META_FILES_FILE_FIELD
    meta_files_file_content_type: str = DEFAULT_META_FILES_FILE_CONTENT_TYPE
    meta_files_expires_anchor_field: str = DEFAULT_META_FILES_EXPIRES_ANCHOR_FIELD
    meta_files_expires_seconds_field: str = (
        DEFAULT_META_FILES_EXPIRES_SECONDS_FIELD
    )
    meta_files_expires_anchor: str = DEFAULT_META_FILES_EXPIRES_ANCHOR
    meta_files_expiry_skew_s: float = DEFAULT_META_FILES_EXPIRY_SKEW_S
    meta_files_max_error_bytes: int = DEFAULT_META_FILES_MAX_ERROR_BYTES
    max_video_s: float = DEFAULT_MAX_VIDEO_S
    chunk_s: float = DEFAULT_CHUNK_S
    chunk_overlap_s: float = DEFAULT_CHUNK_OVERLAP_S
    chunk_cut_timeout_s: float = DEFAULT_CHUNK_CUT_TIMEOUT_S
    chunk_deadline_allowance_s: float = DEFAULT_CHUNK_DEADLINE_ALLOWANCE_S
    deadline_extension_max_s: float = DEFAULT_DEADLINE_EXTENSION_MAX_S
    deadline_extension_max_attempts: int = DEFAULT_DEADLINE_EXTENSION_MAX_ATTEMPTS
    job_cost_cap_usd: Decimal = DEFAULT_JOB_COST_CAP_USD
    chunk_max_attempts: int = DEFAULT_CHUNK_MAX_ATTEMPTS
    chunk_retry_budget_usd: Decimal = DEFAULT_CHUNK_RETRY_BUDGET_USD
    extract_retry_count: int = DEFAULT_EXTRACT_RETRY_COUNT
    extract_retry_backoff_s: float = DEFAULT_EXTRACT_RETRY_BACKOFF_S
    extract_retry_min_remaining_s: float = DEFAULT_EXTRACT_RETRY_MIN_REMAINING_S
    extract_retry_failures: frozenset[str] = DEFAULT_EXTRACT_RETRY_FAILURES
    chunk_bundle_ttl_s: float = DEFAULT_CHUNK_BUNDLE_TTL_S
    step_span_mode: str = DEFAULT_STEP_SPAN_MODE
    contiguity_tolerance_s: float = DEFAULT_CONTIGUITY_TOLERANCE_S
    whole_video_max_s: float = DEFAULT_WHOLE_VIDEO_MAX_S
    spark_input_tokens_per_video_s: float = (
        DEFAULT_SPARK_INPUT_TOKENS_PER_VIDEO_S
    )
    clock_burn_in: bool = DEFAULT_CLOCK_BURN_IN
    scene_cut_threshold: float = DEFAULT_SCENE_CUT_THRESHOLD
    scene_cuts_in_prompt: bool = DEFAULT_SCENE_CUTS_IN_PROMPT
    snap_window_s: float = DEFAULT_SNAP_WINDOW_S
    min_phase_length_s: float = DEFAULT_MIN_PHASE_LENGTH_S
    spark_reservation_usd: Decimal = DEFAULT_SPARK_RESERVATION_USD
    merge_same_action: bool = DEFAULT_MERGE_SAME_ACTION
    exclude_outro_phases: bool = DEFAULT_EXCLUDE_OUTRO_PHASES

    def __post_init__(self) -> None:
        """Fail loudly on missing or misshaped pipeline config values."""
        if self.pipeline_mode not in ("replay", "live"):
            raise ValueError(
                f"pipeline_mode must be 'replay' or 'live', got {self.pipeline_mode!r}"
            )
        for name in (
            "guide_filename",
            "guide_clip_filename",
            "pending_settlement_filename",
            "sample_pointer_filename",
            "sample_bundle_dirname",
            "sample_bundle_clip_filename",
            "sample_bundle_notes_filename",
            "corrections_filename",
            "image_zip_name",
            "export_html_name",
            "export_zip_name",
            "export_images_dir",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value in (".", "..")
                or "/" in value
                or "\\" in value
                or "\x00" in value
                or any(not ch.isprintable() or ch == '"' for ch in value)
                or bool(ntpath.splitdrive(value)[0])
                or len(value) > 64
            ):
                raise ValueError(f"{name} must be a plain filename, got {value!r}")
        outputs = {}
        for name, required in (
            ("image_zip_entry_pattern", {"number"}),
            ("export_frame_pattern", {"number"}),
            ("export_label_pattern", {"number", "part"}),
        ):
            pattern_value = getattr(self, name)
            try:
                fields = {field for _, field, _, _ in string.Formatter().parse(pattern_value) if field is not None}
                if fields != required:
                    raise ValueError(f"requires fields {sorted(required)}")
                made = [pattern_value.format(number=n, part=p)
                        for n in range(1, MAX_STEPS + 1)
                        for p in (range(1, MAX_PARTS_PER_STEP + 1) if 'part' in required else (1,))]
            except (TypeError, AttributeError, KeyError, IndexError, ValueError) as error:
                raise ValueError(f"{name} does not format: {error}") from error
            if any(not v or v in ('.', '..') or '/' in v or '\\' in v or ntpath.splitdrive(v)[0] or any(not ch.isprintable() for ch in v) for v in made):
                raise ValueError(f"{name} must render a plain filename")
            if len(set(made)) != len(made):
                raise ValueError(f"{name} renders duplicate filenames")
            outputs[name] = set(made)
        if outputs['export_frame_pattern'] & outputs['export_label_pattern']:
            raise ValueError("export image patterns render duplicate filenames")
        for name in ("correction_phrase_max_chars", "correction_max_ops", "export_link_retention_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        pattern = self.label_image_pattern
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("label_image_pattern must be a non-empty string")
        if "{step}" not in pattern or "{part}" not in pattern:
            raise ValueError(
                "label_image_pattern must contain {step} and {part},"
                f" got {pattern!r}"
            )
        try:
            rendered = pattern.format(step=1, part=0)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                f"label_image_pattern does not format: {error}"
            ) from error
        if "/" in rendered or "\\" in rendered or not rendered:
            raise ValueError(
                f"label_image_pattern must render a plain filename, got {pattern!r}"
            )
        attempts = self.settle_max_attempts
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
        ):
            raise ValueError(
                f"settle_max_attempts must be an int >= 1, got {attempts!r}"
            )
        for name in ("max_concurrent_jobs", "job_status_refresh_s", "deadline_extension_max_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        for name in (
            "max_video_s",
            "chunk_s",
            "chunk_cut_timeout_s",
            "chunk_deadline_allowance_s",
            "deadline_extension_max_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not value > 0
            ):
                raise ValueError(
                    f"{name} must be a finite number > 0, got {value!r}"
                )
        overlap = self.chunk_overlap_s
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, (int, float))
            or not math.isfinite(overlap)
            or not overlap >= 0
        ):
            raise ValueError(
                f"chunk_overlap_s must be a finite number >= 0, got {overlap!r}"
            )
        if overlap >= self.chunk_s:
            raise ValueError(
                "chunk_overlap_s must be below chunk_s, else chunks never advance,"
                f" got overlap={overlap!r} chunk_s={self.chunk_s!r}"
            )
        cap = self.job_cost_cap_usd
        if not isinstance(cap, Decimal) or not cap.is_finite() or not cap > 0:
            raise ValueError(
                f"job_cost_cap_usd must be a finite positive Decimal, got {cap!r}"
            )
        reservation = self.spark_reservation_usd
        if (
            not isinstance(reservation, Decimal)
            or not reservation.is_finite()
            or not reservation > 0
        ):
            raise ValueError(
                "spark_reservation_usd must be a finite positive Decimal,"
                f" got {reservation!r}"
            )
        attempts = self.chunk_max_attempts
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
        ):
            raise ValueError(
                f"chunk_max_attempts must be an int >= 1, got {attempts!r}"
            )
        retry_budget = self.chunk_retry_budget_usd
        if (
            not isinstance(retry_budget, Decimal)
            or not retry_budget.is_finite()
            or not retry_budget >= 0
        ):
            raise ValueError(
                "chunk_retry_budget_usd must be a finite Decimal >= 0,"
                f" got {retry_budget!r}"
            )
        if type(self.extract_retry_count) is not int or self.extract_retry_count not in (0, 1):
            raise ValueError("extract_retry_count must be 0 or 1")
        for name in ("extract_retry_backoff_s", "extract_retry_min_remaining_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not isinstance(self.extract_retry_failures, frozenset) or not self.extract_retry_failures <= DEFAULT_EXTRACT_RETRY_FAILURES:
            raise ValueError("extract_retry_failures contains an unknown service failure")
        bundle_ttl = self.chunk_bundle_ttl_s
        if (
            isinstance(bundle_ttl, bool)
            or not isinstance(bundle_ttl, (int, float))
            or not math.isfinite(bundle_ttl)
            or not bundle_ttl > 0
        ):
            raise ValueError(
                "chunk_bundle_ttl_s must be a finite number > 0,"
                f" got {bundle_ttl!r}"
            )
        if self.step_span_mode not in SPAN_MODES:
            raise ValueError(
                "step_span_mode must be one of"
                f" {', '.join(SPAN_MODES)}, got {self.step_span_mode!r}"
            )
        tolerance = self.contiguity_tolerance_s
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or not tolerance >= 0
        ):
            raise ValueError(
                "contiguity_tolerance_s must be a finite number >= 0,"
                f" got {tolerance!r}"
            )
        for name in ("whole_video_max_s", "spark_input_tokens_per_video_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not value > 0
            ):
                raise ValueError(
                    f"{name} must be a finite number > 0, got {value!r}"
                )
        for name in (
            "clock_burn_in",
            "scene_cuts_in_prompt",
            "merge_same_action",
            "exclude_outro_phases",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool, got {value!r}")
        threshold = self.scene_cut_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 < threshold <= 1
        ):
            raise ValueError(
                "scene_cut_threshold must be a finite number in (0, 1],"
                f" got {threshold!r}"
            )
        snap = self.snap_window_s
        if (
            isinstance(snap, bool)
            or not isinstance(snap, (int, float))
            or not math.isfinite(snap)
            or not snap >= 0
        ):
            raise ValueError(
                f"snap_window_s must be a finite number >= 0, got {snap!r}"
            )
        minimum = self.min_phase_length_s
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum)
            or not minimum > 0
        ):
            raise ValueError(
                "min_phase_length_s must be a finite number > 0,"
                f" got {minimum!r}"
            )

    def label_image_name(self, *, step: int, part: int) -> str:
        """Render one provisional label filename from the configured pattern."""
        return self.label_image_pattern.format(step=step, part=part)


def _validate_https_url(raw: str, var_name: str) -> str:
    """Return the normalized https endpoint, or raise ``RuntimeError``.

    Only ``https`` URLs with a host and no embedded credentials are
    accepted; anything else fails closed so a keyed client can never
    point at plaintext or a non-URL. Test doubles pass their stub base
    URL straight to the client constructor instead.
    """
    candidate = (raw or "").strip().rstrip("/")
    try:
        parts = urlsplit(candidate)
    except ValueError as error:
        raise RuntimeError(f"invalid {var_name}: not a URL") from error
    if (
        parts.scheme.lower() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise RuntimeError(
            f"invalid {var_name}: expected an https URL "
            "with a host and no credentials"
        )
    return candidate


def validate_meta_files_base_url(raw: str) -> str:
    """Return the normalized Meta files endpoint, or raise ``RuntimeError``."""
    return _validate_https_url(raw, META_FILES_BASE_URL_ENV_VAR)


def validate_meta_api_base_url(raw: str) -> str:
    """Return the normalized Model API endpoint, or raise ``RuntimeError``."""
    return _validate_https_url(raw, META_API_BASE_URL_ENV_VAR)


def _provider_int(raw: Mapping[str, str], var: str, default: int) -> int:
    text = raw.get(var, "").strip()
    if not text:
        return default
    try:
        value = int(text, 10)
    except ValueError as error:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected an int)") from error
    if value <= 0:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a positive int)")
    return value


def _extract_retry_count(raw: Mapping[str, str]) -> int:
    text = raw.get(EXTRACT_RETRY_COUNT_ENV_VAR, "").strip()
    if not text:
        return DEFAULT_EXTRACT_RETRY_COUNT
    if text not in ("0", "1"):
        raise RuntimeError(f"invalid {EXTRACT_RETRY_COUNT_ENV_VAR}: expected 0 or 1")
    return int(text)


def _multipart_text(raw: Mapping[str, str], var: str, default: str) -> str:
    """Return a multipart knob, or its default when blank.

    Anything operator-set must be one header-safe line: a newline would
    corrupt the multipart body the client builds from it.
    """
    text = (raw.get(var, "") or "").strip()
    if not text:
        return default
    if "\r" in text or "\n" in text:
        raise RuntimeError(f"invalid {var}: must not contain a newline")
    return text


def _multipart_field_name(raw: Mapping[str, str], var: str, default: str) -> str:
    """Return a multipart field name, or its default when blank.

    Field names are interpolated into ``Content-Disposition`` parameters,
    so anything outside the token/bracket grammar (quotes, semicolons,
    whitespace) is refused instead of risking a parameter injection.
    """
    text = (raw.get(var, "") or "").strip()
    if not text:
        return default
    if not is_multipart_field_name(text):
        raise RuntimeError(
            f"invalid {var}: expected letters, digits, '_' or '[...]', "
            f"got {text!r}"
        )
    return text


def _timeout_setting(raw: Mapping[str, str], var: str, default: float) -> float:
    """Return a positive finite timeout in seconds, or its default."""
    text = (raw.get(var, "") or "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise RuntimeError(
            f"invalid {var}: expected seconds as a number, got {text!r}"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid {var}: expected positive seconds, got {text!r}")
    return value


def _max_bytes_setting(raw: Mapping[str, str], var: str, default: int) -> int:
    """Return a positive byte bound, or its default."""
    text = (raw.get(var, "") or "").strip()
    if not text:
        return default
    try:
        value = int(text, 10)
    except ValueError as error:
        raise RuntimeError(
            f"invalid {var}: expected an integer byte count, got {text!r}"
        ) from error
    if value <= 0:
        raise RuntimeError(f"invalid {var}: expected a positive count, got {text!r}")
    return value


def _provider_float(raw: Mapping[str, str], var: str, default: float) -> float:
    text = raw.get(var, "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a number)") from error
    if not math.isfinite(value) or not value > 0:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a positive number)")
    return value


def _nonnegative_float(raw: Mapping[str, str], var: str, default: float) -> float:
    """Return a finite float >= 0, or its default when blank."""
    text = raw.get(var, "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a number)") from error
    if not math.isfinite(value) or not value >= 0:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a number >= 0)")
    return value


def _provider_threshold(raw: Mapping[str, str], var: str, default: float) -> float:
    text = raw.get(var, "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a number)") from error
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected 0..1)")
    return value


def _provider_price(raw: Mapping[str, str], var: str, default: Decimal) -> Decimal:
    text = raw.get(var, "").strip()
    if not text:
        return default
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a decimal)") from error
    # Zero is refused: a paid API has no free calls, and the Spark run
    # shares and SAM holds derived from these prices must stay positive.
    if not value.is_finite() or not value > 0:
        raise RuntimeError(f"invalid {var} value: {text!r} (expected a finite positive decimal)")
    return value


def _nonnegative_price(
    raw: Mapping[str, str], var: str, default: Decimal
) -> Decimal:
    """Return a finite Decimal >= 0, or its default when blank.

    Unlike :func:`_provider_price`, zero is allowed: a zero retry
    budget is the honest way to configure no retries.
    """
    text = raw.get(var, "").strip()
    if not text:
        return default
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise RuntimeError(
            f"invalid {var} value: {text!r} (expected a decimal)"
        ) from error
    if not value.is_finite() or not value >= 0:
        raise RuntimeError(
            f"invalid {var} value: {text!r} (expected a finite decimal >= 0)"
        )
    return value


def _provider_text(raw: Mapping[str, str], var: str, default: str) -> str:
    text = raw.get(var, "").strip()
    return text if text else default


def _provider_model(
    raw: Mapping[str, str], var: str, default: str, known: Mapping[str, str]
) -> str:
    model = _provider_text(raw, var, default)
    if model not in known:
        raise RuntimeError(
            f"invalid {var} value: {model!r} (expected one of {', '.join(sorted(known))})"
        )
    return model


def _provider_alignment_mode(raw: Mapping[str, str]) -> str:
    """Return the step alignment mode, ``align`` unless set to ``exact``."""
    mode = _provider_text(
        raw, STEP_ALIGNMENT_MODE_ENV_VAR, step_alignment.DEFAULT_ALIGNMENT_MODE
    )
    if mode not in step_alignment.ALIGNMENT_MODES:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_MODE_ENV_VAR} value: {mode!r}"
            f" (expected one of {', '.join(step_alignment.ALIGNMENT_MODES)})"
        )
    return mode


def _provider_generic_tokens(raw: Mapping[str, str]) -> frozenset[str]:
    """Return the generic alignment tokens, default unless overridden.

    An override is comma-separated (``battery,cell``); every item must be
    a valid token after stripping, so a typo fails loudly instead of
    silently matching nothing.
    """
    text = (raw.get(STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR, "") or "").strip()
    if not text:
        return DEFAULT_STEP_ALIGNMENT_GENERIC_TOKENS
    try:
        return step_alignment.normalize_generic_tokens(text.split(","))
    except ValueError as error:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR} value:"
            f" {raw.get(STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR)!r} ({error})"
        ) from error


def _provider_verb_synonyms(
    raw: Mapping[str, str],
) -> tuple[frozenset[str], ...]:
    """Return the verb synonym groups, default unless overridden.

    An override lists groups separated by ``;`` with comma-separated
    verbs inside (``remove,detach;disconnect,unplug``); every verb must
    be valid after stripping, so a misspelled override fails loudly
    instead of silently matching nothing.
    """
    text = (raw.get(STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR, "") or "").strip()
    if not text:
        return DEFAULT_STEP_ALIGNMENT_VERB_SYNONYMS
    try:
        return step_alignment.normalize_verb_synonyms(
            [frozenset(group.split(",")) for group in text.split(";")]
        )
    except ValueError as error:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR} value:"
            f" {raw.get(STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR)!r} ({error})"
        ) from error


def _provider_negation_words(raw: Mapping[str, str]) -> frozenset[str]:
    """Return the negation words, default unless overridden.

    An override is comma-separated (``not,never``); every item must be
    a valid word or phrase after stripping, so a typo fails loudly
    instead of silently matching nothing.
    """
    text = (raw.get(STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR, "") or "").strip()
    if not text:
        return DEFAULT_STEP_ALIGNMENT_NEGATION_WORDS
    try:
        return step_alignment.normalize_negation_words(text.split(","))
    except ValueError as error:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR} value:"
            f" {raw.get(STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR)!r} ({error})"
        ) from error


def _provider_qualifiers(raw: Mapping[str, str]) -> frozenset[str]:
    """Return the qualifier words, default unless overridden.

    An override is comma-separated (``internal,external``); every item
    must be a valid word after stripping, so a typo fails loudly
    instead of silently matching nothing.
    """
    text = (raw.get(STEP_ALIGNMENT_QUALIFIERS_ENV_VAR, "") or "").strip()
    if not text:
        return DEFAULT_STEP_ALIGNMENT_QUALIFIERS
    try:
        return step_alignment.normalize_qualifiers(text.split(","))
    except ValueError as error:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_QUALIFIERS_ENV_VAR} value:"
            f" {raw.get(STEP_ALIGNMENT_QUALIFIERS_ENV_VAR)!r} ({error})"
        ) from error


def _provider_part_synonyms(
    raw: Mapping[str, str],
) -> tuple[frozenset[str], ...]:
    """Return the part synonym groups, default unless overridden.

    An override lists groups separated by ``;`` with comma-separated
    words inside (``cable,connector;screw,fastener``); every word must
    be valid after stripping, so a misspelled override fails loudly
    instead of silently matching nothing.
    """
    text = (raw.get(STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR, "") or "").strip()
    if not text:
        return DEFAULT_STEP_ALIGNMENT_PART_SYNONYMS
    try:
        return step_alignment.normalize_part_synonyms(
            [frozenset(group.split(",")) for group in text.split(";")]
        )
    except ValueError as error:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR} value:"
            f" {raw.get(STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR)!r} ({error})"
        ) from error


def _provider_transition_verbs(raw: Mapping[str, str]) -> frozenset[str]:
    """Return the transition verbs, default unless overridden.

    An override is comma-separated (``flip,turn over``); every item
    must be a valid verb word or phrase after stripping, so a typo
    fails loudly instead of silently matching nothing.
    """
    text = (raw.get(STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR, "") or "").strip()
    if not text:
        return DEFAULT_STEP_ALIGNMENT_TRANSITION_VERBS
    try:
        return step_alignment.normalize_transition_verbs(text.split(","))
    except ValueError as error:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR} value:"
            f" {raw.get(STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR)!r} ({error})"
        ) from error


def _provider_verb_less_overlap_fraction(raw: Mapping[str, str]) -> float:
    """Return the verb-less overlap fraction, default unless overridden.

    Anything outside 0 < f <= 1 fails loudly: zero would link
    mutual-best phases that never overlap, above one nothing links.
    """
    text = (
        raw.get(STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR, "") or ""
    ).strip()
    if not text:
        return DEFAULT_STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION
    try:
        return step_alignment.normalize_verb_less_overlap_fraction(float(text))
    except ValueError as error:
        raise RuntimeError(
            f"invalid {STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR} value:"
            f" {raw.get(STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR)!r}"
            f" ({error})"
        ) from error


def _provider_boundary_tolerance(raw: Mapping[str, str]) -> float | None:
    """Return the consensus boundary tolerance, None (the flag line) unless set.

    A set value must be a positive number of seconds, like the flag
    line it defaults to; anything else fails loudly.
    """
    text = (raw.get(CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR, "") or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise RuntimeError(
            f"invalid {CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR} value:"
            f" {raw.get(CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR)!r}"
            " (expected a number)"
        ) from error
    if not math.isfinite(value) or not value > 0:
        raise RuntimeError(
            f"invalid {CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR} value:"
            f" {raw.get(CONSENSUS_BOUNDARY_TOLERANCE_ENV_VAR)!r}"
            " (expected a positive number)"
        )
    return value


def _span_mode_setting(raw: Mapping[str, str]) -> str:
    """Return the step-span mode, ``contiguous`` unless set to ``legacy``."""
    mode = _provider_text(raw, STEP_SPAN_MODE_ENV_VAR, DEFAULT_STEP_SPAN_MODE)
    if mode not in SPAN_MODES:
        raise RuntimeError(
            f"invalid {STEP_SPAN_MODE_ENV_VAR} value: {mode!r}"
            f" (expected one of {', '.join(SPAN_MODES)})"
        )
    return mode


def _bool_setting(raw: Mapping[str, str], var: str, default: bool) -> bool:
    """Return a 1/true/yes switch, or its default when blank."""
    text = (raw.get(var, "") or "").strip().lower()
    if not text:
        return default
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    raise RuntimeError(
        f"invalid {var} value: {raw.get(var)!r}"
        " (expected one of 1/true/yes or 0/false/no)"
    )


def _unit_interval(raw: Mapping[str, str], var: str, default: float) -> float:
    """Return a finite number in (0, 1], or its default when blank."""
    text = (raw.get(var, "") or "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise RuntimeError(
            f"invalid {var} value: {text!r} (expected a number)"
        ) from error
    if not math.isfinite(value) or not 0 < value <= 1:
        raise RuntimeError(
            f"invalid {var} value: {text!r} (expected a number in (0, 1])"
        )
    return value


def load_provider_settings(env: Mapping[str, str]) -> ProviderSettings:
    """Build provider tunables from the environment mapping. Loud on bad values."""
    mode = _provider_text(env, PROVIDER_MODE_ENV_VAR, "replay").lower()
    if mode not in PROVIDER_MODES:
        raise RuntimeError(
            f"invalid {PROVIDER_MODE_ENV_VAR} value: {env.get(PROVIDER_MODE_ENV_VAR)!r}"
            f" (expected one of {', '.join(PROVIDER_MODES)})"
        )
    base_raw = env.get(META_API_BASE_URL_ENV_VAR, "").strip()
    base_url = validate_meta_api_base_url(base_raw) if base_raw else DEFAULT_META_API_BASE_URL
    return ProviderSettings(
        mode=mode,
        meta_api_base_url=base_url,
        spark_model_id=_provider_model(
            env, SPARK_MODEL_ENV_VAR, DEFAULT_SPARK_MODEL_ID, SPARK_MODEL_RECORD_NAMES
        ),
        sam_model_id=_provider_model(
            env, SAM_MODEL_ENV_VAR, DEFAULT_SAM_MODEL_ID, SAM_MODEL_RECORD_NAMES
        ),
        spark_runs=_provider_int(env, SPARK_RUNS_ENV_VAR, SPARK_RUNS_PER_GUIDE),
        time_spread_flag_s=_provider_float(env, TIME_SPREAD_ENV_VAR, float(TIME_SPREAD_FLAG_S)),
        sam_max_starts_per_minute=_provider_int(
            env, SAM_RATE_ENV_VAR, SAM_MAX_REQUEST_STARTS_PER_MINUTE
        ),
        sam_threshold_fan=_provider_threshold(
            env, SAM_THRESHOLD_FAN_ENV_VAR, PART_CHECK_THRESHOLDS["fan"]
        ),
        sam_threshold_battery=_provider_threshold(
            env, SAM_THRESHOLD_BATTERY_ENV_VAR, PART_CHECK_THRESHOLDS["battery"]
        ),
        sam_phrase_fan=_provider_text(env, SAM_PHRASE_FAN_ENV_VAR, DEFAULT_SAM_PHRASE_FAN),
        sam_phrase_battery=_provider_text(
            env, SAM_PHRASE_BATTERY_ENV_VAR, DEFAULT_SAM_PHRASE_BATTERY
        ),
        spark_timeout_s=_provider_float(env, SPARK_TIMEOUT_ENV_VAR, DEFAULT_SPARK_TIMEOUT_S),
        sam_timeout_s=_provider_float(env, SAM_TIMEOUT_ENV_VAR, DEFAULT_SAM_TIMEOUT_S),
        readback_timeout_s=_provider_float(
            env, READBACK_TIMEOUT_ENV_VAR, DEFAULT_READBACK_TIMEOUT_S
        ),
        spark_input_usd_per_token=_provider_price(
            env, SPARK_INPUT_PRICE_ENV_VAR, DEFAULT_SPARK_INPUT_USD_PER_TOKEN
        ),
        spark_output_usd_per_token=_provider_price(
            env, SPARK_OUTPUT_PRICE_ENV_VAR, DEFAULT_SPARK_OUTPUT_USD_PER_TOKEN
        ),
        sam_usd_per_call=_provider_price(env, SAM_PRICE_ENV_VAR, SAM_COST_PER_CALL_USD),
        step_alignment_mode=_provider_alignment_mode(env),
        step_alignment_generic_tokens=_provider_generic_tokens(env),
        step_alignment_verb_synonyms=_provider_verb_synonyms(env),
        step_alignment_negation_words=_provider_negation_words(env),
        step_alignment_qualifiers=_provider_qualifiers(env),
        step_alignment_part_synonyms=_provider_part_synonyms(env),
        step_alignment_transition_verbs=_provider_transition_verbs(env),
        step_alignment_verb_less_overlap_fraction=(
            _provider_verb_less_overlap_fraction(env)
        ),
        consensus_boundary_tolerance_s=_provider_boundary_tolerance(env),
    )


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings. Production refuses to start without required paths/key."""
    raw = os.environ if env is None else env
    production_raw = raw.get(PRODUCTION_ENV_VAR, "").strip().lower()
    if production_raw in ("", "0", "false", "no"):
        production = False
    elif production_raw in ("1", "true", "yes"):
        production = True
    else:
        raise RuntimeError(
            f"invalid {PRODUCTION_ENV_VAR} value: {raw.get(PRODUCTION_ENV_VAR)!r}"
            ' (expected one of 1/true/yes or 0/false/no)'
        )
    data_root_raw = raw.get(DATA_ROOT_ENV_VAR, "").strip()
    db_path_raw = raw.get(DB_PATH_ENV_VAR, "").strip()
    api_key = raw.get(API_KEY_ENV_VAR, "").strip() or None
    endpoint_raw = raw.get(META_FILES_BASE_URL_ENV_VAR, "").strip()
    endpoint = (
        validate_meta_files_base_url(endpoint_raw) if endpoint_raw else None
    )
    mode_raw = raw.get(PIPELINE_MODE_ENV_VAR, "").strip().lower()
    if mode_raw == "":
        pipeline_mode = DEFAULT_PIPELINE_MODE
    elif mode_raw in ("replay", "live"):
        pipeline_mode = mode_raw
    else:
        raise RuntimeError(
            f"invalid {PIPELINE_MODE_ENV_VAR} value: {raw.get(PIPELINE_MODE_ENV_VAR)!r}"
            " (expected one of replay/live)"
        )
    provider_raw = raw.get(PROVIDER_MODE_ENV_VAR, "").strip()
    if production and not provider_raw:
        # Production has one mode switch: unset, the providers follow the
        # pipeline, so a live pipeline never meets replay providers.
        providers = load_provider_settings(
            {**raw, PROVIDER_MODE_ENV_VAR: pipeline_mode}
        )
    else:
        providers = load_provider_settings(raw)
    if production and pipeline_mode == "live" and providers.mode != "live":
        raise RuntimeError(
            f"production mode refuses {PIPELINE_MODE_ENV_VAR}=live with"
            f" {PROVIDER_MODE_ENV_VAR}={providers.mode}: no live job could start"
        )
    attempts_raw = raw.get(SETTLE_MAX_ATTEMPTS_ENV_VAR, "").strip()
    if attempts_raw == "":
        settle_max_attempts = DEFAULT_SETTLE_MAX_ATTEMPTS
    else:
        try:
            settle_max_attempts = int(attempts_raw)
        except ValueError as error:
            raise RuntimeError(
                f"invalid {SETTLE_MAX_ATTEMPTS_ENV_VAR} value:"
                f" {raw.get(SETTLE_MAX_ATTEMPTS_ENV_VAR)!r} (expected an int >= 1)"
            ) from error
        if settle_max_attempts < 1:
            raise RuntimeError(
                f"invalid {SETTLE_MAX_ATTEMPTS_ENV_VAR} value:"
                f" {raw.get(SETTLE_MAX_ATTEMPTS_ENV_VAR)!r} (expected an int >= 1)"
            )
    if production:
        missing = [
            name
            for value, name in (
                (data_root_raw, DATA_ROOT_ENV_VAR),
                (api_key, API_KEY_ENV_VAR),
                (endpoint, META_FILES_BASE_URL_ENV_VAR),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"production mode refuses to start without: {', '.join(missing)}")
    if providers.mode == "live" and not api_key:
        raise RuntimeError(
            f"provider mode 'live' refuses to start without: {API_KEY_ENV_VAR}"
        )
    data_root = Path(data_root_raw) if data_root_raw else Path("var")
    max_video_s = _provider_float(raw, MAX_VIDEO_S_ENV_VAR, DEFAULT_MAX_VIDEO_S)
    chunk_s = _provider_float(raw, CHUNK_S_ENV_VAR, DEFAULT_CHUNK_S)
    chunk_overlap_s = _nonnegative_float(raw, CHUNK_OVERLAP_S_ENV_VAR, DEFAULT_CHUNK_OVERLAP_S)
    chunk_deadline_allowance_s = _timeout_setting(
        raw, CHUNK_DEADLINE_ALLOWANCE_S_ENV_VAR, DEFAULT_CHUNK_DEADLINE_ALLOWANCE_S,
    )
    if chunk_overlap_s >= chunk_s:
        raise ValueError("chunk_overlap_s must be below chunk_s")
    needed_extension_s = math.ceil(max_video_s / (chunk_s - chunk_overlap_s)) * chunk_deadline_allowance_s
    extension_ceiling_s = _provider_float(raw, DEADLINE_EXTENSION_MAX_S_ENV_VAR, needed_extension_s)
    if extension_ceiling_s < needed_extension_s:
        raise ValueError(f"{DEADLINE_EXTENSION_MAX_S_ENV_VAR} must be >= {needed_extension_s:g}")
    return Settings(
        data_root=data_root,
        db_path=(
            Path(db_path_raw) if db_path_raw else data_root / "partfidelity.sqlite3"
        ),
        production=production,
        meta_api_key=api_key,
        meta_files_base_url=endpoint,
        providers=providers,
        image_match=ImageMatchSettings(
            iou_cutoff=_unit_interval(raw, MATCH_IOU_ENV_VAR, DEFAULT_MATCH_IOU),
            ssim_cutoff=(_unit_interval(raw, MATCH_SSIM_ENV_VAR, DEFAULT_MATCH_SSIM_PARSE)
                         if raw.get(MATCH_SSIM_ENV_VAR, "").strip() else DEFAULT_MATCH_SSIM),
            noop_floor=_unit_interval(raw, MATCH_NOOP_ENV_VAR, DEFAULT_MATCH_NOOP),
            ecc_iterations=_provider_int(raw, MATCH_ECC_ITERATIONS_ENV_VAR, DEFAULT_MATCH_ECC_ITERATIONS),
            ecc_epsilon=_provider_float(raw, MATCH_ECC_EPSILON_ENV_VAR, DEFAULT_MATCH_ECC_EPSILON),
            ssim_window=_provider_int(raw, MATCH_SSIM_WINDOW_ENV_VAR, DEFAULT_MATCH_SSIM_WINDOW),
            ssim_sigma=_provider_float(raw, MATCH_SSIM_SIGMA_ENV_VAR, DEFAULT_MATCH_SSIM_SIGMA),
            small_sample_count=_provider_int(raw, MATCH_SMALL_SAMPLE_ENV_VAR, DEFAULT_MATCH_SMALL_SAMPLE),
            corruption_blur=_provider_int(raw, MATCH_CORRUPTION_BLUR_ENV_VAR, DEFAULT_MATCH_CORRUPTION_BLUR),
            corruption_shift=_provider_int(raw, MATCH_CORRUPTION_SHIFT_ENV_VAR, DEFAULT_MATCH_CORRUPTION_SHIFT),
            cases_filename=_provider_text(raw, MATCH_CASES_FILENAME_ENV_VAR, DEFAULT_MATCH_CASES_FILENAME),
        ),
        pipeline_mode=pipeline_mode,
        settle_max_attempts=settle_max_attempts,
        max_concurrent_jobs=_provider_int(
            raw, MAX_CONCURRENT_JOBS_ENV_VAR, DEFAULT_MAX_CONCURRENT_JOBS
        ),
        job_status_refresh_s=_provider_int(
            raw, JOB_STATUS_REFRESH_S_ENV_VAR, DEFAULT_JOB_STATUS_REFRESH_S
        ),
        meta_files_timeout_s=_timeout_setting(
            raw, META_FILES_TIMEOUT_ENV_VAR, DEFAULT_META_FILES_TIMEOUT_S
        ),
        meta_files_upload_timeout_s=_timeout_setting(
            raw,
            META_FILES_UPLOAD_TIMEOUT_ENV_VAR,
            DEFAULT_META_FILES_UPLOAD_TIMEOUT_S,
        ),
        meta_files_max_upload_bytes=_max_bytes_setting(
            raw,
            META_FILES_MAX_UPLOAD_BYTES_ENV_VAR,
            DEFAULT_META_FILES_MAX_UPLOAD_BYTES,
        ),
        meta_files_upload_purpose=_multipart_text(
            raw,
            META_FILES_UPLOAD_PURPOSE_ENV_VAR,
            DEFAULT_META_FILES_UPLOAD_PURPOSE,
        ),
        meta_files_purpose_field=_multipart_field_name(
            raw, META_FILES_PURPOSE_FIELD_ENV_VAR, DEFAULT_META_FILES_PURPOSE_FIELD
        ),
        meta_files_file_field=_multipart_field_name(
            raw, META_FILES_FILE_FIELD_ENV_VAR, DEFAULT_META_FILES_FILE_FIELD
        ),
        meta_files_file_content_type=_multipart_text(
            raw,
            META_FILES_FILE_CONTENT_TYPE_ENV_VAR,
            DEFAULT_META_FILES_FILE_CONTENT_TYPE,
        ),
        meta_files_expires_anchor_field=_multipart_field_name(
            raw,
            META_FILES_EXPIRES_ANCHOR_FIELD_ENV_VAR,
            DEFAULT_META_FILES_EXPIRES_ANCHOR_FIELD,
        ),
        meta_files_expires_seconds_field=_multipart_field_name(
            raw,
            META_FILES_EXPIRES_SECONDS_FIELD_ENV_VAR,
            DEFAULT_META_FILES_EXPIRES_SECONDS_FIELD,
        ),
        meta_files_expires_anchor=_multipart_text(
            raw,
            META_FILES_EXPIRES_ANCHOR_ENV_VAR,
            DEFAULT_META_FILES_EXPIRES_ANCHOR,
        ),
        meta_files_expiry_skew_s=_timeout_setting(
            raw,
            META_FILES_EXPIRY_SKEW_ENV_VAR,
            DEFAULT_META_FILES_EXPIRY_SKEW_S,
        ),
        meta_files_max_error_bytes=_max_bytes_setting(
            raw,
            META_FILES_MAX_ERROR_BYTES_ENV_VAR,
            DEFAULT_META_FILES_MAX_ERROR_BYTES,
        ),
        max_video_s=max_video_s,
        chunk_s=chunk_s,
        chunk_overlap_s=chunk_overlap_s,
        chunk_cut_timeout_s=_timeout_setting(
            raw, CHUNK_CUT_TIMEOUT_S_ENV_VAR, DEFAULT_CHUNK_CUT_TIMEOUT_S
        ),
        chunk_deadline_allowance_s=chunk_deadline_allowance_s,
        deadline_extension_max_s=extension_ceiling_s,
        deadline_extension_max_attempts=_provider_int(
            raw, DEADLINE_EXTENSION_MAX_ATTEMPTS_ENV_VAR,
            DEFAULT_DEADLINE_EXTENSION_MAX_ATTEMPTS,
        ),
        job_cost_cap_usd=_provider_price(
            raw, JOB_COST_CAP_ENV_VAR, DEFAULT_JOB_COST_CAP_USD
        ),
        chunk_max_attempts=_provider_int(
            raw, CHUNK_MAX_ATTEMPTS_ENV_VAR, DEFAULT_CHUNK_MAX_ATTEMPTS
        ),
        chunk_retry_budget_usd=_nonnegative_price(
            raw, CHUNK_RETRY_BUDGET_ENV_VAR, DEFAULT_CHUNK_RETRY_BUDGET_USD
        ),
        extract_retry_count=_extract_retry_count(raw),
        extract_retry_backoff_s=_nonnegative_float(raw, EXTRACT_RETRY_BACKOFF_ENV_VAR, DEFAULT_EXTRACT_RETRY_BACKOFF_S),
        extract_retry_min_remaining_s=_nonnegative_float(raw, EXTRACT_RETRY_MIN_REMAINING_ENV_VAR, DEFAULT_EXTRACT_RETRY_MIN_REMAINING_S),
        extract_retry_failures=frozenset(
            value.strip() for value in raw.get(EXTRACT_RETRY_FAILURES_ENV_VAR, ",".join(sorted(DEFAULT_EXTRACT_RETRY_FAILURES))).split(",") if value.strip()
        ),
        chunk_bundle_ttl_s=_timeout_setting(
            raw, CHUNK_BUNDLE_TTL_ENV_VAR, DEFAULT_CHUNK_BUNDLE_TTL_S
        ),
        step_span_mode=_span_mode_setting(raw),
        contiguity_tolerance_s=_nonnegative_float(
            raw, CONTIGUITY_TOLERANCE_ENV_VAR, DEFAULT_CONTIGUITY_TOLERANCE_S
        ),
        whole_video_max_s=_provider_float(
            raw, WHOLE_VIDEO_MAX_S_ENV_VAR, DEFAULT_WHOLE_VIDEO_MAX_S
        ),
        spark_input_tokens_per_video_s=_provider_float(
            raw,
            SPARK_INPUT_TOKENS_PER_VIDEO_S_ENV_VAR,
            DEFAULT_SPARK_INPUT_TOKENS_PER_VIDEO_S,
        ),
        clock_burn_in=_bool_setting(
            raw, CLOCK_BURN_IN_ENV_VAR, DEFAULT_CLOCK_BURN_IN
        ),
        scene_cut_threshold=_unit_interval(
            raw, SCENE_CUT_THRESHOLD_ENV_VAR, DEFAULT_SCENE_CUT_THRESHOLD
        ),
        scene_cuts_in_prompt=_bool_setting(
            raw, SCENE_CUTS_IN_PROMPT_ENV_VAR, DEFAULT_SCENE_CUTS_IN_PROMPT
        ),
        snap_window_s=_nonnegative_float(
            raw, SNAP_WINDOW_S_ENV_VAR, DEFAULT_SNAP_WINDOW_S
        ),
        min_phase_length_s=_provider_float(
            raw, MIN_PHASE_LENGTH_S_ENV_VAR, DEFAULT_MIN_PHASE_LENGTH_S
        ),
        spark_reservation_usd=_provider_price(
            raw, SPARK_RESERVATION_ENV_VAR, DEFAULT_SPARK_RESERVATION_USD
        ),
        merge_same_action=_bool_setting(
            raw, MERGE_SAME_ACTION_ENV_VAR, DEFAULT_MERGE_SAME_ACTION
        ),
        exclude_outro_phases=_bool_setting(
            raw, EXCLUDE_OUTRO_PHASES_ENV_VAR, DEFAULT_EXCLUDE_OUTRO_PHASES
        ),
    )
