"""Settings smoke: health route, exact contract surface, settings refusal, ignore parity."""

import dataclasses
import inspect
import typing
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Mapping, Sequence

import app.contracts as contracts
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings, load_settings

EXPECTED_CONTRACTS = (
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

# Field (name, type) in declaration order, mirroring the frozen contract exactly.
EXPECTED_FIELDS = {
    "PartRequest": (
        ("phrase", str),
        ("position_hint", str),
    ),
    "LabelSlot": (
        ("step_index", int),
        ("part_index", int),
        ("part", contracts.PartRequest),
    ),
    "Step": (
        ("number", int),
        ("start_s", float),
        ("end_s", float),
        ("action", str),
        ("tools", tuple[str, ...]),
        ("parts", tuple[contracts.PartRequest, ...]),
        ("hazards", tuple[str, ...]),
        ("time_spread_s", float),
        ("time_needs_check", bool),
        ("kind", str),
    ),
    "ExcludedSpan": (
        ("start_s", float),
        ("end_s", float),
        ("kind", str),
    ),
    "StepBundle": (
        ("device_guess", str),
        ("confidence", float),
        ("steps", tuple[contracts.Step, ...]),
        ("source_runs", int),
        ("timestamps_approximate", bool),
        ("boundaries", tuple[contracts.BoundaryInfo, ...]),
        ("excluded_spans", tuple[contracts.ExcludedSpan, ...]),
    ),
    "BoundaryInfo": (
        ("time_s", float),
        ("support_runs", int),
        ("total_runs", int),
        ("nearest_cut_s", float | None),
        ("cut_distance_s", float | None),
        ("snapped", bool),
        ("snap_delta_s", float),
        ("needs_check", bool),
    ),
    "Detection": (
        ("phrase", str),
        ("mask_path", Path),
        ("score", float),
        ("box_xyxy", tuple[int, int, int, int]),
    ),
    "ProvisionalRender": (
        ("detection", contracts.Detection),
        ("artifact_id", contracts.ArtifactId),
        ("rendered_image_path", Path),
        ("position_hint", str),
    ),
    "ArtifactRecord": (
        ("id", contracts.ArtifactId),
        ("job_id", contracts.JobId),
        ("kind", str),
        ("relative_path", Path),
        ("expires_at", datetime),
    ),
    "MetaFileRecord": (
        ("id", contracts.MetaFileId),
        ("job_id", contracts.JobId),
        ("expires_at", datetime),
        ("delete_state", str),
    ),
    "LabelDecision": (
        ("slot", contracts.LabelSlot),
        ("detection", contracts.Detection | None),
        ("provisional", contracts.ProvisionalRender | None),
        ("checked", bool),
        ("status", Literal["part_pixels_unchanged", "not_checked"]),
        ("readback", Literal["yes", "no", "not_run"]),
    ),
    "GuideHazardAssessment": (
        ("difficulty", Literal["Easy", "Medium", "Hard"]),
        ("verdict", str),
        ("draft_reasons", tuple[str, ...]),
    ),
    "Guide": (
        ("job_id", contracts.JobId),
        ("steps", tuple[contracts.Step, ...]),
        ("frame_paths", tuple[Path, ...]),
        ("labels", tuple[contracts.LabelDecision, ...]),
        ("difficulty", Literal["Easy", "Medium", "Hard"]),
        ("verdict", str),
        ("draft_reasons", tuple[str, ...]),
        ("excluded_spans", tuple[contracts.ExcludedSpan, ...]),
    ),
    "CallRecord": (
        ("job_id", contracts.JobId),
        ("stage", str),
        ("model", str),
        ("endpoint", str),
        ("latency_ms", int),
        ("units_in", int | None),
        ("units_out", int | None),
        ("estimated_cost_usd", Decimal),
        ("status", str),
        ("evidence", Mapping[str, contracts.JSONValue]),
    ),
    "UploadPreview": (
        ("job_id", contracts.JobId),
        ("local_video", Path),
        ("duration_s", float),
        ("size_bytes", int),
        ("audio_removed", bool),
        ("confirmed_at", datetime | None),
    ),
    "UploadRecord": (
        ("preview", contracts.UploadPreview),
        ("preview_artifact_id", contracts.ArtifactId),
        ("preview_expires_at", datetime),
        (
            "state",
            Literal["preview", "confirming", "confirmed", "cancelling", "cancelled", "failed"],
        ),
        ("consented_at", datetime | None),
        ("claim_id", str | None),
        ("claim_started_at", datetime | None),
        ("guard_deadline_utc", datetime | None),
        ("meta_file_id", contracts.MetaFileId | None),
        ("meta_expires_at", datetime | None),
        ("confirmed_at", datetime | None),
        ("cancel_requested_at", datetime | None),
        ("hidden", bool),
        ("failure_count", int),
        ("last_error", str | None),
    ),
    "ConfirmationClaim": (
        ("record", contracts.UploadRecord),
        ("acquired", bool),
    ),
    "DeleteRequestRecord": (
        ("id", contracts.DeleteRequestId),
        ("job_id", contracts.JobId | None),
        ("contact", str),
        ("requested_at", datetime),
        ("expires_at", datetime),
    ),
    "MetaUpload": (
        ("file_id", contracts.MetaFileId),
        ("expires_at", datetime),
    ),
    "GlimmerProof": (
        ("model_id", str),
        ("licence", str),
        ("quantisation", str),
        ("peak_ram_bytes", int),
        ("latency_ms", int),
        ("platform", Literal["mac", "vm_cpu"]),
        ("graded_image", bool),
    ),
}

# Module-function stubs: parameter names in order, keyword-only subset, defaults,
# and raw annotation strings, mirroring the frozen contract exactly. Stubs that name
# module-owned shapes (HazardAssessment, DeletionResult, SweepResult) keep those
# names unresolved: contracts declares the signature, never the shape.
EXPECTED_LANE_FUNCTIONS = {
    "read_back_label": (
        ["ctx", "provisional", "slot", "client"],
        ("slot", "client"),
        {},
        {
            "ctx": "JobContext",
            "provisional": "ProvisionalRender",
            "slot": "LabelSlot",
            "client": "SparkClient",
            "return": "LabelDecision",
        },
    ),
    "assemble_guide": (
        ["ctx", "steps", "frames", "labels", "hazards", "excluded_spans"],
        ("hazards", "excluded_spans"),
        {"hazards": None, "excluded_spans": ()},
        {
            "ctx": "JobContext",
            "steps": "Sequence[Step]",
            "frames": "Sequence[Path]",
            "labels": "Sequence[LabelDecision]",
            "hazards": "GuideHazardAssessment | None",
            "excluded_spans": "Sequence[ExcludedSpan]",
            "return": "Guide",
        },
    ),
    "guide_hazard_assessment": (
        ["steps", "assessment"],
        (),
        {},
        {
            "steps": "Sequence[Step]",
            "assessment": "HazardAssessment",
            "return": "GuideHazardAssessment",
        },
    ),
    "prepare_preview": (
        ["ctx", "upload", "declared_name", "uploads"],
        ("uploads",),
        {},
        {
            "ctx": "JobContext",
            "upload": "BinaryIO",
            "declared_name": "str",
            "uploads": "UploadStore",
            "return": "UploadPreview",
        },
    ),
    "confirm_and_send": (
        ["job_id", "uploads", "client"],
        ("uploads", "client"),
        {},
        {
            "job_id": "JobId",
            "uploads": "UploadStore",
            "client": "MetaFilesClient",
            "return": "MetaFileId",
        },
    ),
    "cancel_and_delete": (
        ["job_id", "uploads", "lifecycle", "client"],
        ("uploads", "lifecycle", "client"),
        {},
        {
            "job_id": "JobId",
            "uploads": "UploadStore",
            "lifecycle": "LifecycleRepository",
            "client": "MetaFilesClient",
            "return": "DeletionResult",
        },
    ),
    "sweep_expired": (
        ["now", "uploads", "lifecycle", "client"],
        ("uploads", "lifecycle", "client"),
        {},
        {
            "now": "datetime",
            "uploads": "UploadStore",
            "lifecycle": "LifecycleRepository",
            "client": "MetaFilesClient",
            "return": "SweepResult",
        },
    ),
}

# Sensitive .gitignore lines that must also appear in .dockerignore so ignored
# local secrets and private artifacts never enter the Docker build context.
SENSITIVE_GITIGNORE_PATTERNS = (
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    ".secrets/",
    "footage/",
    "raw/",
    "inputs/",
    "incoming/",
    "consent*",
    "*consent*",
    "var/",
    "*.db",
    "*.sqlite3",
    "*.sqlite3-journal",
    "*.sqlite3-wal",
    "*.sqlite3-shm",
    "artifacts/",
    "out/",
    "renders/",
)


def test_healthz_ok_with_temp_data_root_and_no_key(tmp_path):
    settings = Settings(data_root=tmp_path, db_path=tmp_path / "test.sqlite3", production=False)
    assert settings.meta_api_key is None
    response = TestClient(create_app(settings)).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_contract_exported_symbol_set_is_exact():
    assert tuple(contracts.__all__) == EXPECTED_CONTRACTS
    for name in EXPECTED_CONTRACTS:
        assert hasattr(contracts, name), name


def test_contract_dataclass_fields_match_interfaces():
    for cls_name, expected in EXPECTED_FIELDS.items():
        cls = getattr(contracts, cls_name)
        assert dataclasses.is_dataclass(cls), cls_name
        assert cls.__dataclass_params__.frozen is True, cls_name
        assert [f.name for f in dataclasses.fields(cls)] == [n for n, _ in expected], cls_name
        hints = typing.get_type_hints(cls)
        for field_name, field_type in expected:
            assert hints[field_name] == field_type, (cls_name, field_name)


def test_contract_protocols_available():
    for proto in (
        contracts.SparkClient,
        contracts.SamClient,
        contracts.MetaFilesClient,
        contracts.ArtifactStore,
        contracts.CallReservation,
        contracts.JobContext,
        contracts.LifecycleRepository,
        contracts.UploadStore,
    ):
        assert issubclass(proto, typing.Protocol), proto


def _assert_protocol_methods_match(proto, expected, *, check_members=True):
    if check_members:
        assert {m for m in vars(proto) if not m.startswith("_")} == set(expected)
    for name, (params, hints, returns) in expected.items():
        method = getattr(proto, name)
        assert list(inspect.signature(method).parameters) == params, (proto, name)
        resolved = typing.get_type_hints(method)
        assert resolved.pop("return") == returns, (proto, name)
        assert resolved == hints, (proto, name)


def test_artifact_store_signatures_match_interfaces():
    _assert_protocol_methods_match(
        contracts.ArtifactStore,
        {
            "register": (
                ["self", "kind", "path"],
                {"kind": str, "path": Path},
                contracts.ArtifactId,
            ),
            "mark_kept": (
                ["self", "artifact_id"],
                {"artifact_id": contracts.ArtifactId},
                type(None),
            ),
            "mark_deleted": (
                ["self", "artifact_id"],
                {"artifact_id": contracts.ArtifactId},
                type(None),
            ),
        },
    )


def test_call_reservation_signatures_match_interfaces():
    _assert_protocol_methods_match(
        contracts.CallReservation,
        {
            "settle": (
                ["self", "actual_usd", "outcome"],
                {
                    "actual_usd": Decimal,
                    "outcome": Literal["succeeded", "failed"],
                },
                type(None),
            ),
        },
    )


def test_job_context_signatures_match_interfaces():
    assert {m for m in vars(contracts.JobContext) if not m.startswith("_")} == {
        "job_id",
        "artifacts",
        "reserve",
        "record_call_with_event",
        "finish",
    }
    for prop_name, returns in (
        ("job_id", contracts.JobId),
        ("artifacts", contracts.ArtifactStore),
    ):
        prop = getattr(contracts.JobContext, prop_name)
        assert isinstance(prop, property), prop_name
        resolved = typing.get_type_hints(prop.fget)
        assert resolved == {"return": returns}, prop_name
    _assert_protocol_methods_match(
        contracts.JobContext,
        {
            "reserve": (
                ["self", "kind", "worst_case_usd"],
                {"kind": str, "worst_case_usd": Decimal},
                contracts.CallReservation,
            ),
            "record_call_with_event": (
                ["self", "key", "call", "event"],
                {
                    "key": str,
                    "call": contracts.CallRecord,
                    "event": tuple[str, Mapping[str, contracts.JSONValue]],
                },
                type(None),
            ),
            "finish": (
                ["self", "outcome", "keep_artifact_ids", "failed_output_names"],
                {
                    "outcome": Literal["done", "failed"],
                    "keep_artifact_ids": Sequence[contracts.ArtifactId],
                    "failed_output_names": tuple[str, str] | None,
                },
                type(None),
            ),
        },
        check_members=False,
    )
    params = inspect.signature(contracts.JobContext.finish).parameters
    assert params["keep_artifact_ids"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["keep_artifact_ids"].default == ()
    assert params["failed_output_names"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["failed_output_names"].default is None


def test_lifecycle_repository_signatures_match_interfaces():
    _assert_protocol_methods_match(
        contracts.LifecycleRepository,
        {
            "list_due_artifacts": (
                ["self", "now"],
                {"now": datetime},
                tuple[contracts.ArtifactRecord, ...],
            ),
            "list_due_meta_files": (
                ["self", "now"],
                {"now": datetime},
                tuple[contracts.MetaFileRecord, ...],
            ),
            "record_artifact_deletion": (
                ["self", "artifact_id", "outcome", "error"],
                {
                    "artifact_id": contracts.ArtifactId,
                    "outcome": Literal["tombstoned", "retry"],
                    "error": str | None,
                },
                type(None),
            ),
            "record_meta_file_deletion": (
                ["self", "meta_file_id", "outcome", "error"],
                {
                    "meta_file_id": contracts.MetaFileId,
                    "outcome": Literal["verified_absent", "acknowledged_only", "retry"],
                    "error": str | None,
                },
                type(None),
            ),
        },
    )
    for name in ("record_artifact_deletion", "record_meta_file_deletion"):
        params = inspect.signature(getattr(contracts.LifecycleRepository, name)).parameters
        assert params["error"].default is None, name


def test_meta_files_client_signatures_match_interfaces():
    _assert_protocol_methods_match(
        contracts.MetaFilesClient,
        {
            "upload_file": (
                ["self", "path", "expires_after_hours", "idempotency_key"],
                {"path": Path, "expires_after_hours": int, "idempotency_key": str},
                contracts.MetaUpload,
            ),
            "find_upload": (
                ["self", "idempotency_key"],
                {"idempotency_key": str},
                contracts.MetaUpload | None,
            ),
            "delete_file": (
                ["self", "file_id"],
                {"file_id": contracts.MetaFileId},
                type(None),
            ),
            "get_file": (
                ["self", "file_id"],
                {"file_id": contracts.MetaFileId},
                contracts.MetaUpload | None,
            ),
        },
    )
    params = inspect.signature(contracts.MetaFilesClient.upload_file).parameters
    assert params["expires_after_hours"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["idempotency_key"].kind is inspect.Parameter.KEYWORD_ONLY


def test_meta_upload_error_surface_matches_interfaces():
    assert issubclass(contracts.MetaFilesError, Exception)
    assert issubclass(contracts.MetaReadUnsupported, contracts.MetaFilesError)
    assert issubclass(contracts.MetaUploadExpiryError, contracts.MetaFilesError)
    assert typing.get_type_hints(contracts.MetaUploadExpiryError) == {
        "file_id": contracts.MetaFileId,
        "expires_at": datetime | None,
    }
    init = contracts.MetaUploadExpiryError.__init__
    assert list(inspect.signature(init).parameters) == ["self", "file_id", "expires_at"]
    assert typing.get_type_hints(init) == {
        "file_id": contracts.MetaFileId,
        "expires_at": datetime | None,
        "return": type(None),
    }


def test_upload_store_signatures_match_interfaces():
    _assert_protocol_methods_match(
        contracts.UploadStore,
        {
            "register_preview": (
                ["self", "ctx", "preview", "artifact_id", "expires_at"],
                {
                    "ctx": contracts.JobContext,
                    "preview": contracts.UploadPreview,
                    "artifact_id": contracts.ArtifactId,
                    "expires_at": datetime,
                },
                contracts.UploadRecord,
            ),
            "load_job": (
                ["self", "job_id"],
                {"job_id": contracts.JobId},
                tuple[contracts.JobContext, contracts.UploadRecord] | None,
            ),
            "claim_confirmation": (
                ["self", "job_id", "consented_at"],
                {"job_id": contracts.JobId, "consented_at": datetime},
                contracts.ConfirmationClaim,
            ),
            "claim_chunk_upload": (
                ["self", "job_id", "claim_id", "artifact_id", "duration_s", "size_bytes", "now"],
                {
                    "job_id": contracts.JobId,
                    "claim_id": str,
                    "artifact_id": contracts.ArtifactId,
                    "duration_s": float,
                    "size_bytes": int,
                    "now": datetime,
                },
                contracts.UploadRecord,
            ),
            "chunk_claim_taken": (
                ["self", "claim_id"],
                {"claim_id": str},
                bool,
            ),
            "load_chunk_claim": (
                ["self", "claim_id"],
                {"claim_id": str},
                tuple[contracts.JobId, contracts.MetaFileId | None] | None,
            ),
            "complete_confirmation": (
                ["self", "claim_id", "meta_file_id", "expires_at", "request_time_utc", "now"],
                {
                    "claim_id": str,
                    "meta_file_id": contracts.MetaFileId,
                    "expires_at": datetime,
                    "request_time_utc": datetime,
                    "now": datetime,
                },
                contracts.UploadRecord,
            ),
            "request_cancellation": (
                ["self", "job_id", "now"],
                {"job_id": contracts.JobId, "now": datetime},
                contracts.UploadRecord,
            ),
            "record_failure": (
                ["self", "job_id", "failure_key", "stage", "error", "now", "meta_file_id", "meta_expires_at"],
                {
                    "job_id": contracts.JobId,
                    "failure_key": str,
                    "stage": Literal[
                        "meta_upload", "meta_persist", "local_delete", "meta_delete", "recovery"
                    ],
                    "error": str,
                    "now": datetime,
                    "meta_file_id": contracts.MetaFileId | None,
                    "meta_expires_at": datetime | None,
                },
                contracts.UploadRecord,
            ),
            "list_recoverable": (
                ["self", "now"],
                {"now": datetime},
                tuple[contracts.UploadRecord, ...],
            ),
            "recover_job": (
                ["self", "job_id", "now", "no_match_claim_id"],
                {
                    "job_id": contracts.JobId,
                    "now": datetime,
                    "no_match_claim_id": str | None,
                },
                contracts.UploadRecord,
            ),
            "public_uploads_blocked": (
                ["self"],
                {},
                bool,
            ),
            "submit_delete_request": (
                ["self", "job_id", "contact", "now"],
                {"job_id": contracts.JobId | None, "contact": str, "now": datetime},
                contracts.DeleteRequestId,
            ),
            "list_delete_requests": (
                ["self", "limit"],
                {"limit": int},
                tuple[contracts.DeleteRequestRecord, ...],
            ),
            "resolve_delete_request": (
                ["self", "request_id"],
                {"request_id": contracts.DeleteRequestId},
                type(None),
            ),
            "purge_delete_requests": (
                ["self", "now"],
                {"now": datetime},
                int,
            ),
        },
    )
    failure_params = inspect.signature(contracts.UploadStore.record_failure).parameters
    for name in ("meta_file_id", "meta_expires_at"):
        assert failure_params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert failure_params[name].default is None, name
    recover_params = inspect.signature(contracts.UploadStore.recover_job).parameters
    assert recover_params["no_match_claim_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert recover_params["no_match_claim_id"].default is None


def test_guide_hazard_assessment_fail_closed_defaults():
    assert contracts.GuideHazardAssessment() == contracts.GuideHazardAssessment(
        difficulty="Hard",
        verdict="DRAFT",
        draft_reasons=("Hazard assessment unavailable.",),
    )


def test_lane_function_signatures_match_interfaces():
    for name, (params, keyword_only, defaults, annotations) in EXPECTED_LANE_FUNCTIONS.items():
        func = getattr(contracts, name)
        assert inspect.isfunction(func), name
        sig = inspect.signature(func)
        assert list(sig.parameters) == params, name
        assert func.__annotations__ == annotations, name
        for pname in params:
            param = sig.parameters[pname]
            if pname in keyword_only:
                assert param.kind is inspect.Parameter.KEYWORD_ONLY, (name, pname)
            else:
                assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (name, pname)
            if pname in defaults:
                assert param.default == defaults[pname], (name, pname)
            else:
                assert param.default is inspect.Parameter.empty, (name, pname)


def test_lane_function_stubs_resolve_contract_types():
    assert typing.get_type_hints(contracts.read_back_label) == {
        "ctx": contracts.JobContext,
        "provisional": contracts.ProvisionalRender,
        "slot": contracts.LabelSlot,
        "client": contracts.SparkClient,
        "return": contracts.LabelDecision,
    }
    assert typing.get_type_hints(contracts.assemble_guide) == {
        "ctx": contracts.JobContext,
        "steps": Sequence[contracts.Step],
        "frames": Sequence[Path],
        "labels": Sequence[contracts.LabelDecision],
        "hazards": contracts.GuideHazardAssessment | None,
        "excluded_spans": Sequence[contracts.ExcludedSpan],
        "return": contracts.Guide,
    }


def test_lane_function_stubs_are_declarations_only():
    sentinel = object()
    assert contracts.read_back_label(sentinel, sentinel, slot=sentinel, client=sentinel) is None
    assert contracts.assemble_guide(sentinel, sentinel, sentinel, sentinel) is None
    assert contracts.assemble_guide(sentinel, sentinel, sentinel, sentinel, hazards=sentinel) is None
    assert contracts.guide_hazard_assessment(sentinel, sentinel) is None
    assert contracts.prepare_preview(sentinel, sentinel, sentinel, uploads=sentinel) is None
    assert contracts.confirm_and_send(sentinel, uploads=sentinel, client=sentinel) is None
    assert (
        contracts.cancel_and_delete(sentinel, uploads=sentinel, lifecycle=sentinel, client=sentinel)
        is None
    )
    assert (
        contracts.sweep_expired(sentinel, uploads=sentinel, lifecycle=sentinel, client=sentinel)
        is None
    )


def test_contract_constants_match_interfaces():
    assert contracts.SPARK_RUNS_PER_GUIDE == 3
    assert contracts.TIME_SPREAD_FLAG_S == 5
    assert contracts.MAX_STEPS == 12
    assert contracts.MAX_PARTS_PER_STEP == 3
    assert contracts.MAX_READBACKS_PER_JOB == 3
    assert contracts.JOB_DEADLINE_S == 240
    assert contracts.SAM_MAX_REQUEST_STARTS_PER_MINUTE == 55
    assert dict(contracts.PART_CHECK_THRESHOLDS) == {"fan": 0.6, "battery": 0.5}
    assert contracts.JOB_COST_CAP_USD == Decimal("0.50")
    assert contracts.AGGREGATE_SPEND_CEILING_USD == Decimal("100")
    assert contracts.AGGREGATE_SPEND_MIN_UNSPENT_USD == Decimal("50")
    assert contracts.UPLOAD_PREVIEW_TTL_HOURS == 24
    assert contracts.META_UPLOAD_MAX_HOURS == 24
    assert contracts.META_EXPIRY_GUARD_MARGIN_HOURS == 1
    assert contracts.CONFIRMATION_CLAIM_TIMEOUT_S == 120
    assert contracts.UPLOAD_FAILURE_BLOCK_THRESHOLD == 3
    assert contracts.UPLOAD_ERROR_MAX_CHARS == 512
    assert contracts.DELETE_REQUEST_MAX_PENDING == 1000
    assert contracts.DELETE_REQUEST_RETENTION_DAYS == 30
    assert contracts.DELETE_REQUEST_CONTACT_MAX_CHARS == 254
    assert contracts.DELETE_REQUEST_PAGE_SIZE == 100
    assert contracts.SPARK_3RUNS_RESERVATION_USD == Decimal("0.26")
    assert contracts.SAM_MAX_CALLS_PER_JOB == 32
    assert contracts.SAM_COST_PER_CALL_USD == Decimal("0.0025")
    assert contracts.SAM_MAX_RESERVATION_USD == Decimal("0.08")
    assert contracts.READBACK_COST_PER_CALL_USD == Decimal("0.04")
    assert contracts.READBACK_MAX_RESERVATION_USD == Decimal("0.12")
    assert contracts.JOB_RESERVATION_TOTAL_USD == Decimal("0.46")
    assert contracts.JOB_RESERVATION_HEADROOM_USD == Decimal("0.04")
    assert contracts.READBACK_CONFIDENCE_CUTOFF == 0.70
    for name in (
        "JOB_COST_CAP_USD",
        "SPARK_3RUNS_RESERVATION_USD",
        "SAM_COST_PER_CALL_USD",
        "SAM_MAX_RESERVATION_USD",
        "READBACK_COST_PER_CALL_USD",
        "READBACK_MAX_RESERVATION_USD",
        "JOB_RESERVATION_TOTAL_USD",
        "JOB_RESERVATION_HEADROOM_USD",
    ):
        assert isinstance(getattr(contracts, name), Decimal), name
    assert (
        contracts.SAM_MAX_CALLS_PER_JOB * contracts.SAM_COST_PER_CALL_USD
        <= contracts.SAM_MAX_RESERVATION_USD
    )
    assert (
        contracts.MAX_READBACKS_PER_JOB * contracts.READBACK_COST_PER_CALL_USD
        == contracts.READBACK_MAX_RESERVATION_USD
    )
    assert (
        contracts.SPARK_3RUNS_RESERVATION_USD
        + contracts.SAM_MAX_RESERVATION_USD
        + contracts.READBACK_MAX_RESERVATION_USD
        == contracts.JOB_RESERVATION_TOTAL_USD
    )
    assert (
        contracts.JOB_RESERVATION_TOTAL_USD + contracts.JOB_RESERVATION_HEADROOM_USD
        == contracts.JOB_COST_CAP_USD
    )
    assert (
        contracts.META_UPLOAD_MAX_HOURS + contracts.META_EXPIRY_GUARD_MARGIN_HOURS
        == contracts.UPLOAD_PREVIEW_TTL_HOURS + contracts.META_EXPIRY_GUARD_MARGIN_HOURS
        == 25
    )


def test_dockerignore_mirrors_sensitive_gitignore_patterns():
    repo_root = Path(contracts.__file__).resolve().parents[1]
    gitignore = (repo_root / ".gitignore").read_text().splitlines()
    dockerignore = (repo_root / ".dockerignore").read_text().splitlines()
    for pattern in SENSITIVE_GITIGNORE_PATTERNS:
        assert pattern in gitignore, pattern
        assert pattern in dockerignore, pattern


def test_production_refuses_missing_paths_and_key():
    with pytest.raises(RuntimeError):
        load_settings({"PARTFIDELITY_PRODUCTION": "1"})
    with pytest.raises(RuntimeError):
        load_settings({"PARTFIDELITY_PRODUCTION": "1", "META_API_KEY": "k"})
    with pytest.raises(RuntimeError):
        load_settings({"PARTFIDELITY_PRODUCTION": "1", "PARTFIDELITY_DATA_ROOT": "/tmp/x"})


def test_production_rejects_unknown_flag_value():
    with pytest.raises(RuntimeError, match="invalid PARTFIDELITY_PRODUCTION"):
        load_settings({"PARTFIDELITY_PRODUCTION": "treu"})
    with pytest.raises(RuntimeError, match="invalid PARTFIDELITY_PRODUCTION"):
        load_settings({"PARTFIDELITY_PRODUCTION": "2"})


def test_production_rejects_whitespace_only_key():
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        load_settings(
            {
                "PARTFIDELITY_PRODUCTION": "1",
                "PARTFIDELITY_DATA_ROOT": "/tmp/x",
                "META_API_KEY": "   ",
            }
        )


def test_flag_values_parse_case_insensitively_and_key_is_stripped():
    for value in ("", "0", "false", "no", "FALSE", " No "):
        assert load_settings({"PARTFIDELITY_PRODUCTION": value}).production is False
    for value in ("1", "true", "yes", "TRUE", " Yes "):
        with pytest.raises(RuntimeError, match="PARTFIDELITY_DATA_ROOT"):
            load_settings({"PARTFIDELITY_PRODUCTION": value})
    assert load_settings({"META_API_KEY": "  k  "}).meta_api_key == "k"


def test_create_app_refuses_production_without_paths(monkeypatch):
    monkeypatch.setenv("PARTFIDELITY_PRODUCTION", "1")
    with pytest.raises(RuntimeError):
        create_app()
