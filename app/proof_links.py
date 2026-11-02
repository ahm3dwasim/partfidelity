"""Validated destinations for the recorded sample's proof links."""
from pathlib import Path
import re
import tomllib

KEYS = {"sample_calls", "hard_stop", "catch_cta", "meta_roles", "meta_roles_link", "meta_roles_catch_link", "meta_roles_built_link", "open_weights_job", "open_weights_run_date", "open_weights_calls", "open_weights_replay", "open_weights_note", "open_weights_calls_text", "open_weights_replay_text", "open_weights_second_check_replay_text", "sample_reason_from", "sample_reason_to", "open_weights_fixture_sha256", "open_weights_receipt_count", "open_weights_real_second_check_min"}
# Each role is a distinct phrase that must stand alone: not followed by a word character or ".digit", so
# "SAM 3" is not satisfied by "SAM 3.1" and "Muse Spark" is not satisfied by "Muse Spark Open".
ROLE_PATTERNS = tuple(re.compile(r"(?<![\w.])" + re.escape(t) + r"(?![\w]|\.\d)") for t in (
    "Muse Spark steps", "SAM 3.1 part boxes", "SAM 3 contours", "Muse Spark Open", "Muse Image"))


def load():
    with (Path(__file__).resolve().parents[1] / "config/proof_links.toml").open("rb") as source:
        data = tomllib.load(source)
    # open_weights_run_date may be "" (no date in the saved evidence); every other value is non-empty text.
    ints = ("open_weights_receipt_count", "open_weights_real_second_check_min")
    if set(data) != KEYS or any(not isinstance(data[k], int) or isinstance(data[k], bool) or data[k] < 1 for k in ints) or any(
            not isinstance(v, str) or (k != "open_weights_run_date" and not v.strip())
            for k, v in data.items() if k not in ints):
        raise ValueError("invalid sample proof link configuration")
    if data["open_weights_run_date"] and not re.fullmatch(r"20[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])", data["open_weights_run_date"]):
        raise ValueError("invalid sample proof link configuration: open_weights_run_date must be YYYY-MM-DD or empty")
    if not re.fullmatch(r"/calls/[0-9a-f]{32}", data["sample_calls"]) or not re.fullmatch(r"/guide/sample#step-[1-9][0-9]*", data["hard_stop"]):
        raise ValueError("invalid sample proof link configuration")
    if not re.fullmatch(r"[0-9a-f]{32}", data["open_weights_job"]) or any("verified" in data[k].lower() for k in KEYS - set(ints)):
        raise ValueError("invalid sample proof link configuration: open_weights_job must be a job id and no text may say verified")
    if not re.fullmatch(r"[0-9a-f]{64}", data["open_weights_fixture_sha256"]):
        raise ValueError("invalid sample proof link configuration: open_weights_fixture_sha256 must be a sha256 hex digest")
    if data["open_weights_calls"] != f"/calls/{data['open_weights_job']}" or not re.fullmatch(r"/[a-z][a-z-]*", data["open_weights_replay"]):
        raise ValueError("invalid sample proof link configuration: open_weights_calls must be the job's calls path")
    if not all(p.search(data["meta_roles"]) for p in ROLE_PATTERNS) or "verified" in data["meta_roles"].lower():
        raise ValueError("invalid sample proof link configuration: meta_roles must name every model and never say verified")
    return data


def open_weights_note_html() -> str:
    """The sample's open-weights line: what it lacks, and links to the newer job's receipts and the replay."""
    import html
    links = load()
    return (
        f'<p class="gd-open-weights" role="note">{html.escape(links["open_weights_note"])} '
        f'<a href="{links["open_weights_calls"]}">{html.escape(links["open_weights_calls_text"])}</a> &#183; '
        f'<a href="{links["open_weights_replay"]}">{html.escape(links["open_weights_replay_text"])}</a></p>'
    )


def sample_wording(text: str) -> str:
    """Sample-only wording: the stored draft reason says "your video", but the visitor did not film the sample."""
    links = load()
    return text.replace(links["sample_reason_from"], links["sample_reason_to"])


def fixture_id_prefix(job_id: str) -> str:
    return f"receipts:{job_id}:"


def receipts_source(database, job_id: str) -> str:
    """"fixture" when every receipt of the job is the shipped copy, "real" when any row is not, "" when it has none."""
    from app import db
    with db.connect(database) as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM model_calls WHERE job_id = ?", (job_id,))]
    if not ids:
        return ""
    prefix = fixture_id_prefix(job_id)
    return "fixture" if all(i.startswith(prefix) for i in ids) else "real"


def is_saved_copy(job_id: str, database=None) -> bool:
    """True only for the shipped job while its receipts are the installed fixture, never for a real run of it."""
    return job_id == load()["open_weights_job"] and database is not None and receipts_source(database, job_id) == "fixture"


def saved_copy_line(job_id: str, database=None) -> str:
    """One provenance line for the shipped job: its rows were installed from saved evidence, not made at boot."""
    if not is_saved_copy(job_id, database):
        return ""
    date = load()["open_weights_run_date"]
    return "Saved copy of a recorded run" + (f" from {date}" if date else "")
