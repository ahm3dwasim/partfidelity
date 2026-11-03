"""config/wait.toml loads strictly, and the wait screen reads only real job state."""
import sqlite3

import pytest

from app import wait_config, wait_view

SRC = wait_config.CONFIG.read_text()


def _load(tmp_path, text):
    path = tmp_path / "wait.toml"
    path.write_text(text)
    return wait_config.load(path)


def test_shipped_config_loads_with_named_stages_in_order():
    cfg = wait_config.load()
    assert [s["id"] for s in cfg["stages"]] == ["upload", "steps", "outline", "check", "guide"]
    assert cfg["page"]["log_lines"] == 4


@pytest.mark.parametrize(
    "old,new",
    [
        ("log_lines = 4", "log_linez = 4"),
        ("log_lines = 4", "log_lines = 0"),
        ("log_lines = 4", 'log_lines = "4"'),
        ('title = "Piecing it together."', 'title = ""'),
        ('receipts = ["extract"]', 'receipts = ["extrakt"]'),
        ('marker = "extract"', 'marker = "frames"'),
        ('marker = "readback"', 'marker = "segment"'),
        ('receipts = ["readback"]', 'receipts = ["upload"]'),
        ('id = "check"', 'id = "steps"'),
        ('log_done = "Guide ready"', ""),
    ],
)
def test_bad_config_fails_loudly(tmp_path, old, new):
    assert old in SRC
    with pytest.raises(wait_config.WaitConfigError):
        _load(tmp_path, SRC.replace(old, new, 1))


def test_bad_toml_fails_loudly(tmp_path):
    with pytest.raises(wait_config.WaitConfigError):
        _load(tmp_path, SRC + "\n[[[")


def _job(tmp_path, *, state="active", calls=(), confirmed="2026-10-30T10:00:00Z", completed=None):
    database = tmp_path / "pf.sqlite3"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE jobs (id TEXT, state TEXT, created_at TEXT, confirmed_at TEXT, completed_at TEXT)")
    conn.execute("CREATE TABLE model_calls (id INTEGER PRIMARY KEY, job_id TEXT, stage TEXT, status TEXT, created_at TEXT)")
    conn.execute("INSERT INTO jobs VALUES ('j1', ?, '2026-10-30T09:59:00Z', ?, ?)", (state, confirmed, completed))
    conn.executemany(
        "INSERT INTO model_calls (job_id, stage, status, created_at) VALUES ('j1', ?, ?, ?)", calls
    )
    conn.commit()
    conn.close()
    return database


NOW = wait_view.datetime.fromisoformat("2026-10-30T10:05:00+00:00")


def test_view_marks_stages_from_receipts_and_marker(tmp_path):
    database = _job(tmp_path, calls=[
        ("upload", "succeeded", "2026-10-30T10:00:20Z"),
        ("extract", "succeeded", "2026-10-30T10:01:20Z"),
    ])
    view = wait_view.build("j1", "active", "segment", database=database, data_root=tmp_path, now=NOW)
    assert [s.status for s in view.stages] == ["done", "done", "working", "waiting", "waiting"]
    assert [s.took_s for s in view.stages[:2]] == [20.0, 60.0]
    assert view.ended_at is None and view.elapsed_s == 300.0
    assert [(when, line) for _, when, line in view.log] == [
        ("10:00:20", "upload call succeeded"), ("10:00:20", "Clip received"),
        ("10:01:20", "extract call succeeded"), ("10:01:20", "Steps found"),
    ]
    # Nothing on disk yet: no frame counter, never a guessed one.
    assert (view.frames_done, view.frames_total) == (0, 0)


def test_view_counts_real_frames_and_masks(tmp_path):
    database = _job(tmp_path)
    frames = tmp_path / "jobs" / "j1" / "preview_frames"
    frames.mkdir(parents=True)
    for name in ("step1_t0.00s_0.png", "step1_t0.00s_0_fan_0.mask.png", "step1_t2.00s_1.png", "step2_t2.50s_0.png"):
        (frames / name).write_bytes(b"x")
    assert wait_view.frame_counts("j1", tmp_path) == (1, 3)


def test_stopped_job_clock_stops_at_last_activity(tmp_path):
    database = _job(tmp_path, state="failed", calls=[("extract", "failed", "2026-10-30T10:02:00Z")])
    view = wait_view.build("j1", "failed", None, database=database, data_root=tmp_path, now=NOW)
    assert view.stages[1].status == "stopped"
    assert view.elapsed_s == 120.0


def test_queued_job_is_never_shown_as_working(tmp_path):
    database = _job(tmp_path, state="waiting")
    view = wait_view.build("j1", "waiting", None, database=database, data_root=tmp_path, now=NOW)
    assert [s.status for s in view.stages] == ["queued"] + ["waiting"] * 4


def test_queued_job_with_upload_receipt_shows_upload_done_and_next_stage_queued(tmp_path):
    database = _job(tmp_path, state="waiting", calls=[("upload", "succeeded", "2026-10-30T10:00:20Z")])
    view = wait_view.build("j1", "waiting", None, database=database, data_root=tmp_path, now=NOW)
    assert [s.status for s in view.stages] == ["done", "queued", "waiting", "waiting", "waiting"]


def test_stage_without_receipts_gives_the_next_stage_no_duration(tmp_path):
    # Chunked extraction keeps its receipts on child jobs: "steps" has none here.
    database = _job(tmp_path, calls=[
        ("upload", "succeeded", "2026-10-30T10:00:20Z"),
        ("locate", "succeeded", "2026-10-30T10:03:00Z"),
    ])
    view = wait_view.build("j1", "active", "readback", database=database, data_root=tmp_path, now=NOW)
    took = {s.id: s.took_s for s in view.stages}
    assert took["upload"] == 20.0 and took["steps"] is None
    assert took["outline"] is None


def test_staged_temp_files_are_not_counted_as_frames(tmp_path):
    folder = wait_view.job_dir("j1", root=tmp_path) / "c_frames"
    folder.mkdir(parents=True)
    (folder / "step1_t0.00s_0.png").write_bytes(b"x")
    (folder / "step1_t0.00s_0.png.abcd.tmp.png").write_bytes(b"x")
    (folder / "step1_t0.00s_0_fan_0.mask.png.zz.tmp.png").write_bytes(b"x")
    assert wait_view.frame_counts("j1", tmp_path) == (0, 1)


def test_log_names_confirmation_only_when_recorded(tmp_path):
    database = _job(tmp_path, state="sending", confirmed=None)
    view = wait_view.build("j1", "sending", None, database=database, data_root=tmp_path, now=NOW)
    lines = [row[2] for row in view.log]
    assert "Clip confirmed" not in lines and "Job created" in lines
    database = _job(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, state="active")
    view = wait_view.build("j1", "active", None, database=database, data_root=tmp_path, now=NOW)
    assert "Clip confirmed" in [row[2] for row in view.log]


def test_failed_or_empty_segmentation_does_not_claim_parts_outlined(tmp_path):
    database = _job(tmp_path, calls=[
        ("segment", "failed", "2026-10-30T10:03:00Z"),
        ("readback", "succeeded", "2026-10-30T10:04:00Z"),
    ])
    view = wait_view.build("j1", "active", "readback", database=database, data_root=tmp_path, now=NOW)
    lines = [line for _, _, line in view.log]
    assert "segment call failed" in lines
    assert "Parts outlined" not in lines


def test_progress_stages_gloss_muse_spark_and_sam_on_first_use():
    from app import wait_config
    details = {s["id"]: s["detail"] for s in wait_config.load()["stages"]}
    assert details["steps"].startswith("Muse Spark (Meta's model that reads the video)")
    assert details["outline"].startswith("SAM 3.1 (Meta's model that outlines the part)")
