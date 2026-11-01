"""Local frame sampling: 2 fps grids, sharpness/motion rank, top two kept."""

import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.contracts import Step
from app.frame_scoring import (
    ScoredFrame,
    combined_score,
    frame_motion,
    laplacian_sharpness,
    pick_best_two,
    sample_times,
    score_frames,
)

WIDTH, HEIGHT, FPS = 64, 48, 10.0


def checker_bgr() -> np.ndarray:
    tile = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    tile[::8, ::8] = 255
    tile[4::8, 4::8] = 255
    return tile


def solid_bgr() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 128, np.uint8)


def make_video(path: Path, segments) -> Path:
    """Write an MJPG video; segments are (frame, count) pairs."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    assert writer.isOpened()
    for frame, count in segments:
        for _ in range(count):
            writer.write(frame)
    writer.release()
    return path


def make_step(number: int, start_s: float, end_s: float) -> Step:
    return Step(
        number=number,
        start_s=start_s,
        end_s=end_s,
        action="open the panel",
        tools=(),
        parts=(),
        hazards=(),
        time_spread_s=0.0,
        time_needs_check=False,
    )


def stamp_of(path: Path) -> float:
    return float(path.name.split("_t")[1].split("s_")[0])


def test_score_frames_signature_matches_interfaces():
    assert list(inspect.signature(score_frames).parameters) == [
        "video_path",
        "steps",
    ]


def test_two_best_frames_by_sharpness_and_motion(tmp_path):
    video = make_video(
        tmp_path / "clip.avi",
        [(checker_bgr(), 20), (solid_bgr(), 20)],
    )
    paths = score_frames(video, [make_step(1, 0.0, 4.0)])[1]
    assert len(paths) == 2
    for path in paths:
        assert path.is_file()
        assert cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is not None
    assert all(stamp_of(p) < 2.0 for p in paths)
    still = ScoredFrame(timestamp_s=0.5, sharpness=100.0, motion=0.0)
    shaky = ScoredFrame(timestamp_s=0.0, sharpness=100.0, motion=50.0)
    flat = ScoredFrame(timestamp_s=1.0, sharpness=1.0, motion=0.0)
    assert pick_best_two([shaky, still, flat]) == (still, shaky)


def test_no_hand_detector_without_fixture_trigger():
    source = (
        Path(__file__).resolve().parent.parent
        / "app"
        / "frame_scoring.py"
    ).read_text(encoding="utf-8")
    assert "hand" not in source.lower()
    fixtures = Path(__file__).resolve().parent / "fixtures"
    assert [p for p in fixtures.iterdir() if "hand" in p.name.lower()] == []


def test_sampler_never_exceeds_step_window():
    cases = [
        (0.0, 10.0),
        (2.3, 2.3),
        (0.0, 0.4),
        (0.0, 0.5),
        (5.1, 7.77),
        (0.25, 3.75),
    ]
    for start, end in cases:
        stamps = sample_times(start, end)
        assert len(stamps) == int((end - start) * 2.0) + 1
        assert all(start <= t <= end for t in stamps), (start, end, stamps)
    with pytest.raises(ValueError):
        sample_times(3.0, 2.0)
    with pytest.raises(ValueError):
        sample_times(0.0, 1.0, fps=0.0)


def test_sample_times_two_fps_grid():
    assert sample_times(0.0, 2.0) == (0.0, 0.5, 1.0, 1.5, 2.0)


def test_pick_best_two_breaks_ties_toward_earlier_stamp():
    early = ScoredFrame(timestamp_s=0.0, sharpness=10.0, motion=0.0)
    late = ScoredFrame(timestamp_s=1.0, sharpness=10.0, motion=0.0)
    assert pick_best_two([late, early]) == (early, late)
    assert pick_best_two([early]) == (early,)
    assert pick_best_two([]) == ()


def test_combined_score_discounts_motion():
    assert combined_score(100.0, 0.0) == pytest.approx(100.0)
    assert combined_score(100.0, 3.0) == pytest.approx(25.0)
    assert combined_score(50.0, 0.0) > combined_score(100.0, 3.0)


def test_laplacian_prefers_edges_over_flat():
    gray_checker = cv2.cvtColor(checker_bgr(), cv2.COLOR_BGR2GRAY)
    gray_solid = cv2.cvtColor(solid_bgr(), cv2.COLOR_BGR2GRAY)
    assert laplacian_sharpness(gray_checker) > laplacian_sharpness(gray_solid)


def test_frame_motion_zero_for_identical_positive_for_shift():
    gray = cv2.cvtColor(checker_bgr(), cv2.COLOR_BGR2GRAY)
    shifted = np.roll(gray, 4, axis=1)
    assert frame_motion(gray, gray) == pytest.approx(0.0)
    assert frame_motion(gray, shifted) > 0.0


def test_score_frames_maps_step_numbers_and_writes_pngs(tmp_path):
    video = make_video(
        tmp_path / "clip.avi",
        [(checker_bgr(), 20), (solid_bgr(), 20)],
    )
    result = score_frames(video, [make_step(1, 0.0, 2.0), make_step(2, 2.0, 4.0)])
    assert set(result) == {1, 2}
    assert len(result[1]) == 2
    assert len(result[2]) == 2
    assert all(p.suffix == ".png" for paths in result.values() for p in paths)
    assert all(stamp_of(p) < 2.0 for p in result[1])
    assert all(stamp_of(p) >= 2.0 for p in result[2])


def test_saved_winning_frame_keeps_source_colour(tmp_path):
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    frame[:, :, 2] = 220
    frame[::8, ::8, 0] = 255
    video = make_video(tmp_path / "colour.avi", [(frame, 20)])
    winner = score_frames(video, [make_step(1, 0.0, 2.0)])[1][0]
    saved = cv2.imread(str(winner), cv2.IMREAD_UNCHANGED)
    assert saved is not None and saved.ndim == 3
    assert np.any(saved[:, :, 2] != saved[:, :, 1])


def test_score_frames_window_past_end_of_video_maps_to_empty(tmp_path):
    video = make_video(tmp_path / "clip.avi", [(checker_bgr(), 20)])
    result = score_frames(video, [make_step(1, 100.0, 104.0)])
    assert result[1] == ()


def test_score_frames_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_frames(tmp_path / "absent.avi", [make_step(1, 0.0, 1.0)])


def test_score_frames_empty_steps_returns_empty(tmp_path):
    video = make_video(tmp_path / "clip.avi", [(checker_bgr(), 10)])
    assert score_frames(video, []) == {}


def test_score_frames_imwrite_failure_returns_empty_and_cleans_partial(
    tmp_path, monkeypatch
):
    video = make_video(
        tmp_path / "clip.avi",
        [(checker_bgr(), 20), (solid_bgr(), 20)],
    )
    real_imwrite = cv2.imwrite
    outcomes = iter([True, False])

    def flaky_imwrite(path, img):
        if next(outcomes):
            return real_imwrite(path, img)
        return False

    monkeypatch.setattr(cv2, "imwrite", flaky_imwrite)
    assert score_frames(video, [make_step(1, 0.0, 4.0)])[1] == ()
    assert list((tmp_path / "clip_frames").glob("*.png")) == []


def test_score_frames_imwrite_failure_on_first_write_returns_empty(
    tmp_path, monkeypatch
):
    video = make_video(tmp_path / "clip.avi", [(checker_bgr(), 20)])
    monkeypatch.setattr(cv2, "imwrite", lambda path, img: False)
    assert score_frames(video, [make_step(1, 0.0, 2.0)])[1] == ()
    assert list((tmp_path / "clip_frames").glob("*.png")) == []


def test_score_frames_second_write_failure_preserves_earlier_winners(
    tmp_path, monkeypatch
):
    video = make_video(
        tmp_path / "clip.avi",
        [(checker_bgr(), 20), (solid_bgr(), 20)],
    )
    step = make_step(1, 0.0, 4.0)
    first = score_frames(video, [step])[1]
    assert len(first) == 2
    before = {path: path.read_bytes() for path in first}
    real_imwrite = cv2.imwrite
    outcomes = iter([True, False])

    def flaky_imwrite(path, img):
        if next(outcomes):
            return real_imwrite(path, img)
        return False

    monkeypatch.setattr(cv2, "imwrite", flaky_imwrite)
    assert score_frames(video, [step])[1] == ()
    for path, content in before.items():
        assert path.read_bytes() == content
    assert sorted(
        p.name
        for p in (tmp_path / "clip_frames").iterdir()
        if p.name != ".publish.lock"
    ) == sorted(p.name for p in before)


def test_score_frames_failed_publication_restores_prior_winners(
    tmp_path, monkeypatch
):
    video = make_video(
        tmp_path / "clip.avi",
        [(checker_bgr(), 20), (solid_bgr(), 20)],
    )
    step = make_step(1, 0.0, 4.0)
    first = score_frames(video, [step])[1]
    assert len(first) == 2
    for rank, path in enumerate(first):
        path.write_bytes(f"prior-{rank}".encode())
    before = {path: path.read_bytes() for path in first}
    real_replace = os.replace
    calls = 0

    def flaky_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk fault")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk fault"):
        score_frames(video, [step])
    for path, content in before.items():
        assert path.read_bytes() == content
    assert sorted(
        p.name
        for p in (tmp_path / "clip_frames").iterdir()
        if p.name != ".publish.lock"
    ) == sorted(p.name for p in before)


def test_score_frames_unopenable_video_creates_no_output_dir(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="cannot open video"):
        score_frames(bad, [make_step(1, 0.0, 1.0)])
    assert not (tmp_path / "bad_frames").exists()


def test_score_frames_unopenable_video_releases_capture(tmp_path, monkeypatch):
    created: list = []

    class StubCapture:
        def __init__(self, *args, **kwargs) -> None:
            self.released = False
            created.append(self)

        def isOpened(self):  # noqa: N802 — mirrors the cv2 name
            return False

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", StubCapture)
    video = tmp_path / "clip.avi"
    video.write_bytes(b"junk")
    with pytest.raises(ValueError, match="cannot open video"):
        score_frames(video, [make_step(1, 0.0, 1.0)])
    assert created and all(capture.released for capture in created)


def test_score_frames_mkdir_collision_releases_capture(tmp_path, monkeypatch):
    created: list = []

    class StubCapture:
        def __init__(self, *args, **kwargs) -> None:
            self.released = False
            created.append(self)

        def isOpened(self):  # noqa: N802 — mirrors the cv2 name
            return True

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", StubCapture)
    video = tmp_path / "clip.avi"
    video.write_bytes(b"junk")
    (tmp_path / "clip_frames").write_bytes(b"a regular file blocks mkdir")
    with pytest.raises(FileExistsError):
        score_frames(video, [make_step(1, 0.0, 1.0)])
    assert created and all(capture.released for capture in created)
