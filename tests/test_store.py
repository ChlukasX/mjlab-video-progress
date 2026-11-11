"""Contract tests for store: store_video and last_stored_iteration."""

from __future__ import annotations

import pytest

from mjlab_video_progress.store import last_stored_iteration, store_video

RUN = "exp/2026-11-09_12-00-00"


@pytest.fixture
def run_dir(tmp_path):
    """The run's dir in the logs. Not created: store only uses its two last names."""
    return tmp_path / "logs/rsl_rl" / RUN


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


def make_video(tmp_path, content=b"video", name="rl-video-step-0.mp4"):
    path = tmp_path / "work" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_stores_the_video_at_the_layout_path(tmp_path, run_dir, out):
    video = make_video(tmp_path)
    stored = store_video(video, out, run_dir, 150)
    assert stored == out / RUN / "iter_000150" / "video.mp4"
    assert stored.read_bytes() == b"video"
    assert not run_dir.exists()  # only its names were used


def test_wide_iteration_keeps_all_digits(tmp_path, run_dir, out):
    stored = store_video(make_video(tmp_path), out, run_dir, 1234567)
    assert stored.parent.name == "iter_1234567"


def test_source_video_is_untouched(tmp_path, run_dir, out):
    video = make_video(tmp_path)
    mtime = video.stat().st_mtime_ns
    store_video(video, out, run_dir, 150)
    assert (video.read_bytes(), video.stat().st_mtime_ns) == (b"video", mtime)


def test_storing_an_iteration_again_replaces_the_video(tmp_path, run_dir, out):
    store_video(make_video(tmp_path, b"old"), out, run_dir, 150)
    stored = store_video(make_video(tmp_path, b"new"), out, run_dir, 150)
    assert stored.read_bytes() == b"new"
    assert [p.name for p in stored.parent.iterdir()] == ["video.mp4"]  # no temp file left


def test_missing_source_keeps_the_earlier_video_and_leaves_no_temp_file(tmp_path, run_dir, out):
    stored = store_video(make_video(tmp_path, b"old"), out, run_dir, 150)
    with pytest.raises(FileNotFoundError):
        store_video(tmp_path / "work" / "gone.mp4", out, run_dir, 150)
    assert [p.name for p in stored.parent.iterdir()] == ["video.mp4"]
    assert stored.read_bytes() == b"old"


def test_missing_source_for_a_new_iteration_leaves_no_files(tmp_path, run_dir, out):
    with pytest.raises(FileNotFoundError):
        store_video(tmp_path / "gone.mp4", out, run_dir, 150)
    assert not any(p.is_file() for p in out.rglob("*"))


def test_last_stored_iteration_is_none_when_nothing_is_stored(tmp_path, run_dir, out):
    assert last_stored_iteration(out, run_dir) is None  # output dir does not exist yet
    store_video(make_video(tmp_path), out, run_dir, 50)
    other = run_dir.parent / "2026-11-10_08-00-00"
    assert last_stored_iteration(out, other) is None  # nothing stored for that run


def test_last_stored_iteration_is_the_numeric_max(tmp_path, run_dir, out):
    for iteration in (50, 150, 1000000, 900):  # lexical order would put 150 above 1000000
        store_video(make_video(tmp_path), out, run_dir, iteration)
    assert last_stored_iteration(out, run_dir) == 1000000


def test_last_stored_iteration_ignores_everything_that_is_not_a_stored_video(
    tmp_path, run_dir, out
):
    store_video(make_video(tmp_path), out, run_dir, 150)
    run_out = out / RUN
    (run_out / "iter_000900").mkdir()  # half-built: no video.mp4
    (run_out / "iter_000800").mkdir()
    (run_out / "iter_000800" / "strip.png").write_bytes(b"x")  # other files, still no video.mp4
    (run_out / "iter_000700").write_bytes(b"x")  # a file, not a dir
    (run_out / "iter_x").mkdir()
    (run_out / "iter_x" / "video.mp4").write_bytes(b"x")
    (run_out / "notes.txt").write_bytes(b"x")
    store_video(make_video(tmp_path), out, run_dir.parent / "2026-12-01_00-00-00", 9999)
    store_video(make_video(tmp_path), out, run_dir.parent.parent / "other-exp" / run_dir.name, 9999)
    assert last_stored_iteration(out, run_dir) == 150
