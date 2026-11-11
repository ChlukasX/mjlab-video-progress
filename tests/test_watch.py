"""Contract tests for watch: poll_run, with a fake render and fake sinks."""

from __future__ import annotations

import logging

import pytest
from helpers import snapshot, zip_bytes

from mjlab_video_progress import watch
from mjlab_video_progress.config import Config, OutputConfig, RenderConfig, WatchConfig
from mjlab_video_progress.discovery import Run
from mjlab_video_progress.exceptions import RenderError, SinkError
from mjlab_video_progress.scan import Checkpoint
from mjlab_video_progress.sinks import Delivery
from mjlab_video_progress.watch import poll_run

RUN_NAME = "exp/2026-11-09_12-00-00"


class Bug(Exception):
    """Stands for any exception poll_run does not expect."""


@pytest.fixture
def cfg(tmp_path):
    return Config(
        watch=WatchConfig(every=100),
        render=RenderConfig(command=("play",), timeout_seconds=60),
        output=OutputConfig(dir=tmp_path / "out"),
    )


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "logs/rsl_rl" / RUN_NAME
    run_dir.mkdir(parents=True)
    return Run(run_dir=run_dir, task="Task-X")


def add_checkpoints(run, *iterations):
    for it in iterations:
        (run.run_dir / f"model_{it}.pt").write_bytes(zip_bytes())


class FakeRender:
    """Stands in for render_checkpoint: writes a video in the work dir, or raises."""

    def __init__(self, raises=None, on_call=None):
        self.raises, self.on_call, self.calls = raises, on_call, []

    def __call__(self, render_cfg, task, checkpoint, work_dir):
        self.calls.append((render_cfg, task, checkpoint, work_dir))
        if self.on_call:
            self.on_call(work_dir)
        if self.raises:
            raise self.raises
        video = work_dir / "videos/play/rl-video-step-0.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        return video


class FakeSink:
    def __init__(self, raises=None):
        self.raises, self.deliveries = raises, []

    def __call__(self, delivery):
        self.deliveries.append(delivery)
        if self.raises:
            raise self.raises


def stored_video(cfg, iteration):
    return cfg.output.dir / RUN_NAME / f"iter_{iteration:06d}" / "video.mp4"


def test_nothing_is_done_without_checkpoints(cfg, run):
    render, state = FakeRender(), {}
    poll_run(cfg, run, state, [FakeSink()], render)
    assert render.calls == [] and state == {}


@pytest.mark.parametrize(("last", "iterations"), [(None, (0, 50)), (100, (0, 50, 100, 150))])
def test_nothing_is_done_before_the_next_checkpoint_is_due(cfg, run, last, iterations):
    add_checkpoints(run, *iterations)
    render, state = FakeRender(), {run.run_dir: last}
    poll_run(cfg, run, state, [FakeSink()], render)
    assert render.calls == [] and state == {run.run_dir: last}


def test_due_checkpoint_is_rendered_stored_and_delivered(cfg, run):
    add_checkpoints(run, 0, 50, 100, 150)
    render, sink, state = FakeRender(), FakeSink(), {}
    poll_run(cfg, run, state, [sink], render)
    assert render.calls == [
        (
            cfg.render,
            "Task-X",
            Checkpoint(150, run.run_dir / "model_150.pt"),
            cfg.output.dir / ".work",
        )
    ]
    video = stored_video(cfg, 150)
    assert video.read_bytes() == b"video"
    assert sink.deliveries == [Delivery(video, "Task-X", run.run_dir, 150)]
    assert state == {run.run_dir: 150}


def test_several_due_renders_only_the_newest(cfg, run):
    add_checkpoints(run, *range(0, 501, 100))
    render = FakeRender()
    poll_run(cfg, run, {}, [FakeSink()], render)
    assert [call[2].iteration for call in render.calls] == [500]


def test_state_is_per_run(cfg, run, tmp_path):
    add_checkpoints(run, 0, 100)
    render, other = FakeRender(), tmp_path / "logs/rsl_rl/exp/other-run"
    poll_run(cfg, run, {other: 5000}, [FakeSink()], render)
    assert len(render.calls) == 1


def test_iteration_is_marked_handled_before_the_render_starts(cfg, run):
    add_checkpoints(run, 0, 100)
    state, seen = {}, []
    render = FakeRender(on_call=lambda work_dir: seen.append(dict(state)))
    poll_run(cfg, run, state, [], render)
    assert seen == [{run.run_dir: 100}]


@pytest.mark.parametrize("error", [RenderError("play timed out"), FileNotFoundError("gone")])
def test_render_failure_is_logged_and_nothing_is_stored_or_delivered(cfg, run, caplog, error):
    add_checkpoints(run, 0, 100)
    sink, state = FakeSink(), {}
    with caplog.at_level(logging.WARNING):
        poll_run(cfg, run, state, [sink], FakeRender(raises=error))
    assert state == {run.run_dir: 100}  # consumed: not tried again
    assert sink.deliveries == [] and not stored_video(cfg, 100).exists()
    assert all(s in caplog.text for s in (RUN_NAME, "100", str(error)))


def test_store_failure_is_logged_and_nothing_is_delivered(cfg, run, caplog, monkeypatch):
    def full_disk(*args):
        raise OSError("no space left on device")

    monkeypatch.setattr(watch, "store_video", full_disk)
    add_checkpoints(run, 0, 100)
    sink = FakeSink()
    with caplog.at_level(logging.WARNING):
        poll_run(cfg, run, {}, [sink], FakeRender())
    assert sink.deliveries == [] and "no space left" in caplog.text


@pytest.mark.parametrize("error", [SinkError("telegram is down"), OSError("video vanished")])
def test_sink_failure_is_logged_and_the_other_sinks_still_run(cfg, run, caplog, error):
    add_checkpoints(run, 0, 100)
    failing, working, state = FakeSink(raises=error), FakeSink(), {}
    with caplog.at_level(logging.WARNING):
        poll_run(cfg, run, state, [failing, working], FakeRender())
    assert len(working.deliveries) == 1 and stored_video(cfg, 100).exists()
    assert str(error) in caplog.text and state == {run.run_dir: 100}


def test_a_bug_in_a_sink_propagates_after_the_iteration_is_marked_handled(cfg, run):
    add_checkpoints(run, 0, 100)
    state = {}
    with pytest.raises(Bug):
        poll_run(cfg, run, state, [FakeSink(raises=Bug("bug"))], FakeRender())
    assert state == {run.run_dir: 100}


def test_work_dir_is_emptied_before_the_render_and_left_afterwards(cfg, run):
    add_checkpoints(run, 0, 100)
    work = cfg.output.dir / ".work"
    work.mkdir(parents=True)
    (work / "old.log").write_text("from the last render")
    seen = []
    poll_run(cfg, run, {}, [], FakeRender(on_call=lambda w: seen.append((w / "old.log").exists())))
    assert seen == [False]
    assert (work / "videos/play/rl-video-step-0.mp4").exists()


def test_run_dir_that_vanished_propagates_without_rendering(cfg, run):
    run.run_dir.rmdir()
    render = FakeRender()
    with pytest.raises(FileNotFoundError):
        poll_run(cfg, run, {}, [FakeSink()], render)
    assert render.calls == []


def test_poll_leaves_the_logs_tree_identical(cfg, run, tmp_path):
    add_checkpoints(run, 0, 50, 100)
    logs = tmp_path / "logs"
    before = snapshot(logs)
    poll_run(cfg, run, {}, [FakeSink()], FakeRender())
    assert snapshot(logs) == before
