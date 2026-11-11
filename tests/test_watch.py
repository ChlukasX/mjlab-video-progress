"""Contract tests for watch: poll_run, with a fake render and fake sinks."""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import pytest
from helpers import snapshot, zip_bytes

from mjlab_video_progress import watch
from mjlab_video_progress.config import (
    Config,
    OutputConfig,
    RenderConfig,
    SinksConfig,
    TelegramConfig,
    WatchConfig,
)
from mjlab_video_progress.discovery import Run
from mjlab_video_progress.exceptions import ConfigError, DiscoveryError, RenderError, SinkError
from mjlab_video_progress.scan import Checkpoint
from mjlab_video_progress.sinks import Delivery
from mjlab_video_progress.store import store_video
from mjlab_video_progress.watch import Session, poll_run, start
from mjlab_video_progress.watch import watch as watch_loop

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


# --- start ---------------------------------------------------------------------------------------

TOKEN = "123456:SECRET-token_value"


@pytest.fixture
def wcfg(cfg, tmp_path):
    """cfg with what start needs: the log root (for the directory fallback), a task, a poll time."""
    watch_cfg = replace(cfg.watch, poll_seconds=7, log_root=tmp_path / "logs/rsl_rl", task="Task-X")
    return replace(cfg, watch=watch_cfg)


def telegram_cfg(wcfg, **kwargs):
    return replace(wcfg, sinks=SinksConfig(telegram=TelegramConfig(chat_id=42, **kwargs)))


def no_processes(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    return proc_root


def test_start_discovers_the_run_and_has_no_sinks_without_config(wcfg, run, tmp_path):
    session = start(wcfg, {}, no_processes(tmp_path))
    assert session.runs == [Run(run.run_dir, "Task-X")]
    assert session.state == {run.run_dir: None}
    assert session.sinks == []


def test_start_seeds_the_state_from_the_stored_videos(wcfg, run, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    for iteration in (200, 300):
        store_video(video, wcfg.output.dir, run.run_dir, iteration)
    assert start(wcfg, {}, no_processes(tmp_path)).state == {run.run_dir: 300}


def test_start_propagates_a_discovery_error(wcfg, tmp_path):  # the log root does not exist
    with pytest.raises(DiscoveryError):
        start(wcfg, {}, no_processes(tmp_path))


def test_start_builds_a_telegram_sink_that_posts_with_the_token(wcfg, run, tmp_path):
    urls = []

    def post(url, headers, body):
        urls.append(url)
        return 200, b'{"ok": true}'

    session = start(telegram_cfg(wcfg), {"TELEGRAM_BOT_TOKEN": TOKEN}, no_processes(tmp_path), post)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    assert len(session.sinks) == 1
    session.sinks[0](Delivery(video, "Task-X", run.run_dir, 100))
    assert urls == [f"https://api.telegram.org/bot{TOKEN}/sendVideo"]


def test_session_repr_never_contains_the_token(wcfg, run, tmp_path):
    session = start(telegram_cfg(wcfg), {"TELEGRAM_BOT_TOKEN": TOKEN}, no_processes(tmp_path))
    assert TOKEN not in repr(session)


@pytest.mark.parametrize("name", ["TELEGRAM_BOT_TOKEN", "MY_BOT"])
def test_start_removes_the_token_variable_from_the_environment(wcfg, run, tmp_path, name):
    environ = {name: TOKEN, "OTHER": "x"}
    start(telegram_cfg(wcfg, token_env=name), environ, no_processes(tmp_path))
    assert environ == {"OTHER": "x"}


def test_start_empties_os_environ_by_default(wcfg, run, tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)  # what a child process such as play inherits
    start(telegram_cfg(wcfg), proc_root=no_processes(tmp_path))
    assert "TELEGRAM_BOT_TOKEN" not in os.environ


def test_start_leaves_the_environment_alone_when_the_token_comes_from_a_file(wcfg, run, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n")
    token_file.chmod(0o600)
    environ = {"TELEGRAM_BOT_TOKEN": "unrelated"}
    start(telegram_cfg(wcfg, token_file=token_file), environ, no_processes(tmp_path))
    assert environ == {"TELEGRAM_BOT_TOKEN": "unrelated"}


def test_start_refuses_a_missing_token(wcfg, run, tmp_path):
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        start(telegram_cfg(wcfg), {}, no_processes(tmp_path))


# --- watch ---------------------------------------------------------------------------------------


class Stop(Exception):
    """Ends the endless loop in a test."""


def stopping_after(passes, sleeps, between=lambda n: None):
    """A fake sleep: records its argument, runs `between(n)`, raises Stop on the last call."""

    def sleep(seconds):
        sleeps.append(seconds)
        between(len(sleeps))
        if len(sleeps) >= passes:
            raise Stop

    return sleep


def second_run(run):
    run_dir = run.run_dir.parent / "2026-11-09_13-00-00"
    run_dir.mkdir()
    return Run(run_dir, "Task-X")


def test_watch_polls_then_sleeps_poll_seconds_until_stopped(wcfg, run):
    add_checkpoints(run, 0, 100)
    render, sleeps = FakeRender(), []
    session = Session([run], {run.run_dir: None}, [FakeSink()])

    def new_checkpoint_after_the_first_pass(n):
        if n == 1:
            add_checkpoints(run, 200)

    sleep = stopping_after(3, sleeps, new_checkpoint_after_the_first_pass)
    with pytest.raises(Stop):
        watch_loop(wcfg, session, sleep, render)
    assert sleeps == [7, 7, 7]
    assert [call[2].iteration for call in render.calls] == [100, 200]  # nothing new in pass 3


def test_watch_handles_the_runs_in_order_and_one_at_a_time(wcfg, run):
    other = second_run(run)
    add_checkpoints(run, 0, 100)
    add_checkpoints(other, 0, 100)
    events, fake = [], FakeRender()

    def render(render_cfg, task, checkpoint, work_dir):
        events.append(("enter", checkpoint.path.parent.name))
        video = fake(render_cfg, task, checkpoint, work_dir)
        events.append(("exit", checkpoint.path.parent.name))
        return video

    session = Session([run, other], {run.run_dir: None, other.run_dir: None}, [])
    with pytest.raises(Stop):
        watch_loop(wcfg, session, stopping_after(1, []), render)
    first, second = run.run_dir.name, other.run_dir.name
    assert events == [("enter", first), ("exit", first), ("enter", second), ("exit", second)]


def test_a_bug_in_one_run_is_logged_and_the_other_runs_and_later_passes_go_on(wcfg, run, caplog):
    other = second_run(run)
    add_checkpoints(run, 0, 100)
    add_checkpoints(other, 0, 100)
    fake, sink, rendered = FakeRender(), FakeSink(), []

    def render(render_cfg, task, checkpoint, work_dir):
        rendered.append(checkpoint.path.parent.name)
        if checkpoint.path.parent == run.run_dir:
            raise Bug("render blew up")
        return fake(render_cfg, task, checkpoint, work_dir)

    session = Session([run, other], {run.run_dir: None, other.run_dir: None}, [sink])
    with caplog.at_level(logging.ERROR), pytest.raises(Stop):
        watch_loop(wcfg, session, stopping_after(2, []), render)
    assert [d.run_dir for d in sink.deliveries] == [other.run_dir]  # the other run still ran
    assert rendered == [run.run_dir.name, other.run_dir.name]  # and nothing was retried in pass 2
    assert "render blew up" in caplog.text and run.run_dir.name in caplog.text
    assert all(record.exc_info for record in caplog.records)  # with the traceback


@pytest.mark.parametrize("where", ["sleep", "render"])
def test_watch_does_not_swallow_ctrl_c(wcfg, run, where):
    add_checkpoints(run, 0, 100)
    session = Session([run], {run.run_dir: None}, [])

    def interrupt(*args):
        raise KeyboardInterrupt

    sleep = interrupt if where == "sleep" else stopping_after(5, [])
    render = interrupt if where == "render" else FakeRender()
    with pytest.raises(KeyboardInterrupt):
        watch_loop(wcfg, session, sleep, render)
