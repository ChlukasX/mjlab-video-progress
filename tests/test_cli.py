"""Contract tests for cli.main: arguments, exit codes, signals. The loop itself is patched out
except in the two tests that run a real process."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest
from helpers import assert_silent, fake_play, zip_bytes

from mjlab_video_progress import cli, telegram
from mjlab_video_progress import watch as watch_module
from mjlab_video_progress import watchers as watchers_module
from mjlab_video_progress.config import load_config
from mjlab_video_progress.discovery import Run
from mjlab_video_progress.exceptions import ConfigError, RunDirNotFound, SinkError, WatcherConflict
from mjlab_video_progress.watch import Session
from mjlab_video_progress.watchers import Watcher, find_watchers

CONFIG = """\
[watch]
every = 100

[render]
command = ["play"]
timeout_seconds = 60

[output]
dir = "out"
"""


class Bug(Exception):
    """Stands for any exception cli does not expect."""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text(CONFIG)
    return path


@pytest.fixture
def session(tmp_path):
    return Session([Run(tmp_path / "logs/exp/2026-11-09_12-00-00", "Task-X")], {}, [])


def patch_loop(monkeypatch, session, run=None):
    """Replace start and watch; `run(cfg, session)` is what the fake watch does."""
    calls = []
    monkeypatch.setattr(watch_module, "start", lambda cfg: session)

    def fake_watch(cfg, sess):
        calls.append((cfg, sess))
        if run:
            run(cfg, sess)

    monkeypatch.setattr(watch_module, "watch", fake_watch)
    return calls


# --- arguments ----------------------------------------------------------------------------------


def test_help_names_the_watch_command(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0 and "watch" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["watch"],
        ["watch", "--config", "c.toml", "--log", "x.log"],  # --log needs -d
        ["stop"],
        ["stop", "1", "--all"],
    ],
)
def test_bad_arguments_are_argparse_errors(capsys, argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    if argv == ["watch"]:
        assert "--config" in capsys.readouterr().err


def test_console_script_points_at_main():
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    assert scripts == {"mjlab-video-progress": "mjlab_video_progress.cli:main"}


# --- errors: message on stderr, exit code 1 -----------------------------------------------------


def test_missing_config_file(tmp_path, capsys, monkeypatch, session):
    calls = patch_loop(monkeypatch, session)
    assert cli.main(["watch", "--config", str(tmp_path / "nope.toml")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "nope.toml" in err and calls == []


def test_invalid_config_names_the_key(config, capsys, monkeypatch, session):
    config.write_text(CONFIG.replace("every = 100", "every = 0"))
    calls = patch_loop(monkeypatch, session)
    assert cli.main(["watch", "--config", str(config)]) == 1
    assert "watch.every" in capsys.readouterr().err and calls == []


@pytest.mark.parametrize(
    "error",
    [
        RunDirNotFound("no run dir under /x"),
        ConfigError("sinks.telegram: no token"),
        WatcherConflict("exp/run is already watched by pid 7"),
    ],
)
def test_errors_from_start(config, capsys, monkeypatch, session, error):
    def failing_start(cfg):
        raise error

    calls = patch_loop(monkeypatch, session)
    monkeypatch.setattr(watch_module, "start", failing_start)
    assert cli.main(["watch", "--config", str(config)]) == 1
    assert capsys.readouterr().err == f"error: {error}\n" and calls == []


def test_a_bug_is_not_turned_into_an_exit_code(config, monkeypatch, session):
    def bug(cfg, sess):
        raise Bug

    patch_loop(monkeypatch, session, bug)
    with pytest.raises(Bug):
        cli.main(["watch", "--config", str(config)])


# --- running and stopping -----------------------------------------------------------------------


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit])
def test_a_stop_is_a_clean_exit(config, monkeypatch, session, caplog, stop):
    def stopping(cfg, sess):
        raise stop

    calls = patch_loop(monkeypatch, session, stopping)
    with caplog.at_level(logging.INFO):
        assert cli.main(["watch", "--config", str(config)]) == 0
    assert calls == [(load_config(config), session)]
    assert "watching 1 run(s): exp/2026-11-09_12-00-00" in caplog.text


def test_sigterm_becomes_a_clean_stop_and_the_old_handler_returns(config, monkeypatch, session):
    before = signal.getsignal(signal.SIGTERM)

    def receive_sigterm(cfg, sess):
        handler = signal.getsignal(signal.SIGTERM)
        assert handler != before
        handler(signal.SIGTERM, None)  # what the signal would do: raise SystemExit

    patch_loop(monkeypatch, session, receive_sigterm)
    assert cli.main(["watch", "--config", str(config)]) == 0
    assert signal.getsignal(signal.SIGTERM) == before


# --- a real process: the stop signal must not leave play behind ----------------------------------

DRIVER = """
import signal, sys
from functools import partial
from pathlib import Path
from mjlab_video_progress import cli, watch

signal.signal(signal.SIGINT, signal.default_int_handler)  # even if the parent ignores SIGINT
watch.start = partial(watch.start, proc_root=Path(sys.argv[1]))  # macOS has no /proc
watch.free_mib = lambda extra_args: None  # never the real nvidia-smi
sys.exit(cli.main(sys.argv[2:]))  # argv: proc_root, then the CLI arguments
"""


def wait_until_play_runs(proc, work, stderr_log):
    child_beat = work / "child.beat"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        assert proc.poll() is None, f"the CLI exited early:\n{stderr_log.read_text()}"
        if child_beat.exists() and child_beat.stat().st_size > 0:
            return
        time.sleep(0.05)
    pytest.fail("play never started")


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_stop_signal_during_a_render_stops_play_and_its_child(tmp_path, sig):
    run_dir = tmp_path / "logs/rsl_rl/exp/2026-11-09_12-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "model_100.pt").write_bytes(zip_bytes())  # due at once: every = 100
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    play = fake_play(tmp_path, "start_beats()\nbeat()")  # never writes a video: the render hangs
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    config = tmp_path / "cfg.toml"
    config.write_text(
        f"""\
[watch]
every = 100
poll_seconds = 1
log_root = {json.dumps(str(tmp_path / "logs/rsl_rl"))}
task = "Task-X"

[render]
command = [{json.dumps(sys.executable)}, {json.dumps(str(play))}]
timeout_seconds = 60

[output]
dir = {json.dumps(str(tmp_path / "out"))}
"""
    )
    stderr_log = tmp_path / "stderr.log"
    with open(stderr_log, "wb") as err:
        cmd = [sys.executable, str(driver), str(proc_root), "watch", "--config", str(config)]
        proc = subprocess.Popen(cmd, stderr=err, start_new_session=True)
    work = tmp_path / "out/exp/2026-11-09_12-00-00/.work"
    try:
        wait_until_play_runs(proc, work, stderr_log)
        proc.send_signal(sig)
        assert proc.wait(timeout=30) == 0, stderr_log.read_text()
        assert_silent(work / "play.beat")
        assert_silent(work / "child.beat")
        assert "watching 1 run(s): exp/2026-11-09_12-00-00" in stderr_log.read_text()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


# --- test ---------------------------------------------------------------------------------------


@pytest.fixture
def telegram_config(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text(CONFIG + '\n[sinks.telegram]\nchat_id = 42\ntoken_env = "MY_BOT"\n')
    return path


def test_test_sends_the_message_and_reports_ok(telegram_config, monkeypatch, capsys):
    sent = []
    monkeypatch.setenv("MY_BOT", "tok")
    monkeypatch.setattr(telegram, "send_text", lambda cfg, token, text: sent.append((cfg, token)))
    assert cli.main(["test", "--config", str(telegram_config)]) == 0
    assert [(cfg.chat_id, token) for cfg, token in sent] == [(42, "tok")]
    assert capsys.readouterr().out == "telegram: ok\n"
    assert os.environ["MY_BOT"] == "tok"  # test leaves the environment alone


def test_test_reports_a_failing_sink(telegram_config, monkeypatch, capsys):
    def failing(cfg, token, text):
        raise SinkError("Telegram error 400: chat not found")

    monkeypatch.setenv("MY_BOT", "tok")
    monkeypatch.setattr(telegram, "send_text", failing)
    assert cli.main(["test", "--config", str(telegram_config)]) == 1
    assert capsys.readouterr().out == "telegram: Telegram error 400: chat not found\n"


def test_test_reports_a_missing_token(telegram_config, monkeypatch, capsys):
    monkeypatch.delenv("MY_BOT", raising=False)
    monkeypatch.setattr(telegram, "send_text", lambda *a: pytest.fail("sent without a token"))
    assert cli.main(["test", "--config", str(telegram_config)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("telegram: ") and "MY_BOT" in out


def test_test_without_a_sink_fails(config, capsys):
    assert cli.main(["test", "--config", str(config)]) == 1
    assert "no sink configured" in capsys.readouterr().err


# --- chat-id ------------------------------------------------------------------------------------


def chat_id_token(tmp_path):
    path = tmp_path / "token"
    path.write_text("tok\n")
    path.chmod(0o600)
    return path


def test_chat_id_prints_every_chat(tmp_path, monkeypatch, capsys):
    seen = []
    chats = {7: "ann", -100: "Lab", 8: ""}
    monkeypatch.setattr(telegram, "get_chats", lambda token: seen.append(token) or chats)
    assert cli.main(["chat-id", "--token-file", str(chat_id_token(tmp_path))]) == 0
    assert seen == ["tok"]
    assert capsys.readouterr().out == "7  ann\n-100  Lab\n8\n"


def test_chat_id_reads_the_environment_without_a_token_file(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "envtok")
    monkeypatch.setattr(telegram, "get_chats", lambda token: {1: token})
    assert cli.main(["chat-id"]) == 0
    assert capsys.readouterr().out == "1  envtok\n"


def test_chat_id_without_chats_points_elsewhere(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(telegram, "get_chats", lambda token: {})
    assert cli.main(["chat-id", "--token-file", str(chat_id_token(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "send your bot a message" in err.lower() and "@userinfobot" in err


def test_chat_id_errors_are_reported(tmp_path, monkeypatch, capsys):
    loose = chat_id_token(tmp_path)
    loose.chmod(0o644)
    monkeypatch.setattr(telegram, "get_chats", lambda token: pytest.fail("token not checked"))
    assert cli.main(["chat-id", "--token-file", str(loose)]) == 1
    assert "chmod 600" in capsys.readouterr().err

    def failing(token):
        raise SinkError("Telegram error 401: Unauthorized")

    monkeypatch.setattr(telegram, "get_chats", failing)
    assert cli.main(["chat-id", "--token-file", str(chat_id_token(tmp_path))]) == 1
    assert "error: Telegram error 401: Unauthorized" in capsys.readouterr().err


# --- watch -d, ps, stop -------------------------------------------------------------------------


def test_detach_returns_in_the_original_process(config, monkeypatch, session, capsys, tmp_path):
    calls = patch_loop(monkeypatch, session)
    detached = []
    monkeypatch.setattr(watchers_module, "detach", lambda log: detached.append(log) or 4242)
    log = tmp_path / "w.log"
    assert cli.main(["watch", "--config", str(config), "-d", "--log", str(log)]) == 0
    assert detached == [log] and calls == []
    assert capsys.readouterr().out == f"watcher 4242 started, log: {log}\n"


def test_detach_runs_the_loop_in_the_detached_process(config, monkeypatch, session, tmp_path):
    calls = patch_loop(monkeypatch, session)
    monkeypatch.setattr(watchers_module, "detach", lambda log: None)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert cli.main(["watch", "--config", str(config), "-d"]) == 0
    assert calls == [(load_config(config), session)]


def test_ps_prints_a_table(monkeypatch, capsys, tmp_path):
    lock = tmp_path / "out/exp/2026-11-09_12-00-00/.watch.lock"
    found = [
        Watcher(7, tmp_path / "a.toml", (lock,), tmp_path / "a.log", time.time() - 3725),
        Watcher(12345, None, (lock, lock), None, None),
    ]
    monkeypatch.setattr(watchers_module, "find_watchers", lambda: found)
    assert cli.main(["ps"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["PID", "UPTIME", "CONFIG", "RUNS", "LOG"]
    assert lines[1].split() == [
        "7",
        "1:02:05",
        str(tmp_path / "a.toml"),
        "exp/2026-11-09_12-00-00",
        str(tmp_path / "a.log"),
    ]
    assert lines[2].split() == [
        "12345",
        "-",
        "-",
        "exp/2026-11-09_12-00-00,exp/2026-11-09_12-00-00",
        "-",
    ]


def test_ps_without_watchers(monkeypatch, capsys):
    monkeypatch.setattr(watchers_module, "find_watchers", lambda: [])
    assert cli.main(["ps"]) == 0
    assert capsys.readouterr().out == "no watchers running\n"


def watchers_with_pids(monkeypatch, *pids):
    found = [Watcher(pid, None, (), None, None) for pid in pids]
    monkeypatch.setattr(watchers_module, "find_watchers", lambda: found)
    stopped = []

    def fake_stop(targets):
        stopped.append(list(targets))
        return [pid for pid in targets if pid == 13]  # 13 never stops

    monkeypatch.setattr(watchers_module, "stop", fake_stop)
    return stopped


def test_stop_refuses_a_pid_that_is_not_a_watcher(monkeypatch, capsys):
    stopped = watchers_with_pids(monkeypatch, 7)
    assert cli.main(["stop", "7", "8"]) == 1
    assert stopped == [] and "8 is not a running watcher" in capsys.readouterr().err


def test_stop_named_and_all(monkeypatch, capsys):
    stopped = watchers_with_pids(monkeypatch, 7, 9, 13)
    assert cli.main(["stop", "9"]) == 0
    assert capsys.readouterr().out == "stopped 9\n"
    assert cli.main(["stop", "--all"]) == 1
    out, err = capsys.readouterr()
    assert out == "stopped 7\nstopped 9\n" and err == "error: 13 still running\n"
    assert stopped == [[9], [7, 9, 13]]


def test_stop_all_without_watchers(monkeypatch):
    stopped = watchers_with_pids(monkeypatch)
    assert cli.main(["stop", "--all"]) == 0
    assert stopped == [[]]


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")
def test_detached_watcher_is_listed_refuses_a_second_and_stops(tmp_path, capsys):
    run_dir = tmp_path / "logs/rsl_rl/exp/2026-11-09_12-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "model_100.pt").write_bytes(zip_bytes())
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    play = fake_play(tmp_path, "start_beats()\nbeat()")  # the render hangs until stopped
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    config = tmp_path / "cfg.toml"
    config.write_text(
        f"""\
[watch]
every = 100
log_root = {json.dumps(str(tmp_path / "logs/rsl_rl"))}
task = "Task-X"

[render]
command = [{json.dumps(sys.executable)}, {json.dumps(str(play))}]
timeout_seconds = 60

[output]
dir = {json.dumps(str(tmp_path / "out"))}
"""
    )
    log = tmp_path / "state/w.log"
    cmd = [sys.executable, str(driver), str(proc_root), "watch", "--config", str(config)]
    cmd += ["-d", "--log", str(log)]
    first = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stderr
    pid = int(first.stdout.split()[1])
    work = tmp_path / "out/exp/2026-11-09_12-00-00/.work"
    try:
        wait_until_play_runs_detached(pid, work, log)
        [me] = [w for w in find_watchers() if w.pid == pid]
        assert me.log == log and me.config == config
        assert me.locks == (tmp_path / "out/exp/2026-11-09_12-00-00/.watch.lock",)
        second = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert second.returncode == 1
        assert f"is already watched by pid {pid}" in second.stderr
        assert cli.main(["stop", str(pid)]) == 0
        assert capsys.readouterr().out == f"stopped {pid}\n"
        assert_silent(work / "play.beat")
        assert_silent(work / "child.beat")
        assert "watching 1 run(s)" in log.read_text() and "stopped" in log.read_text()
    finally:  # SIGTERM first: a SIGKILLed watcher leaves its play running
        if pid in [w.pid for w in find_watchers()] and watchers_module.stop([pid], timeout=20):
            os.kill(pid, signal.SIGKILL)


def wait_until_play_runs_detached(pid, work, log):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        beat = work / "child.beat"
        if beat.exists() and beat.stat().st_size > 0:
            return
        time.sleep(0.05)
    pytest.fail(f"play never started:\n{log.read_text() if log.exists() else ''}")
