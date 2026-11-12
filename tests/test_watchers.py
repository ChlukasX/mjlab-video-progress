"""Contract tests for watchers: the run lock, the /proc scan, the log path, stop."""

from __future__ import annotations

import os
import signal

import pytest

from mjlab_video_progress.watchers import (
    LOCK_NAME,
    Watcher,
    default_log,
    find_watchers,
    lock_run,
    stop,
)


def test_a_run_lock_is_exclusive_until_closed(tmp_path):
    run_out = tmp_path / "out/exp/run"
    first = lock_run(run_out)
    assert first is not None and (run_out / LOCK_NAME).is_file()
    assert lock_run(run_out) is None  # flock is per open file: also within one process
    first.close()
    second = lock_run(run_out)
    assert second is not None
    second.close()


def fake_process(proc_root, pid, *, cmdline, cwd, fds, state="S", starttime=500):
    entry = proc_root / str(pid)
    (entry / "fd").mkdir(parents=True)
    (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in cmdline) + b"\0")
    (entry / "cwd").symlink_to(cwd)
    for fd, target in fds.items():
        (entry / "fd" / str(fd)).symlink_to(target)
    fields = [state, *["0"] * 18, str(starttime)]  # field 22 (starttime) is the 20th after ")"
    (entry / "stat").write_text(f"{pid} (python3) {' '.join(fields)}\n")
    return entry


@pytest.fixture
def proc_root(tmp_path):
    root = tmp_path / "proc"
    root.mkdir()
    (root / "stat").write_text("cpu 1 2 3\nbtime 1000\n")
    return root


def test_find_watchers_reads_locks_config_log_and_start(tmp_path, proc_root):
    lock = tmp_path / "out/exp/run" / LOCK_NAME
    lock.parent.mkdir(parents=True)
    lock.touch()
    log = tmp_path / "w.log"
    log.touch()
    home = tmp_path / "home"
    home.mkdir()
    fake_process(
        proc_root,
        42,
        cmdline=["python3", "/v/bin/mjlab-video-progress", "watch", "--config", "c.toml", "-d"],
        cwd=home,
        fds={0: "/dev/null", 1: log, 2: log, 3: lock},
    )
    fake_process(proc_root, 43, cmdline=["bash"], cwd=home, fds={0: "/dev/pts/1"})  # no lock
    fake_process(  # attached: stdout on a terminal
        proc_root,
        44,
        cmdline=["python3", "x", "watch", f"--config={tmp_path / 'abs.toml'}"],
        cwd=home,
        fds={1: "/dev/pts/1", 3: lock},
    )
    hz = os.sysconf("SC_CLK_TCK")
    assert find_watchers(proc_root) == [
        Watcher(42, home / "c.toml", (lock,), log, 1000 + 500 / hz),
        Watcher(44, tmp_path / "abs.toml", (lock,), None, 1000 + 500 / hz),
    ]
    assert find_watchers(proc_root, uid=os.getuid() + 1) == []


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")
def test_real_proc_lists_this_process_while_it_holds_a_lock(tmp_path):
    lock = lock_run(tmp_path / "exp/run")
    try:
        [me] = [w for w in find_watchers() if w.pid == os.getpid()]
        assert me.locks == (tmp_path / "exp/run" / LOCK_NAME,)
        assert me.started is not None
    finally:
        lock.close()
    assert os.getpid() not in [w.pid for w in find_watchers()]


def test_default_log(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / "cfg/progress-af.toml"
    assert default_log(config, {}) == tmp_path / ".local/state/mjlab-video-progress/progress-af.log"
    assert default_log(config, {"XDG_STATE_HOME": "/s"}).as_posix() == (
        "/s/mjlab-video-progress/progress-af.log"
    )


def test_stop_sends_sigterm_and_waits_until_gone(tmp_path, proc_root):
    for pid in (42, 43):
        fake_process(proc_root, pid, cmdline=["x"], cwd=tmp_path, fds={})
    sent = []

    def kill(pid, sig):
        sent.append((pid, sig))
        if pid == 42:  # exits at once; 43 becomes a zombie: both count as stopped
            (proc_root / "42/stat").unlink()
        else:
            (proc_root / "43/stat").write_text("43 (x) Z 1\n")

    assert stop([42, 43], proc_root, kill, timeout=5, sleep=lambda s: None) == []
    assert sent == [(42, signal.SIGTERM), (43, signal.SIGTERM)]


def test_stop_reports_what_is_still_running(tmp_path, proc_root):
    fake_process(proc_root, 42, cmdline=["x"], cwd=tmp_path, fds={})

    def gone(pid, sig):
        raise ProcessLookupError

    assert stop([42], proc_root, lambda pid, sig: None, timeout=0, sleep=lambda s: None) == [42]
    assert stop([7], proc_root, gone, timeout=0, sleep=lambda s: None) == []
