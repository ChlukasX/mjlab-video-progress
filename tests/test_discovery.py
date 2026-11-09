"""Contract tests for discovery: parse_train_argv, find_train_processes and find_run_dir."""

from __future__ import annotations

import dataclasses
import os
import socket

import pytest

from mjlab_video_progress.discovery import (
    RunDirNotFound,
    TrainCommand,
    TrainProcess,
    find_run_dir,
    find_train_processes,
    parse_train_argv,
)

UV_BIN = "/home/user/project/.venv/bin"  # `uv sync` / `uv run` layout
SYS_BIN = "/usr/local/bin"  # plain pip install into a default environment
TASK = "Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-MidThigh"
# Shape of the real command line captured from a live uv-based run (flags abbreviated).
REAL = [
    f"{UV_BIN}/python", f"{UV_BIN}/train", TASK,
    "--agent.logger", "wandb", "--env.scene.num-envs", "4096",
    "--agent.run-name", "cur7-midthigh", "--agent.resume", "True",
]  # fmt: skip


def test_real_command_line_is_parsed():
    assert parse_train_argv(REAL) == TrainCommand(task=TASK, log_root=None)


def test_result_is_frozen():
    cmd = parse_train_argv(REAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.task = "x"  # type: ignore[union-attr,misc]


@pytest.mark.parametrize("flags", [["--log-root", "logs/x"], ["--log-root=logs/x"]])
def test_log_root_both_spellings(flags):
    assert parse_train_argv(REAL + flags).log_root == "logs/x"


@pytest.mark.parametrize(
    ("interp", "script"),
    [
        (f"{UV_BIN}/python", f"{UV_BIN}/train"),  # uv venv (the captured shape)
        ("/usr/bin/python3", f"{SYS_BIN}/train"),  # default system python
        ("python3.13", f"{SYS_BIN}/train"),  # bare interpreter name
    ],
)
def test_python_interpreter_and_script_locations_match(interp, script):
    assert parse_train_argv([interp, script, TASK]) == TrainCommand(task=TASK, log_root=None)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["uv", "run", "train", TASK, "--agent.logger", "wandb"],  # wrapper
        ["bash", "scripts/train.sh", TASK],  # wrapper script
        ["zsh", "-c", f"bash scripts/train.sh {TASK}"],  # shell
        ["notpython", f"{UV_BIN}/train", TASK],
        [f"{UV_BIN}/python", f"{UV_BIN}/play", TASK],  # other mjlab script
        [f"{UV_BIN}/python", f"{UV_BIN}/train"],  # no task
        [f"{UV_BIN}/python", f"{UV_BIN}/train", "--agent.logger", "wandb"],  # flag, not a task
        [f"{UV_BIN}/python", f"{UV_BIN}/train", TASK, "--log-root"],  # flag without a value
    ],
)
def test_non_matching_command_lines_return_none(argv):
    assert parse_train_argv(argv) is None


def add_proc(proc, pid, argv):
    """Create proc/<pid>/cmdline the way Linux stores it: a NUL after every argument."""
    d = proc / str(pid)
    d.mkdir()
    (d / "cmdline").write_bytes(b"".join(os.fsencode(a) + b"\0" for a in argv))
    return d


def test_finds_matching_process(tmp_path):
    add_proc(tmp_path, 1234, REAL)
    assert find_train_processes(tmp_path) == [TrainProcess(1234, TrainCommand(task=TASK))]


def test_non_matching_processes_skipped(tmp_path):
    add_proc(tmp_path, 10, ["uv", "run", "train", TASK])  # wrapper
    add_proc(tmp_path, 11, ["bash", "scripts/train.sh", TASK])  # wrapper script
    add_proc(tmp_path, 12, [])  # kernel thread: empty cmdline
    assert find_train_processes(tmp_path) == []


def test_non_pid_entries_ignored(tmp_path):
    add_proc(tmp_path, "self", REAL)
    add_proc(tmp_path, "sys", REAL)
    (tmp_path / "cpuinfo").write_text("x")
    assert find_train_processes(tmp_path) == []


def test_other_user_skipped(tmp_path):
    add_proc(tmp_path, 1234, REAL)
    assert find_train_processes(tmp_path, uid=os.getuid() + 1) == []


def test_vanished_process_skipped(tmp_path):
    (tmp_path / "77").mkdir()  # exited between listing and reading cmdline
    add_proc(tmp_path, 1234, REAL)
    assert [p.pid for p in find_train_processes(tmp_path)] == [1234]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file")
def test_unreadable_cmdline_skipped(tmp_path):
    (add_proc(tmp_path, 77, REAL) / "cmdline").chmod(0)
    add_proc(tmp_path, 1234, REAL)
    assert [p.pid for p in find_train_processes(tmp_path)] == [1234]


def test_sorted_by_pid_numerically(tmp_path):
    add_proc(tmp_path, 20, REAL)
    add_proc(tmp_path, 3, REAL)
    assert [p.pid for p in find_train_processes(tmp_path)] == [3, 20]


def test_non_utf8_argument_does_not_crash(tmp_path):
    d = add_proc(tmp_path, 1234, REAL)
    with (d / "cmdline").open("ab") as f:
        f.write(b"--agent.run-name\0caf\xe9\0")  # Latin-1 byte, invalid UTF-8
    assert [p.pid for p in find_train_processes(tmp_path)] == [1234]


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")
def test_real_proc_smoke():
    assert isinstance(find_train_processes(), list)


PID = 1234


def train_process(tmp_path, log_root=None, pid=PID):
    """Fake proc/<pid>/cwd -> repo (like Linux's magic link); returns (proc, repo, process)."""
    proc, repo = tmp_path / "proc", tmp_path / "repo"
    (proc / str(pid)).mkdir(parents=True)
    repo.mkdir()
    (proc / str(pid) / "cwd").symlink_to(repo)
    return proc, repo, TrainProcess(pid, TrainCommand(task=TASK, log_root=log_root))


def add_run(log_root, pid, run="exp/2026-11-09_12-00-00", host=None):
    """Create <log_root>/<experiment>/<run>/ with params/ and the tfevents file of pid."""
    d = log_root / run
    (d / "params").mkdir(parents=True)
    (d / "params" / "agent.yaml").write_text("x")
    host = socket.gethostname() if host is None else host
    (d / f"events.out.tfevents.1762689600.{host}.{pid}.0").write_bytes(b"\0")
    return d


def test_run_dir_under_default_log_root(tmp_path):
    proc, repo, process = train_process(tmp_path)
    run = add_run(repo / "logs/rsl_rl", PID)
    assert find_run_dir(process, proc) == run


def test_relative_log_root_resolves_against_cwd(tmp_path):
    proc, repo, process = train_process(tmp_path, log_root="custom/x")
    run = add_run(repo / "custom/x", PID)
    assert find_run_dir(process, proc) == run


def test_absolute_log_root_ignores_cwd(tmp_path):
    proc, _, process = train_process(tmp_path, log_root=str(tmp_path / "elsewhere"))
    run = add_run(tmp_path / "elsewhere", PID)
    assert find_run_dir(process, proc) == run


def test_any_experiment_name(tmp_path):
    proc, repo, process = train_process(tmp_path)
    add_run(repo / "logs/rsl_rl", 999, "a_exp/2026-11-01_10-00-00")
    run = add_run(repo / "logs/rsl_rl", PID, "z_exp/2026-11-09_12-00-00_cur7")
    assert find_run_dir(process, proc) == run


def test_pid_with_matching_suffix_is_not_a_match(tmp_path):
    proc, repo, process = train_process(tmp_path)
    add_run(repo / "logs/rsl_rl", 51234)
    with pytest.raises(RunDirNotFound):
        find_run_dir(process, proc)


def test_dotted_hostname_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu1.lab.local")
    proc, repo, process = train_process(tmp_path)
    run = add_run(repo / "logs/rsl_rl", PID, host="gpu1.lab.local")
    assert find_run_dir(process, proc) == run


def test_same_pid_on_other_host_is_not_a_match(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu1")
    proc, repo, process = train_process(tmp_path)
    add_run(repo / "logs/rsl_rl", PID, host="gpu2")  # another node on a shared NFS home
    with pytest.raises(RunDirNotFound):
        find_run_dir(process, proc)


def test_starting_run_without_tfevents_not_found(tmp_path):
    proc, repo, process = train_process(tmp_path)
    (repo / "logs/rsl_rl/exp/2026-11-09_12-00-00/params").mkdir(parents=True)
    with pytest.raises(RunDirNotFound) as e:
        find_run_dir(process, proc)
    assert f"pid {PID}" in str(e.value) and str(repo / "logs/rsl_rl") in str(e.value)


def test_missing_log_root_not_found(tmp_path):
    proc, _, process = train_process(tmp_path)
    with pytest.raises(RunDirNotFound):
        find_run_dir(process, proc)


def test_vanished_process_not_found(tmp_path):
    proc, repo, _ = train_process(tmp_path)
    add_run(repo / "logs/rsl_rl", 77)
    with pytest.raises(RunDirNotFound, match="pid 77"):
        find_run_dir(TrainProcess(77, TrainCommand(task=TASK)), proc)  # no proc/77/cwd


def test_two_runs_for_same_pid_are_ambiguous(tmp_path):
    proc, repo, process = train_process(tmp_path)
    old = add_run(repo / "logs/rsl_rl", PID, "exp/2026-01-01_09-00-00")  # pid reused
    new = add_run(repo / "logs/rsl_rl", PID, "exp/2026-11-09_12-00-00")
    with pytest.raises(RunDirNotFound) as e:
        find_run_dir(process, proc)
    assert str(old) in str(e.value) and str(new) in str(e.value)


def snapshot(root):
    """Bytes (files) and st_mtime_ns of every path under root, root included."""
    return {
        p: (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
        for p in [root, *root.rglob("*")]
    }


def test_lookup_leaves_logs_tree_identical(tmp_path):
    proc, repo, process = train_process(tmp_path)
    add_run(repo / "logs/rsl_rl", PID)
    add_run(repo / "logs/rsl_rl", 999, "exp/2026-11-01_10-00-00")
    before = snapshot(repo / "logs")
    find_run_dir(process, proc)
    assert snapshot(repo / "logs") == before


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")
def test_real_proc_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = add_run(tmp_path / "logs/rsl_rl", os.getpid())
    process = TrainProcess(os.getpid(), TrainCommand(task=TASK))
    assert find_run_dir(process) == run.resolve()
