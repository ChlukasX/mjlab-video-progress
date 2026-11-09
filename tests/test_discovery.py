"""Contract tests for discovery: parse_train_argv and find_train_processes."""

from __future__ import annotations

import dataclasses
import os

import pytest

from mjlab_video_progress.discovery import (
    TrainCommand,
    TrainProcess,
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
