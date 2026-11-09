"""Contract tests for discovery.parse_train_argv."""

from __future__ import annotations

import dataclasses

import pytest

from mjlab_video_progress.discovery import TrainCommand, parse_train_argv

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
