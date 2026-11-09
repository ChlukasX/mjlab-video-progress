"""Contract tests for discovery: parse_train_argv, find_train_processes, find_run_dir,
find_latest_run_dir and discover_runs."""

from __future__ import annotations

import dataclasses
import os
import socket

import pytest

from mjlab_video_progress.discovery import (
    DiscoveryError,
    Run,
    RunDirNotFound,
    TrainCommand,
    TrainProcess,
    discover_runs,
    find_latest_run_dir,
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


def test_tilde_log_root_is_literal_like_mjlab(tmp_path):
    proc, repo, process = train_process(tmp_path, log_root="~/x")  # `--log-root=~/x`: unexpanded
    run = add_run(repo / "~/x", PID)  # mjlab joins it as-is, so it writes <cwd>/~/x
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


# find_latest_run_dir: the pid add_run writes into the tfevents name does not matter here.


def test_latest_single_run(tmp_path):
    run = add_run(tmp_path, PID)
    assert find_latest_run_dir(tmp_path) == run


def test_latest_by_name_across_experiments(tmp_path):
    new = add_run(tmp_path, PID, "a_exp/2026-11-09_12-00-00")
    add_run(tmp_path, PID, "z_exp/2026-11-01_10-00-00")  # experiment name sorts later
    assert find_latest_run_dir(tmp_path) == new


def test_run_name_suffix_sorted_by_timestamp(tmp_path):
    new = add_run(tmp_path, PID, "exp/2026-11-09_12-00-00_cur7")
    add_run(tmp_path, PID, "exp/2026-11-09_11-00-00_zzz")
    assert find_latest_run_dir(tmp_path) == new


def test_name_wins_over_mtime(tmp_path):
    new = add_run(tmp_path, PID, "exp/2026-11-09_12-00-00")
    add_run(tmp_path, PID, "exp/2026-11-01_10-00-00")  # written just now
    for p in [new, *new.rglob("*")]:
        os.utime(p, (1_000_000_000, 1_000_000_000))  # newer-named run untouched since 2001
    assert find_latest_run_dir(tmp_path) == new


def test_starting_run_without_tfevents_is_latest(tmp_path):
    add_run(tmp_path, PID, "exp/2026-11-01_10-00-00")
    starting = tmp_path / "exp/2026-11-09_12-00-00"
    (starting / "params").mkdir(parents=True)  # params dumped, runner not built yet
    assert find_latest_run_dir(tmp_path) == starting


def test_non_run_dirs_ignored(tmp_path):
    run = add_run(tmp_path, PID, "exp/2026-11-01_10-00-00")
    cache = tmp_path / "exp/wandb_checkpoints/abc123"  # mjlab's W&B download cache
    cache.mkdir(parents=True)
    (cache / "model_100.pt").write_bytes(b"\0")
    (tmp_path / "exp/zzz").mkdir()  # sorts after every timestamp
    (tmp_path / "exp/2026-12-01_00-00-00.bak").mkdir()  # not mjlab's name format
    (tmp_path / "exp/2026-12-02_00-00-00").write_text("x")  # a file, not a dir
    (tmp_path / "exp/deep/2026-12-03_00-00-00").mkdir(parents=True)  # three levels down
    (tmp_path / "2026-12-04_00-00-00").mkdir()  # one level down (an experiment dir)
    assert find_latest_run_dir(tmp_path) == run


def test_latest_missing_log_root_not_found(tmp_path):
    with pytest.raises(RunDirNotFound) as e:
        find_latest_run_dir(tmp_path / "missing")
    assert str(tmp_path / "missing") in str(e.value)


def test_no_runs_not_found(tmp_path):
    (tmp_path / "exp/wandb_checkpoints/abc123").mkdir(parents=True)
    with pytest.raises(RunDirNotFound) as e:
        find_latest_run_dir(tmp_path)
    assert str(tmp_path) in str(e.value)


def test_same_name_in_two_experiments_ambiguous(tmp_path):
    a = add_run(tmp_path, PID, "a_exp/2026-11-09_12-00-00")
    b = add_run(tmp_path, PID, "b_exp/2026-11-09_12-00-00")
    add_run(tmp_path, PID, "c_exp/2026-11-01_10-00-00")
    with pytest.raises(RunDirNotFound) as e:
        find_latest_run_dir(tmp_path)
    assert str(a) in str(e.value) and str(b) in str(e.value)


def test_fallback_leaves_logs_tree_identical(tmp_path):
    add_run(tmp_path, PID, "a_exp/2026-11-09_12-00-00")
    add_run(tmp_path, PID, "b_exp/2026-11-01_10-00-00")
    (tmp_path / "a_exp/wandb_checkpoints/abc123").mkdir(parents=True)
    before = snapshot(tmp_path)
    find_latest_run_dir(tmp_path)
    assert snapshot(tmp_path) == before


# discover_runs: fake proc/<pid> gets both cmdline and the cwd link; logs is mjlab's default root.

OTHER = "Mjlab-Velocity-Flat-Unitree-G1"


def glue_dirs(tmp_path):
    """Returns (proc, repo, logs); proc exists and is empty, repo is every process's cwd."""
    proc, repo = tmp_path / "proc", tmp_path / "repo"
    proc.mkdir()
    return proc, repo, repo / "logs/rsl_rl"


def add_train_proc(proc, pid, cwd, task=TASK):
    d = add_proc(proc, pid, [f"{UV_BIN}/python", f"{UV_BIN}/train", task])
    (d / "cwd").symlink_to(cwd)


def test_process_run_found(tmp_path, caplog):
    proc, repo, logs = glue_dirs(tmp_path)
    add_train_proc(proc, PID, repo)
    run = add_run(logs, PID)
    assert discover_runs(None, TASK, proc) == [Run(run, TASK)]
    assert caplog.records == []  # same task as watch.task: nothing to warn about


def test_several_processes_all_returned(tmp_path):
    proc, repo, logs = glue_dirs(tmp_path)
    add_train_proc(proc, 20, repo, task=OTHER)
    add_train_proc(proc, 3, repo)
    late = add_run(logs, 3, "exp/2026-11-09_13-00-00")  # pid order, not run-name order
    early = add_run(logs, 20, "exp/2026-11-09_12-00-00")
    assert discover_runs(None, None, proc) == [Run(late, TASK), Run(early, OTHER)]


def test_process_task_wins_over_watch_task(tmp_path, caplog):
    proc, repo, logs = glue_dirs(tmp_path)
    add_train_proc(proc, PID, repo)
    run = add_run(logs, PID)
    assert discover_runs(None, OTHER, proc) == [Run(run, TASK)]
    (record,) = caplog.records
    assert record.levelname == "WARNING"
    assert TASK in record.getMessage() and OTHER in record.getMessage()


def test_process_without_run_dir_skipped_with_warning(tmp_path, caplog):
    proc, repo, logs = glue_dirs(tmp_path)
    add_train_proc(proc, 3, repo)  # still starting: no tfevents yet
    add_train_proc(proc, PID, repo)
    run = add_run(logs, PID)
    assert discover_runs(None, None, proc) == [Run(run, TASK)]
    (record,) = caplog.records
    assert record.levelname == "WARNING"
    assert "pid 3" in record.getMessage() and str(logs) in record.getMessage()


def test_processes_found_but_none_has_run_dir_raises(tmp_path):
    proc, repo, logs = glue_dirs(tmp_path)
    add_train_proc(proc, 3, repo)
    add_train_proc(proc, PID, repo)
    add_run(logs, 999)  # the fallback would pick this run; it must not run
    with pytest.raises(RunDirNotFound) as e:
        discover_runs(logs, TASK, proc)
    assert "pid 3" in str(e.value) and f"pid {PID}" in str(e.value)


def test_fallback_when_no_process(tmp_path):
    proc, _, logs = glue_dirs(tmp_path)
    add_proc(proc, 10, ["uv", "run", "train", TASK])  # wrapper only: no training process
    add_run(logs, 999, "exp/2026-11-01_10-00-00")
    latest = add_run(logs, 999, "exp/2026-11-09_12-00-00")
    assert discover_runs(logs, OTHER, proc) == [Run(latest, OTHER)]


def test_fallback_needs_watch_task(tmp_path):
    proc, _, logs = glue_dirs(tmp_path)
    run = add_run(logs, 999)
    with pytest.raises(DiscoveryError) as e:
        discover_runs(logs, None, proc)
    assert not isinstance(e.value, RunDirNotFound)  # the run dir was found
    assert str(run) in str(e.value) and "watch.task" in str(e.value)


def test_no_process_and_no_log_root_raises(tmp_path):
    proc, _, _ = glue_dirs(tmp_path)
    with pytest.raises(DiscoveryError) as e:
        discover_runs(None, TASK, proc)
    assert not isinstance(e.value, RunDirNotFound)
    assert "watch.log_root" in str(e.value)


def test_fallback_error_propagates(tmp_path):
    proc, _, _ = glue_dirs(tmp_path)
    with pytest.raises(RunDirNotFound) as e:
        discover_runs(tmp_path / "missing", TASK, proc)
    assert isinstance(e.value, DiscoveryError)  # one type for the CLI to catch
    assert str(tmp_path / "missing") in str(e.value)
