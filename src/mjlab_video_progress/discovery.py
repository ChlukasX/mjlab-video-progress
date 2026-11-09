"""Discovery: find the running mjlab training and its run directory (read-only).

Contract for ``parse_train_argv`` (tests/test_discovery.py is the specification):

- Takes a process argument list and returns a frozen ``TrainCommand(task, log_root)``, or
  ``None`` when it is not a mjlab ``train`` process.
- Strict default rule: ``argv[0]`` is a Python interpreter (basename ``python``, ``python3`` or
  ``python3.N``), ``argv[1]`` has the basename ``train``, and ``argv[2]`` is the task id and
  does not start with ``-``. Wrapper processes (``uv run``, ``bash scripts/train.sh``) do not
  match, so one training is counted once.
- ``--log-root X`` and ``--log-root=X`` are read; a ``--log-root`` with no value makes the
  whole command line a non-match.

Contract for ``find_train_processes(proc_root, uid)``:

- Lists the numeric entries of ``proc_root`` (default ``/proc``), keeps those whose directory is
  owned by ``uid`` (default ``os.getuid()``), and runs their NUL-separated ``cmdline`` (decoded
  with ``os.fsdecode``) through ``parse_train_argv``.
- Returns ``TrainProcess(pid, command)`` for every match, sorted by pid. A process that vanishes
  or cannot be read mid-scan is skipped. Read-only: never signals or writes anything.

Contract for ``find_run_dir(process, proc_root)``:

- Reads the process cwd with ``os.readlink(proc_root/<pid>/cwd)`` and joins the log root
  (``--log-root``, else mjlab's default ``logs/rsl_rl``; an absolute one ignores the cwd).
- Returns the run dir ``<log root>/<experiment>/<run>`` that holds the one file matching
  ``events.out.tfevents.*.<hostname>.<pid>.0`` (hostname from ``socket.gethostname()``).
- Raises ``RunDirNotFound`` naming the pid and why: cwd unreadable (process gone), no match
  (wrong root, or the run is still starting: tfevents appear only after env setup), or several
  matches (a reused pid; it never guesses). Read-only: one readlink and one glob.

Contract for ``find_latest_run_dir(log_root)`` (the directory fallback):

- ``log_root`` means the same as mjlab's ``--log-root``: the dir holding ``<experiment>/<run>``.
- Candidates are the dirs exactly two levels down named like mjlab's run dirs,
  ``YYYY-MM-DD_HH-MM-SS[_<run_name>]``; anything else (``wandb_checkpoints``, stray dirs, files,
  deeper dirs) is ignored. Returns the candidate with the highest name across all experiments
  (name order is start order). mtime is never read, so a starting run with only ``params/``
  counts and a finished run is returned when it is the newest.
- Raises ``RunDirNotFound`` naming ``log_root`` and why: not a directory, no candidate, or the
  newest name in several experiments (it never guesses). Read-only: one glob and a stat per
  candidate.

Contract for ``discover_runs(log_root, task, proc_root)`` (``watch.log_root``, ``watch.task``):

- Training processes found: a ``Run(run_dir, task)`` per process with a run dir, in pid order,
  with the process's own task (a different ``task`` is logged as a warning). The rest are skipped
  with a warning; if none is left, raises ``RunDirNotFound`` with each pid's reason (no fallback).
- No training process: ``Run(find_latest_run_dir(log_root), task)``; raises ``DiscoveryError``
  (base of ``RunDirNotFound``) if ``log_root`` or ``task`` is ``None``.

Unverified: only a uv-venv command line from a single live run was captured. A system-Python
install, ``python -m mjlab.scripts.train``, other launchers, multi-GPU children and wrapper
variations have not been observed. Processes that rewrite their own cmdline are not handled.
``find_run_dir`` follows the mjlab 1.6.0 and tensorboard source and has not been run against a
live training. Not handled: multi-GPU runs (torchrunx workers write tfevents with their own
pid), PID namespaces (a container watched from the host) and a single stale match left by a
reused pid while the live run is still starting. ``find_latest_run_dir`` follows the mjlab
1.6.0 source only; it cannot tell a finished run from a live one, returns the previous run while
a new one is still in env setup (no run dir yet), and can misorder runs across a clock or
time-zone change (names are local time).

Status: implemented; verified only by tests/test_discovery.py against a fake ``/proc`` tree and
fake logs trees. The real-``/proc`` tests run on Linux only.
"""

from __future__ import annotations

import logging
import os
import re
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)
_PYTHON = re.compile(r"python(3(\.\d+)*)?")


@dataclass(frozen=True)
class TrainCommand:
    task: str
    log_root: str | None = None


def parse_train_argv(argv: Sequence[str]) -> TrainCommand | None:
    if len(argv) < 3 or not _PYTHON.fullmatch(os.path.basename(argv[0])):
        return None
    if os.path.basename(argv[1]) != "train" or argv[2].startswith("-"):
        return None
    log_root = None
    for i, tok in enumerate(argv[3:], start=3):
        if tok == "--log-root":
            if i + 1 >= len(argv):
                return None
            log_root = argv[i + 1]
        elif tok.startswith("--log-root="):
            log_root = tok.removeprefix("--log-root=")
    return TrainCommand(task=argv[2], log_root=log_root)


@dataclass(frozen=True)
class TrainProcess:
    pid: int
    command: TrainCommand


def find_train_processes(
    proc_root: Path = Path("/proc"), uid: int | None = None
) -> list[TrainProcess]:
    uid = os.getuid() if uid is None else uid
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            raw = (entry / "cmdline").read_bytes()
        except OSError:  # process exited or is unreadable mid-scan
            continue
        argv = [os.fsdecode(a) for a in raw.removesuffix(b"\0").split(b"\0")]
        command = parse_train_argv(argv)
        if command:
            found.append(TrainProcess(int(entry.name), command))
    return sorted(found, key=lambda p: p.pid)


_DEFAULT_LOG_ROOT = "logs/rsl_rl"  # mjlab 1.6.0 TrainConfig.log_root


class DiscoveryError(LookupError):
    """Discovery found nothing to watch; the message says why."""


class RunDirNotFound(DiscoveryError):
    """The run dir of a training process could not be located; the message says why."""


def find_run_dir(process: TrainProcess, proc_root: Path = Path("/proc")) -> Path:
    pid, log_root = process.pid, process.command.log_root
    try:
        cwd = os.readlink(proc_root / str(pid) / "cwd")
    except OSError as e:  # process exited or is unreadable
        raise RunDirNotFound(f"pid {pid}: cannot read cwd ({e})") from e
    root = Path(cwd) / (_DEFAULT_LOG_ROOT if log_root is None else log_root)
    pattern = f"*/*/events.out.tfevents.*.{socket.gethostname()}.{pid}.0"
    runs = sorted(p.parent for p in root.glob(pattern))
    if not runs:
        raise RunDirNotFound(f"pid {pid}: no {pattern} under {root} (run may still be starting)")
    if len(runs) > 1:
        raise RunDirNotFound(f"pid {pid}: ambiguous, {len(runs)} run dirs match: {runs}")
    return runs[0]


# mjlab 1.6.0 launch_training: datetime.now().strftime("%Y-%m-%d_%H-%M-%S") [+ "_" + run_name].
_RUN_NAME = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_.+)?")


def find_latest_run_dir(log_root: Path) -> Path:
    if not log_root.is_dir():
        raise RunDirNotFound(f"log root {log_root} is not a directory")
    runs = [p for p in log_root.glob("*/*") if _RUN_NAME.fullmatch(p.name) and p.is_dir()]
    if not runs:
        raise RunDirNotFound(f"no <experiment>/<YYYY-MM-DD_HH-MM-SS[_name]> dir under {log_root}")
    newest = max(p.name for p in runs)
    latest = sorted(p for p in runs if p.name == newest)
    if len(latest) > 1:
        raise RunDirNotFound(f"ambiguous, {len(latest)} run dirs named {newest}: {latest}")
    return latest[0]


@dataclass(frozen=True)
class Run:
    run_dir: Path
    task: str


def discover_runs(
    log_root: Path | None, task: str | None, proc_root: Path = Path("/proc")
) -> list[Run]:
    processes = find_train_processes(proc_root)
    if not processes:
        if log_root is None:
            raise DiscoveryError("no training process found and watch.log_root is not set")
        run_dir = find_latest_run_dir(log_root)
        if task is None:
            raise DiscoveryError(f"no training process found; set watch.task for {run_dir}")
        return [Run(run_dir, task)]
    runs, reasons = [], []
    for p in processes:
        try:
            runs.append(Run(find_run_dir(p, proc_root), p.command.task))
        except RunDirNotFound as e:
            reasons.append(str(e))
            continue
        if task is not None and task != p.command.task:
            _log.warning("pid %d trains %s, not watch.task %s", p.pid, p.command.task, task)
    if not runs:
        raise RunDirNotFound("no training process has a run dir: " + "; ".join(reasons))
    for reason in reasons:
        _log.warning("skipping training process, %s", reason)
    return runs
