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

Unverified: only a uv-venv command line from a single live run was captured. A system-Python
install, ``python -m mjlab.scripts.train``, other launchers, multi-GPU children and wrapper
variations have not been observed. Processes that rewrite their own cmdline are not handled.

Status: implemented; verified only by tests/test_discovery.py against a fake ``/proc`` tree.
The real-``/proc`` smoke test runs on Linux only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
