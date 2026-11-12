"""Watchers: one watcher per run (a lock), list them, detach one, stop them.

Contract:

- ``lock_run(run_out)`` creates ``run_out`` (a run's output dir, ``store.run_output_dir``) and
  takes an exclusive, non-blocking ``flock`` on ``run_out/.watch.lock``. It returns the open
  file, which holds the lock until it is closed or the process dies (also on SIGKILL), or
  ``None`` when another process holds it. The file stays empty: who holds it is read from
  ``/proc``. An ``OSError`` (output dir not creatable) propagates.
- ``find_watchers(proc_root=Path("/proc"), uid=None)`` returns a ``Watcher`` for every process
  of ``uid`` (default: this user) that holds a ``.watch.lock`` file open (read from
  ``<pid>/fd``), sorted by pid. ``locks`` are those files; ``config`` is the value of
  ``--config`` on its command line, made absolute against its cwd (``None`` if absent);
  ``log`` is the file its stdout points to when that is a regular file (a detached watcher),
  else ``None``; ``started`` is its start time in Unix seconds (``<pid>/stat`` field 22 and
  ``btime`` from ``stat``), ``None`` if unreadable. A process that vanishes mid-scan is skipped.
  Read-only.
- ``default_log(config, environ=None)`` is ``$XDG_STATE_HOME/mjlab-video-progress/<config
  stem>.log`` (``~/.local/state`` when ``XDG_STATE_HOME`` is unset or empty).
- ``detach(log)`` forks twice with a ``setsid`` in between, like a daemon. In the original
  process it returns the pid of the detached process; in the detached process it returns
  ``None``, with stdin on ``/dev/null`` and stdout and stderr appended to ``log`` (created
  ``0600`` with its parent dirs). Open files, such as the run locks, are inherited. The cwd is
  kept. Call it before any thread starts.
- ``stop(pids, proc_root, kill=os.kill, timeout=STOP_TIMEOUT_SECONDS, sleep=time.sleep)`` sends
  SIGTERM to each pid, then waits until every one is gone or a zombie, or the timeout passes.
  It returns the pids still running. Callers pass only pids ``find_watchers`` listed: the tool
  never signals any other process (hard rule 1). No SIGKILL: a watcher killed mid-render would
  leave ``play`` running.
"""

from __future__ import annotations

import fcntl
import os
import signal
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

LOCK_NAME = ".watch.lock"
STOP_TIMEOUT_SECONDS = 30  # a watcher stops play first: SIGINT, then SIGKILL after 10 s


def lock_run(run_out: Path) -> IO[bytes] | None:
    run_out.mkdir(parents=True, exist_ok=True)
    f = open(run_out / LOCK_NAME, "ab")  # held open for the watcher's lifetime
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


@dataclass(frozen=True)
class Watcher:
    pid: int
    config: Path | None
    locks: tuple[Path, ...]
    log: Path | None
    started: float | None


def _config_arg(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.partition("=")[2]
    return None


def _started(proc_root: Path, entry: Path) -> float | None:
    try:
        fields = (entry / "stat").read_text().rpartition(")")[2].split()
        btime = next(
            int(line.split()[1])
            for line in (proc_root / "stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        return btime + int(fields[19]) / os.sysconf("SC_CLK_TCK")  # field 22: starttime
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _readlink(path: Path) -> str:
    try:
        return os.readlink(path)
    except OSError:  # closed mid-scan (the scan's own fd, for this process)
        return ""


def find_watchers(proc_root: Path = Path("/proc"), uid: int | None = None) -> list[Watcher]:
    uid = os.getuid() if uid is None else uid
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            targets = {fd.name: _readlink(fd) for fd in (entry / "fd").iterdir()}
            locks = tuple(sorted(Path(t) for t in targets.values() if Path(t).name == LOCK_NAME))
            if not locks:
                continue
            raw = (entry / "cmdline").read_bytes()
            cwd = Path(os.readlink(entry / "cwd"))
        except OSError:  # gone, or not ours to read
            continue
        config = _config_arg([os.fsdecode(a) for a in raw.removesuffix(b"\0").split(b"\0")])
        out = targets.get("1")
        log = Path(out) if out and os.path.isfile(out) else None
        found.append(
            Watcher(
                pid=int(entry.name),
                config=cwd / config if config else None,
                locks=locks,
                log=log,
                started=_started(proc_root, entry),
            )
        )
    return sorted(found, key=lambda w: w.pid)


def default_log(config: Path, environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    state = environ.get("XDG_STATE_HOME") or Path.home() / ".local/state"
    return Path(state) / "mjlab-video-progress" / f"{config.stem}.log"


def detach(log: Path) -> int | None:
    log.parent.mkdir(parents=True, exist_ok=True)
    out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    read_end, write_end = os.pipe()
    for stream in (sys.stdout, sys.stderr):
        stream.flush()  # or the children print it again
    child = os.fork()
    if child:  # the original process
        os.close(write_end)
        os.close(out)
        pid = os.read(read_end, 32)
        os.close(read_end)
        os.waitpid(child, 0)
        return int(pid)
    os.close(read_end)
    os.setsid()
    if os.fork():
        os._exit(0)  # the middle process: the detached one is no session leader, gets no tty
    os.write(write_end, str(os.getpid()).encode())
    os.close(write_end)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(out, 1)
    os.dup2(out, 2)
    os.close(null)
    os.close(out)
    return None


def _running(proc_root: Path, pid: int) -> bool:
    try:
        return (proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def stop(
    pids: Iterable[int],
    proc_root: Path = Path("/proc"),
    kill: Callable[[int, int], None] = os.kill,
    timeout: float = STOP_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int]:
    pids = list(pids)
    for pid in pids:
        try:
            kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone
    deadline = time.monotonic() + timeout
    while True:
        left = [pid for pid in pids if _running(proc_root, pid)]
        if not left or time.monotonic() >= deadline:
            return left
        sleep(0.2)
