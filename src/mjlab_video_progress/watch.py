"""Watch: find the runs, then poll them, render what is due, store it and hand it to the sinks.

Contract:

- ``start(cfg, environ=None, proc_root=Path("/proc"), post=urlopen_post)`` returns a
  ``Session(runs, state, sinks)``. In this order it: reads the Telegram token if
  ``cfg.sinks.telegram`` is set (``read_token``; a problem raises ``ConfigError``) and, when the
  token came from the environment (no ``token_file``), removes that variable from ``environ``
  (``os.environ`` by default) so ``play`` never inherits it; discovers the runs once with
  ``discover_runs(cfg.watch.log_root, cfg.watch.task, proc_root)`` (a ``DiscoveryError``
  propagates); seeds ``state[run_dir]`` with ``last_stored_iteration(cfg.output.dir, run_dir)``
  for each run (``None`` when nothing is stored); and builds the sinks list: one callable for
  Telegram when configured (using ``post``), none otherwise. The Telegram sink is a closure, not a
  ``functools.partial``, so ``repr(session)`` never contains the token.
- ``watch(cfg, session, sleep=time.sleep, render=render_checkpoint)`` loops forever: each pass
  calls ``poll_run`` for every run in ``session.runs`` order, one at a time (renders never
  overlap), then ``sleep(cfg.watch.poll_seconds)``. An ``Exception`` from one run's pass is logged
  with its traceback and the loop goes on with the next run and the next pass; nothing else is
  caught, so ``KeyboardInterrupt`` and ``SystemExit`` (Ctrl-C, SIGTERM) end it, and
  ``render_checkpoint`` stops its ``play`` on the way out. It never returns normally.

- ``poll_run(cfg, run, state, sinks, render=render_checkpoint)`` does one pass for one run.
  ``state`` maps ``run.run_dir`` to the last iteration handled (a missing key means none yet) and
  is updated in place. It scans the run dir, asks ``next_due`` (``cfg.watch.every``) and returns
  when nothing is due. Otherwise, for the one checkpoint it picked (the newest due):
  1. it records that iteration in ``state`` **before** rendering, so whatever fails afterwards the
     same checkpoint is never tried again (no retry storm; the next attempt is the next due
     checkpoint);
  2. it removes ``<cfg.output.dir>/.work`` and calls
     ``render(cfg.render, run.task, checkpoint, <that work dir>)``, which recreates it; the work
     dir is left behind afterwards, so the last ``play.log`` can be read after a success or a
     failure;
  3. it stores the video with ``store_video(video, cfg.output.dir, run.run_dir, iteration)``;
  4. it calls each sink with ``Delivery(stored video, run.task, run.run_dir, iteration)``.
  A ``RenderError`` or ``OSError`` from steps 2-3 is logged as a warning (run, iteration, the
  error) and the pass ends: nothing is delivered. A ``SinkError`` or ``OSError`` from one sink is
  logged and the other sinks still run; the stored video stays. Any other exception (a bug) and
  an ``OSError`` from scanning (the run dir vanished) propagate; the loop that calls this decides.
- A sink is a callable taking a ``Delivery``. Logging is stdlib ``logging``.
- Read-only on the run dir; writes only under ``cfg.output.dir`` (hard rule 1).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mjlab_video_progress.config import Config, RenderConfig
from mjlab_video_progress.discovery import Run, discover_runs
from mjlab_video_progress.exceptions import RenderError, SinkError
from mjlab_video_progress.render import render_checkpoint
from mjlab_video_progress.scan import Checkpoint, next_due, scan_checkpoints
from mjlab_video_progress.sinks import Delivery
from mjlab_video_progress.store import last_stored_iteration, store_video
from mjlab_video_progress.telegram import Post, deliver_telegram, read_token, urlopen_post

log = logging.getLogger(__name__)


def _name(run: Run) -> str:
    return f"{run.run_dir.parent.name}/{run.run_dir.name}"


Render = Callable[[RenderConfig, str, Checkpoint, Path], Path]
Sink = Callable[[Delivery], None]


def poll_run(
    cfg: Config,
    run: Run,
    state: dict[Path, int | None],
    sinks: Sequence[Sink],
    render: Render = render_checkpoint,
) -> None:
    due = next_due(scan_checkpoints(run.run_dir), state.get(run.run_dir), cfg.watch.every)
    if due is None:
        return
    state[run.run_dir] = due.iteration  # before the render: whatever fails, never tried again
    name = _name(run)
    work_dir = cfg.output.dir / ".work"
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
        video = render(cfg.render, run.task, due, work_dir)
        stored = store_video(video, cfg.output.dir, run.run_dir, due.iteration)
    except (RenderError, OSError) as e:
        log.warning("render of %s iteration %d failed: %s", name, due.iteration, e)
        return
    log.info("stored %s iteration %d at %s", name, due.iteration, stored)
    delivery = Delivery(stored, run.task, run.run_dir, due.iteration)
    for sink in sinks:
        try:
            sink(delivery)
        except (SinkError, OSError) as e:  # never log the sink itself: its repr can hold a token
            log.warning("delivery of %s iteration %d failed: %s", name, due.iteration, e)


@dataclass(frozen=True)
class Session:
    runs: list[Run]
    state: dict[Path, int | None]
    sinks: list[Sink]


def start(
    cfg: Config,
    environ: MutableMapping[str, str] | None = None,
    proc_root: Path = Path("/proc"),
    post: Post = urlopen_post,
) -> Session:
    environ = os.environ if environ is None else environ
    sinks: list[Sink] = []
    telegram = cfg.sinks.telegram
    if telegram is not None:
        token = read_token(telegram, environ)
        if telegram.token_file is None:
            environ.pop(telegram.token_env, None)  # play, started later, must not inherit it

        def send(delivery: Delivery) -> None:  # a closure, not a partial: its repr hides the token
            deliver_telegram(telegram, token, delivery, post)

        sinks.append(send)
    runs = discover_runs(cfg.watch.log_root, cfg.watch.task, proc_root)
    state = {run.run_dir: last_stored_iteration(cfg.output.dir, run.run_dir) for run in runs}
    return Session(runs, state, sinks)


def watch(
    cfg: Config,
    session: Session,
    sleep: Callable[[float], None] = time.sleep,
    render: Render = render_checkpoint,
) -> None:
    while True:
        for run in session.runs:
            try:
                poll_run(cfg, run, session.state, session.sinks, render)
            except Exception:  # a bug costs this pass of this run, not the watcher
                log.exception("pass for %s failed", _name(run))
        sleep(cfg.watch.poll_seconds)
