"""Watch: poll the runs, render what is due, store it and hand it to the sinks.

Contract:

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
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from mjlab_video_progress.config import Config, RenderConfig
from mjlab_video_progress.discovery import Run
from mjlab_video_progress.exceptions import RenderError, SinkError
from mjlab_video_progress.render import render_checkpoint
from mjlab_video_progress.scan import Checkpoint, next_due, scan_checkpoints
from mjlab_video_progress.sinks import Delivery
from mjlab_video_progress.store import store_video

log = logging.getLogger(__name__)

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
    name = f"{run.run_dir.parent.name}/{run.run_dir.name}"
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
