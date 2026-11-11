"""Store: where finished videos live in the output dir.

Contract:

- Layout: ``<output_dir>/<experiment>/<run>/iter_<N>/video.mp4``. Experiment and run are
  ``run_dir.parent.name`` and ``run_dir.name``; ``run_dir`` is used for those two names only,
  need not exist and is never touched. ``N`` is the iteration zero-padded to 6 digits (a wider
  number keeps all its digits). The ``iter_<N>`` dir is the bundle; in v1 it holds only
  ``video.mp4``, so later files can sit next to it.
- ``store_video(video, output_dir, run_dir, iteration)`` copies ``video`` to that path and
  returns it, creating the dirs. It copies to a temp file in the same dir and ``os.replace``s it,
  so ``video.mp4`` is whole or absent, and storing an iteration again replaces the earlier video
  atomically. The source is left untouched (the caller cleans up its work dir). An ``OSError``
  (``FileNotFoundError`` for a missing source) propagates, leaves no temp file and leaves any
  earlier ``video.mp4`` intact.
- ``last_stored_iteration(output_dir, run_dir)`` returns the highest ``N`` (numerically) among
  the run's ``iter_<N>`` dirs that contain a ``video.mp4``, or ``None`` if there is none, or the
  output dir or the run's dir does not exist yet. A dir without ``video.mp4``, files, other
  names and other runs are ignored. ``watch`` uses it to seed ``last_rendered`` after a restart.
  Read-only.
- Writes only under ``output_dir``.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

_ITER = re.compile(r"iter_([0-9]+)")


def _run_out(output_dir: Path, run_dir: Path) -> Path:
    return Path(output_dir) / run_dir.parent.name / run_dir.name


def store_video(video: Path, output_dir: Path, run_dir: Path, iteration: int) -> Path:
    dest = _run_out(output_dir, run_dir) / f"iter_{iteration:06d}" / "video.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("video.mp4.tmp")  # one render at a time, so a fixed name is enough
    try:
        shutil.copyfile(video, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)  # only left behind when the copy failed
    return dest


def last_stored_iteration(output_dir: Path, run_dir: Path) -> int | None:
    try:
        entries = list(_run_out(output_dir, run_dir).iterdir())
    except FileNotFoundError:
        return None
    found = [
        int(m[1]) for p in entries if (m := _ITER.fullmatch(p.name)) and (p / "video.mp4").is_file()
    ]
    return max(found, default=None)
