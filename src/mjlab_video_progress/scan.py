"""Scan: list the finished checkpoints in a run dir and pick the one due for rendering.

Contract:

- ``scan_checkpoints(run_dir)`` returns a ``Checkpoint(iteration, path)`` for every regular file
  named exactly ``model_<int>.pt`` directly in ``run_dir``, sorted by iteration (numerically;
  ``model_0.pt`` counts). Everything else is ignored: other names (``*.onnx``, ``policy.pt``,
  ``model_5.pt.tmp``), directories (even one named ``model_7.pt``) and subdirectories.
- Only finished files are returned. mjlab writes a checkpoint in place with ``torch.save``
  (a zip whose end record is written last), so a file without a valid zip end record is still
  being written and is skipped; the next scan sees it once complete.
- A missing or unreadable ``run_dir`` raises ``OSError`` (``FileNotFoundError`` if missing).
- Read-only on the run dir (hard rule 1): lists the directory and reads the tail of each
  matching file; never writes or changes mtimes.
- ``next_due(checkpoints, last_rendered, every)`` is pure. It returns the newest checkpoint
  (by iteration, whatever the input order) with ``iteration >= last_rendered + every``, or
  ``None``. ``last_rendered=None`` (nothing rendered yet) counts as 0, so the first render waits
  for a checkpoint at iteration ``every`` or later. Several due: only the newest is returned, the
  rest are skipped. The threshold is relative to ``last_rendered``, not a multiple of ``every``,
  so an ``every`` that is not a multiple of mjlab's ``save_interval`` still fires (the effective
  interval is ``every`` rounded up to a multiple of ``save_interval``). ``every`` is not
  validated; config guarantees a positive int.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

_NAME = re.compile(r"model_([0-9]+)\.pt")


@dataclass(frozen=True)
class Checkpoint:
    iteration: int
    path: Path


def scan_checkpoints(run_dir: Path) -> list[Checkpoint]:
    found = []
    for path in Path(run_dir).iterdir():
        m = _NAME.fullmatch(path.name)
        # is_zipfile reads only the file's tail and is False for a file still being written.
        if m and path.is_file() and zipfile.is_zipfile(path):
            found.append(Checkpoint(int(m[1]), path))
    return sorted(found, key=lambda c: c.iteration)


def next_due(
    checkpoints: list[Checkpoint], last_rendered: int | None, every: int
) -> Checkpoint | None:
    threshold = (last_rendered or 0) + every
    return max(
        (c for c in checkpoints if c.iteration >= threshold),
        key=lambda c: c.iteration,
        default=None,
    )
