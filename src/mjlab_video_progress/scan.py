"""Scan: list the finished checkpoints in a run dir.

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
