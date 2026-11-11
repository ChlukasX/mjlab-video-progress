"""Sinks: what every sink is handed for one stored video.

Contract:

- ``Delivery(video, task, run_dir, iteration)`` is frozen. ``video`` is the stored mp4 (see
  ``store``), ``task`` the mjlab task name, ``run_dir`` the run's dir in the logs (a sink uses
  only its last two names, experiment and run, and never touches it) and ``iteration`` the
  checkpoint iteration the video shows.
- A sink delivers the video only (v1). It raises ``SinkError`` when it cannot deliver and never
  writes under ``logs/`` or ``wandb/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Delivery:
    video: Path
    task: str
    run_dir: Path
    iteration: int
