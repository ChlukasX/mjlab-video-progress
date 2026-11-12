"""Every test runs without the real nvidia-smi: the GPU gate reports "no GPU" unless a test
replaces ``watch.free_mib`` itself."""

from __future__ import annotations

import pytest

from mjlab_video_progress import watch


@pytest.fixture(autouse=True)
def no_gpu_gate(monkeypatch):
    monkeypatch.setattr(watch, "free_mib", lambda extra_args: None)
