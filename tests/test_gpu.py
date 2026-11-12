"""Contract tests for gpu: the device parse and the nvidia-smi query, with a fake runner."""

from __future__ import annotations

import subprocess

import pytest

from mjlab_video_progress.exceptions import GpuError
from mjlab_video_progress.gpu import free_mib, gpu_index


@pytest.mark.parametrize(
    ("args", "index"),
    [
        ((), 0),
        (("--video-width", "720"), 0),
        (("--device", "cuda:1"), 1),
        (("--device=cuda:2",), 2),
        (("--device", "cuda:1", "--device=cuda:3"), 3),  # the last one wins
        (("--device", "cuda"), 0),
        (("--device", "cpu"), 0),
        (("--device",), 0),  # dangling
    ],
)
def test_gpu_index(args, index):
    assert gpu_index(args) == index


def runner(stdout="", raises=None, calls=None):
    def run(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        if raises:
            raise raises
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    return run


def test_no_nvidia_smi_means_no_gate():
    calls = []
    assert free_mib((), runner(calls=calls), which=lambda name: None) is None
    assert calls == []


def test_queries_the_gpu_play_uses():
    calls = []
    free = free_mib(("--device", "cuda:1"), runner("9743\n", calls=calls), which=lambda n: n)
    assert free == 9743
    [(cmd, kwargs)] = calls
    assert cmd == [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
        "--id=1",
    ]
    assert kwargs["timeout"] == 30 and kwargs["check"] is True


@pytest.mark.parametrize(
    "run",
    [
        runner(raises=FileNotFoundError("nvidia-smi")),
        runner(raises=subprocess.CalledProcessError(6, "nvidia-smi")),
        runner(raises=subprocess.TimeoutExpired("nvidia-smi", 30)),
        runner("No devices were found\n"),
        runner("100\n200\n"),
    ],
    ids=["oserror", "exit", "timeout", "text", "two-lines"],
)
def test_failures_raise_gpu_error(run):
    with pytest.raises(GpuError, match="GPU 0"):
        free_mib((), run, which=lambda n: n)
