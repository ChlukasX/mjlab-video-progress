"""GPU gate: free memory on the GPU that ``play`` will use, read with ``nvidia-smi``.

Contract:

- ``gpu_index(extra_args)`` returns the index from the last ``--device cuda:N`` or
  ``--device=cuda:N`` in ``render.extra_args``, else 0 (mjlab's ``play`` defaults to ``cuda:0``;
  ``cuda`` alone, ``cpu`` or anything else also give 0, since rendering uses the GPU anyway).
  Pure function.
- ``free_mib(extra_args, run=subprocess.run, which=shutil.which)`` returns the free memory in MiB
  of that GPU, or ``None`` when ``nvidia-smi`` is not on ``PATH`` (no gate). It runs
  ``nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=<index>`` with a 30 s
  timeout. A start failure, a non-zero exit, a timeout or output that is not one integer raises
  ``GpuError``. ``run`` and ``which`` are injected so tests never call the real ``nvidia-smi``.
- The index is the CUDA index, given to ``nvidia-smi`` as is. They match unless
  ``CUDA_VISIBLE_DEVICES`` or ``CUDA_DEVICE_ORDER`` renumber the GPUs (not handled).
- Read-only: it only queries ``nvidia-smi``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from mjlab_video_progress.exceptions import GpuError

TIMEOUT_SECONDS = 30


def gpu_index(extra_args) -> int:
    device = None
    for i, arg in enumerate(extra_args):
        if arg == "--device" and i + 1 < len(extra_args):
            device = extra_args[i + 1]
        elif arg.startswith("--device="):
            device = arg.partition("=")[2]
    m = re.fullmatch(r"cuda:(\d+)", device or "")
    return int(m.group(1)) if m else 0


def free_mib(extra_args, run=subprocess.run, which=shutil.which) -> int | None:
    if which("nvidia-smi") is None:
        return None
    index = gpu_index(extra_args)
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
        f"--id={index}",
    ]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=True)
        return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        raise GpuError(f"nvidia-smi could not read the free memory of GPU {index}: {e}") from e
