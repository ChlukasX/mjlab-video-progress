"""Render: run mjlab's ``play`` for one checkpoint and return the finished video.

Contract:

- ``video_complete(path)`` is True only when ``path`` is a whole mp4: its top-level boxes
  (4-byte big-endian size, 4-byte type) tile the file exactly from the first byte to the last,
  and a ``moov`` and an ``mdat`` box are among them. Otherwise False: a missing file or
  directory, an empty or non-mp4 file, a file cut anywhere, or a file ffmpeg is still writing
  (``mdat`` with its size placeholder 0 and no ``moov`` yet). Both layouts count, ``moov`` last
  (ffmpeg's default) and ``moov`` first (``+faststart``). Reads only the box headers, never
  writes. Boxes with a size below 8 (0 and 1, the to-end-of-file and 64-bit forms) are not
  accepted; a render video never reaches 4 GB.
- ``render_checkpoint(cfg, task, checkpoint, work_dir)`` returns ``work_dir/videos/play/
  rl-video-step-0.mp4``, a video ``video_complete`` accepts, or raises ``RenderError``.
  It creates ``work_dir`` and removes any video already at that path (so a stale one is never
  returned), then copies ``checkpoint.path`` to ``work_dir/<its name>``; a copy that is not a
  complete zip (the file was being rewritten) raises ``RenderError`` before ``play`` starts.
  ``play`` then runs as ``[*cfg.command, task, "--checkpoint-file", <copy>, "--video", "True",
  "--video-length", N, "--num-envs", N, *cfg.extra_args]`` in ``cfg.cwd``, in its own session,
  with stdout and stderr in ``work_dir/play.log``. mjlab's ``play --video`` never exits by
  itself, so the call polls every ``POLL_SECONDS`` and returns as soon as the video is complete
  (or ``play`` has exited and the video is complete), after stopping ``play``: SIGINT to its
  process group, SIGKILL to the group after ``KILL_GRACE_SECONDS``. After ``cfg.timeout_seconds``
  it stops ``play`` the same way and raises ``RenderError`` ("timed out after Ns"). If ``play``
  exits with no complete video it raises ``RenderError`` ("play exited with code N ...") with
  the last lines of ``play.log`` (the timeout message carries them too). Only the process group
  it created is ever signalled, and it writes only under ``work_dir`` (never under the
  checkpoint's own dir). An ``OSError`` from creating ``work_dir``, copying the checkpoint or
  starting ``play`` propagates; it is not a ``RenderError``.
"""

from __future__ import annotations

import os
import shutil
import signal
import struct
import subprocess
import time
import zipfile
from pathlib import Path

from mjlab_video_progress.config import RenderConfig
from mjlab_video_progress.exceptions import RenderError
from mjlab_video_progress.scan import Checkpoint

POLL_SECONDS = 0.2
KILL_GRACE_SECONDS = 10
VIDEO_NAME = "videos/play/rl-video-step-0.mp4"  # where mjlab's play writes, next to the checkpoint


def video_complete(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            end = os.fstat(f.fileno()).st_size
            kinds = set()
            pos = 0
            while pos < end:
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    return False
                size, kind = struct.unpack(">I4s", header)
                if size < 8 or pos + size > end:
                    return False
                kinds.add(kind)
                pos += size
    except (FileNotFoundError, IsADirectoryError):
        return False
    return {b"moov", b"mdat"} <= kinds


def render_checkpoint(cfg: RenderConfig, task: str, checkpoint: Checkpoint, work_dir: Path) -> Path:
    work_dir = Path(work_dir).absolute()  # play runs in cfg.cwd, so a relative path would break
    work_dir.mkdir(parents=True, exist_ok=True)
    video = work_dir / VIDEO_NAME
    video.unlink(missing_ok=True)  # a leftover would count as done before play even starts
    copy = work_dir / checkpoint.path.name
    shutil.copyfile(checkpoint.path, copy)
    if not zipfile.is_zipfile(copy):
        raise RenderError(f"checkpoint {checkpoint.path} was incomplete when copied; try again")
    argv = [
        *cfg.command,
        task,
        "--checkpoint-file",
        str(copy),
        "--video",
        "True",  # mjlab turns off tyro's flag conversion
        "--video-length",
        str(cfg.video_length),
        "--num-envs",
        str(cfg.num_envs),
        *cfg.extra_args,
    ]
    log_path = work_dir / "play.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            argv,
            cwd=cfg.cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + cfg.timeout_seconds
    try:
        while True:
            code = proc.poll()  # before the video check: play's last write precedes its exit
            if video_complete(video):
                return video
            if code is not None:
                reason = f"play exited with code {code} before the video was complete"
                raise RenderError(f"{reason}\n{_tail(log_path)}")
            if time.monotonic() >= deadline:
                reason = f"timed out after {cfg.timeout_seconds}s without a finished video"
                raise RenderError(f"{reason}\n{_tail(log_path)}")
            time.sleep(POLL_SECONDS)
    finally:
        _stop(proc)


def _tail(path: Path, lines: int = 20) -> str:
    return "\n".join(path.read_bytes()[-4000:].decode(errors="replace").splitlines()[-lines:])


def _stop(proc: subprocess.Popen) -> None:
    """Stop play and what it started: SIGINT to its process group, SIGKILL after the grace.

    The group is the one start_new_session gave it (id == proc.pid); the kernel does not reuse
    that pid while any member is alive, so this never reaches another process.
    """
    try:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        os.killpg(proc.pid, signal.SIGKILL)  # also reaps a child that outlived its parent
    except ProcessLookupError:
        pass  # the group is already gone
    proc.wait()
