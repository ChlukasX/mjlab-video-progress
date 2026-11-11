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
"""

from __future__ import annotations

import os
import struct
from pathlib import Path


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
