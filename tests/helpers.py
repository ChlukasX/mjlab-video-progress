"""Helpers shared by several test modules (tests/ has no package, so: ``from helpers import``)."""

from __future__ import annotations

import io
import struct
import textwrap
import time
import zipfile


def zip_bytes():
    """A complete zip, like the file torch.save writes (the end record comes last)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", b"x" * 5000)
    return buf.getvalue()


def snapshot(root):
    """Bytes (files) and st_mtime_ns of every path under root, root included."""
    return {
        p: (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
        for p in [root, *root.rglob("*")]
    }


def box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


FTYP = box(b"ftyp", b"isom0000")
MDAT = box(b"mdat", b"x" * 50)
MOOV = box(b"moov", b"y" * 20)


VIDEO = FTYP + MDAT + MOOV
VIDEO_NAME = "videos/play/rl-video-step-0.mp4"

# The fake play never exits on its own, like the real one under viser: it keeps a heartbeat file
# growing, and so does a grandchild process (the real `uv run play` has a python child).
PRELUDE = """
import json, signal, subprocess, sys, time
from pathlib import Path

VIDEO = @VIDEO@
BEAT = ("import sys, time\\nfor _ in range(1200):\\n open(sys.argv[1], 'ab').write(b'x')\\n"
        " time.sleep(0.05)")
args = sys.argv[1:]
work = Path(args[args.index("--checkpoint-file") + 1]).parent
(work / "argv.json").write_text(json.dumps({"argv": args, "cwd": str(Path.cwd().resolve())}))

def start_beats():
    (work / "play.beat").touch()
    (work / "child.beat").touch()
    subprocess.Popen([sys.executable, "-c", BEAT, str(work / "child.beat")])

def beat():
    for _ in range(1200):  # at most a minute, so a failed test leaves nothing behind
        with open(work / "play.beat", "ab") as f:
            f.write(b"x")
        time.sleep(0.05)

def write_video():
    path = work / "@VIDEO_NAME@"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(VIDEO)
"""


def fake_play(tmp_path, body):
    script = tmp_path / "fake_play.py"
    prelude = PRELUDE.replace("@VIDEO@", repr(VIDEO)).replace("@VIDEO_NAME@", VIDEO_NAME)
    script.write_text(prelude + textwrap.dedent(body))
    return script


def assert_silent(path):
    """The process writing `path` is gone: the file stops growing."""
    size = path.stat().st_size
    time.sleep(0.5)
    assert path.stat().st_size == size, f"{path.name} is still being written"
