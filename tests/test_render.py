"""Contract tests for render: video_complete."""

from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

from mjlab_video_progress.render import video_complete


def box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


FTYP = box(b"ftyp", b"isom0000")
MDAT = box(b"mdat", b"x" * 50)
MOOV = box(b"moov", b"y" * 20)


@pytest.fixture(params=[(), ("-movflags", "+faststart")], ids=["moov-last", "moov-first"])
def mp4(request, tmp_path):
    """A real 2 s mp4 made by ffmpeg, in both layouts."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "rl-video-step-0.mp4"
    cmd = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10", "-t", "2"]
    subprocess.run([*cmd, "-pix_fmt", "yuv420p", *request.param, "-y", str(path)], check=True)
    return path


def test_real_complete_video(mp4):
    assert video_complete(mp4)


def test_real_video_cut_anywhere_is_incomplete(mp4, tmp_path):
    # ffprobe alone accepts many cuts inside the trailing moov, so cut at every byte.
    data = mp4.read_bytes()
    partial = tmp_path / "partial.mp4"
    for cut in range(len(data)):
        partial.write_bytes(data[:cut])
        assert not video_complete(partial), f"cut at byte {cut} of {len(data)}"


@pytest.mark.parametrize(
    "data, expected",
    [
        (FTYP + MDAT + MOOV, True),
        (FTYP + MOOV + MDAT, True),  # +faststart order
        (FTYP + box(b"free", b"") + MDAT + MOOV, True),  # other boxes are allowed
        (FTYP + MDAT, False),  # no moov: still being written
        (FTYP + MOOV, False),  # no mdat
        (FTYP + MDAT + MOOV + b"\0\0\0", False),  # trailing bytes: boxes do not tile the file
        (FTYP + MDAT + MOOV[:-1], False),  # last box cut by one byte
        (FTYP + struct.pack(">I4s", 0, b"mdat") + b"x" * 50, False),  # size placeholder 0
        (b"x" * 100, False),  # not an mp4
        (b"", False),
    ],
    ids=[
        "moov-last",
        "moov-first",
        "free-box",
        "no-moov",
        "no-mdat",
        "trailing-bytes",
        "moov-cut-by-one",
        "size-placeholder-0",
        "not-mp4",
        "empty",
    ],
)
def test_box_layout(tmp_path, data, expected):
    path = tmp_path / "v.mp4"
    path.write_bytes(data)
    assert video_complete(path) is expected


def test_missing_file_or_directory(tmp_path):
    assert video_complete(tmp_path / "nope.mp4") is False
    assert video_complete(tmp_path) is False
