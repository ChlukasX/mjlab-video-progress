"""Contract tests for render: video_complete and render_checkpoint (against a fake play)."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from helpers import snapshot, zip_bytes

from mjlab_video_progress import render
from mjlab_video_progress.config import RenderConfig
from mjlab_video_progress.exceptions import RenderError
from mjlab_video_progress.render import render_checkpoint, video_complete
from mjlab_video_progress.scan import Checkpoint


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


# --- render_checkpoint, against a fake `play` script -------------------------------------------

VIDEO = FTYP + MDAT + MOOV
VIDEO_NAME = "videos/play/rl-video-step-0.mp4"

# The fake play never exits on its own, like the real one under viser: it keeps a heartbeat file
# growing, and so does a grandchild process (the real `uv run play` has a python child).
PRELUDE = """
import json, signal, subprocess, sys, time
from pathlib import Path

VIDEO = @VIDEO@
BEAT = "import sys, time\\nwhile True:\\n open(sys.argv[1], 'ab').write(b'x')\\n time.sleep(0.05)"
args = sys.argv[1:]
work = Path(args[args.index("--checkpoint-file") + 1]).parent
(work / "argv.json").write_text(json.dumps({"argv": args, "cwd": str(Path.cwd().resolve())}))

def start_beats():
    (work / "play.beat").touch()
    (work / "child.beat").touch()
    subprocess.Popen([sys.executable, "-c", BEAT, str(work / "child.beat")])

def beat():
    while True:
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


def render_cfg(script, **overrides):
    fields = {"command": (sys.executable, str(script), "prefix"), "timeout_seconds": 30}
    return RenderConfig(**{**fields, **overrides})


def assert_silent(path):
    """The process writing `path` is gone: the file stops growing."""
    size = path.stat().st_size
    time.sleep(0.5)
    assert path.stat().st_size == size, f"{path.name} is still being written"


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(render, "POLL_SECONDS", 0.02)


@pytest.fixture
def ckpt(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / "model_150.pt"
    path.write_bytes(zip_bytes())
    return Checkpoint(150, path)


HANG = "start_beats()\nbeat()"
FINISH_AND_HANG = "start_beats()\nwrite_video()\nbeat()"


def test_returns_finished_video_and_stops_play_and_its_child(tmp_path, ckpt):
    work = tmp_path / "work" / "1"  # does not exist yet
    script = fake_play(tmp_path, FINISH_AND_HANG)
    result = render_checkpoint(render_cfg(script), "Task-X", ckpt, work)
    assert result == work / VIDEO_NAME
    assert video_complete(result)
    assert_silent(work / "play.beat")
    assert_silent(work / "child.beat")


def test_play_that_ignores_sigint_is_killed_after_the_grace(tmp_path, ckpt, monkeypatch):
    monkeypatch.setattr(render, "KILL_GRACE_SECONDS", 0.3)
    script = fake_play(tmp_path, "signal.signal(signal.SIGINT, signal.SIG_IGN)\n" + FINISH_AND_HANG)
    work = tmp_path / "work"
    render_checkpoint(render_cfg(script), "Task-X", ckpt, work)
    assert_silent(work / "play.beat")
    assert_silent(work / "child.beat")


def test_argv_and_cwd(tmp_path, ckpt):
    repo = tmp_path / "repo"
    repo.mkdir()
    work = tmp_path / "work"
    cfg = render_cfg(
        fake_play(tmp_path, FINISH_AND_HANG),
        num_envs=2,
        video_length=50,
        extra_args=("--device", "cpu"),
        cwd=repo,
    )
    render_checkpoint(cfg, "Task-X", ckpt, work)
    seen = json.loads((work / "argv.json").read_text())
    assert seen["argv"] == [
        "prefix",
        "Task-X",
        "--checkpoint-file",
        str(work / "model_150.pt"),
        "--video",
        "True",
        "--video-length",
        "50",
        "--num-envs",
        "2",
        "--device",
        "cpu",
    ]
    assert seen["cwd"] == str(repo.resolve())


def test_play_gets_a_copy_and_the_checkpoint_dir_is_untouched(tmp_path, ckpt):
    run_dir = ckpt.path.parent
    before = snapshot(run_dir)
    work = tmp_path / "work"
    render_checkpoint(render_cfg(fake_play(tmp_path, FINISH_AND_HANG)), "T", ckpt, work)
    argv = json.loads((work / "argv.json").read_text())["argv"]
    used = Path(argv[argv.index("--checkpoint-file") + 1])
    assert used != ckpt.path and used.read_bytes() == ckpt.path.read_bytes()
    assert snapshot(run_dir) == before


def test_timeout_stops_play_and_raises(tmp_path, ckpt):
    work = tmp_path / "work"
    with pytest.raises(RenderError, match="timed out after 1s"):
        render_checkpoint(render_cfg(fake_play(tmp_path, HANG), timeout_seconds=1), "T", ckpt, work)
    assert_silent(work / "play.beat")
    assert_silent(work / "child.beat")


def test_exit_without_video_raises_with_the_log_tail(tmp_path, ckpt):
    work = tmp_path / "work"
    script = fake_play(tmp_path, 'print("CUDA out of memory", file=sys.stderr)\nsys.exit(1)')
    with pytest.raises(RenderError, match="code 1") as e:
        render_checkpoint(render_cfg(script), "T", ckpt, work)
    assert "CUDA out of memory" in str(e.value)
    assert "CUDA out of memory" in (work / "play.log").read_text()


def test_clean_exit_without_video_raises(tmp_path, ckpt):
    script = fake_play(tmp_path, "sys.exit(0)")
    with pytest.raises(RenderError, match="code 0"):
        render_checkpoint(render_cfg(script), "T", ckpt, tmp_path / "work")


def test_play_that_writes_the_video_and_exits_succeeds(tmp_path, ckpt):
    result = render_checkpoint(
        render_cfg(fake_play(tmp_path, "write_video()")), "T", ckpt, tmp_path / "work"
    )
    assert video_complete(result)


def test_half_written_video_is_waited_for(tmp_path, ckpt):
    body = """
        start_beats()
        path = work / "@VIDEO_NAME@"
        path.parent.mkdir(parents=True)
        with path.open("wb") as f:
            f.write(VIDEO[:20])
            f.flush()
            time.sleep(1)
            f.write(VIDEO[20:])
        beat()
    """.replace("@VIDEO_NAME@", VIDEO_NAME)
    result = render_checkpoint(render_cfg(fake_play(tmp_path, body)), "T", ckpt, tmp_path / "work")
    assert result.read_bytes() == VIDEO


def test_stale_video_is_not_returned(tmp_path, ckpt):
    work = tmp_path / "work"
    (work / VIDEO_NAME).parent.mkdir(parents=True)
    (work / VIDEO_NAME).write_bytes(VIDEO)  # left by an earlier render
    with pytest.raises(RenderError, match="code 1"):
        render_checkpoint(render_cfg(fake_play(tmp_path, "sys.exit(1)")), "T", ckpt, work)


def test_incomplete_checkpoint_is_not_rendered(tmp_path, ckpt):
    ckpt.path.write_bytes(ckpt.path.read_bytes()[:50])  # still being written
    work = tmp_path / "work"
    with pytest.raises(RenderError, match="checkpoint"):
        render_checkpoint(render_cfg(fake_play(tmp_path, FINISH_AND_HANG)), "T", ckpt, work)
    assert not (work / "argv.json").exists()  # play was never started


def test_missing_checkpoint_raises_oserror_not_render_error(tmp_path, ckpt):
    ckpt.path.unlink()  # deleted between the scan and the render
    work = tmp_path / "work"
    with pytest.raises(FileNotFoundError):
        render_checkpoint(render_cfg(fake_play(tmp_path, FINISH_AND_HANG)), "T", ckpt, work)
    assert not (work / "argv.json").exists()


def test_work_dir_that_cannot_be_created_raises_oserror(tmp_path, ckpt):
    blocker = tmp_path / "blocker"
    blocker.write_text("")  # a file where a directory is needed
    with pytest.raises(OSError):
        render_checkpoint(
            render_cfg(fake_play(tmp_path, FINISH_AND_HANG)), "T", ckpt, blocker / "work"
        )


def test_play_command_that_cannot_start_raises_oserror(tmp_path, ckpt):
    cfg = RenderConfig(command=(str(tmp_path / "no-such-play"),), timeout_seconds=5)
    with pytest.raises(FileNotFoundError):
        render_checkpoint(cfg, "T", ckpt, tmp_path / "work")


def test_stop_survives_a_play_that_already_exited_unreaped():
    # macOS answers PermissionError when signalling a group that is only an unreaped zombie.
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    time.sleep(0.5)  # exited, but nobody has waited on it yet
    render._stop(proc)
    assert proc.returncode == 0
