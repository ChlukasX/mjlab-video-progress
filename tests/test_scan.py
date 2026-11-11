"""Contract tests for scan: scan_checkpoints and next_due."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from mjlab_video_progress.scan import Checkpoint, next_due, scan_checkpoints


def zip_bytes():
    """A complete zip, like the file torch.save writes (the end record comes last)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", b"x" * 5000)
    return buf.getvalue()


def add_ckpt(run_dir, name, data=None):
    path = run_dir / name
    path.write_bytes(zip_bytes() if data is None else data)
    return path


def snapshot(root):
    """Bytes (files) and st_mtime_ns of every path under root, root included."""
    return {
        p: (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
        for p in [root, *root.rglob("*")]
    }


def test_empty_run_dir_returns_empty_list(tmp_path):
    assert scan_checkpoints(tmp_path) == []


def test_sorted_numerically_and_includes_iteration_zero(tmp_path):
    for it in (100, 0, 50, 1000, 9):  # lexical order would put 1000 before 9
        add_ckpt(tmp_path, f"model_{it}.pt")
    assert scan_checkpoints(tmp_path) == [
        Checkpoint(it, tmp_path / f"model_{it}.pt") for it in (0, 9, 50, 100, 1000)
    ]


def test_ignores_entries_that_are_not_model_n_pt(tmp_path):
    add_ckpt(tmp_path, "model_50.pt")
    for name in [
        "model_5.pt.tmp",
        "model_x.pt",
        "model_1_backup.pt",
        "model_.pt",
        "policy.pt",
        "run.onnx",
    ]:
        add_ckpt(tmp_path, name)  # complete files: only the name rules them out
    for name in ["model_7.pt", "params", "videos"]:
        (tmp_path / name).mkdir()
    assert scan_checkpoints(tmp_path) == [Checkpoint(50, tmp_path / "model_50.pt")]


@pytest.mark.parametrize("cut", [0, 10, "half", "all but last byte"])
def test_incomplete_checkpoint_is_skipped(tmp_path, cut):
    data = zip_bytes()
    cut = {"half": len(data) // 2, "all but last byte": len(data) - 1}.get(cut, cut)
    add_ckpt(tmp_path, "model_50.pt")
    add_ckpt(tmp_path, "model_100.pt", data[:cut])  # still being written
    assert scan_checkpoints(tmp_path) == [Checkpoint(50, tmp_path / "model_50.pt")]


def test_scan_leaves_logs_tree_identical(tmp_path):
    add_ckpt(tmp_path, "model_0.pt")
    add_ckpt(tmp_path, "model_50.pt", zip_bytes()[:100])
    (tmp_path / "videos/play").mkdir(parents=True)
    before = snapshot(tmp_path)
    scan_checkpoints(tmp_path)
    assert snapshot(tmp_path) == before


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_checkpoints(tmp_path / "gone")


def ckpts(*iterations):
    return [Checkpoint(it, Path(f"model_{it}.pt")) for it in iterations]


def test_next_due_no_checkpoints():
    assert next_due([], None, 100) is None
    assert next_due([], 100, 100) is None


@pytest.mark.parametrize(
    "last, newest, expected",
    [
        (None, 50, None),  # first render waits for iteration `every`; None counts as 0
        (None, 100, 100),
        (0, 50, None),  # 0 is a real value, not "unset"
        (0, 100, 100),
        (100, 150, None),
        (100, 200, 200),  # exactly last_rendered + every is due
    ],
)
def test_next_due_threshold_is_last_rendered_plus_every(last, newest, expected):
    due = next_due(ckpts(*range(0, newest + 1, 50)), last, 100)
    assert (due.iteration if due else None) == expected


def test_next_due_several_due_returns_only_the_newest():
    cs = ckpts(*range(0, 501, 100))
    assert next_due(cs, None, 100) == cs[-1]
    assert next_due(cs, 100, 100) == cs[-1]


def test_next_due_every_not_a_multiple_of_save_interval_still_fires():
    cs = ckpts(0, 150, 300)  # save_interval 150, every 100
    assert next_due(cs[:2], None, 100) == cs[1]
    assert next_due(cs, 150, 100) == cs[2]


def test_next_due_nothing_newer_than_last_rendered():
    cs = ckpts(0, 50, 100)
    assert next_due(cs, 100, 50) is None  # the one just rendered is never due again
    assert next_due(cs, 500, 100) is None


def test_next_due_ignores_input_order():
    cs = ckpts(500, 0, 300, 100)
    assert next_due(cs, 0, 100) == cs[0]
