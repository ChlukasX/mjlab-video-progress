"""Helpers shared by several test modules (tests/ has no package, so: ``from helpers import``)."""

from __future__ import annotations

import io
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
