# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: see `README.md`.

## What this is

`mjlab-video-progress` is a `watch` loop that renders new local
[mjlab](https://github.com/mujocolab/mjlab) checkpoints (every X iterations) via the configured
`play` command into a **bundle** (`video.mp4`, `strip.png`, `eval.json`, `manifest.json`) and
delivers it to **sinks** (`local`, `docs`, `telegram`; others are deferred). It must never
interrupt the training run. Linux is the supported platform.

Goal: low impact on the training run. A render uses `--num-envs 1` and a hard timeout by
default. The actual cost has not yet been measured.

## Commands

```bash
uv sync                                  # install incl. dev group (pytest, ruff)
uv run pytest                            # tests/ ; ffmpeg-dependent tests skip if ffmpeg is missing
uv run ruff check .
uv run ruff format --check .             # line length 100; rules E, F, I, B, UP
```

Run all three of pytest, `ruff check` and `ruff format --check` before calling work done.
Needs `ffmpeg` and `ffprobe` on `PATH`. Python `>=3.11` (`tomllib` from the standard library).

## Code map (`src/mjlab_video_progress/`)

## Hard rules (do not break)

1. **Read-only on training output.** The tool never starts, stops or signals a training
   process, and never writes, moves or deletes under `logs/` or `wandb/`. The one exception is
   files mjlab's own `play` subprocess creates (`<checkpoint dir>/videos/play/`). A test asserts
   a **scan** leaves the logs tree byte- and mtime-identical. The tool writes only: the output
   dir, run-log files named in the `docs` sink, and an optional `image_dir` / `dest`. Never run
   git from the tool. The tool may kill its own render subprocess on timeout, and never
   anything else.

## Conventions

- Prefer the standard library (`urllib` for Telegram, `subprocess` for ffmpeg; no imaging lib).
- Every module starts with a docstring stating its contract; keep it accurate when behavior
  changes. `from __future__ import annotations` is used throughout.
- Sinks get injected transports so tests never hit the network. New behavior needs a unit test.

## Verification status (do not overclaim)

Nothing is implemented or verified yet.
