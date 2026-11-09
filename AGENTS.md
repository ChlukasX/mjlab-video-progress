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

- `config.py`: loads and validates the TOML config (`load_config`, frozen dataclasses,
  `ConfigError`). Rejects unknown keys, wrong types, non-positive ints and empty strings
  (except inside `render.extra_args`).
- `discovery.py`: `parse_train_argv(argv)` turns a process argument list into
  `TrainCommand(task, log_root)`, or `None` if it is not a `python .../train <task>` command.
  Pure function, no filesystem or process access.
  `find_train_processes(proc_root, uid)` lists the same-user processes under `/proc` (root
  injectable for tests) whose `cmdline` matches, as `TrainProcess(pid, command)` sorted by pid.
  Read-only.

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

## Known gaps (deliberately not handled in v1)

Edge cases in `discovery.parse_train_argv` that were considered and left out because real
command lines are unlikely to hit them. Not covered by tests; revisit if discovery misfires.

- Repeated `--log-root`: the last occurrence wins (argparse behaviour); this is not specified.
- A value that is itself `--log-root` (`--log-root --log-root`) is also scanned as a flag, so
  the dangling second one makes the whole command line a non-match (`None`).
- The interpreter pattern `python(3(\.\d+)*)?` rejects `python2` and `python3.` but accepts
  odd versions such as `python3.1.2`.

Also deferred: a config key to override the process match. Launchers the strict rule does not
match (`python -m mjlab.scripts.train`, wrappers) are left to the planned `watch.log_root`
directory fallback, which is not built yet. Add the key only when a real launcher is missed.

## Verification status (do not overclaim)

Only the config loader (`config.py`), the train-command parser and the `/proc` process
listing (`discovery.py`) are implemented. They are covered by unit tests only
(`tests/test_config.py`, 41 passing, and `tests/test_discovery.py`, 24 passing plus 1 skipped
on macOS; 65 passing in total, plus ruff clean). The skipped test is the real-`/proc` smoke test,
which has not been run on Linux yet; the process listing is otherwise tested against a fake
`/proc` tree only. Run-dir lookup, the watch loop, render, bundle and sinks are not implemented
yet. Nothing has been run against a real mjlab training run; the parser has only been checked
against the shape of one captured uv-venv command line.
