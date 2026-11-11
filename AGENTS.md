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

- `exceptions.py`: every custom exception, in one place: `ConfigError`, and `DiscoveryError`
  with its subclass `RunDirNotFound`. `config` and `discovery` import them from here, so the
  old import paths (`config.ConfigError`, `discovery.RunDirNotFound`) still work.
- `config.py`: loads and validates the TOML config (`load_config`, frozen dataclasses,
  raises `ConfigError`). Rejects unknown keys, wrong types, non-positive ints and empty strings
  (except inside `render.extra_args`). Expands a leading `~` in paths, like a shell would for
  `--log-root ~/x`, then resolves relative paths against the config file.
- `discovery.py`: `parse_train_argv(argv)` turns a process argument list into
  `TrainCommand(task, log_root)`, or `None` if it is not a `python .../train <task>` command.
  Pure function, no filesystem or process access.
  `find_train_processes(proc_root, uid)` lists the same-user processes under `/proc` (root
  injectable for tests) whose `cmdline` matches, as `TrainProcess(pid, command)` sorted by pid.
  Read-only.
  `find_run_dir(process, proc_root)` returns the run dir: under `<cwd>/<log root>` (default
  `logs/rsl_rl`), the one `*/*/` dir holding `events.out.tfevents.*.<hostname>.<pid>.0`.
  Otherwise it raises `RunDirNotFound` with the reason (no cwd, no match, ambiguous). Read-only.
  A `~` that reaches `--log-root` unexpanded (`--log-root=~/x`; bash and zsh leave it) is taken
  literally, as mjlab does: mjlab writes `<cwd>/~/x`, and the lookup looks there.
  `find_latest_run_dir(log_root)` is the directory fallback: among the `*/*` dirs named like
  mjlab's `YYYY-MM-DD_HH-MM-SS[_<run_name>]`, it returns the one with the highest name across
  all experiments (never reads mtime). Otherwise `RunDirNotFound` (not a dir, no run dir, newest
  name in several experiments). Read-only.
  `discover_runs(log_root, task, proc_root)` is the glue `watch` will call (once per start, to
  be confirmed in roadmap item 8; there is no `watch` yet). It takes `watch.log_root` and
  `watch.task` as arguments and does not import `config`. If training processes are found, it
  returns a `Run(run_dir, task)` per process with a run dir, in pid order, using the process's
  own task (a different `watch.task` is only logged as a warning). Processes without a run dir
  are skipped with a warning; if none is left it raises `RunDirNotFound` and never falls back
  (so a training that exits mid-lookup gives an error, not the fallback). Only when no process
  is found does it return `find_latest_run_dir(watch.log_root)` with `watch.task`; either one
  unset raises `DiscoveryError`, the base class of `RunDirNotFound`. Warnings go to stdlib
  `logging`. Read-only.
- `scan.py`: `scan_checkpoints(run_dir)` returns `Checkpoint(iteration, path)` for every regular
  file named exactly `model_<int>.pt` in the run dir, sorted by iteration. A file without a valid
  zip end record (mjlab's `torch.save` writes it last, in place) is still being written and is
  skipped. A missing run dir raises `FileNotFoundError`. Read-only. Which checkpoint is due is
  not decided here.

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

Cases `discovery.find_run_dir` does not handle (by design it reports "not found" and never
guesses; only the ambiguous case is tested):

- Multi-GPU runs: torchrunx workers write the tfevents file with their own pid, not the train
  pid, so the lookup always reports "not found".
- PID namespaces: a training in a container watched from the host has a different pid in the
  filename.
- A reused pid: an old run with the same pid and hostname is reported as ambiguous. While the
  live run is still starting (no tfevents yet), that old run is returned as the only match.
  The fix would be to filter by the process start time (`/proc/<pid>/stat` + `/proc/stat` btime).

Cases `discovery.find_latest_run_dir` does not handle (it picks by dir name only; not tested):

- A finished run: if the newest run dir is a finished run, it is returned anyway; no new
  checkpoints appear, so `watch` would idle.
- A run in env setup: mjlab creates the run dir only when it dumps `params/`, after the env is
  built. Until then the previous run is returned.
- Clock or time-zone changes between runs: names are local time, so name order can differ from
  start order.

Also deferred: a config key to override the process match. Launchers the strict rule does not
match (`python -m mjlab.scripts.train`, wrappers) are left to the `watch.log_root` directory
fallback (`find_latest_run_dir`, used by `discover_runs` when no process is found). Add the key
only when a real launcher is missed.

## Verification status (do not overclaim)

Only the config loader (`config.py`), the train-command parser, the `/proc` process listing, the
run-dir lookup, the directory fallback and the glue choosing between them (`discover_runs`, all
in `discovery.py`), and the checkpoint scan (`scan.py`) are implemented. They are covered by unit
tests only (`tests/test_config.py`, 44 passing, `tests/test_discovery.py`, 56 passing plus 2
skipped on macOS, and `tests/test_scan.py`, 9 passing; 109 passing in total, plus ruff clean).
The skipped tests are the real-`/proc` tests (process listing smoke test, run-dir lookup via
the real `cwd` link), which have not been run on Linux yet; otherwise
discovery is tested against a fake `/proc` tree and fake logs trees only. `discover_runs` is not
called by anything yet, nor is `scan_checkpoints`. The "due" logic, watch loop, render, bundle
and sinks are not implemented yet.
Nothing has been run against a real mjlab training run; the parser has only been checked
against the shape of one captured uv-venv command line, and the run-dir lookup and the
fallback only against the mjlab 1.6.0, rsl_rl 5.4.2 and tensorboard source. The scan's
completeness check (zip end record) was checked only by truncating one `torch.save` file on
macOS, not against a checkpoint being written by a live run on Linux.
