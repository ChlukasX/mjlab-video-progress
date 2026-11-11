# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: see `README.md`.

## What this is

`mjlab-video-progress` is a `watch` loop that renders new local
[mjlab](https://github.com/mujocolab/mjlab) checkpoints (every X iterations) via the configured
`play` command into a **bundle** (v1: just `video.mp4`; `strip.png`, `eval.json` and
`manifest.json` are deferred to a later version) and delivers it to **sinks** (`local`, `docs`,
`telegram`; others are deferred). The sinks deliver the video only. It must never interrupt the
training run. Linux is the supported platform.

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
  with its subclass `RunDirNotFound`, and `RenderError`. `config` and `discovery` import them
  from here, so the old import paths (`config.ConfigError`, `discovery.RunDirNotFound`) still
  work.
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
  skipped. A missing run dir raises `FileNotFoundError`. Read-only.
  `next_due(checkpoints, last_rendered, every)` is a pure function returning the newest
  checkpoint with `iteration >= last_rendered + every`, or `None`. `last_rendered=None` counts as
  0, so the first render waits for iteration `every`. Several due: only the newest. The threshold
  is relative to `last_rendered`, so an `every` that is not a multiple of mjlab's `save_interval`
  still fires. A failed render, retries and seeding `last_rendered` after a restart are not
  handled here (left to `watch`).
- `render.py`: `video_complete(path)` is True only for a whole mp4: the top-level boxes tile the
  file exactly and a `moov` and an `mdat` box are present (both `moov` last and `+faststart`
  layouts). A missing file, a directory, a cut or still-being-written file is False. It reads
  box headers only; it replaces a plain `ffprobe` exit code, which accepted many cuts inside the
  trailing `moov`.
  `render_checkpoint(cfg, task, checkpoint, work_dir)` returns `work_dir/videos/play/
  rl-video-step-0.mp4` or raises `RenderError`. It copies the checkpoint into `work_dir` (a copy
  that is not a complete zip raises before `play` starts), removes any leftover video, runs
  `[*command, task, --checkpoint-file <copy>, --video True, --video-length, --num-envs,
  *extra_args]` in its own session with output in `play.log`, and polls. As soon as
  `video_complete` is true it stops `play` (SIGINT to its process group, SIGKILL after
  `KILL_GRACE_SECONDS`) and returns; on timeout or an exit without a video it stops it the same
  way and raises, with the log tail in the message. An `OSError` (missing checkpoint, `work_dir`
  not creatable, `play` not startable) propagates for `watch` to handle. Not called by anything
  yet.
- `store.py`: where finished videos live. `store_video(video, output_dir, run_dir, iteration)`
  copies the video to `<output_dir>/<experiment>/<run>/iter_<N>/video.mp4` (experiment and run are
  the last two names of `run_dir`, which is never touched; `N` zero-padded to 6 digits) through a
  temp file and `os.replace`, so the video is whole or absent and a repeat replaces it. The source
  is left alone; an `OSError` propagates. `last_stored_iteration(output_dir, run_dir)` returns the
  highest `N` whose dir holds a `video.mp4`, or `None` (including when the output dir does not
  exist yet): `watch` seeds `last_rendered` from it after a restart. The `iter_<N>` dir is the
  bundle; it holds only the video in v1.

## Hard rules (do not break)

1. **Read-only on training output.** The tool never starts, stops or signals a training
   process, and never writes, moves or deletes under `logs/` or `wandb/`. There is no exception:
   `play` runs on a copy of the checkpoint in the output dir, so its `videos/play/` lands there.
   A test asserts a **scan** leaves the logs tree byte- and mtime-identical. The tool writes only: the output
   dir, run-log files named in the `docs` sink, and an optional `image_dir` / `dest`. Never run
   git from the tool. The tool may signal the process group of its own render subprocess
   (once the video is done, or on timeout), and never anything else.

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

Render, from reading mjlab 1.6.0 source (`play.py`, `video_recorder.py`, `viewer/viser`); none
of it run:

- `play --video` never exits by itself: after the `video_length` frames it keeps running the
  viewer, and headless that is `ViserPlayViewer`, whose `is_running()` is always True. The render
  must stop `play` itself once the video is complete, with the timeout as the hard cap.
- mjlab turns off tyro's flag conversion (`TYRO_FLAGS`), so the flag is `--video True`.
- The video is always `<checkpoint dir>/videos/play/rl-video-step-0.mp4`, overwritten by each
  render, so a failed render can leave the previous video behind. Hence the render runs `play`
  on a copy in `work_dir` and removes any leftover video first.
- `uv run play` starts the real python as a child; signalling only the wrapper would leave it
  running. The render signals the process group it created, never anything else.
- The stop sequence (SIGINT to the process group, SIGKILL after 10 s) is tested only with a fake
  `play` and a grandchild process. Whether it stops a real `uv run play`, its viser server and
  its GPU job cleanly is untested. A group member that is a zombie nobody reaps (a container
  without an init process) would make the stop wait out the full grace.
- Each render starts a viser web server on a port. A clash with a concurrent viewer, and what it
  exposes on the host, are untested.

## Verification status (do not overclaim)

Implemented: the config loader (`config.py`); the train-command parser, the `/proc` process
listing, the run-dir lookup, the directory fallback and the glue choosing between them
(`discover_runs`, all in `discovery.py`); the checkpoint scan and the "due" choice (`scan.py`);
the mp4 completeness check and running `play` (`render.py`); and the video store (`store.py`).
They are covered by unit tests only (`tests/test_config.py`, 44 passing, `tests/test_discovery.py`,
56 passing plus 2 skipped on macOS, `tests/test_scan.py`, 20 passing, `tests/test_render.py`, 30
passing and `tests/test_store.py`, 9 passing; 159 passing in total, plus ruff clean). The skipped
tests are the real-`/proc` tests (process listing smoke test, run-dir lookup via the real `cwd`
link), which have not been run on Linux yet; otherwise discovery is tested against a fake `/proc`
tree and fake logs trees only. `render_checkpoint` has never run a real `play`: its tests use a
fake script. Nothing above is called by anything yet. The watch loop and sinks are not
implemented yet.
Nothing has been run against a real mjlab training run; the parser has only been checked
against the shape of one captured uv-venv command line, and the run-dir lookup and the
fallback only against the mjlab 1.6.0, rsl_rl 5.4.2 and tensorboard source. The scan's
completeness check (zip end record) was checked only by truncating one `torch.save` file on
macOS, not against a checkpoint being written by a live run on Linux. `video_complete` was
checked only against mp4s made by ffmpeg itself (both layouts, every cut); the file mediapy
writes inside `play` has not been looked at.
