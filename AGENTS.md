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
uv run mjlab-video-progress watch --config PATH   # the tool itself (console script)
uv run mjlab-video-progress test --config PATH    # send a test message through every sink
```

Run all three of pytest, `ruff check` and `ruff format --check` before calling work done.
Needs `ffmpeg` and `ffprobe` on `PATH`. Python `>=3.11` (`tomllib` from the standard library).

## Code map (`src/mjlab_video_progress/`)

- `exceptions.py`: every custom exception, in one place: `ConfigError`, and `DiscoveryError`
  with its subclass `RunDirNotFound`, `RenderError` and `SinkError`. `config` and `discovery`
  import them from here, so the old import paths (`config.ConfigError`,
  `discovery.RunDirNotFound`) still work.
- `config.py`: loads and validates the TOML config (`load_config`, frozen dataclasses,
  raises `ConfigError`). Rejects unknown keys, wrong types, non-positive ints and empty strings
  (except inside `render.extra_args`). Expands a leading `~` in paths, like a shell would for
  `--log-root ~/x`, then resolves relative paths against the config file. `[sinks.telegram]` is
  optional (`Config.sinks.telegram` is `None` without it): `chat_id` (int, may be negative, or a
  non-empty string), `token_file` (path) and `token_env` (default `TELEGRAM_BOT_TOKEN`); loading
  never reads the token file or the environment. Other sink tables are rejected until built.
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
  `discover_runs(log_root, task, proc_root)` is the glue `watch.start` calls, once per start.
  It takes `watch.log_root` and `watch.task` as arguments and does not import `config`. If training processes are found, it
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
  exist yet): `watch` seeds `last_rendered` from it after a restart.
  `run_output_dir(output_dir, run_dir)` is `<output_dir>/<experiment>/<run>`; `watch` puts the
  run's `.work` dir there. The `iter_<N>` dir is the bundle; it holds only the video in v1.
- `sinks.py`: `Delivery(video, task, run_dir, iteration)`, what every sink is handed. A sink
  delivers the video only and raises `SinkError` when it cannot.
- `telegram.py`: the Telegram sink. `read_token(cfg, environ=None)` returns the bot token from
  `token_file` if set (a regular file, not readable by group or others, non-empty; otherwise
  `ConfigError` with a `chmod 600` hint), else from the environment variable `token_env`; it
  never puts the token in a message. `deliver_telegram(cfg, token, delivery, post)` sends one
  multipart `sendVideo` request (`chat_id`, `caption` = task, `experiment/run`, iteration, cut to
  1024 characters, `supports_streaming`, the video) and raises `SinkError` for an `ok: false`
  answer (code, description, `retry_after`), an answer that is not a JSON object with `ok`, a
  transport `OSError`, or a video over 50,000,000 bytes (no request is sent). The token is
  redacted from every message and the original exception is not chained, since the URL holds the
  token. `post` is the injected transport; `urlopen_post` is the `urllib` default and returns an
  HTTP error status instead of raising. One attempt, no retry. `send_text(cfg, token, text, post)`
  sends a plain `sendMessage` through the same request and error handling (used by `test`).
- `watch.py`: `start(cfg, environ, proc_root, post)` returns a `Session(runs, state, sinks)`: it
  reads the Telegram token (`read_token`; `ConfigError` on a problem) and, when it came from the
  environment, removes that variable from `environ` (`os.environ` by default) so `play` never
  inherits it; discovers the runs once (`discover_runs`; `DiscoveryError` propagates); seeds
  `state[run_dir]` from `last_stored_iteration`; and builds the sink callables. The Telegram sink is
  a closure, not a `partial`, so `repr(session)` never shows the token.
  `watch(cfg, session, sleep, render)` loops forever: every pass calls `poll_run` for each run in
  order, one at a time, then `sleep(poll_seconds)`. An `Exception` from one run's pass is logged
  with its traceback and the loop goes on; `KeyboardInterrupt` and `SystemExit` end it.
  `poll_run(cfg, run, state, sinks, render=render_checkpoint)` is one pass for one run. `state`
  maps `run_dir` to the last iteration handled and is updated in place. When `next_due` picks a
  checkpoint it records the iteration **first**, so a failure afterwards never causes a retry; it
  removes the run's work dir `<output.dir>/<experiment>/<run>/.work`, renders into it (left behind afterwards, for `play.log`), stores
  the video and calls each sink with a `Delivery` for the stored video. A `RenderError` or
  `OSError` from the render or the store, and a `SinkError` or `OSError` from a sink, are logged as
  warnings (the other sinks still run); anything else, and a vanished run dir, propagates to
  `watch`. It never logs a sink (its repr can hold the token). `run_name(run)` is
  `<experiment>/<run>`, used in log lines.
- `cli.py`: `main(argv)` is the `mjlab-video-progress` console script (`[project.scripts]`), two
  subcommands. `test --config PATH` sends a test message through each configured sink
  (`telegram.send_text`) and prints `<sink>: ok` or `<sink>: <error>`; exit 0 only if all
  worked (1 also for no sink). No discovery, no render, the environment is left alone.
  `watch --config PATH` logs INFO to stderr, loads the config, calls
  `watch.start`, logs `watching N run(s): ...` and runs `watch.watch`. A `ConfigError` or
  `DiscoveryError` prints `error: <message>` and returns 1; any other exception shows its
  traceback. While the loop runs SIGTERM is turned into `SystemExit` (the old handler is restored
  afterwards), and `KeyboardInterrupt` and `SystemExit` end the loop with exit code 0, after
  `render_checkpoint` has stopped `play` and its child. argparse errors exit 2.

## Hard rules (do not break)

1. **Read-only on training output.** The tool never starts, stops or signals a training
   process, and never writes, moves or deletes under `logs/` or `wandb/`. There is no exception:
   `play` runs on a copy of the checkpoint in the output dir, so its `videos/play/` lands there.
   A test asserts a **scan** leaves the logs tree byte- and mtime-identical. The tool writes
   only: the output dir, run-log files named in the `docs` sink, and an optional `image_dir` /
   `dest`. Never run git from the tool. The tool may signal the process group of its own render
   subprocess (once the video is done, or on timeout), and never anything else.

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

Render, from reading mjlab 1.6.0 source (`play.py`, `video_recorder.py`, `viewer/viser`), and
from a few real renders on 2026-11-12 (see Verification status):

- `play --video` never exits by itself: after the `video_length` frames it keeps running the
  viewer, and headless that is `ViserPlayViewer`, whose `is_running()` is always True. The render
  must stop `play` itself once the video is complete, with the timeout as the hard cap.
- mjlab turns off tyro's flag conversion (`TYRO_FLAGS`), so the flag is `--video True`.
- The video is always `<checkpoint dir>/videos/play/rl-video-step-0.mp4`, overwritten by each
  render, so a failed render can leave the previous video behind. Hence the render runs `play`
  on a copy in `work_dir` and removes any leftover video first.
- `uv run play` starts the real python as a child; signalling only the wrapper would leave it
  running. The render signals the process group it created, never anything else.
- The stop sequence (SIGINT to the process group, SIGKILL after 10 s) is tested with a fake
  `play` and a grandchild process. With a real `uv run --frozen --no-sync play` the last kept
  `play.log` ends with `Ctrl+C received. Shutting down viewer...` / `(viser) Server stopped`, and
  no `play` was left running after four renders. Only the last render's `play.log` survives and
  a SIGKILL escalation is not logged, so whether earlier renders needed the SIGKILL is unknown.
  A group member that is a zombie nobody reaps (a container without an init process) would make
  the stop wait out the full grace.
- Each render starts a viewer: `play --viewer auto` picks `native` only with `DISPLAY` or
  `WAYLAND_DISPLAY` set, else viser (the real watcher ran headless and got viser). viser binds
  `0.0.0.0:8080` and moves to the next port when one is taken (viser `_infra.py`, read, not run
  with two renders), so concurrent renders do not clash. While a render runs its viewer is
  reachable from other hosts; mjlab has no option to bind it to localhost.
- The work dir is per run (`<output.dir>/<experiment>/<run>/.work`), so watchers sharing an
  `output.dir` never touch each other's. Two watchers on the same run would still clash.

Watch, by design:

- A failed render, store or delivery is never retried: the iteration is marked handled before the
  render. A transient failure loses that iteration's video (and Telegram message); the next
  attempt is the next due checkpoint. A restart retries it, because seeding uses stored videos
  only, and a sink that failed is not retried either (no per-sink delivery state exists).
- The GPU load gate (see `TODO.md`) is not built; it would be a check just before the render.
- Runs are discovered once at start. If the training process starts later, or ends, `watch` does
  not notice; it polls the run dirs it found until it is stopped (final-checkpoint detection is
  deferred).
- `watch` relies on its caller to turn SIGTERM into an exception (`SystemExit`): Python's default
  SIGTERM ends the process without running `render_checkpoint`'s `finally`, which would leave
  `play` running. The CLI installs that handler; anything else that calls `watch` must too.
  A SIGKILL of the watcher cannot be handled: a render in progress then leaves `play` running,
  and under viser it never exits by itself, so someone has to stop it.

Telegram, from the Bot API docs at core.telegram.org (`sendVideo`, multipart upload, answer
format), and four real `sendVideo` deliveries on 2026-11-12:

- The request shape works with a real bot (the videos arrived). The 50 MB limit (the docs say
  MB, read as 10^6 bytes, so the code refuses above 50,000,000 bytes) and the error answers were
  read from the docs, not tried. `parameters.retry_after` was not found in the part of the docs
  read; the code uses it only if present.
- One attempt with a 60 s timeout, no retry or backoff: a 429 is only reported. `urllib` follows
  the proxy environment variables.
- The caption is cut at 1024 code points; Telegram counts characters its own way, which only
  matters for non-BMP characters in a task name.
- `watch.start` removes the token environment variable from its own environment before any
  `play` is spawned, so `play` and mjlab code never inherit it. A token from `token_file` is
  never in the environment.

## Verification status (do not overclaim)

Implemented: the config loader (`config.py`); the train-command parser, the `/proc` process
listing, the run-dir lookup, the directory fallback and the glue choosing between them
(`discover_runs`, all in `discovery.py`); the checkpoint scan and the "due" choice (`scan.py`);
the mp4 completeness check and running `play` (`render.py`); the video store (`store.py`); the
Telegram sink (`telegram.py`, `sinks.py`); and the startup and polling loop (`start`, `watch` and
`poll_run` in `watch.py`). Unit tests (`tests/test_config.py` 64, `tests/test_discovery.py` 58,
`tests/test_scan.py` 20, `tests/test_render.py` 30, `tests/test_store.py` 9,
`tests/test_telegram.py` 41, `tests/test_watch.py` 33, `tests/test_cli.py` 19; 274 passing on
Linux, none skipped, plus ruff clean). The two real-`/proc` tests (process listing smoke test,
run-dir lookup via the real `cwd` link) skip on macOS and pass on Linux. The CLI tests patch out
`start` and `watch`, plus two run the real CLI in a subprocess against a fake `play` and send it
SIGTERM and SIGINT. The `test` command's `send_text` (`sendMessage`) sent one real message on
2026-11-12 (`telegram: ok`); its error paths only ran against a fake transport. The `docs` sink and the GPU load gate are not implemented yet.

Real runs (Linux, 2026-11-12): `watch` ran from the CLI against two live mjlab trainings
(one resumed from a checkpoint, each with its own config), with
`command = ["uv", "run", "--frozen", "--no-sync", "play"]` and the Telegram sink. It rendered
iterations 27300, 27800, 28300 (first run) and 24450 (second run); each video was stored and
arrived in Telegram, with no warning in the logs. `find_train_processes` and `find_run_dir`
found the live resumed run through the real `/proc` (checked by hand afterwards; the logs do not
say whether `start` used the process or the directory fallback). That is the whole extent of the
real-world check: a handful of renders on one machine, never two at once. Not seen live: a
failing render, a timeout, a SIGKILL escalation, a Telegram error answer, a checkpoint caught
mid-write by the scan (the zip end record check was only tried by truncating a file on macOS),
or a cut mediapy mp4 (`video_complete` accepted mediapy's finished files, and was otherwise
checked only against mp4s made by ffmpeg, both layouts, every cut). The parser has seen one real
command-line shape (a uv venv `.../bin/train <task> ...`).
