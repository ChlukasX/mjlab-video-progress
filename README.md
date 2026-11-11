# mjlab-video-progress

A tool that watches an [mjlab](https://github.com/mujocolab/mjlab)
training run and, every X iterations, makes a **bundle** for a checkpoint and hands it to
**sinks**. Allowing you to get a video of your training on the go!

The renderer is mjlab's own `play --video` command, run as a separate low-impact subprocess
beside the live training. The training run is never interrupted.

## Scope (v1)

- A `watch` command that polls a run's local `logs/` directory (read-only) and renders a
  checkpoint every X iterations.
- A bundle per render: for now just `video.mp4`, stored at
  `<output.dir>/<experiment>/<run>/iter_<N>/video.mp4`.
- Sinks: `local`, `docs` (markdown run log) and `telegram`. They deliver the video only.
- Safety: a hard render timeout, `--num-envs 1` by default, and a GPU load gate that turns on
  automatically when `nvidia-smi` is found.
- Platform: Linux. Needs `ffmpeg` and `ffprobe` on `PATH`.

## Deferred

`strip.png`, `eval.json`, `manifest.json` and any other sink content beyond the video (to be
decided in a later version), `discord`, rendering a single checkpoint on demand,
final-checkpoint detection, backfilling past checkpoints, a W&B checkpoint source, and macOS
and Windows.

## Installation and setup

### Requirements

- Linux (the supported platform), Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- `ffmpeg` and `ffprobe` on `PATH`.
- An mjlab checkout in which `play` already works. This tool does not import mjlab and has no
  runtime dependencies: it runs the command you configure, from the directory you configure.

### Install

```bash
git clone git@github.com:ChlukasX/mjlab-video-progress.git
uv tool install ./mjlab-video-progress     # puts `mjlab-video-progress` on your PATH
```

or, without a clone, `uv tool install git+ssh://git@github.com/ChlukasX/mjlab-video-progress.git`.
If the command is not found afterwards, run `uv tool update-shell` and open a new shell.
To work on the tool itself, use `uv sync` and `uv run mjlab-video-progress ...` instead.

#### Alternative: add it to your mjlab project

If you would rather have the tool next to mjlab and pinned in the project's lock file, add it to
the mjlab project's `pyproject.toml`:

```bash
uv add --dev "git+ssh://git@github.com/ChlukasX/mjlab-video-progress.git"
```

This adds the following, and `uv run mjlab-video-progress watch --config progress.toml` then
works from the project:

```toml
[dependency-groups]
dev = ["mjlab-video-progress"]

[tool.uv.sources]
mjlab-video-progress = { git = "ssh://git@github.com/ChlukasX/mjlab-video-progress.git" }
```

The dev group keeps the tool out of the project's normal dependencies, and because the tool has
no dependencies of its own it adds nothing else to the lock file. To try it without changing
`pyproject.toml` at all, run it from a temporary overlay instead:

```bash
uv run --with "git+ssh://git@github.com/ChlukasX/mjlab-video-progress.git" \
    mjlab-video-progress watch --config progress.toml
```

On a server with a training running, be careful: `uv add` and a plain `uv run` re-lock and sync
the project's environment, which is the one the training uses. Either install the tool with
`uv tool install` as above, or add the dependency while no training is running and start it
afterwards with `uv run --frozen --no-sync mjlab-video-progress ...`, which leaves both the lock
file and the environment alone. In every case the server needs SSH access to the GitHub
repository (a deploy key or a forwarded agent).

### Check that `play` works by hand

The tool renders by running mjlab's own `play` on a copy of a checkpoint. Run the same thing once
yourself, from your mjlab checkout, on a checkpoint of your task:

```bash
uv run play <task> --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_<N>.pt \
    --video True --video-length 200 --num-envs 1
```

The video is written to `<checkpoint dir>/videos/play/rl-video-step-0.mp4`. From the mjlab 1.6.0
source, `play` does not exit after writing it (it keeps its viewer running), so stop it with
Ctrl-C; the tool stops it for you. If this does not work by hand, for example because the server
has no display and offscreen rendering is not set up, it will not work through the tool either.
Anything `play` needs from the environment (such as a `MUJOCO_GL` setting) has to be set in the
shell you start `watch` from, because `play` inherits it.

### Configure

Create a TOML file, for example `progress.toml`. Relative paths are relative to this file, and a
leading `~` is expanded.

```toml
[watch]
every = 500                # render when a checkpoint is at least this many iterations after the
                           # last rendered one
poll_seconds = 60          # how often to look for new checkpoints (default 60)
# Only used when no training process is found: take the newest run under log_root.
log_root = "~/mjlab/logs/rsl_rl"
task = "<your mjlab task name>"

[render]
command = ["uv", "run", "play"]   # the tool appends: <task> --checkpoint-file ... --video True
cwd = "~/mjlab"                   # run it from your mjlab checkout
timeout_seconds = 600             # hard cap per render; play is stopped after this
num_envs = 1                      # default 1, to keep the load on the GPU low
video_length = 200                # frames (default 200)
extra_args = []                   # appended last, e.g. ["--device", "cuda:0"]

[output]
dir = "~/mjlab-progress"          # the videos are stored here

[sinks.telegram]                  # optional; leave this table out to only store videos
chat_id = 123456789               # a number (groups and channels are negative) or "@channelname"
token_file = "~/.config/mjlab-video-progress/telegram_token"
```

The first render happens once a checkpoint at iteration `every` or later exists, then one for
each checkpoint at least `every` iterations after the last one. If mjlab saves every 50
iterations and `every = 500`, you get videos at 500, 1000, and so on. If the renders fall behind,
only the newest due checkpoint is rendered.

### Telegram bot (optional)

1. Create a bot: message [@BotFather](https://t.me/BotFather), send `/newbot`, and keep the token
   it gives you. On a shared server everyone should make their own bot. If a token leaks, send
   `/revoke` to BotFather.
2. Store the token in a file only you can read, outside the repository:

   ```bash
   mkdir -p ~/.config/mjlab-video-progress
   install -m 600 /dev/null ~/.config/mjlab-video-progress/telegram_token
   $EDITOR ~/.config/mjlab-video-progress/telegram_token     # paste the token and save
   ```

   The tool refuses a token file that group or others can read. Instead of `token_file` you can
   set the environment variable `TELEGRAM_BOT_TOKEN` (or the name given in `token_env`); `watch`
   removes it from its own environment before it starts `play`. Never put the token in the
   config file or on a command line, where other users can see it.
3. Find your chat id: send any message to your bot first (a bot cannot write first), then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and look for `"chat":{"id":...}`. Do this on
   your own machine, not on a shared server, because the token is part of that URL and shows up
   in process lists. For a group, add the bot and write a message; for a channel, add the bot as
   an administrator and use `chat_id = "@channelname"`.

Videos over 50 MB are not sent (Telegram's limit for bots).

### Run

```bash
mjlab-video-progress watch --config progress.toml
```

- Start it while the training is running and after the training has created its run directory
  (mjlab does that once the environment is built). It looks for your own `train` process under
  `/proc` and takes the task and the run directory from it. If it finds none, it uses the newest
  run under `watch.log_root` with `watch.task`. This happens once, at start.
- It does not stop when the training ends. Stop it with Ctrl-C or SIGTERM, which also stops a
  render in progress. To keep it running after you log out, use `tmux`, `screen` or a systemd
  user service.
- It renders one checkpoint at a time and logs to stderr. A failed render or delivery is logged
  and not retried; the next attempt is the next due checkpoint.
- The videos are stored as `<output.dir>/<experiment>/<run>/iter_000500/video.mp4`. The output
  of the latest render is in `<output.dir>/.work/play.log`; look there when renders fail. After a
  restart, `watch` continues after the last stored video.
- It only reads the training's `logs/` directory; it never writes there.

## Status

Early development. `mjlab-video-progress watch --config PATH` runs end to end in the tests, against
a fake `play` and a fake Telegram transport. It has not yet been run against a real mjlab training
or the real Telegram API. The `docs` sink and the GPU load gate are not built.
