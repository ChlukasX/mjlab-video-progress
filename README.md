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

Planned:

- Starting automatically with a training: a long-running `watch` that picks up each new `train`
  process (and its run directory) as it appears, instead of finding the runs once at start, so
  it does not have to be started by hand for every training.

## Setup

You need Linux, Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), `ffmpeg` and `ffprobe`
on `PATH`, and an mjlab checkout in which `play` works.

### 1. Install

```bash
uv tool install git+ssh://git@github.com/ChlukasX/mjlab-video-progress.git
```

If `mjlab-video-progress` is not found afterwards, run `uv tool update-shell` and open a new
shell. Other ways to install are under [Other install options](#other-install-options).

### 2. Check that `play` works

The tool renders by running mjlab's `play` on a copy of a checkpoint. Run it once by hand from
your mjlab checkout:

```bash
uv run play <task> --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_<N>.pt \
    --video True --video-length 200 --num-envs 1
```

Once `<checkpoint dir>/videos/play/rl-video-step-0.mp4` exists, stop it with Ctrl-C (`play`
keeps running after the video is written; the tool stops it for you). If this fails, for example
because offscreen rendering is not set up, the tool fails too. Environment variables `play` needs
(such as `MUJOCO_GL`) must be set in the shell you start `watch` from.

### 3. Write the config

Save this as `~/.config/mjlab-video-progress/progress.toml` and change `cwd`:

```toml
[watch]
every = 500                  # render every 500 iterations

[render]
command = ["uv", "run", "--frozen", "--no-sync", "play"]
cwd = "~/mjlab"              # your mjlab checkout
timeout_seconds = 600        # hard cap per render

[output]
dir = "~/mjlab-progress"     # videos are stored here
```

`--frozen --no-sync` stops uv from re-syncing the environment the running training uses. A
leading `~` is expanded, and relative paths are relative to the config file. Optional keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `watch.poll_seconds` | `60` | How often to look for new checkpoints. |
| `watch.log_root`, `watch.task` | none | Used only when no `train` process is found: the newest run under `log_root` (e.g. `"~/mjlab/logs/rsl_rl"`) with this task. |
| `render.num_envs` | `1` | Kept at 1 to keep the GPU load low. |
| `render.video_length` | `200` | Frames per video. |
| `render.extra_args` | `[]` | Appended to the `play` command, e.g. `["--device", "cuda:0"]`. |

With `every = 500` and mjlab saving every 50 iterations you get videos at 500, 1000, and so on.
If renders fall behind, only the newest due checkpoint is rendered.

### 4. Telegram (optional)

1. Message [@BotFather](https://t.me/BotFather), send `/newbot` and copy the token. On a shared
   server everyone makes their own bot. If the token leaks, send `/revoke` to BotFather.
2. Save the token in a file only you can read (the tool refuses any other):

   ```bash
   install -m 600 /dev/null ~/.config/mjlab-video-progress/telegram_token
   $EDITOR ~/.config/mjlab-video-progress/telegram_token     # paste the token, save
   ```

3. Send your bot any message (a bot cannot write first), then print your chat id. This reads
   the token from the file, so it never appears on a command line:

   ```bash
   python3 -c 'import json,os,urllib.request as u; t=open(os.path.expanduser("~/.config/mjlab-video-progress/telegram_token")).read().strip(); r=json.load(u.urlopen(f"https://api.telegram.org/bot{t}/getUpdates"))["result"]; print({c["id"]: c.get("title") or c.get("username") for c in (x.get("message", x.get("channel_post", {})).get("chat") for x in r) if c} or "no messages yet: send your bot a message first")'
   ```

4. Add to the config:

   ```toml
   [sinks.telegram]
   chat_id = 123456789          # the number printed above
   token_file = "~/.config/mjlab-video-progress/telegram_token"
   ```

For a group, add the bot, post a message and use the (negative) id. For a channel, make the bot
an administrator and use `chat_id = "@channelname"`. Instead of `token_file` you can set
`TELEGRAM_BOT_TOKEN` (or the variable named in `token_env`); `watch` removes it from its own
environment before it starts `play`. Never put the token in the config or on a command line.
Videos over 50 MB are not sent (Telegram's limit for bots).

Check the setup (token, chat id, network) without waiting for the first render:

```bash
mjlab-video-progress test --config ~/.config/mjlab-video-progress/progress.toml
```

It sends one test message to the chat and prints `telegram: ok`, or the error.

### 5. Run

Start it after the training has created its run directory (mjlab does that once the environment
is built), inside `tmux` so it survives logging out:

```bash
tmux new -s progress
mjlab-video-progress watch --config ~/.config/mjlab-video-progress/progress.toml
```

Detach with `Ctrl-b d`; stop it with Ctrl-C (or SIGTERM), which also stops a render in progress.

- At start it finds your own `train` process under `/proc` and takes the task and run directory
  from it (falling back to `watch.log_root`). It does this once and does not stop when the
  training ends.
- Videos are stored as `<output.dir>/<experiment>/<run>/iter_000500/video.mp4`. After a restart
  it continues after the last stored video.
- A failed render or delivery is logged as a warning and not retried. The output of the latest
  render of a run is in `<output.dir>/<experiment>/<run>/.work/play.log`.
- It only reads the training's `logs/` directory; it never writes there.

### Other install options

To pin the tool in your mjlab project's lock file instead, add it as a dev dependency
**while no training is running** (`uv add` re-syncs the environment the training uses):

```bash
uv add --dev "git+ssh://git@github.com/ChlukasX/mjlab-video-progress.git"
uv run --frozen --no-sync mjlab-video-progress watch --config progress.toml
```

The tool has no dependencies of its own, so this adds nothing else to the lock file. To work on
the tool itself, clone it and use `uv sync` and `uv run mjlab-video-progress ...`. Every git
install needs SSH access to the GitHub repository (a deploy key or a forwarded agent).

## Status

Early development. `mjlab-video-progress watch --config PATH` runs end to end in the tests, and
has run against two real mjlab trainings, delivering a handful of videos to Telegram. Failure
paths (timeouts, Telegram errors) have only been tested with fakes. The `docs` sink and the GPU
load gate are not built.
