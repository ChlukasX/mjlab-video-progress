# mjlab-video-progress

Get videos of your [mjlab](https://github.com/mujocolab/mjlab) training on the go. Every X
iterations it renders the newest checkpoint with mjlab's own `play --video` and sends the video to
Telegram.

It never touches the training: it only reads `logs/`, and renders a copy of the checkpoint in a
separate process with `--num-envs 1` and a hard timeout. Linux only. Plans are in
[TODO.md](TODO.md).

## Setup

You need Linux, Python 3.11+, [uv](https://docs.astral.sh/uv/), `ffmpeg` and `ffprobe` on
`PATH`, and an mjlab checkout in which `play` works.

### 1. Install

```bash
uv tool install git+ssh://git@github.com/ChlukasX/mjlab-video-progress.git
```

If the command is not found afterwards, run `uv tool update-shell` and open a new shell.

### 2. Check that `play` renders

From your mjlab checkout:

```bash
uv run play <task> --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_<N>.pt \
    --video True --video-length 200 --num-envs 1
```

When `videos/play/rl-video-step-0.mp4` appears next to the checkpoint, press Ctrl-C (`play` keeps
running after the video; the tool stops it for you). If this fails, the tool fails too. Variables
`play` needs, such as `MUJOCO_GL=egl` on a headless server, must be set where you start `watch`.

### 3. Write the config

`~/.config/mjlab-video-progress/progress.toml`:

```toml
[watch]
every = 500                  # render every 500 iterations

[render]
command = ["uv", "run", "--frozen", "--no-sync", "play"]   # --frozen --no-sync: leave the training's env alone
cwd = "~/mjlab"              # your mjlab checkout
timeout_seconds = 600

[output]
dir = "~/mjlab-progress"     # videos are stored here
```

Relative paths are relative to the config file. Optional keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `watch.poll_seconds` | `60` | How often to look for new checkpoints. |
| `watch.log_root`, `watch.task` | none | Fallback when no `train` process is found: the newest run under `log_root` (e.g. `"~/mjlab/logs/rsl_rl"`), with this task. |
| `render.num_envs` | `1` | |
| `render.video_length` | `200` | Frames per video. |
| `render.extra_args` | `[]` | Appended to the `play` command, e.g. `["--device", "cuda:0"]`. |
| `render.min_free_gpu_mib` | `2048` | With `nvidia-smi` on `PATH`, a render waits for the next poll while the GPU from `--device` (default `cuda:0`) has less free memory. A render used about 1.3 GB. |

### 4. Telegram

1. Message [@BotFather](https://t.me/BotFather), send `/newbot` and copy the token (`/revoke` if
   it ever leaks).
2. Save it in a file only you can read (the tool refuses any other):

   ```bash
   install -m 600 /dev/null ~/.config/mjlab-video-progress/telegram_token
   $EDITOR ~/.config/mjlab-video-progress/telegram_token
   ```

3. Send your bot any message, then print your chat id:

   ```bash
   mjlab-video-progress chat-id --token-file ~/.config/mjlab-video-progress/telegram_token
   ```

   If it finds no chat, it says where else to get the id.

4. Add to the config, then check it with `mjlab-video-progress test --config <config>` (prints
   `telegram: ok` or the error):

   ```toml
   [sinks.telegram]
   chat_id = 123456789
   token_file = "~/.config/mjlab-video-progress/telegram_token"
   ```

For a group use its (negative) id; for a channel make the bot an admin and use
`chat_id = "@channelname"`. Instead of `token_file` you can set `TELEGRAM_BOT_TOKEN` (or the
variable named in `token_env`); `play` never sees it. Videos over 50 MB are not sent.

### 5. Run

Once the training has created its run directory:

```bash
tmux new -s progress
mjlab-video-progress watch --config ~/.config/mjlab-video-progress/progress.toml
```

Detach with `Ctrl-b d`. Ctrl-C or SIGTERM stops it, including a render in progress.

- It finds your running `train` process once at start (falling back to `watch.log_root`), and
  keeps polling that run until stopped.
- Videos land in `<output.dir>/<experiment>/<run>/iter_000500/video.mp4`; after a restart it
  continues after the last one.
- Failures are logged as warnings and not retried. A run's last `play` output is in
  `<output.dir>/<experiment>/<run>/.work/play.log`.

### Other install options

To pin it in your mjlab project instead (it has no dependencies of its own), run this **while no
training is running**, since `uv add` re-syncs the training's environment:

```bash
uv add --dev "git+ssh://git@github.com/ChlukasX/mjlab-video-progress.git"
uv run --frozen --no-sync mjlab-video-progress watch --config progress.toml
```

To work on the tool: clone it, `uv sync`, `uv run mjlab-video-progress ...`. Git installs need
SSH access to the GitHub repository.

## Status

Early development. Has run against real mjlab trainings and delivered videos to Telegram; failure
paths (timeouts, Telegram errors) are only tested with fakes.
