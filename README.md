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
decided in a later version), `discord`, rendering a single checkpoint on demand, final-checkpoint detection, backfilling past
checkpoints, a W&B checkpoint source, and macOS and Windows.

## Status

Early development. Nothing is implemented yet.
