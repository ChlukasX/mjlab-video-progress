# TODO

## v1 (in scope, not done)

- [x] Run `watch` against a real mjlab training (real `play`, real `/proc`, real checkpoints).
- [x] Send one real message through the Telegram sink.
- [ ] `docs` sink: markdown run log.
- [x] GPU load gate: check just before the render, on automatically when `nvidia-smi` is found
  (free memory only, `render.min_free_gpu_mib`).

## Bugs

- [x] `.work` clash: every watcher with the same `output.dir` renders into the same
  `<output.dir>/.work`, so two concurrent renders delete each other's work dir. Fix: a work dir
  per run (`<output.dir>/<experiment>/<run>/.work`).

## Planned

- [x] `chat-id` command: print the chat ids the bot has seen (`getUpdates`, token read like
  `watch` does), or say where else to get the id; replaces the Python one-liner in the README.
- [x] `test` command: send a test message to every configured sink (token, chat id, network).
- [ ] Pick up each new `train` process as it appears, instead of discovering runs once at start.

## Deferred (later version)

- [ ] `strip.png`, `eval.json`, `manifest.json`
- [ ] `discord` sink
- [ ] Render a single checkpoint on demand
- [ ] Final-checkpoint detection
- [ ] Backfill past checkpoints
- [ ] W&B checkpoint source
- [ ] macOS and Windows

## Cleanup

- [x] `AGENTS.md`/`CLAUDE.md`: drop the stale "roadmap item 8; there is no `watch` yet" note.
- [x] Simplify the `README.md` file, installation, and setup steps specifically.
