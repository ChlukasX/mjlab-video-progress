"""CLI: ``mjlab-video-progress watch|test --config PATH``, ``ps``, ``stop`` and ``chat-id``.

Contract:

- ``main(argv=None) -> int`` is the console script's entry point (``pyproject.toml`` points
  ``[project.scripts] mjlab-video-progress`` at it). ``argparse`` has five subcommands:
  ``watch`` and ``test`` need ``--config PATH``, ``chat-id`` takes ``--token-file PATH``, ``ps``
  takes nothing, ``stop`` takes pids or ``--all`` (exactly one of them); a missing subcommand, a
  missing ``--config``, ``--log`` without ``-d`` or an unknown argument is argparse's own error
  (usage on stderr, ``SystemExit`` with code 2), and ``--help`` exits 0.
- ``watch``: sets up logging (``logging.basicConfig``, INFO, to stderr, with time, level and
  logger name), loads the config with ``load_config``, calls ``watch.start(cfg)``, logs
  ``watching N run(s): <experiment>/<run>, ...`` and calls ``watch.watch(cfg, session)``.
  ``watch.start`` and ``watch.watch`` are looked up on the ``watch`` module when called.
- ``watch -d`` (``--detach``) does the same up to and including ``start`` in the foreground, so
  a config, discovery or lock error still shows in the terminal; then ``watchers.detach(log)``
  (``--log PATH``, default ``watchers.default_log(config)``). The original process prints
  ``watcher <pid> started, log: <path>`` and returns 0; the detached one logs to that file and
  runs the loop as above. ``detach`` and the other ``watchers`` functions are looked up when
  called.
- ``ps``: ``watchers.find_watchers()`` as a table (``PID UPTIME CONFIG RUNS LOG``, ``-`` for
  unknown, runs as ``<experiment>/<run>``), or ``no watchers running``. Returns 0.
- ``stop PID... | --all``: a pid that ``find_watchers`` does not list is an error (nothing is
  signalled, returns 1). Otherwise ``watchers.stop`` on those pids (``--all``: every watcher;
  none is not an error), printing ``stopped <pid>`` or ``error: <pid> still running`` for each.
  Returns 0 only if all stopped.
- ``test``: loads the config and sends ``TEST_MESSAGE`` through every configured sink
  (``telegram.send_text``, looked up when called), printing ``<sink>: ok`` or
  ``<sink>: <error>`` per sink to stdout. Returns 0 if every sink worked, 1 if one failed (a
  ``ConfigError`` such as a missing token, or a ``SinkError``) or no sink is configured. It does
  no discovery and no render, and leaves the environment alone.
- ``chat-id``: reads the bot token with ``telegram.read_token`` (from ``--token-file``, with its
  permission checks, else from ``TELEGRAM_BOT_TOKEN``), calls ``telegram.get_chats`` (looked up
  when called) and prints ``<id>  <name>`` per chat to stdout, returning 0. It needs no config,
  since the chat id is what it finds. No chat prints ``NO_CHATS`` (how to get a message to the
  bot, or the id from elsewhere); that, a ``ConfigError`` or a ``SinkError`` goes to stderr
  and returns 1.
- A ``ConfigError`` (from the config or from ``start``, for example the Telegram token), a
  ``DiscoveryError`` or a ``WatcherConflict`` prints ``error: <message>`` to stderr and
  returns 1. Nothing else is caught, so a bug shows its traceback.
- While ``watch.watch`` runs, SIGTERM is turned into ``SystemExit``, because Python's default
  SIGTERM ends the process without running ``render_checkpoint``'s ``finally`` and would leave
  ``play`` running; the previous handler is restored on the way out. ``KeyboardInterrupt``
  (SIGINT) and ``SystemExit`` both mean a clean stop: logged, return 0. In both cases a running
  render is stopped first, so ``play`` and its child are gone when the process exits.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from mjlab_video_progress import telegram as telegram_module
from mjlab_video_progress import watch as watch_module
from mjlab_video_progress import watchers as watchers_module
from mjlab_video_progress.config import Config, TelegramConfig, load_config
from mjlab_video_progress.exceptions import (
    ConfigError,
    DiscoveryError,
    SinkError,
    WatcherConflict,
)

log = logging.getLogger(__name__)

NO_CHATS = """\
error: your bot has not seen any chat yet (Telegram keeps its messages for about a day).
Send your bot a message (for a group: add the bot and post there), then run this again.
Or get the id elsewhere: message @userinfobot for your own id, or use "@channelname" as
chat_id for a public channel the bot administers."""

TEST_MESSAGE = "mjlab-video-progress: test message, this sink works."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mjlab-video-progress",
        description="Render a video of an mjlab training run every X iterations and deliver it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    watch = commands.add_parser("watch", help="watch a training run until stopped")
    watch.add_argument("--config", type=Path, required=True, help="path to the TOML config")
    watch.add_argument("-d", "--detach", action="store_true", help="run in the background")
    watch.add_argument("--log", type=Path, help="log file with -d (default: ~/.local/state/...)")
    commands.add_parser("ps", help="list the running watchers")
    stop = commands.add_parser("stop", help="stop running watchers (SIGTERM, then wait)")
    stop.add_argument("pids", nargs="*", type=int, metavar="PID", help="pids from `ps`")
    stop.add_argument("--all", action="store_true", help="stop every watcher")
    test = commands.add_parser("test", help="send a test message through every configured sink")
    test.add_argument("--config", type=Path, required=True, help="path to the TOML config")
    chat_id = commands.add_parser("chat-id", help="print the chat ids your Telegram bot has seen")
    chat_id.add_argument(
        "--token-file", type=Path, help="bot token file (default: $TELEGRAM_BOT_TOKEN)"
    )
    return parser


def _chat_id(token_file: Path | None) -> int:
    path = token_file.expanduser() if token_file else None
    try:
        token = telegram_module.read_token(TelegramConfig(chat_id=0, token_file=path))
        chats = telegram_module.get_chats(token)
    except (ConfigError, SinkError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not chats:
        print(NO_CHATS, file=sys.stderr)
        return 1
    for chat, name in chats.items():
        print(f"{chat}  {name}".rstrip())
    return 0


def _test(cfg: Config) -> int:
    telegram = cfg.sinks.telegram
    if telegram is None:
        print("error: no sink configured", file=sys.stderr)
        return 1
    try:
        token = telegram_module.read_token(telegram)
        telegram_module.send_text(telegram, token, TEST_MESSAGE)
    except (ConfigError, SinkError) as e:
        print(f"telegram: {e}")
        return 1
    print("telegram: ok")
    return 0


def _ps() -> int:
    found = watchers_module.find_watchers()
    if not found:
        print("no watchers running")
        return 0
    now = time.time()
    rows = [("PID", "UPTIME", "CONFIG", "RUNS", "LOG")]
    for w in found:
        uptime = str(timedelta(seconds=int(now - w.started))) if w.started else "-"
        runs = ",".join(f"{lock.parent.parent.name}/{lock.parent.name}" for lock in w.locks)
        rows.append((str(w.pid), uptime, str(w.config or "-"), runs, str(w.log or "-")))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for row in rows:
        print("  ".join([*(c.ljust(n) for c, n in zip(row, widths, strict=False)), row[4]]))
    return 0


def _stop(pids: list[int], everything: bool) -> int:
    known = {w.pid for w in watchers_module.find_watchers()}
    unknown = [pid for pid in pids if pid not in known]
    if unknown:
        for pid in unknown:
            print(f"error: {pid} is not a running watcher (see `ps`)", file=sys.stderr)
        return 1
    targets = sorted(known) if everything else pids
    left = watchers_module.stop(targets)
    for pid in targets:
        if pid in left:
            print(f"error: {pid} still running", file=sys.stderr)
        else:
            print(f"stopped {pid}")
    return 1 if left else 0


def _exit_on_sigterm(signum, frame):
    raise SystemExit(0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "watch" and args.log and not args.detach:
        parser.error("--log needs -d")
    if args.command == "stop" and bool(args.pids) == args.all:
        parser.error("stop needs pids or --all, not both")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.command == "chat-id":
        return _chat_id(args.token_file)
    if args.command == "ps":
        return _ps()
    if args.command == "stop":
        return _stop(args.pids, args.all)
    if args.command == "test":
        try:
            return _test(load_config(args.config))
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    try:
        cfg = load_config(args.config)
        session = watch_module.start(cfg)
    except (ConfigError, DiscoveryError, WatcherConflict) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.detach:
        log_file = (args.log or watchers_module.default_log(args.config)).expanduser().absolute()
        pid = watchers_module.detach(log_file)
        if pid is not None:  # the original process; the detached one goes on below
            print(f"watcher {pid} started, log: {log_file}")
            return 0
    log.info(
        "watching %d run(s): %s",
        len(session.runs),
        ", ".join(watch_module.run_name(run) for run in session.runs),
    )
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        watch_module.watch(cfg, session)
    except (KeyboardInterrupt, SystemExit):
        pass  # a stop: render_checkpoint has already stopped play on the way out
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
    log.info("stopped")
    return 0
