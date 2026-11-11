"""CLI: ``mjlab-video-progress watch --config PATH``.

Contract:

- ``main(argv=None) -> int`` is the console script's entry point (``pyproject.toml`` points
  ``[project.scripts] mjlab-video-progress`` at it). ``argparse`` has one subcommand, ``watch``,
  which needs ``--config PATH``; a missing subcommand, a missing ``--config`` or an unknown
  argument is argparse's own error (usage on stderr, ``SystemExit`` with code 2), and ``--help``
  exits 0.
- ``watch``: sets up logging (``logging.basicConfig``, INFO, to stderr, with time, level and
  logger name), loads the config with ``load_config``, calls ``watch.start(cfg)``, logs
  ``watching N run(s): <experiment>/<run>, ...`` and calls ``watch.watch(cfg, session)``.
  ``watch.start`` and ``watch.watch`` are looked up on the ``watch`` module when called.
- A ``ConfigError`` (from the config or from ``start``, for example the Telegram token) or a
  ``DiscoveryError`` prints ``error: <message>`` to stderr and returns 1. Nothing else is caught,
  so a bug shows its traceback.
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
from collections.abc import Sequence
from pathlib import Path

from mjlab_video_progress import watch as watch_module
from mjlab_video_progress.config import load_config
from mjlab_video_progress.exceptions import ConfigError, DiscoveryError

log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mjlab-video-progress",
        description="Render a video of an mjlab training run every X iterations and deliver it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    watch = commands.add_parser("watch", help="watch a training run until stopped")
    watch.add_argument("--config", type=Path, required=True, help="path to the TOML config")
    return parser


def _exit_on_sigterm(signum, frame):
    raise SystemExit(0)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        cfg = load_config(args.config)
        session = watch_module.start(cfg)
    except (ConfigError, DiscoveryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
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
