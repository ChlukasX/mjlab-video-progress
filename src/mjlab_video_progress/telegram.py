"""Telegram sink: send a stored video to a chat with the Bot API's ``sendVideo``.

Contract:

- ``read_token(cfg, environ=None)`` returns the bot token, or raises ``ConfigError`` (never with
  the token in the message). If ``cfg.token_file`` is set it is used, and the environment is not
  looked at: the file must be a regular file, must not be readable by group or others (any of
  the 0o077 mode bits is refused, with a ``chmod 600`` hint, like ``ssh`` does for keys) and must
  hold a non-empty token; surrounding whitespace, such as the trailing newline, is stripped.
  Otherwise the token is ``environ[cfg.token_env]`` (``os.environ`` by default), stripped, and
  must be set and non-empty. It does not remove anything from the environment; ``watch`` does
  that so ``play`` never inherits the token.
- ``deliver_telegram(cfg, token, delivery, post=urlopen_post)`` sends one request to
  ``https://api.telegram.org/bot<token>/sendVideo`` and returns ``None`` when Telegram answers
  ``ok``. The request is ``multipart/form-data`` with ``chat_id`` (``str(cfg.chat_id)``),
  ``caption``, ``supports_streaming=true`` and the file as ``video`` (``video.mp4``,
  ``video/mp4``). The caption is ``"<task>\n<experiment>/<run>\niteration <N>"`` as plain text
  (no parse mode), cut to 1024 characters. A missing or unreadable video raises ``OSError``
  before any request. A video over ``MAX_VIDEO_BYTES`` raises ``SinkError`` without a request.
  One attempt only; retries are the caller's business.
- ``SinkError`` covers: an answer with ``ok`` false (the message has the error code, the
  description and, if given, ``parameters.retry_after``), an answer that is not a JSON object
  with an ``ok`` field, and a transport that raises ``OSError`` (``URLError``, timeouts,
  connection errors). No ``SinkError`` message or traceback contains the token: it is redacted
  from any text taken from the transport or from Telegram, and the original exception is not
  chained (``raise ... from None``), since the token is part of the URL.
- ``post(url, headers, body) -> (status, body)`` is the injected transport, so tests never reach
  the network. ``urlopen_post`` is the default: ``urllib``, ``TIMEOUT_SECONDS`` timeout, and an
  HTTP error status is returned like any other (Telegram explains errors in the body); a
  connection failure raises ``OSError``.
"""

from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from mjlab_video_progress.config import TelegramConfig
from mjlab_video_progress.exceptions import ConfigError, SinkError
from mjlab_video_progress.sinks import Delivery

API_BASE = "https://api.telegram.org"
MAX_VIDEO_BYTES = 50_000_000  # the Bot API docs say 50 MB, read as 10^6 bytes per MB
TIMEOUT_SECONDS = 60
CAPTION_LIMIT = 1024

Post = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]


def _read_token_file(path: Path) -> str:
    key = "sinks.telegram.token_file"
    try:
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ConfigError(f"{key}: {path} is not a regular file")
        if mode & 0o077:
            raise ConfigError(
                f"{key}: {path} is readable by group or others; run: chmod 600 {path}"
            )
        return path.read_text().strip()
    except OSError as e:
        raise ConfigError(f"{key}: cannot read {path}: {e.strerror}") from e
    except UnicodeDecodeError:
        raise ConfigError(f"{key}: {path} is not a text file") from None


def read_token(cfg: TelegramConfig, environ: Mapping[str, str] | None = None) -> str:
    if cfg.token_file is not None:
        token = _read_token_file(cfg.token_file)
        if not token:
            raise ConfigError(f"sinks.telegram.token_file: {cfg.token_file} is empty")
        return token
    token = (os.environ if environ is None else environ).get(cfg.token_env, "").strip()
    if not token:
        raise ConfigError(
            f"sinks.telegram.token_env: environment variable {cfg.token_env} is not set or empty"
        )
    return token


def urlopen_post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:  # Telegram explains an error in the body
        return e.code, e.read()


def _form(fields: dict[str, str], video: bytes) -> tuple[bytes, str]:
    """A multipart/form-data body with the text fields and the file as ``video``."""
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        + value.encode()
        + b"\r\n"
        for name, value in fields.items()
    ]
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="video.mp4"'
            "\r\nContent-Type: video/mp4\r\n\r\n"
        ).encode()
        + video
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def deliver_telegram(
    cfg: TelegramConfig, token: str, delivery: Delivery, post: Post = urlopen_post
) -> None:
    def redact(text: str) -> str:
        return text.replace(token, "<token>") if token else text

    size = delivery.video.stat().st_size
    if size > MAX_VIDEO_BYTES:
        raise SinkError(f"video is {size} bytes, over Telegram's limit of {MAX_VIDEO_BYTES} bytes")
    run_dir = delivery.run_dir
    caption = (
        f"{delivery.task}\n{run_dir.parent.name}/{run_dir.name}\niteration {delivery.iteration}"
    )
    body, content_type = _form(
        {
            "chat_id": str(cfg.chat_id),
            "caption": caption[:CAPTION_LIMIT],
            "supports_streaming": "true",
        },
        delivery.video.read_bytes(),
    )
    try:
        status, answer = post(
            f"{API_BASE}/bot{token}/sendVideo", {"Content-Type": content_type}, body
        )
    except OSError as e:  # the URL, and so the token, can be in the message: do not chain it
        raise SinkError(
            f"request to Telegram failed: {redact(f'{type(e).__name__}: {e}')}"
        ) from None
    try:
        data = json.loads(answer)
    except ValueError:
        data = None
    if not isinstance(data, dict) or "ok" not in data:
        raise SinkError(f"unexpected response from Telegram (HTTP {status})")
    if data["ok"] is True:
        return
    reason = f"Telegram error {data.get('error_code', status)}: {data.get('description')}"
    params = data.get("parameters")
    if isinstance(params, dict) and "retry_after" in params:
        reason += f" (retry after {params['retry_after']}s)"
    raise SinkError(redact(reason))
