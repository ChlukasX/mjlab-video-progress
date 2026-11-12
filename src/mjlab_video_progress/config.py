"""Config: load and validate the TOML config file.

Contract:

- ``load_config(path)`` returns a frozen ``Config`` or raises ``ConfigError``.
- A missing file or a TOML syntax error raises ``ConfigError`` naming the file.
- Unknown keys, wrong types, missing required keys and non-positive integers raise
  ``ConfigError`` naming the full key path (for example ``render.timeout_seconds``).
- Integers must be ``int`` (not ``bool``) and strictly positive.
- Empty strings are rejected (``watch.task``, paths, ``render.command`` elements), except
  inside ``render.extra_args``.
- A leading ``~`` or ``~user`` in a path expands to that home directory, the way a shell expands
  ``--log-root ~/x`` before mjlab sees it (mjlab itself never expands ``~``). An unknown user or
  a missing home directory raises ``ConfigError`` naming the key. A ``~`` elsewhere is literal.
- ``render.min_free_gpu_mib`` (positive int, default 2048) is the free memory the GPU gate in
  ``watch.poll_run`` requires before a render (a real render peaked at 1289 MiB).
- Relative paths resolve against the config file's directory; absolute paths are unchanged.
- Loading never touches the filesystem under ``watch.log_root`` (hard rule 1).

- ``[sinks.telegram]`` is optional; without it ``Config.sinks.telegram`` is ``None``. Its keys:
  ``chat_id`` (required: an int, which may be negative, or a non-empty string, never a bool),
  ``token_file`` (optional path, ``~`` expanded, relative paths resolved against the config file)
  and ``token_env`` (non-empty string, default ``TELEGRAM_BOT_TOKEN``). Both token keys may be
  set; the sink uses ``token_file`` if set, else the environment variable. Loading never reads
  the token file or the environment. Any other sink table is rejected until that sink exists.

Tables: ``[watch]``, ``[render]``, ``[output]`` and ``[sinks]``. Any other key is rejected.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from mjlab_video_progress.exceptions import ConfigError


@dataclass(frozen=True)
class WatchConfig:
    every: int
    poll_seconds: int = 60
    log_root: Path | None = None
    task: str | None = None


@dataclass(frozen=True)
class RenderConfig:
    command: tuple[str, ...]
    timeout_seconds: int
    num_envs: int = 1
    video_length: int = 200
    extra_args: tuple[str, ...] = ()
    cwd: Path | None = None
    min_free_gpu_mib: int = 2048


@dataclass(frozen=True)
class OutputConfig:
    dir: Path


@dataclass(frozen=True)
class TelegramConfig:
    chat_id: int | str
    token_file: Path | None = None
    token_env: str = "TELEGRAM_BOT_TOKEN"


@dataclass(frozen=True)
class SinksConfig:
    telegram: TelegramConfig | None = None


@dataclass(frozen=True)
class Config:
    watch: WatchConfig
    render: RenderConfig
    output: OutputConfig
    sinks: SinksConfig = SinksConfig()


_REQUIRED = object()


def _int(value, key, base):
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{key}: must be a positive integer, got {value!r}")
    return value


def _str(value, key, base):
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key}: must be a non-empty string, got {value!r}")
    return value


def _path(value, key, base):
    try:
        path = Path(_str(value, key, base)).expanduser()
    except RuntimeError as e:  # ~user of an unknown user, or no home directory
        raise ConfigError(f"{key}: cannot expand ~ in {value!r}") from e
    return base / path  # joining an absolute path keeps it unchanged


def _strs(value, key, base):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{key}: must be a list of strings, got {value!r}")
    return tuple(value)


def _chat_id(value, key, base):
    if type(value) is int or (isinstance(value, str) and value):  # not bool; ids can be negative
        return value
    raise ConfigError(f"{key}: must be an integer or a non-empty string, got {value!r}")


def _command(value, key, base):
    if not _strs(value, key, base) or "" in value:
        raise ConfigError(f"{key}: must be a non-empty list of non-empty strings")
    return tuple(value)


# Schema: table -> key -> (validator, default or _REQUIRED).
_SPEC = {
    "watch": {
        "every": (_int, _REQUIRED),
        "poll_seconds": (_int, 60),
        "log_root": (_path, None),
        "task": (_str, None),
    },
    "render": {
        "command": (_command, _REQUIRED),
        "timeout_seconds": (_int, _REQUIRED),
        "num_envs": (_int, 1),
        "video_length": (_int, 200),
        "extra_args": (_strs, ()),
        "cwd": (_path, None),
        "min_free_gpu_mib": (_int, 2048),
    },
    "output": {"dir": (_path, _REQUIRED)},
    "sinks.telegram": {
        "chat_id": (_chat_id, _REQUIRED),
        "token_file": (_path, None),
        "token_env": (_str, "TELEGRAM_BOT_TOKEN"),
    },
}


def _table(data: dict, name: str, base: Path) -> dict:
    raw = data.get(name.rpartition(".")[2], {})  # "sinks.telegram" is data["telegram"] here
    if not isinstance(raw, dict):
        raise ConfigError(f"{name}: must be a table")
    spec = _SPEC[name]
    for key in raw:
        if key not in spec:
            raise ConfigError(f"unknown key {name}.{key}")
    out = {}
    for key, (check, default) in spec.items():
        if key in raw:
            out[key] = check(raw[key], f"{name}.{key}", base)
        elif default is _REQUIRED:
            raise ConfigError(f"missing required key {name}.{key}")
        else:
            out[key] = default
    return out


def _sinks(raw, base: Path) -> SinksConfig:
    if not isinstance(raw, dict):
        raise ConfigError("sinks: must be a table")
    for key in raw:
        if key != "telegram":
            raise ConfigError(f"unknown key sinks.{key}")
    if "telegram" not in raw:
        return SinksConfig()
    return SinksConfig(telegram=TelegramConfig(**_table(raw, "sinks.telegram", base)))


def load_config(path: Path) -> Config:
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except OSError as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    except ValueError as e:  # covers TOMLDecodeError and UnicodeDecodeError
        raise ConfigError(f"invalid TOML in {path}: {e}") from e
    for key in data:
        if key not in ("watch", "render", "output", "sinks"):
            raise ConfigError(f"unknown key {key} in {path}")
    base = path.absolute().parent
    return Config(
        watch=WatchConfig(**_table(data, "watch", base)),
        render=RenderConfig(**_table(data, "render", base)),
        output=OutputConfig(**_table(data, "output", base)),
        sinks=_sinks(data["sinks"], base) if "sinks" in data else SinksConfig(),
    )
