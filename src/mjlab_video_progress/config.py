"""Config: load and validate the TOML config file.

Contract:

- ``load_config(path)`` returns a frozen ``Config`` or raises ``ConfigError``.
- A missing file or a TOML syntax error raises ``ConfigError`` naming the file.
- Unknown keys, wrong types, missing required keys and non-positive integers raise
  ``ConfigError`` naming the full key path (for example ``render.timeout_seconds``).
- Integers must be ``int`` (not ``bool``) and strictly positive.
- Empty strings are rejected (``watch.task``, paths, ``render.command`` elements), except
  inside ``render.extra_args``.
- Relative paths resolve against the config file's directory; absolute paths are unchanged.
- Loading never touches the filesystem under ``watch.log_root`` (hard rule 1).

Tables: ``[watch]``, ``[render]`` and ``[output]``. Any other key is rejected.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """The config file is missing, malformed or invalid."""


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


@dataclass(frozen=True)
class OutputConfig:
    dir: Path


@dataclass(frozen=True)
class Config:
    watch: WatchConfig
    render: RenderConfig
    output: OutputConfig


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
    return base / _str(value, key, base)  # joining an absolute path keeps it unchanged


def _strs(value, key, base):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{key}: must be a list of strings, got {value!r}")
    return tuple(value)


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
    },
    "output": {"dir": (_path, _REQUIRED)},
}


def _table(data: dict, name: str, base: Path) -> dict:
    raw = data.get(name, {})
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
        if key not in _SPEC:
            raise ConfigError(f"unknown key {key} in {path}")
    base = path.absolute().parent
    return Config(
        watch=WatchConfig(**_table(data, "watch", base)),
        render=RenderConfig(**_table(data, "render", base)),
        output=OutputConfig(**_table(data, "output", base)),
    )
