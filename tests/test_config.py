"""Contract tests for config.py: load_config(path) -> frozen Config, or ConfigError."""

from __future__ import annotations

import dataclasses

import pytest

from mjlab_video_progress.config import ConfigError, load_config

MINIMAL = """\
[watch]
every = 500

[render]
command = ["uv", "run", "play"]
timeout_seconds = 600

[output]
dir = "progress"
"""


def write(tmp_path, text=MINIMAL, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def sub(old, new):
    assert old in MINIMAL
    return MINIMAL.replace(old, new)


def test_minimal_loads_typed_and_immutable(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.watch.every == 500
    assert cfg.render.command == ("uv", "run", "play")
    assert cfg.render.timeout_seconds == 600
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.render.num_envs = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("old", "key"),
    [
        ("[watch]\nevery = 500\n\n", "watch.every"),  # whole table absent
        ("every = 500\n", "watch.every"),
        ('command = ["uv", "run", "play"]\n', "render.command"),
        ("timeout_seconds = 600\n", "render.timeout_seconds"),
        ('dir = "progress"\n', "output.dir"),
    ],
)
def test_missing_required_key_is_named(tmp_path, old, key):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(write(tmp_path, sub(old, "")))


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ("every = 500", 'every = "500"', "watch.every"),
        ("every = 500", "every = true", "watch.every"),  # bool is not an int here
        ("every = 500", "every = 500\ntask = 3", "watch.task"),
        ("every = 500", "every = 500\nlog_root = 3", "watch.log_root"),
        ("timeout_seconds = 600", "timeout_seconds = 1.5", "render.timeout_seconds"),
    ],
)
def test_wrong_type_names_key(tmp_path, old, new, key):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(write(tmp_path, sub(old, new)))


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ("every = 500", "every = 0", "watch.every"),
        ("every = 500", "every = -5", "watch.every"),
        ("every = 500", "every = 500\npoll_seconds = 0", "watch.poll_seconds"),
        ("timeout_seconds = 600", "timeout_seconds = 0", "render.timeout_seconds"),
        ("timeout_seconds = 600", "timeout_seconds = 600\nnum_envs = 0", "render.num_envs"),
        (
            "timeout_seconds = 600",
            "timeout_seconds = 600\nvideo_length = 0",
            "render.video_length",
        ),
    ],
)
def test_non_positive_integers_rejected(tmp_path, old, new, key):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(write(tmp_path, sub(old, new)))


def test_defaults_apply(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.watch.poll_seconds == 60
    assert cfg.render.num_envs == 1
    assert cfg.render.video_length == 200
    assert cfg.render.extra_args == ()
    # None means "use what discovery finds"; Config must not invent a value.
    assert cfg.render.cwd is None
    assert cfg.watch.log_root is None
    assert cfg.watch.task is None


@pytest.mark.parametrize("bad", ['"uv run play"', "[]", '["uv", 3]'])
def test_command_must_be_nonempty_list_of_strings(tmp_path, bad):
    text = sub('command = ["uv", "run", "play"]', f"command = {bad}")
    with pytest.raises(ConfigError, match=r"render\.command"):
        load_config(write(tmp_path, text))


def test_load_does_not_touch_log_root(tmp_path):
    ckpt = tmp_path / "logs" / "exp" / "run" / "model_500.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    def snapshot():
        return [
            (p, p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
            for p in sorted(tmp_path.glob("logs/**/*"))
        ]

    before = snapshot()
    load_config(write(tmp_path, sub("every = 500", 'every = 500\nlog_root = "logs"')))
    assert snapshot() == before


def test_missing_file_and_syntax_error(tmp_path):
    with pytest.raises(ConfigError, match="nope.toml"):
        load_config(tmp_path / "nope.toml")
    with pytest.raises(ConfigError, match="cfg.toml"):
        load_config(write(tmp_path, "[render\n"))


def test_relative_paths_resolve_against_config_file(tmp_path, monkeypatch):
    conf = tmp_path / "conf"
    conf.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    text = sub("every = 500", 'every = 500\nlog_root = "logs"')
    text = text.replace("timeout_seconds = 600", 'timeout_seconds = 600\ncwd = "repo"')
    cfg = load_config(write(conf, text))
    assert cfg.watch.log_root == conf / "logs"
    assert cfg.render.cwd == conf / "repo"
    assert cfg.output.dir == conf / "progress"


def test_absolute_paths_are_kept(tmp_path):
    cfg = load_config(write(tmp_path, sub('dir = "progress"', 'dir = "/abs/out"')))
    assert str(cfg.output.dir) == "/abs/out"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("bogus = 1\n" + MINIMAL, "bogus"),
        (sub("every = 500", "every = 500\nbogus = 1"), r"watch\.bogus"),
        (sub("timeout_seconds = 600", "timeout_seconds = 600\nbogus = 1"), r"render\.bogus"),
        (sub('dir = "progress"', 'dir = "progress"\nbogus = 1'), r"output\.bogus"),
        (MINIMAL + '\n[[runs]]\nname = "a"\n', "runs"),  # removed schema fails loudly
    ],
)
def test_unknown_key_names_full_path(tmp_path, text, key):
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, text))
