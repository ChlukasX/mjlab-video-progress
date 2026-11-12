"""Contract tests for config.py: load_config(path) -> frozen Config, or ConfigError."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from helpers import snapshot

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
        (
            "timeout_seconds = 600",
            "timeout_seconds = 600\nmin_free_gpu_mib = 0",
            "render.min_free_gpu_mib",
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
    assert cfg.render.min_free_gpu_mib == 2048
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

    before = snapshot(tmp_path / "logs")
    load_config(write(tmp_path, sub("every = 500", 'every = 500\nlog_root = "logs"')))
    assert snapshot(tmp_path / "logs") == before


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


def test_leading_tilde_expands_to_home_in_every_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    text = sub("every = 500", 'every = 500\nlog_root = "~/logs/rsl_rl"')
    text = text.replace("timeout_seconds = 600", 'timeout_seconds = 600\ncwd = "~"')
    text = text.replace('dir = "progress"', 'dir = "~/progress"')
    cfg = load_config(write(tmp_path, text))
    assert cfg.watch.log_root == tmp_path / "home/logs/rsl_rl"
    assert cfg.render.cwd == tmp_path / "home"
    assert cfg.output.dir == tmp_path / "home/progress"


def test_tilde_not_at_start_is_literal(tmp_path):
    cfg = load_config(write(tmp_path, sub('dir = "progress"', 'dir = "out/~/x"')))
    assert cfg.output.dir == tmp_path / "out/~/x"


def test_tilde_of_unknown_user_names_key(tmp_path):
    text = sub('dir = "progress"', 'dir = "~no_such_user_mjvp/x"')
    with pytest.raises(ConfigError, match=r"output\.dir"):
        load_config(write(tmp_path, text))


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


def test_optional_string_values_are_loaded(tmp_path):
    text = sub("every = 500", 'every = 500\ntask = "Task-A"')
    text = text.replace("timeout_seconds = 600", 'timeout_seconds = 600\nextra_args = ["--a", "b"]')
    cfg = load_config(write(tmp_path, text))
    assert cfg.watch.task == "Task-A"
    assert cfg.render.extra_args == ("--a", "b")


@pytest.mark.parametrize("bad", ["[1]", '"--a"', '["--a", 2]'])
def test_extra_args_must_be_list_of_strings(tmp_path, bad):
    text = sub("timeout_seconds = 600", f"timeout_seconds = 600\nextra_args = {bad}")
    with pytest.raises(ConfigError, match=r"render\.extra_args"):
        load_config(write(tmp_path, text))


def test_table_that_is_not_a_table_is_rejected(tmp_path):
    text = "watch = 3\n" + sub("[watch]\nevery = 500\n\n", "")
    with pytest.raises(ConfigError, match="watch"):
        load_config(write(tmp_path, text))


def test_non_utf8_file_is_a_config_error(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ConfigError, match="cfg.toml"):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ('dir = "progress"', 'dir = ""', "output.dir"),
        ('command = ["uv", "run", "play"]', 'command = [""]', "render.command"),
        ("every = 500", 'every = 500\ntask = ""', "watch.task"),
        ("every = 500", 'every = 500\nlog_root = ""', "watch.log_root"),
        ("timeout_seconds = 600", 'timeout_seconds = 600\ncwd = ""', "render.cwd"),
    ],
)
def test_empty_strings_rejected(tmp_path, old, new, key):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(write(tmp_path, sub(old, new)))


# --- [sinks.telegram] ---------------------------------------------------------------------------

TELEGRAM = "\n[sinks.telegram]\nchat_id = 123\n"


def load_telegram(tmp_path, text):
    return load_config(write(tmp_path, MINIMAL + text)).sinks.telegram


def test_no_sinks_table_means_no_telegram(tmp_path):
    assert load_config(write(tmp_path)).sinks.telegram is None


def test_telegram_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + TELEGRAM))
    assert cfg.sinks.telegram.chat_id == 123
    assert cfg.sinks.telegram.token_file is None
    assert cfg.sinks.telegram.token_env == "TELEGRAM_BOT_TOKEN"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.sinks.telegram.chat_id = 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-1001234567890", -1001234567890),  # supergroups and channels are negative
        ('"@my_channel"', "@my_channel"),
        ('"123456"', "123456"),  # a string stays a string
    ],
)
def test_telegram_chat_id_forms(tmp_path, value, expected):
    chat_id = load_telegram(tmp_path, f"\n[sinks.telegram]\nchat_id = {value}\n").chat_id
    assert chat_id == expected and type(chat_id) is type(expected)


@pytest.mark.parametrize("bad", ["true", '""', "1.5", "[1]"])
def test_telegram_chat_id_wrong_type_names_key(tmp_path, bad):
    with pytest.raises(ConfigError, match=r"sinks\.telegram\.chat_id"):
        load_telegram(tmp_path, f"\n[sinks.telegram]\nchat_id = {bad}\n")


def test_telegram_chat_id_is_required(tmp_path):
    with pytest.raises(ConfigError, match=r"missing required key sinks\.telegram\.chat_id"):
        load_telegram(tmp_path, "\n[sinks.telegram]\n")


def test_telegram_token_env_is_loaded_and_checked(tmp_path):
    assert load_telegram(tmp_path, TELEGRAM + 'token_env = "MY_BOT"\n').token_env == "MY_BOT"
    for bad in ['""', "3"]:
        with pytest.raises(ConfigError, match=r"sinks\.telegram\.token_env"):
            load_telegram(tmp_path, TELEGRAM + f"token_env = {bad}\n")


def test_telegram_token_file_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cases = [
        ('"secrets/token"', tmp_path / "secrets/token"),  # relative: against the config file
        ('"/abs/token"', Path("/abs/token")),
        ('"~/.config/mvp/token"', tmp_path / "home/.config/mvp/token"),
    ]
    for value, expected in cases:
        got = load_telegram(tmp_path, TELEGRAM + f"token_file = {value}\n").token_file
        assert got == expected


@pytest.mark.parametrize("bad", ['""', "3"])
def test_telegram_token_file_wrong_type_names_key(tmp_path, bad):
    with pytest.raises(ConfigError, match=r"sinks\.telegram\.token_file"):
        load_telegram(tmp_path, TELEGRAM + f"token_file = {bad}\n")


def test_telegram_accepts_both_token_sources(tmp_path):
    got = load_telegram(tmp_path, TELEGRAM + 'token_file = "t"\ntoken_env = "X"\n')
    assert (got.token_file, got.token_env) == (tmp_path / "t", "X")


def test_loading_does_not_read_the_token_file_or_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    got = load_telegram(tmp_path, TELEGRAM + 'token_file = "no/such/file"\n')
    assert got.token_file == tmp_path / "no/such/file"
    assert not (tmp_path / "no").exists()


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (TELEGRAM + "bogus = 1\n", r"sinks\.telegram\.bogus"),
        ("\n[sinks.docs]\nx = 1\n", r"sinks\.docs"),  # added together with that sink
        ('\n[sinks.local]\ndest = "d"\n', r"sinks\.local"),
    ],
)
def test_unknown_sink_keys_name_full_path(tmp_path, text, key):
    with pytest.raises(ConfigError, match=key):
        load_telegram(tmp_path, text)


def test_sinks_that_is_not_a_table_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match=r"sinks: must be a table"):
        load_config(write(tmp_path, "sinks = 5\n" + MINIMAL))
    with pytest.raises(ConfigError, match=r"sinks\.telegram: must be a table"):
        load_config(write(tmp_path, MINIMAL + "\n[sinks]\ntelegram = 5\n"))
