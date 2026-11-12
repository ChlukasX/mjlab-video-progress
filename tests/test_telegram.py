"""Contract tests for telegram: read_token, deliver_telegram, send_text,
get_chats and urlopen_post.
No real network."""

from __future__ import annotations

import email
import email.policy
import json
import socket
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs

import pytest

from mjlab_video_progress import telegram
from mjlab_video_progress.config import TelegramConfig
from mjlab_video_progress.exceptions import ConfigError, SinkError
from mjlab_video_progress.sinks import Delivery
from mjlab_video_progress.telegram import (
    deliver_telegram,
    get_chats,
    read_token,
    send_text,
    urlopen_post,
)

TOKEN = "123456:SECRET-token_value"
VIDEO = b"\x00\xffvideo\r\nbytes\x80"
RUN_DIR = Path("/logs/rsl_rl/exp/2026-11-09_12-00-00")
OK = json.dumps({"ok": True, "result": {"message_id": 1}}).encode()


# --- read_token ---------------------------------------------------------------------------------


def token_file(tmp_path, text=TOKEN + "\n", mode=0o600):
    path = tmp_path / "token"
    path.write_text(text)
    path.chmod(mode)
    return path


@pytest.mark.parametrize("mode", [0o600, 0o400, 0o700])
def test_token_file_is_read_and_stripped(tmp_path, mode):
    cfg = TelegramConfig(chat_id=1, token_file=token_file(tmp_path, f"  {TOKEN}\n", mode))
    assert read_token(cfg, {}) == TOKEN


@pytest.mark.parametrize("mode", [0o640, 0o644, 0o604, 0o660, 0o666])
def test_token_file_readable_by_others_is_refused(tmp_path, mode):
    path = token_file(tmp_path, mode=mode)
    with pytest.raises(ConfigError, match="chmod 600") as e:
        read_token(TelegramConfig(chat_id=1, token_file=path), {})
    assert str(path) in str(e.value) and TOKEN not in str(e.value)


@pytest.mark.parametrize("text", ["", "  \n"])
def test_empty_token_file_is_refused(tmp_path, text):
    cfg = TelegramConfig(chat_id=1, token_file=token_file(tmp_path, text))
    with pytest.raises(ConfigError, match="empty"):
        read_token(cfg, {})


def test_missing_token_file_and_directory_are_refused(tmp_path):
    for path in (tmp_path / "gone", tmp_path):
        with pytest.raises(ConfigError, match=str(path)):
            read_token(TelegramConfig(chat_id=1, token_file=path), {})


def test_token_comes_from_the_environment_when_there_is_no_file():
    assert read_token(TelegramConfig(chat_id=1), {"TELEGRAM_BOT_TOKEN": f" {TOKEN}\n"}) == TOKEN
    cfg = TelegramConfig(chat_id=1, token_env="MY_BOT")
    assert read_token(cfg, {"MY_BOT": TOKEN, "TELEGRAM_BOT_TOKEN": "other"}) == TOKEN


def test_environment_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    assert read_token(TelegramConfig(chat_id=1)) == TOKEN


@pytest.mark.parametrize("environ", [{}, {"TELEGRAM_BOT_TOKEN": ""}, {"TELEGRAM_BOT_TOKEN": " "}])
def test_unset_or_empty_environment_variable_is_refused(environ):
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        read_token(TelegramConfig(chat_id=1), environ)


def test_token_file_wins_over_the_environment(tmp_path):
    cfg = TelegramConfig(chat_id=1, token_file=token_file(tmp_path))
    assert read_token(cfg, {"TELEGRAM_BOT_TOKEN": "from-env"}) == TOKEN


def test_unusable_token_file_does_not_fall_back_to_the_environment(tmp_path):
    cfg = TelegramConfig(chat_id=1, token_file=tmp_path / "gone")
    with pytest.raises(ConfigError):
        read_token(cfg, {"TELEGRAM_BOT_TOKEN": TOKEN})


# --- deliver_telegram ---------------------------------------------------------------------------


class FakePost:
    """The injected transport: records calls, then answers or raises."""

    def __init__(self, status=200, body=OK, raises=None):
        self.status, self.body, self.raises = status, body, raises
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, body))
        if self.raises:
            raise self.raises(url)
        return self.status, self.body


def api_error(code, description, **parameters):
    body = {"ok": False, "error_code": code, "description": description}
    if parameters:
        body["parameters"] = parameters
    return json.dumps(body).encode()


def parse_request(headers, body):
    """The multipart form the request carries, as {field name: email part}."""
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    raw = f"Content-Type: {headers['Content-Type']}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    msg = email.message_from_bytes(raw, policy=email.policy.HTTP)
    return {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}


def text(part):
    return part.get_payload(decode=True).decode()


@pytest.fixture
def delivery(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(VIDEO)
    return Delivery(video=video, task="Task-X", run_dir=RUN_DIR, iteration=150)


def send(delivery, post=None, chat_id=123):
    post = post or FakePost()
    deliver_telegram(TelegramConfig(chat_id=chat_id), TOKEN, delivery, post)
    return post


def test_one_send_video_request_with_the_form_fields(delivery):
    post = send(delivery)
    assert len(post.calls) == 1
    url, headers, body = post.calls[0]
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendVideo"
    parts = parse_request(headers, body)
    assert sorted(parts) == ["caption", "chat_id", "supports_streaming", "video"]
    assert text(parts["chat_id"]) == "123"
    assert text(parts["supports_streaming"]) == "true"
    assert text(parts["caption"]) == "Task-X\nexp/2026-11-09_12-00-00\niteration 150"
    video = parts["video"]
    assert video.get_filename() == "video.mp4" and video.get_content_type() == "video/mp4"
    assert video.get_payload(decode=True) == VIDEO


@pytest.mark.parametrize(
    ("chat_id", "sent"), [(-1001234567890, "-1001234567890"), ("@chan", "@chan")]
)
def test_chat_id_is_sent_as_a_string(delivery, chat_id, sent):
    post = send(delivery, chat_id=chat_id)
    assert text(parse_request(*post.calls[0][1:])["chat_id"]) == sent


def test_caption_is_utf8_and_cut_to_1024_characters(delivery):
    delivery = Delivery(delivery.video, "Tâsk-" + "x" * 2000, RUN_DIR, 150)
    post = send(delivery)
    caption = text(parse_request(*post.calls[0][1:])["caption"])
    assert caption.startswith("Tâsk-x") and len(caption) == 1024


def test_ok_answer_returns_none(delivery):
    assert deliver_telegram(TelegramConfig(chat_id=1), TOKEN, delivery, FakePost()) is None


def test_error_answer_raises_with_code_and_description(delivery):
    post = FakePost(400, api_error(400, "Bad Request: chat not found"))
    with pytest.raises(SinkError, match="400") as e:
        send(delivery, post)
    assert "chat not found" in str(e.value)


def test_flood_control_answer_names_retry_after(delivery):
    post = FakePost(429, api_error(429, "Too Many Requests: retry after 17", retry_after=17))
    with pytest.raises(SinkError, match="retry after 17"):
        send(delivery, post)


@pytest.mark.parametrize(
    ("status", "body"),
    [(502, b"<html>Bad Gateway</html>"), (200, b"[]"), (200, b'{"result": 1}'), (200, b"")],
)
def test_unexpected_answer_raises(delivery, status, body):
    with pytest.raises(SinkError, match=rf"unexpected response.*{status}"):
        send(delivery, FakePost(status, body))


@pytest.mark.parametrize("error", [URLError, TimeoutError, ConnectionResetError])
def test_transport_failure_raises_without_the_token_anywhere(delivery, error):
    post = FakePost(raises=lambda url: error(f"cannot reach {url}"))
    with pytest.raises(SinkError) as e:
        send(delivery, post)
    assert TOKEN not in str(e.value)
    assert TOKEN not in "".join(traceback.format_exception(e.value))  # the cause is not chained


def test_token_echoed_by_telegram_is_redacted(delivery):
    post = FakePost(401, api_error(401, f"Unauthorized: {TOKEN}"))
    with pytest.raises(SinkError, match="Unauthorized") as e:
        send(delivery, post)
    assert TOKEN not in str(e.value)


def test_oversize_video_raises_before_any_request(delivery, monkeypatch):
    monkeypatch.setattr(telegram, "MAX_VIDEO_BYTES", len(VIDEO) - 1)
    post = FakePost()
    with pytest.raises(SinkError, match="over Telegram's limit"):
        send(delivery, post)
    assert post.calls == []


def test_missing_video_raises_oserror_before_any_request(delivery):
    delivery.video.unlink()
    post = FakePost()
    with pytest.raises(FileNotFoundError):
        send(delivery, post)
    assert post.calls == []


# --- send_text ----------------------------------------------------------------------------------


def test_send_text_is_one_send_message_request():
    post = FakePost()
    send_text(TelegramConfig(chat_id=-100), TOKEN, "hi & bye", post)
    [(url, headers, body)] = post.calls
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert headers == {"Content-Type": "application/x-www-form-urlencoded"}
    assert parse_qs(body.decode(), strict_parsing=True) == {
        "chat_id": ["-100"],
        "text": ["hi & bye"],
    }


@pytest.mark.parametrize(
    "post",
    [FakePost(400, api_error(400, f"chat not found for {TOKEN}")), FakePost(raises=URLError)],
)
def test_send_text_failure_is_a_sink_error_without_the_token(post):
    with pytest.raises(SinkError) as e:
        send_text(TelegramConfig(chat_id=1), TOKEN, "hi", post)
    assert TOKEN not in str(e.value)


# --- get_chats ----------------------------------------------------------------------------------


def updates(*items):
    return json.dumps({"ok": True, "result": list(items)}).encode()


def test_get_chats_collects_every_chat_from_get_updates():
    post = FakePost(
        body=updates(
            {
                "update_id": 1,
                "message": {"chat": {"id": 7, "first_name": "Ann", "username": "ann"}},
            },
            {"update_id": 2, "message": {"chat": {"id": 7, "username": "ann"}}},
            {"update_id": 3, "my_chat_member": {"chat": {"id": -100, "title": "Lab"}}},
            {"update_id": 4, "channel_post": {"chat": {"id": -200, "title": "Runs"}}},
            {"update_id": 5, "message": {"chat": {"id": 8}}},
            {"update_id": 6, "poll": {"id": "x"}},
        )
    )
    assert get_chats(TOKEN, post) == {7: "ann", -100: "Lab", -200: "Runs", 8: ""}
    [(url, _, body)] = post.calls
    assert url == f"https://api.telegram.org/bot{TOKEN}/getUpdates" and body == b""


def test_get_chats_with_no_updates_is_empty():
    assert get_chats(TOKEN, FakePost(body=updates())) == {}


@pytest.mark.parametrize(
    "post",
    [
        FakePost(body=json.dumps({"ok": True, "result": {}}).encode()),
        FakePost(409, api_error(409, "Conflict: can't use getUpdates while webhook is active")),
    ],
)
def test_get_chats_failure_is_a_sink_error(post):
    with pytest.raises(SinkError):
        get_chats(TOKEN, post)


# --- urlopen_post, against a loopback server -----------------------------------------------------


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    seen = []

    class Handler(BaseHTTPRequestHandler):
        status, reply = 200, OK

        def do_POST(self):
            size = int(self.headers["Content-Length"])
            seen.append((self.path, dict(self.headers), self.rfile.read(size)))
            self.send_response(self.status)
            self.send_header("Content-Length", str(len(self.reply)))
            self.end_headers()
            self.wfile.write(self.reply)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", Handler, seen
    httpd.shutdown()
    httpd.server_close()


def test_urlopen_post_sends_the_request_and_returns_status_and_body(server):
    base, _, seen = server
    result = urlopen_post(f"{base}/botX/sendVideo", {"Content-Type": "x/y"}, b"payload")
    assert result == (200, OK)
    path, headers, body = seen[0]
    assert (path, headers["Content-Type"], body) == ("/botX/sendVideo", "x/y", b"payload")


def test_urlopen_post_returns_an_http_error_status_instead_of_raising(server):
    base, handler, _ = server
    handler.status, handler.reply = 400, api_error(400, "Bad Request")
    assert urlopen_post(f"{base}/x", {}, b"") == (400, handler.reply)


def test_urlopen_post_connection_failure_raises_oserror():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # closed again on exit: nothing listens there
    with pytest.raises(OSError):
        urlopen_post(f"http://127.0.0.1:{port}/x", {}, b"")
