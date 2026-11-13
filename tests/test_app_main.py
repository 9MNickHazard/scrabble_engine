"""`scrabble_ai.app.__main__`, the launcher: finding Edge, finding an app that is already running,
and drawing the icon. Nothing here opens a window or touches the Desktop or data/app/."""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from scrabble_ai.app import __main__ as launcher
from scrabble_ai.app.server import AppState, create_server

pytest.register_assert_rewrite("app_fixtures")
import app_fixtures  # noqa: E402,F401

pytest_plugins = ["app_fixtures"]


def test_find_edge_returns_a_path_or_none():
    edge = launcher.find_edge()
    assert edge is None or (isinstance(edge, str) and Path(edge).exists() and edge.lower().endswith("msedge.exe"))


def test_find_edge_without_edge(monkeypatch, tmp_path):
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "x86"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "x64"))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    assert launcher.find_edge() is None


@pytest.fixture
def server_file(monkeypatch, tmp_path):
    path = tmp_path / "server.json"
    monkeypatch.setattr(launcher, "SERVER_FILE", path)
    return path


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve(server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    return thread


def stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(10)


def test_running_instance_without_a_server_file(server_file):
    assert launcher.running_instance() is None


@pytest.mark.parametrize("content", ["", "{not json", '{"port": 80', "{}", '{"pid": 1}'])
def test_running_instance_with_a_broken_server_file(server_file, content):
    server_file.write_text(content, encoding="utf-8")
    assert launcher.running_instance() is None


@pytest.mark.parametrize("content", ['{"port": null}', '{"port": "abc"}', "[]", "null", '"text"', '{"port": [1]}',
                                     b"\xff\xfe\x00garbage"])
def test_running_instance_with_a_mangled_server_file(server_file, content):
    """A non-numeric port once raised http.client.InvalidURL (not a ValueError) and a non-object file a
    TypeError, killing the launcher at start-up."""
    if isinstance(content, bytes):
        server_file.write_bytes(content)
    else:
        server_file.write_text(content, encoding="utf-8")
    assert launcher.running_instance() is None


def test_running_instance_with_a_stale_server_file(server_file):
    server_file.write_text(json.dumps({"port": free_port(), "pid": 1, "started": 0}), encoding="utf-8")
    assert launcher.running_instance() is None


def test_running_instance_ignores_another_program_on_the_port(server_file):
    class NotTheApp(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_error(404)

        def log_message(self, *args):
            pass

    other = ThreadingHTTPServer(("127.0.0.1", 0), NotTheApp)
    thread = serve(other)
    try:
        server_file.write_text(json.dumps({"port": other.server_address[1]}), encoding="utf-8")
        assert launcher.running_instance() is None
    finally:
        stop(other, thread)


def test_running_instance_finds_the_app(server_file, trained, tmp_path):
    state = AppState(tmp_path / "runs", tmp_path / "app", trained.loaded)
    state.players._device = torch.device("cpu")
    state.gpu = lambda: None
    app = create_server(0, state=state)
    thread = serve(app)
    try:
        port = app.server_address[1]
        server_file.write_text(json.dumps({"port": port, "pid": 1, "started": 0}), encoding="utf-8")
        assert launcher.running_instance() == f"http://127.0.0.1:{port}"
    finally:
        stop(app, thread)


def test_make_icon(tmp_path):
    from PIL import Image

    path = tmp_path / "icons" / "scrabble-lab.ico"
    assert launcher.make_icon(path) is True
    with Image.open(path) as icon:
        assert icon.format == "ICO"
        assert {(256, 256), (48, 48), (32, 32), (16, 16)} <= set(icon.info["sizes"])
        icon.size = (16, 16)
        icon.load()
        assert icon.getbbox() is not None, "something is drawn"


def test_make_icon_without_pillow(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)  # makes `from PIL import ...` raise ImportError
    path = tmp_path / "icon.ico"
    assert launcher.make_icon(path) is False
    assert not path.exists()
