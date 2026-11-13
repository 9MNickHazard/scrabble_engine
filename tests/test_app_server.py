"""`scrabble_ai.app.server`: the HTTP API of docs/APP_API.md and the static files, served on a free
port by `create_server` with an `AppState` on temp folders (nothing is started detached)."""

from __future__ import annotations

import dataclasses
import http.client
import json
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import scrabble_ai
from scrabble_ai.app import server as server_module
from scrabble_ai.app.matches import COLUMNS
from scrabble_ai.app.server import STATIC_DIR, AppState, create_server

pytest.register_assert_rewrite("app_fixtures")
from app_fixtures import DEAD_PID, TINY_RUN, copy_run, make_run, record, write_records  # noqa: E402

pytest_plugins = ["app_fixtures"]

SUMMARY_KEYS = {"name", "status", "phase", "network", "parameters", "games", "step", "positions", "hours",
                "games_per_second", "created", "updated", "limit", "progress", "latest_eval", "best_step", "has_best",
                "checkpoints", "changed", "error"}
STATE_KEYS = {"id", "seed", "players", "advisor", "board", "last_placed", "racks", "scores", "to_move", "turn", "bag",
              "scoreless_turns", "can_exchange", "unseen", "unseen_by_seat", "over", "result", "history"}
CANDIDATE_KEYS = {"text", "kind", "word", "position", "score", "tiles", "leave", "exchanged", "leave_value", "value",
                  "total", "win", "chosen"}
HIGH_GPU = {"name": "NVIDIA GeForce RTX 5070 Ti", "utilization": 87, "memory_used_mb": 15565, "memory_total_mb": 16303,
            "temperature_c": 61}


@pytest.fixture
def app(tmp_path, trained, spawned, fast_scorer):
    runs_dir, app_dir = tmp_path / "runs", tmp_path / "app"
    runs_dir.mkdir()
    app_dir.mkdir()
    copy_run(trained, runs_dir)
    state = AppState(runs_dir, app_dir, trained.loaded)
    state.players._device = torch.device("cpu")
    state.gpu = lambda: None  # no nvidia-smi from the tests
    server = create_server(0, state=state, log_path=app_dir / "app.log")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(state=state, server=server, port=server.server_address[1], runs_dir=runs_dir,
                              app_dir=app_dir, spawned=spawned, trained=trained)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)
        assert not thread.is_alive()


def request(app, method: str, path: str, body=None, raw: bytes | None = None, headers: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", app.port, timeout=60)
    try:
        data = raw if raw is not None else None if body is None else json.dumps(body).encode("utf-8")
        all_headers = dict(headers or {})
        if data is not None:
            all_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=data, headers=all_headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type") or "", response.read()
    finally:
        connection.close()


def strict_json(content: bytes):
    """JSON as a browser parses it: no NaN or Infinity."""
    def refuse(constant):
        raise ValueError(f"{constant} is not JSON")
    return json.loads(content, parse_constant=refuse)


def api(app, method: str, path: str, body=None, raw: bytes | None = None):
    status, content_type, content = request(app, method, path, body, raw)
    assert content_type == "application/json; charset=utf-8", (path, content_type)
    data = strict_json(content)
    if status >= 400:
        assert set(data) == {"error"} and isinstance(data["error"], str) and data["error"], (path, data)
    return status, data


def ok(app, method: str, path: str, body=None):
    status, data = api(app, method, path, body)
    assert status == 200, (method, path, data)
    return data


# ---- status ------------------------------------------------------------------------------------


def test_status(app):
    data = ok(app, "GET", "/api/status")
    assert set(data) == {"version", "lexicon", "leaves", "gpu", "warnings", "active_runs", "queued_runs", "active_matches"}
    loaded = app.trained.loaded
    assert data["version"] == scrabble_ai.__version__
    assert data["lexicon"] == {"id": loaded.lexicon_id, "words": loaded.engine.word_count}
    assert data["leaves"] == {"ok": True, "file": loaded.leaves_path.name}
    assert (data["gpu"], data["warnings"], data["active_runs"], data["queued_runs"], data["active_matches"]) == (
        None, [], [], [], [])
    make_run(app.runs_dir, "night1", records=[record("session", time.time(), event="start")], pid=os.getpid())
    make_run(app.runs_dir, "night2", queued={"queue": "q", "overrides": {}, "created": time.time()})
    match_id = ok(app, "POST", "/api/matches", {"a": "bot:greedy", "b": "bot:random", "pairs": 3})["id"]
    data = ok(app, "GET", "/api/status")
    assert (data["active_runs"], data["queued_runs"], data["active_matches"]) == (["night1"], ["night2"], [match_id])


def test_the_gpu_memory_warning(app):
    app.state.gpu = lambda: dict(HIGH_GPU)
    data = ok(app, "GET", "/api/status")
    assert data["gpu"] == HIGH_GPU
    assert data["warnings"] == [{"kind": "gpu_memory", "message": "Other programs are using 15.2 of 16 GB of GPU memory. "
                                 "Training started now would be slow until they are closed."}]
    app.state.gpu = lambda: {**HIGH_GPU, "memory_used_mb": 2140}
    assert ok(app, "GET", "/api/status")["warnings"] == []
    app.state.gpu = lambda: {**HIGH_GPU, "memory_used_mb": int(0.6 * 16303)}
    assert ok(app, "GET", "/api/status")["warnings"] == [], "60% exactly is not over the line"
    # A run of our own is using the memory: no warning.
    app.state.gpu = lambda: dict(HIGH_GPU)
    make_run(app.runs_dir, "busy", records=[record("session", time.time(), event="start")], pid=os.getpid())
    data = ok(app, "GET", "/api/status")
    assert data["warnings"] == [] and data["active_runs"] == ["busy"] and data["gpu"] == HIGH_GPU


def test_gpu_reads_nvidia_smi_and_caches_it(monkeypatch, trained, tmp_path):
    state = AppState(tmp_path / "runs", tmp_path / "app", trained.loaded)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="NVIDIA GeForce RTX 5070 Ti, 87, 2140, 16303, 61\nSecond GPU, 1, 2, 3, 4\n")

    monkeypatch.setattr(server_module.subprocess, "run", run)
    assert state.gpu() == {"name": "NVIDIA GeForce RTX 5070 Ti", "utilization": 87, "memory_used_mb": 2140,
                           "memory_total_mb": 16303, "temperature_c": 61}
    state.gpu()
    assert len(calls) == 1 and calls[0][0] == "nvidia-smi", "refreshed at most every two seconds"
    for failure in (OSError("no nvidia-smi"), None):
        def broken(command, failure=failure, **kwargs):
            if failure:
                raise failure
            return SimpleNamespace(stdout="")
        monkeypatch.setattr(server_module.subprocess, "run", broken)
        state._gpu = (0.0, None)
        assert state.gpu() is None


# ---- configuration and runs --------------------------------------------------------------------


def test_config_schema(app):
    data = ok(app, "GET", "/api/config/schema")
    assert set(data) == {"sections", "presets", "speed"}
    assert [s["name"] for s in data["sections"]] == ["run", "model", "selfplay", "warmstart", "train"]


def test_runs_list_and_detail(app):
    runs = ok(app, "GET", "/api/runs")["runs"]
    assert [r["name"] for r in runs] == [TINY_RUN] and set(runs[0]) == SUMMARY_KEYS
    detail = ok(app, "GET", f"/api/runs/{TINY_RUN}")
    assert set(detail) == {"run", "config", "checkpoints", "best"}
    assert set(detail["config"]) == {"run", "model", "selfplay", "warmstart", "train"}
    assert {k: detail["best"][k] for k in ("file", "step", "model_id")} == {
        "file": "best.pt", "step": app.trained.best_step, "model_id": f"{TINY_RUN}/best.pt"}
    assert detail["best"]["size_bytes"] > 0 and detail["best"]["modified"] > 0
    for checkpoint in detail["checkpoints"]:
        assert set(checkpoint) == {"file", "step", "size_bytes", "modified", "model_id"}
    assert ok(app, "GET", f"/api/runs/{TINY_RUN}/") == detail, "a trailing slash is fine"
    for path in ("/api/runs/nope", "/api/runs/bad.name", "/api/runs/..", "/api/runs/%2e%2e", "/api/runs/nope/metrics",
                 "/api/runs/nope/log", "/api/runs/nope/report", "/api/runs/..%2F..%2Fx"):
        assert api(app, "GET", path)[0] == 404, path


def test_metrics_and_log_and_report(app):
    records = ok(app, "GET", f"/api/runs/{TINY_RUN}/metrics")
    assert set(records) == {"records", "next"} and records["next"] == len(records["records"]) > 0
    assert all({"kind", "time", "step", "games", "positions", "hours"} <= set(r) for r in records["records"])
    tail = ok(app, "GET", f"/api/runs/{TINY_RUN}/metrics?since={records['next'] - 2}")
    assert tail == {"records": records["records"][-2:], "next": records["next"]}
    assert ok(app, "GET", f"/api/runs/{TINY_RUN}/metrics?since={records['next']}") == {"records": [], "next": records["next"]}
    assert api(app, "GET", f"/api/runs/{TINY_RUN}/metrics?since=abc")[0] == 400

    (app.runs_dir / TINY_RUN / "console.log").write_text("".join(f"line {i}\n" for i in range(300)), encoding="utf-8")
    assert ok(app, "GET", f"/api/runs/{TINY_RUN}/log") == {"lines": [f"line {i}" for i in range(100, 300)]}
    assert ok(app, "GET", f"/api/runs/{TINY_RUN}/log?tail=3") == {"lines": ["line 297", "line 298", "line 299"]}
    assert api(app, "GET", f"/api/runs/{TINY_RUN}/log?tail=x")[0] == 400

    report = ok(app, "GET", f"/api/runs/{TINY_RUN}/report")
    assert {"headline", "summary", "events"} <= set(report)
    assert not any(e["title"] == "Run ended unexpectedly" for e in report["events"])


def test_the_report_of_a_run_that_died(app):
    make_run(app.runs_dir, "died", pid=DEAD_PID, console=["[17:09:24] step 10", "Traceback (most recent call last):",
                                                           "RuntimeError: CUDA out of memory", ""],
             records=[record("session", 1, event="start"), record("selfplay", 2, games=1234, hours=0.5)])
    event = ok(app, "GET", "/api/runs/died/report")["events"][-1]
    assert event == {"games": 1234, "hours": 0.5, "kind": "stop", "tone": "warning", "title": "Run ended unexpectedly",
                     "detail": "RuntimeError: CUDA out of memory. It had not saved a checkpoint; Resume starts it again "
                               "from the beginning."}
    # With a checkpoint on disk, resuming continues from it.
    (app.runs_dir / "died" / "checkpoints").mkdir()
    (app.runs_dir / "died" / "checkpoints" / "step_000000010.pt").write_bytes(b"x")
    event = ok(app, "GET", "/api/runs/died/report")["events"][-1]
    assert event["detail"] == "RuntimeError: CUDA out of memory. Resume continues it from its last checkpoint."
    make_run(app.runs_dir, "silent", pid=DEAD_PID)
    event = ok(app, "GET", "/api/runs/silent/report")["events"][-1]
    assert event["title"] == "Run ended unexpectedly" and event["games"] == 0
    assert event["detail"] == ("The training process ended unexpectedly. It had not saved a checkpoint; Resume starts "
                               "it again from the beginning.")
    make_run(app.runs_dir, "asked", pid=DEAD_PID, console=["Is the GPU busy?"])
    assert ok(app, "GET", "/api/runs/asked/report")["events"][-1]["detail"].startswith("Is the GPU busy? It had")


def test_metrics_with_a_nan_loss_are_still_json(app):
    """A diverged loss is logged as NaN, which once went into the reply as-is: not JSON a browser accepts."""
    make_run(app.runs_dir, "diverged", records=[
        record("train", 1, step=1, loss=float("nan"), value_loss=float("inf"), win_loss=float("-inf"), learning_rate=1e-3),
        record("selfplay", 2, games=5, games_per_second=float("nan"), mean_score=400.5)])
    data = ok(app, "GET", "/api/runs/diverged/metrics")  # `ok` parses strictly: no NaN or Infinity
    train, selfplay = data["records"]
    assert (train["loss"], train["value_loss"], train["win_loss"], train["learning_rate"]) == (None, None, None, 1e-3)
    assert (selfplay["games_per_second"], selfplay["mean_score"]) == (None, 400.5)
    for path in ("/api/runs", "/api/runs/diverged", "/api/runs/diverged/report", "/api/status"):
        ok(app, "GET", path)


def test_start_runs(app, test_lexicon):
    result = ok(app, "POST", "/api/runs", {"mode": "parallel", "runs": [
        {"name": "night1", "overrides": {"run.hours": 8, "model.blocks": 6}},
        {"name": "night1-wide", "overrides": {"run.hours": 8, "model.blocks": 10, "model.channels": 128}}]})
    assert result == {"started": ["night1", "night1-wide"], "queued": []}
    assert [call[0][6] for call in app.spawned.calls] == ["night1", "night1-wide"]
    # Those two are running, so a queue waits behind them, its first run included.
    result = ok(app, "POST", "/api/runs", {"mode": "queue", "runs": [{"name": "q1"}, {"name": "q2"}]})
    assert result == {"started": [], "queued": ["q1", "q2"]}
    assert ok(app, "POST", "/api/runs", {"mode": "queue", "runs": [{"name": "q3"}]}) == {"started": [], "queued": ["q3"]}
    assert ok(app, "POST", "/api/runs", {"runs": [{"name": "default-mode"}]}) == {"started": ["default-mode"], "queued": []}
    statuses = {r["name"]: r["status"] for r in ok(app, "GET", "/api/runs")["runs"]}
    assert statuses["night1"] == "running" and {statuses[n] for n in ("q1", "q2", "q3")} == {"queued"}
    data = ok(app, "GET", "/api/status")
    assert set(data["queued_runs"]) == {"q1", "q2", "q3"} and "night1" in data["active_runs"]
    status, data = api(app, "POST", "/api/runs", {"runs": [{"name": "ok1"}, {"name": "bad", "overrides": {"model.blocks": 0}}]})
    assert status == 400 and data["error"] == "bad: model.blocks must be at least 1"


def test_start_runs_refusals(app, test_lexicon):
    cases = [
        ({"runs": [{"name": "bad name"}]}, "letters"),
        ({"runs": [{"name": TINY_RUN}]}, "already exists"),
        ({"runs": [{"name": "x", "overrides": {"model.nope": 1}}]}, "model.nope"),
        ({"runs": [{"name": "x", "overrides": {"model.blocks": "many"}}]}, "model.blocks"),
        ({"runs": [{"name": "x"}, {"name": "x"}]}, "same name"),
        ({"runs": [{"name": "x"}], "mode": "sideways"}, "mode"),
        ({"runs": "x"}, "list"),
        ({"runs": [{"overrides": {}}]}, "list"),
        ({"runs": [{"name": 5}]}, "list"),
        ({}, "list"),
        ({"runs": []}, "nothing"),
    ]
    for body, message in cases:
        status, data = api(app, "POST", "/api/runs", body)
        assert status == 400 and message in data["error"], (body, data)
    assert app.spawned.calls == []


def test_start_runs_needs_fitted_leave_values(app, test_lexicon):
    app.state.loaded = dataclasses.replace(app.state.loaded, leaves_path=None)
    status, data = api(app, "POST", "/api/runs", {"runs": [{"name": "x"}]})
    assert status == 400 and "fit-leaves" in data["error"] and app.spawned.calls == []
    assert ok(app, "GET", "/api/status")["leaves"] == {"ok": False, "file": None}


def test_stop_resume_delete(app):
    make_run(app.runs_dir, "live", records=[record("session", time.time(), event="start")], pid=os.getpid())
    assert ok(app, "POST", "/api/runs/live/stop") == {"ok": True}
    assert (app.runs_dir / "live" / "STOP").exists()
    assert ok(app, "GET", "/api/runs/live")["run"]["status"] == "stopping"
    assert api(app, "POST", "/api/runs/live/resume", {})[0] == 400
    status, data = api(app, "DELETE", "/api/runs/live")
    assert status == 400 and "stop it first" in data["error"]
    assert ok(app, "POST", "/api/runs/live/stop", {"force": True}) == {"ok": True}
    assert app.spawned.killed == [os.getpid()]
    assert ok(app, "GET", "/api/runs/live")["run"]["status"] == "stopped"
    status, data = api(app, "POST", "/api/runs/live/stop")
    assert status == 400 and "not running" in data["error"]

    summary = ok(app, "POST", f"/api/runs/{TINY_RUN}/resume", {"hours": 8, "max_games": 0,
                                                              "overrides": {"train.learning_rate": 0.0003}})
    assert set(summary) == SUMMARY_KEYS and summary["status"] == "running"
    command = app.spawned.calls[-1][0]
    assert command[-7:] == ["--set", "train.learning_rate=0.0003", "--hours", "8", "--set", "run.max_games=0",
                            "--resume", "latest"][-7:]
    assert api(app, "POST", f"/api/runs/{TINY_RUN}/resume")[0] == 400, "already running"

    make_run(app.runs_dir, "old", records=[record("session", 1, event="stop", finished=False)])
    assert ok(app, "DELETE", "/api/runs/old") == {"ok": True}
    assert not (app.runs_dir / "old").exists()
    for method, path in (("DELETE", "/api/runs/old"), ("POST", "/api/runs/old/stop"), ("POST", "/api/runs/old/resume"),
                         ("DELETE", "/api/runs/.."), ("DELETE", "/api/runs/%2e%2e"), ("DELETE", "/api/runs/..%5Cx")):
        assert api(app, method, path)[0] == 404, path
    assert app.runs_dir.is_dir() and (app.runs_dir / TINY_RUN).is_dir()


# ---- models ------------------------------------------------------------------------------------


def test_models_and_meta(app):
    data = ok(app, "GET", "/api/models")
    assert set(data) == {"bots", "models"}
    assert [b["spec"] for b in data["bots"]] == ["bot:greedy-leave", "bot:greedy", "bot:random"]
    assert f"{TINY_RUN}/best.pt" in {m["id"] for m in data["models"]}
    model = ok(app, "POST", f"/api/models/{TINY_RUN}/best.pt/meta", {"starred": True, "note": "first to beat the baseline"})
    assert model["id"] == f"{TINY_RUN}/best.pt" and model["starred"] is True and model["note"] == "first to beat the baseline"
    model = ok(app, "POST", f"/api/models/{TINY_RUN}%2Fbest.pt/meta", {"note": "encoded slash"})
    assert model["starred"] is True and model["note"] == "encoded slash"
    listed = {m["id"]: m for m in ok(app, "GET", "/api/models")["models"]}
    assert listed[f"{TINY_RUN}/best.pt"]["note"] == "encoded slash"
    for path, expected in ((f"/api/models/{TINY_RUN}/other.pt/meta", 400),
                           (f"/api/models/{TINY_RUN}/step_000999999.pt/meta", 404), ("/api/models/ghost/best.pt/meta", 404),
                           ("/api/models/..%2F..%2Fbest.pt/meta", 404), (f"/api/models/{TINY_RUN}/meta", 404)):
        assert api(app, "POST", path, {"starred": True})[0] == expected, path
    for body in ({"starred": "yes"}, {"starred": 1}, {"note": 5}, {"note": ["x"]}):
        status, data = api(app, "POST", f"/api/models/{TINY_RUN}/best.pt/meta", body)
        assert status == 400, body
    library = json.loads((app.app_dir / "library.json").read_text(encoding="utf-8"))
    assert set(library) == {f"{TINY_RUN}/best.pt"}, "refused requests wrote nothing"


def test_deleting_a_run_forgets_its_stars_and_notes(app):
    copy_run(app.trained, app.runs_dir, "doomed")
    ok(app, "POST", "/api/models/doomed/best.pt/meta", {"starred": True, "note": "gone soon"})
    ok(app, "POST", f"/api/models/{TINY_RUN}/best.pt/meta", {"note": "stays"})
    assert ok(app, "DELETE", "/api/runs/doomed") == {"ok": True}
    library = json.loads((app.app_dir / "library.json").read_text(encoding="utf-8"))
    assert set(library) == {f"{TINY_RUN}/best.pt"}
    copy_run(app.trained, app.runs_dir, "doomed")  # a new run of the same name starts unstarred
    model = next(m for m in ok(app, "GET", "/api/models")["models"] if m["id"] == "doomed/best.pt")
    assert model["starred"] is False and model["note"] == ""


# ---- matches -----------------------------------------------------------------------------------


def test_matches(app):
    match_id = ok(app, "POST", "/api/matches", {"a": f"model:{TINY_RUN}/best.pt", "b": "bot:greedy-leave", "pairs": 500,
                                                "candidates": 64})["id"]
    assert match_id.startswith("m_")
    command = app.spawned.calls[-1][0]
    assert command[-2:] == ["scrabble_ai.app.match_job", str(app.app_dir / "matches" / match_id)]
    matches = ok(app, "GET", "/api/matches")["matches"]
    assert [m["id"] for m in matches] == [match_id]
    assert {"id", "a", "b", "a_name", "b_name", "pairs", "status", "progress", "games_done", "created", "seconds",
            "spread", "spread_ci", "win_rate"} <= set(matches[0])
    assert matches[0]["a_name"] == f"{TINY_RUN} (best)" and matches[0]["status"] == "running"
    detail = ok(app, "GET", f"/api/matches/{match_id}")
    assert set(detail) == {"match", "summary", "games", "highlights"}
    assert detail["games"] == {name: [] for name in COLUMNS}
    assert ok(app, "POST", f"/api/matches/{match_id}/cancel") == {"ok": True}
    assert (app.app_dir / "matches" / match_id / "CANCEL").exists()
    for method, path in (("GET", "/api/matches/m_nope"), ("POST", "/api/matches/m_nope/cancel"),
                         ("GET", "/api/matches/..%2F..%2Fx")):
        assert api(app, method, path)[0] == 404, path
    for body in ({"a": "human", "b": "bot:greedy"}, {"a": "bot:greedy", "b": "bot:nope"}, {"a": "bot:greedy"},
                 {"a": "bot:greedy", "b": "bot:random", "pairs": 0}, {"a": "bot:greedy", "b": "bot:random", "pairs": "x"},
                 {"a": "bot:greedy", "b": "bot:random", "candidates": 1}):
        assert api(app, "POST", "/api/matches", body)[0] == 400, body


# ---- interactive games -------------------------------------------------------------------------


def test_games(app):
    state = ok(app, "POST", "/api/games", {"players": ["human", f"model:{TINY_RUN}/best.pt"], "seed": 12345})
    assert set(state) == STATE_KEYS and state["seed"] == 12345 and state["id"].startswith("g_")
    assert state["players"] == [{"spec": "human", "name": "You"},
                                {"spec": f"model:{TINY_RUN}/best.pt", "name": f"{TINY_RUN} (best)"}]
    game = state["id"]
    assert ok(app, "GET", f"/api/games/{game}") == state
    hint = ok(app, "GET", f"/api/games/{game}/hint?top=3")
    assert hint["advisor"] == f"{TINY_RUN} (best)" and len(hint["candidates"]) == 3
    assert all(set(c) == CANDIDATE_KEYS and c["chosen"] is False for c in hint["candidates"])
    assert len(ok(app, "GET", f"/api/games/{game}/hint")["candidates"]) <= 15
    status, data = api(app, "POST", f"/api/games/{game}/ai")
    assert status == 400 and "person" in data["error"]
    status, data = api(app, "POST", f"/api/games/{game}/move", {"tiles": [{"row": 15, "col": 7, "letter": "A"}]})
    assert status == 400 and "off the board" in data["error"]
    assert ok(app, "GET", f"/api/games/{game}") == state, "a refused move changes nothing"
    play = next(c for c in ok(app, "GET", f"/api/games/{game}/hint?top=50")["candidates"] if c["kind"] == "play")
    moved = ok(app, "POST", f"/api/games/{game}/move", {"tiles": [
        {"row": r, "col": c, "letter": letter, "blank": blank} for r, c, letter, blank in play["tiles"]]})
    assert moved["history"][0]["word"] == play["word"] and moved["to_move"] == 1
    response = ok(app, "POST", f"/api/games/{game}/ai")
    assert set(response) == {"state", "analysis"} and response["state"]["turn"] == 2
    assert set(response["analysis"]) == {"player", "turn", "considered", "candidates"}
    assert response["analysis"]["turn"] == 2
    assert sum(c["chosen"] for c in response["analysis"]["candidates"]) == 1
    assert [h["analysed"] for h in response["state"]["history"]] == [False, True]
    assert ok(app, "GET", f"/api/games/{game}/analysis?turn=2") == response["analysis"]
    for query, expected in (("turn=1", 404), ("turn=3", 404), ("turn=0", 400), ("turn=abc", 400), ("", 400),
                            ("turn=2.0", 400), ("turn=-2", 400), ("turn=10001", 400)):
        assert api(app, "GET", f"/api/games/{game}/analysis?{query}")[0] == expected, query
    state = response["state"]
    assert state["advisor"] == f"model:{TINY_RUN}/best.pt" and state["unseen"] == state["unseen_by_seat"][state["to_move"]]
    status, data = api(app, "POST", f"/api/games/{game}/move", {"exchange": "?????"})
    assert status == 400
    assert ok(app, "POST", f"/api/games/{game}/move", {"pass": True})["scoreless_turns"] == 1
    assert ok(app, "DELETE", f"/api/games/{game}") == {"ok": True}
    for method, path in (("GET", f"/api/games/{game}"), ("POST", f"/api/games/{game}/ai"),
                         ("GET", f"/api/games/{game}/hint"), ("POST", f"/api/games/{game}/move")):
        assert api(app, method, path, {} if method == "POST" else None)[0] == 404, path
    watch = ok(app, "POST", "/api/games", {"players": ["bot:greedy", "bot:random"]})
    assert 0 <= watch["seed"] < 2**48 and watch["players"][1] == {"spec": "bot:random", "name": "Random"}


def test_game_refusals(app):
    for body in ({"players": ["human"]}, {"players": "human"}, {}, {"players": ["human", "bot:nope"]},
                 {"players": ["human", "model:nope/best.pt"]}, {"players": ["human", "bot:greedy"], "advisor": "human"},
                 {"players": ["human", "bot:greedy"], "seed": "abc"}):
        assert api(app, "POST", "/api/games", body)[0] == 400, body
    game = ok(app, "POST", "/api/games", {"players": ["human", "bot:greedy"], "seed": 3})["id"]
    for body in ({}, {"tiles": [{"row": "x", "col": 7, "letter": "A"}]}, {"exchange": "Q1"}):
        assert api(app, "POST", f"/api/games/{game}/move", body)[0] == 400, body
    assert api(app, "GET", f"/api/games/{game}/hint?top=abc")[0] == 400


@pytest.mark.parametrize("method, path, body", [
    pytest.param("POST", "/api/games/{game}/move", {"tiles": [{"col": 7, "letter": "A"}]}, id="tile-without-row"),
    pytest.param("POST", "/api/games/{game}/move", {"tiles": [5]}, id="tile-not-an-object"),
    pytest.param("POST", "/api/games/{game}/move", {"tiles": "AB"}, id="tiles-not-a-list"),
    pytest.param("POST", "/api/games", {"players": ["human", "bot:greedy"], "seed": -1}, id="negative-seed"),
    pytest.param("POST", "/api/games", {"players": [1, 2]}, id="players-not-strings"),
    pytest.param("GET", "/api/leaves?rack=ERS&bag=-1", None, id="negative-bag"),
    pytest.param("POST", "/api/matches", {"a": "bot:greedy", "b": "bot:random", "pairs": None}, id="pairs-null"),
    pytest.param("POST", "/api/runs", {"runs": [{"name": "x", "overrides": [1]}]}, id="overrides-not-an-object"),
    pytest.param("POST", "/api/runs/tiny/resume", {"overrides": [1]}, id="resume-overrides-not-an-object"),
    pytest.param("POST", "/api/runs/tiny/resume", {"hours": "8"}, id="resume-hours-a-string"),
    pytest.param("POST", "/api/runs/tiny/resume", {"max_games": -5}, id="resume-negative-games"),
    pytest.param("POST", "/api/runs/tiny/resume", {"overrides": {"model.blocks": 3}}, id="resume-locked-setting"),
    pytest.param("POST", "/api/runs/tiny/resume", {"overrides": {"run.out_dir": "C:/x"}}, id="resume-out-dir"),
    pytest.param("POST", "/api/runs", {"runs": [{"name": "x", "overrides": {"run.out_dir": "C:/x"}}]}, id="start-out-dir"),
    pytest.param("POST", "/api/runs", {"runs": [{"name": "x"}], "mode": 5}, id="mode-not-a-string"),
    pytest.param("POST", "/api/runs", {"runs": [{"name": "x"}], "mode": ["queue"]}, id="mode-a-list"),
    pytest.param("POST", "/api/runs/tiny/stop", {"force": "false"}, id="force-a-string"),
    pytest.param("POST", "/api/runs/tiny/stop", {"force": 1}, id="force-a-number"),
    pytest.param("POST", "/api/games", {"players": ["human", "bot:greedy"], "seed": 2**63}, id="seed-too-big"),
    pytest.param("POST", "/api/games", {"players": ["human", "bot:greedy"], "seed": 1.5}, id="seed-a-float"),
    pytest.param("POST", "/api/games", {"players": ["human", "bot:greedy"], "advisor": 5}, id="advisor-not-a-string"),
    pytest.param("POST", "/api/games", {"players": ["human", "bot:greedy", "bot:random"]}, id="three-players"),
    pytest.param("POST", "/api/games/{game}/move", {"exchange": 5}, id="exchange-not-letters"),
    pytest.param("POST", "/api/matches", {"a": "bot:greedy", "b": "bot:random", "pairs": True}, id="pairs-a-bool"),
    pytest.param("POST", "/api/matches", {"a": "bot:greedy", "b": "bot:random", "pairs": 2.5}, id="pairs-a-float"),
    pytest.param("POST", "/api/matches", {"a": "bot:greedy", "b": "bot:random", "candidates": "lots"}, id="candidates-text"),
    pytest.param("POST", "/api/matches", {"a": 5, "b": "bot:random"}, id="player-not-a-string"),
    pytest.param("POST", "/api/models/tiny/best.pt/meta", {"starred": "yes"}, id="starred-a-string"),
    pytest.param("GET", "/api/leaves?bag=101", None, id="bag-over-100"),
    pytest.param("GET", "/api/leaves?bag=1e2", None, id="bag-not-digits"),
    pytest.param("GET", "/api/runs/tiny/metrics?since=-1", None, id="since-negative"),
    pytest.param("GET", "/api/runs/tiny/log?tail=-1", None, id="tail-negative"),
    pytest.param("GET", "/api/games/{game}/hint?top=0", None, id="top-zero"),
    pytest.param("GET", "/api/games/{game}/hint?top=201", None, id="top-over-200"),
])
def test_malformed_input_is_a_400(app, test_lexicon, method, path, body):
    """Malformed values once reached code that raised KeyError (then a 404), TypeError, AttributeError
    or OverflowError (500s)."""
    game = ok(app, "POST", "/api/games", {"players": ["human", "bot:greedy"], "seed": 3})["id"]
    status, data = api(app, method, path.format(game=game), body)
    assert status == 400, data
    assert not app.spawned.calls and not app.spawned.killed


def test_a_plain_key_error_is_a_bug_not_a_404(app, monkeypatch):
    def broken():
        return {}["missing"]

    monkeypatch.setattr(app.state.runs, "list", broken)
    status, data = api(app, "GET", "/api/runs")
    assert status == 500 and data == {"error": "KeyError: 'missing'"}
    assert "KeyError" in (app.app_dir / "app.log").read_text(encoding="utf-8")


def test_a_bug_is_a_500_logged_and_the_server_carries_on(app, monkeypatch):
    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(app.state.runs, "list", broken)
    status, data = api(app, "GET", "/api/runs")
    assert status == 500 and data == {"error": "RuntimeError: boom"}
    log = (app.app_dir / "app.log").read_text(encoding="utf-8")
    assert "GET /api/runs" in log and "RuntimeError: boom" in log and "Traceback" in log
    assert ok(app, "GET", "/api/models")["bots"]


# ---- other -------------------------------------------------------------------------------------


def test_leaves(app):
    leaves = app.trained.loaded.leaves
    value = leaves.value("ERS", 60)
    assert ok(app, "GET", "/api/leaves?rack=ers") == {"rack": "ERS", "bag": 60, "value": round(value, 2),
                                                      "relative": round(value - leaves.value("", 60), 2)}
    assert ok(app, "GET", "/api/leaves?rack=Q%3F&bag=10")["rack"] == "Q?"
    assert ok(app, "GET", "/api/leaves")["rack"] == ""
    for query in ("rack=ABCDEFGH", "rack=E1", "rack=ERS&bag=x"):
        assert api(app, "GET", f"/api/leaves?{query}")[0] == 400, query


def test_errors(app):
    for method, path in (("GET", "/api/nope"), ("POST", "/api/status"), ("DELETE", "/api/status"),
                         ("GET", "/api/games"), ("PUT", "/api/status")):
        status, content_type, content = request(app, method, path)
        if method == "PUT":
            assert status == 501  # http.server answers methods it has no handler for
            continue
        assert status == 404 and strict_json(content)["error"].startswith("no such endpoint"), (method, path)
    status, data = api(app, "POST", "/api/games", raw=b"{not json")
    assert status == 400 and "not valid JSON" in data["error"]
    for raw in (b"[1, 2]", b'"text"', b"null", b"5"):
        status, data = api(app, "POST", "/api/games", raw=raw)
        assert status == 400 and "JSON object" in data["error"], raw
    status, data = api(app, "POST", "/api/games", raw=b"\xff\xfe{}")
    assert status == 400
    for method in ("POST", "DELETE"):
        status, _, content = request(app, method, "/index.html")
        assert status == 405 and content == b"method not allowed"


def raw_request(app, data: bytes) -> bytes:
    """Send bytes as they are and return whatever comes back before the server closes or times out."""
    with socket.create_connection(("127.0.0.1", app.port), timeout=5) as sock:
        sock.sendall(data)
        chunks = []
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\r\n\r\n" in b"".join(chunks):
                    break
        except (socket.timeout, ConnectionError):
            pass
    return b"".join(chunks)


def test_a_malformed_content_length_is_a_400(app):
    reply = raw_request(app, b"POST /api/games HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n{}")
    assert request(app, "GET", "/api/status")[0] == 200, "the server is still up"
    assert reply.startswith(b"HTTP/1.1 400"), reply[:80]


def test_two_requests_on_one_connection(app):
    connection = http.client.HTTPConnection("127.0.0.1", app.port, timeout=30)
    try:
        for _ in range(2):
            connection.request("POST", "/api/games", body=json.dumps({"players": ["human", "bot:greedy"]}),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            assert response.status == 200 and strict_json(response.read())["id"]
    finally:
        connection.close()


# ---- static files ------------------------------------------------------------------------------


def test_static_files(app):
    status, content_type, content = request(app, "GET", "/")
    assert status == 200 and content_type == "text/html; charset=utf-8"
    assert content == (STATIC_DIR / "index.html").read_bytes()
    status, content_type, content = request(app, "GET", "/js/app.js")
    assert status == 200 and content_type in ("text/javascript; charset=utf-8", "application/javascript; charset=utf-8")
    assert content == (STATIC_DIR / "js" / "app.js").read_bytes()
    assert request(app, "GET", "/css/app.css")[1] == "text/css; charset=utf-8"
    assert request(app, "GET", "/icon.svg")[1] == "image/svg+xml; charset=utf-8"
    assert request(app, "GET", "/js/views/play.js?v=2")[0] == 200
    for path in ("/nope.js", "/js", "/js/", "/css/../nope.css"):
        assert request(app, "GET", path)[0] == 404, path


@pytest.mark.parametrize("path", [
    "/../pyproject.toml", "/..%2f..%2fpyproject.toml", "/..%2F..%2F..%2F..%2Fpyproject.toml",
    "/js/..%2f..%2f..%2fserver.py", "/js/..%2f..%2fserver.py", "/..%5c..%5cpyproject.toml", "/js/..\\..\\server.py",
    "/%2e%2e/%2e%2e/server.py", "/C:/Windows/win.ini", "/C:%5cWindows%5cwin.ini", "/%2fWindows/win.ini",
    "//server.py", "/static/../server.py", "/./../server.py",
])
def test_static_paths_cannot_leave_the_static_folder(app, path):
    status, _, content = request(app, "GET", path)
    assert status == 404, path
    assert b"[project]" not in content and b"import" not in content and b"[fonts]" not in content


@pytest.mark.parametrize("path", ["/index%00.html", "/..%00/server.py", "/js/app.js%00", "/%00"])
def test_a_nul_byte_in_a_static_path_is_a_404(app, path):
    """It once made Path.resolve raise outside any try: the connection dropped with no response."""
    status, _, content = request(app, "GET", path)
    assert status == 404 and content == b"not found"
    assert request(app, "GET", "/api/status")[0] == 200


def test_the_server_listens_on_loopback_only(app):
    assert app.server.server_address[0] == "127.0.0.1"
    others = {ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")}
    if not others:
        pytest.skip("this machine has no other IPv4 address to try")
    for ip in others:
        with pytest.raises(OSError):
            socket.create_connection((ip, app.port), timeout=0.5).close()
