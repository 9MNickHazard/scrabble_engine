"""Runs that train from saved games, as the app handles them (`RunManager`, the run report route
and the settings schema), against docs/APP_API.md, "Training from saved games". Nothing is
started: `spawned` (app_fixtures.py) records what would have been."""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scrabble_ai.app import schema as schema_module
from scrabble_ai.app.runs import LOCKED_ON_RESUME, RunError, RunManager, python_executable
from scrabble_ai.app.server import AppState, create_server
from scrabble_ai.config import OFFLINE_DEFAULTS, load_config

pytest.register_assert_rewrite("app_fixtures", "app_games_helpers")
from app_fixtures import DEAD_PID, TINY_RUN, copy_run, make_run, record  # noqa: E402
from app_games_helpers import loaded_for, make_set  # noqa: E402

pytest_plugins = ["app_fixtures"]

GAMES_FIELDS = ("mode", "game_set", "total_steps", "passes", "pass", "positions_per_second")
REPORT_SUMMARY_KEYS = {"steps", "positions", "hours", "passes", "game_set", "train_games", "mean_positions_per_second",
                       "best", "final", "first_significant", "held_out", "evaluations", "health"}
MATCH_KEYS = {"games", "step", "hours", "win_rate", "win_ci", "spread", "ci", "pairs"}


@pytest.fixture
def dirs(tmp_path, spawned, test_lexicon):
    runs_dir, app_dir = tmp_path / "runs", tmp_path / "app"
    runs_dir.mkdir()
    app_dir.mkdir()
    return SimpleNamespace(runs=runs_dir, app=app_dir, games=tmp_path / "games", spawned=spawned)


@pytest.fixture
def manager(dirs):
    return RunManager(dirs.runs, dirs.app)


@pytest.fixture
def loaded(engine, leaves):
    return loaded_for(engine, leaves)


def games_records(t0: float, steps=(0, 4, 8), total_steps=16, steps_per_pass=8, batch=32, stop=None) -> list[dict]:
    """What a games run logs: a start, a train record per step given, and maybe a stop."""
    records = [record("session", t0, event="start", mode="games", parameters=5000, limit_hours=0.0, game_set="s1",
                      train_games=270, validation_games=30, total_steps=total_steps, steps_per_pass=steps_per_pass,
                      batch=batch)]
    for i, step in enumerate(steps[1:], 1):
        records.append(record("train", t0 + 60 * i, step=step, games=step * batch, positions=step * batch,
                              loss=1.0, win_loss=0.69, spread_loss=0.3, learning_rate=1e-4,
                              positions_per_second=100.0 * i, **{"pass": round(step / steps_per_pass, 3)}))
    if stop is not None:
        last = steps[-1]
        records.append(record("session", t0 + 1000, step=last, games=last * batch, positions=last * batch, event="stop",
                              reason=stop[0], finished=stop[1]))
    return records


# ---- the kind of run ---------------------------------------------------------------------------


def test_mode_comes_from_the_settings(manager, dirs):
    make_run(dirs.runs, "sp", settings={})
    make_run(dirs.runs, "gm", settings={"offline.games": str(dirs.games / "s1")})
    make_run(dirs.runs, "queued", queued={"queue": "q", "overrides": {"offline.games": str(dirs.games / "s1")},
                                          "created": time.time()})
    assert (manager.mode("sp"), manager.mode("gm"), manager.mode("queued")) == ("selfplay", "games", "games")
    assert manager.summary("queued")["game_set"] == "s1"


def test_starting_a_games_run(manager, dirs, loaded):
    make_set(dirs.games, "s1", loaded, 40)
    result = manager.start([{"name": "g1", "overrides": {"offline.games": "s1", "run.hours": 2}}], "parallel")
    assert result == {"started": ["g1"], "queued": []}
    command, log = dirs.spawned.calls[-1]
    folder = str(dirs.games / "s1")
    assert command == [python_executable(), "-u", "-m", "scrabble_ai.cli", "train-games", "--name", "g1",
                       "--set", f"run.out_dir={dirs.runs}", "--set", f"offline.games={folder}", "--set", "run.hours=2"]
    stored = json.loads((dirs.runs / "g1" / "unstarted.json").read_text(encoding="utf-8"))["overrides"]
    assert stored == {"offline.games": folder, "run.hours": 2}, "the set is named by its folder"
    summary = manager.summary("g1")
    assert summary["status"] == "running" and summary["mode"] == "games" and summary["game_set"] == "s1"
    assert (summary["total_steps"], summary["pass"], summary["positions_per_second"], summary["progress"]) == (
        None, None, None, None), "nothing is known before the trainer has started"
    assert summary["passes"] == 6.0
    assert manager.using_games("s1") == ["g1"] and manager.using_games("other") == []
    # A self-play run started beside it trains with `train`.
    manager.start([{"name": "sp", "overrides": {"run.hours": 1}}], "parallel")
    assert dirs.spawned.calls[-1][0][4] == "train"
    assert manager.summary("sp")["mode"] == "selfplay"


def test_a_queue_of_both_kinds(manager, dirs, loaded):
    make_set(dirs.games, "s1", loaded, 40)
    make_run(dirs.runs, "busy", records=[record("session", time.time(), event="start")], pid=os.getpid())
    result = manager.start([{"name": "q-games", "overrides": {"offline.games": "s1"}}, {"name": "q-self"}], "queue")
    assert result == {"started": [], "queued": ["q-games", "q-self"]}
    queue_file = next((dirs.app / "queues").glob("queue_*.json"))
    entries = json.loads(queue_file.read_text(encoding="utf-8"))["runs"]
    assert [e["command"][4] for e in entries] == ["train-games", "train"]
    assert f"offline.games={dirs.games / 's1'}" in entries[0]["command"]
    assert manager.summary("q-games")["status"] == "queued" and manager.using_games("s1") == ["q-games"]


def test_a_set_that_cannot_be_trained_from_is_refused(manager, dirs, loaded):
    make_set(dirs.games, "empty", loaded, 0, target=50)
    make_set(dirs.games, "building", loaded, 20, target=100, pid=os.getpid())
    make_set(dirs.games, "was-building", loaded, 20, target=100, pid=DEAD_PID)
    for value, message in (("nope", "no game set"), ("empty", "no game set"), ("building", "still being built"),
                           ("bad name", "no game set"), (5, "must be the name")):
        with pytest.raises(RunError, match=message):
            manager.start([{"name": "g", "overrides": {"offline.games": value}}], "parallel")
    assert dirs.spawned.calls == []
    # A set whose process died can be trained from with what it has.
    assert manager.start([{"name": "g", "overrides": {"offline.games": "was-building"}}], "parallel")["started"] == ["g"]
    # With two runs, the message says which.
    with pytest.raises(RunError, match="^g2: .*no game set"):
        manager.start([{"name": "g1"}, {"name": "g2", "overrides": {"offline.games": "nope"}}], "parallel")


def test_the_games_defaults_apply_before_the_runs_own_settings(manager, dirs):
    folder = str(dirs.games / "s1")
    config = manager._check_overrides({"offline.games": folder})
    for key, value in OFFLINE_DEFAULTS.items():
        section, name = key.split(".")
        assert getattr(getattr(config, section), name) == value, key
    config = manager._check_overrides({"offline.games": folder, "model.head": "flat", "selfplay.max_candidates": 48})
    assert (config.model.head, config.selfplay.max_candidates, config.model.win_head) == ("flat", 48, "wdl")
    selfplay = manager._check_overrides({"model.blocks": 4})
    assert (selfplay.model.head, selfplay.model.win_head, selfplay.selfplay.max_candidates) == ("flat", "binary", 64)
    # Validated with those defaults: 40 exchanges among 32 candidates is not a setting.
    with pytest.raises(RunError, match="max_exchanges"):
        manager._check_overrides({"offline.games": folder, "selfplay.max_exchanges": 40})
    manager._check_overrides({"selfplay.max_exchanges": 40})


def test_the_summary_of_a_games_run(manager, dirs):
    folder = str(dirs.games / "s1")
    t0 = time.time() - 600
    make_run(dirs.runs, "live", settings={"offline.games": folder, "offline.passes": 2.0}, pid=os.getpid(),
             records=games_records(t0, steps=(0, 4, 6)))
    summary = manager.summary("live")
    assert summary["mode"] == "games" and summary["game_set"] == "s1"
    assert (summary["total_steps"], summary["passes"], summary["pass"], summary["positions_per_second"]) == (16, 2.0, 0.75, 200.0)
    assert summary["progress"] == pytest.approx(6 / 16)
    assert (summary["step"], summary["games"], summary["positions"]) == (6, 192, 192)
    assert summary["status"] == "running" and summary["phase"] == "training", "a games run has one phase"
    assert set(summary["trend"]) == {"games", "spread", "ci", "step", "win_rate", "win_ci"}
    assert summary["best_eval"] is None and summary["health"] is None and summary["latest_eval"] is None

    make_run(dirs.runs, "done", settings={"offline.games": folder}, records=games_records(
        t0, steps=(0, 8, 16), stop=("all 16 steps done", True)))
    done = manager.summary("done")
    assert (done["status"], done["progress"], done["pass"]) == ("finished", 1.0, 2.0)
    make_run(dirs.runs, "stopped", settings={"offline.games": folder}, records=games_records(
        t0, steps=(0, 4), stop=("STOP file found", False)))
    stopped = manager.summary("stopped")
    assert stopped["status"] == "stopped" and stopped["progress"] == pytest.approx(0.25)
    assert manager.using_games("s1") == ["live"]


def test_a_games_run_is_judged_by_its_win_rate(manager, dirs):
    folder = str(dirs.games / "s1")
    t0 = time.time() - 600
    evals = [record("eval", t0 + 100 * (i + 1), step=s, games=s * 32, positions=s * 32, opponent="greedy-leave",
                    spread=spread, spread_ci=4.0, win_rate=rate, win_ci=0.02, score_for=420.0, score_against=418.0, pairs=500)
             for i, (s, rate, spread) in enumerate(((4, 0.51, 2.0), (8, 0.56, -1.0), (12, 0.55, -3.0)))]
    make_run(dirs.runs, "judged", settings={"offline.games": folder}, records=games_records(t0) + evals)
    summary = manager.summary("judged")
    assert summary["health"]["state"] == "ahead" and summary["health"]["edge"] is None
    assert "55.0% +/- 2.0" in summary["health"]["note"]
    assert summary["trend"] == {"games": [128, 256, 384], "spread": [2.0, -1.0, -3.0], "ci": [4.0, 4.0, 4.0],
                                "step": [4, 8, 12], "win_rate": [0.51, 0.56, 0.55], "win_ci": [0.02, 0.02, 0.02]}
    assert summary["best_eval"] == {"games": 256, "step": 8, "hours": 0.0, "win_rate": 0.56, "win_ci": 0.02, "spread": -1.0,
                                    "ci": 4.0, "pairs": 500}
    assert summary["latest_eval"] == {"games": 384, "greedy-leave": {"spread": -3.0, "ci": 4.0, "win_rate": 0.55,
                                                                     "win_ci": 0.02}}
    # The same records in a self-play run are judged by points.
    make_run(dirs.runs, "points", settings={}, records=games_records(t0) + evals)
    assert manager.summary("points")["health"]["state"] == "level"


def test_resuming_a_games_run(manager, dirs):
    folder = str(dirs.games / "s1")
    make_run(dirs.runs, "g", settings={"offline.games": folder}, latest="step_000000008.pt",
             records=games_records(time.time() - 100, stop=("STOP file found", False)))
    with pytest.raises(RunError, match="offline.games is fixed"):
        manager.resume("g", overrides={"offline.games": str(dirs.games / "other")})
    with pytest.raises(RunError, match="model.head is fixed"):
        manager.resume("g", overrides={"model.head": "flat"})
    for key, value in (("offline.batch_size", 512), ("offline.validation_games", 0)):
        with pytest.raises(RunError, match=f"{key} is fixed"):
            manager.resume("g", overrides={key: value})
    summary = manager.resume("g", hours=3, overrides={"offline.learning_rate": 0.0001})
    assert summary["mode"] == "games"
    command = dirs.spawned.calls[-1][0]
    assert command[4] == "train-games"
    assert command[5:9] == ["--name", "g", "--set", f"run.out_dir={dirs.runs}"]
    assert command[9:] == ["--set", "offline.learning_rate=0.0001", "--hours", "3", "--resume", "latest"]
    # A run that died before its first checkpoint starts again as the run that was asked for.
    make_run(dirs.runs, "early", pid=DEAD_PID)
    (dirs.runs / "early" / "unstarted.json").write_text(json.dumps({"overrides": {"offline.games": folder}}),
                                                        encoding="utf-8")
    manager.resume("early")
    command = dirs.spawned.calls[-1][0]
    assert command[4] == "train-games" and f"offline.games={folder}" in command and "--resume" not in command


def test_the_schema_agrees_with_what_resume_refuses(fields_by_key):
    """`locked_on_resume` tells the form which settings it may not offer to change on resume;
    runs.py refuses exactly those under LOCKED_ON_RESUME ("offline.games cannot change on
    resume", docs/APP_API.md)."""
    for key, field in fields_by_key.items():
        assert field["locked_on_resume"] == key.startswith(LOCKED_ON_RESUME), key


# ---- the schema --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema():
    return schema_module.schema()


@pytest.fixture(scope="module")
def fields_by_key(schema):
    return {field["key"]: field for section in schema["sections"] for field in section["fields"]}


# What the trainer that learns from games reads outside its own section (offline.py).
READ_BY_GAMES = {"selfplay.max_candidates", "selfplay.max_exchanges", "selfplay.win_weight", "selfplay.win_scale",
                 "train.value_scale", "train.huber_delta"}


def test_every_field_says_which_form_shows_it(fields_by_key):
    for key, field in fields_by_key.items():
        section = key.split(".")[0]
        modes = field["modes"]
        if section == "offline":
            assert modes == ["games"], key
        elif key == "run.max_games":
            assert modes == ["selfplay"], "training from games ends when its passes are done"
        elif section in ("run", "model") or key in READ_BY_GAMES:
            assert modes == ["selfplay", "games"], key
        else:
            assert modes == ["selfplay"], key
    offline = [s for s in schema_module.schema()["sections"] if s["name"] == "offline"][0]
    assert offline["title"] == "Training from games"
    assert [f["key"] for f in offline["fields"]][0] == "offline.games"


def test_the_new_model_and_train_settings_are_described(fields_by_key):
    assert fields_by_key["model.kind"]["options"] == ["cnn", "cnn-global", "transformer"]
    assert fields_by_key["model.head"]["options"] == ["flat", "pooled"]
    assert fields_by_key["model.win_head"]["options"] == ["binary", "wdl"]
    assert fields_by_key["model.aux_heads"]["type"] == "bool"
    assert fields_by_key["train.target"]["options"] == ["td", "q"] and fields_by_key["train.target"]["default"] == "td"
    for key in ("model.kind", "model.head", "model.win_head", "model.aux_heads"):
        assert fields_by_key[key]["locked_on_resume"] is True, key
    assert fields_by_key["offline.games"]["advanced"] is False and fields_by_key["offline.passes"]["advanced"] is False


def test_games_presets_are_real_settings_on_top_of_the_games_defaults(schema, fields_by_key):
    assert schema["games_defaults"] == OFFLINE_DEFAULTS
    defaults = [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in schema["games_defaults"].items()]
    for key, value in schema["games_defaults"].items():
        assert "games" in fields_by_key[key]["modes"], key
        field = fields_by_key[key]
        if field["options"]:
            assert value in field["options"], key
    presets = schema["games_presets"]
    names = [p["name"] for p in presets]
    assert presets and len(names) == len(set(names)) and presets[0]["overrides"] == {}
    for preset in presets:
        assert set(preset) == {"name", "description", "overrides"} and preset["name"] and preset["description"]
        overrides = preset["overrides"]
        for key, value in overrides.items():
            field = fields_by_key[key]
            assert "games" in field["modes"], (preset["name"], key)
            if field["options"]:
                assert value in field["options"], (preset["name"], key)
            if field["min"] is not None:
                assert value >= field["min"], (preset["name"], key)
            if field["max"] is not None:
                assert value <= field["max"], (preset["name"], key)
            if field["multiple_of"]:
                assert value % field["multiple_of"] == 0, (preset["name"], key)
        items = defaults + [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in overrides.items()]
        config = load_config(None, items)
        assert config.model.head == "pooled", preset["name"]


def test_the_schema_route_carries_both_kinds(app):
    data = ok(app, "GET", "/api/config/schema")
    assert data["games_defaults"] == OFFLINE_DEFAULTS and data["games_presets"] == schema_module.GAMES_PRESETS
    assert all("modes" in f for s in data["sections"] for f in s["fields"])


# ---- the server: report and runs ---------------------------------------------------------------


@pytest.fixture
def app(tmp_path, trained, spawned, fast_scorer, test_lexicon):
    runs_dir, app_dir = tmp_path / "runs", tmp_path / "app"
    runs_dir.mkdir()
    app_dir.mkdir()
    copy_run(trained, runs_dir)
    state = AppState(runs_dir, app_dir, trained.loaded)
    state.players._device = torch.device("cpu")
    state.gpu = lambda: None
    server = create_server(0, state=state, log_path=app_dir / "app.log")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(state=state, port=server.server_address[1], runs_dir=runs_dir, games_dir=tmp_path / "games",
                              spawned=spawned, loaded=trained.loaded)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


def api(app, method: str, path: str, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", app.port, timeout=60)
    try:
        data = None if body is None else json.dumps(body).encode("utf-8")
        connection.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        response = connection.getresponse()
        content = json.loads(response.read())
    finally:
        connection.close()
    return response.status, content


def ok(app, method: str, path: str, body=None):
    status, data = api(app, method, path, body)
    assert status == 200, (method, path, data)
    return data


def test_the_report_of_a_games_run(app):
    folder = str(app.games_dir / "s1")
    t0 = time.time() - 600
    records = games_records(t0, steps=(0, 4, 8), stop=("all 16 steps done", True)) + [
        record("valid", t0 + 200, step=8, games=256, positions=256, win_loss=0.65, spread_loss=0.3, win_accuracy=0.61,
               win_brier=0.22, value_rmse_points=70.0, leave_only_rmse_points=74.0, positions_checked=30),
        record("eval", t0 + 300, step=8, games=256, positions=256, opponent="greedy-leave", spread=2.1, spread_ci=5.2,
               win_rate=0.534, win_ci=0.013, score_for=420.0, score_against=418.0, pairs=2000),
        record("best", t0 + 301, step=8, games=256, positions=256, win_rate=0.534, win_ci=0.013, pairs=2000),
    ]
    make_run(app.runs_dir, "night5", settings={"offline.games": folder}, records=records)
    report = ok(app, "GET", "/api/runs/night5/report")
    assert set(report) == {"headline", "summary", "events"}
    assert report["headline"] == ("night5 wins 53.4% +/- 1.3 of games against the static evaluator, more than half "
                                  "beyond the margin of error.")
    summary = report["summary"]
    assert set(summary) == REPORT_SUMMARY_KEYS
    assert (summary["steps"], summary["positions"], summary["passes"], summary["game_set"], summary["train_games"]) == (
        8, 256, 1.0, "s1", 270)
    assert set(summary["best"]) == MATCH_KEYS and summary["best"]["win_rate"] == 0.534
    assert summary["held_out"] == {"win_loss": 0.65, "win_accuracy": 0.61, "value_rmse_points": 70.0,
                                   "leave_only_rmse_points": 74.0}
    assert summary["evaluations"] == 1 and summary["health"]["state"] == "ahead"
    assert {e["kind"] for e in report["events"]} <= {"phase", "significant", "promotion", "stop"}
    # A self-play run gets the self-play report.
    assert "promotions" in ok(app, "GET", f"/api/runs/{TINY_RUN}/report")["summary"]
    summary = ok(app, "GET", "/api/runs/night5")["run"]
    assert summary["mode"] == "games" and summary["game_set"] == "s1"


def test_the_report_of_a_games_run_that_died(app):
    """As for a self-play run: the log of a process that died has no stop record, so the report
    is told (server.py adds "Run ended unexpectedly" for self-play runs)."""
    folder = str(app.games_dir / "s1")
    make_run(app.runs_dir, "died", settings={"offline.games": folder}, pid=DEAD_PID,
             console=["Traceback (most recent call last):", "RuntimeError: CUDA out of memory"],
             records=games_records(time.time() - 100))
    events = ok(app, "GET", "/api/runs/died/report")["events"]
    assert events and events[-1]["title"] == "Run ended unexpectedly" and events[-1]["kind"] == "stop"


def test_starting_a_games_run_through_the_server(app):
    make_set(app.games_dir, "s1", app.loaded, 40)
    result = ok(app, "POST", "/api/runs", {"runs": [{"name": "g1", "overrides": {"offline.games": "s1"}}]})
    assert result == {"started": ["g1"], "queued": []}
    assert app.spawned.calls[-1][0][4] == "train-games"
    status, data = api(app, "POST", "/api/runs", {"runs": [{"name": "g2", "overrides": {"offline.games": "missing"}}]})
    assert status == 400 and "no game set" in data["error"]
    runs = {r["name"]: r for r in ok(app, "GET", "/api/runs")["runs"]}
    assert runs["g1"]["mode"] == "games" and runs[TINY_RUN]["mode"] == "selfplay"
    status, data = api(app, "DELETE", "/api/gamesets/s1")
    assert status == 400 and "g1" in data["error"]
    assert Path(app.games_dir / "s1").is_dir()
