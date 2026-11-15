"""`scrabble_ai.app.gamesets.GameSets` and the `/api/gamesets` routes of docs/APP_API.md
("Training from saved games"), with the process that plays the games replaced by a recorder
(`spawned` in app_fixtures.py): nothing is started detached and nothing is killed."""

from __future__ import annotations

import dataclasses
import http.client
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from scrabble_ai import games as games_module
from scrabble_ai.app import gamesets as gamesets_module
from scrabble_ai.app.gamesets import BYTES_PER_GAME, MAX_GAMES, GameSets
from scrabble_ai.app.players import NotFound
from scrabble_ai.app.runs import RunError, python_executable
from scrabble_ai.app.server import AppState, create_server

pytest.register_assert_rewrite("app_fixtures", "app_games_helpers")
from app_fixtures import DEAD_PID, TINY_RUN, copy_run, make_run, record  # noqa: E402
from app_games_helpers import SET_KEYS, loaded_for, make_set  # noqa: E402

pytest_plugins = ["app_fixtures"]


@pytest.fixture
def loaded(engine, leaves):
    return loaded_for(engine, leaves)


@pytest.fixture
def sets(tmp_path, spawned):
    games_dir = tmp_path / "games"
    return SimpleNamespace(manager=GameSets(games_dir), dir=games_dir, spawned=spawned)


# ---- statuses ----------------------------------------------------------------------------------


def test_complete_partial_and_empty(sets, loaded):
    make_set(sets.dir, "done", loaded, 40)
    make_set(sets.dir, "half", loaded, 40, target=100)
    make_set(sets.dir, "none", loaded, 0, target=50)
    done = sets.manager.summary("done")
    assert set(done) == SET_KEYS
    assert (done["status"], done["games"], done["target"], done["progress"], done["error"]) == ("complete", 40, 40, 1.0, None)
    assert done["lexicon_id"] == loaded.lexicon_id and done["leaves_hash"] == loaded.leaves_hash
    assert done["bytes"] == (sets.dir / "done" / "games.bin").stat().st_size
    assert done["mean_turns"] > 0 and done["opening_plies"] == 4 and done["opening_temperature"] == 6.0
    half = sets.manager.summary("half")
    assert (half["status"], half["games"], half["target"], half["progress"]) == ("partial", 40, 100, 0.4)
    none = sets.manager.summary("none")
    assert (none["status"], none["games"], none["target"], none["progress"]) == ("empty", 0, 50, 0.0)
    assert none["mean_turns"] is None and none["games_per_second"] is None


def test_building_stopping_and_failed(sets, loaded):
    make_set(sets.dir, "live", loaded, 20, target=200, pid=os.getpid(), request={"games": 300, "opening_plies": None,
                                                                                  "opening_temperature": None,
                                                                                  "requested": time.time()})
    live = sets.manager.summary("live")
    assert live["status"] == "building" and live["error"] is None
    assert live["target"] == 300, "while it builds, the target is what was last asked for"
    assert live["progress"] == pytest.approx(20 / 300)
    (sets.dir / "live" / "STOP").write_text("", encoding="utf-8")
    assert sets.manager.summary("live")["status"] == "stopping"

    lines = [f"line {i}" for i in range(20)]
    make_set(sets.dir, "dead", loaded, 20, target=200, pid=DEAD_PID, console=lines)
    dead = sets.manager.summary("dead")
    assert dead["status"] == "failed" and dead["error"] == "\n".join(lines[-8:])
    assert dead["target"] == 200
    make_set(sets.dir, "silent", loaded, 20, pid=DEAD_PID)
    assert sets.manager.summary("silent")["error"] == "The process that was playing the games ended unexpectedly."


def test_a_set_known_only_by_its_request(sets):
    """The app has asked for a set and the process has not written meta.json yet."""
    request = {"games": 1000, "opening_plies": 3, "opening_temperature": 2.5, "requested": 1794700000.0}
    make_set(sets.dir, "asked", None, 0, request=request, pid=os.getpid())
    summary = sets.manager.summary("asked")
    assert set(summary) == SET_KEYS
    assert (summary["status"], summary["games"], summary["target"], summary["progress"]) == ("building", 0, 1000, 0.0)
    assert (summary["opening_plies"], summary["opening_temperature"]) == (3, 2.5)
    assert summary["lexicon_id"] is None and summary["leaves_hash"] is None and summary["mean_score"] is None
    assert summary["created"] == summary["updated"] == 1794700000.0
    (sets.dir / "asked" / "running.json").unlink()
    after = sets.manager.summary("asked")
    assert (after["status"], after["progress"]) == ("empty", None)
    with pytest.raises(NotFound):
        sets.manager.summary("nothing")
    with pytest.raises(NotFound):  # a name that could not be a set's is simply not there
        sets.manager.summary("bad name")


def test_list_order_and_estimate(sets, loaded):
    assert sets.manager.list() == {"sets": [], "estimate": {"bytes_per_game": BYTES_PER_GAME, "games_per_second": None}}
    make_set(sets.dir, "older", loaded, 40)
    time.sleep(0.02)
    make_set(sets.dir, "newer", loaded, 20)
    make_set(sets.dir, "going", loaded, 20, target=100, pid=os.getpid())
    (sets.dir / "not-a-set").mkdir()  # no meta.json or request.json: not listed
    (sets.dir / "bad.name").mkdir()
    (sets.dir / "bad.name" / "meta.json").write_text("{}", encoding="utf-8")
    listing = sets.manager.list()
    assert [s["name"] for s in listing["sets"]] == ["going", "newer", "older"], "building first, then the newest"
    measured = [games_module.read_meta(sets.dir / n) for n in ("going", "newer", "older")]
    assert listing["estimate"]["bytes_per_game"] == pytest.approx(sum(m["bytes"] for m in measured) / 80)
    first_rate = next(s["games_per_second"] for s in listing["sets"] if s["games_per_second"])
    assert listing["estimate"]["games_per_second"] == first_rate
    # One set that cannot be read does not hide the others.
    (sets.dir / "broken").mkdir()
    (sets.dir / "broken" / "meta.json").write_text(json.dumps({"format": games_module.FORMAT, "games": 3}), encoding="utf-8")
    listing = sets.manager.list()
    broken = next(s for s in listing["sets"] if s["name"] == "broken")
    assert broken["status"] == "failed" and "could not read" in broken["error"]
    assert {s["name"] for s in listing["sets"]} == {"going", "newer", "older", "broken"}


def test_log(sets, loaded):
    make_set(sets.dir, "logged", loaded, 20, console=[f"line {i}" for i in range(300)])
    assert sets.manager.log("logged") == [f"line {i}" for i in range(100, 300)]
    assert sets.manager.log("logged", 2) == ["line 298", "line 299"]
    assert sets.manager.log("logged", 0) == []
    make_set(sets.dir, "quiet", loaded, 20)
    assert sets.manager.log("quiet") == []


# ---- starting ----------------------------------------------------------------------------------


def test_start_a_new_set(sets):
    summary = sets.manager.start("fresh", 5000, 3, 2.5)
    assert summary["status"] == "building" and summary["name"] == "fresh" and summary["target"] == 5000
    command, log = sets.spawned.calls[-1]
    assert command == [python_executable(), "-u", "-m", "scrabble_ai.cli", "games", "--name", "fresh", "--games", "5000",
                       "--dir", str(sets.dir), "--opening-plies", "3", "--opening-temperature", "2.5"]
    assert log == sets.dir / "fresh" / "console.log"
    request = json.loads((sets.dir / "fresh" / "request.json").read_text(encoding="utf-8"))
    assert (request["games"], request["opening_plies"], request["opening_temperature"]) == (5000, 3, 2.5)
    marker = json.loads((sets.dir / "fresh" / "running.json").read_text(encoding="utf-8"))
    assert marker["pid"] == sets.spawned.pid
    # The opening settings are optional: the generator's own defaults then apply.
    sets.manager.start("plain", 10)
    assert sets.spawned.calls[-1][0][-6:] == ["--name", "plain", "--games", "10", "--dir", str(sets.dir)]


def test_start_tops_up_a_set_that_exists(sets, loaded):
    make_set(sets.dir, "grow", loaded, 40, target=100)
    (sets.dir / "grow" / "STOP").write_text("", encoding="utf-8")  # left from the last stop
    with pytest.raises(RunError, match="already has 40 games"):
        sets.manager.start("grow", 40)
    with pytest.raises(RunError, match="keeps the settings it was made with"):
        sets.manager.start("grow", 100, opening_plies=2)
    with pytest.raises(RunError, match="keeps the settings it was made with"):
        sets.manager.start("grow", 100, opening_temperature=1.0)
    assert sets.spawned.calls == []
    summary = sets.manager.start("grow", 100, opening_plies=4, opening_temperature=6)
    assert summary["status"] == "building" and not (sets.dir / "grow" / "STOP").exists()
    command = sets.spawned.calls[-1][0]
    assert command[-6:] == ["--name", "grow", "--games", "100", "--dir", str(sets.dir)], \
        "the set's own opening settings are not passed again"
    with pytest.raises(RunError, match="already being built"):
        sets.manager.start("grow", 200)


def test_start_refusals(sets, loaded):
    cases = [
        (("bad name", 10), "letters"), (("a.b", 10), "letters"), (("x" * 65, 10), "at most 64"), ((5, 10), "letters"),
        (("ok", 0), "between 1 and"), (("ok", -3), "between 1 and"), (("ok", MAX_GAMES + 1), "between 1 and"),
        (("ok", True), "whole number"), (("ok", 2.5), "whole number"), (("ok", "10"), "whole number"),
        (("ok", None), "whole number"),
        (("ok", 10, -1), "opening_plies"), (("ok", 10, 21), "opening_plies"), (("ok", 10, True), "opening_plies"),
        (("ok", 10, 1.5), "opening_plies"),
        (("ok", 10, None, -0.5), "opening_temperature"), (("ok", 10, None, 101), "opening_temperature"),
        (("ok", 10, None, False), "opening_temperature"), (("ok", 10, None, "6"), "opening_temperature"),
    ]
    for args, message in cases:
        with pytest.raises(RunError, match=message):
            sets.manager.start(*args)
    sets.manager.start("x" * 64, MAX_GAMES)  # the limits themselves are fine
    # A folder that holds something else is not taken over.
    (sets.dir / "occupied").mkdir(parents=True)
    (sets.dir / "occupied" / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(RunError, match="already exists"):
        sets.manager.start("occupied", 10)
    make_set(sets.dir, "Mixed", loaded, 20)
    with pytest.raises(RunError, match="already exists"):
        sets.manager.start("mixed", 30)
    assert len(sets.spawned.calls) == 1


# ---- stopping and deleting ---------------------------------------------------------------------


def test_stop_and_force_stop(sets, loaded):
    make_set(sets.dir, "live", loaded, 20, target=100, pid=os.getpid())
    sets.manager.stop("live")
    assert (sets.dir / "live" / "STOP").exists() and sets.manager.summary("live")["status"] == "stopping"
    sets.manager.stop("live", force=True)
    assert sets.spawned.killed == [os.getpid()]
    assert not (sets.dir / "live" / "running.json").exists() and not (sets.dir / "live" / "STOP").exists()
    assert sets.manager.summary("live")["status"] == "partial", "it keeps what was complete"
    with pytest.raises(RunError, match="not being built"):
        sets.manager.stop("live")
    make_set(sets.dir, "dead", loaded, 20, pid=DEAD_PID)
    with pytest.raises(RunError, match="not being built"):
        sets.manager.stop("dead")
    with pytest.raises(NotFound):
        sets.manager.stop("nothing")
    assert sets.spawned.killed == [os.getpid()]


def test_delete(sets, loaded):
    make_set(sets.dir, "old", loaded, 20)
    make_set(sets.dir, "live", loaded, 20, target=100, pid=os.getpid())
    make_set(sets.dir, "used", loaded, 20)
    with pytest.raises(RunError, match="stop it first"):
        sets.manager.delete("live")
    with pytest.raises(RunError, match="night1, night2"):
        sets.manager.delete("used", ["night1", "night2"])
    assert (sets.dir / "live").is_dir() and (sets.dir / "used").is_dir()
    sets.manager.delete("used", [])
    sets.manager.delete("old")
    assert not (sets.dir / "old").exists() and not (sets.dir / "used").exists()
    with pytest.raises(NotFound):
        sets.manager.delete("old")
    with pytest.raises(NotFound):
        sets.manager.delete("..")
    assert sets.dir.is_dir() and sets.manager.names() == ["live"]


def test_exists_and_names(sets, loaded):
    assert sets.manager.names() == [] and not sets.manager.exists("a")
    make_set(sets.dir, "a", loaded, 20)
    make_set(sets.dir, "b", None, 0, request={"games": 5})
    assert sets.manager.names() == ["a", "b"] and sets.manager.exists("a") and not sets.manager.exists("bad name")


# ---- the routes --------------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path, trained, spawned, fast_scorer):
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
    if response.status >= 400:
        assert set(content) == {"error"} and content["error"], (path, content)
    return response.status, content


def ok(app, method: str, path: str, body=None):
    status, data = api(app, method, path, body)
    assert status == 200, (method, path, data)
    return data


def test_the_routes_list_and_describe_sets(app):
    assert app.state.gamesets.games_dir == app.state.runs.games_dir == app.games_dir
    assert ok(app, "GET", "/api/gamesets") == {"sets": [], "estimate": {"bytes_per_game": BYTES_PER_GAME,
                                                                         "games_per_second": None}}
    make_set(app.games_dir, "twenty", app.loaded, 40)
    make_set(app.games_dir, "live", app.loaded, 20, target=100, pid=os.getpid(), console=["  20 games (100/s), 0 MB"])
    listing = ok(app, "GET", "/api/gamesets")
    assert set(listing) == {"sets", "estimate"} and set(listing["estimate"]) == {"bytes_per_game", "games_per_second"}
    assert [s["name"] for s in listing["sets"]] == ["live", "twenty"]
    for entry in listing["sets"]:
        assert set(entry) == SET_KEYS, entry["name"]
    summary = ok(app, "GET", "/api/gamesets/twenty")
    assert set(summary) == SET_KEYS and summary["status"] == "complete" and summary["games"] == 40
    assert ok(app, "GET", "/api/gamesets/twenty/") == summary
    assert ok(app, "GET", "/api/gamesets/live/log?tail=5") == {"lines": ["  20 games (100/s), 0 MB"]}
    assert ok(app, "GET", "/api/gamesets/twenty/log") == {"lines": []}
    assert api(app, "GET", "/api/gamesets/live/log?tail=x")[0] == 400


def test_the_routes_say_404_for_a_set_that_does_not_exist(app):
    assert api(app, "GET", "/api/gamesets/nope")[0] == 404
    assert api(app, "POST", "/api/gamesets/nope/stop", {})[0] == 404
    assert api(app, "DELETE", "/api/gamesets/nope")[0] == 404
    # "404 for a set that does not exist", for its log too.
    assert api(app, "GET", "/api/gamesets/nope/log")[0] == 404


def test_a_name_that_cannot_be_a_set_is_not_found(app):
    """As with runs (`/api/runs/bad.name` is a 404), a set that cannot exist is not found."""
    for path in ("/api/gamesets/bad.name", "/api/gamesets/%2e%2e"):
        assert api(app, "GET", path)[0] == 404, path


def test_the_route_that_starts_a_set(app):
    summary = ok(app, "POST", "/api/gamesets", {"name": "twenty-million", "games": 20000000, "opening_plies": 4,
                                                "opening_temperature": 6.0})
    assert set(summary) == SET_KEYS and summary["status"] == "building" and summary["target"] == 20000000
    command = app.spawned.calls[-1][0]
    assert command[4:12] == ["games", "--name", "twenty-million", "--games", "20000000", "--dir", str(app.games_dir),
                             "--opening-plies"]
    assert [s["name"] for s in ok(app, "GET", "/api/gamesets")["sets"]] == ["twenty-million"]
    calls = len(app.spawned.calls)
    for body, message in (({"name": "twenty-million", "games": 30000000}, "already being built"),
                          ({"name": "bad name", "games": 10}, "letters"), ({"games": 10}, "name"),
                          ({"name": "x", "games": 0}, "between 1 and"), ({"name": "x", "games": 500_000_001}, "between"),
                          ({"name": "x"}, "whole number"), ({"name": "x", "games": 10, "opening_plies": 30}, "opening_plies")):
        status, data = api(app, "POST", "/api/gamesets", body)
        assert status == 400 and message in data["error"], (body, data)
    assert len(app.spawned.calls) == calls


def test_starting_a_set_needs_fitted_leave_values(app):
    app.state.loaded = dataclasses.replace(app.state.loaded, leaves_path=None)
    status, data = api(app, "POST", "/api/gamesets", {"name": "x", "games": 10})
    assert status == 400 and "fit-leaves" in data["error"] and app.spawned.calls == []


def test_the_stop_route(app):
    make_set(app.games_dir, "live", app.loaded, 20, target=100, pid=os.getpid())
    assert ok(app, "POST", "/api/gamesets/live/stop", {}) == {"ok": True}
    assert ok(app, "GET", "/api/gamesets/live")["status"] == "stopping"
    status, data = api(app, "POST", "/api/gamesets/live/stop", {"force": "yes"})
    assert status == 400 and "force" in data["error"]
    assert ok(app, "POST", "/api/gamesets/live/stop", {"force": True}) == {"ok": True}
    assert app.spawned.killed == [os.getpid()]
    assert ok(app, "GET", "/api/gamesets/live")["status"] == "partial"
    status, data = api(app, "POST", "/api/gamesets/live/stop", {})
    assert status == 400 and "not being built" in data["error"]


def test_the_delete_route_refuses_while_building_or_trained_from(app):
    make_set(app.games_dir, "live", app.loaded, 20, target=100, pid=os.getpid())
    status, data = api(app, "DELETE", "/api/gamesets/live")
    assert status == 400 and "stop it first" in data["error"]
    used = make_set(app.games_dir, "used", app.loaded, 20)
    settings = {"offline.games": str(used)}
    make_run(app.runs_dir, "learning", records=[record("session", time.time(), event="start", mode="games")],
             settings=settings, pid=os.getpid())
    make_run(app.runs_dir, "waiting", queued={"queue": "q", "overrides": settings, "created": time.time()})
    status, data = api(app, "DELETE", "/api/gamesets/used")
    assert status == 400 and "learning" in data["error"] and "waiting" in data["error"]
    # Once those runs have stopped (or gone), it can go.
    (app.runs_dir / "learning" / "running.json").unlink()
    (app.runs_dir / "waiting" / "queued.json").unlink()
    assert ok(app, "DELETE", "/api/gamesets/used") == {"ok": True}
    assert not used.exists() and (app.runs_dir / TINY_RUN).is_dir()


def test_the_spawn_is_replaced_for_game_sets_too(spawned):
    """The fixture that keeps processes from starting covers the names gamesets.py imported."""
    assert gamesets_module.spawn_detached is spawned and gamesets_module.kill_process == spawned.kill
