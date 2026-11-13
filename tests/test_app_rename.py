"""Renaming a training run, `POST /api/runs/<name>/rename` (docs/APP_API.md): the folder moves,
config.toml takes the new name, and the stars and notes of the run's models, the matches they
played and games in progress follow it; every refusal leaves everything as it was. On temp
folders, with nothing started detached and every network on the CPU."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scrabble_ai import cli
from scrabble_ai import train as train_module
from scrabble_ai.app import runs as runs_module
from scrabble_ai.app.matches import write_json
from scrabble_ai.app.players import name as player_name
from scrabble_ai.app.runs import RunError, RunManager
from scrabble_ai.app.server import AppState, create_server

pytest.register_assert_rewrite("app_fixtures")
from app_fixtures import DEAD_PID, TINY_RUN, copy_run, make_run, record  # noqa: E402

pytest_plugins = ["app_fixtures"]

NEW = "sixty-four-run-1"


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
        yield SimpleNamespace(state=state, port=server.server_address[1], runs_dir=runs_dir, app_dir=app_dir,
                              spawned=spawned, trained=trained)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)
        assert not thread.is_alive()


@pytest.fixture
def dirs(tmp_path, spawned):
    runs_dir, app_dir = tmp_path / "runs", tmp_path / "app"
    runs_dir.mkdir()
    app_dir.mkdir()
    return runs_dir, app_dir


@pytest.fixture
def manager(dirs):
    return RunManager(*dirs)


def api(app, method: str, path: str, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", app.port, timeout=60)
    try:
        data = None if body is None else json.dumps(body).encode("utf-8")
        connection.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        response = connection.getresponse()
        status, content = response.status, response.read()
    finally:
        connection.close()
    data = json.loads(content)
    if status >= 400:
        assert set(data) == {"error"} and isinstance(data["error"], str) and data["error"], (path, data)
    return status, data


def ok(app, method: str, path: str, body=None):
    status, data = api(app, method, path, body)
    assert status == 200, (method, path, data)
    return data


def rename(app, old: str, new: str):
    return api(app, "POST", f"/api/runs/{old}/rename", {"name": new})


def step_files(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "checkpoints").glob("step_*.pt"))


def craft_match(app_dir: Path, match_id: str, a: str, b: str, state: str = "finished", pid: int = DEAD_PID) -> Path:
    """A match folder as `Matches.start` and the job leave it (created ten minutes ago)."""
    folder = app_dir / "matches" / match_id
    folder.mkdir(parents=True)
    created = time.time() - 600
    write_json(folder / "spec.json", {"id": match_id, "a": a, "b": b, "a_name": player_name(a), "b_name": player_name(b),
                                      "pairs": 2, "candidates": 64, "created": created, "runs_dir": "x",
                                      "first_seed": 8_000_000_000_000})
    write_json(folder / "status.json", {"status": state, "games_done": 4 if state == "finished" else 1, "started": created,
                                        "seconds": 1.5, "error": None})
    write_json(folder / "pid.json", {"pid": pid})
    return folder


def spec_of(folder: Path) -> dict:
    return json.loads((folder / "spec.json").read_text(encoding="utf-8"))


def folders(runs_dir: Path) -> list[str]:
    return sorted(p.name for p in runs_dir.iterdir())


# ---- the whole route on a genuine run ----------------------------------------------------------


def test_rename_a_genuine_run(app):
    old_dir, new_dir = app.runs_dir / TINY_RUN, app.runs_dir / NEW
    config_before = (old_dir / "config.toml").read_bytes()
    assert config_before.count(f'name = "{TINY_RUN}"'.encode()) == 1
    files_before = sorted(p.relative_to(old_dir).as_posix() for p in old_dir.rglob("*"))
    step = step_files(old_dir)[0]
    ok(app, "POST", f"/api/models/{TINY_RUN}/best.pt/meta", {"starred": True, "note": "really a 6x64"})
    ok(app, "POST", f"/api/models/{TINY_RUN}/{step}/meta", {"note": "a checkpoint"})
    metrics = ok(app, "GET", f"/api/runs/{TINY_RUN}/metrics")  # cached under the old name now
    summary_before = ok(app, "GET", f"/api/runs/{TINY_RUN}")["run"]
    models_before = {m["file"]: m for m in ok(app, "GET", "/api/models")["models"]}

    summary = ok(app, "POST", f"/api/runs/{TINY_RUN}/rename", {"name": NEW})

    # The summary, under the new name, of the same run.
    assert summary["name"] == NEW
    for key in ("status", "network", "parameters", "games", "step", "best_step", "has_best", "checkpoints", "changed"):
        assert summary[key] == summary_before[key], key
    assert [r["name"] for r in ok(app, "GET", "/api/runs")["runs"]] == [NEW]
    # The folder moved, with everything in it.
    assert folders(app.runs_dir) == [NEW]
    assert sorted(p.relative_to(new_dir).as_posix() for p in new_dir.rglob("*")) == files_before
    # config.toml: the run's name changed, and not one other byte.
    config_after = (new_dir / "config.toml").read_bytes()
    assert config_after == config_before.replace(f'name = "{TINY_RUN}"'.encode(), f'name = "{NEW}"'.encode())
    expected = tomllib.loads(config_before.decode("utf-8"))
    expected["run"]["name"] = NEW
    assert tomllib.loads(config_after.decode("utf-8")) == expected
    assert ok(app, "GET", f"/api/runs/{NEW}")["config"]["run"]["name"] == NEW
    # The old name is gone everywhere.
    for method, path in (("GET", f"/api/runs/{TINY_RUN}"), ("GET", f"/api/runs/{TINY_RUN}/metrics"),
                         ("GET", f"/api/runs/{TINY_RUN}/log"), ("POST", f"/api/runs/{TINY_RUN}/rename")):
        assert api(app, method, path, {"name": "again"} if method == "POST" else None)[0] == 404, path
    # The models are listed under the new name, with their stars and notes.
    models = {m["file"]: m for m in ok(app, "GET", "/api/models")["models"]}
    assert models.keys() == models_before.keys() and {m["run"] for m in models.values()} == {NEW}
    for file, before in models_before.items():
        assert models[file] == {**before, "id": f"{NEW}/{file}", "spec": f"model:{NEW}/{file}", "run": NEW,
                                "name": player_name(f"model:{NEW}/{file}")}, file
    assert (models["best.pt"]["name"], models["best.pt"]["starred"], models["best.pt"]["note"]) == (
        f"{NEW} (best)", True, "really a 6x64")
    assert models[step]["note"] == "a checkpoint" and models[step]["starred"] is False
    library = json.loads((app.app_dir / "library.json").read_text(encoding="utf-8"))
    assert set(library) == {f"{NEW}/best.pt", f"{NEW}/{step}"}
    # The same records, under the new name.
    assert ok(app, "GET", f"/api/runs/{NEW}/metrics") == metrics
    assert ok(app, "GET", f"/api/runs/{NEW}")["best"]["model_id"] == f"{NEW}/best.pt"
    # A network of the renamed run can be loaded and played.
    game = ok(app, "POST", "/api/games", {"players": [f"model:{NEW}/best.pt", "bot:greedy"], "seed": 7})
    assert game["players"][0] == {"spec": f"model:{NEW}/best.pt", "name": f"{NEW} (best)"}
    moved = ok(app, "POST", f"/api/games/{game['id']}/ai")
    assert moved["state"]["turn"] == 1 and moved["analysis"]["player"] == f"{NEW} (best)"


def test_a_game_in_progress_follows_the_rename(app):
    """A game names its players by spec and looks the network up at every move: without
    following the rename, its next AI move would find the old file gone."""
    game = ok(app, "POST", "/api/games", {"players": [f"model:{TINY_RUN}/best.pt", "bot:greedy"], "seed": 11})["id"]
    other = ok(app, "POST", "/api/games", {"players": ["human", "bot:random"], "seed": 12})
    ok(app, "POST", f"/api/games/{game}/ai")
    ok(app, "POST", f"/api/runs/{TINY_RUN}/rename", {"name": NEW})
    state = ok(app, "GET", f"/api/games/{game}")
    assert state["players"] == [{"spec": f"model:{NEW}/best.pt", "name": f"{NEW} (best)"},
                                {"spec": "bot:greedy", "name": "Greedy"}]
    assert state["advisor"] == f"model:{NEW}/best.pt"
    ok(app, "POST", f"/api/games/{game}/ai")  # Greedy
    response = ok(app, "POST", f"/api/games/{game}/ai")  # the network again, from its new path
    assert response["state"]["turn"] == 3 and response["analysis"]["player"] == f"{NEW} (best)"
    assert ok(app, "GET", f"/api/games/{game}/hint?top=3")["advisor"] == f"{NEW} (best)"
    assert ok(app, "GET", f"/api/games/{other['id']}") == other, "a game without the run is untouched"


# ---- matches -----------------------------------------------------------------------------------


def test_matches_follow_the_rename(app):
    step = step_files(app.runs_dir / TINY_RUN)[0]
    played = craft_match(app.app_dir, "m_20261101_100000", f"model:{TINY_RUN}/best.pt", "bot:greedy")
    second = craft_match(app.app_dir, "m_20261101_100001", "bot:random", f"model:{TINY_RUN}/{step}", state="cancelled")
    mirror = craft_match(app.app_dir, "m_20261101_100002", f"model:{TINY_RUN}/best.pt", f"model:{TINY_RUN}/{step}")
    cased = craft_match(app.app_dir, "m_20261101_100003", "bot:greedy", "model:TINY/best.pt")  # as Windows finds it
    died = craft_match(app.app_dir, "m_20261101_100004", f"model:{TINY_RUN}/best.pt", "bot:greedy", state="running")
    other = craft_match(app.app_dir, "m_20261101_100005", "model:tinyish/best.pt", "bot:greedy-leave")
    bots = craft_match(app.app_dir, "m_20261101_100006", "bot:greedy", "bot:random")
    before = {folder: spec_of(folder) for folder in (played, second, mirror, cased, died)}
    untouched = {folder: (folder / "spec.json").read_bytes() for folder in (other, bots)}

    ok(app, "POST", f"/api/runs/{TINY_RUN}/rename", {"name": NEW})

    best, checkpoint = f"model:{NEW}/best.pt", f"model:{NEW}/{step}"
    best_name, checkpoint_name = f"{NEW} (best)", f"{NEW} step {int(step[5:14]):,}"
    expected = {
        played: {"a": best, "a_name": best_name, "b": "bot:greedy", "b_name": "Greedy"},
        second: {"a": "bot:random", "a_name": "Random", "b": checkpoint, "b_name": checkpoint_name},
        mirror: {"a": best, "a_name": best_name, "b": checkpoint, "b_name": checkpoint_name},
        cased: {"a": "bot:greedy", "a_name": "Greedy", "b": best, "b_name": best_name},
        died: {"a": best, "a_name": best_name, "b": "bot:greedy", "b_name": "Greedy"},  # its job is gone
    }
    for folder, sides in expected.items():
        assert spec_of(folder) == {**before[folder], **sides}, folder.name
    for folder, content in untouched.items():
        assert (folder / "spec.json").read_bytes() == content, folder.name
    listed = {m["id"]: m for m in ok(app, "GET", "/api/matches")["matches"]}
    assert (listed[played.name]["a"], listed[played.name]["a_name"]) == (best, best_name)
    detail = ok(app, "GET", f"/api/matches/{mirror.name}")["match"]
    assert (detail["a_name"], detail["b_name"]) == (best_name, checkpoint_name)
    # A game of the match can still be replayed from the players its spec now names.
    spec = spec_of(played)
    game = ok(app, "POST", "/api/games", {"players": [spec["a"], spec["b"]], "seed": spec["first_seed"]})
    assert ok(app, "POST", f"/api/games/{game['id']}/ai")["state"]["turn"] == 1


def test_a_running_match_blocks_the_rename(app):
    running = craft_match(app.app_dir, "m_20261101_110000", "bot:greedy", f"model:{TINY_RUN}/best.pt",
                          state="running", pid=os.getpid())
    others = craft_match(app.app_dir, "m_20261101_110001", "bot:greedy", "bot:random", state="running", pid=os.getpid())
    spec_before = (running / "spec.json").read_bytes()
    status, data = rename(app, TINY_RUN, NEW)
    assert status == 400 and "match" in data["error"] and "still running" in data["error"], data
    assert folders(app.runs_dir) == [TINY_RUN] and (running / "spec.json").read_bytes() == spec_before
    # Once it has finished the rename goes ahead; a running match of other players never mattered.
    write_json(running / "status.json", {"status": "finished", "games_done": 4, "started": time.time(), "seconds": 9.0})
    ok(app, "POST", f"/api/runs/{TINY_RUN}/rename", {"name": NEW})
    assert spec_of(running)["b"] == f"model:{NEW}/best.pt" and spec_of(others)["b"] == "bot:random"


# ---- refusals ----------------------------------------------------------------------------------


def test_refusals(app):
    make_run(app.runs_dir, "other", records=[record("session", 1, event="stop", finished=True)])
    make_run(app.runs_dir, "live", records=[record("session", time.time(), event="start")], pid=os.getpid())
    make_run(app.runs_dir, "closing", records=[record("session", time.time(), event="start")], pid=os.getpid(), stop=True)
    make_run(app.runs_dir, "waiting", queued={"queue": "q", "overrides": {}, "created": time.time()})
    ok(app, "POST", f"/api/models/{TINY_RUN}/best.pt/meta", {"starred": True})
    played = craft_match(app.app_dir, "m_20261101_120000", f"model:{TINY_RUN}/best.pt", "bot:greedy")
    listing = folders(app.runs_dir)
    library = (app.app_dir / "library.json").read_bytes()
    config = (app.runs_dir / TINY_RUN / "config.toml").read_bytes()
    match_spec = (played / "spec.json").read_bytes()
    tiny = f"/api/runs/{TINY_RUN}/rename"
    cases = [
        ("/api/runs/nope/rename", {"name": "fresh"}, 404, "nope"),
        ("/api/runs/%2e%2e/rename", {"name": "fresh"}, 404, ""),
        ("/api/runs/bad.name/rename", {"name": "fresh"}, 404, ""),
        ("/api/runs/live/rename", {"name": "fresh"}, 400, "live is running"),
        ("/api/runs/closing/rename", {"name": "fresh"}, 400, "closing is stopping"),
        ("/api/runs/waiting/rename", {"name": "fresh"}, 400, "waiting is queued"),
        (tiny, {"name": "other"}, 400, "already exists"),
        (tiny, {"name": "OTHER"}, 400, "'other' already exists"),  # Windows folder names ignore case
        (tiny, {"name": "Live"}, 400, "already exists"),
        (tiny, {"name": TINY_RUN}, 400, "already has that name"),
        (tiny, {}, 400, "new name"),
        (tiny, None, 400, "new name"),
        (tiny, {"name": 5}, 400, "new name"),
        (tiny, {"name": None}, 400, "new name"),
        (tiny, {"name": ["fresh"]}, 400, "new name"),
        (tiny, {"hours": 3}, 400, "new name"),
    ]
    for bad in ("", "bad name", " fresh", "fresh ", "-fresh", "_fresh", "a.b", "..", ".", "x/y", "x\\y", "naïve", "x" * 65,
                "fresh\n", "C:fresh"):
        cases.append((tiny, {"name": bad}, 400, "letters, digits"))
    for path, body, expected, message in cases:
        status, data = api(app, "POST", path, body)
        assert status == expected and message in data["error"], (path, body, status, data)
    assert folders(app.runs_dir) == listing
    assert (app.app_dir / "library.json").read_bytes() == library
    assert (app.runs_dir / TINY_RUN / "config.toml").read_bytes() == config
    assert (played / "spec.json").read_bytes() == match_spec
    assert app.spawned.calls == [] and app.spawned.killed == []
    # The longest name allowed is fine.
    assert ok(app, "POST", tiny, {"name": "x" * 64})["name"] == "x" * 64


def test_a_change_of_case_only(app):
    ok(app, "POST", f"/api/models/{TINY_RUN}/best.pt/meta", {"starred": True, "note": "capital"})
    played = craft_match(app.app_dir, "m_20261101_130000", f"model:{TINY_RUN}/best.pt", "bot:greedy")
    upper = TINY_RUN.capitalize()
    assert ok(app, "POST", f"/api/runs/{TINY_RUN}/rename", {"name": upper})["name"] == upper
    assert folders(app.runs_dir) == [upper], "no temporary folder left behind either"
    assert [r["name"] for r in ok(app, "GET", "/api/runs")["runs"]] == [upper]
    assert tomllib.loads((app.runs_dir / upper / "config.toml").read_text(encoding="utf-8"))["run"]["name"] == upper
    best = next(m for m in ok(app, "GET", "/api/models")["models"] if m["file"] == "best.pt")
    assert (best["id"], best["starred"], best["note"]) == (f"{upper}/best.pt", True, "capital")
    assert set(json.loads((app.app_dir / "library.json").read_text(encoding="utf-8"))) == {f"{upper}/best.pt"}
    assert spec_of(played)["a"] == f"model:{upper}/best.pt"
    # And back again.
    assert ok(app, "POST", f"/api/runs/{upper}/rename", {"name": TINY_RUN})["name"] == TINY_RUN
    assert folders(app.runs_dir) == [TINY_RUN] and spec_of(played)["a_name"] == f"{TINY_RUN} (best)"


# ---- runs that never started, resuming, and the old name afterwards --------------------------------


def test_runs_that_never_started(app, test_lexicon):
    # Taken out of a queue: no config.toml, its settings in unstarted.json.
    held = make_run(app.runs_dir, "held")
    (held / "unstarted.json").write_text(json.dumps({"overrides": {"model.blocks": 2, "run.hours": 1.5}}), encoding="utf-8")
    # Its queue died before starting it: queued.json is still there, and the run reads as stopped.
    make_run(app.runs_dir, "orphan", queued={"queue": "queue_20261101_000000", "overrides": {"model.channels": 32},
                                             "created": time.time() - 3600})
    before = {name: ok(app, "GET", f"/api/runs/{name}")["run"] for name in ("held", "orphan")}
    assert before["orphan"]["status"] == "stopped"

    for old, new in (("held", "held-2"), ("orphan", "orphan-2")):
        summary = ok(app, "POST", f"/api/runs/{old}/rename", {"name": new})
        assert summary["name"] == new and not (app.runs_dir / old).exists()
        for key in ("status", "network", "changed", "limit"):
            assert summary[key] == before[old][key], (old, key)
        assert not (app.runs_dir / new / "config.toml").exists()
    assert json.loads((app.runs_dir / "held-2" / "unstarted.json").read_text(encoding="utf-8")) == {
        "overrides": {"model.blocks": 2, "run.hours": 1.5}}
    assert json.loads((app.runs_dir / "orphan-2" / "queued.json").read_text(encoding="utf-8"))["overrides"] == {
        "model.channels": 32}

    # Resuming starts each as the run that was asked for, under its new name.
    for new, setting in (("held-2", "model.blocks=2"), ("orphan-2", "model.channels=32")):
        assert ok(app, "POST", f"/api/runs/{new}/resume", {})["status"] == "running"
        command, log = app.spawned.calls[-1]
        assert command[command.index("--name") + 1] == new and setting in command, command
        assert log == app.runs_dir / new / "console.log"
    assert not (app.runs_dir / "orphan-2" / "queued.json").exists()


def test_resume_after_a_rename_uses_the_new_name(app, monkeypatch):
    ok(app, "POST", f"/api/runs/{TINY_RUN}/rename", {"name": NEW})
    summary = ok(app, "POST", f"/api/runs/{NEW}/resume", {"hours": 1})
    assert summary["name"] == NEW and summary["status"] == "running"
    command, log = app.spawned.calls[-1]
    assert command[command.index("--name") + 1] == NEW and command[-2:] == ["--resume", "latest"]
    assert f"run.out_dir={app.runs_dir}" in command and log == app.runs_dir / NEW / "console.log"

    # The checkpoints still hold the old name, but the trainer that command starts uses the new
    # one: run the command's own arguments through the CLI, with a Trainer that only listens.
    latest = (app.runs_dir / NEW / "checkpoints" / "latest.txt").read_text(encoding="utf-8").strip()
    assert train_module.load_checkpoint_config(app.runs_dir / NEW / "checkpoints" / latest).run.name == TINY_RUN
    seen = {}

    class ListeningTrainer:
        def __init__(self, config, resume=None, quiet=False):
            seen.update(config=config, resume=resume)

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(train_module, "Trainer", ListeningTrainer)
    assert command[2:5] == ["-m", "scrabble_ai.cli", "train"]
    assert cli.main(command[4:]) == 0 and seen["ran"]
    assert seen["config"].run.name == NEW and Path(seen["config"].run.out_dir) == app.runs_dir
    assert seen["resume"] == app.runs_dir / NEW / "checkpoints" / latest


def test_a_new_run_can_take_the_old_name(app, test_lexicon):
    ok(app, "POST", f"/api/models/{TINY_RUN}/best.pt/meta", {"starred": True, "note": "moves with the run"})
    old_records = ok(app, "GET", f"/api/runs/{TINY_RUN}/metrics")["records"]
    ok(app, "POST", f"/api/runs/{TINY_RUN}/rename", {"name": NEW})
    assert ok(app, "POST", "/api/runs", {"runs": [{"name": TINY_RUN}]}) == {"started": [TINY_RUN], "queued": []}
    assert app.spawned.calls[-1][0][app.spawned.calls[-1][0].index("--name") + 1] == TINY_RUN
    assert ok(app, "GET", f"/api/runs/{TINY_RUN}/metrics") == {"records": [], "next": 0}
    assert ok(app, "GET", f"/api/runs/{TINY_RUN}")["run"]["status"] == "running"
    # Its trainer writes a log of its own: none of the renamed run's records come with it.
    (app.runs_dir / TINY_RUN / "metrics.jsonl").write_text(
        "".join(json.dumps(record("session", time.time() + n, event="start")) + "\n" for n in range(len(old_records) + 5)),
        encoding="utf-8")
    fresh = ok(app, "GET", f"/api/runs/{TINY_RUN}/metrics")["records"]
    assert len(fresh) == len(old_records) + 5 and {r["kind"] for r in fresh} == {"session"}
    assert ok(app, "GET", f"/api/runs/{NEW}/metrics")["records"] == old_records
    models = ok(app, "GET", "/api/models")["models"]
    assert {m["run"] for m in models} == {NEW} and next(m for m in models if m["file"] == "best.pt")["starred"] is True
    assert set(json.loads((app.app_dir / "library.json").read_text(encoding="utf-8"))) == {f"{NEW}/best.pt"}


# ---- RunManager.rename on its own ----------------------------------------------------------------


def test_the_metrics_cache_forgets_both_names(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "a", records=[record("train", n, step=n) for n in range(1, 9)])
    make_run(runs_dir, "b", records=[record("train", 1, step=100)])
    assert len(manager.records("a")) == 8 and len(manager.records("b")) == 1
    shutil.rmtree(runs_dir / "b")  # deleted by hand: its records stay in the cache
    manager.rename("a", "b")
    assert [r["step"] for r in manager.records("b")] == list(range(1, 9))
    make_run(runs_dir, "a", records=[record("selfplay", 50 + n, games=n) for n in range(12)])
    assert [r["games"] for r in manager.records("a")] == list(range(12))


def test_a_config_that_cannot_be_written_puts_the_folder_back(manager, dirs, monkeypatch):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", records=[record("session", 1, event="stop", finished=True)], settings={"run.hours": 2.0})
    config = (runs_dir / "r" / "config.toml").read_bytes()
    real_replace = os.replace

    def replace(source, target):
        if Path(target).name == "config.toml":
            raise PermissionError("the file is in use")
        return real_replace(source, target)

    monkeypatch.setattr(runs_module.os, "replace", replace)
    with pytest.raises(RunError, match="r was not renamed: its config.toml could not be written"):
        manager.rename("r", "r2")
    assert folders(runs_dir) == ["r"]
    assert sorted(p.name for p in (runs_dir / "r").iterdir()) == ["config.toml", "metrics.jsonl"]
    assert (runs_dir / "r" / "config.toml").read_bytes() == config


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to rename a folder with a file open in it")
def test_a_folder_with_a_file_open_is_retried_then_refused(manager, dirs, monkeypatch):
    runs_dir, _ = dirs
    make_run(runs_dir, "closing", records=[record("session", 1, event="stop", finished=True)], console=["done."])
    sleeps = []
    monkeypatch.setattr(runs_module.time, "sleep", sleeps.append)
    with open(runs_dir / "closing" / "console.log", "ab"):  # as a trainer still closing down holds its log
        with pytest.raises(RunError, match="closing still has files open"):
            manager.rename("closing", "closed")
        assert len(sleeps) == 19 and folders(runs_dir) == ["closing"]
    manager.rename("closing", "closed")
    assert folders(runs_dir) == ["closed"]


def test_rename_refuses_without_the_server_too(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", records=[record("session", 1, event="stop", finished=True)])
    for old, new, message in (("nope", "x", "no run"), ("r", "r", "has that name"), ("r", "x y", "letters"),
                              ("r", 5, "text"), ("r", None, "text")):
        with pytest.raises(RunError, match=message):
            manager.rename(old, new)
    assert folders(runs_dir) == ["r"]


def test_renamed_config_keeps_every_other_byte():
    renamed = runs_module._renamed_config
    text = '# settings\r\n[run]\r\nname = "old"   # the run\r\nlexicon = "TEST"\r\n\r\n[model]\r\nname = "kept"\r\nblocks = 6\r\n'
    assert renamed(text, "new") == text.replace('name = "old"', 'name = "new"', 1)
    assert renamed("[run]\nseed = 3\nname = 'old'\n[train]\nlr = 0.001\n", "N-1") == '[run]\nseed = 3\nname = "N-1"\n[train]\nlr = 0.001\n'
    assert renamed('[model]\nname = "x"\n[run]\nseed = 3\n', "new") is None, "only the name under [run] is the run's"
    assert renamed('name = "top"\n[model]\nblocks = 6\n', "new") is None
    assert renamed('[run]\nname = "old"\nname = "twice"\n', "new") is None, "not TOML before, so not touched"
    assert renamed('[run]\nname = """old"""\n', "new") is None
