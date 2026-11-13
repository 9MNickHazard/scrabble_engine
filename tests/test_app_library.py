"""`scrabble_ai.app.players` and `scrabble_ai.app.library`: player specs, loading networks, and the
model library, on a genuine (tiny) run folder."""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

from scrabble_ai.app import players as players_module
from scrabble_ai.app.library import Library
from scrabble_ai.app.players import NotFound, PlayerError
from scrabble_ai.app.runs import RunManager

pytest.register_assert_rewrite("app_fixtures")
from app_fixtures import TINY_RUN, copy_run, record, write_records  # noqa: E402

pytest_plugins = ["app_fixtures"]

BEST = f"model:{TINY_RUN}/best.pt"


@pytest.fixture
def runs_dir(tmp_path, trained):
    directory = tmp_path / "runs"
    directory.mkdir()
    copy_run(trained, directory)
    return directory


@pytest.fixture
def library(runs_dir, tmp_path, trained):
    return Library(RunManager(runs_dir, tmp_path / "app"), trained.loaded, tmp_path / "app")


def step_files(runs_dir, run=TINY_RUN):
    return sorted(p.name for p in (runs_dir / run / "checkpoints").glob("step_*.pt"))


# ---- specs -------------------------------------------------------------------------------------


def test_names_and_kinds():
    assert (players_module.kind("human"), players_module.name("human")) == ("human", "You")
    assert players_module.name("bot:greedy-leave") == "Greedy + leave"
    assert players_module.name("bot:greedy") == "Greedy"
    assert players_module.name("bot:random") == "Random"
    assert {players_module.kind(s) for s in players_module.BOTS} == {"bot"}
    assert players_module.bot_name("bot:greedy-leave") == "greedy-leave"
    assert players_module.name("model:night1/best.pt") == "night1 (best)"
    assert players_module.name("model:night1/step_000000123.pt") == "night1 step 123"
    assert players_module.name("model:night1/step_001234567.pt") == "night1 step 1,234,567"
    assert players_module.split_model("model:night1/best.pt") == ("night1", "best.pt")
    assert players_module.split_model("night1/best.pt") == ("night1", "best.pt"), "a bare model id works too"


@pytest.mark.parametrize("spec", ["", "Human", "bot:", "bot:nope", "bot:greedy_leave", "greedy", "model", "robot:greedy",
                                  " human"])
def test_unknown_players_are_refused(spec):
    with pytest.raises(PlayerError, match="unknown player"):
        players_module.kind(spec)


@pytest.mark.parametrize("spec", [None, 5, ["human"], {"spec": "human"}, b"human"])
def test_a_player_that_is_not_a_string_is_refused(spec):
    with pytest.raises(PlayerError, match="named by a string"):
        players_module.kind(spec)


def test_not_found_is_a_key_error_with_a_plain_message():
    error = players_module.NotFound("there is no game g_1")
    assert isinstance(error, KeyError) and str(error) == "there is no game g_1"
    assert str(players_module.NotFound()) == "not found"


@pytest.mark.parametrize("spec", ["model:", "model:night1", "model:night1/", "model:/best.pt", "model:../x/best.pt",
                                  "model:run/../../best.pt", "model:run/other.pt", "model:run/best.pt/extra",
                                  "model:run/sub/best.pt", "model:run/step_123.pt", "model:run/step_0000001234.pt",
                                  "model:run\\..\\best.pt", "model:..\\x/best.pt", "model:C:/best.pt", "model:run/BEST.PT",
                                  "model:run/best.pt.bak", "model:run/.best.pt", "model:-run/best.pt",
                                  "model:run/../run/best.pt"])
def test_malformed_model_ids_are_refused(spec):
    with pytest.raises(PlayerError, match="not a model id"):
        players_module.split_model(spec)
    with pytest.raises(PlayerError):
        players_module.name(spec)


@pytest.mark.parametrize("spec", ["model:run/best.pt\n", "model:run\n/best.pt", "model:run/step_000000001.pt\n"])
def test_a_model_id_with_a_trailing_newline_is_refused(spec):
    """RUN_NAME and MODEL_FILE once ended in `$`, which also matches before a trailing newline."""
    with pytest.raises(PlayerError, match="not a model id"):
        players_module.split_model(spec)


def test_model_path(runs_dir, cpu_players):
    players = cpu_players(runs_dir)
    assert players.model_path(BEST) == runs_dir / TINY_RUN / "checkpoints" / "best.pt"
    with pytest.raises(PlayerError, match="no longer exists"):
        players.model_path(f"model:{TINY_RUN}/step_000999999.pt")
    with pytest.raises(PlayerError, match="no longer exists"):
        players.model_path("model:nosuchrun/best.pt")


def test_players_default_device_is_a_torch_device(trained, tmp_path):
    players = players_module.Players(trained.loaded, tmp_path)
    assert isinstance(players.device, torch.device) and players.device.type in ("cuda", "cpu")


# ---- loading networks --------------------------------------------------------------------------


def test_scorer_is_cached_and_reloaded_when_the_file_changes(runs_dir, cpu_players, fast_scorer):
    players = cpu_players(runs_dir)
    first = players.scorer(BEST)
    assert players.scorer(BEST) is first
    path = players.model_path(BEST)
    later = path.stat().st_mtime + 10
    os.utime(path, (later, later))
    second = players.scorer(BEST)
    assert second is not first, "a promoted best.pt is picked up"
    assert players.scorer(BEST) is second
    assert next(second.net.parameters()).device.type == "cpu"


def test_the_cache_keeps_only_the_most_recent(runs_dir, cpu_players, fast_scorer):
    players = cpu_players(runs_dir, cache_size=1)
    step = f"model:{TINY_RUN}/{step_files(runs_dir)[-1]}"
    best = players.scorer(BEST)
    players.scorer(step)
    assert len(players._scorers) == 1
    assert players.scorer(BEST) is not best, "evicted, so loaded again"


def doctored_copy(runs_dir, trained, name, **changes):
    """A copy of the run whose best.pt claims other training data."""
    target = copy_run(trained, runs_dir, name)
    path = target / "checkpoints" / "best.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    state.update(changes)
    torch.save(state, path)
    return f"model:{name}/best.pt"


def test_a_network_from_another_word_list_or_leave_values_is_refused(runs_dir, trained, cpu_players, fast_scorer):
    players = cpu_players(runs_dir)
    other_lexicon = doctored_copy(runs_dir, trained, "otherlex", lexicon_id="OTHER-123456789abc")
    with pytest.raises(PlayerError, match="word list OTHER-123456789abc"):
        players.scorer(other_lexicon)
    other_leaves = doctored_copy(runs_dir, trained, "otherleaves", leaves_hash="000000000000")
    with pytest.raises(PlayerError, match="different leave values"):
        players.scorer(other_leaves)
    assert players.scorer(BEST) is not None, "the genuine one still loads"


# ---- the library -------------------------------------------------------------------------------


def evals_at(records, step):
    return {r["opponent"]: {"spread": r["spread"], "ci": r["spread_ci"], "win_rate": r["win_rate"]}
            for r in records if r["kind"] == "eval" and r["step"] == step}


def test_models_of_a_genuine_run(library, runs_dir, trained):
    result = library.models()
    assert [b["spec"] for b in result["bots"]] == ["bot:greedy-leave", "bot:greedy", "bot:random"]
    assert all(b["name"] and b["description"] for b in result["bots"])
    models = {m["id"]: m for m in result["models"]}
    files = step_files(runs_dir)
    assert set(models) == {f"{TINY_RUN}/best.pt"} | {f"{TINY_RUN}/{f}" for f in files}
    index = json.loads((runs_dir / TINY_RUN / "checkpoints" / "index.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (runs_dir / TINY_RUN / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]

    best = models[f"{TINY_RUN}/best.pt"]
    assert set(best) == {"id", "spec", "name", "run", "file", "kind", "network", "parameters", "step", "games", "modified",
                         "size_bytes", "compatible", "incompatible_reason", "eval", "starred", "note"}
    assert all(m["incompatible_reason"] is None for m in result["models"])
    assert (best["spec"], best["name"], best["run"], best["file"], best["kind"]) == (
        BEST, f"{TINY_RUN} (best)", TINY_RUN, "best.pt", "best")
    assert best["step"] == trained.best_step == index["best"]["step"]
    assert best["network"] == "1x16" and best["parameters"] == index["parameters"] > 0
    assert best["games"] == [r for r in records if r["step"] == trained.best_step][-1]["games"]
    assert best["compatible"] is True and best["starred"] is False and best["note"] == ""
    assert best["eval"] == evals_at(records, trained.best_step) and set(best["eval"]) == {"greedy-leave", "greedy"}
    assert best["size_bytes"] == (runs_dir / TINY_RUN / "checkpoints" / "best.pt").stat().st_size

    for file in files:
        entry = models[f"{TINY_RUN}/{file}"]
        step = int(file[5:14])
        assert (entry["kind"], entry["step"], entry["name"]) == ("checkpoint", step, f"{TINY_RUN} step {step:,}")
        assert entry["eval"] == (evals_at(records, step) or None)
        assert entry["size_bytes"] == (runs_dir / TINY_RUN / "checkpoints" / file).stat().st_size
    assert [m["modified"] for m in result["models"]] == sorted((m["modified"] for m in result["models"]), reverse=True)
    assert library.model(f"{TINY_RUN}/best.pt") == best
    with pytest.raises(NotFound, match="no model"):
        library.model(f"{TINY_RUN}/step_000999999.pt")


def test_models_without_index_json(library, runs_dir, trained):
    (runs_dir / TINY_RUN / "checkpoints" / "index.json").unlink()
    models = {m["id"]: m for m in library.models()["models"]}
    best = models[f"{TINY_RUN}/best.pt"]
    records = [json.loads(line) for line in (runs_dir / TINY_RUN / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    promoted = [r for r in records if r["kind"] == "gate" and r.get("promoted")]
    assert best["step"] == promoted[-1]["step"] == trained.best_step
    assert best["network"] == "1x16" and best["compatible"] is True and best["incompatible_reason"] is None
    assert best["size_bytes"] > 0 and best["modified"] > 0
    start = next(r for r in records if r["kind"] == "session" and r["event"] == "start")
    assert best["parameters"] == start["parameters"] > 0, "from the session start record when index.json is gone"
    assert best["eval"] is not None
    assert len(models) == 1 + len(step_files(runs_dir))


def test_compatibility_comes_from_index_json(library, runs_dir):
    path = runs_dir / TINY_RUN / "checkpoints" / "index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**index, "lexicon_id": "OTHER-000000000000"}), encoding="utf-8")
    models = library.models()["models"]
    assert {m["compatible"] for m in models} == {False}
    assert all("OTHER-000000000000" in m["incompatible_reason"] and library.loaded.lexicon_id in m["incompatible_reason"]
               for m in models)
    path.write_text(json.dumps({**index, "leaves_hash": "ffffffffffff"}), encoding="utf-8")
    models = library.models()["models"]
    assert {m["compatible"] for m in models} == {False}
    assert all("leave values" in m["incompatible_reason"] and "refitted" in m["incompatible_reason"] for m in models)
    path.write_text(json.dumps(index), encoding="utf-8")
    assert {(m["compatible"], m["incompatible_reason"]) for m in library.models()["models"]} == {(True, None)}


def test_runs_without_checkpoints_are_skipped(library, runs_dir):
    (runs_dir / "empty").mkdir()
    write_records(runs_dir / "empty" / "metrics.jsonl", [record("session", time.time(), event="start")])
    assert {m["run"] for m in library.models()["models"]} == {TINY_RUN}


def test_set_meta_persists_and_validates(library, runs_dir, trained, tmp_path):
    model_id = f"{TINY_RUN}/best.pt"
    entry = library.set_meta(model_id, starred=True)
    assert entry["starred"] is True and entry["note"] == ""
    entry = library.set_meta(model_id, note="first to beat the baseline")
    assert entry["starred"] is True and entry["note"] == "first to beat the baseline", "fields are kept independently"
    again = Library(RunManager(runs_dir, tmp_path / "app"), trained.loaded, tmp_path / "app")
    assert again.model(model_id)["starred"] is True and again.model(model_id)["note"] == "first to beat the baseline"
    assert library.set_meta(model_id, note="x" * 5000)["note"] == "x" * 2000
    assert library.set_meta(model_id, starred=False)["starred"] is False
    assert not (tmp_path / "app" / "library.json.tmp").exists()
    for bad in ("../x/best.pt", "tiny/other.pt", "tiny", "bot:greedy"):
        with pytest.raises(PlayerError):
            library.set_meta(bad, starred=True)
    with pytest.raises(NotFound, match="no model"):
        library.set_meta(f"{TINY_RUN}/step_000999999.pt", starred=True)


def test_set_meta_refuses_wrong_types_before_writing_anything(library, tmp_path):
    model_id = f"{TINY_RUN}/best.pt"
    for starred, note, message in (("yes", None, "starred must be true or false"), (1, None, "starred"),
                                   (None, 5, "note must be text"), (None, ["x"], "note"), (True, {"a": 1}, "note")):
        with pytest.raises(PlayerError, match=message):
            library.set_meta(model_id, starred=starred, note=note)
    assert not (tmp_path / "app" / "library.json").exists()


def test_set_meta_for_an_unknown_model_writes_nothing(library, tmp_path):
    """It once wrote library.json before finding out the model did not exist."""
    library.set_meta(f"{TINY_RUN}/best.pt", note="kept")
    before = (tmp_path / "app" / "library.json").read_text(encoding="utf-8")
    with pytest.raises(NotFound):
        library.set_meta(f"{TINY_RUN}/step_000999999.pt", starred=True)
    with pytest.raises(NotFound):
        library.set_meta("ghost/best.pt", note="x")
    assert (tmp_path / "app" / "library.json").read_text(encoding="utf-8") == before


def test_forget_run(library, runs_dir, trained, tmp_path):
    copy_run(trained, runs_dir, "other")
    library.set_meta(f"{TINY_RUN}/best.pt", starred=True, note="mine")
    step = step_files(runs_dir)[-1]
    library.set_meta(f"{TINY_RUN}/{step}", note="a step")
    library.set_meta("other/best.pt", starred=True, note="theirs")
    path = tmp_path / "app" / "library.json"
    library.forget_run(TINY_RUN)
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"other/best.pt"}
    assert library.model(f"{TINY_RUN}/best.pt")["starred"] is False and library.model(f"{TINY_RUN}/best.pt")["note"] == ""
    assert library.model("other/best.pt")["note"] == "theirs"
    mtime = path.stat().st_mtime_ns
    library.forget_run("never-existed")
    library.forget_run("oth")  # a prefix of another run's name is a different run
    assert path.stat().st_mtime_ns == mtime and library.model("other/best.pt")["starred"] is True
