"""`scrabble_ai.app.matches` and `match_job`: fast matches between bots and models, the job's
files, and the match list (nothing is started detached)."""

from __future__ import annotations

import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scrabble_ai import engine as engine_module
from scrabble_ai.app import match_job
from scrabble_ai.app import matches as matches_module
from scrabble_ai.app.matches import COLUMNS, Matches, play_match_detailed, quick_stats, write_json
from scrabble_ai.app.players import PlayerError
from scrabble_ai.app.runs import python_executable

pytest.register_assert_rewrite("app_fixtures")
from app_fixtures import DEAD_PID, TINY_RUN, copy_run, lexicon_entry_for  # noqa: E402

pytest_plugins = ["app_fixtures"]

BEST = f"model:{TINY_RUN}/best.pt"


@pytest.fixture(autouse=True)
def _no_real_processes(spawned):
    """spawn_detached is replaced in every test here."""


@pytest.fixture
def setup(tmp_path, trained, cpu_players, fast_scorer):
    runs_dir, app_dir = tmp_path / "runs", tmp_path / "app"
    runs_dir.mkdir()
    copy_run(trained, runs_dir)
    players = cpu_players(runs_dir)
    return SimpleNamespace(runs_dir=runs_dir, app_dir=app_dir, players=players, matches=Matches(app_dir, players),
                           engine=trained.loaded.engine, leaves=trained.loaded.leaves)


def seeds(n, first=5000):
    return np.arange(n, dtype=np.uint64) + np.uint64(first)


def check_columns(games: dict, pairs: int, engine) -> None:
    assert set(games) == set(COLUMNS)
    assert {len(column) for column in games.values()} == {2 * pairs}
    for k in range(pairs):
        assert games["seed"][2 * k] == games["seed"][2 * k + 1]
        assert (games["a_first"][2 * k], games["a_first"][2 * k + 1]) == (True, False)
    for i in range(2 * pairs):
        assert games["ended_by"][i] in ("out", "scoreless")
        assert games["turns"][i] >= 2
        moves = sum(games[f"{what}_{side}"][i] for what in ("bingos", "exchanges", "passes") for side in "ab")
        assert moves <= games["turns"][i]
        if games["ended_by"][i] == "scoreless":
            assert games["exchanges_a"][i] + games["exchanges_b"][i] + games["passes_a"][i] + games["passes_b"][i] >= 6
        for side in "ab":
            assert -200 < games[f"score_{side}"][i] < 2000
            best, word = games[f"best_play_{side}"][i], games[f"best_word_{side}"][i]
            assert (best > 0) == (word != ""), (i, side, best, word)
            if word:
                assert engine.is_word(word), word
                assert best <= games[f"score_{side}"][i] + 200
    json.dumps(games)  # plain Python values, ready for games.json


# ---- play_match_detailed -----------------------------------------------------------------------


def test_bots_agree_with_the_engine(setup):
    s = seeds(6)
    games = play_match_detailed(setup.engine, setup.leaves, {}, "bot:greedy-leave", "bot:greedy", s, None, slots=4)
    check_columns(games, 6, setup.engine)
    assert games["seed"] == [int(x) for x in np.repeat(s, 2)]
    reference = setup.engine.play_pairs("greedy-leave", setup.leaves, "greedy", setup.leaves, s)
    assert games["score_a"] == [int(x) for x in reference["score_a"]]
    assert games["score_b"] == [int(x) for x in reference["score_b"]]
    assert games["turns"] == [int(x) for x in reference["turns"]]
    spread = np.array(games["score_a"]) - np.array(games["score_b"])
    np.testing.assert_array_equal(spread[0::2], reference["spread_first"])
    np.testing.assert_array_equal(spread[1::2], reference["spread_second"])


def test_the_same_seeds_give_the_same_games_whatever_the_slots(setup):
    s = seeds(5)
    one = play_match_detailed(setup.engine, setup.leaves, {}, "bot:random", "bot:greedy", s, None, slots=1)
    many = play_match_detailed(setup.engine, setup.leaves, {}, "bot:random", "bot:greedy", s, None, slots=512)
    assert one == many


class FakeClock:
    """time.time() that moves on a second per call, so progress is reported on every pass."""

    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        self.now += 1.0
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


def test_progress_sees_only_complete_pairs(setup, monkeypatch):
    monkeypatch.setattr(matches_module, "time", FakeClock())
    calls = []
    s = seeds(7)
    games = play_match_detailed(setup.engine, setup.leaves, {}, "bot:greedy-leave", "bot:random", s, None, slots=3,
                                progress=lambda columns, finished: calls.append((json.loads(json.dumps(columns)), finished)))
    assert len(calls) >= 3, "progress is reported along the way, not only at the end"
    for columns, finished in calls:
        pairs = len(columns["seed"]) // 2
        assert len(columns["seed"]) == 2 * pairs and finished >= 2 * pairs
        assert all(len(column) == 2 * pairs for column in columns.values())
        for k in range(pairs):
            assert columns["seed"][2 * k] == columns["seed"][2 * k + 1]
            assert (columns["a_first"][2 * k], columns["a_first"][2 * k + 1]) == (True, False)
    assert any(0 < len(c["seed"]) < 14 for c, _ in calls), "some report came mid-match"
    assert calls[-1] == (games, 14)
    check_columns(games, 7, setup.engine)


def test_cancel_stops_early_with_complete_pairs_only(setup):
    s = seeds(40)
    state = {"checks": 0}

    def cancelled():
        state["checks"] += 1
        return state["checks"] > 3

    games = play_match_detailed(setup.engine, setup.leaves, {}, "bot:greedy-leave", "bot:greedy", s, None, slots=5,
                                cancelled=cancelled)
    pairs = len(games["seed"]) // 2
    assert 0 < pairs < 40 and len(games["seed"]) == 2 * pairs
    check_columns(games, pairs, setup.engine)
    reference = setup.engine.play_pairs("greedy-leave", setup.leaves, "greedy", setup.leaves, np.array(sorted(set(games["seed"])), np.uint64))
    assert sorted(zip(games["seed"], games["score_a"])) == sorted(zip(np.repeat(sorted(set(games["seed"])), 2).tolist(),
                                                                        [int(x) for x in reference["score_a"]]))


def test_a_model_against_a_bot_on_the_cpu(setup):
    cpu = torch.device("cpu")
    scorers = {BEST: setup.players.scorer(BEST)}
    games = play_match_detailed(setup.engine, setup.leaves, scorers, BEST, "bot:greedy-leave", seeds(3), cpu, slots=4)
    check_columns(games, 3, setup.engine)
    # Seats swapped: game 2k now has the bot first, which is game 2k+1 of the first match. Same games, so same scores
    # (and a different slot count does not change what the network plays).
    swapped = play_match_detailed(setup.engine, setup.leaves, scorers, "bot:greedy-leave", BEST, seeds(3), cpu)
    for k in range(3):
        assert swapped["score_a"][2 * k] == games["score_b"][2 * k + 1]
        assert swapped["score_b"][2 * k] == games["score_a"][2 * k + 1]
        assert swapped["score_a"][2 * k + 1] == games["score_b"][2 * k]
    model_vs_model = play_match_detailed(setup.engine, setup.leaves, scorers, BEST, BEST, seeds(2), cpu)
    check_columns(model_vs_model, 2, setup.engine)


@pytest.mark.parametrize("candidates", [2, 3, 4, 5])
def test_few_candidates_still_play(setup, candidates):
    """Every candidate count Matches.start accepts plays: exchanges are capped at candidates - 1 (a fixed 4
    once made BatchEnv refuse 4 or fewer, even between bots)."""
    games = play_match_detailed(setup.engine, setup.leaves, {}, "bot:greedy", "bot:random", seeds(2), None, candidates)
    assert len(games["seed"]) == 4
    cpu = torch.device("cpu")
    games = play_match_detailed(setup.engine, setup.leaves, {BEST: setup.players.scorer(BEST)}, BEST, "bot:random",
                                seeds(1), cpu, candidates)
    check_columns(games, 1, setup.engine)


# ---- quick_stats -------------------------------------------------------------------------------


def test_quick_stats_by_hand():
    # Three pairs. A's spreads: pair 0: +20, -10; pair 1: +5, +5; pair 2: -30, +10.
    games = {"score_a": [120, 90, 105, 105, 70, 110], "score_b": [100, 100, 100, 100, 100, 100],
             "a_first": [True, False] * 3, "turns": [20, 22, 21, 21, 19, 25]}
    per_pair = np.array([(20 - 10) / 2, (5 + 5) / 2, (-30 + 10) / 2])  # 5, 5, -10
    wins = np.array([(1 + 0) / 2, (1 + 1) / 2, (0 + 1) / 2])
    stats = quick_stats(games)
    assert stats["spread"] == pytest.approx(per_pair.mean())
    assert stats["spread_ci"] == pytest.approx(1.96 * per_pair.std(ddof=1) / math.sqrt(3))
    assert stats["win_rate"] == pytest.approx(wins.mean())
    tie = {"score_a": [100, 100, 100, 100], "score_b": [100, 100, 100, 100], "a_first": [True, False] * 2,
           "turns": [20] * 4}
    assert quick_stats(tie) == {"spread": 0.0, "spread_ci": 0.0, "win_rate": 0.5}


def test_quick_stats_needs_two_pairs():
    empty = {"spread": None, "spread_ci": None, "win_rate": None}
    assert quick_stats({"score_a": [], "score_b": [], "a_first": [], "turns": []}) == empty
    assert quick_stats({"score_a": [1, 2], "score_b": [2, 1], "a_first": [True, False], "turns": [3, 3]}) == empty


# ---- Matches: start, list, get, cancel ---------------------------------------------------------


def test_start_writes_the_job_files(setup, spawned):
    match_id = setup.matches.start(BEST, "bot:greedy-leave", 25, 32)
    folder = setup.app_dir / "matches" / match_id
    assert match_id.startswith("m_") and len(match_id) == len("m_20261112_173005")
    spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
    assert {k: spec[k] for k in ("id", "a", "b", "a_name", "b_name", "pairs", "candidates", "runs_dir")} == {
        "id": match_id, "a": BEST, "b": "bot:greedy-leave", "a_name": f"{TINY_RUN} (best)", "b_name": "Greedy + leave",
        "pairs": 25, "candidates": 32, "runs_dir": str(setup.runs_dir)}
    assert spec["created"] == pytest.approx(time.time(), abs=60)
    assert 8_000_000_000_000 <= spec["first_seed"] < 2**63
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "running" and status["games_done"] == 0
    [(command, log_path)] = spawned.calls
    assert command == [python_executable(), "-u", "-m", "scrabble_ai.app.match_job", str(folder)]
    assert log_path == folder / "job.log"
    assert json.loads((folder / "pid.json").read_text(encoding="utf-8")) == {"pid": spawned.pid}
    summary = setup.matches.list()[0]
    assert summary["status"] == "running" and summary["progress"] == 0.0 and summary["spread"] is None
    second = setup.matches.start("bot:greedy", "bot:random", 3)
    assert second != match_id and (setup.app_dir / "matches" / second / "spec.json").exists()
    assert json.loads((setup.app_dir / "matches" / second / "spec.json").read_text(encoding="utf-8"))["candidates"] == 64


@pytest.mark.parametrize("a, b, pairs, candidates, message", [
    ("human", "bot:greedy", 10, 64, "bots and models"),
    ("bot:greedy", "human", 10, 64, "bots and models"),
    ("bot:nope", "bot:greedy", 10, 64, "unknown player"),
    ("model:../x/best.pt", "bot:greedy", 10, 64, "not a model id"),
    (f"model:{TINY_RUN}/step_000999999.pt", "bot:greedy", 10, 64, "no longer exists"),
    ("bot:greedy", "bot:random", 0, 64, "pairs"),
    ("bot:greedy", "bot:random", 100_001, 64, "pairs"),
    ("bot:greedy", "bot:random", 10, 1, "candidates"),
    ("bot:greedy", "bot:random", 10, 513, "candidates"),
    ("bot:greedy", "bot:random", "10", 64, "pairs must be a whole number"),
    ("bot:greedy", "bot:random", 10.0, 64, "pairs must be a whole number"),
    ("bot:greedy", "bot:random", True, 64, "pairs must be a whole number"),
    ("bot:greedy", "bot:random", None, 64, "pairs must be a whole number"),
    ("bot:greedy", "bot:random", 10, 64.0, "candidates must be a whole number"),
    ("bot:greedy", "bot:random", 10, False, "candidates must be a whole number"),
    (None, "bot:random", 10, 64, "named by a string"),
])
def test_start_refusals(setup, spawned, a, b, pairs, candidates, message):
    with pytest.raises(PlayerError, match=message):
        setup.matches.start(a, b, pairs, candidates)
    assert spawned.calls == [] and not (setup.app_dir / "matches").exists()


def craft(app_dir, match_id, created, state="running", pid=None, games_done=0, pairs=10, games=None, **status):
    folder = app_dir / "matches" / match_id
    folder.mkdir(parents=True)
    write_json(folder / "spec.json", {"id": match_id, "a": "bot:greedy", "b": "bot:random", "a_name": "Greedy",
                                      "b_name": "Random", "pairs": pairs, "candidates": 64, "created": created,
                                      "runs_dir": "x", "first_seed": 8_000_000_000_000})
    write_json(folder / "status.json", {"status": state, "games_done": games_done, "started": created, "seconds": 1.5,
                                        "error": None, **status})
    if pid is not None:
        write_json(folder / "pid.json", {"pid": pid})
    if games is not None:
        write_json(folder / "games.json", games)
    return folder


def test_list_get_and_cancel_on_crafted_folders(setup):
    now = time.time()
    craft(setup.app_dir, "m_20261112_100000", now - 100, pid=os.getpid(), games_done=5)
    craft(setup.app_dir, "m_20261112_100001", now - 90, pid=DEAD_PID, games_done=4)  # died 90 s ago
    craft(setup.app_dir, "m_20261112_100002", now - 5, pid=DEAD_PID)  # just started: not yet failed
    craft(setup.app_dir, "m_20261112_100003", now - 80, state="finished", pid=DEAD_PID, games_done=20,
          spread=6.1, spread_ci=3.4, win_rate=0.53)
    craft(setup.app_dir, "m_20261112_100004", now - 70, state="cancelled", pid=DEAD_PID, games_done=8)
    craft(setup.app_dir, "m_20261112_100005", now - 60, state="failed", pid=DEAD_PID, error="ValueError: boom")
    (setup.app_dir / "matches" / "m_20261112_100006").mkdir()  # no spec yet: skipped
    summaries = setup.matches.list()
    assert [s["id"] for s in summaries] == ["m_20261112_100002", "m_20261112_100005", "m_20261112_100004",
                                            "m_20261112_100003", "m_20261112_100001", "m_20261112_100000"]
    by_id = {s["id"]: s for s in summaries}
    assert by_id["m_20261112_100000"]["status"] == "running" and by_id["m_20261112_100000"]["progress"] == 0.25
    assert by_id["m_20261112_100001"]["status"] == "failed"
    assert by_id["m_20261112_100002"]["status"] == "running"
    finished = by_id["m_20261112_100003"]
    assert (finished["status"], finished["progress"], finished["spread"], finished["spread_ci"], finished["win_rate"]) == (
        "finished", 1.0, 6.1, 3.4, 0.53)
    assert by_id["m_20261112_100004"]["status"] == "cancelled"
    assert by_id["m_20261112_100005"]["status"] == "failed" and by_id["m_20261112_100005"]["error"] == "ValueError: boom"
    for summary in summaries:
        assert {"id", "a", "b", "a_name", "b_name", "pairs", "status", "progress", "games_done", "created", "seconds",
                "spread", "spread_ci", "win_rate"} <= set(summary)
    assert setup.matches.active() == ["m_20261112_100002", "m_20261112_100000"]

    detail = setup.matches.get("m_20261112_100000")
    assert set(detail) == {"match", "summary", "games", "highlights"}
    assert detail["games"] == {name: [] for name in COLUMNS}
    setup.matches.cancel("m_20261112_100000")
    assert (setup.app_dir / "matches" / "m_20261112_100000" / "CANCEL").exists()
    with pytest.raises(KeyError):
        setup.matches.get("m_20261112_100006")  # a folder without a spec yet
    for bad in ("m_nope", "x_20261112_100000", "m_../m_20261112_100000", "", "m_", "m_..", "..\\m_20261112_100000",
                "m_20261112_100000/../m_20261112_100003"):
        with pytest.raises(KeyError):
            setup.matches.get(bad)
        with pytest.raises(KeyError):
            setup.matches.cancel(bad)
    assert not (setup.app_dir / "CANCEL").exists() and not (setup.app_dir / "matches" / "CANCEL").exists()


def test_a_match_pid_older_than_its_process_is_not_that_process(setup):
    """pid.json names the job; a later process given the same number must not keep the match 'running'."""
    from scrabble_ai.app import procs

    born = procs.process_start_time(os.getpid())
    assert born is not None
    now = time.time()
    cases = {"m_20261112_110000": ({"pid": os.getpid(), "started": born - 1000}, None, "failed"),
             "m_20261112_110001": ({"pid": os.getpid()}, born - 1000, "failed"),
             "m_20261112_110002": ({"pid": os.getpid()}, None, "running"),
             "m_20261112_110003": ({"pid": os.getpid(), "started": now}, None, "running"),
             "m_20261112_110004": ({"pid": "abc"}, None, "failed")}
    for match_id, (marker, file_time, _) in cases.items():
        folder = craft(setup.app_dir, match_id, now - 100)
        (folder / "pid.json").write_text(json.dumps(marker), encoding="utf-8")
        if file_time is not None:
            os.utime(folder / "pid.json", (file_time, file_time))
    states = {s["id"]: s["status"] for s in setup.matches.list()}
    assert states == {match_id: expected for match_id, (_, _, expected) in cases.items()}
    # Within its first 20 seconds a match is running whatever its pid file says.
    folder = craft(setup.app_dir, "m_20261112_110005", time.time() - 5)
    (folder / "pid.json").write_text(json.dumps({"pid": os.getpid(), "started": born - 1000}), encoding="utf-8")
    assert {s["id"]: s["status"] for s in setup.matches.list()}["m_20261112_110005"] == "running"


def test_unknown_matches_are_not_found(setup):
    from scrabble_ai.app.players import NotFound

    with pytest.raises(NotFound, match="no match m_nope") as raised:
        setup.matches.get("m_nope")
    assert isinstance(raised.value, KeyError) and str(raised.value) == "there is no match m_nope"
    with pytest.raises(NotFound):
        setup.matches.cancel("m_²³")  # not ASCII digits


def test_no_matches_yet(setup):
    assert setup.matches.list() == [] and setup.matches.active() == []


# ---- the job, in-process -----------------------------------------------------------------------


@pytest.fixture
def job_engine(monkeypatch, trained):
    """match_job loads the engine itself; hand it the test one."""
    monkeypatch.setattr(engine_module, "load", lambda *args, **kwargs: trained.loaded)


def test_match_job_plays_a_bot_match(setup, job_engine):
    match_id = setup.matches.start("bot:greedy-leave", "bot:greedy", 4)
    folder = setup.app_dir / "matches" / match_id
    assert match_job.main(str(folder)) == 0
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "finished" and status["games_done"] == 8 and status["error"] is None
    assert status["spread"] is not None and status["win_rate"] is not None and status["seconds"] >= 0
    games = json.loads((folder / "games.json").read_text(encoding="utf-8"))
    check_columns(games, 4, setup.engine)
    spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
    assert games["seed"][0] == spec["first_seed"] and games["seed"][-1] == spec["first_seed"] + 3
    reference = setup.engine.play_pairs("greedy-leave", setup.leaves, "greedy", setup.leaves,
                                        np.arange(4, dtype=np.uint64) + np.uint64(spec["first_seed"]))
    assert games["score_a"] == [int(x) for x in reference["score_a"]]
    detail = setup.matches.get(match_id)
    assert detail["match"]["status"] == "finished" and detail["match"]["progress"] == 1.0
    assert detail["summary"] is not None and detail["games"] == games
    assert all(0 <= h["game"] < 8 for h in detail["highlights"])


def test_match_job_honours_cancel(setup, job_engine):
    match_id = setup.matches.start("bot:greedy-leave", "bot:greedy", 50)
    folder = setup.app_dir / "matches" / match_id
    setup.matches.cancel(match_id)
    assert match_job.main(str(folder)) == 0
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "cancelled" and status["games_done"] == 0
    assert setup.matches.get(match_id)["match"]["status"] == "cancelled"


def test_match_job_reports_a_failure(setup, job_engine):
    match_id = setup.matches.start(BEST, "bot:greedy", 2)
    folder = setup.app_dir / "matches" / match_id
    (setup.runs_dir / TINY_RUN / "checkpoints" / "best.pt").unlink()  # gone before the job loads it
    assert match_job.main(str(folder)) == 1
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and "PlayerError" in status["error"] and "no longer exists" in status["error"]
    assert setup.matches.list()[0]["status"] == "failed"


def test_match_job_without_fitted_leave_values(setup, monkeypatch, lexicon_dir, tmp_path):
    """The job loads the engine as the app does, so bots can play before leave values are fitted."""
    monkeypatch.setattr(engine_module, "lexicon_entry", lexicon_entry_for(lexicon_dir))
    monkeypatch.setattr(engine_module, "DATA_DIR", tmp_path / "no-data")
    app_side = engine_module.load(allow_missing_leaves=True)
    assert app_side.leaves_path is None, "the app itself runs with zero leave values"
    match_id = setup.matches.start("bot:greedy", "bot:random", 2)
    folder = setup.app_dir / "matches" / match_id
    match_job.main(str(folder))
    assert json.loads((folder / "status.json").read_text(encoding="utf-8"))["status"] == "finished"
