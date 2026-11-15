"""How a saved network picks its move: `Scorer(..., win_weight, win_scale)` carries the rule it was
trained and evaluated with (`selfplay.win_weight` and `selfplay.win_scale` in its checkpoint), and
the app (players, sessions, matches) and `scrabble-ai eval` use it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scrabble_ai import cli
from scrabble_ai import evaluate as evaluate_module
from scrabble_ai.app.matches import play_match_detailed
from scrabble_ai.app.sessions import Sessions
from scrabble_ai.evaluate import MatchResult, NetPlayer, play_match
from scrabble_ai.selfplay import Scorer
from scrabble_ai.train import load_network

pytest.register_assert_rewrite("app_fixtures")
from app_fixtures import TINY_RUN, copy_run  # noqa: E402

pytest_plugins = ["app_fixtures"]

MODEL = f"model:{TINY_RUN}/best.pt"
CANDIDATE_KEYS = {"text", "kind", "word", "position", "score", "tiles", "leave", "exchanged", "leave_value", "value",
                  "total", "win", "chosen"}


def set_rule(best_file, win_weight: float, win_scale: float) -> None:
    state = torch.load(best_file, map_location="cpu", weights_only=False)
    state["config"]["selfplay"]["win_weight"] = win_weight
    state["config"]["selfplay"]["win_scale"] = win_scale
    torch.save(state, best_file)


@pytest.fixture
def runs(tmp_path, trained):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    copy_run(trained, runs_dir)
    return runs_dir


def test_the_scorer_carries_the_rule_it_is_given(trained):
    net, _ = load_network(trained.run_dir / "checkpoints" / "best.pt", torch.device("cpu"))
    plain = Scorer(net, trained.loaded.engine, 50.0, torch.device("cpu"))
    assert (plain.win_weight, plain.win_scale) == (0.0, 100.0)
    ruled = Scorer(net, trained.loaded.engine, 50.0, torch.device("cpu"), win_weight=0.9, win_scale=60.0)
    assert (ruled.win_weight, ruled.win_scale) == (0.9, 60.0)


def test_players_read_the_rule_from_the_checkpoint(runs, cpu_players):
    players = cpu_players(runs)
    scorer = players.scorer(MODEL)
    assert (scorer.win_weight, scorer.win_scale) == (0.0, 100.0), "the tiny run was trained by points"
    set_rule(runs / TINY_RUN / "checkpoints" / "best.pt", 0.75, 40.0)
    scorer = players.scorer(MODEL)  # the file changed: read again
    assert (scorer.win_weight, scorer.win_scale) == (0.75, 40.0)


def test_a_session_ranks_by_utility_for_a_network_that_plays_to_win(runs, cpu_players, fast_scorer):
    set_rule(runs / TINY_RUN / "checkpoints" / "best.pt", 1.0, 100.0)
    sessions = Sessions(cpu_players(runs))
    session = sessions.create([MODEL, "bot:greedy"], seed=11)
    analysis = session.ai_move(top=200)["analysis"]
    candidates = analysis["candidates"]
    assert candidates and all(set(c) == CANDIDATE_KEYS | {"utility"} for c in candidates)
    utilities = [c["utility"] for c in candidates]
    assert utilities == sorted(utilities, reverse=True), "listed in the order the network ranks them"
    for c in candidates:
        assert c["utility"] == pytest.approx(100.0 * (2.0 * c["win"] - 1.0), abs=0.01), "win chance alone at weight 1"
    chosen = [c for c in candidates if c["chosen"]]
    assert len(chosen) == 1 and chosen[0]["utility"] == max(utilities)


def test_a_session_of_a_network_that_plays_by_points_has_no_utility(runs, cpu_players, fast_scorer):
    sessions = Sessions(cpu_players(runs))
    session = sessions.create([MODEL, "bot:greedy"], seed=11)
    candidates = session.ai_move(top=200)["analysis"]["candidates"]
    assert all(set(c) == CANDIDATE_KEYS for c in candidates)
    totals = [c["total"] for c in candidates]
    assert totals == sorted(totals, reverse=True)


def test_matches_play_each_network_by_its_own_rule(runs, cpu_players, trained, fast_scorer):
    set_rule(runs / TINY_RUN / "checkpoints" / "best.pt", 0.8, 30.0)
    scorer = cpu_players(runs).scorer(MODEL)
    seeds = np.arange(3, dtype=np.uint64) + np.uint64(321)
    engine, leaves = trained.loaded.engine, trained.loaded.leaves
    columns = play_match_detailed(engine, leaves, {MODEL: scorer}, MODEL, "bot:greedy-leave", seeds, torch.device("cpu"),
                                  candidates=16, exchanges=2, slots=6)
    reference = play_match(engine, leaves, NetPlayer("net", scorer, 0.8, 30.0), "greedy-leave", seeds, torch.device("cpu"),
                           16, 2)
    assert np.mean(columns["score_a"]) == pytest.approx(reference.score_a)
    assert np.mean(columns["score_b"]) == pytest.approx(reference.score_b)


def test_scrabble_ai_eval_plays_a_network_by_its_own_rule(runs, trained, monkeypatch, test_lexicon):
    best = runs / TINY_RUN / "checkpoints" / "best.pt"
    set_rule(best, 0.6, 70.0)
    seen = []

    def fake_match(engine, leaves, a, b, seeds, device, candidates, exchanges):
        seen.append(a)
        return MatchResult("a", "b", len(seeds), 0.0, 1.0, 0.5, 0.1, 400.0, 400.0, 24.0)

    monkeypatch.setattr(evaluate_module, "play_match", fake_match)
    common = ["--pairs", "2", "--lexicon", "TEST", "--leaves", str(trained.fitted), "--device", "cpu"]
    cli.main(["eval", str(best), "greedy-leave", *common])
    assert (seen[-1].win_weight, seen[-1].win_scale) == (0.6, 70.0)
    cli.main(["eval", str(best), "greedy-leave", *common, "--win-weight", "0.2", "--win-scale", "10"])
    assert (seen[-1].win_weight, seen[-1].win_scale) == (0.2, 10.0)
