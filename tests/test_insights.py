"""Tests for the run report and match summary analysis (scrabble_ai.app.insights).

Records and game columns are built by hand so that every expected headline, event and number is
known in advance.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from scrabble_ai.app import insights
from scrabble_ai.app.insights import match_summary, run_overview, run_report

EVENT_KEYS = {"games", "hours", "kind", "tone", "title", "detail"}
SUMMARY_KEYS = {"games", "hours", "steps", "mean_games_per_second", "best", "final", "first_significant",
                "promotions", "value_rmse_gain", "health"}
MATCH_KEYS = {"spread", "spread_ci", "win_rate", "win_ci", "significant", "score_a", "score_b", "turns", "a_first",
              "a_second", "bingos_a", "bingos_b", "exchanges_a", "exchanges_b", "passes_a", "passes_b",
              "ended_by_going_out"}


# ---- record builders -------------------------------------------------------------------------


def rec(kind, games=0, hours=0.0, step=0, **fields):
    return {"kind": kind, "time": 1794520000.0 + hours * 3600, "step": step, "games": games,
            "positions": games * 24, "hours": hours, **fields}


def ev(games, hours, spread, ci, opponent="greedy-leave", old=False, step=0, pairs=250):
    if old:
        fields = {"opponent": opponent, f"spread_vs_{opponent}": spread, f"spread_ci_vs_{opponent}": ci,
                  f"win_rate_vs_{opponent}": 0.5, "pairs": pairs}
    else:
        fields = {"opponent": opponent, "spread": spread, "spread_ci": ci, "win_rate": 0.5, "win_ci": 0.03,
                  "score_for": 430.0, "score_against": 425.0, "pairs": pairs}
    return rec("eval", games, hours, step, **fields)


def warm_start(steps=3000):
    """A finished warm start and pretraining, as the trainer logs them."""
    out = [rec("warmstart", 0, 0.0, 0, games_played=100000, positions_added=2400000, seconds=176.0, mean_score=415.7)]
    for s in range(500, steps + 1, 500):
        out.append(rec("train", 0, 0.0, s, loss=1.2, value_loss=0.95, win_loss=0.45, value_error_points=54.0,
                       learning_rate=1e-3))
    out.append(rec("checkpoint", 0, 0.05, steps, file=f"step_{steps:09d}.pt", replay_positions=2400000))
    return out


def make_run(evals=(), *, minutes_per_eval=20, extra_minutes=0, warm=True, old=False, greedy=True, gps=None,
             loss=None, gates=None, plant=None, seed=0, records_fields=True):
    """A run of self-play minutes with an evaluation against greedy-leave every `minutes_per_eval`.

    evals: (spread, ci) per evaluation. gps(minute) / loss(minute) override the steady defaults
    (56 games/s, loss 1.17 with mild noise). gates[k]: None for no gate record after evaluation k,
    True / False for `promoted`, "nokey" for a gate record without `promoted`. plant: {minute:
    {field: value}} written into that minute's self-play record."""
    rng = np.random.default_rng(seed)
    records = warm_start() if warm else []
    state = {"games": 0, "minute": 0}
    step0 = 3000 if warm else 0

    def minute():
        state["minute"] += 1
        m = state["minute"]
        rate = gps(m) if gps else 56.0
        state["games"] += int(round(rate * 60))
        hours = round(0.05 + m / 60, 4)
        step = step0 + 160 * m
        value = loss(m) if loss else 1.17 + float(rng.normal(0, 0.004))
        records.append(rec("train", state["games"], hours, step, loss=value, value_loss=0.94, win_loss=0.45,
                           value_error_points=54.0, learning_rate=1e-3))
        fields = {"games_per_second": rate, "positions_per_second": rate * 24, "mean_score": 415.0,
                  "mean_turns": 24.7, "mean_abs_spread": 85.0, "first_player_win_rate": 0.54, "temperature": 3.0,
                  "value_rmse_points": 73.0, "leave_only_rmse_points": 74.0, "win_brier": 0.15,
                  "replay_positions": 3000000, "replay_gb": 0.9}
        if records_fields:
            fields.update(bingos_per_game=1.9, exchanges_per_game=0.3, passes_per_game=0.02,
                          max_game_score=int(550 + rng.integers(0, 100)), max_play_score=int(120 + rng.integers(0, 60)))
        fields.update((plant or {}).get(m, {}))
        records.append(rec("selfplay", state["games"], hours, step, **fields))
        return hours, step

    for k, (spread, ci) in enumerate(evals):
        for _ in range(minutes_per_eval):
            hours, step = minute()
        records.append(ev(state["games"], hours, spread, ci, old=old, step=step))
        if greedy:
            records.append(ev(state["games"], hours, spread + 40.0, 9.0, opponent="greedy", old=old, step=step))
        flag = gates[k] if gates is not None and k < len(gates) else None
        if flag is not None:
            fields = {"spread_vs_best" if old else "spread": 6.0, "spread_ci": 4.0, "win_rate": 0.55,
                      "best_step": step - 3200, "pairs": 100}
            if flag != "nokey":
                fields["promoted"] = flag
            records.append(rec("gate", state["games"], hours, step, **fields))
    for _ in range(extra_minutes):
        minute()
    return records


def kinds(report, kind):
    return [e for e in report["events"] if e["kind"] == kind]


def check_shape(report):
    assert set(report) == {"headline", "summary", "events"}
    assert SUMMARY_KEYS <= set(report["summary"])
    for e in report["events"]:
        assert set(e) == EVENT_KEYS
        assert e["tone"] in ("good", "info", "warning")
    games = [(e["games"], e["hours"]) for e in report["events"]]
    assert games == sorted(games)
    assert "!" not in report["headline"]
    json.dumps(report)  # serializable as is


def says_significant(text):
    return "significant" in text.lower()


# ---- headlines -------------------------------------------------------------------------------


def test_empty_records():
    for records in ([], None):
        report = run_report("night1", records)
        check_shape(report)
        assert report["headline"] == "night1 has not logged anything yet."
        assert report["events"] == []
        assert all(report["summary"][k] is None for k in SUMMARY_KEYS)


def test_odd_input_never_raises():
    records = [None, 5, "text", [], {"kind": 3}, {"kind": "eval"}, {"kind": "eval", "spread": "abc"},
               {"kind": "eval", "opponent": "greedy-leave", "spread": float("nan"), "spread_ci": 2.0},
               {"kind": "train", "loss": "nan"}, {"kind": "train", "loss": None, "games": "lots"},
               {"kind": "selfplay", "games": {"a": 1}, "games_per_second": "fast"},
               {"kind": "gate", "promoted": "yes", "spread": [1]}, {"kind": "session", "event": "stop", "reason": 5},
               {"kind": "warmstart", "games_played": "many"}, {"kind": "checkpoint", "hours": float("inf")},
               {"kind": "eval", "opponent": "greedy-leave", "spread": 3.0, "spread_ci": -1.0, "games": 10}]
    report = run_report("odd", records, config="not a dict")
    check_shape(report)
    # The one readable evaluation has no usable interval: nothing may be called significant.
    assert report["summary"]["final"]["spread"] == 3.0 and report["summary"]["final"]["ci"] is None
    assert not says_significant(report["headline"])
    assert report["summary"]["promotions"] is None
    assert match_summary(None, None, None) == (None, [])
    assert match_summary({"score_a": "abc", "score_b": 5}, "A", "B") == (None, [])
    summary, highlights = match_summary({"score_a": [1, None, float("nan"), "x"], "score_b": [2, 3]}, "A", "B")
    assert summary is None and isinstance(highlights, list)


def test_warm_start_in_progress():
    records = [rec("session", 0, 0.0, 0, event="start"),
               rec("warmstart", 0, 0.0, 0, games_played=20008, positions_added=473244, seconds=35.2, mean_score=415.7)]
    report = run_report("night1", records, {"warmstart": {"games": 100000, "steps": 3000}})
    check_shape(report)
    assert report["headline"] == "night1 has not finished its warm start (20,008 of 100,000 games), so there is no evaluation yet."
    assert report["events"] == []  # not finished: no phase event
    # Without the config the lone record could be the end of the warm start: say only what is certain.
    report = run_report("night1", records)
    assert report["headline"] == ("night1 has not started self-play yet (20,008 warm-start games so far), "
                                  "so there is no evaluation yet.")
    assert run_report("night1", records[:1])["headline"] == "night1 has started but has not reported any results yet."


def test_pretraining():
    records = warm_start()[:3]  # warm start and pretraining to step 1,000
    report = run_report("night1", records, {"warmstart": {"games": 100000, "steps": 3000}})
    check_shape(report)
    assert report["headline"] == ("night1 has finished its warm start but has not started self-play yet "
                                  "(pretraining has reached step 1,000 of 3,000), so there is no evaluation yet.")
    [phase] = report["events"]
    assert (phase["kind"], phase["title"]) == ("phase", "Warm start finished")
    assert phase["detail"] == "100,000 games and 2,400,000 positions in 176 s."


def test_self_play_without_evaluation():
    records = make_run(extra_minutes=5)
    report = run_report("night1", records, {"train": {"eval_minutes": 20}})
    check_shape(report)
    assert report["headline"] == ("night1 has played 16,800 self-play games in 8 minutes but has not been "
                                  "evaluated yet. Evaluations run every 20 minutes of self-play.")
    phases = kinds(report, "phase")
    assert [e["title"] for e in phases] == ["Warm start finished", "Self-play began"]
    assert phases[1]["games"] == 0 and phases[1]["hours"] == 0.05
    assert phases[1]["detail"] == "Pretraining ran to step 3,000; the network then began playing itself."
    assert report["summary"]["best"] is None and report["summary"]["first_significant"] is None


def test_not_shown_to_beat_baseline():
    report = run_report("night1", make_run([(4.99, 8.09), (4.17, 7.81)]))
    check_shape(report)
    assert report["headline"] == ("night1 has not been shown to beat greedy-leave: its best evaluation was "
                                  "+5.0 +/- 8.1 points a game, an interval that includes zero. The latest is +4.2 +/- 7.8.")
    assert report["summary"]["first_significant"] is None
    assert kinds(report, "significant") == []


def test_significant_at_best_and_at_end():
    report = run_report("night1", make_run([(1.0, 5.0), (6.1, 3.9), (5.2, 4.1)]))
    check_shape(report)
    s = report["summary"]
    assert report["headline"] == (f"night1 beats greedy-leave by 5.2 points a game at its latest evaluation "
                                  f"(+5.2 +/- 4.1), a significant margin. Its best was +6.1 +/- 3.9 at "
                                  f"{s['best']['games']:,} games.")
    assert s["best"]["spread"] == 6.1 and s["final"]["spread"] == 5.2
    assert s["first_significant"]["games"] == s["best"]["games"]
    [event] = kinds(report, "significant")
    assert event["tone"] == "good" and event["games"] == s["best"]["games"]
    assert event["detail"] == "+6.1 +/- 3.9 points a game over 250 pairs, the first interval to clear zero."


def test_significant_earlier_but_not_at_end():
    report = run_report("night1", make_run([(1.0, 5.0), (6.1, 3.9), (1.2, 4.0)]))
    check_shape(report)
    best = report["summary"]["best"]
    assert report["headline"] == (f"night1 beat greedy-leave significantly earlier in the run (+6.1 +/- 3.9 points "
                                  f"a game at {best['games']:,} games), but its latest evaluation, +1.2 +/- 4.0, "
                                  f"no longer clears zero.")
    # A 4.9-point fall with intervals of 3.9 and 4.0 is ordinary noise, not a regression.
    assert kinds(report, "regression") == []


def test_significantly_worse_than_baseline():
    report = run_report("night1", make_run([(-2.0, 5.0), (-8.2, 4.0)]))
    check_shape(report)
    assert report["headline"] == ("night1 is significantly worse than greedy-leave: -8.2 +/- 4.0 points a game at "
                                  "its latest evaluation. Every evaluation so far has been behind it; the best was "
                                  "-2.0 +/- 5.0.")
    [event] = kinds(report, "significant")
    assert event["tone"] == "warning" and event["title"] == "Significantly worse than greedy-leave"

    report = run_report("night1", make_run([(1.0, 5.0), (-6.5, 4.0)]))
    assert report["headline"].endswith("It has not been shown to beat it at any evaluation; the best was +1.0 +/- 5.0.")

    report = run_report("night1", make_run([(6.0, 3.0), (-6.0, 3.0)]))
    first = report["summary"]["first_significant"]
    assert report["headline"] == (f"night1 is now significantly worse than greedy-leave, -6.0 +/- 3.0 points a game at "
                                  f"its latest evaluation, although it was significantly better earlier (+6.0 +/- 3.0 "
                                  f"at {first['games']:,} games).")
    assert [e["tone"] for e in kinds(report, "significant")] == ["good", "warning"]


def test_every_evaluation_behind_but_last_one_unclear():
    report = run_report("night1", make_run([(-6.0, 3.0), (-4.0, 6.0)]))
    assert report["headline"] == ("night1 has not been shown to beat greedy-leave: every evaluation so far has been "
                                  "behind it, the best being -4.0 +/- 6.0 points a game.")
    assert not says_significant(report["headline"])


def test_only_greedy_evaluations():
    records = [r for r in make_run([(1.0, 5.0)]) if r.get("opponent") != "greedy-leave"]
    report = run_report("night1", records)
    check_shape(report)
    assert report["headline"] == ("night1 has no evaluation against greedy-leave yet; its latest, against greedy, "
                                  "is +41.0 +/- 9.0 points a game, a significant margin.")
    assert report["summary"]["best"] is None and report["summary"]["evaluations"] == 0


def test_old_and_new_key_layouts_agree():
    evals = [(1.0, 5.0), (6.1, 3.9), (1.2, 4.0)]
    gates = [None, "nokey", "nokey"]
    new, old = run_report("r", make_run(evals, gates=gates)), run_report("r", make_run(evals, gates=gates, old=True))
    assert new == old
    assert new["summary"]["final"]["spread"] == 1.2 and new["summary"]["final"]["ci"] == 4.0


# ---- summary numbers -------------------------------------------------------------------------


def test_summary_numbers_by_hand():
    records = warm_start()
    # Three reports of 10 s each: 600 games at 60/s, 500 at 50/s, 300 at 30/s -> 1,400 games in 30 s.
    gaps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0]
    for k, (games, rate) in enumerate([(600, 60.0), (1100, 50.0), (1400, 30.0)]):
        records.append(rec("selfplay", games, 0.06 + k / 1000, 3000 + k, games_per_second=rate,
                           value_rmse_points=70.0 - gaps[k], leave_only_rmse_points=70.0))
    for k, gap in enumerate(gaps[3:]):  # more value-RMSE reports without a rate
        records.append(rec("selfplay", 1400 + k, 0.07, 3010 + k, value_rmse_points=70.0 - gap,
                           leave_only_rmse_points=70.0))
    records.append(ev(1400, 0.07, 3.0, 5.0, step=3020))
    records.append(ev(1500, 0.08, 3.0, 4.0, step=3030))  # same spread: the earlier stays the best
    s = run_report("r", records)["summary"]
    assert s["mean_games_per_second"] == pytest.approx(1400 / 30, abs=0.01)
    assert s["value_rmse_gain"] == pytest.approx(2.0)  # last quarter of 8 reports: gaps 1 and 3
    assert s["best"]["step"] == 3020 and s["final"]["step"] == 3030
    assert s["games"] == 1500 and s["hours"] == 0.08 and s["steps"] == 3030
    assert s["promotions"] is None


def test_mean_rate_over_two_sessions_on_real_shapes():
    records = make_run([(1.0, 5.0)], gps=lambda m: 50.0)
    records.append(rec("session", records[-1]["games"], records[-1]["hours"], records[-1]["step"], event="stop",
                       reason="STOP file found", finished=False))
    games, hours = records[-1]["games"], records[-1]["hours"]
    records.append(rec("session", games, hours, 9000, event="start"))
    for k in range(1, 4):
        records.append(rec("selfplay", games + 4200 * k, hours + k / 60, 9000 + k, games_per_second=70.0))
    s = run_report("r", records)["summary"]
    # 60,000 games at 50/s (1,200 s) and 12,600 at 70/s (180 s).
    assert s["mean_games_per_second"] == pytest.approx((60000 + 12600) / (1200 + 180), abs=0.01)


# ---- events ----------------------------------------------------------------------------------


def test_promotions_counted_and_reported():
    report = run_report("r", make_run([(1.0, 5.0)] * 4, gates=[None, True, False, True]))
    check_shape(report)
    assert report["summary"]["promotions"] == 2
    promos = kinds(report, "promotion")
    assert [e["title"] for e in promos] == ["New best network"] * 2
    # The first gate follows the second evaluation (step 3,000 + 40 x 160); its best is 20 minutes older.
    assert promos[0]["detail"] == "Beat the best so far (step 6,200) by +6.0 +/- 4.0 over 100 pairs and replaced it."

    report = run_report("r", make_run([(1.0, 5.0)] * 3, gates=[None, "nokey", "nokey"]))
    assert report["summary"]["promotions"] is None and kinds(report, "promotion") == []


def test_first_network_is_not_described_as_a_win():
    """The trainer logs the first evaluated network as a gate record with no match behind it."""
    records = make_run([(1.0, 5.0)] * 3, gates=[None, True, False])
    first_eval = next(i for i, r in enumerate(records) if r["kind"] == "eval")
    at = records[first_eval]
    records.insert(first_eval + 1, rec("gate", at["games"], at["hours"], at["step"], spread=None, spread_ci=None,
                                       win_rate=None, best_step=at["step"], promoted=True, pairs=0))
    report = run_report("r", records)
    check_shape(report)
    assert report["summary"]["promotions"] == 1  # the one real gate win
    promos = kinds(report, "promotion")
    assert [e["title"] for e in promos] == ["First network to beat", "New best network"]
    assert "Beat" not in promos[0]["detail"] and promos[0]["tone"] == "info"

    only_first = [r for r in records if r["kind"] != "gate" or r["pairs"] == 0]
    report = run_report("r", only_first)
    assert report["summary"]["promotions"] is None
    assert [e["title"] for e in kinds(report, "promotion")] == ["First network to beat"]


def test_self_play_start_uses_the_trainers_marker():
    records = warm_start() + [rec("session", 0, 0.05, 3000, event="selfplay")]
    report = run_report("r", records, {"warmstart": {"games": 100000, "steps": 3000}})
    check_shape(report)
    starts = [e for e in kinds(report, "phase") if e["title"] == "Self-play began"]
    assert len(starts) == 1 and "step 3,000" in starts[0]["detail"]


def test_many_promotions_are_folded():
    report = run_report("r", make_run([(1.0, 5.0)] * 12, gates=[None] + [True] * 11))
    assert report["summary"]["promotions"] == 11
    promos = kinds(report, "promotion")
    assert [e["title"] for e in promos] == ["New best network", "9 more promotions", "New best network"]
    assert promos[0]["games"] < promos[1]["games"] < promos[2]["games"]


def test_regression_fires_on_a_real_drop():
    report = run_report("r", make_run([(5.0, 3.0), (-3.0, 3.0)]))
    [event] = kinds(report, "regression")
    assert event["tone"] == "warning"
    assert event["detail"] == ("Fell 8.0 points against greedy-leave, from +5.0 +/- 3.0 to -3.0 +/- 3.0, more than "
                               "the 4.2 that noise between two evaluations would explain.")


def test_regression_ignores_noise():
    # Evaluations zig-zagging by 5 points with 4-point intervals: ordinary noise.
    evals = [(3.0 + (2.5 if k % 2 else -2.5), 4.0) for k in range(30)]
    assert kinds(run_report("r", make_run(evals, minutes_per_eval=2)), "regression") == []
    # Just under the threshold (1.3 x 5.66 = 7.35).
    assert kinds(run_report("r", make_run([(5.0, 4.0), (-2.3, 4.0)])), "regression") == []


def test_plateau():
    evals = [(1.0, 4.0), (2.0, 4.0), (6.0, 4.0)] + [(3.0, 4.0)] * 6
    report = run_report("r", make_run(evals))
    [event] = kinds(report, "plateau")
    best = report["summary"]["best"]
    assert event["title"] == "No new best in 6 evaluations"
    assert event["games"] == report["summary"]["final"]["games"]
    assert event["detail"] == (f"The best evaluation against greedy-leave, +6.0 +/- 4.0 at {best['games']:,} games, "
                               f"has not been beaten in the 6 evaluations since (2.0 hours and "
                               f"{6 * 20 * 60 * 56:,} games ago).")
    # Best inside the last third: still improving recently.
    assert kinds(run_report("r", make_run([(1.0, 4.0)] * 7 + [(6.0, 4.0), (3.0, 4.0)])), "plateau") == []
    # The best exactly a third from the end counts.
    assert len(kinds(run_report("r", make_run([(1.0, 4.0)] * 5 + [(6.0, 4.0)] + [(3.0, 4.0)] * 3)), "plateau")) == 1
    # Too few evaluations to call it.
    assert kinds(run_report("r", make_run([(6.0, 4.0)] + [(3.0, 4.0)] * 6)), "plateau") == []


def test_loss_spike_fires_once_per_episode():
    report = run_report("r", make_run([(1.0, 5.0)] * 3, loss=lambda m: 3.51 if m == 30 else 1.17))
    [event] = kinds(report, "loss_spike")
    assert event["title"] == "Loss spike" and event["tone"] == "warning"
    assert event["detail"] == "Training loss reached 3.510 at step 7,800, 3.0 times its recent median of 1.170."

    report = run_report("r", make_run([(1.0, 5.0)] * 3, loss=lambda m: 2.4 if m in (30, 31) else 1.17))
    [event] = kinds(report, "loss_spike")
    assert "over 2 reports" in event["detail"]


def test_loss_noise_and_drift_are_not_spikes():
    rng = np.random.default_rng(5)
    noisy = {m: 1.17 + float(rng.normal(0, 0.02)) for m in range(1, 200)}
    assert kinds(run_report("r", make_run([(1.0, 5.0)] * 6, loss=noisy.get)), "loss_spike") == []
    # A loss that climbs steadily by a quarter over a run is a trend, not a spike.
    assert kinds(run_report("r", make_run([(1.0, 5.0)] * 6, loss=lambda m: 1.0 + m / 480)), "loss_spike") == []
    # Pretraining's falling loss at the start has too little history to be judged.
    assert kinds(run_report("r", warm_start()), "loss_spike") == []


def test_non_finite_loss():
    report = run_report("r", make_run([(1.0, 5.0)] * 2, loss=lambda m: float("nan") if m in (12, 13) else 1.17))
    [event] = [e for e in kinds(report, "loss_spike") if e["title"] == "Loss is not a number"]
    assert event["detail"] == "The training loss was NaN at step 4,920, the first of 2 reports with a non-finite loss."
    report = run_report("r", make_run([(1.0, 5.0)], loss=lambda m: float("inf") if m == 5 else 1.17))
    [event] = kinds(report, "loss_spike")
    assert event["detail"] == "The training loss was infinite at step 3,800."


def test_slowdown_needs_a_sustained_stretch():
    report = run_report("r", make_run([(1.0, 5.0)] * 3, gps=lambda m: 28.0 if 40 <= m < 45 else 56.0))
    [event] = kinds(report, "slowdown")
    assert event["detail"] == ("Self-play averaged 28.0 games/s over 5 consecutive reports (about 5 minutes), against "
                               "the run's typical 56.0.")
    # One slow report, then two in a row: routine.
    rate = lambda m: 20.0 if m in (10, 30, 31) else 56.0  # noqa: E731
    assert kinds(run_report("r", make_run([(1.0, 5.0)] * 3, gps=rate)), "slowdown") == []
    # Mild drift (10%) is not a slowdown.
    rate = lambda m: 56.0 if m < 30 else 50.0  # noqa: E731
    assert kinds(run_report("r", make_run([(1.0, 5.0)] * 3, gps=rate)), "slowdown") == []
    # Slow to the end.
    report = run_report("r", make_run([(1.0, 5.0)] * 3, gps=lambda m: 30.0 if m > 55 else 56.0))
    [event] = kinds(report, "slowdown")
    assert event["detail"].endswith(", and had not recovered by the last report.")


def test_records_reported_once_where_they_happened():
    plant = {30: {"max_game_score": 812, "max_play_score": 230}}
    report = run_report("r", make_run([(1.0, 5.0)] * 3, plant=plant))
    records = kinds(report, "record")
    assert [e["title"] for e in records] == ["Highest self-play game score", "Highest-scoring self-play move"]
    assert all(e["games"] == 30 * 3360 for e in records)
    assert records[0]["detail"] == "A self-play game reached 812 points, the highest of the run."
    assert records[1]["detail"] == "A single self-play move scored 230 points, the most of the run."
    # Older runs do not log these fields: nothing to report.
    assert kinds(run_report("r", make_run([(1.0, 5.0)], records_fields=False)), "record") == []


def test_sessions_stops_and_resume():
    records = [rec("session", 0, 0.0, 0, event="start")] + make_run([(1.0, 5.0)])
    g, h, s = records[-1]["games"], records[-1]["hours"], records[-1]["step"]
    records.append(rec("session", g, h, s, event="stop", reason="STOP file found", finished=False))
    records.append(rec("session", g, h, s, event="start"))
    later = make_run([(2.0, 5.0)], warm=False)
    for r in later:
        r["games"] += g
        r["hours"] = round(r["hours"] + h, 4)
        r["step"] += s
    records += later
    g2, h2 = records[-1]["games"], records[-1]["hours"]
    records.append(rec("session", g2, h2, 0, event="stop", reason="time limit of 0.3 h reached", finished=True))
    records.append(rec("session", g2, h2, 0, event="start"))
    records.append(rec("session", g2, h2, 0, event="stop", reason="killed from the app", finished=False))
    report = run_report("r", records)
    check_shape(report)
    stops = kinds(report, "stop")
    assert [(e["title"], e["tone"]) for e in stops] == [("Session stopped", "info"), ("Run finished", "info"),
                                                        ("Session stopped", "warning")]
    # 3 minutes of warm start and pretraining (0.05 h) plus 20 of self-play.
    assert stops[0]["detail"] == f"Stopped at {g:,} games after 23 minutes: STOP file found."
    resumed = [e for e in kinds(report, "phase") if e["title"] == "Run resumed"]
    assert len(resumed) == 2 and resumed[0]["games"] == g
    assert [e["title"] for e in kinds(report, "phase")].count("Self-play began") == 1


def test_quiet_run_has_only_ordinary_events():
    """A healthy, steady run with noisy evaluations: phases, the significance milestone and records
    are fine, but nothing alarming may be invented."""
    rng = np.random.default_rng(11)
    evals = [(2.0 + 0.25 * k + float(rng.normal(0, 1.5)), 4.0) for k in range(12)]
    rates = {m: 56.0 + float(rng.normal(0, 1.5)) for m in range(1, 300)}
    report = run_report("r", make_run(evals, gps=rates.get, gates=[None] + [False] * 11, seed=3))
    check_shape(report)
    assert {e["kind"] for e in report["events"]} <= {"phase", "significant", "record", "plateau"}
    assert all(e["tone"] != "warning" for e in report["events"])


# ---- the real dry run ------------------------------------------------------------------------

DRY_RUN_CANDIDATES = [
    os.environ.get("SCRABBLE_DRYRUN_METRICS", ""),
    str(Path(__file__).resolve().parents[1] / "runs" / "dryrun" / "metrics.jsonl"),
]


def test_real_dry_run_metrics():
    path = next((Path(p) for p in DRY_RUN_CANDIDATES if p and Path(p).is_file()), None)
    if path is None:
        pytest.skip("no dry-run metrics file on this machine")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = run_report("dryrun", records)
    check_shape(report)
    s = report["summary"]
    # Its interval straddles zero: no claim of significance anywhere.
    assert s["first_significant"] is None
    assert not says_significant(report["headline"])
    assert report["headline"].startswith("dryrun has not been shown to beat greedy-leave")
    assert kinds(report, "significant") == []
    assert s["final"]["ci"] > abs(s["final"]["spread"])
    assert s["promotions"] is None  # its gate record predates `promoted`
    assert 50 < s["mean_games_per_second"] < 65
    assert [e["title"] for e in kinds(report, "phase")][:2] == ["Warm start finished", "Self-play began"]


# ---- matches ---------------------------------------------------------------------------------

A, B = "night1 (best)", "Greedy + leave"


def match_games():
    """Three complete pairs and a trailing unpaired game. Spreads from A's side:
    pair 0: +50 first, -20 second (mean 15); pair 1: +10, 0 (mean 5); pair 2: -150, +200 (mean 25)."""
    return {
        "seed": [1, 1, 2, 2, 3, 3, 4],
        "a_first": [True, False, True, False, True, False, True],
        "score_a": [400, 380, 420, 390, 300, 500, 600],
        "score_b": [350, 400, 410, 390, 450, 300, 200],
        "turns": [22, 25, 24, 30, 21, 23, 26],
        "bingos_a": [2, 1, 3, 1, 0, 2, 4], "bingos_b": [1, 2, 2, 1, 3, 0, 0],
        "exchanges_a": [0, 1, 0, 0, 2, 0, 0], "exchanges_b": [0, 0, 1, 0, 0, 4, 0],
        "passes_a": [0, 0, 0, 1, 0, 0, 0], "passes_b": [0, 0, 0, 2, 0, 0, 0],
        "best_play_a": [98, 74, 212, 50, 40, 110, 150], "best_play_b": [83, 92, 60, 66, 120, 45, 30],
        "best_word_a": ["RETAINS", "QUIZ", "QUIXOTIC", "CAT", "AX", "RETINA", "ORIENTS"],
        "best_word_b": ["ZANIEST", "OUTJINX", "JO", "DOG", "STAINER", "OX", "ZA"],
        "ended_by": ["out", "out", "out", "scoreless", "out", "out", "out"],
    }


def test_match_summary_by_hand():
    summary, _ = match_summary(match_games(), A, B)
    assert MATCH_KEYS <= set(summary)
    assert summary["pairs"] == 3
    assert summary["spread"] == pytest.approx(15.0)
    assert summary["spread_ci"] == pytest.approx(1.96 * 10 / math.sqrt(3), abs=1e-3)  # per-pair sd is 10
    assert summary["win_rate"] == pytest.approx(7 / 12, abs=1e-4)  # pairs win 0.5, 0.75, 0.5
    assert summary["win_ci"] == pytest.approx(1.96 * math.sqrt(0.0625 / 3) / math.sqrt(3), abs=1e-4)
    assert summary["significant"] is True
    assert summary["score_a"] == pytest.approx(2390 / 6, abs=0.01)
    assert summary["score_b"] == pytest.approx(2300 / 6, abs=0.01)
    assert summary["turns"] == pytest.approx(145 / 6, abs=0.01)
    assert summary["a_first"] == {"spread": -30.0, "win_rate": pytest.approx(2 / 3, abs=1e-4)}
    assert summary["a_second"] == {"spread": 60.0, "win_rate": 0.5}
    assert summary["bingos_a"] == 1.5 and summary["bingos_b"] == 1.5
    assert summary["exchanges_a"] == 0.5 and summary["exchanges_b"] == pytest.approx(5 / 6, abs=1e-3)
    assert summary["passes_a"] == pytest.approx(1 / 6, abs=1e-3) and summary["passes_b"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["ended_by_going_out"] == pytest.approx(5 / 6, abs=1e-4)
    json.dumps(summary)


def test_match_summary_partial():
    games = match_games()
    three = {k: v[:3] for k, v in games.items()}  # one complete pair and one game of the next
    summary, highlights = match_summary(three, A, B)
    assert summary is None
    assert highlights and all(0 <= h["game"] < 3 for h in highlights)
    four = {k: v[:4] for k, v in games.items()}
    assert match_summary(four, A, B)[0] is not None  # two complete pairs is enough
    assert match_summary({k: [] for k in games}, A, B) == (None, [])


def test_inconsistent_pairs_are_left_out():
    games = match_games()
    games["seed"][3] = 99  # pair 1 no longer shares a seed
    summary, _ = match_summary(games, A, B)
    assert summary["pairs"] == 2 and summary["spread"] == pytest.approx(20.0)
    games = match_games()
    games["a_first"][1] = True  # pair 0: A moved first in both games
    summary, _ = match_summary(games, A, B)
    assert summary["pairs"] == 2 and summary["spread"] == pytest.approx(15.0)
    games = match_games()
    del games["a_first"], games["seed"]  # no seat or seed columns: the runner's order (A first in even games)
    assert match_summary(games, A, B)[0]["spread"] == pytest.approx(15.0)


def test_match_highlights():
    _, highlights = match_summary(match_games(), A, B)
    got = [(h["title"], h["detail"], h["game"]) for h in highlights]
    assert got == [
        ("Highest-scoring play", "QUIXOTIC for 212 points by night1 (best).", 2),
        ("Biggest win", "night1 (best) won 600-200, a margin of 400 points; 600 is also the highest score in the match.", 6),
        ("Closest game", "Tied at 390.", 3),
        ("Most bingos", "5 bingos in one game: 3 by night1 (best) and 2 by Greedy + leave.", 2),
        ("Biggest swing in a pair", "With the same seed and the seats swapped, night1 (best) lost by 150 moving first "
                                    "and won by 200 moving second, a swing of 350 points.", 4),
        ("Longest game", "30 turns against a match average of 24.4; tied at 390.", 3),
        ("Ended without going out", "Ended on six scoreless turns in a row, with night1 (best) on 390 and "
                                    "Greedy + leave on 390.", 3),
        ("Most exchanges", "Greedy + leave exchanged 4 times in one game; night1 (best) won 500-300.", 5),
    ]


def test_match_highlights_skip_what_is_not_there():
    games = match_games()
    games["bingos_a"] = [0] * 7
    games["bingos_b"] = [1, 0, 0, 0, 0, 0, 0]  # never more than one bingo in a game
    games["ended_by"] = ["out"] * 7
    games["exchanges_b"] = [0] * 7
    games["score_b"][3] = 389  # no tie: the closest game is won by one point
    del games["turns"]
    titles = {h["title"]: h for h in match_summary(games, A, B)[1]}
    assert "Most bingos" not in titles and "Ended without going out" not in titles
    assert "Longest game" not in titles and "Most exchanges" not in titles
    assert titles["Closest game"]["detail"] == "night1 (best) won 390-389, by 1 point."
    games["best_word_a"] = [""] * 7
    titles = {h["title"]: h for h in match_summary(games, A, B)[1]}
    assert titles["Highest-scoring play"]["detail"] == "A 212-point play by night1 (best)."


def test_match_against_itself_names_the_sides():
    _, highlights = match_summary(match_games(), "night1 (best)", "night1 (best)")
    assert highlights[0]["detail"] == "QUIXOTIC for 212 points by night1 (best) (A)."


def test_fallback_statistics_match_evaluate():
    from scrabble_ai.evaluate import summarize

    rng = np.random.default_rng(2)
    for n in (2, 3, 50):
        sf = rng.integers(-120, 120, n).astype(np.float64)
        ss = rng.integers(-120, 120, n).astype(np.float64)
        ss[0] = 0.0  # a tie
        sa = rng.integers(300, 500, 2 * n).astype(np.float64)
        sb = rng.integers(300, 500, 2 * n).astype(np.float64)
        ours = insights._summarize_here(sf, ss, sa, sb)
        theirs = summarize("A", "B", sf, ss, sa, sb, np.zeros(2 * n))
        assert ours == pytest.approx((theirs.spread, theirs.spread_ci, theirs.win_rate, theirs.win_ci,
                                      theirs.score_a, theirs.score_b))


# ---- health, and the events of a run that got worse -------------------------------------------


def worse_edge(from_minute, value=77.0, leave=74.0, minutes=400):
    """Self-play reports from `from_minute` on in which the network predicts new games worse than
    leave values alone."""
    return {m: {"value_rmse_points": value, "leave_only_rmse_points": leave} for m in range(from_minute, minutes)}


def test_health_is_null_before_the_first_evaluation():
    assert run_report("r", [])["summary"]["health"] is None
    assert run_report("r", make_run([], extra_minutes=30))["summary"]["health"] is None
    overview = run_overview(make_run([], extra_minutes=30))
    assert overview == {"trend": {"games": [], "spread": [], "ci": []}, "best_eval": None, "health": None}


@pytest.mark.parametrize("evals, state, start", [
    ([(2.0, 5.0), (9.0, 4.0)], "ahead", "Ahead of greedy-leave: +9.0 +/- 4.0"),
    ([(2.0, 5.0), (3.0, 5.0)], "level", "Level with greedy-leave: +3.0 +/- 5.0"),
    ([(-9.0, 5.0), (-8.0, 5.0)], "behind", "Behind greedy-leave: -8.0 +/- 5.0"),
    ([(-3.0, 7.0), (-47.0, 8.0), (-45.0, 8.0)], "worse", "It got worse as it trained: -45.0 +/- 8.0"),
    # Below its best by less than noise between two evaluations explains: not "worse".
    ([(2.0, 6.0), (-7.0, 5.0)], "behind", "Behind greedy-leave"),
    # Still significantly ahead, although below its best: ahead, and the note says so.
    ([(40.0, 5.0), (10.0, 5.0)], "ahead", "Ahead of greedy-leave: +10.0 +/- 5.0"),
])
def test_health_states(evals, state, start):
    health = run_report("r", make_run(evals))["summary"]["health"]
    assert set(health) == {"state", "edge", "note"}
    assert health["state"] == state and health["note"].startswith(start), health
    assert health["edge"] == 1.0, "the mean of leave-only error minus the network's, over the last ten reports"
    assert run_overview(make_run(evals))["health"] == health


def test_health_of_a_run_that_was_further_ahead_says_so():
    health = run_report("r", make_run([(40.0, 5.0), (10.0, 5.0)]))["summary"]["health"]
    assert health["note"].endswith("It was further ahead earlier (+40.0 +/- 5.0 at 67,200 games).")


def test_a_run_that_got_worse_is_described_as_such():
    records = make_run([(-3.0, 7.0), (-47.0, 8.0), (-45.0, 8.0)], gates=[None, False, False], plant=worse_edge(15))
    # The trainer logs the first evaluated network as a gate record with no spread and no pairs.
    first = next(i for i, r in enumerate(records) if r["kind"] == "eval")
    records.insert(first + 2, rec("gate", records[first]["games"], records[first]["hours"], records[first]["step"],
                                  spread=None, spread_ci=None, win_rate=None, best_step=records[first]["step"],
                                  promoted=True, pairs=0))
    report = run_report("night1", records)
    check_shape(report)
    assert report["headline"] == ("night1 got worse as it trained: its latest evaluation against greedy-leave is "
                                  "-45.0 +/- 8.0 points a game, down from -3.0 +/- 7.0 at 67,200 games. It was never "
                                  "shown to beat greedy-leave.")
    health = report["summary"]["health"]
    assert health["state"] == "worse" and health["edge"] == -3.0
    assert health["note"] == ("It got worse as it trained: -45.0 +/- 8.0 points a game against greedy-leave at the "
                              "latest evaluation, down from -3.0 +/- 7.0 at 67,200 games. Its best network is still "
                              "the first one evaluated. On games it has not trained on, its predictions are 3.0 points "
                              "of RMSE worse than leave values alone, as they are when a network memorises its "
                              "training games.")
    [event] = kinds(report, "overfit")
    assert event["tone"] == "warning" and event["title"] == "Predicting new games worse than leave values alone"
    # The reports go bad at minute 15. The mean of ten reports is first below -0.5 at minute 18, when four
    # of the ten are bad: (6 x +1 + 4 x -3) / 10 = -0.6, the network's error averaging (6 x 73 + 4 x 77) / 10.
    assert "averaged 74.6 points of RMSE, against 74.0 for leave values alone" in event["detail"]
    assert event["hours"] == round(0.05 + 18 / 60, 4) and event["hours"] < records[first]["hours"]


def test_no_overfit_event_while_the_network_predicts_new_games_better():
    report = run_report("r", make_run([(2.0, 5.0)] * 4))
    assert kinds(report, "overfit") == []
    # Fewer than ten reports are too few to say, however bad they look.
    report = run_report("r", make_run([], extra_minutes=9, plant=worse_edge(1)))
    assert kinds(report, "overfit") == []
    # A short dip does not average below the margin over ten reports.
    report = run_report("r", make_run([], extra_minutes=40, plant={m: {"value_rmse_points": 76.0} for m in (5, 6, 7)}))
    assert kinds(report, "overfit") == []


def test_the_overfit_event_is_reported_once():
    report = run_report("r", make_run([], extra_minutes=120, plant=worse_edge(30)))
    assert len(kinds(report, "overfit")) == 1


def test_guard_records_become_events():
    records = make_run([(-3.0, 7.0), (-30.0, 7.0)])
    last = records[-1]
    records.append(rec("guard", last["games"], last["hours"], last["step"], action="rollback", to_step=6200,
                       learning_rate_scale=0.5,
                       reason="2 evaluations in a row significantly behind best@6200 (latest -28.0 +/- 6.0)"))
    records.append(rec("guard", last["games"] + 5000, last["hours"] + 0.5, last["step"] + 900, action="rollback"))
    records.append("not a record")
    report = run_report("r", records)
    check_shape(report)
    first, second = kinds(report, "guard")
    assert first["tone"] == "warning" and first["title"] == "Guard went back to the best network"
    assert first["detail"] == ("Training had made the network worse, so the trainer went back to its best network "
                               "(step 6,200) and lowered the learning rate to 0.5 times its setting. 2 evaluations in a "
                               "row significantly behind best@6200 (latest -28.0 +/- 6.0).")
    assert second["detail"] == "Training had made the network worse, so the trainer went back to its best network."


def test_run_overview_gives_the_trend_and_the_best_evaluation():
    records = make_run([(-3.0, 7.0), (4.0, 6.0), (1.0, None)])
    overview = run_overview(records)
    assert overview["trend"] == {"games": [67200, 134400, 201600], "spread": [-3.0, 4.0, 1.0], "ci": [7.0, 6.0, None]}
    assert overview["best_eval"] == {"games": 134400, "step": 3000 + 160 * 40, "spread": 4.0, "ci": 6.0}
    assert overview["health"]["state"] == "level"
    # Evaluations against other opponents are not part of the trend.
    assert len(run_overview(make_run([(1.0, 5.0)] * 3, greedy=True))["trend"]["games"]) == 3


@pytest.mark.parametrize("records", [None, "nonsense", [None, 3, "x"], [{"kind": "eval"}], [{"kind": "eval", "opponent": "greedy-leave", "spread": "abc"}],
                                     [{"kind": "selfplay", "value_rmse_points": float("nan")}]])
def test_run_overview_never_raises(records):
    assert run_overview(records) == {"trend": {"games": [], "spread": [], "ci": []}, "best_eval": None, "health": None}
