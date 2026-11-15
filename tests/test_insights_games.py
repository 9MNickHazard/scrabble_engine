"""`insights.games_overview` and `insights.games_report`: runs that train from saved games are
judged by their win rate against the static evaluator (docs/APP_API.md, "Training from saved
games"). Records are built by hand, so every expected number and sentence is known."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scrabble_ai.app.insights import games_overview, games_report

EVENT_KEYS = {"games", "hours", "kind", "tone", "title", "detail"}
SUMMARY_KEYS = {"steps", "positions", "hours", "passes", "game_set", "train_games", "mean_positions_per_second",
                "best", "final", "first_significant", "held_out", "evaluations", "health"}
MATCH_KEYS = {"games", "step", "hours", "win_rate", "win_ci", "spread", "ci", "pairs"}
TREND_KEYS = {"games", "spread", "ci", "step", "win_rate", "win_ci"}
BATCH = 2048


def rec(kind, step=0, hours=0.0, **fields):
    return {"kind": kind, "time": 1794700000.0 + hours * 3600, "step": step, "games": step * BATCH,
            "positions": step * BATCH, "hours": hours, **fields}


def start(step=0, hours=0.0):
    return rec("session", step, hours, event="start", mode="games", parameters=610_000, limit_hours=0.0,
               game_set="twenty-million", train_games=19_980_000, validation_games=20_000, total_steps=58_500,
               steps_per_pass=9_750, batch=BATCH)


def match(step, hours, rate, ci=0.013, spread=1.0, spread_ci=5.0, pairs=2000, opponent="greedy-leave"):
    return rec("eval", step, hours, opponent=opponent, spread=spread, spread_ci=spread_ci, win_rate=rate, win_ci=ci,
               score_for=420.0, score_against=419.0, pairs=pairs)


def trained(step, hours, rate=40_000.0):
    return rec("train", step, hours, loss=1.0, win_loss=0.66, spread_loss=0.3, learning_rate=1e-4,
               positions_per_second=rate, **{"pass": round(step / 9_750, 3)})


def run(rates, cis=None, best_records=True):
    """A games run with one match per rate, a train record before each, and best records."""
    records = [start()]
    best = -1.0
    for i, rate in enumerate(rates):
        step, hours = 4_000 * (i + 1), 0.5 * (i + 1)
        records.append(trained(step, hours - 0.1, 40_000.0 + 100 * i))
        ci = 0.013 if cis is None else cis[i]
        records.append(match(step, hours, rate, ci))
        if best_records and rate > best:
            best = rate
            records.append(rec("best", step, hours, win_rate=rate, win_ci=ci, pairs=2000))
    return records


# ---- overview ----------------------------------------------------------------------------------


def test_overview_before_any_match():
    for records in ([], [start()], [start(), trained(100, 0.1)]):
        overview = games_overview(records)
        assert overview == {"trend": {key: [] for key in TREND_KEYS}, "best_eval": None, "health": None}


def test_overview_trend_and_best():
    overview = games_overview(run([0.51, 0.56, 0.55]))
    assert set(overview) == {"trend", "best_eval", "health"} and set(overview["trend"]) == TREND_KEYS
    trend = overview["trend"]
    assert trend["step"] == [4_000, 8_000, 12_000] and trend["games"] == [4_000 * BATCH, 8_000 * BATCH, 12_000 * BATCH]
    assert trend["win_rate"] == [0.51, 0.56, 0.55] and trend["win_ci"] == [0.013] * 3
    assert trend["spread"] == [1.0] * 3 and trend["ci"] == [5.0] * 3
    assert overview["best_eval"] == {"games": 8_000 * BATCH, "step": 8_000, "hours": 1.0, "win_rate": 0.56, "win_ci": 0.013,
                                     "spread": 1.0, "ci": 5.0, "pairs": 2000}
    # Ties keep the earliest.
    assert games_overview(run([0.53, 0.53]))["best_eval"]["step"] == 4_000
    # Only matches against the static evaluator count.
    records = run([0.52]) + [match(8_000, 1.0, 0.9, opponent="greedy")]
    assert games_overview(records)["trend"]["win_rate"] == [0.52]


@pytest.mark.parametrize("rates,cis,state", [
    ([0.50, 0.534], None, "ahead"),             # 0.534 - 0.013 > 0.5
    ([0.50, 0.51], None, "level"),              # the interval includes one half
    ([0.50, 0.48], None, "behind"),             # 0.48 + 0.013 < 0.5
    ([0.55, 0.50], None, "worse"),              # 0.05 > 1.3 * hypot(0.013, 0.013) and not ahead
    ([0.60, 0.55], None, "ahead"),              # fell, but still ahead
    ([0.52, 0.51], [0.013, 0.013], "level"),    # a small drop is noise
    ([0.60], [None], "level"),                  # no interval: never called ahead
])
def test_health_by_win_rate(rates, cis, state):
    health = games_overview(run(rates, cis))["health"]
    assert set(health) == {"state", "edge", "note"} and health["edge"] is None
    assert health["state"] == state, health["note"]
    assert health["note"].endswith(".") and "against the static evaluator" in health["note"]


def test_health_notes():
    assert games_overview(run([0.534]))["health"]["note"] == (
        "Wins 53.4% +/- 1.3 of games against the static evaluator at the latest match: more than half, beyond the "
        "margin of error.")
    assert games_overview(run([0.55, 0.50]))["health"]["note"] == (
        "It got worse as it trained: 50.0% +/- 1.3 of games against the static evaluator at the latest match, down "
        "from 55.0% +/- 1.3 of games at step 4,000. The best network is kept.")
    assert games_overview(run([0.60, 0.55]))["health"]["note"].endswith(
        "It won more earlier (60.0% +/- 1.3 of games at step 4,000).")
    assert games_overview(run([0.48]))["health"]["note"] == (
        "Wins 48.0% +/- 1.3 of games against the static evaluator at the latest match: less than half, beyond the "
        "margin of error.")
    assert games_overview(run([0.51]))["health"]["note"] == (
        "Wins 51.0% +/- 1.3 of games against the static evaluator at the latest match: not distinguishable from half.")


@pytest.mark.parametrize("records", [
    None, "not records", 42, [1, "x", None, []], [{"kind": "eval"}],
    [{"kind": "eval", "opponent": "greedy-leave", "win_rate": "abc", "win_ci": None}],
    [{"kind": "eval", "opponent": "greedy-leave", "win_rate": float("nan"), "step": "x"}],
    [{"kind": "eval", "opponent": "greedy-leave", "win_rate": 0.6, "win_ci": -1.0, "games": float("inf")}],
    [{"kind": 5, "opponent": None}, {"kind": "session", "event": "start", "steps_per_pass": 0}],
])
def test_odd_input_never_raises(records):
    overview = games_overview(records)
    assert set(overview) == {"trend", "best_eval", "health"}
    report = games_report("odd", records, {"offline": "not a dict"})
    assert set(report) == {"headline", "summary", "events"} and isinstance(report["headline"], str)


def test_odd_values_are_skipped_not_believed():
    records = [{"kind": "eval", "opponent": "greedy-leave", "win_rate": 0.6, "win_ci": -1.0, "step": 3}]
    overview = games_overview(records)
    assert overview["trend"]["win_rate"] == [0.6] and overview["trend"]["win_ci"] == [None]
    assert overview["health"]["state"] == "level", "a negative interval is no interval"
    assert games_overview([{"kind": "eval", "opponent": "greedy-leave", "win_rate": float("nan")}])["health"] is None


# ---- report ------------------------------------------------------------------------------------


def full_run():
    records = run([0.50, 0.534, 0.52])
    records.insert(3, rec("valid", 4_000, 0.45, win_loss=0.70, spread_loss=0.31, win_accuracy=0.58, win_brier=0.24,
                          value_rmse_points=72.0, leave_only_rmse_points=74.0, positions_checked=20_000))
    records.append(rec("valid", 12_000, 1.5, win_loss=0.6521, spread_loss=0.3, win_accuracy=0.612, win_brier=0.23,
                       value_rmse_points=71.3, leave_only_rmse_points=73.9, positions_checked=20_000))
    records.append(rec("checkpoint", 12_000, 1.5, file="step_000012000.pt"))
    records.append(rec("session", 12_000, 1.5, event="stop", reason="STOP file found", finished=False))
    return records


def test_the_report_summary():
    report = games_report("night5", full_run(), None)
    summary = report["summary"]
    assert set(summary) == SUMMARY_KEYS
    assert (summary["steps"], summary["positions"], summary["hours"]) == (12_000, 12_000 * BATCH, 1.5)
    assert summary["passes"] == round(12_000 / 9_750, 2) and summary["game_set"] == "twenty-million"
    assert summary["train_games"] == 19_980_000
    assert summary["mean_positions_per_second"] == pytest.approx(np.mean([40_000.0, 40_100.0, 40_200.0]))
    for key in ("best", "final", "first_significant"):
        assert set(summary[key]) == MATCH_KEYS, key
    assert summary["best"]["win_rate"] == 0.534 and summary["best"]["step"] == 8_000
    assert summary["final"]["win_rate"] == 0.52 and summary["final"]["step"] == 12_000
    assert summary["first_significant"]["step"] == 8_000
    assert summary["held_out"] == {"win_loss": 0.6521, "win_accuracy": 0.612, "value_rmse_points": 71.3,
                                   "leave_only_rmse_points": 73.9}, "the latest check"
    assert summary["evaluations"] == 3 and summary["health"]["state"] == "ahead", "0.52 - 0.013 is above one half"


def test_the_report_headlines():
    assert games_report("night5", run([0.534]))["headline"] == (
        "night5 wins 53.4% +/- 1.3 of games against the static evaluator, more than half beyond the margin of error.")
    assert games_report("n", run([0.48]))["headline"] == (
        "n wins 48.0% +/- 1.3 of games against the static evaluator: less than half.")
    assert games_report("n", run([0.51]))["headline"] == (
        "n has not been shown to beat the static evaluator: it wins 51.0% +/- 1.3 of games, which is not "
        "distinguishable from half.")
    assert games_report("n", run([0.534, 0.51]))["headline"].endswith(
        " Its best match was 53.4% +/- 1.3 of games at step 4,000.")
    assert games_report("n", [start(), trained(500, 0.2)])["headline"] == (
        "n has trained for 500 steps and has not played the static evaluator yet.")
    assert games_report("n", [])["headline"] == "n has not reported any results yet."
    assert games_report("", [])["headline"] == "This run has not reported any results yet."


def test_the_report_before_anything_happened():
    summary = games_report("n", [start()])["summary"]
    assert set(summary) == SUMMARY_KEYS
    assert summary["best"] is None and summary["final"] is None and summary["first_significant"] is None
    assert summary["held_out"] is None and summary["evaluations"] == 0 and summary["health"] is None
    assert summary["mean_positions_per_second"] is None and summary["passes"] == 0.0


def test_the_report_events():
    records = full_run()
    records.append(start(12_000, 1.5))  # resumed
    records.append(rec("session", 12_500, 1.7, event="stop", reason="all 58,500 steps done", finished=True))
    records.append(rec("session", 12_500, 1.8, event="stop", reason="killed from the app", finished=False))
    events = games_report("night5", records)["events"]
    for event in events:
        assert set(event) == EVENT_KEYS
    assert [e["hours"] for e in events] == sorted(e["hours"] for e in events)
    kinds = [(e["kind"], e["title"]) for e in events]
    assert kinds[0] == ("phase", "Training began")
    assert ("phase", "Run resumed") in kinds
    assert ("significant", "First win rate clearly above half") in kinds
    assert [t for k, t in kinds if k == "promotion"] == ["New best network", "New best network"]
    stops = [e for e in events if e["kind"] == "stop"]
    assert [(e["title"], e["tone"]) for e in stops] == [("Session stopped", "info"), ("Run finished", "info"),
                                                        ("Session stopped", "warning")]
    assert stops[0]["detail"] == "Stopped at step 12,000 after 1.5 hours: STOP file found."
    began = events[0]
    assert began["detail"] == "Learning from 19,980,000 games of the set twenty-million, one position from each per pass."
    significant = next(e for e in events if e["kind"] == "significant")
    assert significant["detail"] == "53.4% +/- 1.3 of games against the static evaluator over 2,000 pairs."
    assert significant["games"] == 8_000 * BATCH and significant["hours"] == 1.0
    promotion = [e for e in events if e["kind"] == "promotion"][-1]
    assert promotion["detail"] == "Won 53.4% +/- 1.3 of games against the static evaluator, the most so far."
    assert {e["kind"] for e in events} <= {"phase", "significant", "promotion", "stop"}


def test_the_report_numbers_are_json():
    records = full_run()
    records.append(trained(13_000, 1.6, rate=float("nan")))
    records.append(match(13_000, 1.6, float("inf")))
    report = games_report("n", records)

    def walk(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(report)
    walk(games_overview(records))
