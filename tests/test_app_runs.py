"""`scrabble_ai.app.runs` and `queue_runner`: run status from the files a run leaves behind,
incremental metrics, and starting / stopping / resuming / deleting (with nothing really started).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from scrabble_ai.app import procs, queue_runner
from scrabble_ai.app import runs as runs_module
from scrabble_ai.app.runs import RunError, RunManager, flatten, pid_alive
from scrabble_ai.config import Config

pytest.register_assert_rewrite("app_fixtures")
from app_fixtures import DEAD_PID, TINY_RUN, copy_run, make_run, record, write_records  # noqa: E402

pytest_plugins = ["app_fixtures"]

NOW = time.time()


@pytest.fixture(autouse=True)
def _no_real_processes(spawned):
    """Every test here runs with spawn_detached and kill_process replaced."""


@pytest.fixture
def dirs(tmp_path):
    runs_dir, app_dir = tmp_path / "runs", tmp_path / "app"
    runs_dir.mkdir()
    app_dir.mkdir()
    return runs_dir, app_dir


@pytest.fixture
def manager(dirs):
    return RunManager(*dirs)


def status(manager, name):
    summary = manager.summary(name)
    return summary["status"], summary["error"]


def test_the_dead_pid_is_dead_and_this_process_is_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(DEAD_PID)
    assert not pid_alive(0) and not pid_alive(-4)


# ---- status ------------------------------------------------------------------------------------


def test_running_and_stopping(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", records=[record("session", NOW - 10, event="start")], pid=os.getpid())
    assert status(manager, "r") == ("running", None)
    (runs_dir / "r" / "STOP").write_text("", encoding="utf-8")
    assert status(manager, "r") == ("stopping", None)


def test_a_dead_process_is_failed_with_the_tail_of_its_log(manager, dirs):
    runs_dir, _ = dirs
    lines = [f"line {i}" for i in range(30)]
    make_run(runs_dir, "r", records=[record("session", NOW - 10, event="start")], pid=DEAD_PID, console=lines)
    state, error = status(manager, "r")
    assert state == "failed"
    assert error == "\n".join(lines[-12:])
    # No console log: a generic message instead.
    make_run(runs_dir, "quiet", pid=DEAD_PID)
    assert status(manager, "quiet") == ("failed", "The training process ended unexpectedly.")


def test_a_running_json_with_a_dead_pid_and_a_stop_file_is_failed_not_stopping(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", pid=DEAD_PID, stop=True)
    assert status(manager, "r")[0] == "failed"


def test_queued_while_the_queue_runner_lives(manager, dirs):
    runs_dir, app_dir = dirs
    (app_dir / "queues").mkdir()
    (app_dir / "queues" / "queue_x.pid").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    make_run(runs_dir, "q", queued={"queue": "queue_x", "overrides": {}, "created": NOW - 3600})
    assert status(manager, "q") == ("queued", None)


def test_queued_during_the_grace_period_before_the_runner_writes_its_pid(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "fresh", queued={"queue": "queue_y", "overrides": {}, "created": time.time() - 25})
    assert status(manager, "fresh") == ("queued", None)
    make_run(runs_dir, "late", queued={"queue": "queue_y", "overrides": {}, "created": time.time() - 31})
    state, error = status(manager, "late")
    assert state == "stopped" and "queue was interrupted" in error


def test_an_interrupted_queue(manager, dirs):
    runs_dir, app_dir = dirs
    (app_dir / "queues").mkdir()
    (app_dir / "queues" / "queue_z.pid").write_text(json.dumps({"pid": DEAD_PID}), encoding="utf-8")
    make_run(runs_dir, "q", queued={"queue": "queue_z", "overrides": {"model.blocks": 3}, "created": NOW - 3600})
    state, error = status(manager, "q")
    assert state == "stopped"
    assert error == "The queue was interrupted before this run started. Resume starts it."
    # Its settings come from the marker while it has no config.toml.
    assert manager.config("q")["model"]["blocks"] == 3
    assert manager.summary("q")["network"] == "3x64"


def test_finished_and_stopped(manager, dirs):
    runs_dir, _ = dirs
    start = record("session", NOW - 100, event="start")
    make_run(runs_dir, "done", records=[start, record("session", NOW - 50, event="stop", reason="game limit", finished=True)])
    assert status(manager, "done") == ("finished", None)
    make_run(runs_dir, "halted", records=[start, record("session", NOW - 50, event="stop", reason="STOP file found", finished=False)])
    assert status(manager, "halted") == ("stopped", None)
    # Finished once, then resumed and stopped by the user: the last stop record decides.
    make_run(runs_dir, "again", records=[start, record("session", NOW - 90, event="stop", finished=True), start,
                                          record("session", NOW - 10, event="stop", finished=False)])
    assert status(manager, "again") == ("stopped", None)
    make_run(runs_dir, "nosession", records=[record("train", NOW - 10, loss=1.0)])
    assert status(manager, "nosession") == ("stopped", None)
    make_run(runs_dir, "empty")
    assert status(manager, "empty") == ("stopped", None)


# ---- phase, progress, evaluation, speed --------------------------------------------------------


def test_phase(manager, dirs):
    runs_dir, _ = dirs
    start = record("session", NOW - 100, event="start")
    make_run(runs_dir, "boot", pid=os.getpid())
    assert manager.summary("boot")["phase"] == "warmstart"
    make_run(runs_dir, "warm", records=[start], pid=os.getpid())
    assert manager.summary("warm")["phase"] == "warmstart"
    make_run(runs_dir, "pre", records=[start, record("warmstart", NOW - 50, games_played=100)], pid=os.getpid())
    assert manager.summary("pre")["phase"] == "pretraining"
    make_run(runs_dir, "began", records=[start, record("warmstart", NOW - 50), record("session", NOW - 40, event="selfplay")],
             pid=os.getpid())
    assert manager.summary("began")["phase"] == "selfplay"
    make_run(runs_dir, "sp", records=[start, record("selfplay", NOW - 40, games=10, games_per_second=1.0)], pid=os.getpid())
    assert manager.summary("sp")["phase"] == "selfplay"
    make_run(runs_dir, "never", records=[])
    assert manager.summary("never")["phase"] is None


def test_progress_against_a_time_limit(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"run.hours": 2.0}, pid=os.getpid(),
             records=[record("session", time.time() - 1800, event="start")])
    summary = manager.summary("r")
    assert summary["progress"] == pytest.approx(0.25, abs=0.01)
    assert summary["limit"] == {"hours": 2.0, "max_games": 0}


def test_progress_against_a_game_limit_and_both(manager, dirs):
    runs_dir, _ = dirs
    start = record("session", time.time() - 1800, event="start")
    make_run(runs_dir, "games", settings={"run.max_games": 1000}, pid=os.getpid(),
             records=[start, record("selfplay", time.time(), games=250, games_per_second=5.0)])
    assert manager.summary("games")["progress"] == pytest.approx(0.25)
    make_run(runs_dir, "both", settings={"run.max_games": 1000, "run.hours": 2.0}, pid=os.getpid(),
             records=[start, record("selfplay", time.time(), games=500, games_per_second=5.0)])
    assert manager.summary("both")["progress"] == pytest.approx(0.5)
    make_run(runs_dir, "over", settings={"run.max_games": 1000}, pid=os.getpid(),
             records=[start, record("selfplay", time.time(), games=1500, games_per_second=5.0)])
    assert manager.summary("over")["progress"] == 1.0
    make_run(runs_dir, "nolimit", pid=os.getpid(), records=[start])
    assert manager.summary("nolimit")["progress"] is None
    make_run(runs_dir, "finished", settings={"run.max_games": 1000},
             records=[start, record("session", time.time(), games=1000, event="stop", finished=True)])
    assert manager.summary("finished")["progress"] == 1.0


def evaluation(t, games, opponent, spread, old=False, step=0):
    if old:
        return record("eval", t, step=step, games=games, opponent=opponent, **{
            f"spread_vs_{opponent}": spread, f"spread_ci_vs_{opponent}": 2.0, f"win_rate_vs_{opponent}": 0.6})
    return record("eval", t, step=step, games=games, opponent=opponent, spread=spread, spread_ci=2.0, win_rate=0.6, pairs=10)


def test_latest_eval_in_both_key_layouts(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "new", records=[evaluation(1, 100, "greedy-leave", 1.0), evaluation(2, 100, "greedy", 30.0),
                                       evaluation(3, 200, "greedy-leave", 4.0), evaluation(4, 200, "greedy", 40.0)])
    assert manager.summary("new")["latest_eval"] == {
        "games": 200, "greedy-leave": {"spread": 4.0, "ci": 2.0, "win_rate": 0.6},
        "greedy": {"spread": 40.0, "ci": 2.0, "win_rate": 0.6}}
    make_run(runs_dir, "old", records=[evaluation(1, 100, "greedy-leave", 1.5, old=True),
                                       evaluation(2, 100, "greedy", 31.0, old=True)])
    assert manager.summary("old")["latest_eval"] == {
        "games": 100, "greedy-leave": {"spread": 1.5, "ci": 2.0, "win_rate": 0.6},
        "greedy": {"spread": 31.0, "ci": 2.0, "win_rate": 0.6}}
    make_run(runs_dir, "none", records=[record("train", 1, loss=1.0)])
    assert manager.summary("none")["latest_eval"] is None
    # The newest evaluation only covered greedy-leave: the older greedy figure is not carried over.
    make_run(runs_dir, "partial", records=[evaluation(1, 100, "greedy-leave", 1.0), evaluation(2, 100, "greedy", 30.0),
                                           evaluation(3, 200, "greedy-leave", 4.0)])
    assert manager.summary("partial")["latest_eval"] == {"games": 200, "greedy-leave": {"spread": 4.0, "ci": 2.0, "win_rate": 0.6}}


def test_latest_eval_after_the_game_counter_went_back(manager, dirs):
    """The latest evaluation is the last in the log, even when a forced stop and resume put the game
    counter back below an earlier evaluation's (it once took the one with the most games)."""
    runs_dir, _ = dirs
    make_run(runs_dir, "r", records=[
        evaluation(1, 1000, "greedy-leave", 5.0), evaluation(2, 1000, "greedy", 40.0),
        record("session", 3, games=1000, event="stop", reason="killed from the app", finished=False),
        record("session", 4, games=900, event="start"),  # resumed from the checkpoint at 900 games
        evaluation(5, 950, "greedy-leave", 2.0), evaluation(6, 950, "greedy", 35.0)])
    assert manager.summary("r")["latest_eval"] == {
        "games": 950, "greedy-leave": {"spread": 2.0, "ci": 2.0, "win_rate": 0.6},
        "greedy": {"spread": 35.0, "ci": 2.0, "win_rate": 0.6}}


def test_games_per_second(manager, dirs):
    runs_dir, _ = dirs
    t0 = time.time() - 100
    selfplay = [record("selfplay", t0 + 60, games=600, games_per_second=10.0),
                record("selfplay", t0 + 120, games=1800, games_per_second=20.0)]
    make_run(runs_dir, "live", records=[record("session", t0, event="start"), *selfplay], pid=os.getpid())
    assert manager.summary("live")["games_per_second"] == 20.0, "the latest rate while running"
    make_run(runs_dir, "done", records=[record("session", t0, event="start"), *selfplay,
                                        record("session", t0 + 130, games=1800, event="stop", finished=True)])
    assert manager.summary("done")["games_per_second"] == 15.0, "the average afterwards"
    # Before the first self-play report: worked out from the game counter since self-play began.
    began = record("session", t0, games=100, event="selfplay")
    make_run(runs_dir, "early", records=[record("session", t0 - 5, event="start"), began,
                                         record("train", t0 + 10, games=150, loss=1.0)], pid=os.getpid())
    assert manager.summary("early")["games_per_second"] == pytest.approx(5.0)
    make_run(runs_dir, "tooearly", records=[began, record("train", t0 + 4, games=150, loss=1.0)], pid=os.getpid())
    assert manager.summary("tooearly")["games_per_second"] is None
    make_run(runs_dir, "warm", records=[record("session", t0, event="start"), record("train", t0 + 30, loss=1.0)],
             pid=os.getpid())
    assert manager.summary("warm")["games_per_second"] is None
    make_run(runs_dir, "nothing")
    assert manager.summary("nothing")["games_per_second"] is None


def test_changed_lists_only_real_differences(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"model.blocks": 10, "model.channels": 128, "train.learning_rate": 0.0003,
                                      "train.amp": False, "run.hours": 8.0, "run.max_games": 5000,
                                      "run.name": "elsewhere", "run.out_dir": "C:/somewhere", "run.vram_fraction": 0.9})
    assert manager.summary("r")["changed"] == {"model.blocks": 10, "model.channels": 128, "train.learning_rate": 0.0003,
                                               "train.amp": False}
    # An int written for a float setting is the same value, not a change.
    run_dir = make_run(runs_dir, "ints")
    (run_dir / "config.toml").write_text("[train]\ncheckpoint_minutes = 10\nlearning_rate = 0.001\n[run]\nhours = 3\n",
                                         encoding="utf-8")
    assert manager.summary("ints")["changed"] == {}


def test_network_and_parameters(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"model.blocks": 4, "model.channels": 32},
             records=[record("session", 1, event="start", parameters=1234)], index={"parameters": 999})
    summary = manager.summary("r")
    assert (summary["network"], summary["parameters"]) == ("4x32", 1234)
    make_run(runs_dir, "idx", records=[record("session", 1, event="start")], index={"parameters": 999})
    assert manager.summary("idx")["parameters"] == 999
    make_run(runs_dir, "bare")
    assert manager.summary("bare")["network"] == "6x64" and manager.summary("bare")["parameters"] is None


def test_summary_of_the_genuine_run(manager, dirs, trained):
    runs_dir, _ = dirs
    copy_run(trained, runs_dir)
    summary = manager.summary(TINY_RUN)
    assert set(summary) == {"name", "status", "phase", "network", "parameters", "games", "step", "positions", "hours",
                            "games_per_second", "created", "updated", "limit", "progress", "latest_eval", "best_step",
                            "has_best", "checkpoints", "changed", "error"}
    assert summary["status"] == "finished" and summary["error"] is None
    assert summary["phase"] == "selfplay" and summary["progress"] == 1.0
    assert summary["network"] == "1x16" and summary["parameters"] > 0
    assert summary["games"] == trained.games and summary["step"] == trained.step
    assert summary["best_step"] == trained.best_step and summary["has_best"] is True
    assert summary["checkpoints"] == len(list((runs_dir / TINY_RUN / "checkpoints").glob("step_*.pt"))) >= 1
    assert summary["limit"] == {"hours": 0.0, "max_games": 24}
    assert set(summary["latest_eval"]) == {"games", "greedy-leave", "greedy"}
    assert summary["latest_eval"]["games"] == trained.games
    assert summary["games_per_second"] > 0
    assert summary["created"] <= summary["updated"]
    defaults = flatten(Config().to_dict())
    excluded = {"run.name", "run.out_dir", "run.hours", "run.max_games"}
    expected = {k: v for k, v in trained.settings.items() if k not in excluded and v != defaults[k]}
    assert summary["changed"] == expected

    detail = manager.detail(TINY_RUN)
    assert set(detail) == {"run", "config", "checkpoints", "best"}
    best_file = trained.run_dir / "checkpoints" / "best.pt" if hasattr(trained, "run_dir") else None
    assert {k: detail["best"][k] for k in ("file", "step", "model_id")} == {
        "file": "best.pt", "step": trained.best_step, "model_id": f"{TINY_RUN}/best.pt"}
    assert detail["best"]["size_bytes"] > 0 and detail["best"]["modified"] > 0
    if best_file is not None:
        assert detail["best"]["size_bytes"] == best_file.stat().st_size
    for checkpoint in detail["checkpoints"]:
        assert checkpoint["model_id"] == f"{TINY_RUN}/{checkpoint['file']}"
        assert checkpoint["file"] == f"step_{checkpoint['step']:09d}.pt" and checkpoint["size_bytes"] > 0
    assert detail["config"]["model"]["blocks"] == 1


def test_list_order(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "a_old", records=[record("session", 1000, event="stop", finished=False)])
    make_run(runs_dir, "b_new", records=[record("session", 3000, event="stop", finished=True)])
    make_run(runs_dir, "c_running", records=[record("session", 500, event="start")], pid=os.getpid())
    make_run(runs_dir, "d_queued", queued={"queue": "q", "overrides": {}, "created": time.time()})
    make_run(runs_dir, "e_stopping", records=[record("session", 100, event="start")], pid=os.getpid(), stop=True)
    (runs_dir / "not a run").mkdir()
    (runs_dir / "stray.txt").write_text("x", encoding="utf-8")
    assert [s["name"] for s in manager.list()] == ["c_running", "e_stopping", "d_queued", "b_new", "a_old"]
    assert manager.active() == ["c_running", "e_stopping"]
    assert manager.queued() == ["d_queued"]


def test_list_on_a_missing_runs_folder(tmp_path):
    assert RunManager(tmp_path / "nope", tmp_path / "app").list() == []


# ---- reading metrics.jsonl ---------------------------------------------------------------------


def test_metrics_are_read_incrementally_and_a_half_written_line_waits(manager, dirs):
    runs_dir, _ = dirs
    run_dir = make_run(runs_dir, "r", records=[record("train", i, step=i, loss=1.0) for i in range(3)])
    path = run_dir / "metrics.jsonl"
    assert [r["step"] for r in manager.records("r")] == [0, 1, 2]
    line = json.dumps(record("train", 3, step=3, loss=0.5)) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line[:17])  # the trainer is mid-write
    assert [r["step"] for r in manager.records("r")] == [0, 1, 2]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line[17:])
    assert [r["step"] for r in manager.records("r")] == [0, 1, 2, 3]
    assert [r["step"] for r in manager.records("r")] == [0, 1, 2, 3], "nothing read twice"
    records, next_index = manager.metrics("r", 2)
    assert [r["step"] for r in records] == [2, 3] and next_index == 4
    assert manager.metrics("r", 4) == ([], 4)
    assert manager.metrics("r", 99) == ([], 4)
    assert len(manager.metrics("r", -5)[0]) == 4


def test_corrupt_lines_are_skipped(manager, dirs):
    runs_dir, _ = dirs
    run_dir = make_run(runs_dir, "r")
    good = [json.dumps(record("train", i, step=i, loss=1.0)) for i in range(3)]
    (run_dir / "metrics.jsonl").write_text(
        good[0] + "\n" + '{"kind": "train", "ti' + "\n" + "\x00\x00\x00\n" + "\n" + good[1] + "\r\n" + good[2] + "\n",
        encoding="utf-8")
    assert [r["step"] for r in manager.records("r")] == [0, 1, 2]
    assert manager.summary("r")["step"] == 2


def test_a_line_that_is_json_but_not_an_object_is_skipped(manager, dirs):
    """Such a line once raised AttributeError and made every refresh append the earlier records again."""
    runs_dir, _ = dirs
    run_dir = make_run(runs_dir, "r")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(record("train", 0, step=0)) + "\nnull\n42\n[1, 2]\n\"text\"\n" + json.dumps(record("train", 1, step=1)) + "\n",
        encoding="utf-8")
    assert [r["step"] for r in manager.records("r")] == [0, 1]
    assert [r["step"] for r in manager.records("r")] == [0, 1]
    assert manager.summary("r")["step"] == 1


def test_a_file_that_shrinks_is_read_again_from_the_start(manager, dirs):
    runs_dir, _ = dirs
    run_dir = make_run(runs_dir, "r", records=[record("train", i, step=i, loss=1.0) for i in range(5)])
    assert len(manager.records("r")) == 5
    write_records(run_dir / "metrics.jsonl", [record("train", 9, step=99, loss=1.0)])
    assert [r["step"] for r in manager.records("r")] == [99]
    (run_dir / "metrics.jsonl").unlink()
    assert manager.records("r") == []


def test_log_tail(manager, dirs):
    runs_dir, _ = dirs
    lines = [f"[12:00:{i:02d}] line {i}" for i in range(300)]
    make_run(runs_dir, "r", console=lines)
    assert manager.log("r") == lines[-200:]
    assert manager.log("r", 5) == lines[-5:]
    make_run(runs_dir, "nolog")
    assert manager.log("nolog") == []


def test_log_tail_zero_or_less_is_empty(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", console=["a", "b", "c"])
    assert manager.log("r", 0) == [] and manager.log("r", -5) == []
    assert manager.log("r", 1) == ["c"] and manager.log("r", 99) == ["a", "b", "c"]


def test_log_tail_with_long_lines(manager, dirs):
    """Lines far longer than 400 bytes come back whole, as many as asked for, never a cut fragment."""
    runs_dir, _ = dirs
    lines = [f"{i:03d} " + "x" * (996 + 37 * i) for i in range(10)]
    make_run(runs_dir, "r", console=lines)
    for tail in (1, 2, 3, 9, 10, 50):
        assert manager.log("r", tail) == lines[-tail:], tail
    huge = [f"{i} " + "y" * 30_000 for i in range(5)]
    make_run(runs_dir, "huge", console=huge)
    assert manager.log("huge", 2) == huge[-2:]


def test_log_drops_a_first_line_cut_by_the_read_window(manager, dirs):
    runs_dir, _ = dirs
    lines = [f"line {i:04d} " + "z" * 50 for i in range(2000)]
    make_run(runs_dir, "r", console=lines)
    for tail in (1, 7, 200, 1999, 2000):
        assert manager.log("r", tail) == lines[-tail:], tail
    run_dir = make_run(runs_dir, "noeol")
    (run_dir / "console.log").write_text("first\nsecond\nthird, still being written", encoding="utf-8")
    assert manager.log("noeol", 2) == ["second", "third, still being written"]


# ---- starting ----------------------------------------------------------------------------------


def base_command(name, runs_dir):
    return [runs_module.python_executable(), "-u", "-m", "scrabble_ai.cli", "train", "--name", name,
            "--set", f"run.out_dir={runs_dir}"]


def test_start_one_run_with_overrides(manager, dirs, spawned, test_lexicon):
    runs_dir, _ = dirs
    overrides = {"run.hours": 8, "model.blocks": 6, "train.amp": False, "model.cross_checks": True,
                 "train.learning_rate": 0.0003, "model.norm": "group", "run.lexicon": "TEST"}
    assert manager.start([{"name": "night1", "overrides": overrides}], "parallel") == {"started": ["night1"], "queued": []}
    [(command, log_path)] = spawned.calls
    assert command == base_command("night1", runs_dir) + [
        "--set", "run.hours=8", "--set", "model.blocks=6", "--set", "train.amp=false", "--set", "model.cross_checks=true",
        "--set", "train.learning_rate=0.0003", "--set", "model.norm=group", "--set", "run.lexicon=TEST"]
    assert log_path == runs_dir / "night1" / "console.log"
    running = json.loads((runs_dir / "night1" / "running.json").read_text(encoding="utf-8"))
    assert running["pid"] == spawned.pid and running["started"] == pytest.approx(time.time(), abs=60)
    assert manager.summary("night1")["status"] == "running"


def test_start_in_parallel(manager, dirs, spawned, test_lexicon):
    runs_dir, _ = dirs
    result = manager.start([{"name": "a"}, {"name": "b", "overrides": {"model.blocks": 2}}], "parallel")
    assert result == {"started": ["a", "b"], "queued": []}
    assert [command for command, _ in spawned.calls] == [base_command("a", runs_dir),
                                                         base_command("b", runs_dir) + ["--set", "model.blocks=2"]]
    assert all((runs_dir / n / "running.json").exists() for n in "ab")


def test_start_a_queue(manager, dirs, spawned, test_lexicon):
    runs_dir, app_dir = dirs
    seen_at_spawn = {}

    def on_spawn(command, log_path):
        seen_at_spawn.update({n: (runs_dir / n / "queued.json").exists() for n in ("q1", "q2", "q3")})

    spawned.on_spawn = on_spawn
    specs = [{"name": "q1", "overrides": {"model.blocks": 2}}, {"name": "q2"}, {"name": "q3", "overrides": {"train.amp": False}}]
    assert manager.start(specs, "queue") == {"started": ["q1"], "queued": ["q2", "q3"]}
    assert seen_at_spawn == {"q1": True, "q2": True, "q3": True}, "the markers are written before the runner starts"

    [(command, log_path)] = spawned.calls
    [queue_file] = list((app_dir / "queues").glob("queue_*.json"))
    assert command == [runs_module.python_executable(), "-u", "-m", "scrabble_ai.app.queue_runner", str(queue_file)]
    assert log_path == queue_file.with_suffix(".log")
    queue = json.loads(queue_file.read_text(encoding="utf-8"))
    assert queue["runs_dir"] == str(runs_dir)
    assert [entry["name"] for entry in queue["runs"]] == ["q1", "q2", "q3"]
    assert queue["runs"][0]["command"] == base_command("q1", runs_dir) + ["--set", "model.blocks=2"]
    assert queue["runs"][2]["command"] == base_command("q3", runs_dir) + ["--set", "train.amp=false"]
    assert [entry["log"] for entry in queue["runs"]] == [str(runs_dir / n / "console.log") for n in ("q1", "q2", "q3")]
    for spec in specs:
        marker = json.loads((runs_dir / spec["name"] / "queued.json").read_text(encoding="utf-8"))
        assert marker["queue"] == queue_file.stem and marker["overrides"] == spec.get("overrides", {})
        assert marker["created"] == pytest.approx(time.time(), abs=60)
        assert not (runs_dir / spec["name"] / "running.json").exists()
        assert manager.summary(spec["name"])["status"] == "queued"
    assert manager.summary("q1")["network"] == "2x64", "a queued run's settings come from its marker"


def test_a_queue_of_one_starts_at_once_when_nothing_else_is_going(manager, dirs, spawned, test_lexicon):
    runs_dir, app_dir = dirs
    make_run(runs_dir, "old", records=[record("session", NOW, event="stop", finished=True)])
    make_run(runs_dir, "crashed", pid=DEAD_PID)
    assert manager.start([{"name": "solo"}], "queue") == {"started": ["solo"], "queued": []}
    assert spawned.calls[0][0] == base_command("solo", runs_dir)
    assert not (app_dir / "queues").exists()


@pytest.mark.parametrize("busy", ["running", "stopping", "queued"])
def test_a_queue_waits_behind_a_run_already_going(manager, dirs, spawned, test_lexicon, busy):
    runs_dir, app_dir = dirs
    if busy == "queued":
        make_run(runs_dir, "other", queued={"queue": "queue_20200101_000000", "overrides": {}, "created": time.time()})
    else:
        make_run(runs_dir, "other", pid=os.getpid(), stop=busy == "stopping")
    assert manager.start([{"name": "solo"}], "queue") == {"started": [], "queued": ["solo"]}
    assert manager.start([{"name": "a"}, {"name": "b"}], "queue") == {"started": [], "queued": ["a", "b"]}
    queue_files = sorted((app_dir / "queues").glob("queue_*.json"))
    assert len(queue_files) == 2 and len(spawned.calls) == 2
    assert all(command[-2:] == ["scrabble_ai.app.queue_runner", str(path)]
               for (command, _), path in zip(spawned.calls, queue_files))
    for name in ("solo", "a", "b"):
        assert (runs_dir / name / "queued.json").exists() and not (runs_dir / name / "running.json").exists()
        assert manager.summary(name)["status"] == "queued"
    # Parallel ignores what is going.
    assert manager.start([{"name": "now"}], "parallel") == {"started": ["now"], "queued": []}


@pytest.mark.parametrize("name", ["bad name", "bad/name", "bad\\name", "..", "../x", "-x", "_x", "", "a" * 65, "na.me",
                                  "C:x", "nämé"])
def test_start_refuses_bad_names(manager, dirs, spawned, test_lexicon, name):
    runs_dir, _ = dirs
    with pytest.raises(RunError, match="letters, digits"):
        manager.start([{"name": name}], "parallel")
    assert spawned.calls == [] and list(runs_dir.iterdir()) == []


def test_start_refuses_a_name_with_a_trailing_newline(manager, dirs, spawned, test_lexicon):
    """NAME_PATTERN once ended in `$`, which also matches before a trailing newline."""
    for name in ("night1\n", "night1\r\n", "\nnight1"):
        with pytest.raises(RunError, match="letters, digits"):
            manager.start([{"name": name}], "parallel")
        assert not manager.exists(name)
    assert spawned.calls == []


def test_start_refusals(manager, dirs, spawned, test_lexicon):
    runs_dir, _ = dirs
    make_run(runs_dir, "taken")
    cases = [
        ([{"name": "taken"}], "parallel", "already exists"),
        ([{"name": "ok", "overrides": {"model.nope": 1}}], "parallel", "model.nope"),
        ([{"name": "ok", "overrides": {"nosection.x": 1}}], "parallel", "nosection"),
        ([{"name": "ok", "overrides": {"model.blocks": "many"}}], "parallel", "model.blocks"),
        ([{"name": "ok", "overrides": {"model.channels": 12}}], "parallel", "multiple of 8"),
        ([{"name": "ok", "overrides": {"train.amp": "maybe"}}], "parallel", "train.amp"),
        ([{"name": "ok", "overrides": {"run.lexicon": "NOPE"}}], "parallel", "NOPE"),
        ([{"name": "twin"}, {"name": "twin"}], "parallel", "same name"),
        ([{"name": "twin"}, {"name": "twin"}], "queue", "same name"),
        ([{"name": "ok"}], "sideways", "mode"),
        ([], "parallel", "nothing to start"),
        # One bad run refuses the whole request: nothing is started.
        ([{"name": "first"}, {"name": "second", "overrides": {"model.nope": 1}}], "parallel", "model.nope"),
        ([{"name": "first"}, {"name": "taken"}], "queue", "already exists"),
    ]
    for specs, mode, message in cases:
        with pytest.raises(RunError, match=message):
            manager.start(specs, mode)
    assert spawned.calls == []
    assert sorted(p.name for p in runs_dir.iterdir()) == ["taken"]


def test_refusal_messages_name_the_run_when_there_are_several(manager, dirs, spawned, test_lexicon):
    with pytest.raises(RunError) as one:
        manager.start([{"name": "solo", "overrides": {"model.blocks": 0}}], "parallel")
    assert str(one.value) == "model.blocks must be at least 1"
    with pytest.raises(RunError) as several:
        manager.start([{"name": "fine"}, {"name": "broken", "overrides": {"model.blocks": 0}}], "queue")
    assert str(several.value) == "broken: model.blocks must be at least 1"
    with pytest.raises(RunError) as unknown:
        manager.start([{"name": "fine"}, {"name": "odd", "overrides": {"model.nope": 1}}], "parallel")
    assert str(unknown.value).startswith("odd: unknown setting model.nope")
    assert spawned.calls == []


def test_start_refuses_names_that_differ_only_in_case(manager, dirs, spawned, test_lexicon):
    runs_dir, _ = dirs
    with pytest.raises(RunError, match="same name"):
        manager.start([{"name": "Night"}, {"name": "night"}], "parallel")
    make_run(runs_dir, "night1")
    with pytest.raises(RunError, match="already exists"):
        manager.start([{"name": "NIGHT1"}], "parallel")
    assert spawned.calls == []


@pytest.mark.parametrize("key, value", [("run.out_dir", "C:/elsewhere"), ("run.name", "other")])
@pytest.mark.parametrize("mode", ["parallel", "queue"])
def test_start_refuses_the_settings_the_app_decides(manager, dirs, spawned, test_lexicon, key, value, mode):
    """A user's --set run.out_dir came after the app's own and won: the trainer wrote elsewhere."""
    runs_dir, _ = dirs
    with pytest.raises(RunError, match=f"{key} is set by the app"):
        manager.start([{"name": "moved", "overrides": {key: value}}], mode)
    with pytest.raises(RunError, match=f"second: {key} is set by the app"):
        manager.start([{"name": "first"}, {"name": "second", "overrides": {key: value}}], mode)
    assert spawned.calls == [] and list(runs_dir.iterdir()) == []


def test_start_refuses_overrides_that_are_not_an_object(manager, dirs, spawned, test_lexicon):
    for overrides in ([1], "model.blocks=3", 5, {1: 2}):
        with pytest.raises(RunError, match="overrides must be an object"):
            manager.start([{"name": "r", "overrides": overrides}], "parallel")
    assert spawned.calls == []


def test_a_started_run_records_its_settings_before_it_is_spawned(manager, dirs, spawned, test_lexicon):
    """Until the trainer writes config.toml, unstarted.json is the only record of what was asked for."""
    runs_dir, _ = dirs
    seen = {}
    spawned.on_spawn = lambda command, log: seen.update(
        json.loads((runs_dir / command[6] / "unstarted.json").read_text(encoding="utf-8")))
    overrides = {"model.blocks": 3, "train.amp": False}
    manager.start([{"name": "early", "overrides": overrides}], "parallel")
    assert seen == {"overrides": overrides}
    # It dies in its first seconds, before config.toml: it is failed, and resumes as itself.
    (runs_dir / "early" / "running.json").write_text(json.dumps({"pid": DEAD_PID, "started": time.time()}),
                                                    encoding="utf-8")
    assert manager.summary("early")["status"] == "failed"
    assert manager.summary("early")["network"] == "3x64"
    manager.resume("early")
    command = spawned.calls[-1][0]
    assert command == base_command("early", runs_dir) + ["--set", "model.blocks=3", "--set", "train.amp=false"]


def test_a_run_with_config_toml_gets_no_unstarted_marker(manager, dirs, spawned):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"model.blocks": 2}, latest="step_000000100.pt")
    manager.resume("r")
    assert not (runs_dir / "r" / "unstarted.json").exists()


def test_two_queues_started_in_the_same_second_get_their_own_files(manager, dirs, spawned, test_lexicon, monkeypatch):
    runs_dir, app_dir = dirs
    monkeypatch.setattr(runs_module.time, "strftime", lambda fmt, *args: "20261112_120000")
    manager.start([{"name": "a1"}, {"name": "a2"}], "queue")
    manager.start([{"name": "b1"}, {"name": "b2"}], "queue")
    manager.start([{"name": "c1"}, {"name": "c2"}], "queue")
    stems = [json.loads((runs_dir / n / "queued.json").read_text(encoding="utf-8"))["queue"] for n in ("a1", "b1", "c1")]
    assert stems == ["queue_20261112_120000", "queue_20261112_120000_002", "queue_20261112_120000_003"]
    assert stems == sorted(stems), "a later queue sorts after an earlier one (the runner waits on that order)"
    queues = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (app_dir / "queues").glob("*.json")}
    assert [e["name"] for e in queues[stems[0]]["runs"]] == ["a1", "a2"]
    assert [e["name"] for e in queues[stems[1]]["runs"]] == ["b1", "b2"]
    assert [call[1].name for call in spawned.calls] == [f"{s}.log" for s in stems]


def test_the_tenth_queue_in_a_second_still_sorts_last(manager, dirs, spawned, test_lexicon, monkeypatch):
    """Queues are ordered by comparing stems as strings: '_10' once sorted before '_2'."""
    runs_dir, _ = dirs
    monkeypatch.setattr(runs_module.time, "strftime", lambda fmt, *args: "20261112_120000")
    for i in range(10):
        manager.start([{"name": f"q{i}a"}, {"name": f"q{i}b"}], "queue")
    stems = [json.loads((runs_dir / f"q{i}a" / "queued.json").read_text(encoding="utf-8"))["queue"] for i in range(10)]
    assert stems == sorted(stems)


# ---- stopping ----------------------------------------------------------------------------------


def test_stop_writes_the_stop_file(manager, dirs, spawned):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", records=[record("session", NOW, event="start")], pid=os.getpid())
    manager.stop("r")
    assert (runs_dir / "r" / "STOP").exists()
    assert spawned.killed == []
    assert manager.summary("r")["status"] == "stopping"


def test_stop_refusals(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "idle", records=[record("session", NOW, event="stop", finished=False)])
    with pytest.raises(RunError, match="not running"):
        manager.stop("idle")
    make_run(runs_dir, "dead", pid=DEAD_PID)
    with pytest.raises(RunError, match="not running"):
        manager.stop("dead")
    with pytest.raises(RunError, match="no run named"):
        manager.stop("missing")
    assert not (runs_dir / "idle" / "STOP").exists() and not (runs_dir / "dead" / "STOP").exists()


def test_stopping_a_queued_run_takes_it_out_of_the_queue(manager, dirs, spawned):
    runs_dir, _ = dirs
    make_run(runs_dir, "q", queued={"queue": "queue_x", "overrides": {"model.blocks": 3}, "created": time.time()})
    manager.stop("q")
    assert not (runs_dir / "q" / "queued.json").exists() and not (runs_dir / "q" / "STOP").exists()
    assert json.loads((runs_dir / "q" / "unstarted.json").read_text(encoding="utf-8")) == {"overrides": {"model.blocks": 3}}
    summary = manager.summary("q")
    assert summary["status"] == "stopped" and summary["network"] == "3x64" and summary["phase"] is None
    assert spawned.killed == []
    with pytest.raises(RunError, match="not running"):
        manager.stop("q")


def test_force_stop(manager, dirs, spawned):
    runs_dir, _ = dirs
    records = [record("session", NOW - 100, event="start"),
               record("selfplay", NOW - 50, step=40, games=800, positions=9000, hours=0.02, games_per_second=5.0)]
    make_run(runs_dir, "r", records=records, pid=os.getpid(), stop=True)
    assert manager.summary("r")["status"] == "stopping"
    manager.stop("r", force=True)
    assert spawned.killed == [os.getpid()]
    assert not (runs_dir / "r" / "running.json").exists() and not (runs_dir / "r" / "STOP").exists()
    last = manager.records("r")[-1]
    assert (last["kind"], last["event"], last["finished"]) == ("session", "stop", False)
    assert (last["step"], last["games"], last["positions"], last["hours"]) == (40, 800, 9000, 0.02)
    assert last["time"] == pytest.approx(time.time(), abs=60)
    assert len(manager.records("r")) == 3
    assert manager.summary("r")["status"] == "stopped"


def test_force_stop_after_a_half_written_line(manager, dirs, spawned):
    """The stop record starts on its own line even when the killed trainer left half a line."""
    runs_dir, _ = dirs
    run_dir = make_run(runs_dir, "r", records=[record("session", NOW - 100, event="start")], pid=os.getpid())
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"kind": "train", "time": 17')
    manager.stop("r", force=True)
    assert [(r["kind"], r["event"]) for r in manager.records("r")] == [("session", "start"), ("session", "stop")]
    # And a run that never wrote a line at all.
    make_run(runs_dir, "silent", pid=os.getpid())
    manager.stop("silent", force=True)
    assert [(r["kind"], r["event"]) for r in manager.records("silent")] == [("session", "stop")]


# ---- resuming ----------------------------------------------------------------------------------


def test_resume_from_a_checkpoint(manager, dirs, spawned):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", records=[record("session", NOW, event="stop", finished=False)], settings={"model.blocks": 2},
             latest="step_000000100.pt", pid=DEAD_PID, stop=True)
    summary = manager.resume("r", hours=2.5, max_games=1000, overrides={"train.learning_rate": 0.0003, "train.amp": False})
    [(command, log_path)] = spawned.calls
    assert command == base_command("r", runs_dir) + [
        "--set", "train.learning_rate=0.0003", "--set", "train.amp=false", "--hours", "2.5",
        "--set", "run.max_games=1000", "--resume", "latest"]
    assert log_path == runs_dir / "r" / "console.log"
    assert json.loads((runs_dir / "r" / "running.json").read_text(encoding="utf-8"))["pid"] == spawned.pid
    assert not (runs_dir / "r" / "STOP").exists()
    assert summary["name"] == "r" and summary["status"] == "running"


def test_resume_without_hours_keeps_the_runs_own_time_limit(manager, dirs, spawned):
    """`--hours` is passed only when hours is given; otherwise the checkpoint's run.hours applies again."""
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"run.hours": 8.0}, latest="step_000000100.pt")
    manager.resume("r")
    command = spawned.calls[0][0]
    assert command == base_command("r", runs_dir) + ["--resume", "latest"]
    manager.stop("r", force=True)
    manager.resume("r", hours=0)
    assert spawned.calls[-1][0][-4:] == ["--hours", "0", "--resume", "latest"], "0 asks for no time limit"
    manager.stop("r", force=True)
    manager.resume("r", hours=3, max_games=0)
    assert spawned.calls[-1][0][-6:] == ["--hours", "3", "--set", "run.max_games=0", "--resume", "latest"]


def test_resume_without_a_checkpoint_restarts_from_the_saved_settings(manager, dirs, spawned):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"model.blocks": 2, "model.channels": 32, "train.amp": False}, pid=DEAD_PID)
    manager.resume("r", max_games=500, overrides={"model.blocks": 3})
    command = spawned.calls[0][0]
    sets = [command[i + 1] for i, a in enumerate(command) if a == "--set"]
    assert "--resume" not in command and "--hours" not in command
    assert "model.channels=32" in sets and "train.amp=false" in sets
    assert "model.blocks=3" in sets and "model.blocks=2" not in sets, "the request's overrides win"
    assert "run.max_games=500" in sets and "run.max_games=0" not in sets
    assert len({s.split("=")[0] for s in sets}) == len(sets), "each setting is passed once"
    assert not any(s.startswith("run.name=") for s in sets)
    assert [s for s in sets if s.startswith("run.out_dir=")] == [f"run.out_dir={runs_dir}"]


@pytest.mark.parametrize("stored_in", ["unstarted.json", "queued.json", "config.toml"])
def test_resume_without_a_checkpoint_keeps_the_runs_own_time_limit(manager, dirs, spawned, stored_in):
    runs_dir, _ = dirs
    settings = {"run.hours": 8.0, "model.blocks": 3}
    if stored_in == "config.toml":
        make_run(runs_dir, "r", settings=settings, pid=DEAD_PID)
    elif stored_in == "queued.json":
        make_run(runs_dir, "r", queued={"queue": "queue_gone", "overrides": settings, "created": NOW - 3600})
    else:
        run_dir = make_run(runs_dir, "r", pid=DEAD_PID)
        (run_dir / "unstarted.json").write_text(json.dumps({"overrides": settings}), encoding="utf-8")
    manager.resume("r")
    command = spawned.calls[0][0]
    hours = [command[i + 1] for i, a in enumerate(command) if a == "--hours"]
    hours += [command[i + 1][len("run.hours="):] for i, a in enumerate(command)
              if a == "--set" and command[i + 1].startswith("run.hours=")]
    assert hours and float(hours[-1]) == 8.0


def test_resume_overrides_are_kept_for_a_run_that_has_not_started(manager, dirs, spawned):
    runs_dir, _ = dirs
    run_dir = make_run(runs_dir, "r", pid=DEAD_PID)
    (run_dir / "unstarted.json").write_text(json.dumps({"overrides": {"model.blocks": 3}}), encoding="utf-8")
    manager.resume("r", overrides={"model.channels": 32})
    (run_dir / "running.json").write_text(json.dumps({"pid": DEAD_PID, "started": time.time()}), encoding="utf-8")
    manager.resume("r")
    assert "model.channels=32" in spawned.calls[-1][0]


def test_resume_an_interrupted_queued_run(manager, dirs, spawned):
    runs_dir, _ = dirs
    make_run(runs_dir, "q", queued={"queue": "queue_gone", "overrides": {"model.blocks": 3}, "created": NOW - 3600})
    assert manager.summary("q")["status"] == "stopped"
    manager.resume("q", hours=1)
    command = spawned.calls[0][0]
    assert command == base_command("q", runs_dir) + ["--set", "model.blocks=3", "--set", "run.hours=1"]
    assert not (runs_dir / "q" / "queued.json").exists()
    assert manager.summary("q")["status"] == "running"


def test_resume_refusals(manager, dirs, spawned):
    runs_dir, _ = dirs
    make_run(runs_dir, "running", pid=os.getpid(), latest="step_000000001.pt")
    make_run(runs_dir, "stopping", pid=os.getpid(), stop=True, latest="step_000000001.pt")
    make_run(runs_dir, "queued", queued={"queue": "q", "overrides": {}, "created": time.time()})
    for name, state in (("running", "running"), ("stopping", "stopping"), ("queued", "queued")):
        with pytest.raises(RunError, match=f"already {state}"):
            manager.resume(name)
    with pytest.raises(RunError, match="no run named"):
        manager.resume("missing")
    with pytest.raises(RunError):
        manager.resume("../runs")
    assert spawned.calls == []


@pytest.mark.parametrize("checkpoint", [True, False])
def test_resume_validates_its_input(manager, dirs, spawned, checkpoint):
    """Resume once passed overrides and hours to the command line unchecked."""
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"model.blocks": 2}, latest="step_000000100.pt" if checkpoint else None)
    bad = [
        ({"overrides": {"train.nope": 1}}, "unknown setting train.nope"),
        # With a checkpoint this one is refused as locked ("model." prefix) rather than as unknown.
        ({"overrides": {"model.nope": 1}}, "model.nope"),
        ({"overrides": {"nosection.x": 1}}, "nosection"),
        ({"overrides": {"train.learning_rate": "fast"}}, "train.learning_rate"),
        ({"overrides": {"train.min_replay": 1}}, "train.min_replay must be at least train.batch_size"),
        ({"overrides": [1]}, "overrides must be an object"),
        ({"overrides": "train.amp=false"}, "overrides must be an object"),
        ({"overrides": {"run.out_dir": "C:/elsewhere"}}, "run.out_dir is set by the app"),
        ({"overrides": {"run.name": "other"}}, "run.name is set by the app"),
        ({"hours": -1}, "hours must be a number"),
        ({"hours": "8"}, "hours must be a number"),
        ({"hours": True}, "hours must be a number"),
        ({"hours": float("nan")}, "hours must be a number"),
        ({"max_games": -1}, "max_games must be a whole number"),
        ({"max_games": 2.5}, "max_games must be a whole number"),
        ({"max_games": "100"}, "max_games must be a whole number"),
        ({"max_games": False}, "max_games must be a whole number"),
    ]
    for arguments, message in bad:
        with pytest.raises(RunError, match=message):
            manager.resume("r", **arguments)
    assert spawned.calls == [] and not (runs_dir / "r" / "running.json").exists()
    manager.resume("r", hours=0.5, max_games=10, overrides={"train.learning_rate": 0.0003})
    assert len(spawned.calls) == 1


@pytest.mark.parametrize("key, value", [("run.lexicon", "OTHER"), ("run.leaves", "zero"), ("model.blocks", 3),
                                        ("model.channels", 32), ("model.norm", "group"), ("model.cross_checks", False),
                                        ("train.value_scale", 25.0)])
def test_resume_from_a_checkpoint_refuses_locked_settings(manager, dirs, spawned, key, value):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", settings={"model.blocks": 2}, latest="step_000000100.pt")
    with pytest.raises(RunError, match=f"{key} is fixed by the run's checkpoints"):
        manager.resume("r", overrides={key: value, "train.learning_rate": 0.0003})
    assert spawned.calls == []
    # Without a checkpoint nothing is fixed yet: it starts again with the new value.
    make_run(runs_dir, "fresh", settings={"model.blocks": 2}, pid=DEAD_PID)
    manager.resume("fresh", overrides={key: value})
    assert f"{key}={str(value).lower() if isinstance(value, bool) else value}" in spawned.calls[-1][0]


def test_a_run_taken_out_of_the_queue_keeps_its_settings(manager, dirs, spawned):
    """Stopping a queued run once deleted queued.json, the only place its settings were kept."""
    runs_dir, _ = dirs
    make_run(runs_dir, "q", queued={"queue": "queue_x", "overrides": {"model.blocks": 3, "train.amp": False},
                                    "created": time.time()})
    manager.stop("q")
    manager.resume("q", hours=1)
    assert spawned.calls[0][0] == base_command("q", runs_dir) + [
        "--set", "model.blocks=3", "--set", "train.amp=false", "--set", "run.hours=1"]
    assert manager.summary("q")["status"] == "running"


# ---- deleting ----------------------------------------------------------------------------------


def test_delete(manager, dirs, tmp_path):
    runs_dir, _ = dirs
    make_run(runs_dir, "doomed", records=[record("train", 1, step=1)], latest="step_000000001.pt")
    make_run(runs_dir, "keeper", records=[record("train", 1, step=7)])
    assert len(manager.records("doomed")) == 1
    manager.delete("doomed")
    assert not (runs_dir / "doomed").exists() and (runs_dir / "keeper" / "metrics.jsonl").exists()
    # A new run with the same name starts with fresh metrics, not the cached ones.
    make_run(runs_dir, "doomed", records=[record("train", 2, step=5), record("train", 3, step=6)])
    assert [r["step"] for r in manager.records("doomed")] == [5, 6]


def test_delete_refusals(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "running", pid=os.getpid())
    make_run(runs_dir, "stopping", pid=os.getpid(), stop=True)
    for name in ("running", "stopping"):
        with pytest.raises(RunError, match="running; stop it first"):
            manager.delete(name)
        assert (runs_dir / name).is_dir()
    with pytest.raises(RunError, match="no run named"):
        manager.delete("missing")


def test_a_queued_run_can_be_deleted(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "q", queued={"queue": "q", "overrides": {}, "created": time.time()})
    manager.delete("q")
    assert not (runs_dir / "q").exists()


def test_delete_cannot_reach_outside_the_runs_folder(manager, dirs, tmp_path):
    runs_dir, _ = dirs
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep", encoding="utf-8")
    for name in ("..", "../outside", "..\\outside", str(outside), "/outside", "\\outside", "a/b", "a\\b", "C:outside",
                 "runs/../outside", ".", "", "outside\x00"):
        with pytest.raises(RunError):
            manager.delete(name)
    assert (outside / "precious.txt").read_text(encoding="utf-8") == "keep"
    assert runs_dir.is_dir()


def test_delete_does_not_follow_a_junction_out_of_the_runs_folder(manager, dirs, tmp_path):
    _winapi = pytest.importorskip("_winapi")
    if not hasattr(_winapi, "CreateJunction"):
        pytest.skip("no junctions here")
    runs_dir, _ = dirs
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep", encoding="utf-8")
    link = runs_dir / "sneaky"
    _winapi.CreateJunction(str(outside), str(link))
    try:
        with pytest.raises(RunError, match="outside the runs folder"):
            manager.delete("sneaky")
        assert (outside / "precious.txt").exists()
    finally:
        os.rmdir(link)  # removes the junction only


# ---- the queue runner --------------------------------------------------------------------------


OLDER, NEWER = "queue_20261112_120000", "queue_20261112_120000_2"


def write_queue(runs_dir, app_dir, stem, names, order, created=None, markers=None):
    """A queue file whose commands are tiny Python scripts that append their run's name to `order`."""
    queue_dir = app_dir / "queues"
    queue_dir.mkdir(exist_ok=True)
    script = "import sys; open(sys.argv[1], 'a').write(sys.argv[2] + '\\n')"
    entries = []
    for name in names:
        run_dir = runs_dir / name
        entries.append({"name": name, "command": [sys.executable, "-c", script, str(order), name],
                        "log": str(run_dir / "console.log")})
        if markers is None or name in markers:
            run_dir.mkdir(exist_ok=True)
            (run_dir / "queued.json").write_text(json.dumps(
                {"queue": stem, "overrides": {}, "created": time.time() if created is None else created}), encoding="utf-8")
    queue_file = queue_dir / f"{stem}.json"
    queue_file.write_text(json.dumps({"runs": entries, "runs_dir": str(runs_dir)}), encoding="utf-8")
    return queue_file


def started(order):
    return order.read_text(encoding="utf-8").splitlines() if order.exists() else []


class FakeAlive:
    """pid_alive for the queue runner: the given pids are alive for their first `checks` questions."""

    def __init__(self, checks: dict[int, int], on_check=None):
        self.left = dict(checks)
        self.asked: list[int] = []
        self.on_check = on_check

    def __call__(self, pid: int) -> bool:
        self.asked.append(pid)
        if self.on_check is not None:
            self.on_check(len(self.asked))
        if self.left.get(pid, 0) > 0:
            self.left[pid] -= 1
            return True
        return False


def test_queue_runner_runs_in_order_and_skips_runs_taken_out(dirs, tmp_path):
    runs_dir, app_dir = dirs
    order = tmp_path / "order.txt"
    queue_file = write_queue(runs_dir, app_dir, OLDER, ["a", "b", "c", "d"], order, markers={"a", "b", "c"})
    (runs_dir / "b" / "queued.json").unlink()  # b was taken out of the queue; d's folder was deleted

    assert queue_runner.main(str(queue_file), poll_seconds=0.01) == 0
    assert started(order) == ["a", "c"]
    runner = json.loads(queue_file.with_suffix(".pid").read_text(encoding="utf-8"))
    assert set(runner) == {"pid", "started"} and runner["pid"] == os.getpid()
    assert runner["started"] == pytest.approx(time.time(), abs=60)
    for name in ("a", "c"):
        running = json.loads((runs_dir / name / "running.json").read_text(encoding="utf-8"))
        assert isinstance(running["pid"], int) and running["pid"] > 0 and running["started"] > 0
        assert not (runs_dir / name / "queued.json").exists()
    assert not (runs_dir / "b" / "running.json").exists() and not (runs_dir / "d").exists()
    # The children ended without a trainer clearing running.json: the app calls them failed.
    manager = RunManager(runs_dir, app_dir)
    assert manager.summary("a")["status"] == "failed"
    assert manager.summary("b")["status"] == "stopped"


def test_a_queued_run_that_dies_at_once_resumes_as_itself(dirs, tmp_path, spawned):
    runs_dir, app_dir = dirs
    order = tmp_path / "order.txt"
    queue_file = write_queue(runs_dir, app_dir, OLDER, ["early"], order)
    (runs_dir / "early" / "queued.json").write_text(json.dumps(
        {"queue": OLDER, "overrides": {"model.blocks": 3}, "created": time.time()}), encoding="utf-8")
    assert queue_runner.main(str(queue_file), poll_seconds=0.01) == 0  # the "trainer" exits at once
    manager = RunManager(runs_dir, app_dir)
    assert manager.summary("early")["status"] == "failed"
    manager.resume("early")
    assert "model.blocks=3" in spawned.calls[-1][0]


def test_queue_runner_waits_for_a_run_that_is_training(dirs, tmp_path, monkeypatch, capsys):
    runs_dir, app_dir = dirs
    order = tmp_path / "order.txt"
    make_run(runs_dir, "busy", pid=4242)
    alive = FakeAlive({4242: 3})
    monkeypatch.setattr(queue_runner, "pid_alive", alive)
    monkeypatch.setattr(queue_runner, "started_after", lambda pid, moment: False)  # fake pids: no real process
    queue_file = write_queue(runs_dir, app_dir, OLDER, ["a"], order)
    assert queue_runner.main(str(queue_file), poll_seconds=0.01) == 0
    assert started(order) == ["a"]
    assert alive.asked.count(4242) == 4, "asked until the pid was gone"
    out = capsys.readouterr().out
    assert out.count("a is waiting for busy") == 1 and out.index("waiting") < out.index("starting a")


def test_queue_runner_does_not_wait_for_a_dead_process(dirs, tmp_path, capsys):
    runs_dir, app_dir = dirs
    order = tmp_path / "order.txt"
    make_run(runs_dir, "crashed", pid=DEAD_PID)
    queue_file = write_queue(runs_dir, app_dir, OLDER, ["a"], order)
    began = time.time()
    assert queue_runner.main(str(queue_file), poll_seconds=5.0) == 0
    assert started(order) == ["a"] and time.time() - began < 4.5
    assert "waiting" not in capsys.readouterr().out


def test_an_older_queue_blocks_a_newer_one(dirs, tmp_path, monkeypatch, capsys):
    runs_dir, app_dir = dirs
    order = tmp_path / "order.txt"
    write_queue(runs_dir, app_dir, OLDER, ["o1"], order, created=time.time() - 3600)
    (app_dir / "queues" / f"{OLDER}.pid").write_text(json.dumps({"pid": 4343}), encoding="utf-8")
    newer = write_queue(runs_dir, app_dir, NEWER, ["n1"], order)
    alive = FakeAlive({4343: 2})
    monkeypatch.setattr(queue_runner, "pid_alive", alive)
    monkeypatch.setattr(queue_runner, "started_after", lambda pid, moment: False)  # fake pids: no real process
    assert queue_runner.main(str(newer), poll_seconds=0.01) == 0
    assert started(order) == ["n1"] and alive.asked.count(4343) == 3
    assert "n1 is waiting for o1" in capsys.readouterr().out


def test_a_newer_queue_does_not_block_an_older_one(dirs, tmp_path, monkeypatch, capsys):
    runs_dir, app_dir = dirs
    order = tmp_path / "order.txt"
    older = write_queue(runs_dir, app_dir, OLDER, ["o1"], order, created=time.time() - 3600)
    write_queue(runs_dir, app_dir, NEWER, ["n1"], order)  # its marker is fresh too
    (app_dir / "queues" / f"{NEWER}.pid").write_text(json.dumps({"pid": 4444}), encoding="utf-8")
    alive = FakeAlive({4444: 10_000})
    monkeypatch.setattr(queue_runner, "pid_alive", alive)
    monkeypatch.setattr(queue_runner, "started_after", lambda pid, moment: False)  # fake pids: no real process
    assert queue_runner.main(str(older), poll_seconds=0.01) == 0
    assert started(order) == ["o1"] and 4444 not in alive.asked
    assert "waiting" not in capsys.readouterr().out
    assert (runs_dir / "n1" / "queued.json").exists(), "the other queue's run is left alone"


def test_a_run_taken_out_while_its_queue_waits_is_skipped(dirs, tmp_path, monkeypatch, capsys):
    runs_dir, app_dir = dirs
    order = tmp_path / "order.txt"
    make_run(runs_dir, "busy", pid=4242)
    queue_file = write_queue(runs_dir, app_dir, OLDER, ["a", "b"], order)

    def take_a_out(checks):
        if checks == 3:
            RunManager(runs_dir, app_dir).stop("a")  # the app's way of taking it out of the queue

    alive = FakeAlive({4242: 5}, on_check=take_a_out)
    monkeypatch.setattr(queue_runner, "pid_alive", alive)
    monkeypatch.setattr(queue_runner, "started_after", lambda pid, moment: False)  # fake pids: no real process
    assert queue_runner.main(str(queue_file), poll_seconds=0.01) == 0
    assert started(order) == ["b"]
    assert not (runs_dir / "a" / "running.json").exists()
    assert "skipping a: no longer queued" in capsys.readouterr().out


def test_waiting_for(dirs, monkeypatch):
    runs_dir, app_dir = dirs
    queues = app_dir / "queues"
    queues.mkdir()
    now = time.time()
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) is None
    assert queue_runner.waiting_for(runs_dir / "missing", queues, NEWER) is None
    make_run(runs_dir, "dead", pid=DEAD_PID)
    make_run(runs_dir, "done", records=[record("session", now, event="stop", finished=True)])
    make_run(runs_dir, "mine", queued={"queue": NEWER, "overrides": {}, "created": now})
    make_run(runs_dir, "later", queued={"queue": "queue_20261112_120001", "overrides": {}, "created": now})
    make_run(runs_dir, "orphan", queued={"queue": "queue_20200101_000000", "overrides": {}, "created": now - 60})
    (runs_dir / "junk").mkdir()
    (runs_dir / "junk" / "running.json").write_text("[1, 2]", encoding="utf-8")
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) is None, \
        "dead pids, this queue's own runs, newer queues and an older queue whose runner died are no reason to wait"
    make_run(runs_dir, "fresh", queued={"queue": "queue_20200101_000001", "overrides": {}, "created": now - 5})
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) == "fresh", "an older queue whose runner may be starting"
    (runs_dir / "fresh" / "queued.json").unlink()
    (queues / "queue_20200101_000000.pid").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) == "orphan", "an older queue whose runner is alive"
    (runs_dir / "orphan" / "queued.json").unlink()
    make_run(runs_dir, "training", pid=os.getpid())
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) == "training"


# ---- process identity: a pid Windows has handed to a later process -----------------------------


def born() -> float:
    created = procs.process_start_time(os.getpid())
    assert created is not None
    return created


def test_process_start_time():
    assert time.time() - 30 * 24 * 3600 < born() < time.time()
    assert procs.process_start_time(DEAD_PID) is None
    assert procs.process_start_time(0) is None and procs.process_start_time(-4) is None


def test_started_after():
    me = os.getpid()
    assert procs.started_after(me, born() - 1000), "the marker is older than the process: not the process it named"
    assert not procs.started_after(me, born() + 1), "written after the process started"
    assert not procs.started_after(me, born() - procs.MARKER_SLACK_SECONDS + 1), "within the slack"
    assert not procs.started_after(DEAD_PID, 1.0), "no start time to compare: assume it is the process"
    assert not procs.started_after(me, 0)


def test_marker_helpers(tmp_path):
    assert procs.marker_pid({"pid": 12}) == 12 and procs.marker_pid({"pid": "12"}) == 12
    for marker in (None, {}, {"pid": None}, {"pid": "abc"}, {"pid": [1]}):
        assert procs.marker_pid(marker) == 0, marker
    path = tmp_path / "marker.json"
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (1_000_000, 1_000_000))
    assert procs.marker_time({"started": 5.5}, path) == 5.5
    assert procs.marker_time({}, path) == 1_000_000 == procs.marker_time({"started": "x"}, path)
    assert procs.marker_time({}, tmp_path / "missing.json") == 0.0


def write_marker(path, pid, started=None, file_time=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid} if started is None else {"pid": pid, "started": started}), encoding="utf-8")
    if file_time is not None:
        os.utime(path, (file_time, file_time))


def test_a_running_marker_older_than_its_process_is_not_that_process(manager, dirs, spawned):
    runs_dir, _ = dirs
    start = record("session", NOW - 100, event="start")
    for name, started, file_time, expected in (
            ("reused", born() - 1000, None, "failed"),
            ("reused_file", None, born() - 1000, "failed"),
            ("genuine", time.time(), None, "running"),
            ("genuine_file", None, None, "running"),
            ("garbled", None, None, "failed")):
        run_dir = make_run(runs_dir, name, records=[start], console=["the last line"])
        write_marker(run_dir / "running.json", "abc" if name == "garbled" else os.getpid(), started, file_time)
        state, error = manager._status(run_dir, manager.records(name))
        assert state == expected, name
        assert (error == "the last line") == (expected == "failed"), name
    for name in ("reused", "reused_file"):
        with pytest.raises(RunError, match="not running"):
            manager.stop(name, force=True)
        manager.delete(name)  # not running, so it can go
    assert spawned.killed == [], "a process that merely inherited the number is never killed"
    assert sorted(manager.active()) == ["genuine", "genuine_file"]


def test_a_queue_pid_older_than_its_runner_is_not_that_runner(manager, dirs):
    runs_dir, app_dir = dirs
    queues = app_dir / "queues"
    for name, started, file_time, expected in (("reused", born() - 1000, None, "stopped"),
                                               ("reused_file", None, born() - 1000, "stopped"),
                                               ("genuine", time.time(), None, "queued")):
        make_run(runs_dir, name, queued={"queue": f"queue_{name}", "overrides": {}, "created": NOW - 3600})
        write_marker(queues / f"queue_{name}.pid", os.getpid(), started, file_time)
        assert manager.summary(name)["status"] == expected, name


def test_the_queue_runner_does_not_wait_for_a_reused_pid(dirs):
    runs_dir, app_dir = dirs
    queues = app_dir / "queues"
    queues.mkdir()
    write_marker(runs_dir / "old" / "running.json", os.getpid(), born() - 1000)
    make_run(runs_dir, "older", queued={"queue": "queue_20200101_000000", "overrides": {}, "created": NOW - 3600})
    write_marker(queues / "queue_20200101_000000.pid", os.getpid(), born() - 1000)
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) is None
    write_marker(queues / "queue_20200101_000000.pid", os.getpid(), time.time())
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) == "older"
    write_marker(runs_dir / "old" / "running.json", os.getpid(), time.time())
    assert queue_runner.waiting_for(runs_dir, queues, NEWER) == "old"


class FakeKernel:
    """Stands in for kernel32 in kill_process: records what it is asked and opens nothing."""

    def __init__(self):
        self.opened: list[tuple] = []

    def OpenProcess(self, access, inherit, pid):  # noqa: N802
        self.opened.append((access, inherit, pid))
        return 0  # no handle: nothing is terminated

    def TerminateProcess(self, handle, code):  # noqa: N802
        raise AssertionError("never reached: OpenProcess gave no handle")

    def CloseHandle(self, handle):  # noqa: N802
        pass


@pytest.fixture
def fake_kill(monkeypatch):
    from types import SimpleNamespace

    calls, kernel = [], FakeKernel()
    outcome = {"returncode": 0, "raise": None}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if outcome["raise"] is not None:
            raise outcome["raise"]
        return SimpleNamespace(returncode=outcome["returncode"], stdout=b"", stderr=b"")

    monkeypatch.setattr(procs.subprocess, "run", run)
    monkeypatch.setattr(procs.ctypes, "windll", SimpleNamespace(kernel32=kernel))
    return SimpleNamespace(calls=calls, kernel=kernel, outcome=outcome)


@pytest.mark.skipif(os.name != "nt", reason="taskkill is Windows only")
def test_kill_process_ends_the_whole_tree(fake_kill):
    procs.kill_process(1234)
    [(command, kwargs)] = fake_kill.calls
    assert command == ["taskkill", "/PID", "1234", "/T", "/F"]
    assert kwargs.get("timeout") and kwargs.get("capture_output")
    assert fake_kill.kernel.opened == [], "taskkill did it: no fallback"


@pytest.mark.skipif(os.name != "nt", reason="taskkill is Windows only")
@pytest.mark.parametrize("failure", ["exit code", "no taskkill", "timeout"])
def test_kill_process_falls_back_to_terminate_process(fake_kill, failure):
    import subprocess

    if failure == "exit code":
        fake_kill.outcome["returncode"] = 128
    else:
        fake_kill.outcome["raise"] = (FileNotFoundError("taskkill") if failure == "no taskkill"
                                      else subprocess.TimeoutExpired("taskkill", 15))
    procs.kill_process(1234)
    assert len(fake_kill.calls) == 1
    assert fake_kill.kernel.opened == [(0x0001, False, 1234)], "PROCESS_TERMINATE on that pid"


def test_kill_process_ignores_pids_that_cannot_be_a_process(fake_kill):
    for pid in (0, -1):
        procs.kill_process(pid)
    assert fake_kill.calls == [] and fake_kill.kernel.opened == []


def test_phase_of_a_run_stopped_during_its_warm_start(manager, dirs):
    runs_dir, _ = dirs
    make_run(runs_dir, "r", records=[record("session", NOW - 100, event="start"),
                                     record("session", NOW - 50, event="stop", reason="STOP file found", finished=False)])
    summary = manager.summary("r")
    assert (summary["status"], summary["phase"]) == ("stopped", "warmstart")
