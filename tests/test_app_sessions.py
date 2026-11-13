"""`scrabble_ai.app.sessions`: games with a board on screen, played through `ai_move`, `human_move`
and `hint` on the test word list, with bots and with the tiny trained network on the CPU."""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import pytest

from scrabble_ai import _core
from scrabble_ai import engine as engine_module
from scrabble_ai.app import sessions as sessions_module
from scrabble_ai.app.matches import play_match_detailed
from scrabble_ai.app.players import NotFound, PlayerError, Players
from scrabble_ai.app.sessions import GameSession, Sessions

pytest.register_assert_rewrite("app_fixtures")
from app_fixtures import TINY_RUN, copy_run  # noqa: E402

pytest_plugins = ["app_fixtures"]

BEST = f"model:{TINY_RUN}/best.pt"
STATE_KEYS = {"id", "seed", "players", "advisor", "board", "last_placed", "racks", "scores", "to_move", "turn", "bag",
              "scoreless_turns", "can_exchange", "unseen", "unseen_by_seat", "over", "result", "history"}
HISTORY_KEYS = {"turn", "player", "kind", "text", "word", "position", "score", "total", "tiles", "rack", "leave", "bingo",
                "exchanged", "analysed"}
CANDIDATE_KEYS = {"text", "kind", "word", "position", "score", "tiles", "leave", "exchanged", "leave_value", "value",
                  "total", "win", "chosen"}
BOTS = ("bot:greedy-leave", "bot:greedy", "bot:random")


@pytest.fixture
def sessions(tmp_path, trained, cpu_players, fast_scorer):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    copy_run(trained, runs_dir)
    return Sessions(cpu_players(runs_dir))


def play_out(session: GameSession, top: int = 12, check=None) -> list[dict]:
    analyses = []
    for _ in range(400):
        state = session.state()
        if check is not None:
            check(state)
        if state["over"]:
            return analyses
        analyses.append(session.ai_move(top))
    raise AssertionError("the game did not end")


def tile_letter(letter: str, blank: bool) -> str:
    return letter.lower() if blank else letter.upper()


def rack_points(rack: str) -> int:
    return sum(0 if t == "?" else _core.TILE_VALUES[ord(t) - ord("A")] for t in rack)


def check_unseen(state: dict) -> None:
    """`unseen_by_seat[s]` is the bag plus the other seat's rack; `unseen` is the side to move's."""
    racks, by_seat = state["racks"], state["unseen_by_seat"]
    assert len(by_seat) == 2 and state["unseen"] == by_seat[state["to_move"]]
    on_board = sum(square != "." for row in state["board"] for square in row)
    assert on_board + len(racks[0]) + len(racks[1]) + state["bag"] == 100
    bags = []
    for seat in (0, 1):
        unseen, other = Counter(by_seat[seat]), Counter(racks[1 - seat])
        assert all(count > 0 for count in unseen.values())
        assert all(unseen[tile] >= count for tile, count in other.items()), "the other rack is unseen"
        bag = unseen - other
        assert sum(bag.values()) == state["bag"] and sum(unseen.values()) == state["bag"] + len(racks[1 - seat])
        bags.append(bag)
    assert bags[0] == bags[1], "both seats see the same bag"


def check_finished_game(state: dict) -> None:
    """Everything a finished game's state must agree with."""
    assert set(state) == STATE_KEYS
    history, result = state["history"], state["result"]
    assert state["over"] and result is not None
    assert [h["turn"] for h in history] == list(range(1, len(history) + 1)) and state["turn"] == len(history)
    assert [h["player"] for h in history] == [i % 2 for i in range(len(history))], "players alternate, seat 0 first"

    running = [0, 0]
    for i, move in enumerate(history):
        assert set(move) == HISTORY_KEYS
        running[move["player"]] += move["score"]
        if i < len(history) - 1:
            assert move["total"] == running[move["player"]], f"turn {move['turn']}"
        assert move["kind"] in ("play", "exchange", "pass")
        rack, leave = Counter(move["rack"]), Counter(move["leave"])
        if move["kind"] == "play":
            used = Counter("?" if blank else letter for _, _, letter, blank in move["tiles"])
            assert rack == leave + used, f"turn {move['turn']}: rack = leave + tiles played"
            assert move["word"] and move["position"] and move["text"].startswith(move["position"] + " ")
            assert move["bingo"] == (len(move["tiles"]) == 7) and move["exchanged"] == ""
        elif move["kind"] == "exchange":
            # The history does not say which tiles went back, only how many; `exchanged` has them.
            assert move["tiles"] == [] and move["score"] == 0 and move["text"] == f"exchange {len(move['exchanged'])}"
            assert rack == leave + Counter(move["exchanged"])
        else:
            assert move["tiles"] == [] and move["score"] == 0 and move["text"] == "pass" and rack == leave

    adjustment = result["adjustment"]
    assert result["scores"] == state["scores"] == [running[0] + adjustment[0], running[1] + adjustment[1]]
    last = history[-1]
    assert last["total"] == running[last["player"]] + adjustment[last["player"]]
    expected_winner = 0 if state["scores"][0] > state["scores"][1] else 1 if state["scores"][1] > state["scores"][0] else None
    assert result["winner"] == expected_winner
    assert result["ended_by"] in ("out", "scoreless")
    if result["ended_by"] == "out":
        assert last["kind"] == "play" and last["leave"] == "" and state["bag"] == 0
        assert state["racks"][last["player"]] == ""
        assert adjustment[last["player"]] == 2 * rack_points(state["racks"][1 - last["player"]])
        assert adjustment[1 - last["player"]] == 0
    else:
        assert all(h["kind"] != "play" for h in history[-6:]) and all(a <= 0 for a in adjustment)
    assert state["last_placed"] == [[r, c] for r, c, _, _ in last["tiles"]]

    # The board holds exactly the tiles the history placed.
    placed = {}
    for move in history:
        for r, c, letter, blank in move["tiles"]:
            assert (r, c) not in placed, "a square filled twice"
            placed[(r, c)] = tile_letter(letter, blank)
    board = state["board"]
    assert len(board) == 15 and all(len(row) == 15 for row in board)
    on_board = {(r, c): board[r][c] for r in range(15) for c in range(15) if board[r][c] != "."}
    assert on_board == placed
    check_unseen(state)


def check_candidate(c: dict) -> None:
    assert set(c) == CANDIDATE_KEYS
    assert c["total"] == pytest.approx(c["score"] + c["value"], abs=0.011)
    if c["kind"] == "exchange":
        assert c["exchanged"] and c["text"] == "exchange " + c["exchanged"] and c["tiles"] == [] and c["score"] == 0
    else:
        assert c["exchanged"] == ""
        if c["kind"] == "pass":
            assert c["text"] == "pass" and c["score"] == 0


# ---- bots --------------------------------------------------------------------------------------


def test_a_bot_game_played_to_the_end(sessions):
    session = sessions.create(["bot:greedy-leave", "bot:greedy"], seed=7)
    state = session.state()
    assert state["seed"] == 7 and state["turn"] == 0 and state["history"] == [] and state["result"] is None
    assert state["players"] == [{"spec": "bot:greedy-leave", "name": "Greedy + leave"}, {"spec": "bot:greedy", "name": "Greedy"}]
    assert state["advisor"] == "bot:greedy-leave"
    assert [len(r) for r in state["racks"]] == [7, 7] and state["bag"] == 86 and state["last_placed"] == []
    analyses = play_out(session, check=check_unseen)
    final = session.state()
    check_finished_game(final)
    assert all(h["analysed"] for h in final["history"])
    for i, response in enumerate(analyses):
        analysis = response["analysis"]
        assert set(analysis) == {"player", "turn", "considered", "candidates"}
        assert analysis["player"] == ("Greedy + leave" if i % 2 == 0 else "Greedy") and analysis["turn"] == i + 1
        assert set(response["state"]) == STATE_KEYS and response["state"]["turn"] == i + 1
        candidates = analysis["candidates"]
        for candidate in candidates:
            check_candidate(candidate)
        assert sum(c["chosen"] for c in candidates) == 1
        assert analysis["considered"] >= len(candidates) - 1
        chosen = next(c for c in candidates if c["chosen"])
        played = final["history"][i]
        assert (chosen["kind"], chosen["score"]) == (played["kind"], played["score"])
        if played["kind"] == "play":
            assert chosen["text"] == played["text"]
        assert session.analysis(i + 1) == analysis
    with pytest.raises(PlayerError, match="over"):
        session.ai_move()
    with pytest.raises(PlayerError, match="over"):
        session.hint()


def test_the_same_seed_replays_identically(sessions):
    for specs in (["bot:greedy-leave", "bot:greedy"], ["bot:random", "bot:random"]):
        first, second = sessions.create(specs, seed=4242), sessions.create(specs, seed=4242)
        play_out(first)
        play_out(second)
        a, b = first.state(), second.state()
        assert {k: v for k, v in a.items() if k != "id"} == {k: v for k, v in b.items() if k != "id"}
        check_finished_game(a)


def test_each_bot_follows_its_rule(sessions):
    session = sessions.create(["bot:greedy-leave", "bot:greedy"], seed=99)
    for response in play_out(session, top=4096):
        candidates = response["analysis"]["candidates"]
        chosen = next(c for c in candidates if c["chosen"])
        if response["analysis"]["player"] == "Greedy + leave":
            assert candidates[0]["chosen"], "greedy-leave plays the first listed candidate"
            assert all(c["win"] is None for c in candidates)
            assert chosen["total"] == max(c["total"] for c in candidates)
        else:
            plays = [c for c in candidates if c["kind"] == "play"]
            # Greedy looks at the score alone: value is 0 except the going-out bonus.
            assert all(c["win"] is None and (c["value"] == 0 or (c["leave"] == "" and c["value"] > 0)) for c in candidates)
            if plays:
                assert chosen["kind"] == "play" and chosen["score"] == max(c["score"] for c in plays)


def test_random_varies_but_is_fixed_by_the_seed(sessions):
    session = sessions.create(["bot:random", "bot:random"], seed=5)
    analyses = play_out(session, top=4096)
    not_top = 0
    for response in analyses:
        plays = [c for c in response["analysis"]["candidates"] if c["kind"] == "play"]
        chosen = next(c for c in response["analysis"]["candidates"] if c["chosen"])
        if plays:
            assert chosen["kind"] == "play", "random places a tile whenever it can"
            not_top += chosen["score"] < max(c["score"] for c in plays)
    assert not_top >= 3
    other = sessions.create(["bot:random", "bot:random"], seed=6)
    play_out(other)
    assert other.state()["history"] != session.state()["history"]


def stuck_first_rack_seed(engine, leaves) -> int:
    """A seed whose first rack has no placement on the empty board (the test word list is small)."""
    for seed in range(5000):
        descriptions = _core.Game(engine, leaves, seed).candidates(4096, 127)[0]
        if not any(d["kind"] == "play" for d in descriptions):
            return seed
    pytest.skip("no seed with a stuck first rack found")


@pytest.mark.parametrize("bot", ["bot:greedy", "bot:random"])
def test_with_no_placement_a_simple_bot_exchanges_its_whole_rack(sessions, trained, bot):
    seed = stuck_first_rack_seed(trained.loaded.engine, trained.loaded.leaves)
    session = sessions.create([bot, "human"], seed=seed)
    rack = session.state()["racks"][0]
    session.ai_move()
    move = session.state()["history"][0]
    assert move["kind"] == "exchange" and sorted(move["exchanged"]) == sorted(rack) and move["leave"] == ""
    assert move["text"] == "exchange 7"
    assert session.state()["scoreless_turns"] == 1


@pytest.mark.parametrize("bot", ["bot:greedy", "bot:random", "bot:greedy-leave"])
def test_with_no_placement_and_no_exchange_a_bot_passes(sessions, bot):
    """Found by search: a late position (fewer than 7 in the bag) where the side to move cannot place."""
    found = None
    for seed in range(200):
        session = sessions.create(["bot:greedy-leave", "bot:greedy-leave"], seed=seed)
        for turn in range(200):
            state = session.state()
            if state["over"]:
                break
            if not state["can_exchange"]:
                if not any(d["kind"] == "play" for d in session.game.candidates(4096, 127)[0]):
                    found = (seed, turn)
                    break
            session.ai_move()
        if found:
            break
    if found is None:
        pytest.skip("no stuck late position found")
    seed, turn = found
    session = sessions.create(["bot:greedy-leave", "bot:greedy-leave"], seed=seed)
    for _ in range(turn):
        session.ai_move()
    to_move = session.state()["to_move"]
    session.specs[to_move] = bot  # the same position, now with this bot to move
    session.ai_move()
    assert session.state()["history"][-1]["kind"] == "pass"


def bot_sessions(trained, tmp_path, leaves) -> Sessions:
    loaded = dataclasses.replace(trained.loaded, leaves=leaves, leaves_hash=engine_module.leaves_hash(leaves))
    return Sessions(Players(loaded, tmp_path))


def seat_stats(history: list[dict], seat: int) -> tuple:
    mine = [h for h in history if h["player"] == seat]
    plays = [h for h in mine if h["kind"] == "play"]
    return (sum(h["bingo"] for h in mine), sum(h["kind"] == "exchange" for h in mine),
            sum(h["kind"] == "pass" for h in mine), max((h["score"] for h in plays), default=0))


@pytest.mark.parametrize("a, b", [("bot:greedy-leave", "bot:greedy"), ("bot:greedy", "bot:random"),
                                  ("bot:random", "bot:greedy-leave"), ("bot:random", "bot:random")])
@pytest.mark.parametrize("which_leaves", ["fitted", "zero"])
def test_a_match_game_replays_in_the_viewer(trained, tmp_path, a, b, which_leaves):
    """APP_API.md: any match game replays in the viewer from its seed and seating. (Greedy once broke
    score ties by leave value and random drew from NumPy's generator, so neither replayed.)"""
    leaves = trained.loaded.leaves if which_leaves == "fitted" else _core.LeaveModel.zero()
    sessions = bot_sessions(trained, tmp_path, leaves)
    games = play_match_detailed(trained.loaded.engine, leaves, {}, a, b, np.array([31, 32, 33], np.uint64), None)
    for i, seed in enumerate(games["seed"]):
        a_first = games["a_first"][i]
        session = sessions.create([a, b] if a_first else [b, a], seed=seed)
        play_out(session)
        state = session.state()
        seat_a, seat_b = (0, 1) if a_first else (1, 0)
        assert (state["scores"][seat_a], state["scores"][seat_b]) == (games["score_a"][i], games["score_b"][i]), i
        assert state["turn"] == games["turns"][i] and state["result"]["ended_by"] == games["ended_by"][i]
        for seat, side in ((seat_a, "a"), (seat_b, "b")):
            assert seat_stats(state["history"], seat) == (
                games[f"bingos_{side}"][i], games[f"exchanges_{side}"][i], games[f"passes_{side}"][i],
                games[f"best_play_{side}"][i]), (i, side)


def test_a_match_game_with_a_model_replays_in_the_viewer(sessions, trained):
    import torch

    players = sessions.players
    games = play_match_detailed(trained.loaded.engine, trained.loaded.leaves, {BEST: players.scorer(BEST)}, BEST,
                                "bot:random", np.array([41, 42], np.uint64), torch.device("cpu"))
    for i, seed in enumerate(games["seed"]):
        specs = [BEST, "bot:random"] if games["a_first"][i] else ["bot:random", BEST]
        session = sessions.create(specs, seed=seed)
        play_out(session)
        scores = session.state()["scores"]
        assert (scores if games["a_first"][i] else scores[::-1]) == [games["score_a"][i], games["score_b"][i]], i


def test_bot_choice_adds_a_move_that_was_not_offered(trained):
    engine, leaves = trained.loaded.engine, trained.loaded.leaves
    for seed in range(100):
        game = _core.Game(engine, leaves, seed)
        offered = game.candidates(2, 0)[0]
        index, extra = game.bot_choice("random")
        if extra is not None:
            break
        assert 0 <= index < len(offered)
    else:
        pytest.fail("random always picked one of the two offered moves")
    assert index == len(offered) == 2
    assert {"kind", "text", "word", "position", "score", "tiles", "leave", "leave_value", "exchanged", "bonus"} <= set(extra)
    assert extra["text"] not in {d["text"] for d in offered}
    move = game.play_candidate(index)
    assert (move["text"], move["score"]) == (extra["text"], extra["score"] - extra["bonus"])
    with pytest.raises(ValueError):
        game.bot_choice("nope")


def test_a_bot_move_outside_the_listed_candidates_is_shown_and_replays(trained, tmp_path, monkeypatch):
    """With only two candidates listed, random's move is usually not among them: it is added."""
    sessions = bot_sessions(trained, tmp_path, trained.loaded.leaves)
    reference = sessions.create(["bot:random", "bot:greedy"], seed=77)
    play_out(reference)
    monkeypatch.setattr(sessions_module, "BOT_CANDIDATES", 2)
    monkeypatch.setattr(sessions_module, "BOT_EXCHANGES", 0)
    session = sessions.create(["bot:random", "bot:greedy"], seed=77)
    added = 0
    for response in play_out(session, top=1):
        analysis = response["analysis"]
        chosen = [c for c in analysis["candidates"] if c["chosen"]]
        assert len(chosen) == 1 and analysis["considered"] in (1, 2, 3)
        added += analysis["considered"] == 3
    assert added >= 3
    assert session.state()["history"] == reference.state()["history"], "what is listed does not change what is played"


def test_hints_do_not_change_what_a_random_bot_plays(sessions):
    """hint() with a bot:random advisor once drew from the random player's stream."""
    plain = sessions.create(["bot:random", "bot:random"], seed=15, advisor="bot:random")
    hinted = sessions.create(["bot:random", "bot:random"], seed=15, advisor="bot:random")
    for _ in range(10):
        plain.ai_move()
        before = hinted.state()
        hint = hinted.hint()
        assert hinted.state() == before and not any(c["chosen"] for c in hint["candidates"])
        hinted.ai_move()
    assert hinted.state()["history"] == plain.state()["history"]
    play_out(plain)
    play_out(hinted)
    assert hinted.state()["scores"] == plain.state()["scores"]


# ---- a person playing --------------------------------------------------------------------------


def as_body(candidate: dict) -> dict:
    return {"tiles": [{"row": r, "col": c, "letter": letter, "blank": blank} for r, c, letter, blank in candidate["tiles"]]}


def test_a_human_move_from_the_hint_is_accepted(sessions):
    session = sessions.create(["human", "bot:greedy-leave"], seed=11)
    hint = session.hint(top=50)
    assert hint["advisor"] == "Greedy + leave"
    play = next(c for c in hint["candidates"] if c["kind"] == "play")
    state = session.human_move(as_body(play))
    move = state["history"][0]
    assert (move["kind"], move["word"], move["position"], move["score"]) == ("play", play["word"], play["position"], play["score"])
    assert move["text"] == play["text"] and move["analysed"] is False
    assert state["scores"][0] == play["score"] and state["to_move"] == 1 and state["turn"] == 1
    assert state["last_placed"] == [[r, c] for r, c, _, _ in play["tiles"]]
    check_unseen(state)
    with pytest.raises(PlayerError, match="not a person's turn"):
        session.human_move({"pass": True})
    session.ai_move()
    assert session.state()["to_move"] == 0
    with pytest.raises(PlayerError, match="a person's turn"):
        session.ai_move()


def two_letters_that_are_not_a_word(rack: str) -> str | None:
    from conftest import WORDS

    for x in rack:
        for y in rack:
            if x != "?" and y != "?" and x + y not in WORDS and Counter(x + y) <= Counter(rack):
                return x + y
    return None


def test_illegal_moves_are_refused_and_change_nothing(sessions):
    session = sessions.create(["human", "bot:greedy-leave"], seed=12)
    before = session.state()
    rack = before["racks"][0]
    absent = next(letter for letter in "QZXJKVWYBCDFGHLMNPRSTAEIOU" if letter not in rack)
    pair = two_letters_that_are_not_a_word(rack)
    attempts = [
        ({"tiles": [{"row": 7, "col": 7, "letter": absent}, {"row": 7, "col": 8, "letter": absent}]}, ValueError),
        ({"tiles": [{"row": 15, "col": 7, "letter": rack[0]}]}, PlayerError),
        ({"tiles": [{"row": -1, "col": 7, "letter": rack[0]}]}, PlayerError),
        ({"tiles": [{"row": 7, "col": 7, "letter": "1"}]}, ValueError),
        ({"tiles": [{"row": 0, "col": 0, "letter": rack[0]}, {"row": 0, "col": 1, "letter": rack[1]}]}, ValueError),
        ({"tiles": []}, ValueError),
        ({"exchange": absent}, ValueError),
        ({}, PlayerError),
        # Malformed bodies: once a KeyError (a 404) or a TypeError (a 500).
        ({"tiles": [{"col": 7, "letter": "A"}]}, PlayerError),
        ({"tiles": [{"row": 7, "col": 7}]}, PlayerError),
        ({"tiles": [5]}, PlayerError),
        ({"tiles": "AB"}, PlayerError),
        ({"tiles": [{"row": "7", "col": 7, "letter": "A"}]}, PlayerError),
        ({"tiles": [{"row": True, "col": 7, "letter": "A"}]}, PlayerError),
        ({"tiles": [{"row": 7.0, "col": 7, "letter": "A"}]}, PlayerError),
        ({"exchange": 5}, PlayerError),
        ({"exchange": ["A"]}, PlayerError),
    ]
    if pair:
        attempts.append(({"tiles": [{"row": 7, "col": 7, "letter": pair[0]}, {"row": 7, "col": 8, "letter": pair[1]}]},
                         ValueError))
    for body, error in attempts:
        with pytest.raises(error) as raised:
            session.human_move(body)
        assert str(raised.value), body
        assert session.state() == before, body
    if pair:
        with pytest.raises(ValueError, match="word"):
            session.human_move({"tiles": [{"row": 7, "col": 7, "letter": pair[0]}, {"row": 7, "col": 8, "letter": pair[1]}]})


def test_exchanges_and_passes_and_six_scoreless_turns(sessions):
    session = sessions.create(["human", "human"], seed=13)
    rack = session.state()["racks"][0]
    state = session.human_move({"exchange": rack[:3].lower()})  # letters in either case
    move = state["history"][0]
    assert (move["kind"], move["text"], move["tiles"], move["score"]) == ("exchange", "exchange 3", [], 0)
    assert sorted(move["exchanged"]) == sorted(rack[:3]) and state["scoreless_turns"] == 1
    assert len(state["racks"][0]) == 7 and state["bag"] == 86 and state["to_move"] == 1 and state["last_placed"] == []
    check_unseen(state)
    state = session.human_move({"pass": True})
    assert state["history"][-1]["kind"] == "pass" and state["scoreless_turns"] == 2
    for _ in range(3):
        state = session.human_move({"pass": True})
    assert not state["over"] and state["scoreless_turns"] == 5
    racks = state["racks"]
    state = session.human_move({"exchange": racks[1][:1]})
    assert state["over"] and state["result"]["ended_by"] == "scoreless"
    assert state["result"]["adjustment"] == [-rack_points(state["racks"][0]), -rack_points(state["racks"][1])]
    check_finished_game(state)
    for body in ({"pass": True}, {"exchange": "A"}):
        with pytest.raises(PlayerError, match="over"):
            session.human_move(body)


def test_exchange_candidates_name_their_letters_but_the_history_only_counts_them(sessions):
    session = sessions.create(["human", "bot:greedy"], seed=18, advisor="bot:greedy")
    candidates = session.hint(top=4096)["candidates"]
    for candidate in candidates:
        check_candidate(candidate)
    exchanges = [c for c in candidates if c["kind"] == "exchange"]
    assert exchanges, "a full bag allows exchanges"
    three = next(c for c in exchanges if len(c["exchanged"]) == 3)
    assert three["text"] == f"exchange {three['exchanged']}"
    state = session.human_move({"exchange": three["exchanged"]})
    move = state["history"][0]
    assert move["text"] == "exchange 3" and sorted(move["exchanged"]) == sorted(three["exchanged"])
    response = session.ai_move(top=4096)
    shown = [c for c in response["analysis"]["candidates"] if c["kind"] == "exchange"]
    assert shown and all(c["text"] == "exchange " + c["exchanged"] for c in shown)


def test_hint_marks_nothing_chosen_and_changes_nothing(sessions):
    session = sessions.create(["human", "bot:greedy"], seed=14)
    before = session.state()
    hint = session.hint(top=5)
    assert len(hint["candidates"]) == 5
    for candidate in hint["candidates"]:
        check_candidate(candidate)
        assert candidate["chosen"] is False
    totals = [c["total"] for c in hint["candidates"]]
    assert totals == sorted(totals, reverse=True)
    assert session.state() == before
    assert session.hint(top=5) == hint, "asking again gives the same answer"


def going_out_position(sessions):
    """A session whose side to move can go out (bag empty, a play using the whole rack), found by search."""
    for seed in range(300):
        session = sessions.create(["bot:greedy-leave", "bot:greedy-leave"], seed=seed, advisor="bot:greedy")
        while not session.state()["over"]:
            if session.state()["bag"] == 0:
                outs = [c for c in session.hint(top=4096)["candidates"] if c["kind"] == "play" and c["leave"] == ""]
                if outs:
                    return session, outs[0]
            session.ai_move()
    pytest.skip("no going-out position found")


def test_a_going_out_candidate_shows_the_bonus_as_value(sessions):
    session, candidate = going_out_position(sessions)
    state = session.state()
    mover = state["to_move"]
    bonus = 2 * rack_points(state["racks"][1 - mover])
    check_candidate(candidate)
    assert candidate["value"] == bonus and candidate["total"] == candidate["score"] + bonus
    for advisor in ("bot:greedy-leave", "bot:random", BEST):
        session.advisor = advisor
        same = [c for c in session.hint(top=4096)["candidates"] if c["text"] == candidate["text"]]
        if advisor == BEST and not same:
            continue  # a network lists only the 64 best by static value
        [same] = same
        assert (same["score"], same["value"], same["total"]) == (candidate["score"], bonus, candidate["total"]), advisor
        if advisor == BEST:
            assert same["win"] in (0.0, 0.5, 1.0), "the result of going out is known"
    session.specs[mover] = "human"
    state = session.human_move(as_body(candidate))
    assert state["history"][-1]["score"] == candidate["score"], "the history's score is the candidate's score"
    assert state["result"]["ended_by"] == "out" and state["result"]["adjustment"][mover] == bonus
    check_finished_game(state)


# ---- the stored analyses -----------------------------------------------------------------------


def test_the_analysis_of_each_ai_turn_is_kept(sessions):
    session = sessions.create(["human", "bot:greedy"], seed=21)
    session.human_move({"pass": True})
    response = session.ai_move()
    assert response["analysis"]["turn"] == 2
    assert session.analysis(2) == response["analysis"]
    assert [h["analysed"] for h in session.state()["history"]] == [False, True]
    for turn in (1, 3, 0, 99):
        with pytest.raises(NotFound, match=f"turn {turn}"):
            session.analysis(turn)
    session.human_move({"pass": True})
    session.ai_move()
    assert set(session.analyses) == {2, 4}


# ---- a network as player and as advisor --------------------------------------------------------


def test_a_model_plays_and_advises(sessions):
    session = sessions.create([BEST, "bot:greedy-leave"], seed=16)
    assert session.advisor == BEST and session.state()["advisor"] == BEST, "the advisor defaults to the model"
    response = session.ai_move()
    analysis = response["analysis"]
    assert analysis["player"] == f"{TINY_RUN} (best)" and 1 <= analysis["considered"] <= 64
    candidates = analysis["candidates"]
    assert sum(c["chosen"] for c in candidates) == 1
    chosen = next(c for c in candidates if c["chosen"])
    assert chosen["total"] == max(c["total"] for c in candidates)
    for c in candidates:
        check_candidate(c)
        assert 0.0 <= c["win"] <= 1.0
    hint = session.hint()
    assert hint["advisor"] == f"{TINY_RUN} (best)" and not any(c["chosen"] for c in hint["candidates"])

    human = sessions.create(["human", "bot:greedy"], seed=17, advisor=BEST)
    hint = human.hint(top=3)
    assert hint["advisor"] == f"{TINY_RUN} (best)" and len(hint["candidates"]) == 3
    assert all(c["win"] is not None for c in hint["candidates"])
    play_out(session)
    check_finished_game(session.state())


def test_the_chosen_move_is_listed_even_outside_the_top(sessions):
    for seed in range(60):
        session = sessions.create(["bot:random", "bot:random"], seed=seed)
        response = session.ai_move(top=1)
        candidates = response["analysis"]["candidates"]
        if not candidates[0]["chosen"]:
            assert len(candidates) == 2 and candidates[1]["chosen"] and not candidates[0]["chosen"]
            assert candidates[1]["total"] <= candidates[0]["total"]
            return
    pytest.fail("random always played the top move")


# ---- the session list --------------------------------------------------------------------------


def test_sessions_are_evicted_least_recently_used_first(sessions):
    sessions.limit = 3
    a, b, c = (sessions.create(["bot:greedy", "bot:greedy"], seed=i) for i in range(3))
    sessions.get(a.id)  # a is now the most recent
    d = sessions.create(["bot:greedy", "bot:greedy"], seed=3)
    with pytest.raises(NotFound):
        sessions.get(b.id)
    assert {sessions.get(s.id).id for s in (a, c, d)} == {a.id, c.id, d.id}


def test_unknown_ids_and_delete(sessions):
    with pytest.raises(NotFound, match="no game") as raised:
        sessions.get("g_00000000")
    assert isinstance(raised.value, KeyError) and str(raised.value) == "there is no game g_00000000"
    session = sessions.create(["human", "bot:random"], seed=1)
    assert session.id.startswith("g_") and sessions.get(session.id) is session
    sessions.delete(session.id)
    sessions.delete(session.id)  # twice is fine
    with pytest.raises(NotFound):
        sessions.get(session.id)


def test_create_refusals(sessions):
    for specs in (["human"], ["human", "bot:greedy", "bot:random"], [], ["human", "bot:nope"], "human", None,
                  ["human", "model:../x/best.pt"], ["human", f"model:{TINY_RUN}/step_000999999.pt"], ["human", 5]):
        with pytest.raises(PlayerError):
            sessions.create(specs)
    with pytest.raises(PlayerError, match="advisor"):
        sessions.create(["human", "bot:greedy"], advisor="human")
    with pytest.raises(PlayerError):
        sessions.create(["human", "bot:greedy"], advisor="model:nope/best.pt")
    for seed in (-1, 2**63, "5", True, 1.5):
        with pytest.raises(PlayerError, match="seed must be"):
            sessions.create(["human", "bot:greedy"], seed=seed)
    assert sessions.sessions == {}
    random_seed = sessions.create(["human", "human"])
    assert 0 <= random_seed.seed < 2**48 and random_seed.advisor == "bot:greedy-leave"
    assert sessions.create(["human", "human"], seed=0).seed == 0
    assert sessions.create(["human", "human"], seed=2**63 - 1).seed == 2**63 - 1
