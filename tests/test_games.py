"""Saved games: `Engine.generate_games`, `GameSet` (reading positions back out of a set) and
`scrabble_ai.games` (sets on disk: creating, stopping, continuing, crash safety, refusals).

Positions are checked against the games replayed by hand through `_core.Game`, move by move from
the text `GameSet.game` gives, so the expected numbers do not come from the code under test.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scrabble_ai import _core
from scrabble_ai import engine as engine_module
from scrabble_ai import games as games_module
from scrabble_ai.games import GamesError

SET_GAMES = 240
BLOCK = 80


def quiet(_text: str) -> None:
    pass


@pytest.fixture(scope="module")
def loaded(engine, leaves):
    return engine_module.Loaded(engine, leaves, engine.lexicon_id, None, engine_module.leaves_hash(leaves))


@pytest.fixture(scope="module")
def played(loaded, tmp_path_factory):
    """One set of SET_GAMES games made in blocks of BLOCK, and its games decoded."""
    directory = tmp_path_factory.mktemp("games") / "probe-set"
    meta = games_module.generate(loaded, directory, SET_GAMES, block=BLOCK, say=quiet)
    games = _core.GameSet(directory)
    decoded = [games.game(i) for i in range(len(games))]
    return SimpleNamespace(directory=directory, meta=meta, games=games, decoded=decoded)


# ---- helpers -----------------------------------------------------------------------------------


def parse(text: str):
    """(kind, tiles, score, squares) of a move as `GameSet.game` writes it: "pass", "exchange AB?",
    or a placement "8D QU(I)Z 48" / "H8 JOG 22" (letters in brackets were already on the board,
    lower case is a blank). `tiles` is what `Game.play_tiles` takes, or the exchanged letters."""
    if text == "pass":
        return "pass", None, 0, []
    if text.startswith("exchange "):
        return "exchange", text[len("exchange "):], 0, []
    position, word, score = text.split(" ")
    if position[0].isdigit():  # across: row number then column letter
        row, col, dr, dc = int(position[:-1]) - 1, ord(position[-1]) - ord("A"), 0, 1
    else:  # down: column letter then row number
        row, col, dr, dc = int(position[1:]) - 1, ord(position[0]) - ord("A"), 1, 0
    tiles, through = [], False
    for ch in word:
        if ch in "()":
            through = ch == "("
            continue
        if not through:
            tiles.append((row, col, ch.upper(), ch.islower()))
        row, col = row + dr, col + dc
    return "play", tiles, int(score), sorted(r * 15 + c for r, c, _, _ in tiles)


def letters(state: dict) -> int:
    return sum(ch != "." for row in state["board"] for ch in row)


def replay(engine, leaves, seed: int, moves: list[str]):
    """Play the recorded moves on a fresh game of that seed. Per turn: the mover, the move, the
    tiles on the board after it, the mover's spread after it, and whether the game was over."""
    game = _core.Game(engine, leaves, seed)
    turns = []
    for text in moves:
        before = game.state()
        assert not before["over"], f"seed {seed}: a move after the end"
        kind, tiles, score, squares = parse(text)
        if kind == "pass":
            done = game.pass_turn()
        elif kind == "exchange":
            done = game.exchange(tiles)
        else:
            done = game.play_tiles(tiles)
            assert done["text"] == " ".join(text.split(" ")[:2])
        after = game.state()
        mover = before["to_move"]
        turns.append(SimpleNamespace(kind=kind, score=score, squares=squares, placed=len(squares), mover=mover,
                                     board_after=letters(after), over=after["over"],
                                     spread_after=after["scores"][mover] - after["scores"][1 - mover]))
    return turns, game.state()


def outcome(scores, seat: int) -> float:
    spread = scores[seat] - scores[1 - seat]
    return 1.0 if spread > 0 else 0.5 if spread == 0 else 0.0


def sample_arrays(n: int) -> dict:
    return {"board": np.zeros((n, _core.SQUARES), np.uint8), "leave": np.zeros((n, _core.TILE_TYPES), np.uint8),
            "unseen": np.zeros((n, _core.TILE_TYPES), np.uint8), "meta": np.zeros((n, _core.META), np.int16),
            "leave_value": np.zeros(n, np.float32), "targets": np.zeros((n, _core.TARGETS), np.float32),
            "squares": np.zeros((n, 2 * _core.SQUARES), np.uint8), "turns": np.zeros(n, np.int32)}


def sample(games_set, engine, leaves, games, picks, belief_samples=0) -> dict:
    out = sample_arrays(len(games))
    games_set.sample(engine, leaves, np.asarray(games, np.uint64), np.asarray(picks, np.float32), belief_samples,
                     *out.values())
    return out


def write_set(directory: Path, data: bytes, lengths: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "games.bin").write_bytes(data)
    offsets = np.concatenate([[0], np.cumsum(lengths.astype(np.uint64))]).astype("<u8")
    (directory / "index.bin").write_bytes(offsets.tobytes())


# ---- Engine.generate_games ---------------------------------------------------------------------


def test_the_constants():
    assert _core.TARGETS == 8
    assert [_core.T_WIN, _core.T_SPREAD, _core.T_OPP_SCORE, _core.T_OPP_BINGO, _core.T_HAS_OPP, _core.T_OWN_SCORE,
            _core.T_OWN_BINGO, _core.T_HAS_OWN] == list(range(8))


def test_the_bindings_document_the_right_methods():
    """Each method's docstring is its own: the doc comment of `play_pairs` belongs above
    `play_pairs` in crates/scrabble-py/src/lib.rs, not above `generate_games`."""
    assert _core.Engine.generate_games.__doc__.startswith("Play `count` games by the static evaluator")
    assert "swapping seats" in (_core.Engine.play_pairs.__doc__ or "")
    assert "position" in _core.GameSet.sample.__doc__ and "saved games" in _core.GameSet.__doc__


def test_generate_games_is_deterministic_and_lengths_add_up(engine, leaves):
    data, lengths, totals = engine.generate_games(leaves, 5_000, 60)
    again = engine.generate_games(leaves, 5_000, 60)
    assert isinstance(data, bytes) and data == again[0]
    assert lengths.dtype == np.uint32 and lengths.shape == (60,) and np.array_equal(lengths, again[1])
    assert totals == again[2]
    assert int(lengths.astype(np.int64).sum()) == len(data)
    assert set(totals) == {"games", "moves", "score_sum", "first_player_wins", "bingos"} and totals["games"] == 60
    other = engine.generate_games(leaves, 5_001, 60)
    assert other[0] != data
    # The games are one per seed: those from 5,001 on are the same games shifted by one.
    first = int(lengths[0])
    assert other[0][: len(data) - first] == data[first:]
    assert engine.generate_games(leaves, 5_001, 59)[0] == data[first:]
    # A block is the same games whether made at once or in pieces.
    pieces = engine.generate_games(leaves, 5_000, 25)[0] + engine.generate_games(leaves, 5_025, 35)[0]
    assert pieces == data


def test_totals_agree_with_the_decoded_games(engine, leaves, tmp_path):
    data, lengths, totals = engine.generate_games(leaves, 9_000, 120)
    write_set(tmp_path / "set", data, lengths)
    games = _core.GameSet(tmp_path / "set")
    assert len(games) == 120
    decoded = [games.game(i) for i in range(120)]
    assert [seed for seed, *_ in decoded] == list(range(9_000, 9_120))
    assert totals["moves"] == sum(len(moves) for *_, moves in decoded)
    assert totals["score_sum"] == sum(sum(scores) for _, scores, _, _ in decoded)
    assert totals["first_player_wins"] == sum(outcome(scores, 0) for _, scores, _, _ in decoded)
    bingos = sum(parse(text)[0] == "play" and len(parse(text)[3]) == 7 for *_, moves in decoded for text in moves)
    assert totals["bingos"] == bingos
    for seed, scores, went_out, moves in decoded:
        assert isinstance(seed, int) and isinstance(went_out, bool) and len(scores) == 2 and moves
        assert all(isinstance(text, str) for text in moves)


def test_the_moves_replay_to_the_recorded_result(engine, leaves, played):
    """Every recorded game, played again move by move from its seed, ends where the record says,
    and a game that went out ended on a placement."""
    for seed, scores, went_out, moves in played.decoded[:60]:
        turns, final = replay(engine, leaves, seed, moves)
        assert final["over"], seed
        assert final["scores"] == scores, seed
        assert (final["ended_by"] == "out") == went_out, seed
        if went_out:
            assert turns[-1].kind == "play"
        assert all(not t.over for t in turns[:-1]), seed


def test_without_varied_openings_the_games_are_the_greedy_leave_bots(engine, leaves):
    seeds = np.arange(40, dtype=np.uint64) + np.uint64(777)
    data, lengths, totals = engine.generate_games(leaves, 777, 40, opening_plies=0)
    pairs = engine.play_pairs("greedy-leave", leaves, "greedy-leave", leaves, seeds)
    first_scores = np.stack([pairs["score_a"].reshape(-1, 2)[:, 0], pairs["score_b"].reshape(-1, 2)[:, 0]], axis=1)
    turns = pairs["turns"].reshape(-1, 2)[:, 0]

    at, recorded = 0, []
    for length in lengths:
        recorded.append(data[at:at + int(length)])
        at += int(length)
    for i, record in enumerate(recorded):
        assert int.from_bytes(record[:8], "little") == 777 + i
        score0 = int.from_bytes(record[8:10], "little", signed=True)
        score1 = int.from_bytes(record[10:12], "little", signed=True)
        assert [score0, score1] == first_scores[i].tolist(), i
        assert record[13] == turns[i], f"game {i}: one move recorded per turn played"
    # A temperature of 0 varies nothing either.
    assert engine.generate_games(leaves, 777, 40, opening_plies=4, opening_temperature=0.0)[0] == data
    # With varied openings some games take another course.
    varied = engine.generate_games(leaves, 777, 40)
    assert varied[0] != data


# ---- GameSet -----------------------------------------------------------------------------------


def test_a_game_set_reads_its_games(played):
    games = played.games
    assert len(games) == SET_GAMES
    first_seed = played.meta["first_seed"]
    assert [seed for seed, *_ in played.decoded] == [first_seed + i for i in range(SET_GAMES)]
    for bad in (SET_GAMES, SET_GAMES + 5, 10**9):
        with pytest.raises(ValueError):
            games.game(bad)


def test_opening_a_damaged_set_is_a_value_error(engine, leaves, tmp_path):
    data, lengths, _ = engine.generate_games(leaves, 1, 10)
    with pytest.raises(ValueError):
        _core.GameSet(tmp_path / "missing")
    good = tmp_path / "good"
    write_set(good, data, lengths)
    assert len(_core.GameSet(good)) == 10

    torn = tmp_path / "torn"
    write_set(torn, data, lengths)
    (torn / "index.bin").write_bytes((torn / "index.bin").read_bytes()[:-3])
    with pytest.raises(ValueError):
        _core.GameSet(torn)

    disordered = tmp_path / "disordered"
    write_set(disordered, data, lengths)
    offsets = np.frombuffer((disordered / "index.bin").read_bytes(), "<u8").copy()
    offsets[[2, 3]] = offsets[[3, 2]]
    (disordered / "index.bin").write_bytes(offsets.tobytes())
    with pytest.raises(ValueError):
        _core.GameSet(disordered)

    short = tmp_path / "short"
    write_set(short, data, lengths)
    (short / "games.bin").write_bytes(data[:-10])
    with pytest.raises(ValueError):
        _core.GameSet(short)

    nogames = tmp_path / "nogames"
    write_set(nogames, data, lengths)
    (nogames / "games.bin").unlink()
    with pytest.raises(ValueError):
        _core.GameSet(nogames)

    # A record cut short inside the file is refused when it is read.
    cut = tmp_path / "cut"
    write_set(cut, data, lengths)
    offsets = np.frombuffer((cut / "index.bin").read_bytes(), "<u8").copy()
    offsets[1] -= 5
    (cut / "index.bin").write_bytes(offsets.tobytes())
    with pytest.raises(ValueError):
        _core.GameSet(cut).game(0)


def test_sample_shapes_dtypes_and_determinism(engine, leaves, played):
    n = 50
    games = (np.arange(n) * 7) % SET_GAMES
    picks = (np.arange(n) * 0.137) % 1.0
    out = sample(played.games, engine, leaves, games, picks)
    assert out["targets"].shape == (n, _core.TARGETS) and out["squares"].shape == (n, 450)
    again = sample(played.games, engine, leaves, games, picks)
    for key in out:
        assert np.array_equal(out[key], again[key]), key
    assert np.isfinite(out["targets"]).all() and np.isfinite(out["leave_value"]).all()
    assert set(np.unique(out["squares"])) <= {0, 1}
    # One position from each game, chosen evenly over the turns that leave something to predict.
    for i, game in enumerate(games):
        _, _, went_out, moves = played.decoded[game]
        eligible = len(moves) - went_out
        assert out["turns"][i] == min(int(picks[i] * eligible), eligible - 1)


def test_sample_rows_agree_with_the_games_replayed_by_hand(engine, leaves, played):
    rows = []
    for game in range(0, SET_GAMES, 6):
        _, _, went_out, moves = played.decoded[game]
        eligible = len(moves) - went_out
        for turn in sorted({0, 1, eligible // 2, eligible - 2, eligible - 1}):
            if 0 <= turn < eligible:
                rows.append((game, turn, (turn + 0.5) / eligible))
    out = sample(played.games, engine, leaves, [r[0] for r in rows], [r[2] for r in rows])
    replayed = {}
    checked_spread = 0
    for i, (game, turn, _) in enumerate(rows):
        seed, scores, went_out, moves = played.decoded[game]
        if game not in replayed:
            replayed[game] = replay(engine, leaves, seed, moves)[0]
        turns = replayed[game]
        this = turns[turn]
        board, meta, targets, squares = out["board"][i], out["meta"][i], out["targets"][i], out["squares"][i]
        assert out["turns"][i] == turn
        # The afterstate of the move played on that turn.
        assert int(((board & _core.CODE_LETTER) != 0).sum()) == this.board_after, (game, turn)
        assert int(((board & _core.CODE_PLACED) != 0).sum()) == this.placed, (game, turn)
        total = int(((board & _core.CODE_LETTER) != 0).sum()) + int(out["leave"][i].sum()) + int(out["unseen"][i].sum())
        assert total == 100, "tiles are conserved: board + leave + unseen"
        assert meta[_core.M_KIND] == {"play": 0, "exchange": 1, "pass": 2}[this.kind]
        assert meta[_core.M_TERMINAL] != 1, "a move that goes out is never a training position"
        assert meta[_core.M_BELIEF_INFERRED] == 0
        if not this.over:
            assert meta[_core.M_SCORE] == this.score
            assert meta[_core.M_SPREAD] == this.spread_after, (game, turn)
            checked_spread += 1
        # How it ended for the mover, and what was still to gain.
        assert targets[_core.T_WIN] == outcome(scores, this.mover), (game, turn)
        final_spread = scores[this.mover] - scores[1 - this.mover]
        assert targets[_core.T_SPREAD] == final_spread - meta[_core.M_SPREAD]
        # What the next two moves did.
        for ahead, (score, bingo, has) in ((1, (_core.T_OPP_SCORE, _core.T_OPP_BINGO, _core.T_HAS_OPP)),
                                           (2, (_core.T_OWN_SCORE, _core.T_OWN_BINGO, _core.T_HAS_OWN))):
            plane = squares[(ahead - 1) * 225:ahead * 225]
            if turn + ahead < len(moves):
                following = turns[turn + ahead]
                assert targets[has] == 1.0 and targets[score] == following.score, (game, turn, ahead)
                assert targets[bingo] == float(following.placed == 7)
                assert np.flatnonzero(plane).tolist() == following.squares, (game, turn, ahead)
            else:
                assert targets[has] == 0.0 and targets[score] == 0.0 and targets[bingo] == 0.0
                assert not plane.any()
    assert len(rows) > 150 and checked_spread > 100


def test_belief_inference_only_adds_the_inference(engine, leaves, played):
    games = np.arange(0, SET_GAMES, 2)
    picks = ((np.arange(len(games)) * 0.31) % 1.0)
    plain = sample(played.games, engine, leaves, games, picks, 0)
    inferred = sample(played.games, engine, leaves, games, picks, 3)
    assert (inferred["meta"][:, _core.M_BELIEF_INFERRED] == 1).any()
    assert (plain["meta"][:, _core.M_BELIEF_INFERRED] == 0).all()
    for key in ("board", "leave", "unseen", "leave_value", "targets", "squares", "turns"):
        assert np.array_equal(plain[key], inferred[key]), key
    assert np.array_equal(plain["meta"][:, :_core.M_BELIEF_INFERRED], inferred["meta"][:, :_core.M_BELIEF_INFERRED])
    again = sample(played.games, engine, leaves, games, picks, 3)
    assert np.array_equal(again["meta"], inferred["meta"]), "the same inferences every time"


def test_sample_refuses_bad_shapes_and_missing_games(engine, leaves, played):
    n = 8
    games, picks = np.arange(n, dtype=np.uint64), np.full(n, 0.5, np.float32)
    good = sample_arrays(n)
    played.games.sample(engine, leaves, games, picks, 0, *good.values())
    for key in good:
        bad = dict(sample_arrays(n))
        shape = list(bad[key].shape)
        shape[0] += 1
        bad[key] = np.zeros(shape, bad[key].dtype)
        with pytest.raises(ValueError):
            played.games.sample(engine, leaves, games, picks, 0, *bad.values())
    with pytest.raises(ValueError):
        played.games.sample(engine, leaves, games, picks[:-1], 0, *sample_arrays(n).values())
    for beyond in (SET_GAMES, 10**6):
        far = games.copy()
        far[3] = beyond
        with pytest.raises(ValueError):
            played.games.sample(engine, leaves, far, picks, 0, *sample_arrays(n).values())
    empty = sample_arrays(0)
    played.games.sample(engine, leaves, np.zeros(0, np.uint64), np.zeros(0, np.float32), 0, *empty.values())


# ---- games.py: sets on disk --------------------------------------------------------------------


def test_set_dir_and_seeds(tmp_path):
    assert games_module.set_dir("abc-1_2", tmp_path) == tmp_path / "abc-1_2"
    assert games_module.set_dir("abc") == games_module.GAMES_DIR / "abc"
    assert games_module.set_dir(str(tmp_path / "x")) == tmp_path / "x", "a value with a separator is a folder"
    for bad in ("", "a b", "a.b", "naïve"):
        with pytest.raises(GamesError):
            games_module.set_dir(bad, tmp_path)
    seed = games_module.default_first_seed("twenty-million")
    assert seed == games_module.default_first_seed("twenty-million")
    assert seed == games_module.SEED_BASE + (zlib.crc32(b"twenty-million") % 100_000) * 1_000_000_000
    assert games_module.SEED_BASE <= seed < games_module.SEED_BASE + 100_000 * 1_000_000_000
    assert (seed - games_module.SEED_BASE) % 1_000_000_000 == 0, "a billion seeds of its own"
    assert games_module.default_first_seed("other") != seed


def test_generate_creates_the_files_and_counts(loaded, played, engine, leaves):
    directory, meta = played.directory, played.meta
    assert meta == games_module.read_meta(directory)
    assert meta["format"] == games_module.FORMAT and meta["name"] == "probe-set"
    assert (meta["games"], meta["target"]) == (SET_GAMES, SET_GAMES)
    assert (meta["lexicon_id"], meta["leaves_hash"]) == (loaded.lexicon_id, loaded.leaves_hash)
    assert (meta["opening_plies"], meta["opening_temperature"]) == (4, 6.0)
    assert meta["first_seed"] == games_module.default_first_seed("probe-set")
    assert meta["player"] == "greedy-leave"
    assert meta["bytes"] == (directory / "games.bin").stat().st_size
    offsets = np.frombuffer((directory / "index.bin").read_bytes(), "<u8")
    assert len(offsets) == SET_GAMES + 1 and offsets[0] == 0 and offsets[-1] == meta["bytes"]
    assert meta["moves"] == sum(len(moves) for *_, moves in played.decoded)
    assert meta["score_sum"] == sum(sum(scores) for _, scores, _, _ in played.decoded)
    assert meta["first_player_wins"] == sum(outcome(scores, 0) for _, scores, _, _ in played.decoded)
    assert meta["bingos"] == sum(parse(t)[0] == "play" and len(parse(t)[3]) == 7 for *_, moves in played.decoded for t in moves)
    assert meta["seconds"] >= 0 and meta["created"] <= meta["updated"]
    assert not (directory / "running.json").exists() and not (directory / "STOP").exists()
    # The file is the engine's games for those seeds, block after block.
    data, _, _ = engine.generate_games(leaves, meta["first_seed"], SET_GAMES, 4, 6.0)
    assert (directory / "games.bin").read_bytes() == data


def test_stopping_and_continuing_gives_the_same_files(loaded, played, tmp_path):
    directory = tmp_path / "elsewhere" / "probe-set"  # same name, so the same seeds
    calls, said = [], []

    def should_stop():
        calls.append((directory / "running.json").exists())
        return len(calls) > 1

    meta = games_module.generate(loaded, directory, SET_GAMES, block=BLOCK, should_stop=should_stop, say=said.append)
    assert calls == [True, True], "checked before each block, with the running marker present"
    assert (meta["games"], meta["target"]) == (BLOCK, SET_GAMES)
    assert games_module.read_meta(directory) == meta
    assert any("stopping" in line for line in said)
    assert not (directory / "running.json").exists()
    created = meta["created"]

    meta = games_module.generate(loaded, directory, SET_GAMES, block=BLOCK, say=quiet)
    assert meta["games"] == SET_GAMES and meta["created"] == created
    for name in ("games.bin", "index.bin"):
        assert (directory / name).read_bytes() == (played.directory / name).read_bytes(), name
    for key in ("games", "bytes", "moves", "score_sum", "first_player_wins", "bingos", "first_seed"):
        assert meta[key] == played.meta[key], key


def test_a_stop_file_stops_it_after_the_block(loaded, tmp_path):
    directory = tmp_path / "stopped"
    directory.mkdir()
    (directory / "STOP").write_text("", encoding="utf-8")
    os.utime(directory / "STOP", (1_000_000_000, 1_000_000_000))  # left by a process long gone: not a request now

    def ask_to_stop():
        (directory / "STOP").write_text("", encoding="utf-8")
        return False

    meta = games_module.generate(loaded, directory, 100, block=40, should_stop=ask_to_stop, say=quiet)
    assert meta["games"] == 40
    assert not (directory / "STOP").exists() and not (directory / "running.json").exists()


def test_a_stop_asked_for_while_the_process_was_starting_still_counts(loaded, tmp_path):
    """The app writes STOP when Stop is clicked. If the process is not up yet, the file is newer
    than the process, and it must not be swept away as a leftover."""
    directory = tmp_path / "early"
    directory.mkdir()
    (directory / "STOP").write_text("", encoding="utf-8")  # written after this process began
    meta = games_module.generate(loaded, directory, 100, block=40, say=quiet)
    assert meta["games"] == 0 and meta["target"] == 100
    assert not (directory / "STOP").exists() and not (directory / "running.json").exists()
    assert games_module.generate(loaded, directory, 100, block=40, say=quiet)["games"] == 100


def test_a_torn_tail_is_cut_off_on_the_next_call(loaded, played, tmp_path):
    directory = tmp_path / "crash" / "probe-set"
    stop = iter([False, True])
    meta = games_module.generate(loaded, directory, SET_GAMES, block=BLOCK, should_stop=lambda: next(stop), say=quiet)
    assert meta["games"] == BLOCK
    # A block that did not finish: bytes past what meta.json records in both files.
    with open(directory / "games.bin", "ab") as fh:
        fh.write(os.urandom(777))
    with open(directory / "index.bin", "ab") as fh:
        fh.write(np.array([meta["bytes"] + 300, meta["bytes"] + 777], "<u8").tobytes() + b"\x01\x02\x03")
    again = games_module.generate(loaded, directory, BLOCK, block=BLOCK, say=quiet)  # nothing to add
    assert again["games"] == BLOCK
    assert (directory / "games.bin").stat().st_size == meta["bytes"]
    assert (directory / "index.bin").stat().st_size == 8 * (BLOCK + 1)
    assert len(_core.GameSet(directory)) == BLOCK
    # Torn again, then carried on: the result is the set made in one go.
    with open(directory / "games.bin", "ab") as fh:
        fh.write(b"\xff" * 1000)
    with open(directory / "index.bin", "ab") as fh:
        fh.write(b"\x07" * 13)
    games_module.generate(loaded, directory, SET_GAMES, block=BLOCK, say=quiet)
    for name in ("games.bin", "index.bin"):
        assert (directory / name).read_bytes() == (played.directory / name).read_bytes(), name


def test_open_games_reads_only_what_meta_records(loaded, tmp_path):
    """meta.json is the record of what is complete ("anything past what meta.json records was
    written by a block that did not finish"; a forced stop "keeps what was complete",
    docs/APP_API.md), so a set read while a block is being written, or after a crash or a kill in
    one, gives the games it records and no others."""
    outcomes = {}
    for case in ("block written, meta.json not yet", "killed while writing the index"):
        directory = tmp_path / case.split()[0]
        stop = iter([False, True])
        meta = games_module.generate(loaded, directory, 200, block=60, should_stop=lambda: next(stop), say=quiet)
        data, lengths, _ = loaded.engine.generate_games(loaded.leaves, meta["first_seed"] + 60, 60)
        offsets = (meta["bytes"] + np.cumsum(lengths.astype(np.uint64))).astype("<u8").tobytes()
        with open(directory / "games.bin", "ab") as fh:
            fh.write(data)
        with open(directory / "index.bin", "ab") as fh:
            fh.write(offsets if case.startswith("block") else offsets[:21])
        try:
            games, opened = games_module.open_games(directory, loaded)
            outcomes[case] = (opened["games"], len(games))
        except GamesError as error:
            outcomes[case] = f"GamesError: {error}"
    assert outcomes == {"block written, meta.json not yet": (60, 60), "killed while writing the index": (60, 60)}


def test_generate_refuses_other_settings_word_lists_and_leave_values(loaded, played, tmp_path):
    directory = tmp_path / "copy" / "probe-set"
    shutil.copytree(played.directory, directory)
    before = (directory / "games.bin").read_bytes()
    for kwargs in ({"opening_plies": 3}, {"opening_temperature": 2.0}, {"first_seed": 12345}):
        with pytest.raises(GamesError, match="other settings"):
            games_module.generate(loaded, directory, SET_GAMES + 10, say=quiet, **kwargs)
    # The set's own settings, given again, are fine.
    games_module.generate(loaded, directory, SET_GAMES, opening_plies=4, opening_temperature=6, say=quiet)
    for other in (dataclasses.replace(loaded, lexicon_id="OTHER-0123456789ab"),
                  dataclasses.replace(loaded, leaves_hash="000000000000")):
        with pytest.raises(GamesError, match="another word list or other leave values"):
            games_module.generate(other, directory, SET_GAMES + 10, say=quiet)
    assert (directory / "games.bin").read_bytes() == before
    assert games_module.read_meta(directory)["games"] == SET_GAMES

    for kwargs in ({"games": 0}, {"games": -5}, {"games": 10, "opening_plies": 21}, {"games": 10, "opening_plies": -1},
                   {"games": 10, "opening_temperature": -1.0}, {"games": 10, "opening_temperature": float("nan")}):
        with pytest.raises(GamesError):
            games_module.generate(loaded, tmp_path / "fresh", say=quiet, **kwargs)
    # Games without a readable meta.json are not overwritten.
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "games.bin").write_bytes(b"x" * 100)
    (orphan / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(GamesError, match="no readable meta.json"):
        games_module.generate(loaded, orphan, 10, say=quiet)
    assert (orphan / "games.bin").read_bytes() == b"x" * 100


def test_a_new_set_takes_the_opening_settings_it_is_given(loaded, engine, leaves, tmp_path):
    meta = games_module.generate(loaded, tmp_path / "plain", 30, opening_plies=0, opening_temperature=1.5, first_seed=42,
                                 say=quiet)
    assert (meta["opening_plies"], meta["opening_temperature"], meta["first_seed"]) == (0, 1.5, 42)
    assert (tmp_path / "plain" / "games.bin").read_bytes() == engine.generate_games(leaves, 42, 30, opening_plies=0)[0]


def test_open_games_errors(loaded, played, tmp_path):
    games, meta = games_module.open_games(played.directory, loaded)
    assert len(games) == SET_GAMES and meta["games"] == SET_GAMES
    with pytest.raises(GamesError, match="no game set"):
        games_module.open_games(tmp_path / "nothing", loaded)
    with pytest.raises(GamesError, match="another word list|word list"):
        games_module.open_games(played.directory, dataclasses.replace(loaded, lexicon_id="OTHER-0123456789ab"))
    with pytest.raises(GamesError, match="leave values"):
        games_module.open_games(played.directory, dataclasses.replace(loaded, leaves_hash="000000000000"))

    empty = tmp_path / "empty"
    games_module.generate(loaded, empty, 50, should_stop=lambda: True, say=quiet)
    assert games_module.read_meta(empty)["games"] == 0
    with pytest.raises(GamesError, match="no games yet"):
        games_module.open_games(empty, loaded)

    damaged = tmp_path / "damaged"
    games_module.generate(loaded, damaged, 20, say=quiet)
    meta = json.loads((damaged / "meta.json").read_text(encoding="utf-8"))
    meta["games"] = 25
    (damaged / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(GamesError, match="damaged"):
        games_module.open_games(damaged, loaded)
    (damaged / "index.bin").unlink()
    with pytest.raises(GamesError):
        games_module.open_games(damaged, loaded)


def test_read_meta_and_summary(loaded, played, tmp_path):
    summary = games_module.summary(played.directory)
    meta = played.meta
    assert set(summary) == {"name", "games", "target", "bytes", "lexicon_id", "leaves_hash", "opening_plies",
                            "opening_temperature", "mean_turns", "mean_score", "first_player_win_rate",
                            "bingos_per_game", "games_per_second", "seconds", "created", "updated"}
    assert summary["name"] == "probe-set" and summary["games"] == SET_GAMES and summary["target"] == SET_GAMES
    assert summary["mean_turns"] == pytest.approx(meta["moves"] / SET_GAMES)
    assert summary["mean_score"] == pytest.approx(meta["score_sum"] / (2 * SET_GAMES))
    assert summary["first_player_win_rate"] == pytest.approx(meta["first_player_wins"] / SET_GAMES)
    assert summary["bingos_per_game"] == pytest.approx(meta["bingos"] / SET_GAMES)
    if meta["seconds"]:
        assert summary["games_per_second"] == pytest.approx(SET_GAMES / meta["seconds"])
    assert summary["seconds"] == meta["seconds"]
    assert 300 < summary["mean_score"] * 2 < 1000 and 0 <= summary["first_player_win_rate"] <= 1

    empty = tmp_path / "empty"
    games_module.generate(loaded, empty, 50, should_stop=lambda: True, say=quiet)
    blank = games_module.summary(empty)
    assert blank["games"] == 0 and blank["target"] == 50
    assert [blank[k] for k in ("mean_turns", "mean_score", "first_player_win_rate", "bingos_per_game")] == [None] * 4

    assert games_module.summary(tmp_path / "none") is None
    (tmp_path / "wrong").mkdir()
    (tmp_path / "wrong" / "meta.json").write_text(json.dumps({"format": 99, "games": 3}), encoding="utf-8")
    assert games_module.read_meta(tmp_path / "wrong") is None and games_module.summary(tmp_path / "wrong") is None
    (tmp_path / "list").mkdir()
    (tmp_path / "list" / "meta.json").write_text("[1, 2]", encoding="utf-8")
    assert games_module.read_meta(tmp_path / "list") is None
