"""The network inputs added on 2026-11-14: the move's score and other extra scalars, the plane for
the opponent's last move, board openness, hook squares, and the inference about the opponent's
rack. Also what keeps networks from before then working: their layout is unchanged, and a
checkpoint that does not mention the new settings is read as not having them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scrabble_ai import _core
from scrabble_ai.config import BEFORE_IT_EXISTED, Config, ConfigError, ModelConfig, config_from_dict, load_config
from scrabble_ai.features import (BELIEF_VECTOR, EXTRA_SCALARS, HOOK_PLANES, OPENNESS_PLANES, OPENNESS_SCALARS, VECTOR,
                                  FeatureBuilder, FeatureSet, engine_inputs, plane_count, to_device)
from scrabble_ai.model import ValueNet
from scrabble_ai.replay import Replay
from scrabble_ai.selfplay import CandidateBuffers, Scorer

N, SQUARES, TYPES = 15, 225, 27
ORIGINAL = FeatureSet()
EVERYTHING = FeatureSet.everything()
CPU = torch.device("cpu")


def square(name: str) -> int:
    return (int(name[1:]) - 1) * N + ord(name[0]) - ord("A")


def decisions(engine, leaves, steps: int, belief_samples: int = 4, slots: int = 6, k: int = 8, pick=lambda count: 0):
    """Play `steps` lockstep moves and return the candidates of the decisions then pending:
    (board, leave, unseen, meta, leave_value) with one row per candidate, and the environment."""
    env = _core.BatchEnv(engine, leaves, slots, k, 2, belief_samples)
    buffers = CandidateBuffers(slots, k)
    for slot in range(slots):
        env.reset(slot, 300 + slot, "external", "external")
    for _ in range(steps):
        buffers.prepare(env)
        env.apply(np.array([pick(int(c)) for c in buffers.count], np.int32))
    buffers.prepare(env)
    valid = (np.arange(k)[None, :] < buffers.count[:, None]) & (buffers.status == _core.STATUS_DECISION)[:, None]
    rows = tuple(array[valid].copy() for array in (buffers.board, buffers.leave, buffers.unseen, buffers.meta, buffers.leave_value))
    return rows, env, buffers


def build(engine, inputs: FeatureSet, rows, dtype=torch.float32):
    cross, opened = engine_inputs(engine, inputs, rows[0])
    batch = to_device(*rows, cross, CPU, opened)
    return FeatureBuilder(CPU, inputs, dtype)(batch), batch


# ---- which inputs a network has ------------------------------------------------------------------


def test_sizes_of_every_input_group():
    assert (ORIGINAL.planes, ORIGINAL.vector) == (87, 72) == (plane_count(True), VECTOR)
    assert plane_count(False) == FeatureSet(cross_checks=False).planes == 33
    assert FeatureSet(extra_scalars=True).vector == 72 + len(EXTRA_SCALARS) == 85
    assert FeatureSet(last_move_plane=True).planes == 88
    assert (FeatureSet(openness=True).planes, FeatureSet(openness=True).vector) == (87 + OPENNESS_PLANES, 72 + len(OPENNESS_SCALARS))
    assert FeatureSet(hooks=True).planes == 87 + HOOK_PLANES
    assert FeatureSet(belief=True).vector == 72 + BELIEF_VECTOR == 127
    assert (EVERYTHING.planes, EVERYTHING.vector) == (100, 144)
    assert not FeatureSet(cross_checks=False).needs_cross
    assert FeatureSet(cross_checks=False, hooks=True).needs_cross and FeatureSet(cross_checks=False, openness=True).needs_cross


def test_new_settings_default_to_on_and_old_checkpoints_read_as_off():
    model = Config().model
    assert (model.extra_scalars, model.last_move_plane, model.openness, model.hooks, model.belief_samples) == (True, True, True, True, 4)
    assert FeatureSet.of(model) == EVERYTHING
    assert FeatureSet.of(ModelConfig(belief_samples=0)) == FeatureSet(True, True, True, True, True, False)

    # A checkpoint written before the settings existed does not mention them: it has none of the inputs.
    saved = Config().to_dict()
    for key in BEFORE_IT_EXISTED["model"]:
        del saved["model"][key]
    old = config_from_dict(saved)
    assert FeatureSet.of(old.model) == ORIGINAL and old.model.belief_samples == 0
    assert "extra_scalars" not in saved["model"], "the saved dict is not changed"
    # One written since says what it has, whatever that is.
    saved = Config().to_dict()
    saved["model"].update(openness=False, belief_samples=3)
    new = config_from_dict(saved)
    assert (new.model.openness, new.model.hooks, new.model.belief_samples) == (False, True, 3)
    # A config file, unlike a checkpoint, gets today's defaults for what it leaves out.
    assert load_config(None, ["model.blocks=2"]).model.hooks is True

    for bad in ("model.belief_samples=-1", "model.belief_samples=65"):
        with pytest.raises(ConfigError, match="model.belief_samples"):
            load_config(None, [bad])


def test_the_network_is_built_for_its_inputs():
    small = dict(blocks=1, channels=16, vector_hidden=16, head_hidden=16, value_channels=2)
    new = ValueNet(ModelConfig(**small))
    assert (new.stem.in_channels, new.vector[0].in_features, new.features) == (100, 144, EVERYTHING)
    old = ValueNet(ModelConfig(**small, **BEFORE_IT_EXISTED["model"]))
    assert (old.stem.in_channels, old.vector[0].in_features, old.features) == (87, 72, ORIGINAL)
    # Weights saved by a network from before the new inputs load into one built from its config.
    again = ValueNet(config_from_dict({"model": {**small, "norm": "none", "cross_checks": True, "leave_baseline": True}}).model)
    again.load_state_dict(old.state_dict())


# ---- old networks see exactly what they saw --------------------------------------------------------


def test_new_groups_are_appended_so_the_original_layout_is_unchanged(engine, leaves):
    rows, _, _ = decisions(engine, leaves, steps=7)
    (planes, vector), _ = build(engine, EVERYTHING, rows)
    (old_planes, old_vector), _ = build(engine, ORIGINAL, rows)
    assert planes.shape == (len(rows[0]), 100, N, N) and vector.shape == (len(rows[0]), 144)
    assert torch.equal(planes[:, :87], old_planes)
    assert torch.equal(vector[:, :72], old_vector)
    assert torch.isfinite(planes).all() and torch.isfinite(vector).all()
    # Each group alone lands where it does in the full set.
    (_, scalars_only), _ = build(engine, FeatureSet(extra_scalars=True), rows)
    assert torch.equal(scalars_only[:, 72:], vector[:, 72:85])
    (last, _), _ = build(engine, FeatureSet(last_move_plane=True), rows)
    assert torch.equal(last[:, 87], planes[:, 87])
    (hook_planes, _), _ = build(engine, FeatureSet(hooks=True), rows)
    assert torch.equal(hook_planes[:, 87:], planes[:, 94:100])
    (open_planes, open_vector), _ = build(engine, FeatureSet(openness=True), rows)
    assert torch.equal(open_planes[:, 87:], planes[:, 88:94]) and torch.equal(open_vector[:, 72:], vector[:, 85:89])
    (_, belief_vector), _ = build(engine, FeatureSet(belief=True), rows)
    assert torch.equal(belief_vector[:, 72:], vector[:, 89:])


def test_a_replay_saved_with_the_shorter_metadata_still_loads(tmp_path):
    old_width = 11
    replay = Replay(50)
    rng = np.random.default_rng(0)
    rows = {name: rng.integers(0, 5, size=(20, *array.shape[1:])).astype(array.dtype) for name, array in replay.arrays.items()}
    replay.add(rows)
    replay.save(tmp_path)
    # Rewrite meta.npy as a buffer from before 2026-11-14 had it.
    np.save(tmp_path / "meta.npy", rows["meta"][:, :old_width])
    loaded = Replay(50)
    loaded.load(tmp_path, replay.state())
    assert len(loaded) == 20
    np.testing.assert_array_equal(loaded.arrays["meta"][:20, :old_width], rows["meta"][:, :old_width])
    assert (loaded.arrays["meta"][:20, _core.M_OPPONENT_SINCE_BINGO] == -1).all(), "no bingo known"
    assert (loaded.arrays["meta"][:20, _core.M_BELIEF_INFERRED:] == 0).all(), "nothing inferred"
    np.testing.assert_array_equal(loaded.arrays["board"][:20], rows["board"])
    assert _core.META == 13 + TYPES and replay.bytes_per_position == 225 + 27 + 27 + 2 * _core.META + 12


# ---- the opponent's last move, and the extra scalars -----------------------------------------------


def test_the_plane_marks_exactly_the_tiles_the_opponent_just_placed(engine, leaves):
    game = _core.Game(engine, leaves, 11)
    before = game.candidates(8, 2)
    assert not (before[1] & _core.CODE_OPPONENT).any(), "nobody has moved yet"
    move = game.play_candidate(0)
    placed = {row * N + col for row, col, _letter, _blank in move["tiles"]}
    assert placed, "the first move of this seed is a placement"
    _, board, leave, unseen, meta, leave_value = game.candidates(8, 2)
    for codes in board:
        assert set(np.flatnonzero(codes & _core.CODE_OPPONENT)) == placed
    (planes, _), _ = build(engine, FeatureSet(last_move_plane=True), (board, leave, unseen, meta, leave_value))
    marked = planes[:, 87].reshape(len(board), SQUARES)
    assert all(set(np.flatnonzero(row.numpy())) == placed for row in marked)
    # The mover's own new tiles are in plane 27, never in this one.
    own = planes[:, 27].reshape(len(board), SQUARES)
    assert not (own * marked).any()
    # After the reply, the first player's turn: only the reply is marked.
    reply = game.play_candidate(0)
    replied = {row * N + col for row, col, _letter, _blank in reply.get("tiles", [])}
    codes = game.candidates(8, 2)[1][0]
    assert set(np.flatnonzero(codes & _core.CODE_OPPONENT)) == replied


def test_extra_scalars(engine, leaves):
    rows, _, _ = decisions(engine, leaves, steps=9)
    board, leave, unseen, meta, _ = rows
    meta = meta.copy()
    meta[0, _core.M_OPPONENT_SINCE_BINGO] = -1
    meta[1, _core.M_OPPONENT_SINCE_BINGO] = 0
    meta[2, _core.M_OPPONENT_SINCE_BINGO] = 3
    meta[3, _core.M_OPPONENT_SINCE_BINGO] = 20
    (_, vector), _ = build(engine, FeatureSet(extra_scalars=True), (board, leave, unseen, meta, rows[4]))
    extra = vector[:, 72:].numpy()
    assert extra.shape[1] == 13
    np.testing.assert_allclose(extra[:, 0], np.minimum(meta[:, _core.M_SCORE] / 50.0, 3.0), rtol=1e-6)
    np.testing.assert_allclose(extra[:4, 1:3], [[0, 1], [1, 0], [1, 3 / 8], [1, 1]], rtol=1e-6)
    pool = unseen.sum(axis=1).astype(np.float64)
    for column, tile in enumerate("?SJQXZ"):
        index = 26 if tile == "?" else ord(tile) - ord("A")
        np.testing.assert_allclose(extra[:, 3 + column], unseen[:, index] / pool, rtol=1e-5)
    vowels = [ord(c) - ord("A") for c in "AEIOU"]
    np.testing.assert_allclose(extra[:, 9], unseen[:, vowels].sum(axis=1) / pool, rtol=1e-5)
    # Vowels, consonants and blanks are the whole pool.
    np.testing.assert_allclose(extra[:, 9] + extra[:, 10] + extra[:, 3], 1.0, rtol=1e-5)
    kept = leave.sum(axis=1)
    has = kept > 0
    np.testing.assert_allclose((extra[:, 11] + extra[:, 12])[has], 1.0 - leave[has, 26] / kept[has], rtol=1e-5)
    assert (extra[~has, 11:] == 0).all(), "an empty leave has no vowel balance"


# ---- openness -------------------------------------------------------------------------------------


def test_openness_planes_follow_the_engine_bytes(engine, leaves):
    rows, _, _ = decisions(engine, leaves, steps=8)
    inputs = FeatureSet(openness=True)
    cross, opened = engine_inputs(engine, inputs, rows[0])
    assert opened.shape == (len(rows[0]), SQUARES) and opened.dtype == np.uint8
    (planes, vector), _ = build(engine, inputs, rows)
    planes = planes[:, 87:].reshape(len(opened), OPENNESS_PLANES, SQUARES).numpy()
    for half, shift in ((0, 0), (1, 4)):
        fewest = ((opened >> shift) & _core.OPEN_TILES).astype(np.float32)
        expected = np.where(fewest > 0, (8 - fewest) / 7, 0)
        np.testing.assert_allclose(planes[:, half], expected, rtol=1e-6)
        np.testing.assert_array_equal(planes[:, 2 + half], ((opened >> shift) & _core.OPEN_SEVEN) != 0)
    reach = planes[:, :2].max(axis=1)
    triple = np.array(_core.WORD_MULT) == 3
    double = np.array(_core.WORD_MULT) == 2
    np.testing.assert_allclose(planes[:, 4], reach * triple, rtol=1e-6)
    np.testing.assert_allclose(planes[:, 5], reach * double, rtol=1e-6)
    occupied = (rows[0] & _core.CODE_LETTER) != 0
    assert not planes[:, :, :][np.broadcast_to(occupied[:, None, :], planes.shape)].any(), "occupied squares are never open"
    assert (reach > 0).any() and (planes[:, 2:4] > 0).any()

    scalars = vector[:, 72:].numpy()
    in_reach = reach > 0
    np.testing.assert_allclose(scalars[:, 0], (in_reach & triple).sum(axis=1) / 8, rtol=1e-6)
    np.testing.assert_allclose(scalars[:, 1], (in_reach & double).sum(axis=1) / 17, rtol=1e-6)
    np.testing.assert_allclose(scalars[:, 2], in_reach.sum(axis=1) / (~occupied).sum(axis=1), rtol=1e-6)
    np.testing.assert_allclose(scalars[:, 3], planes[:, 2:4].sum(axis=(1, 2)) / 450, rtol=1e-6)

    with pytest.raises(ValueError, match="openness"):
        FeatureBuilder(CPU, inputs)(to_device(*rows, cross, CPU))


def test_openness_of_a_position_worked_out_by_hand(engine):
    # CAT on H8-J8. With the test word list: CATE? no. CATS? no ("CAT" is the only CAT word), so
    # nothing extends it to the right as a down play; ACT, ARC, etc. cross it.
    text = "\n".join(["." * 15] * 7 + [".......CAT....."] + ["." * 15] * 7)
    board = _core.board_codes_from_text(text)[None, :]
    cross, opened = engine_inputs(engine, FeatureSet(openness=True), board)
    across = lambda name: int(opened[0, square(name)] & 0x0F)  # noqa: E731
    down = lambda name: int(opened[0, square(name)] >> 4)  # noqa: E731
    assert (across("H8"), down("H8")) == (0, 0), "occupied"
    assert (across("A1"), down("A1")) == (0, 0), "out of reach"
    # K8, right of CAT: a row play reaches it with one tile (which words exist is not asked).
    assert across("K8") & _core.OPEN_TILES == 1
    # A column play through K8 would have to make CAT? with its tile: no letter does.
    assert (int(cross[0, SQUARES + square("K8")]) & 0x3FFFFFF) == 0 and down("K8") == 0
    # Under the A (I9): AD, AE, AG... fit, so one tile.
    assert across("I9") & _core.OPEN_TILES == 1 and down("I9") & _core.OPEN_TILES == 1
    # Seven squares to the right of CAT in row 8 are empty to the edge: only five (K8-O8), so no
    # seven-tile play lies wholly to the right, but one can run through CAT from the left.
    assert across("A8") & _core.OPEN_TILES == 7 and across("A8") & _core.OPEN_SEVEN
    assert across("O8") & _core.OPEN_TILES == 5


# ---- hooks ----------------------------------------------------------------------------------------


def test_hook_planes(engine):
    # EAT on H8-J8 (row word) and, down from J8, TEN (column word J8-J10).
    lines = ["." * 15] * 7 + [".......EAT.....", ".........E.....", ".........N....."] + ["." * 15] * 5
    board = _core.board_codes_from_text("\n".join(lines))[None, :]
    inputs = FeatureSet(hooks=True)
    cross, _ = engine_inputs(engine, inputs, board)
    zeros = (np.zeros((1, TYPES), np.uint8), np.zeros((1, TYPES), np.uint8), np.zeros((1, _core.META), np.int16), np.zeros(1, np.float32))
    planes, _ = FeatureBuilder(CPU, inputs)(to_device(board, *zeros, cross, CPU))
    after_row, before_row, below_column, above_column, s_after_row, s_below_column = planes[0, 87:].reshape(HOOK_PLANES, SQUARES).numpy()

    def letters(mask: int) -> int:
        return bin(mask & 0x3FFFFFF).count("1")

    column_masks, row_masks = cross[0, SQUARES:], cross[0, :SQUARES]
    # G8: the square before EAT. Letters that make ?EAT: B, F, H, M, N, P, S (BEAT, FEAT, HEAT, MEAT, NEAT, PEAT? ...).
    g8, k8 = square("G8"), square("K8")
    assert letters(int(column_masks[g8])) >= 5
    assert before_row[g8] == pytest.approx(letters(int(column_masks[g8])) / 26)
    assert after_row[g8] == 0
    # K8: after EAT. EATS is not in the test list; whatever fits is what the plane says.
    assert after_row[k8] == pytest.approx(letters(int(column_masks[k8])) / 26)
    assert s_after_row[k8] == float(bool(int(column_masks[k8]) >> 18 & 1))
    # J11: below TEN. TENS? not in the list; TEND and TENT are not either... the plane follows the mask.
    j11, j7 = square("J11"), square("J7")
    assert below_column[j11] == pytest.approx(letters(int(row_masks[j11])) / 26)
    assert s_below_column[j11] == float(bool(int(row_masks[j11]) >> 18 & 1))
    # J7: above the T of both words. ?TEN: no word, so the square is a hook square with nothing fitting.
    assert above_column[j7] == pytest.approx(letters(int(row_masks[j7])) / 26)
    # Squares that touch nothing in line are not hook squares, and neither are occupied ones.
    for plane in (after_row, before_row, below_column, above_column, s_after_row, s_below_column):
        assert plane[square("A1")] == 0 and plane[square("H8")] == 0
    assert before_row[square("H9")] == 0, "H9 is under the E: a hook for column words, not row words"
    # Every marked square is empty and beside a tile in the right direction.
    occupied = (board[0] & _core.CODE_LETTER) != 0
    for plane, offset in ((after_row, -1), (before_row, 1), (below_column, -N), (above_column, N)):
        for index in np.flatnonzero(plane):
            assert not occupied[index] and occupied[index + offset]


def test_an_s_hook_is_found_with_a_word_that_takes_one(engine):
    # SEA takes an S in front? No. EAT -> SEAT is in the list: S in front of EAT.
    lines = ["." * 15] * 7 + [".......EAT....."] + ["." * 15] * 7
    board = _core.board_codes_from_text("\n".join(lines))[None, :]
    cross, _ = engine_inputs(engine, FeatureSet(hooks=True), board)
    assert int(cross[0, SQUARES + square("G8")]) >> 18 & 1, "S makes SEAT"
    # ONE -> ONES is in the list: S after ONE.
    lines = ["." * 15] * 7 + [".......ONE....."] + ["." * 15] * 7
    board = _core.board_codes_from_text("\n".join(lines))[None, :]
    inputs = FeatureSet(hooks=True)
    cross, _ = engine_inputs(engine, inputs, board)
    zeros = (np.zeros((1, TYPES), np.uint8), np.zeros((1, TYPES), np.uint8), np.zeros((1, _core.META), np.int16), np.zeros(1, np.float32))
    planes, _ = FeatureBuilder(CPU, inputs)(to_device(board, *zeros, cross, CPU))
    s_after_row = planes[0, 87 + 4].reshape(SQUARES)
    assert s_after_row[square("K8")] == 1 and s_after_row.sum() == 1


# ---- the inference about the opponent's rack -------------------------------------------------------


def test_the_environment_infers_racks_only_when_asked(engine, leaves):
    plain, env, _ = decisions(engine, leaves, steps=6, belief_samples=0)
    assert env.belief_samples == 0
    assert (plain[3][:, _core.M_BELIEF_INFERRED:] == 0).all()
    rows, env, buffers = decisions(engine, leaves, steps=6, belief_samples=5)
    assert env.belief_samples == 5
    meta = rows[3]
    inferred = meta[:, _core.M_BELIEF_INFERRED] == 1
    assert inferred.any(), "after six moves most opponents have just placed tiles"
    counts = meta[:, _core.M_BELIEF:].astype(np.float64) / _core.BELIEF_SCALE
    np.testing.assert_allclose(counts[inferred].sum(axis=1), meta[inferred, _core.M_OPPONENT_TILES], atol=0.06)
    assert (counts[inferred] >= 0).all() and (counts[inferred] <= rows[2][inferred] + 1e-9).all(), "never more of a tile than is unseen"
    assert (counts[~inferred] == 0).all()
    # The inference observes the games: the same moves are offered either way.
    np.testing.assert_array_equal(rows[0], plain[0])
    np.testing.assert_array_equal(meta[:, :_core.M_OPPONENT_SINCE_BINGO + 1], plain[3][:, :_core.M_OPPONENT_SINCE_BINGO + 1])


def test_the_single_game_binding_infers_too(engine, leaves):
    plain = _core.Game(engine, leaves, 21)
    watched = _core.Game(engine, leaves, 21, 8)
    for game in (plain, watched):
        game.candidates(8, 2)
        game.play_candidate(0)
    meta_plain, meta_watched = plain.candidates(8, 2)[4], watched.candidates(8, 2)[4]
    assert (meta_plain[:, _core.M_BELIEF_INFERRED] == 0).all()
    assert (meta_watched[:, _core.M_BELIEF_INFERRED] == 1).all()
    assert (meta_watched[:, _core.M_BELIEF:] == meta_watched[0, _core.M_BELIEF:]).all(), "one inference per decision"
    np.testing.assert_array_equal(plain.candidates(8, 2)[1], watched.candidates(8, 2)[1])


def test_belief_inputs(engine, leaves):
    rows, _, _ = decisions(engine, leaves, steps=6, belief_samples=5)
    board, leave, unseen, meta, leave_value = rows
    (_, vector), _ = build(engine, FeatureSet(belief=True), rows)
    belief, shift, flag = vector[:, 72:99].numpy() * 2, vector[:, 99:126].numpy(), vector[:, 126].numpy()
    inferred = meta[:, _core.M_BELIEF_INFERRED] == 1
    np.testing.assert_array_equal(flag, inferred)
    blind = unseen * (meta[:, _core.M_OPPONENT_TILES] / unseen.sum(axis=1))[:, None]
    np.testing.assert_allclose(belief[~inferred], blind[~inferred], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(shift[~inferred], 0, atol=1e-6)
    np.testing.assert_allclose(belief[inferred], meta[inferred, _core.M_BELIEF:] / _core.BELIEF_SCALE, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(shift, belief - blind, atol=1e-5)
    assert np.abs(shift[inferred]).max() > 0.01, "an inference moves the expectation"
    # Either way the expected tiles add up to the opponent's rack.
    np.testing.assert_allclose(belief.sum(axis=1), meta[:, _core.M_OPPONENT_TILES], atol=0.06)


# ---- the transpose augmentation with every input ---------------------------------------------------


def test_transposing_the_tensors_equals_building_from_the_transposed_board(engine, leaves):
    rows, _, _ = decisions(engine, leaves, steps=8)
    board = rows[0]
    (planes, vector), batch = build(engine, EVERYTHING, rows)
    builder = FeatureBuilder(CPU, EVERYTHING)
    flipped_planes, flipped_vector = builder(builder.transpose(batch, torch.ones(len(board), dtype=torch.bool)))
    # The other way: transpose the codes, and let the engine work everything out afresh.
    turned = board.reshape(-1, N, N).transpose(0, 2, 1).reshape(-1, SQUARES).copy()
    (expected_planes, expected_vector), _ = build(engine, EVERYTHING, (turned, *rows[1:]))
    assert torch.equal(flipped_planes, expected_planes)
    assert torch.equal(flipped_vector, expected_vector) and torch.equal(flipped_vector, vector)
    assert not torch.equal(flipped_planes, planes)
    # Rows left alone are untouched.
    same_planes, _ = builder(builder.transpose(batch, torch.zeros(len(board), dtype=torch.bool)))
    assert torch.equal(same_planes, planes)


# ---- end to end -----------------------------------------------------------------------------------


def test_a_network_with_every_input_scores_candidates(engine, leaves, monkeypatch):
    monkeypatch.setattr(Scorer, "PAD", 32)
    torch.manual_seed(0)
    net = ValueNet(ModelConfig(blocks=1, channels=16, vector_hidden=16, head_hidden=16, value_channels=2))
    torch.nn.init.normal_(net.value.weight, std=0.5)
    scorer = Scorer(net, engine, 50.0, CPU)
    assert scorer.belief_samples == 4 and scorer.inputs == EVERYTHING
    rows, _, _ = decisions(engine, leaves, steps=5, belief_samples=scorer.belief_samples)
    value, win = scorer(*rows)
    assert value.shape == win.shape == (len(rows[0]),)
    assert torch.isfinite(value).all() and ((win > 0) & (win < 1)).all()
    assert value.std() > 0, "the network's output depends on its inputs"
    # A network from before the new inputs scores the same candidates and ignores what it never had.
    old = Scorer(ValueNet(ModelConfig(blocks=1, channels=16, vector_hidden=16, head_hidden=16, value_channels=2,
                                      **BEFORE_IT_EXISTED["model"])), engine, 50.0, CPU)
    assert old.belief_samples == 0 and old.inputs == ORIGINAL
    plain, _, _ = decisions(engine, leaves, steps=5, belief_samples=0)
    np.testing.assert_allclose(old(*rows)[0].numpy(), old(*plain)[0].numpy(), rtol=1e-6)


def test_half_precision_planes_match_full_precision(engine, leaves):
    """On the GPU the planes are built as bfloat16. Every plane value here is exactly
    representable in it except the graded ones, which must agree to its precision."""
    rows, _, _ = decisions(engine, leaves, steps=8)
    (full, vector), _ = build(engine, EVERYTHING, rows)
    (half, half_vector), _ = build(engine, EVERYTHING, rows, dtype=torch.bfloat16)
    assert half.dtype == torch.bfloat16 and full.dtype == torch.float32
    assert half.shape == full.shape and half.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(half[:, :88].float(), full[:, :88]), "yes/no planes are exact"
    assert (half[:, 88:].float() - full[:, 88:]).abs().max() < 4e-3
    assert torch.equal(half_vector, vector) and half_vector.dtype == torch.float32
