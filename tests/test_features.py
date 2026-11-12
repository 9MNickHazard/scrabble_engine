"""`features.FeatureBuilder`: plane and vector layout, and the transpose augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scrabble_ai import _core
from scrabble_ai import features
from scrabble_ai.features import FeatureBuilder, to_device
from scrabble_ai.replay import Replay
from scrabble_ai.selfplay import Actor, ActorStats

from conftest import cuda_only

N, SQUARES, TYPES = 15, 225, 27

# The premium layout copied from docs/RULES.md: '=' TW, '-' DW, '"' TL, "'" DL.
RULES_LAYOUT = """
= . . ' . . . = . . . ' . . =
. - . . . " . . . " . . . - .
. . - . . . ' . ' . . . - . .
' . . - . . . ' . . . - . . '
. . . . - . . . . . - . . . .
. " . . . " . . . " . . . " .
. . ' . . . ' . ' . . . ' . .
= . . ' . . . - . . . ' . . =
. . ' . . . ' . ' . . . ' . .
. " . . . " . . . " . . . " .
. . . . - . . . . . - . . . .
' . . - . . . ' . . . - . . '
. . - . . . ' . ' . . . - . .
. - . . . " . . . " . . . - .
= . . ' . . . = . . . ' . . =
"""


def rules_premium() -> np.ndarray:
    """[4, 225] booleans: DL, TL, DW, TW, from the RULES.md diagram."""
    cells = [line.split() for line in RULES_LAYOUT.strip().splitlines()]
    assert len(cells) == N and all(len(row) == N for row in cells)
    flat = np.array([c for row in cells for c in row])
    return np.stack([flat == "'", flat == '"', flat == "-", flat == "="])


def square(name: str) -> int:
    col = ord(name[0]) - ord("A")
    row = int(name[1:]) - 1
    return row * N + col


def letter_bit(c: str) -> int:
    return 1 << (ord(c) - ord("A"))


def bits(letters: str) -> int:
    out = 0
    for c in letters:
        out |= letter_bit(c)
    return out


BOARD_TEXT = "\n".join(
    [
        "..............Z",  # Z on O1, a triple-word square
        *["..............."] * 6,
        ".......CAt.....",  # C on H8 (the centre, double word), A, then a blank T on J8
        "........X......",  # X on I9, a double-letter square
        *["..............."] * 6,
    ]
)


@pytest.fixture(scope="module")
def hand_board() -> np.ndarray:
    codes = _core.board_codes_from_text(BOARD_TEXT).copy()
    # Mark two tiles as placed by the move that produced the position.
    codes[square("J8")] |= _core.CODE_PLACED
    codes[square("I9")] |= _core.CODE_PLACED
    return codes


def one_batch(board: np.ndarray, engine, meta: np.ndarray | None = None, cross: bool = True):
    board = np.ascontiguousarray(board.reshape(-1, SQUARES))
    b = len(board)
    masks = None
    if cross:
        masks = np.empty((b, _core.CROSS_WORDS), np.uint32)
        engine.cross_masks(board, masks)
    meta = np.zeros((b, _core.META), np.int16) if meta is None else meta
    batch = to_device(board, np.zeros((b, TYPES), np.uint8), np.zeros((b, TYPES), np.uint8), meta,
                      np.zeros(b, np.float32), masks, torch.device("cpu"))
    return batch, masks


def test_layout_constants():
    assert features.plane_count(True) == 87 and features.plane_count(False) == 33
    assert features.VECTOR == 2 * TYPES + len(features.SCALARS)
    assert len(features.SCALARS) == 18
    premium = rules_premium()
    np.testing.assert_array_equal(np.array(_core.LETTER_MULT) == 2, premium[0])
    np.testing.assert_array_equal(np.array(_core.LETTER_MULT) == 3, premium[1])
    np.testing.assert_array_equal(np.array(_core.WORD_MULT) == 2, premium[2])
    np.testing.assert_array_equal(np.array(_core.WORD_MULT) == 3, premium[3])
    assert premium.sum(axis=1).tolist() == [24, 12, 17, 8]


def test_board_codes_from_text(hand_board):
    codes = _core.board_codes_from_text(BOARD_TEXT)
    assert codes.dtype == np.uint8 and codes.shape == (SQUARES,)
    assert codes[square("H8")] == 3  # C = letter 2, code letter + 1
    assert codes[square("I8")] == 1  # A
    assert codes[square("J8")] == 20 | _core.CODE_BLANK  # a blank showing T
    assert codes[square("O1")] == 26  # Z
    assert (codes != 0).sum() == 5


def test_board_planes_on_a_hand_made_board(engine, hand_board):
    batch, masks = one_batch(hand_board, engine)
    planes, vector = FeatureBuilder(torch.device("cpu"))(batch)
    assert planes.shape == (1, 87, N, N) and planes.dtype == torch.float32
    assert vector.shape == (1, features.VECTOR)
    p = planes[0].reshape(87, SQUARES).numpy()
    assert set(np.unique(p)) <= {0.0, 1.0}

    occupied = {"O1": "Z", "H8": "C", "I8": "A", "J8": "T", "I9": "X"}
    for plane in range(26):
        expected = np.zeros(SQUARES)
        for name, letter in occupied.items():
            if ord(letter) - ord("A") == plane:
                expected[square(name)] = 1
        np.testing.assert_array_equal(p[plane], expected, err_msg=f"letter plane {plane}")
    assert np.flatnonzero(p[26]).tolist() == [square("J8")]  # blank
    assert sorted(np.flatnonzero(p[27]).tolist()) == sorted([square("J8"), square("I9")])  # just placed
    empty = np.ones(SQUARES)
    empty[[square(s) for s in occupied]] = 0
    np.testing.assert_array_equal(p[28], empty)

    # Premium planes: the RULES layout, on empty squares only.
    premium = rules_premium() & (empty == 1)
    np.testing.assert_array_equal(p[29:33], premium.astype(np.float32))
    assert p[31, square("H8")] == 0, "H8 is a double word but occupied"
    assert p[32, square("A1")] == 1, "A1 is an empty triple word"
    assert p[32, square("O1")] == 0, "O1 is a triple word under the Z"
    assert p[29, square("D1")] == 1 and p[29, square("I9")] == 0, "D1 empty double letter; I9 is one under the X"
    assert p[30, square("F2")] == 1 and p[30, square("B6")] == 1, "triple letters"
    assert p[29:33].sum(axis=1).tolist() == [23, 12, 16, 7]

    # Cross-check planes are exactly the bits of the engine's masks.
    row_masks, column_masks = masks[0, :SQUARES], masks[0, SQUARES:]
    for bit in range(27):
        np.testing.assert_array_equal(p[33 + bit], (row_masks >> bit) & 1, err_msg=f"row bit {bit}")
        np.testing.assert_array_equal(p[60 + bit], (column_masks >> bit) & 1, err_msg=f"column bit {bit}")

    def row_letters(name):
        return "".join(chr(ord("A") + i) for i in range(26) if p[33 + i, square(name)])

    def column_letters(name):
        return "".join(chr(ord("A") + i) for i in range(26) if p[60 + i, square(name)])

    # By hand, from the test lexicon. Row plays (tiles above/below):
    assert row_letters("I7") == "FLMSTW" and p[59, square("I7")] == 1  # ?AX: FAX LAX MAX SAX TAX WAX
    assert row_letters("I10") == "E" and p[59, square("I10")] == 1  # AX?: AXE
    assert row_letters("J7") == "AIU"  # ?T: AT IT UT (the blank counts as its letter)
    assert row_letters("J9") == "AEIO"  # T?: TA TE TI TO
    assert row_letters("H7") == "" and p[59, square("H7")] == 1  # ?C: nothing
    assert row_letters("O2") == "A"  # Z?: ZA
    assert row_letters("C13") == "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and p[59, square("C13")] == 0
    # Column plays (tiles left/right):
    assert column_letters("H9") == "AEO" and p[86, square("H9")] == 1  # ?X: AX EX OX
    assert column_letters("J9") == "IU"  # X?: XI XU
    assert column_letters("G8") == "" and p[86, square("G8")] == 1  # ?CAT
    assert column_letters("N1") == ""  # ?Z
    assert column_letters("C13") == "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and p[86, square("C13")] == 0
    # Occupied squares carry no cross-check information.
    for name in occupied:
        assert p[33:87, square(name)].sum() == 0


def test_without_cross_checks(engine, hand_board):
    batch, _ = one_batch(hand_board, engine, cross=False)
    planes, _ = FeatureBuilder(torch.device("cpu"), cross_checks=False)(batch)
    assert planes.shape == (1, 33, N, N)
    full_batch, _ = one_batch(hand_board, engine)
    full, _ = FeatureBuilder(torch.device("cpu"))(full_batch)
    assert torch.equal(planes, full[:, :33])
    with pytest.raises(ValueError, match="cross"):
        FeatureBuilder(torch.device("cpu"), cross_checks=True)(batch)


def test_vector_layout():
    m = {name: getattr(_core, name) for name in dir(_core) if name.startswith("M_")}
    meta = np.zeros((4, _core.META), np.int16)

    def put(row, **values):
        for key, value in values.items():
            meta[row, m["M_" + key.upper()]] = value

    put(0, score=17, spread=250, bag=40, draws=3, scoreless=0, opponent_tiles=7, kind=0, terminal=0,
        opponent_last_kind=1, opponent_last_tiles=4, opponent_last_score=33)
    put(1, spread=-500, bag=0, draws=0, scoreless=2, opponent_tiles=3, kind=1,
        opponent_last_kind=2, opponent_last_tiles=5, opponent_last_score=0)
    put(2, spread=401, bag=6, draws=2, scoreless=5, opponent_tiles=7, kind=2, opponent_last_kind=3)
    put(3, spread=0, bag=7, kind=0, opponent_last_kind=0, opponent_last_score=200)

    leave = np.zeros((4, TYPES), np.uint8)
    leave[0, 0], leave[0, 4], leave[0, 26] = 2, 1, 1  # A A E ?
    leave[1, 18] = 3  # S S S
    unseen = np.zeros((4, TYPES), np.uint8)
    unseen[:] = np.array(_core.TILE_COUNTS, np.uint8)
    unseen[0, 0] = 3  # 3 of the 9 A's unseen
    unseen[1] = 0
    unseen[1, 16] = 1  # only the Q
    leave_value = np.array([6.5, -3.0, 0.0, 40.0], np.float32)
    board = np.zeros((4, SQUARES), np.uint8)
    batch = to_device(board, leave, unseen, meta, leave_value, None, torch.device("cpu"))
    _, vector = FeatureBuilder(torch.device("cpu"), cross_checks=False)(batch)
    v = vector.numpy()
    assert v.shape == (4, features.VECTOR) and vector.dtype == torch.float32

    # Leave counts / 2.
    assert v[0, :TYPES].tolist() == [1.0, 0, 0, 0, 0.5] + [0] * 21 + [0.5]
    assert v[1, 18] == 1.5
    # Unseen / that tile's total.
    totals = np.array(_core.TILE_COUNTS, np.float32)
    np.testing.assert_allclose(v[0, TYPES:2 * TYPES], np.where(np.arange(TYPES) == 0, 3 / 9, 1.0), rtol=1e-6)
    assert v[1, TYPES + 16] == 1.0 and v[1, TYPES:2 * TYPES].sum() == 1.0
    assert totals.sum() == 100

    s = v[:, 2 * TYPES:]
    assert s.shape[1] == len(features.SCALARS)
    names = list(features.SCALARS)

    def col(prefix):
        matches = [i for i, name in enumerate(names) if name.startswith(prefix)]
        assert len(matches) == 1, prefix
        return s[:, matches[0]]

    np.testing.assert_allclose(col("bag after draw"), [40 / 86, 0, 6 / 86, 7 / 86], rtol=1e-6)
    np.testing.assert_allclose(col("tiles drawn"), [3 / 7, 0, 2 / 7, 0], rtol=1e-6)
    np.testing.assert_allclose(col("leave size"), [4 / 7, 3 / 7, 0, 0], rtol=1e-6)
    np.testing.assert_allclose(col("spread after"), [2.5, -4.0, 4.0, 0.0], rtol=1e-6)  # clipped at +-4
    np.testing.assert_allclose(col("scoreless"), [0, 2 / 6, 5 / 6, 0], rtol=1e-6)
    np.testing.assert_allclose(col("opponent's tiles"), [1.0, 3 / 7, 1.0, 0], rtol=1e-6)
    assert col("bag is empty").tolist() == [0, 1, 0, 0]
    assert col("bag below 7").tolist() == [0, 1, 1, 0]
    assert col("move was a placement").tolist() == [1, 0, 0, 1]
    assert col("move was an exchange").tolist() == [0, 1, 0, 0]
    assert col("move was a pass").tolist() == [0, 0, 1, 0]
    np.testing.assert_allclose(col("unseen tiles"), [94 / 93, 1 / 93, 100 / 93, 100 / 93], rtol=1e-6)
    np.testing.assert_allclose(col("static leave value"), [6.5 / 20, -3 / 20, 0, 2.0], rtol=1e-6)
    # The opponent's previous turn (new fields).
    assert col("opponent's previous turn was a placement").tolist() == [1, 0, 0, 0]
    assert col("opponent's previous turn was an exchange").tolist() == [0, 1, 0, 0]
    assert col("opponent's previous turn was a pass").tolist() == [0, 0, 1, 0]
    np.testing.assert_allclose(col("tiles the opponent placed"), [4 / 7, 5 / 7, 0, 0], rtol=1e-6)
    np.testing.assert_allclose(col("points the opponent scored"), [33 / 50, 0, 0, 3.0], rtol=1e-6)  # 200/50 clipped
    # The scalars are exactly SCALARS in order, nothing else.
    expected_order = [
        [40 / 86, 3 / 7, 4 / 7, 2.5, 0, 1, 0, 0, 1, 0, 0, 94 / 93, 6.5 / 20, 1, 0, 0, 4 / 7, 33 / 50],
    ]
    np.testing.assert_allclose(s[:1], expected_order, rtol=1e-6)


def test_bfloat16_planes(engine, hand_board):
    batch, _ = one_batch(hand_board, engine)
    full, vector_full = FeatureBuilder(torch.device("cpu"))(batch)
    half, vector_half = FeatureBuilder(torch.device("cpu"), dtype=torch.bfloat16)(batch)
    assert half.dtype == torch.bfloat16 and vector_half.dtype == torch.float32
    assert torch.equal(half.float(), full) and torch.equal(vector_half, vector_full)


# ---- transpose augmentation -------------------------------------------------------------------


@pytest.fixture(scope="module")
def game_positions(engine, leaves):
    """A few hundred afterstates from real (sampled) games."""
    replay = Replay(20_000)
    actor = Actor(engine, leaves, replay, 6, 12, 3, 1.0, torch.device("cpu"), first_seed=77, sampling_seed=5)
    stats = ActorStats()
    while stats.games < 8:
        stats.merge(actor.step(None, 6.0))
    rng = np.random.default_rng(0)
    index = np.sort(rng.choice(len(replay), size=min(300, len(replay)), replace=False))
    return {name: np.ascontiguousarray(array[index]) for name, array in replay.arrays.items()}


def transposed_codes(board: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(board.reshape(-1, N, N).transpose(0, 2, 1).reshape(-1, SQUARES))


def test_transpose_matches_recomputing_masks_on_the_transposed_board(engine, game_positions):
    rows = game_positions
    b = len(rows["board"])
    assert b >= 100 and (rows["board"] != 0).sum(axis=1).max() > 20
    masks = np.empty((b, _core.CROSS_WORDS), np.uint32)
    engine.cross_masks(rows["board"], masks)
    batch = to_device(rows["board"], rows["leave"], rows["unseen"], rows["meta"], rows["leave_value"], masks,
                      torch.device("cpu"))
    flip_np = np.random.default_rng(1).random(b) < 0.5
    builder = FeatureBuilder(torch.device("cpu"))
    out = builder.transpose(batch, torch.from_numpy(flip_np))

    expected_board = np.where(flip_np[:, None], transposed_codes(rows["board"]), rows["board"])
    expected_masks = np.empty_like(masks)
    engine.cross_masks(np.ascontiguousarray(expected_board), expected_masks)
    np.testing.assert_array_equal(out.board.numpy(), expected_board)
    np.testing.assert_array_equal(out.cross.numpy().view(np.uint32), expected_masks)
    # The two mask sets really differ on these boards, so the swap is being exercised.
    assert (masks[:, :SQUARES] != masks[:, SQUARES:]).any(axis=1).mean() > 0.9
    # Unflipped rows are untouched; the non-board fields are never changed.
    np.testing.assert_array_equal(out.board.numpy()[~flip_np], rows["board"][~flip_np])
    np.testing.assert_array_equal(out.cross.numpy().view(np.uint32)[~flip_np], masks[~flip_np])
    for name in ("leave", "unseen", "meta", "leave_value"):
        assert torch.equal(getattr(out, name), getattr(batch, name))

    # Twice is the identity.
    again = builder.transpose(out, torch.from_numpy(flip_np))
    assert torch.equal(again.board, batch.board) and torch.equal(again.cross, batch.cross)

    # The expanded planes of a flipped row are the original planes transposed, with the row-play
    # and column-play cross-check planes trading places.
    planes, vector = builder(batch)
    planes_t, vector_t = builder(out)
    assert torch.equal(vector, vector_t)
    swapped = torch.cat([planes[:, :33], planes[:, 60:87], planes[:, 33:60]], dim=1).transpose(2, 3)
    flip = torch.from_numpy(flip_np)
    assert torch.equal(planes_t[flip], swapped[flip])
    assert torch.equal(planes_t[~flip], planes[~flip])


def test_transpose_without_cross_masks(engine, game_positions):
    rows = game_positions
    b = len(rows["board"])
    batch = to_device(rows["board"], rows["leave"], rows["unseen"], rows["meta"], rows["leave_value"], None,
                      torch.device("cpu"))
    flip = torch.ones(b, dtype=torch.bool)
    out = FeatureBuilder(torch.device("cpu"), cross_checks=False).transpose(batch, flip)
    assert out.cross is None
    np.testing.assert_array_equal(out.board.numpy(), transposed_codes(rows["board"]))


@cuda_only
def test_features_on_gpu_match_cpu(engine, game_positions):
    rows = game_positions
    b = 32
    masks = np.empty((b, _core.CROSS_WORDS), np.uint32)
    board = np.ascontiguousarray(rows["board"][:b])
    engine.cross_masks(board, masks)
    args = (board, rows["leave"][:b].copy(), rows["unseen"][:b].copy(), rows["meta"][:b].copy(),
            rows["leave_value"][:b].copy(), masks)
    cpu_planes, cpu_vector = FeatureBuilder(torch.device("cpu"))(to_device(*args, torch.device("cpu")))
    gpu = torch.device("cuda")
    builder = FeatureBuilder(gpu, dtype=torch.bfloat16)
    flip = torch.zeros(b, dtype=torch.bool, device=gpu)
    flip[::2] = True
    batch = builder.transpose(builder.transpose(to_device(*args, gpu), flip), flip)
    gpu_planes, gpu_vector = builder(batch)
    assert torch.equal(gpu_planes.float().cpu(), cpu_planes)
    torch.testing.assert_close(gpu_vector.cpu(), cpu_vector)
