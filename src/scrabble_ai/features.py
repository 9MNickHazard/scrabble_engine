"""Network inputs, built on the GPU from the engine's compact afterstate codes.

An afterstate arrives as small integer arrays (see `scrabble_core::encode`). This module turns a
batch of them into what the network consumes. Which inputs a network has is a `FeatureSet`, read
from its `ModelConfig`; the groups always come in this order, so a network without the later ones
(any checkpoint from before 2026-11-14) sees exactly the layout it was trained with.

Board planes, `[B, planes, 15, 15]`:
     0-25   one plane per letter: a tile showing that letter is on the square
       26   the tile is a blank
       27   the tile was placed by the move that produced this position
       28   the square is empty
    29-32   an empty double-letter / triple-letter / double-word / triple-word square
  cross_checks (54):
    for a play along the row: the letters the tiles above and below allow on this empty square
    (26), and whether there are such tiles, so the play would form a cross-word here (1); then
    the same for a play down the column (tiles to the left and right)
  last_move_plane (1):
    the tile was placed by the opponent's previous move
  openness (6), what the next player could do with the board as the move leaves it:
    how easily a placement along the row reaches this empty square: (8 - fewest tiles) / 7, or 0
    if none can; the same down the column; the square is in reach of a seven-tile placement
    along the row; the same down the column; an empty triple-word square, by how easily it is
    reached in either direction; the same for a double-word square
  hooks (6), the squares where one letter extends a word already on the board:
    after a row word's last letter, by the share of the alphabet that fits (letters / 26); before
    its first letter; below a column word; above it; then whether an S fits after a row word, and
    below a column word

Vector, `[B, vector]`:
     0-26   tiles kept on the rack, by type (count / 2)
    27-53   tiles the mover cannot see, by type (fraction of that type's total)
    54-71   scalars, see `SCALARS`
  extra_scalars (13): see `EXTRA_SCALARS`
  openness (4): see `OPENNESS_SCALARS`
  belief (55): the expected number of each tile type on the opponent's rack (count / 2), how that
    differs from a rack drawn blind from the unseen pool, and whether an inference was made

Everything here is information the mover has. The opponent's actual rack is never an input. What
the opponent did on their previous turn is public, and so is what can be inferred from it about
the tiles they kept (`scrabble_core::belief`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from scrabble_ai import _core

N = 15
SQUARES = _core.SQUARES
TILE_TYPES = _core.TILE_TYPES
META = _core.META

BASE_PLANES = 33
CROSS_PLANES = 54
LAST_MOVE_PLANES = 1
OPENNESS_PLANES = 6
HOOK_PLANES = 6
SCALARS = (
    "bag after draw / 86",
    "tiles drawn / 7",
    "leave size / 7",
    "spread after the move / 100 (clipped to +-4)",
    "scoreless turns in a row / 6",
    "opponent's tiles / 7",
    "bag is empty",
    "bag below 7 (no exchanges)",
    "move was a placement",
    "move was an exchange",
    "move was a pass",
    "unseen tiles / 93",
    "static leave value / 20",
    "opponent's previous turn was a placement",
    "opponent's previous turn was an exchange",
    "opponent's previous turn was a pass",
    "tiles the opponent placed or exchanged on it / 7",
    "points the opponent scored on it / 50 (clipped to 3)",
)
POWER_TILES = "?SJQXZ"
EXTRA_SCALARS = (
    "points the move scored / 50 (clipped to 3)",
    "the opponent has placed all seven tiles at least once",
    "their turns since they last did / 8 (clipped to 1; 1 if never)",
    *(f"share of the unseen tiles that are {tile}" for tile in POWER_TILES),
    "share of the unseen tiles that are vowels",
    "share of the unseen tiles that are consonants",
    "share of the kept tiles that are vowels",
    "share of the kept tiles that are consonants",
)
OPENNESS_SCALARS = (
    "triple-word squares in reach / 8",
    "double-word squares in reach / 17",
    "share of the empty squares in reach",
    "squares in reach of a seven-tile placement, both directions / 450",
)
BELIEF_VECTOR = 2 * TILE_TYPES + 1
VECTOR = 2 * TILE_TYPES + len(SCALARS)  # the vector of a network with none of the optional groups


@dataclass(frozen=True)
class FeatureSet:
    """Which input groups a network has."""

    cross_checks: bool = True
    extra_scalars: bool = False
    last_move_plane: bool = False
    openness: bool = False
    hooks: bool = False
    belief: bool = False

    @classmethod
    def of(cls, model) -> "FeatureSet":
        """The inputs of a network with this `ModelConfig`."""
        return cls(model.cross_checks, model.extra_scalars, model.last_move_plane, model.openness, model.hooks,
                   model.belief_samples > 0)

    @classmethod
    def everything(cls) -> "FeatureSet":
        return cls(True, True, True, True, True, True)

    @property
    def planes(self) -> int:
        return (BASE_PLANES + CROSS_PLANES * self.cross_checks + LAST_MOVE_PLANES * self.last_move_plane
                + OPENNESS_PLANES * self.openness + HOOK_PLANES * self.hooks)

    @property
    def vector(self) -> int:
        return (VECTOR + len(EXTRA_SCALARS) * self.extra_scalars + len(OPENNESS_SCALARS) * self.openness
                + BELIEF_VECTOR * self.belief)

    @property
    def needs_cross(self) -> bool:
        """Cross-check masks are an input themselves and what hooks and openness are worked out from."""
        return self.cross_checks or self.hooks or self.openness


def plane_count(features: FeatureSet | bool) -> int:
    """Board planes of a feature set (or, given a bool, of the original inputs with or without
    cross-check planes)."""
    return (FeatureSet(cross_checks=features) if isinstance(features, bool) else features).planes


@dataclass
class Afterstates:
    """A batch of afterstates as tensors on one device (integer codes, not yet expanded)."""

    board: torch.Tensor  # uint8 [B, 225]
    leave: torch.Tensor  # uint8 [B, 27]
    unseen: torch.Tensor  # uint8 [B, 27]
    meta: torch.Tensor  # int16 [B, META]
    leave_value: torch.Tensor  # float32 [B]
    cross: torch.Tensor | None  # int32 [B, 450], or None when no input needs the cross masks
    open: torch.Tensor | None = None  # uint8 [B, 225], or None without the openness inputs

    def __len__(self) -> int:
        return self.board.shape[0]


def engine_inputs(engine: _core.Engine, features: FeatureSet, board: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(cross masks, openness bytes) for a block of board codes: what `features` needs worked
    out by the engine from the boards alone. Neither is stored anywhere; both are recomputed for
    self-play candidates and for replayed positions alike."""
    cross = opened = None
    if features.needs_cross:
        cross = np.empty((len(board), _core.CROSS_WORDS), np.uint32)
        engine.cross_masks(board, cross)
    if features.openness:
        opened = np.empty((len(board), SQUARES), np.uint8)
        engine.open_squares(board, cross, opened)
    return cross, opened


def to_device(
    board: np.ndarray,
    leave: np.ndarray,
    unseen: np.ndarray,
    meta: np.ndarray,
    leave_value: np.ndarray,
    cross: np.ndarray | None,
    device: torch.device,
    opened: np.ndarray | None = None,
) -> Afterstates:
    def move(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device, non_blocking=True)

    return Afterstates(
        board=move(board),
        leave=move(leave),
        unseen=move(unseen),
        meta=move(meta),
        leave_value=move(leave_value),
        # Masks use bits 0-26 only, so reading the uint32 words as int32 changes nothing.
        cross=None if cross is None else move(cross.view(np.int32)),
        open=None if opened is None else move(opened),
    )


class FeatureBuilder:
    """Expands `Afterstates` into (planes, vector). Holds the constant tables on the device."""

    def __init__(self, device: torch.device, features: FeatureSet | bool = True, dtype: torch.dtype = torch.float32,
                 *, cross_checks: bool | None = None):
        """`features`: the network's inputs. A bool, or the `cross_checks` keyword, means the
        original inputs with or without cross-check planes. `dtype` is the type of the board
        planes: bfloat16 halves their memory under autocast."""
        self.device = device
        if cross_checks is not None:
            features = cross_checks
        self.features = FeatureSet(cross_checks=features) if isinstance(features, bool) else features
        self.cross_checks = self.features.cross_checks
        self.dtype = dtype
        self.planes = self.features.planes
        self.vector = self.features.vector
        letter = torch.tensor(_core.LETTER_MULT, device=device)
        word = torch.tensor(_core.WORD_MULT, device=device)
        self.premium = torch.stack([letter == 2, letter == 3, word == 2, word == 3]).view(1, 4, SQUARES)
        self.tile_totals = torch.tensor(_core.TILE_COUNTS, device=device).float()
        self.letter_ids = torch.arange(1, 27, device=device, dtype=torch.uint8).view(1, 26, 1)
        self.cross_bits = (1 << torch.arange(27, device=device, dtype=torch.int32)).view(1, 1, 27, 1)
        tile = lambda c: 26 if c == "?" else ord(c) - ord("A")  # noqa: E731
        self.power = torch.tensor([tile(c) for c in POWER_TILES], device=device)
        vowels = torch.zeros(TILE_TYPES, device=device)
        vowels[[tile(c) for c in "AEIOU"]] = 1.0
        consonants = 1.0 - vowels
        consonants[26] = 0.0  # a blank is neither
        self.vowels, self.consonants = vowels, consonants

    def transpose(self, batch: Afterstates, flip: torch.Tensor) -> Afterstates:
        """Swap rows and columns in the rows of the batch where `flip` is true.

        Turning every across play into a down play is the one exact symmetry of the game: the
        premium layout is unchanged and words still read forwards. Mirror images are not
        symmetries (words would read backwards), so they are never used.
        """
        b = len(batch)
        board = batch.board.view(b, N, N)
        board = torch.where(flip.view(b, 1, 1), board.transpose(1, 2), board).reshape(b, SQUARES)
        cross = batch.cross
        if cross is not None:
            grid = cross.view(b, 2, N, N)
            # A row play on the transposed board is a column play on the original, so the two
            # sets of masks trade places as well as being transposed.
            swapped = grid.flip(1).transpose(2, 3)
            cross = torch.where(flip.view(b, 1, 1, 1), swapped, grid).reshape(b, 2 * SQUARES)
        opened = batch.open
        if opened is not None:
            # The same for openness: each byte holds the row half and the column half.
            grid = opened.view(b, N, N)
            swapped = ((grid << 4) | (grid >> 4)).transpose(1, 2)
            opened = torch.where(flip.view(b, 1, 1), swapped, grid).reshape(b, SQUARES)
        return Afterstates(board, batch.leave, batch.unseen, batch.meta, batch.leave_value, cross, opened)

    def __call__(self, batch: Afterstates) -> tuple[torch.Tensor, torch.Tensor]:
        b, features = len(batch), self.features
        # The yes/no planes are assembled as booleans (one byte per entry) and converted once.
        code = batch.board.unsqueeze(1)  # [B, 1, 225]
        letter = code & _core.CODE_LETTER  # 0 empty, 1..26
        empty = letter == 0
        flags = [
            letter == self.letter_ids,  # [B, 26, 225]
            (code & _core.CODE_BLANK) != 0,
            (code & _core.CODE_PLACED) != 0,
            empty,
            self.premium & empty,  # [B, 4, 225]
        ]
        bits = None
        if features.needs_cross:
            if batch.cross is None:
                raise ValueError("this model's inputs need cross masks but the batch has none")
            bits = (batch.cross.view(b, 2, 1, SQUARES) & self.cross_bits) != 0  # [B, 2, 27, 225]
        if features.cross_checks:
            flags.append(bits.view(b, CROSS_PLANES, SQUARES))
        if features.last_move_plane:
            flags.append((code & _core.CODE_OPPONENT) != 0)
        graded: list[torch.Tensor] = []  # planes that are not yes/no, each [B, k, 225] float
        scalars_open = None
        if features.openness:
            if batch.open is None:
                raise ValueError("this model's inputs need openness bytes but the batch has none")
            halves = torch.stack([batch.open & 0x0F, batch.open >> 4], dim=1)  # [B, 2, 225]: row, column
            fewest = (halves & _core.OPEN_TILES).float()
            reach = torch.where(fewest > 0, (8.0 - fewest) / 7.0, torch.zeros_like(fewest))
            seven = ((halves & _core.OPEN_SEVEN) != 0).float()
            either = reach.amax(dim=1, keepdim=True)  # [B, 1, 225]
            triple, double = self.premium[:, 3:4], self.premium[:, 2:3]
            graded += [reach, seven, either * triple, either * double]
            in_reach = (either > 0).float()
            scalars_open = torch.stack(
                [
                    (in_reach * triple).sum(dim=(1, 2)) / 8.0,
                    (in_reach * double).sum(dim=(1, 2)) / 17.0,
                    in_reach.sum(dim=(1, 2)) / empty.sum(dim=(1, 2)).clamp(min=1),
                    seven.sum(dim=(1, 2)) / 450.0,
                ],
                dim=1,
            )
        if features.hooks:
            free = empty.view(b, N, N)
            taken = ~free
            left, right = F.pad(taken[:, :, :-1], (1, 0)), F.pad(taken[:, :, 1:], (0, 1))
            above, below = F.pad(taken[:, :-1, :], (0, 0, 1, 0)), F.pad(taken[:, 1:, :], (0, 0, 0, 1))
            # A letter here joins the row word beside it: the masks for column plays say which.
            # A letter above or below a column word: the masks for row plays.
            share = bits[:, :, :26].sum(dim=2).float().view(b, 2, N, N) / 26.0
            fits_s = bits[:, :, ord("S") - ord("A")].view(b, 2, N, N)
            row_word, column_word = share[:, 1], share[:, 0]
            graded.append(
                torch.stack(
                    [
                        row_word * (free & left & ~right),
                        row_word * (free & right & ~left),
                        column_word * (free & above & ~below),
                        column_word * (free & below & ~above),
                        (fits_s[:, 1] & free & left).float(),
                        (fits_s[:, 0] & free & above).float(),
                    ],
                    dim=1,
                ).view(b, HOOK_PLANES, SQUARES)
            )
        board = torch.cat(flags, dim=1).to(dtype=self.dtype)
        if graded:
            board = torch.cat([board, *(plane.to(dtype=self.dtype) for plane in graded)], dim=1)
        board = board.view(b, self.planes, N, N).contiguous(memory_format=torch.channels_last)

        meta = batch.meta.float()
        leave = batch.leave.float()
        unseen = batch.unseen.float()
        bag = meta[:, _core.M_BAG]
        kind = batch.meta[:, _core.M_KIND]
        their_last = batch.meta[:, _core.M_OPPONENT_LAST_KIND]
        their_tiles = meta[:, _core.M_OPPONENT_TILES]
        hidden = unseen.sum(dim=1)
        scalars = [
            bag / 86.0,
            meta[:, _core.M_DRAWS] / 7.0,
            leave.sum(dim=1) / 7.0,
            (meta[:, _core.M_SPREAD] / 100.0).clamp(-4.0, 4.0),
            meta[:, _core.M_SCORELESS] / 6.0,
            their_tiles / 7.0,
            (bag == 0).float(),
            (bag < 7).float(),
            (kind == 0).float(),
            (kind == 1).float(),
            (kind == 2).float(),
            hidden / 93.0,
            batch.leave_value / 20.0,
            (their_last == 1).float(),
            (their_last == 2).float(),
            (their_last == 3).float(),
            meta[:, _core.M_OPPONENT_LAST_TILES] / 7.0,
            (meta[:, _core.M_OPPONENT_LAST_SCORE] / 50.0).clamp(max=3.0),
        ]
        parts = [leave / 2.0, unseen / self.tile_totals, torch.stack(scalars, dim=1)]
        if features.extra_scalars:
            since = meta[:, _core.M_OPPONENT_SINCE_BINGO]
            pool = hidden.clamp(min=1.0).unsqueeze(1)
            kept = leave.sum(dim=1).clamp(min=1.0)
            parts.append(
                torch.cat(
                    [
                        torch.stack(
                            [
                                (meta[:, _core.M_SCORE] / 50.0).clamp(max=3.0),
                                (since >= 0).float(),
                                torch.where(since >= 0, since.clamp(max=8.0) / 8.0, torch.ones_like(since)),
                            ],
                            dim=1,
                        ),
                        unseen[:, self.power] / pool,
                        torch.stack(
                            [
                                (unseen * self.vowels).sum(dim=1) / pool.squeeze(1),
                                (unseen * self.consonants).sum(dim=1) / pool.squeeze(1),
                                (leave * self.vowels).sum(dim=1) / kept,
                                (leave * self.consonants).sum(dim=1) / kept,
                            ],
                            dim=1,
                        ),
                    ],
                    dim=1,
                )
            )
        if scalars_open is not None:
            parts.append(scalars_open)
        if features.belief:
            # A rack drawn blind from the unseen pool; once the bag is empty this is the rack itself.
            blind = unseen * (their_tiles / hidden.clamp(min=1.0)).unsqueeze(1)
            inferred = (batch.meta[:, _core.M_BELIEF_INFERRED] > 0).unsqueeze(1)
            counts = meta[:, _core.M_BELIEF:_core.M_BELIEF + TILE_TYPES] / float(_core.BELIEF_SCALE)
            belief = torch.where(inferred, counts, blind)
            parts += [belief / 2.0, belief - blind, inferred.float()]
        return board, torch.cat(parts, dim=1)
