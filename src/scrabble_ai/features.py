"""Network inputs, built on the GPU from the engine's compact afterstate codes.

An afterstate arrives as small integer arrays (see `scrabble_core::encode`). This module turns a
batch of them into what the network consumes:

Board planes, `[B, PLANES, 15, 15]`:
     0-25   one plane per letter: a tile showing that letter is on the square
       26   the tile is a blank
       27   the tile was placed by the move that produced this position
       28   the square is empty
    29-32   an empty double-letter / triple-letter / double-word / triple-word square
    33-58   for a play along the row: letters the tiles above and below allow on this empty square
       59   ... and whether there are such tiles (the play would form a cross-word here)
    60-85   the same for a play down the column (tiles to the left and right)
       86   ... and whether there are such tiles
  Planes 33-86 are the cross-check planes; they are left out when `cross_checks` is off.

Vector, `[B, VECTOR]`:
     0-26   tiles kept on the rack, by type (count / 2)
    27-53   tiles the mover cannot see, by type (fraction of that type's total)
    54-...  scalars, see `SCALARS`

Everything here is information the mover has. The opponent's rack is never an input; it only
shows up, correctly, inside the unseen pool once the bag is empty. What the opponent did on their
previous turn is public, and is included because it hints at what they kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from scrabble_ai import _core

N = 15
SQUARES = _core.SQUARES
TILE_TYPES = _core.TILE_TYPES
META = _core.META

BASE_PLANES = 33
CROSS_PLANES = 54
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
VECTOR = 2 * TILE_TYPES + len(SCALARS)


def plane_count(cross_checks: bool) -> int:
    return BASE_PLANES + (CROSS_PLANES if cross_checks else 0)


@dataclass
class Afterstates:
    """A batch of afterstates as tensors on one device (integer codes, not yet expanded)."""

    board: torch.Tensor  # uint8 [B, 225]
    leave: torch.Tensor  # uint8 [B, 27]
    unseen: torch.Tensor  # uint8 [B, 27]
    meta: torch.Tensor  # int16 [B, META]
    leave_value: torch.Tensor  # float32 [B]
    cross: torch.Tensor | None  # int32 [B, 450], or None when cross-checks are off

    def __len__(self) -> int:
        return self.board.shape[0]


def to_device(
    board: np.ndarray,
    leave: np.ndarray,
    unseen: np.ndarray,
    meta: np.ndarray,
    leave_value: np.ndarray,
    cross: np.ndarray | None,
    device: torch.device,
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
    )


class FeatureBuilder:
    """Expands `Afterstates` into (planes, vector). Holds the constant tables on the device."""

    def __init__(self, device: torch.device, cross_checks: bool = True, dtype: torch.dtype = torch.float32):
        """`dtype` is the type of the board planes: bfloat16 halves their memory under autocast."""
        self.device = device
        self.cross_checks = cross_checks
        self.dtype = dtype
        self.planes = plane_count(cross_checks)
        letter = torch.tensor(_core.LETTER_MULT, device=device)
        word = torch.tensor(_core.WORD_MULT, device=device)
        self.premium = torch.stack([letter == 2, letter == 3, word == 2, word == 3]).view(1, 4, SQUARES)
        self.tile_totals = torch.tensor(_core.TILE_COUNTS, device=device).float()
        self.letter_ids = torch.arange(1, 27, device=device, dtype=torch.uint8).view(1, 26, 1)
        self.cross_bits = (1 << torch.arange(27, device=device, dtype=torch.int32)).view(1, 1, 27, 1)

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
        return Afterstates(board, batch.leave, batch.unseen, batch.meta, batch.leave_value, cross)

    def __call__(self, batch: Afterstates) -> tuple[torch.Tensor, torch.Tensor]:
        b = len(batch)
        # Everything is assembled as booleans (one byte per plane entry) and converted once.
        code = batch.board.unsqueeze(1)  # [B, 1, 225]
        letter = code & _core.CODE_LETTER  # 0 empty, 1..26
        empty = letter == 0
        planes = [
            letter == self.letter_ids,  # [B, 26, 225]
            (code & _core.CODE_BLANK) != 0,
            (code & _core.CODE_PLACED) != 0,
            empty,
            self.premium & empty,  # [B, 4, 225]
        ]
        if self.cross_checks:
            if batch.cross is None:
                raise ValueError("this model uses cross-check planes but the batch has no cross masks")
            bits = (batch.cross.view(b, 2, 1, SQUARES) & self.cross_bits) != 0  # [B, 2, 27, 225]
            planes.append(bits.view(b, CROSS_PLANES, SQUARES))
        board = torch.cat(planes, dim=1).view(b, self.planes, N, N)
        board = board.to(dtype=self.dtype, memory_format=torch.channels_last)

        meta = batch.meta.float()
        leave = batch.leave.float()
        unseen = batch.unseen.float()
        bag = meta[:, _core.M_BAG]
        kind = batch.meta[:, _core.M_KIND]
        their_last = batch.meta[:, _core.M_OPPONENT_LAST_KIND]
        scalars = torch.stack(
            [
                bag / 86.0,
                meta[:, _core.M_DRAWS] / 7.0,
                leave.sum(dim=1) / 7.0,
                (meta[:, _core.M_SPREAD] / 100.0).clamp(-4.0, 4.0),
                meta[:, _core.M_SCORELESS] / 6.0,
                meta[:, _core.M_OPPONENT_TILES] / 7.0,
                (bag == 0).float(),
                (bag < 7).float(),
                (kind == 0).float(),
                (kind == 1).float(),
                (kind == 2).float(),
                unseen.sum(dim=1) / 93.0,
                batch.leave_value / 20.0,
                (their_last == 1).float(),
                (their_last == 2).float(),
                (their_last == 3).float(),
                meta[:, _core.M_OPPONENT_LAST_TILES] / 7.0,
                (meta[:, _core.M_OPPONENT_LAST_SCORE] / 50.0).clamp(max=3.0),
            ],
            dim=1,
        )
        vector = torch.cat([leave / 2.0, unseen / self.tile_totals, scalars], dim=1)
        return board, vector
