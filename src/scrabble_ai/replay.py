"""Replay buffer: a fixed-size ring of afterstates with their training targets, in RAM.

Each position is stored as the engine's compact codes (about 310 bytes), not as input planes.
Cross-check masks are recomputed from the board when a batch is sampled, so they cost no memory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from scrabble_ai import _core

FIELDS: dict[str, tuple[type, tuple[int, ...]]] = {
    "board": (np.uint8, (_core.SQUARES,)),
    "leave": (np.uint8, (_core.TILE_TYPES,)),
    "unseen": (np.uint8, (_core.TILE_TYPES,)),
    "meta": (np.int16, (_core.META,)),
    "leave_value": (np.float32, ()),
    "value_target": (np.float32, ()),  # spread the mover went on to gain, in points
    "win_target": (np.float32, ()),  # 1 win, 0.5 tie, 0 loss, for the mover
}


class Replay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.arrays = {name: np.zeros((capacity, *shape), dtype) for name, (dtype, shape) in FIELDS.items()}
        self.size = 0  # positions currently held
        self.next = 0  # where the next position is written
        self.total_added = 0  # positions ever added

    def __len__(self) -> int:
        return self.size

    @property
    def bytes_per_position(self) -> int:
        return sum(a[0].nbytes for a in self.arrays.values())

    @property
    def memory_bytes(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())

    def add(self, rows: dict[str, np.ndarray]) -> None:
        """Append a block of positions (same keys as FIELDS), overwriting the oldest when full."""
        count = added = len(rows["board"])
        if count == 0:
            return
        if count > self.capacity:
            rows = {name: array[-self.capacity:] for name, array in rows.items()}
            count = self.capacity
        first = min(count, self.capacity - self.next)
        for name, array in self.arrays.items():
            array[self.next:self.next + first] = rows[name][:first]
            if first < count:
                array[: count - first] = rows[name][first:]
        self.next = (self.next + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.total_added += added

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """A uniformly random batch (with replacement), sorted by index for memory locality."""
        index = np.sort(rng.integers(0, self.size, size=batch_size))
        return {name: array[index] for name, array in self.arrays.items()}

    def state(self) -> dict[str, int]:
        return {"size": self.size, "next": self.next, "total_added": self.total_added, "capacity": self.capacity}

    def save(self, directory: Path) -> None:
        """Write the positions held, oldest first, as `<name>.npy`. Each file goes through a
        temporary name so an interrupted save never leaves a truncated file under the final one."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.size == self.capacity and self.next != 0:
            # The ring has wrapped: the oldest position sits at `next`.
            order = np.concatenate([np.arange(self.next, self.capacity), np.arange(self.next)])
        else:
            order = slice(0, self.size)
        for name, array in self.arrays.items():
            temporary = directory / f"{name}.npy.tmp"
            with open(temporary, "wb") as fh:
                np.save(fh, array[order])
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, directory / f"{name}.npy")

    def load(self, directory: Path, state: dict[str, int]) -> None:
        """Restore what `save` wrote. `state` is the matching `state()` from the checkpoint.

        The files hold positions oldest first and each position carries its own targets, so the
        files alone decide what is restored: the newest positions that fit this buffer, with
        writing set to overwrite the oldest of them next. A different capacity, or files newer
        than the checkpoint, therefore need no special handling.
        """
        saved = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in FIELDS}
        lengths = {len(array) for array in saved.values()}
        if len(lengths) != 1:
            raise ValueError(f"replay files in {directory} have different lengths: a save was interrupted")
        count = lengths.pop()
        keep = min(count, self.capacity)
        for name, array in self.arrays.items():
            array[:keep] = saved[name][count - keep:]
        self.size = keep
        self.next = keep % self.capacity
        self.total_added = state["total_added"]
