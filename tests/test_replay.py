"""`replay.Replay`: the ring buffer, sampling, and save/load."""

from __future__ import annotations

import numpy as np
import pytest

from scrabble_ai import _core
from scrabble_ai.replay import FIELDS, Replay


def rows(ids) -> dict[str, np.ndarray]:
    """Positions whose every field encodes the position's id, so they can be traced."""
    ids = np.asarray(ids)
    n = len(ids)
    out = {}
    for name, (dtype, shape) in FIELDS.items():
        array = np.zeros((n, *shape), dtype)
        array[...] = (ids % 200).reshape(n, *([1] * len(shape))).astype(dtype)
        out[name] = array
    out["value_target"] = ids.astype(np.float32)  # exact up to 2**24
    out["meta"][:, 0] = ids.astype(np.int16)
    return out


def held(replay: Replay) -> list[int]:
    return replay.arrays["value_target"][: replay.size].astype(int).tolist()


def oldest_to_newest(replay: Replay) -> list[int]:
    if replay.size < replay.capacity:
        return held(replay)
    order = np.r_[replay.next:replay.capacity, 0:replay.next]
    return replay.arrays["value_target"][order].astype(int).tolist()


def test_fields_match_the_engine_layout():
    assert FIELDS["meta"] == (np.int16, (_core.META,))
    assert FIELDS["board"][1] == (_core.SQUARES,)
    replay = Replay(10)
    assert replay.bytes_per_position == 225 + 27 + 27 + 2 * _core.META + 4 + 4 + 4
    assert replay.memory_bytes == 10 * replay.bytes_per_position


def test_ring_wraps_and_overwrites_the_oldest():
    replay = Replay(10)
    replay.add(rows(range(0, 4)))
    assert (len(replay), replay.next, replay.total_added) == (4, 4, 4)
    assert held(replay) == [0, 1, 2, 3]
    replay.add(rows(range(4, 10)))
    assert (len(replay), replay.next) == (10, 0)
    replay.add(rows(range(10, 13)))  # wraps from slot 0
    assert (len(replay), replay.next, replay.total_added) == (10, 3, 13)
    assert held(replay) == [10, 11, 12, 3, 4, 5, 6, 7, 8, 9]
    replay.add(rows(range(13, 22)))  # crosses the end of the array
    assert replay.next == 2
    assert held(replay) == [20, 21, 12, 13, 14, 15, 16, 17, 18, 19]
    assert oldest_to_newest(replay) == list(range(12, 22))
    # Every field moved together.
    np.testing.assert_array_equal(replay.arrays["meta"][:, 0], replay.arrays["value_target"].astype(np.int16))
    np.testing.assert_array_equal(replay.arrays["board"][:, 7], replay.arrays["value_target"].astype(int) % 200)
    replay.add(rows([]))
    assert replay.total_added == 22


def test_adding_more_rows_than_the_capacity_keeps_the_newest():
    replay = Replay(5)
    replay.add(rows(range(0, 3)))
    replay.add(rows(range(3, 15)))
    assert len(replay) == 5
    assert sorted(held(replay)) == [10, 11, 12, 13, 14]
    assert oldest_to_newest(replay) == [10, 11, 12, 13, 14]
    replay.add(rows([15]))
    assert oldest_to_newest(replay) == [11, 12, 13, 14, 15]


def test_add_of_more_than_capacity_counts_every_position_in_total_added():
    """`total_added` is documented as "positions ever added"."""
    replay = Replay(5)
    replay.add(rows(range(12)))
    assert replay.total_added == 12


def test_sample_shapes_dtypes_and_only_filled_rows():
    replay = Replay(100)
    replay.add(rows(range(1000, 1030)))
    batch = replay.sample(500, np.random.default_rng(0))
    for name, (dtype, shape) in FIELDS.items():
        assert batch[name].dtype == dtype and batch[name].shape == (500, *shape)
        assert batch[name].flags.c_contiguous
    ids = batch["value_target"].astype(int)
    assert set(ids) <= set(range(1000, 1030)) and len(set(ids)) == 30
    assert (np.diff(ids) >= 0).all(), "sorted by index"
    # Rows stay consistent across fields.
    np.testing.assert_array_equal(batch["meta"][:, 0], batch["value_target"].astype(np.int16))
    # Changing the sample does not change the buffer.
    batch["value_target"][:] = -1
    assert held(replay) == list(range(1000, 1030))


def test_sample_from_an_empty_buffer_is_an_error():
    with pytest.raises(ValueError):
        Replay(10).sample(4, np.random.default_rng(0))


def test_save_and_load_round_trip(tmp_path):
    replay = Replay(10)
    replay.add(rows(range(7)))
    replay.save(tmp_path / "r")
    assert sorted(p.name for p in (tmp_path / "r").iterdir()) == sorted(f"{name}.npy" for name in FIELDS)
    restored = Replay(10)
    restored.load(tmp_path / "r", replay.state())
    assert restored.state() == replay.state()
    for name in FIELDS:
        np.testing.assert_array_equal(restored.arrays[name][:7], replay.arrays[name][:7])
    restored.add(rows([100]))
    assert held(restored) == [0, 1, 2, 3, 4, 5, 6, 100]


def test_save_and_load_after_wrapping_resumes_the_ring(tmp_path):
    replay = Replay(10)
    replay.add(rows(range(10)))
    replay.add(rows(range(10, 15)))  # wraps: physical order 10 11 12 13 14 5 6 7 8 9
    replay.save(tmp_path / "r")
    restored = Replay(10)
    restored.load(tmp_path / "r", replay.state())
    assert replay.state() == {"size": 10, "next": 5, "total_added": 15, "capacity": 10}
    # The files hold positions oldest first, so the restored ring starts at 0 with the same
    # contents in the same age order.
    assert restored.state() == {"size": 10, "next": 0, "total_added": 15, "capacity": 10}
    assert oldest_to_newest(restored) == oldest_to_newest(replay) == list(range(5, 15))
    restored.add(rows([15, 16]))
    replay.add(rows([15, 16]))
    assert oldest_to_newest(restored) == oldest_to_newest(replay) == list(range(7, 17))
    for name in FIELDS:
        np.testing.assert_array_equal(np.sort(restored.arrays[name], axis=0), np.sort(replay.arrays[name], axis=0))
    assert not list((tmp_path / "r").glob("*.tmp"))


def test_load_into_a_larger_capacity_before_wrapping(tmp_path):
    replay = Replay(10)
    replay.add(rows(range(6)))
    replay.save(tmp_path / "r")
    bigger = Replay(20)
    bigger.load(tmp_path / "r", replay.state())
    assert (len(bigger), bigger.next) == (6, 6)
    bigger.add(rows(range(6, 9)))
    assert held(bigger) == list(range(9))


def test_load_into_a_smaller_capacity_before_wrapping_keeps_the_newest(tmp_path):
    replay = Replay(10)
    replay.add(rows(range(8)))
    replay.save(tmp_path / "r")
    smaller = Replay(5)
    smaller.load(tmp_path / "r", replay.state())
    assert sorted(held(smaller)) == [3, 4, 5, 6, 7]
    assert smaller.total_added == 8
    smaller.add(rows([8]))
    assert sorted(held(smaller)) == [4, 5, 6, 7, 8], "the oldest kept position is the one overwritten"


def test_load_into_a_smaller_capacity_after_wrapping_keeps_the_newest(tmp_path):
    replay = Replay(10)
    replay.add(rows(range(10)))
    replay.add(rows(range(10, 15)))  # physical order: 10 11 12 13 14 5 6 7 8 9
    replay.save(tmp_path / "r")
    smaller = Replay(4)
    smaller.load(tmp_path / "r", replay.state())
    assert sorted(held(smaller)) == [11, 12, 13, 14]  # actually keeps 6 7 8 9


def test_load_into_a_larger_capacity_after_wrapping_overwrites_the_oldest_first(tmp_path):
    replay = Replay(10)
    replay.add(rows(range(10)))
    replay.add(rows(range(10, 15)))  # holds 5..14, physical order 10 11 12 13 14 5 6 7 8 9
    replay.save(tmp_path / "r")
    bigger = Replay(12)
    bigger.load(tmp_path / "r", replay.state())
    assert sorted(held(bigger)) == list(range(5, 15))
    bigger.add(rows(range(15, 18)))  # 13 positions for 12 places: position 5, the oldest, goes
    assert sorted(held(bigger)) == list(range(6, 18))  # actually loses 10 instead


def test_files_of_different_lengths_are_rejected(tmp_path):
    replay = Replay(10)
    replay.add(rows(range(6)))
    replay.save(tmp_path / "r")
    np.save(tmp_path / "r" / "leave.npy", replay.arrays["leave"][:4])
    with pytest.raises(ValueError, match="different lengths"):
        Replay(10).load(tmp_path / "r", replay.state())


def test_saving_an_empty_buffer(tmp_path):
    replay = Replay(5)
    replay.save(tmp_path / "r")
    restored = Replay(5)
    restored.load(tmp_path / "r", replay.state())
    assert (len(restored), restored.next, restored.total_added) == (0, 0, 0)
