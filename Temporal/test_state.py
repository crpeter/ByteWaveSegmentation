"""Small state-contract tests, runnable without PyTorch or Core ML.

Run manually: python -m unittest discover -s Temporal -p 'test_*.py'
"""
from fractions import Fraction
import unittest

import numpy as np

from state import TemporalState


def output(index):
    return {"memory_features": np.full((1, 512, 64), index, dtype=np.float32),
            "memory_positions": np.full((1, 512, 64), index + 100, dtype=np.float32),
            "object_pointer": np.full((1, 256), index, dtype=np.float32),
            "low_res_mask": np.ones((1, 1, 256, 256), dtype=np.float32),
            "object_score": np.ones((1, 1), dtype=np.float32)}


class StateContractTests(unittest.TestCase):
    def populate(self, count):
        state = TemporalState()
        for index in range(count):
            state.commit(output(index), Fraction(index, 30), state.token)
        return state

    def test_startup_does_not_duplicate_conditioning_frame(self):
        state = self.populate(1)
        packed = state.pack(Fraction(1, 30))
        self.assertEqual(np.flatnonzero(packed["valid_slots"][0]).tolist(), [0, 7])
        state.commit(output(1), Fraction(1, 30), state.token)
        packed = state.pack(Fraction(2, 30))
        self.assertEqual(np.flatnonzero(packed["valid_slots"][0]).tolist(), [0, 6, 7, 8])
        self.assertEqual(packed["spatial_bank"][0, 6, 0, 0], 1)

    def test_long_stream_preserves_condition_and_evicts_old_history(self):
        state = self.populate(100)
        packed = state.pack(Fraction(100, 30))
        self.assertEqual(packed["spatial_bank"][0, :, 0, 0].tolist(), [0, 94, 95, 96, 97, 98, 99])
        self.assertEqual(packed["pointer_bank"][0, :, 0].tolist(), [0] + list(range(99, 84, -1)))
        self.assertTrue(np.all(packed["valid_slots"] == 1))
        self.assertEqual(state.summary(), {"acceptedFrames": 100, "spatialEntries": 7, "pointerEntries": 16})

    def test_timestamps_are_not_derived_from_nominal_fps(self):
        state = TemporalState()
        for index, pts in enumerate([Fraction(7, 1000), Fraction(44, 1000), Fraction(101, 1000)]):
            state.commit(output(index), pts, state.token)
        self.assertEqual(state.last_timestamp, Fraction(101, 1000))
        with self.assertRaises(ValueError):
            state.pack(Fraction(44, 1000))
        with self.assertRaises(ValueError):
            state.pack(Fraction(101, 1000))

    def test_failed_output_does_not_advance_state(self):
        state = self.populate(2)
        bad = output(2)
        bad["memory_positions"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            state.commit(bad, Fraction(2, 30), state.token)
        self.assertEqual(state.next_index, 2)
        self.assertEqual(state.last_timestamp, Fraction(1, 30))

    def test_reset_rejects_stale_first_frame_and_clears_every_bank(self):
        state = TemporalState()
        old_token = state.token
        state.reset()
        with self.assertRaises(ValueError):
            state.commit(output(0), Fraction(0), old_token)
        with self.assertRaises(ValueError):
            state.pack(Fraction(0))
        self.assertEqual(state.summary(), {"acceptedFrames": 0, "spatialEntries": 0, "pointerEntries": 0})

    def test_absent_subject_is_retained_as_model_output(self):
        state = self.populate(1)
        missing = output(1)
        missing["low_res_mask"].fill(-1024)
        missing["object_score"].fill(-10)
        state.commit(missing, Fraction(1, 30), state.token)
        self.assertEqual(state.next_index, 2)
        self.assertEqual(state.pack(Fraction(2, 30))["spatial_bank"][0, 6, 0, 0], 1)


if __name__ == "__main__":
    unittest.main()
