"""Bounded sequential state for owned v1 only; no community-model compatibility."""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class Entry:
    index: int
    features: np.ndarray
    positions: np.ndarray


def checked(value, shape):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite tensor {shape}; received {array.shape}")
    return array.copy()


class TemporalState:
    """One condition, six recent spatial entries, fifteen recent object pointers.

    Frame indices count accepted decoded frames; time uses exact media PTS.
    New video, selection change or seek requires reset and a new initial mask.
    No automatically reused state after a discontinuity.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.generation = getattr(self, "generation", 0) + 1
        self.next_index = 0
        self.last_timestamp = None
        self.condition = None
        self.condition_pointer = None
        self.recent = deque(maxlen=6)
        self.pointers = deque(maxlen=15)

    @property
    def token(self):
        return self.generation, self.next_index

    def validate_time(self, timestamp):
        if not isinstance(timestamp, Fraction):
            raise ValueError("Use exact rational media timestamps.")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Non-increasing timestamp: reset and reselect after seeking.")

    def pack(self, timestamp):
        self.validate_time(timestamp)
        if self.condition is None:
            raise ValueError("Initialize the subject before propagation.")
        spatial = np.zeros((1, 7, 512, 64), dtype=np.float32)
        positions = np.zeros_like(spatial)
        pointers = np.zeros((1, 16, 256), dtype=np.float32)
        valid = np.zeros((1, 23), dtype=np.float32)
        spatial[0, 0], positions[0, 0] = self.condition.features[0], self.condition.positions[0]
        pointers[0, 0] = self.condition_pointer[0]
        valid[0, 0] = valid[0, 7] = 1
        history = {entry.index: entry for entry in self.recent}
        for slot, lag in enumerate(range(6, 0, -1), start=1):
            entry = history.get(self.next_index - lag)
            if entry is not None:
                spatial[0, slot], positions[0, slot] = entry.features[0], entry.positions[0]
                valid[0, slot] = 1
        pointer_history = dict(self.pointers)
        for lag in range(1, 16):
            pointer = pointer_history.get(self.next_index - lag)
            if pointer is not None:
                pointers[0, lag] = pointer[0]
                valid[0, 7 + lag] = 1
        return dict(spatial_bank=spatial, spatial_positions=positions,
                    pointer_bank=pointers, valid_slots=valid)

    def commit(self, output, timestamp, expected_token):
        self.validate_time(timestamp)
        if expected_token != self.token:
            raise ValueError("Stale prediction; state was advanced or reset.")
        # Validate before mutation: a failed frame must leave the state unchanged.
        features = checked(output["memory_features"], (1, 512, 64))
        positions = checked(output["memory_positions"], (1, 512, 64))
        pointer = checked(output["object_pointer"], (1, 256))
        checked(output["low_res_mask"], (1, 1, 256, 256))
        checked(output["object_score"], (1, 1))
        entry = Entry(self.next_index, features, positions)
        if self.next_index == 0:
            self.condition, self.condition_pointer = entry, pointer
        else:
            self.recent.append(entry)
            self.pointers.append((self.next_index, pointer))
        self.next_index += 1
        self.last_timestamp = timestamp

    def summary(self):
        return {"acceptedFrames": self.next_index,
                "spatialEntries": int(self.condition is not None) + len(self.recent),
                "pointerEntries": int(self.condition_pointer is not None) + len(self.pointers)}
