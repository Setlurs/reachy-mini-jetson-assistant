# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Movements — emotion-driven robot behaviors for Reachy Mini.

Executes head poses and antenna movements on a background thread so the
main pipeline is never blocked. A simple priority system ensures emotion
reactions override idle tracking.
"""

import threading
import time
from typing import Optional

import numpy as np

try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from app.emotion import Emotion


def _head_pose(roll: float = 0, pitch: float = 0, yaw: float = 0) -> np.ndarray:
    """Build a 4x4 head pose from Euler angles in degrees."""
    pose = np.eye(4)
    if HAS_SCIPY:
        pose[:3, :3] = Rotation.from_euler(
            "xyz", [roll, pitch, yaw], degrees=True
        ).as_matrix()
    return pose


class MovementController:
    """Non-blocking emotion-driven movement for Reachy Mini.

    All movements execute on a background thread. Calling react() while
    a previous movement is running will cancel it and start the new one.
    Duplicate emotions and low-confidence detections are suppressed.
    """

    MIN_CONFIDENCE = 0.75
    COOLDOWN_SECS = 8.0

    def __init__(self, reachy, antenna_rest: list[float] | None = None):
        self._reachy = reachy
        self._antenna_rest = antenna_rest or [0.0, 0.0]
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_emotion: Optional[Emotion] = None
        self._last_react_time: float = 0.0
        # When True, emotion-driven reactions skip — keeps the head
        # where the user explicitly aimed it instead of snapping back
        # to neutral on every utterance. Set by move_head, cleared
        # when the user faces forward again.
        self._manual_head: bool = False

    @property
    def manual_head(self) -> bool:
        return self._manual_head

    def set_manual_head(self, on: bool) -> None:
        self._manual_head = bool(on)
        if on:
            # Preempt any in-flight emotion sequence so its trailing
            # goto_target(neutral) can't snap the head back right after
            # we point it. The sequences poll `_cancel` between steps
            # and return early.
            self._cancel.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)
            self._cancel.clear()

    def react(self, emotion: Emotion, confidence: float = 1.0) -> bool:
        """Trigger a movement sequence for the given emotion (non-blocking).

        Returns True if a movement was triggered, False if suppressed.
        Suppressed when: NEUTRAL, confidence too low, same emotion within
        cooldown, or no reachy connected.
        """
        if self._reachy is None:
            return False

        # Respect a user-set head position: skip emotion reactions so
        # "move left" stays put across subsequent turns.
        if self._manual_head:
            return False

        if emotion == Emotion.NEUTRAL:
            return False

        if confidence < self.MIN_CONFIDENCE:
            return False

        now = time.time()
        if (emotion == self._last_emotion
                and (now - self._last_react_time) < self.COOLDOWN_SECS):
            return False

        self._cancel.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        self._cancel.clear()
        self._last_emotion = emotion
        self._last_react_time = now
        self._thread = threading.Thread(
            target=self._run_sequence, args=(emotion,), daemon=True
        )
        self._thread.start()
        return True

    def reset(self) -> None:
        """Return head and antennas to neutral position, and resume
        autonomous emotion reactions (clears the manual-head override)."""
        self._manual_head = False
        if self._reachy is None:
            return
        self._cancel.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._cancel.clear()
        try:
            self._reachy.goto_target(_head_pose(), duration=0.4)
            self._reachy.set_target_antenna_joint_positions(self._antenna_rest)
        except Exception:
            pass

    def play(self, emotion: Emotion) -> bool:
        """Run an emotion sequence on user demand.

        Unlike react(), this bypasses the manual-head override and the
        cooldown — explicit requests always play. Cancels any in-flight
        sequence first so the new one isn't interrupted by old steps.
        """
        if self._reachy is None:
            return False
        if emotion == Emotion.NEUTRAL:
            self.reset()
            return True
        self._cancel.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._cancel.clear()
        self._last_emotion = emotion
        self._last_react_time = time.time()
        self._thread = threading.Thread(
            target=self._run_sequence, args=(emotion,), daemon=True,
        )
        self._thread.start()
        return True

    def _run_sequence(self, emotion: Emotion) -> None:
        try:
            handler = _SEQUENCES.get(emotion, _seq_neutral)
            handler(self._reachy, self._cancel, self._antenna_rest)
        except Exception:
            pass

    @property
    def is_moving(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_emotion(self) -> Optional[Emotion]:
        return self._last_emotion


def _wait(cancel: threading.Event, secs: float) -> bool:
    """Sleep that respects cancellation. Returns True if cancelled."""
    return cancel.wait(timeout=secs)


def _seq_happy(reachy, cancel: threading.Event, rest: list[float]) -> None:
    reachy.goto_target(_head_pose(pitch=-8), duration=0.3)
    if _wait(cancel, 0.2):
        return
    reachy.set_target_antenna_joint_positions([0.4, -0.4])
    if _wait(cancel, 0.3):
        return
    reachy.goto_target(_head_pose(pitch=5), duration=0.2)
    if _wait(cancel, 0.2):
        return
    reachy.goto_target(_head_pose(), duration=0.3)
    reachy.set_target_antenna_joint_positions(rest)


def _seq_sad(reachy, cancel: threading.Event, rest: list[float]) -> None:
    reachy.goto_target(_head_pose(pitch=12), duration=0.6)
    reachy.set_target_antenna_joint_positions([-0.3, -0.3])
    if _wait(cancel, 1.0):
        return
    reachy.goto_target(_head_pose(), duration=0.5)
    reachy.set_target_antenna_joint_positions(rest)


def _seq_curious(reachy, cancel: threading.Event, rest: list[float]) -> None:
    reachy.goto_target(_head_pose(roll=10, pitch=-3), duration=0.4)
    if _wait(cancel, 0.6):
        return
    reachy.goto_target(_head_pose(), duration=0.3)
    reachy.set_target_antenna_joint_positions(rest)


def _seq_excited(reachy, cancel: threading.Event, rest: list[float]) -> None:
    reachy.goto_target(_head_pose(pitch=-10), duration=0.2)
    for _ in range(3):
        if _wait(cancel, 0.15):
            return
        reachy.set_target_antenna_joint_positions([0.6, -0.6])
        if _wait(cancel, 0.15):
            return
        reachy.set_target_antenna_joint_positions([-0.6, 0.6])
    reachy.goto_target(_head_pose(), duration=0.3)
    reachy.set_target_antenna_joint_positions(rest)


def _seq_greeting(reachy, cancel: threading.Event, rest: list[float]) -> None:
    reachy.goto_target(_head_pose(pitch=-10), duration=0.3)
    reachy.set_target_antenna_joint_positions([0.5, -0.5])
    if _wait(cancel, 0.4):
        return
    reachy.set_target_antenna_joint_positions([-0.5, 0.5])
    if _wait(cancel, 0.3):
        return
    reachy.goto_target(_head_pose(), duration=0.3)
    reachy.set_target_antenna_joint_positions(rest)


def _seq_farewell(reachy, cancel: threading.Event, rest: list[float]) -> None:
    reachy.goto_target(_head_pose(pitch=-5), duration=0.4)
    if _wait(cancel, 0.3):
        return
    reachy.goto_target(_head_pose(pitch=5), duration=0.3)
    if _wait(cancel, 0.3):
        return
    reachy.set_target_antenna_joint_positions([-0.2, -0.2])
    if _wait(cancel, 0.5):
        return
    reachy.goto_target(_head_pose(), duration=0.4)
    reachy.set_target_antenna_joint_positions(rest)


def _seq_grateful(reachy, cancel: threading.Event, rest: list[float]) -> None:
    reachy.goto_target(_head_pose(pitch=-10), duration=0.3)
    if _wait(cancel, 0.3):
        return
    reachy.goto_target(_head_pose(pitch=5), duration=0.2)
    if _wait(cancel, 0.15):
        return
    reachy.goto_target(_head_pose(pitch=-5), duration=0.2)
    if _wait(cancel, 0.3):
        return
    reachy.goto_target(_head_pose(), duration=0.3)


def _seq_neutral(reachy, cancel: threading.Event, rest: list[float]) -> None:
    pass


def _seq_surprised(reachy, cancel: threading.Event, rest: list[float]) -> None:
    # Sharp jerk back + antennas wide, hold, then settle.
    reachy.goto_target(_head_pose(pitch=-15, yaw=0), duration=0.18)
    reachy.set_target_antenna_joint_positions([0.9, -0.9])
    if _wait(cancel, 0.35):
        return
    reachy.goto_target(_head_pose(pitch=-8), duration=0.25)
    if _wait(cancel, 0.3):
        return
    reachy.goto_target(_head_pose(), duration=0.3)
    reachy.set_target_antenna_joint_positions(rest)


_SEQUENCES = {
    Emotion.HAPPY: _seq_happy,
    Emotion.SAD: _seq_sad,
    Emotion.CURIOUS: _seq_curious,
    Emotion.EXCITED: _seq_excited,
    Emotion.SURPRISED: _seq_surprised,
    Emotion.GREETING: _seq_greeting,
    Emotion.FAREWELL: _seq_farewell,
    Emotion.GRATEFUL: _seq_grateful,
    Emotion.NEUTRAL: _seq_neutral,
}
