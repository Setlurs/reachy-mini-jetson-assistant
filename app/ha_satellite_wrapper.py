"""Run the HA satellite with motor commands stubbed out.

The satellite ships its own MovementManager that wants to control the
head and antennas — listening pose on wake-word, speaking animations
during HA TTS, idle behavior, "reset to neutral on startup," and a 100
Hz control loop. When our LLM pipeline is also driving the robot, the
two fight and Reachy ends up sleeping mid-conversation.

Solution: in the satellite's own subprocess, patch the ReachyMini SDK's
motor-side methods to no-ops before the satellite imports anything.
Audio, camera, ESPHome protocol, wake-word detection — all unchanged.
The satellite's MovementManager keeps running but its commands silently
drop. Motors stay under our project's exclusive control.

This wrapper is what app/ha_satellite.py spawns instead of running
`python -m reachy_mini_home_assistant` directly. CLI args pass through
to the satellite via the shared sys.argv.
"""

from __future__ import annotations

import sys


# Methods that actuate motors. Getters (get_current_head_pose,
# get_present_antenna_joint_positions) intentionally stay live so the
# satellite's listening / speaking state machines can still read pose
# without changing it.
_MOTOR_METHODS = (
    "async_play_move",
    "play_move",
    "disable_motors",
    "enable_motors",
    "goto_sleep",
    "wake_up",
    "goto_target",
    "set_target",
    "set_target_head_pose",
    "set_target_antenna_joint_positions",
    "set_target_body_yaw",
)


def _stub_motor_methods() -> None:
    from reachy_mini import ReachyMini

    def _make_noop(name: str):
        def _noop(self, *args, **kwargs):
            return None
        _noop.__name__ = name
        _noop.__qualname__ = f"ReachyMini.{name}"
        return _noop

    for name in _MOTOR_METHODS:
        if hasattr(ReachyMini, name):
            setattr(ReachyMini, name, _make_noop(name))


def main() -> None:
    _stub_motor_methods()
    # Importing the satellite's __main__ pulls in its full module tree,
    # which is fine — by now ReachyMini is already patched, so any
    # MovementManager / motion_bridge calls land on no-ops.
    from reachy_mini_home_assistant.__main__ import run
    run()


if __name__ == "__main__":
    main()
