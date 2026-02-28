"""Basic Reachy Mini Lite movement test on Jetson.

Spawns the daemon, connects to the robot, and runs through
a sequence of simple movements: wake up, look around, wiggle
antennas, nod, and go to sleep.
"""

import time
import math
import numpy as np
from reachy_mini import ReachyMini
from scipy.spatial.transform import Rotation as R


def make_head_pose(roll_deg=0, pitch_deg=0, yaw_deg=0):
    """Build a 4x4 head pose from Euler angles (degrees)."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    return pose


def main():
    print("Connecting to Reachy Mini Lite (spawning daemon)...")
    print("  Motor setup takes ~15s on first connect, please wait...")
    reachy = ReachyMini(spawn_daemon=True, use_sim=False, timeout=30.0, media_backend="no_media")
    print("Connected!\n")

    try:
        # 1. Wake up
        print("1. Waking up...")
        reachy.wake_up()
        time.sleep(1)

        # 2. Look left, center, right
        print("2. Looking around...")
        reachy.goto_target(make_head_pose(yaw_deg=25), duration=0.8)
        time.sleep(1)
        reachy.goto_target(make_head_pose(yaw_deg=-25), duration=0.8)
        time.sleep(1)
        reachy.goto_target(make_head_pose(), duration=0.6)
        time.sleep(0.5)

        # 3. Nod (pitch up and down)
        print("3. Nodding...")
        reachy.goto_target(make_head_pose(pitch_deg=-15), duration=0.4)
        time.sleep(0.5)
        reachy.goto_target(make_head_pose(pitch_deg=10), duration=0.3)
        time.sleep(0.4)
        reachy.goto_target(make_head_pose(), duration=0.3)
        time.sleep(0.5)

        # 4. Tilt head (roll)
        print("4. Tilting head...")
        reachy.goto_target(make_head_pose(roll_deg=20), duration=0.5)
        time.sleep(0.8)
        reachy.goto_target(make_head_pose(roll_deg=-20), duration=0.5)
        time.sleep(0.8)
        reachy.goto_target(make_head_pose(), duration=0.4)
        time.sleep(0.5)

        # 5. Wiggle antennas
        print("5. Wiggling antennas...")
        for _ in range(3):
            reachy.set_target_antenna_joint_positions([0.5, -0.5])
            time.sleep(0.3)
            reachy.set_target_antenna_joint_positions([-0.5, 0.5])
            time.sleep(0.3)
        reachy.set_target_antenna_joint_positions([0.0, 0.0])
        time.sleep(0.5)

        # 6. Look at a point in world space (0.3m forward, slightly up)
        print("6. Looking at a world point...")
        reachy.look_at_world(x=0.3, y=0.0, z=0.1, duration=1.0)
        time.sleep(1)
        reachy.look_at_world(x=0.3, y=0.15, z=0.0, duration=0.8)
        time.sleep(1)
        reachy.look_at_world(x=0.3, y=-0.15, z=0.0, duration=0.8)
        time.sleep(1)

        # 7. Return to center and go to sleep
        print("7. Going to sleep...")
        reachy.goto_sleep()
        time.sleep(1)

        print("\nDone! All movements completed.")

    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors...")
    finally:
        reachy.disable_motors()
        print("Motors disabled.")


if __name__ == "__main__":
    main()
