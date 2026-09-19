"""Read-only ball tracking: python -m motion.trajectory_rolling --hz 60.

The rate is a localization request target, not a camera hardware setting.
Segmenter results have no exposure timestamps; velocities are approximate.
"""
import argparse
import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path

from connection import connect
from viam.components.arm import Arm
from viam.services.vision import VisionClient

from motion.viam_ball_catch import DEFAULT_CONFIG, locate_world


# Positions: meters
# Timestamps: monotonic seconds
# Assumes positive world Z points upward.
LAUNCH_THRESHOLD = 0.5  # m/s


class BallPredictor:
    def __init__(self):
        self.samples = deque(maxlen=10)

    def add_observation(self, timestamp, x, y, z):
        if not all(math.isfinite(v) for v in (timestamp, x, y, z)):
            return False
        if self.samples and timestamp <= self.samples[-1][0]:
            return False

        self.samples.append((timestamp, x, y, z))
        return True

    def estimate_velocity(self):
        if len(self.samples) < 2:
            return None

        t1, x1, y1, z1 = self.samples[-2]
        t2, x2, y2, z2 = self.samples[-1]

        dt = t2 - t1
        if dt <= 1e-6:
            return None

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        vz = (z2 - z1) / dt

        return vx, vy, vz

    def reset(self):
        self.samples.clear()


def positive_rate(value):
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise argparse.ArgumentTypeError("rate must be positive and finite")
    return rate


async def main(config_path=DEFAULT_CONFIG, hz=60.0, print_hz=5.0):
    config = json.loads(
        Path(config_path).read_text(encoding="utf-8")
    )

    predictor = BallPredictor()
    sample_period = 1.0 / positive_rate(hz)
    print_period = 1.0 / positive_rate(print_hz)

    async with await connect() as machine:
        arm = Arm.from_robot(machine, config["arm"])
        segmenter = VisionClient.from_robot(
            machine, config["segmenter"]
        )

        previous_xyz_mm = None
        first = True
        above_threshold = False

        last_print = float("-inf")
        rate_started = time.monotonic()
        observations = 0
        print(f"Tracking ball at a target of {hz:g} Hz. Keep the arm stationary.")
        print("Configure camera capture FPS separately in Viam. "
              "Reported rate measures localization, not camera exposure.")
        print("Press Ctrl+C to stop.")

        while True:
            started = time.monotonic()

            try:
                ball, _ = await locate_world(
                    machine,
                    arm,
                    config,
                    segmenter,
                    settle=first,
                    previous_xyz=previous_xyz_mm,
                )
            except (ValueError, asyncio.TimeoutError) as error:
                print(f"Skipping observation: {error}")

                predictor.reset()
                previous_xyz_mm = None
                above_threshold = False

                first = False
                await asyncio.sleep(max(0.0, sample_period - (time.monotonic() - started)))
                continue

            first = False

            # Approximate observation time: localization completion.
            # This is NOT the camera's frame capture timestamp.
            timestamp = time.monotonic()

            # Keep millimeters for locate_world's tracking input.
            previous_xyz_mm = (
                float(ball.x),
                float(ball.y),
                float(ball.z),
            )

            # Convert world coordinates to meters for the predictor.
            x, y, z = (
                value / 1000.0 for value in previous_xyz_mm
            )

            accepted = predictor.add_observation(
                timestamp, x, y, z
            )

            if not accepted:
                predictor.reset()
                previous_xyz_mm = None
                above_threshold = False

            if accepted:
                observations += 1
                show_sample = timestamp - last_print >= print_period
                if show_sample:
                    last_print = timestamp
                    print(
                        f"t={timestamp:.3f} s | "
                        f"x={x:.3f}, y={y:.3f}, z={z:.3f} m"
                    )

                velocity = predictor.estimate_velocity()

                if velocity is not None:
                    vx, vy, vz = velocity

                    if show_sample:
                        print(
                            f"vx={vx:.3f}, vy={vy:.3f}, "
                            f"vz={vz:.3f} m/s"
                        )

                    # Report when upward velocity crosses the threshold.
                    # This flags possible throws; it does not prove release.
                    currently_above = vz > LAUNCH_THRESHOLD

                    if currently_above and not above_threshold:
                        print("Possible throw detected!")

                    above_threshold = currently_above

            if timestamp - rate_started >= 2.0:
                actual_hz = observations / (timestamp - rate_started)
                status = " (below target)" if actual_hz < hz * 0.9 else ""
                print(f"Localization rate: {actual_hz:.1f} Hz / target {hz:g} Hz{status}")
                observations = 0
                rate_started = timestamp

            # Aim for the requested period, including processing time.
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, sample_period - elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hz", type=positive_rate, default=60.0,
                        help="Target localization requests per second (default: 60)")
    parser.add_argument("--print-hz", type=positive_rate, default=5.0,
                        help="Position/velocity output rate (default: 5)")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.config, args.hz, args.print_hz))
    except KeyboardInterrupt:
        print("\nTracking stopped.")
