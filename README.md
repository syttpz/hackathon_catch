# Hackathon Catch

Hackathon Catch is a vision-guided robotic ball-catching system built with a
Viam-controlled arm, a wrist camera, and a fixed side camera. It detects a
lobbed ball, estimates its 3D trajectory, predicts where it will cross a fixed
catch plane, and moves a bowl into position for the catch.

This project was created for the **Viam Hackathon** and is a branched,
catch-focused version of
[`gracexu24/viamhackathon26`](https://github.com/gracexu24/viamhackathon26).
This repository isolates the complete catching workflow, its calibration data,
launch scripts, and hardware-independent tests.

The controller is safe by default: normal runs are preview-only and do not move
the robot. Physical movement must be explicitly enabled with `--execute`.

## Demo

<!-- Replace VIDEO_URL with the final YouTube, Vimeo, or Google Drive URL. -->

> **Video demo coming soon**
>
> The final demo will show camera tracking, trajectory prediction, the planned
> intercept, and the arm completing a catch.

<!-- Optional thumbnail once the video is ready:
[![Hackathon Catch demo](docs/demo-thumbnail.jpg)](VIDEO_URL)
-->

## How it works

The system uses two cameras and a Viam-controlled robot arm to follow a thrown
ball and move a bowl underneath it.

1. The wrist camera and fixed side camera watch for the colored ball.
2. The software combines the camera detections with the calibrated camera
   positions to estimate where the ball is in the robot's world coordinates.
3. Several observations are used to estimate the ball's direction, speed, and
   curved flight path under gravity.
4. The software predicts when and where the ball will descend through the
   bowl's fixed catch height.
5. Before moving, it checks that the prediction is stable, the target is inside
   the arm's configured workspace, and the arm has enough time to reach it.
6. In preview mode, it prints the predicted catch without moving. When started
   with `--execute`, it moves the bowl to the predicted location and then
   returns the arm to its default pose.

Keeping the bowl at one fixed height turns the catch into a short side-to-side
and forward-to-back movement instead of a full 3D motion. This reduces planning
time and gives the arm a better chance of reaching the target before the ball.

The primary workflow is implemented in `motion/catch_plane.py`. Supporting
modules handle ball detection, side-camera calibration, trajectory fitting,
prediction checks, arm timing, and returning the robot safely to its starting
pose. The default run is read-only; physical movement requires `--execute`.

## Hardware and services

- Viam-compatible robot arm (`arm`)
- Bowl or basket mounted to the end effector
- Wrist RGB-D camera (`cam`)
- Fixed side camera (`cam2`)
- Viam Motion service (`builtin`)
- Python 3.10 or newer

Resource names and physical limits are configured in
[`catch_plane.config.json`](catch_plane.config.json).

## Quick start

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On the robot, the supplied launch script expects this repository at
`/opt/viam/trajectory-local` and uses the cached Viam machine configuration.

Start with a preview run. It tracks and predicts but does not move the arm:

```bash
cd /opt/viam/trajectory-local
./catch-plane.sh
```

After checking the camera feeds, calibration, workspace limits, default pose,
and emergency stop, enable the complete catch workflow:

```bash
cd /opt/viam/trajectory-local
./catch-plane.sh --execute
```

The process homes the arm, prints `READY`, waits for a valid throw, performs the
catch attempt, and returns to the default pose. Stop it at any time with
`Ctrl+C`.

Useful alternatives:

```bash
# Run for 30 seconds in preview mode
./catch-plane.sh --duration 30

# Use the wrist camera instead of the side camera as the trajectory source
./catch-plane.sh --trajectory-source wrist

# Run the separate catch-and-grip workflow in preview mode
./catch-throw.sh
```

## Configuration and calibration

The main runtime settings are in `catch_plane.config.json`. They are grouped by
the stage that consumes them:

| Area | Important settings | Used for |
| --- | --- | --- |
| Viam resources | `camera`, `arm`, `gripper`, `motion`, `world_frame` | Resolving hardware and coordinate frames |
| Ball detector | `hue_min/max`, `sat_min`, `val_min`, `min_area_px`, `min_radius_px` | Separating the ball from the image background |
| 3D localization | `ball_radius_mm`, `range_source`, `min/max_distance_mm` | Converting a pixel/radius or aligned depth into range |
| Flight fit | `min_samples`, `max_samples`, `max_gap_s`, `min_span_s`, `max_residual_mm` | Deciding when a trajectory is trustworthy |
| Catch geometry | `catch_plane_z_mm`, `bowl_offset_gripper_mm`, `bowl_axis_flange` | Mapping the ball crossing to the required flange pose |
| Workspace | `throw_volume_mm`, `catch_box_mm`, `min/max_reach_mm` | Rejecting impossible or unsafe observations and targets |
| Timing | `arm_latency_s`, `arm_speed_mm_s`, `arm_acceleration_mm_s2`, `arrival_margin_s` | Determining whether the arm can arrive in time |
| Stability | `stability_mm`, `stability_arrival_s`, `stability_samples`, `stability_span_s` | Requiring agreement across predictions before commit |
| Side camera | `side_calibration`, side HSV thresholds | Loading and validating the fixed-camera model |

Calibration helpers are included for the bowl and side camera:

```bash
./calibrate-bowl.sh
./calibrate-cam2.sh
./calibrate-cam2-pnp.sh
./measure-timing.sh
```

Calibration values are specific to the physical setup. Do not copy bounds,
offsets, or poses to another robot without measuring and validating them.

The side-camera calibration file records camera intrinsics, its world pose, and
quality information. Execution using the side or stereo source is blocked when
that calibration does not pass its quality checks. `measure-timing.sh` performs
real arm moves only when explicitly executed and fits the latency/acceleration
model used by the interception gate.

## Deployment model

The Python process runs on the same machine as `viam-server`. This avoids a
cloud round trip in the frame loop while still using Viam resources and the
machine's configured frame system.

`catch-plane.sh` locates the cached machine configuration, selects the project
virtual environment, and starts `python -m motion.catch_plane`. Secrets remain
in Viam's cached configuration and are not stored in this repository.

## Tests

Run the headless test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Repository layout

The repository includes the current fixed-plane catcher as well as calibration
tools and earlier experiments that led to it.

### Primary catch path

| Path | Responsibility |
| --- | --- |
| `catch-plane.sh` | Robot-side launcher for the primary workflow |
| `catch_plane.config.json` | Physical setup, detector, fit, timing, and safety parameters |
| `motion/catch_plane.py` | Runtime orchestration, wrist localization, gating, movement, and homing |
| `motion/catch_side.py` | Independent `cam2` loop and side-calibration trust checks |
| `motion/ballistic.py` | Free-flight fit, fixed-plane crossing, timing, and reachability |
| `motion/stereo_tracking.py` | Capture-time pairing and triangulation of two camera rays |
| `motion/rolling_catch.py` | Prediction consensus and one-shot catch safety helpers |
| `motion/rough_cycle.py` | Per-throw rearming and optional rough-mode evidence handoff |
| `motion/live_camera_pose.py` | Viam pose conversion and capture-time pose interpolation |
| `motion/trajectory_local.py` | Local credentials, image decoding, and camera utilities |

### Calibration and diagnostics

| Path | Responsibility |
| --- | --- |
| `motion/calibrate_cam2_pnp.py` | Solves the fixed camera pose from pixel/world correspondences |
| `motion/calibrate_cam2.py` | Captures and solves ball-based side-camera calibration samples |
| `motion/calibrate_bowl.py` | Estimates the bowl-mouth offset from the gripper frame |
| `motion/handeye.py` | Eye-to-hand rigid-transform math and degeneracy checks |
| `motion/measure_timing.py` | Measures arm move latency, speed, and acceleration |
| `motion/red_probe.py` | Read-only live color-threshold diagnostic |
| `vision/live_ball.py` | Read-only RGB/depth visualization and annotation |
| `vision/viam_pipeline.py` | Lists and probes configured Viam vision resources |
| `calibration_data/` | Captured samples and fitted geometry retained for reproducibility |
| `cam2_catch_calibration.json` | Active fixed-side-camera calibration and quality metadata |

### Supporting workflows and experiments

| Path | Responsibility |
| --- | --- |
| `motion/catch_throw.py` | Alternative trajectory catch that can coordinate the gripper |
| `motion/rolling_preview.py` | Rolling-object/table-plane interception workflow |
| `motion/track_red.py` | Wrist-camera visual servoing toward a red target |
| `motion/viam_ball_catch.py` | Earlier Viam Vision-service ball-catching approach |
| `motion/viam_stationary_grab.py` | Stationary ball localization and pickup workflow |
| `motion/can_tracking.py`, `can_follower.py` | Red-can detection and following experiments |
| `motion/table_plane.py`, `rolling_intercept.py` | Table fitting and 2D rolling intercept geometry |
| `ball_tracking.py` | Reusable RGB/depth observation and simple trajectory primitives |
| `vision/viam_ball.py` | Read-only Viam detector/segmenter localization helpers |

These supporting workflows are not imported wholesale by the primary catcher;
they remain useful as diagnostics, calibration references, and documented
iterations of the system.

### Tests

The `tests/` directory mirrors the architecture. It covers detection,
ballistics, plane intersections, camera calibration, stereo geometry,
prediction stability, default-pose return, visual following, stationary pickup,
and mocked end-to-end control flow. Hardware calls are mocked so the suite can
run without moving a robot.

## Safety

Keep the robot workspace clear and maintain access to the emergency stop. Run
preview mode first, supervise every execution, and confirm that only one process
has control of the arm. The software checks timing, prediction quality, and
configured workspace limits, but those checks do not replace physical safety
validation.
