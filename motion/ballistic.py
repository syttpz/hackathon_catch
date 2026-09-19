"""Ballistic tracking and interception for a thrown ball.

Free flight in a Z-up world: p(t) = p0 + v0*t + g*t^2/2, with g known, so only
p0 and v0 are fitted and the problem stays linear. Drag, spin and bounces are
not modelled; for a light ball over a short indoor throw the error from that is
small compared with the range noise, but it is an approximation, not physics.

Interception is a feasibility question, not just a geometry one. A throw is
airborne for well under a second, so a predicted point is only usable if the
arm can actually get there in the time remaining. `ArmTiming` carries the
measured cost of a move; `intercept` refuses anything the arm cannot reach in
time rather than committing to a target it will arrive at late.
"""
from dataclasses import dataclass
import numpy as np

GRAVITY_MM_S2 = np.array([0., 0., -9810.])


@dataclass(frozen=True)
class ArmTiming:
    """Cost of a Cartesian move, from `motion.measure_timing`.

    Two models. `acceleration_mm_s2`, when given, uses the triangular profile
    `latency + 2*sqrt(d/a)`: on this arm that fit the measurements to 0.8 ms
    against 13.4 ms for constant velocity, and returned a latency of
    essentially zero. Short corrective moves never reach top speed -- the
    incremental rate climbed 298, 420, 566 mm/s across 25 to 150 mm -- so the
    constant-velocity form understates every short move and overstates the long
    ones. It is kept only for configurations that have not been remeasured.
    """
    latency_s: float
    speed_mm_s: float
    acceleration_mm_s2: float = None

    def reach_seconds(self, distance_mm):
        if self.latency_s < 0:
            raise ValueError('Arm latency cannot be negative')
        distance_mm = max(float(distance_mm), 0.)
        if self.acceleration_mm_s2:
            if self.acceleration_mm_s2 <= 0:
                raise ValueError('Arm acceleration must be positive')
            return self.latency_s+2*np.sqrt(distance_mm/self.acceleration_mm_s2)
        if self.speed_mm_s <= 0:
            raise ValueError('Arm speed must be positive')
        return self.latency_s+distance_mm/self.speed_mm_s


@dataclass(frozen=True)
class Flight:
    """A fitted trajectory. Times are absolute, in the observation clock."""
    origin: np.ndarray            # position at reference_time
    velocity: np.ndarray          # velocity at reference_time
    reference_time: float
    residual_mm: float
    samples: int
    gravity: np.ndarray = None

    def at(self, when):
        g = GRAVITY_MM_S2 if self.gravity is None else self.gravity
        dt = float(when)-self.reference_time
        return self.origin+self.velocity*dt+.5*g*dt*dt, self.velocity+g*dt

    def apex_time(self):
        """When the vertical velocity crosses zero; may be in the past."""
        g = GRAVITY_MM_S2 if self.gravity is None else self.gravity
        if abs(g[2]) < 1e-9:
            return None
        return self.reference_time-self.velocity[2]/g[2]


class BallisticFit:
    """Accumulate timestamped world points and fit free flight through them.

    Gaps, backward timestamps and implausible jumps reset the fit: they mean
    the detector lost the ball or latched onto something else, and carrying
    those samples forward would poison the prediction.
    """

    def __init__(self, *, gravity=GRAVITY_MM_S2, max_samples=40, min_samples=5,
                 max_gap_s=.15, max_jump_mm=900., max_residual_mm=30., min_span_s=.08,
                 release_speed_mm_s=None, lateral_sigma_mm=None, range_sigma_mm=None):
        self.gravity = np.asarray(gravity, float)
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.max_gap_s = max_gap_s
        self.max_jump_mm = max_jump_mm
        self.max_residual_mm = max_residual_mm
        self.min_span_s = min_span_s
        self.release_speed_mm_s = release_speed_mm_s
        self.lateral_sigma_mm = lateral_sigma_mm
        self.range_sigma_mm = range_sigma_mm
        self.camera_origin = None
        self.samples = []

    def reset(self):
        self.samples = []

    def add(self, timestamp, point, camera_origin=None):
        """Add one observation; return a Flight once the fit is trustworthy.

        `camera_origin` enables weighting: a monocular observation pins the two
        directions across the image far better than the one along the ray, and
        on this rig by about sixty to one. Treating all three alike lets the
        range noise drive the fit, which is what made the estimated speed swing
        between 2400 and 5250 mm/s on identical throws.
        """
        point = np.asarray(point, float)
        if camera_origin is not None:
            self.camera_origin = np.asarray(camera_origin, float)
        if not np.isfinite(timestamp) or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid ballistic observation')
        if self.samples:
            gap = timestamp-self.samples[-1][0]
            if gap <= 0:
                return None
            if gap > self.max_gap_s or np.linalg.norm(point-self.samples[-1][1]) > self.max_jump_mm:
                self.reset()
            elif self.release_speed_mm_s is not None:
                # A ball held ready and then thrown produces one window holding
                # both, and no parabola fits that: the still part drags the
                # fitted speed down and the predicted arrival comes out late.
                # A jump in speed is the release, so start the fit from there.
                speed = float(np.linalg.norm(point-self.samples[-1][1])/gap)
                previous = (float(np.linalg.norm(self.samples[-1][1]-self.samples[-2][1])
                                  / (self.samples[-1][0]-self.samples[-2][0]))
                            if len(self.samples) > 1 else 0.)
                if speed > self.release_speed_mm_s > previous*2:
                    self.samples = self.samples[-1:]
        self.samples.append((timestamp, point))
        del self.samples[:-self.max_samples]
        return self.solve()

    def solve(self):
        if len(self.samples) < self.min_samples:
            return None
        times = np.array([s[0] for s in self.samples])
        points = np.array([s[1] for s in self.samples])
        reference = times[-1]
        dt = times-reference
        if np.ptp(dt) < self.min_span_s:
            return None
        # g is known, so remove it and the remainder is linear in (p0, v0).
        target = points-.5*np.outer(dt*dt, self.gravity)
        design = np.column_stack((np.ones(len(dt)), dt))
        weighting = (self.camera_origin is not None and self.lateral_sigma_mm
                     and self.range_sigma_mm)
        if weighting:
            # Whiten each observation along its own viewing ray: rows across the
            # image get their full weight, the row along the ray gets much less.
            rows, values = [], []
            for row, (coefficients, measured) in enumerate(zip(design, target)):
                ray = points[row]-self.camera_origin
                length = np.linalg.norm(ray)
                if length < 1e-6:
                    weighting = False
                    break
                ray = ray/length
                along = np.outer(ray, ray)
                whiten = (np.eye(3)-along)/self.lateral_sigma_mm+along/self.range_sigma_mm
                for axis in range(3):
                    rows.append(np.concatenate([coefficients[i]*whiten[axis]
                                                for i in range(2)]))
                    values.append(float(whiten[axis]@measured))
        if weighting:
            estimate, *_ = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)
            solution = estimate.reshape(2, 3)
        else:
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        origin, velocity = solution
        residual = float(np.sqrt(np.mean(np.sum((design@solution-target)**2, axis=1))))
        if residual > self.max_residual_mm:
            return None
        return Flight(origin, velocity, float(reference), residual, len(self.samples),
                      self.gravity)


def bowl_orientation(velocity, upright):
    """Rotation whose tool +Z is the bowl's opening axis.

    Aiming the mouth straight back along the incoming velocity lets the ball fly
    in along the axis, but a bowl tilted that far spills what it catches.
    Blending toward world up trades entry alignment for retention: `upright` 0
    faces the ball, 1 faces straight up.
    """
    velocity = np.asarray(velocity, float)
    speed = float(np.linalg.norm(velocity))
    up = np.array([0., 0., 1.])
    if not 0 <= upright <= 1:
        raise ValueError('upright must be between 0 and 1')
    axis = up if speed < 1e-6 else (1-upright)*(-velocity/speed)+upright*up
    length = float(np.linalg.norm(axis))
    axis = up if length < 1e-6 else axis/length
    reference = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
    across = np.cross(reference, axis)
    across = across/np.linalg.norm(across)
    return np.column_stack((across, np.cross(axis, across), axis))


def catch_pose(point, velocity, config):
    """Flange position and orientation that place the catcher's mouth at `point`.

    The bowl sits `tool_offset_mm` beyond the flange along the tool axis, so the
    flange must stand back by that much; reachability has to be judged on the
    flange, which is what the arm actually positions.
    """
    point = np.asarray(point, float)
    offset = float(config.get('tool_offset_mm', 0.))
    if config.get('catcher', 'gripper') == 'bowl':
        rotation = bowl_orientation(velocity, config['bowl_upright'])
    elif 'catch_orientation' in config:
        direction = np.asarray(config['catch_orientation'], float)[:3]
        rotation = bowl_orientation(-direction/np.linalg.norm(direction), 0.)
    else:
        # No tool geometry configured: the catch point is the flange point.
        rotation = np.eye(3)
    return point-rotation@np.array([0., 0., offset]), rotation


def reachable(point, config):
    """Inside the configured catch box and the arm's radial envelope."""
    point = np.asarray(point, float)
    lower, upper = np.asarray(config['catch_box_mm'], float)
    radius = float(np.linalg.norm(point))
    return bool((point >= lower).all() and (point <= upper).all()
                and config['min_reach_mm'] <= radius <= config['max_reach_mm'])


@dataclass(frozen=True)
class Intercept:
    point: np.ndarray
    arrival: float          # absolute time the ball reaches `point`
    speed_mm_s: float       # ball speed there; slower is an easier catch
    slack_s: float          # spare time beyond what the move needs
    distance_mm: float


def intercept(flight, flange_position, config, timing, now, *, prefer='slowest'):
    """Best reachable point on the trajectory the arm can still get to in time.

    Returns (Intercept, None) or (None, reason). Candidates must be reachable,
    and must leave `reach_seconds` plus the closing lead and a margin before the
    ball arrives. Among those, 'slowest' picks where the ball is moving least --
    near apex, much the easiest place to catch -- and 'earliest' maximises the
    time buffer instead.
    """
    if prefer not in ('slowest', 'earliest'):
        raise ValueError("prefer must be 'slowest' or 'earliest'")
    flange_position = np.asarray(flange_position, float)
    step = float(config['search_step_s'])
    horizon = float(config['max_prediction_s'])
    # A bowl catches passively: there is no jaw to time, so the closing lead
    # drops out of the budget entirely and only the arrival margin remains.
    closing = 0. if config.get('catcher', 'gripper') == 'bowl' else float(config['close_lead_s'])
    lead = closing+float(config['arrival_margin_s'])
    if step <= 0 or horizon <= 0:
        raise ValueError('search_step_s and max_prediction_s must be positive')

    candidates = []
    unreachable = late = 0
    for offset in np.arange(step, horizon+1e-9, step):
        when = now+offset
        point, velocity = flight.at(when)
        # Reachability applies to the flange, not to the catcher's mouth.
        flange_target, _ = catch_pose(point, velocity, config)
        if not reachable(flange_target, config):
            unreachable += 1
            continue
        distance = float(np.linalg.norm(flange_target-flange_position))
        slack = offset-timing.reach_seconds(distance)-lead
        if slack < 0:
            late += 1
            continue
        candidates.append(Intercept(point, when, float(np.linalg.norm(velocity)), slack, distance))
    if not candidates:
        # Report the actionable cause first: points that were in the box but too
        # late say the throw is catchable and the arm is the limit, which is a
        # different problem from a throw that never comes within reach.
        if late:
            return None, (f'NO CATCH: {late} point(s) on the trajectory are in reach but the arm '
                          'cannot get there in time')
        if unreachable:
            return None, 'NO CATCH: the trajectory never enters the reachable catch box'
        return None, 'NO CATCH: no predicted point within the horizon'
    if prefer == 'slowest':
        best = min(candidates, key=lambda c: (c.speed_mm_s, c.arrival))
    else:
        best = min(candidates, key=lambda c: c.arrival)
    return best, None


def required_lead(distance_mm, config, timing):
    """Seconds needed to commit to a target that far away."""
    closing = 0. if config.get('catcher', 'gripper') == 'bowl' else float(config['close_lead_s'])
    return timing.reach_seconds(distance_mm)+closing+float(config['arrival_margin_s'])


def plane_crossing(flight, plane_z, not_before):
    """When the ball falls through a horizontal plane, and where.

    Returns (time, position, velocity) for the first DESCENDING crossing after
    `not_before`, or None. Descending is the one that matters: a bowl catches a
    ball dropping into it, while the ascending crossing of a lob would be met by
    the rim on its way up.

    Fixing the catch height turns interception into one quadratic instead of a
    search: z(t) = z0 + vz*t + g*t^2/2 = plane_z.
    """
    gravity = float((GRAVITY_MM_S2 if flight.gravity is None else flight.gravity)[2])
    if gravity >= 0:
        raise ValueError('Expected gravity to pull down in a Z-up world')
    origin_z = float(flight.origin[2])
    speed_z = float(flight.velocity[2])
    # (g/2) t^2 + vz t + (z0 - plane) = 0, with t measured from reference_time.
    discriminant = speed_z*speed_z-2*gravity*(origin_z-plane_z)
    if discriminant < 0:
        return None                      # the arc never reaches that height
    root = np.sqrt(discriminant)
    # The descending crossing is the later root; g < 0 flips the usual ordering.
    for candidate in sorted(((-speed_z+root)/gravity, (-speed_z-root)/gravity)):
        when = flight.reference_time+candidate
        if when < not_before:
            continue
        position, velocity = flight.at(when)
        if velocity[2] < 0:
            return when, position, velocity
    return None


def plane_intercept(flight, flange_position, bowl_offset_world, config, timing, now):
    """Interception restricted to a fixed catch height: only XY has to move.

    The wrist orientation is held at whatever was taught, so `bowl_offset_world`
    -- the flange-to-bowl-mouth vector -- is constant and the mouth can be placed
    by translation alone. It is derived from the machine's own gripper frame
    rather than guessed from the flange: a 89 mm error there put the catch plane
    above every throw and refused them all.

    Returns (Intercept, None) or (None, reason).
    """
    flange_position = np.asarray(flange_position, float)
    bowl_offset_world = np.asarray(bowl_offset_world, float)
    crossing = plane_crossing(flight, float(config['catch_plane_z_mm']), now)
    if crossing is None:
        return None, 'NO CATCH: the arc never descends through the catch plane'
    arrival, point, velocity = crossing
    flange_target = point-bowl_offset_world
    if not reachable(flange_target, config):
        return None, (f'NO CATCH: crossing at ({point[0]:.0f}, {point[1]:.0f}) mm is outside the '
                      'reachable catch box')
    # Only XY is commanded; the height never changes, so it costs no travel.
    distance = float(np.linalg.norm((flange_target-flange_position)[:2]))
    slack = (arrival-now)-timing.reach_seconds(distance)-float(config['arrival_margin_s'])
    if config.get('rough_attempt', False):
        anchor = np.asarray(config.get('rough_anchor_mm', flange_position), float)
        excursion = float(np.linalg.norm((flange_target-anchor)[:2]))
        if (not np.isfinite(point).all() or not np.isfinite(distance)
                or not np.isfinite(excursion) or excursion > 150 or distance > 150):
            return None, 'NO CATCH: rough attempt exceeds 150mm from start'
        if arrival-now < .05:
            return None, 'NO CATCH: rough attempt needs at least 0.05s lead'
    if slack < 0 and not config.get('rough_attempt', False):
        # Spell out the margin: without it this reads as a contradiction whenever
        # the arrival time is the larger of the two numbers.
        return None, (f'NO CATCH: needs {timing.reach_seconds(distance):.2f}s to move '
                      f'{distance:.0f} mm plus {float(config["arrival_margin_s"]):.2f}s margin, '
                      f'but the ball arrives in {arrival-now:.2f}s')
    return Intercept(point, arrival, float(np.linalg.norm(velocity)), slack, distance), None
