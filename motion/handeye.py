"""Eye-to-hand calibration from a sphere held in the gripper.

The camera is fixed in the world and watches a ball clamped in the gripper. A
sphere has no orientation, so every observation is a plain 3D point
correspondence and the usual AX=XB rotation machinery is not needed.

For pose i, with the gripper at (R_i, t_i) from forward kinematics:

    R_c @ p_i + t_c == R_i @ b + t_i

where (R_c, t_c) is the camera-to-world transform being solved for, p_i is the
ball centre measured in the camera frame, and b is the ball centre in the
gripper frame -- also unknown, because where the ball sits between the jaws is
not known to millimetres.

Nine unknowns, three equations per pose. Solved by alternating two closed forms
rather than a general optimiser: given b the problem is a Procrustes fit, and
given (R_c, t_c) it is linear in b.

**The gripper orientation must vary across poses.** With a fixed orientation
R_i @ b is constant, b is indistinguishable from t_c, and the ball offset is
silently absorbed into the camera translation -- biasing every later
measurement by it. `orientation_spread_deg` measures this; `solve_eye_to_hand`
refuses a set that is too close to degenerate.
"""
import numpy as np


def kabsch(source, target, *, estimate_scale=False):
    """Best transform mapping source onto target: (R, t) or, with
    `estimate_scale`, the similarity (R, t, s) of Umeyama's method.

    A rigid fit cannot absorb a scale error, so if the camera points were built
    from a wrong assumed ball radius the mismatch surfaces entirely as residual.
    Solving for scale instead both removes that error and measures it: the
    fitted `s` times the assumed radius is the real one.
    """
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('Expected two matching (N, 3) point sets')
    if len(source) < 3:
        raise ValueError('Need at least three point correspondences')
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    centred = source-source_centre
    covariance = centred.T @ (target-target_centre)
    u, singular, vt = np.linalg.svd(covariance)
    # Flip the least-significant axis rather than accepting a reflection.
    correction = np.diag([1., 1., float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rotation = vt.T @ correction @ u.T
    if not estimate_scale:
        return rotation, target_centre-rotation@source_centre
    spread = float((centred**2).sum())
    if spread < 1e-9:
        raise ValueError('Source points coincide; scale is undefined')
    scale = float((singular*np.diag(correction)).sum()/spread)
    return rotation, target_centre-scale*rotation@source_centre, scale


def orientation_spread_deg(rotations):
    """Largest pairwise angle between gripper orientations, in degrees.

    Near zero means every pose used the same wrist orientation, which makes the
    ball offset unobservable.
    """
    rotations = np.asarray(rotations, float)
    worst = 0.
    for i in range(len(rotations)):
        for j in range(i+1, len(rotations)):
            cosine = (np.trace(rotations[i].T@rotations[j])-1)/2
            worst = max(worst, float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    return worst


def solve_eye_to_hand(camera_points, gripper_rotations, gripper_positions, *,
                      min_spread_deg=20., iterations=200, tolerance=1e-10,
                      estimate_scale=False):
    """Camera-to-world transform and the ball's offset in the gripper frame.

    Returns (rotation, translation, ball_offset, residuals, scale). `scale` is
    1.0 unless `estimate_scale`, in which case it is the factor the camera
    points had to be resized by -- multiply the assumed ball radius by it to get
    the real one.
    """
    points = np.asarray(camera_points, float)
    rotations = np.asarray(gripper_rotations, float)
    positions = np.asarray(gripper_positions, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('camera_points must be (N, 3)')
    if rotations.shape != (len(points), 3, 3) or positions.shape != points.shape:
        raise ValueError('Pose arrays must match the number of camera points')
    if len(points) < 4:
        raise ValueError('Need at least four poses; use many more in practice')
    if not (np.isfinite(points).all() and np.isfinite(rotations).all() and np.isfinite(positions).all()):
        raise ValueError('Calibration input contains non-finite values')
    spread = orientation_spread_deg(rotations)
    if spread < min_spread_deg:
        raise ValueError(
            f'Gripper orientation varies by only {spread:.1f} deg across these poses. '
            f'Below {min_spread_deg:.0f} deg the ball offset cannot be separated from the '
            'camera translation and would be absorbed into it. Re-record with the wrist '
            'rotated between poses.')

    def fit(world):
        if estimate_scale:
            return kabsch(points, world, estimate_scale=True)
        rotation, translation = kabsch(points, world)
        return rotation, translation, 1.

    stacked = rotations.reshape(-1, 3)
    offset = np.zeros(3)
    for _ in range(iterations):
        world = np.einsum('nij,j->ni', rotations, offset)+positions
        rotation, translation, scale = fit(world)
        target = (scale*points@rotation.T+translation-positions).reshape(-1)
        updated = np.linalg.lstsq(stacked, target, rcond=None)[0]
        converged = np.linalg.norm(updated-offset) < tolerance
        offset = updated
        if converged:
            break
    world = np.einsum('nij,j->ni', rotations, offset)+positions
    rotation, translation, scale = fit(world)
    residuals = np.linalg.norm(scale*points@rotation.T+translation-world, axis=1)
    return rotation, translation, offset, residuals, float(scale)


def rotation_to_quaternion(rotation):
    """(w, x, y, z) for a Viam frame `orientation` of type `quaternion`."""
    m = np.asarray(rotation, float)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace+1)*2
        q = [s/4, (m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(m[i, i]-m[j, j]-m[k, k]+1)*2
        q = [0., 0., 0., 0.]
        q[0] = (m[k, j]-m[j, k])/s
        q[i+1] = s/4
        q[j+1] = (m[j, i]+m[i, j])/s
        q[k+1] = (m[k, i]+m[i, k])/s
    q = np.array(q, float)
    return q/np.linalg.norm(q)


def viam_frame(rotation, translation, parent='world'):
    """The frame block to paste into the camera's Viam configuration."""
    w, x, y, z = rotation_to_quaternion(rotation)
    return {'parent': parent,
            'translation': {'X': float(translation[0]), 'Y': float(translation[1]),
                            'Z': float(translation[2])},
            'orientation': {'type': 'quaternion',
                            'value': {'W': float(w), 'X': float(x), 'Y': float(y), 'Z': float(z)}}}
