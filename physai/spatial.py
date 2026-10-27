"""Rotations and rigid transforms with plain NumPy (lessons 06-07; lesson 09 uses quat_to_mat and axis_angle_to_mat).

Quaternions are (w, x, y, z), scalar first, as in MuJoCo. Transforms are 4x4 arrays T_AB mapping B to A.
"""
import numpy as np


def quat_to_mat(q):
    """Unit quaternion (w, x, y, z), MuJoCo's order, to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_to_mat(axis, angle):
    """Rotation by `angle` (rad) about the unit vector `axis` (Rodrigues' formula)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


# ---------- lesson 06: quaternions, Euler angles, slerp ----------

def quat_mul(a, b):
    """Hamilton product a * b: the rotation b followed by a (as matrices, R(a) @ R(b))."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    """Conjugate (w, -x, -y, -z): the inverse rotation of a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    """Rotate the 3-vector v by the unit quaternion q: the vector part of q * (0, v) * conj(q)."""
    return quat_mul(quat_mul(q, np.r_[0.0, v]), quat_conj(q))[1:]


def axis_angle_to_quat(axis, angle):
    """Unit quaternion for a rotation by `angle` (rad) about the unit vector `axis`."""
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * np.asarray(axis, float)]


def mat_to_quat(R):
    """Rotation matrix to unit quaternion with w >= 0 (Shepperd: divide by the largest of 4w^2, 4x^2, 4y^2, 4z^2)."""
    t = np.trace(R)
    cands = [t, R[0, 0] - R[1, 1] - R[2, 2], R[1, 1] - R[0, 0] - R[2, 2], R[2, 2] - R[0, 0] - R[1, 1]]
    i = int(np.argmax(cands))
    s = 2 * np.sqrt(1 + cands[i])                      # 4 times the largest component
    if i == 0:
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def euler_zyx_to_mat(yaw, pitch, roll):
    """Intrinsic z-y'-x'' Euler angles (rad) to a rotation matrix: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_euler_zyx(R):
    """Rotation matrix to (yaw, pitch, roll), pitch in [-pi/2, pi/2]; at pitch = +-90 deg yaw and roll are not unique."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    return np.arctan2(R[1, 0], R[0, 0]), pitch, np.arctan2(R[2, 1], R[2, 2])


def euler_rate_matrix(pitch, roll):
    """E with body angular velocity w_b = E @ (roll', pitch', yaw') for z-y'-x'' angles; det E = cos(pitch)."""
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[1, 0, -sp], [0, cr, sr * cp], [0, -sr, cr * cp]])


def quat_angle(q0, q1):
    """Angle (rad) of the rotation that takes orientation q0 to q1, in [0, pi]; q and -q give the same answer."""
    return 2 * np.arccos(np.clip(abs(q0 @ q1), 0, 1))


def slerp(q0, q1, t, shortest=True):
    """Spherical linear interpolation; with shortest=True flip q1 when q0 . q1 < 0 (q1 and -q1 are one rotation)."""
    dot = q0 @ q1
    if shortest and dot < 0:
        q1, dot = -q1, -dot
    om = np.arccos(np.clip(dot, -1, 1))
    if om < 1e-9:
        return q0.copy()
    return (np.sin((1 - t) * om) * q0 + np.sin(t * om) * q1) / np.sin(om)


# ---------- lesson 07: rigid transforms and frame trees ----------

def transform(R, t):
    """4x4 homogeneous transform from a 3x3 rotation R and a translation t."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R).reshape(3, 3), t
    return T


def transform_inv(T):
    """Closed-form inverse [R t; 0 1]^-1 = [R^T  -R^T t; 0 1]."""
    R, t = T[:3, :3], T[:3, 3]
    return transform(R.T, -R.T @ t)


def transform_point(T, p):
    """Apply T to a point: R p + t (the homogeneous 1 is implicit)."""
    return T[:3, :3] @ p + T[:3, 3]


def pose_to_transform(pos, quat):
    """MuJoCo's (pos, quat) pair, quat scalar first, as a 4x4 transform."""
    return transform(quat_to_mat(quat), pos)


def frame_tree(m, qpos):
    """World pose (4x4) of every body, walking MuJoCo's tree from static model data and qpos alone."""
    T = [np.eye(4)]                                    # body 0 is the world
    for b in range(1, m.nbody):
        Tb = T[m.body_parentid[b]] @ pose_to_transform(m.body_pos[b], m.body_quat[b])   # pose at qpos0
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            a, kind = m.jnt_qposadr[j], m.jnt_type[j]
            if kind == 0:                              # free: qpos holds the world pose itself
                Tb = pose_to_transform(qpos[a:a + 3], qpos[a + 3:a + 7])
                continue
            anchor = transform(np.eye(3), m.jnt_pos[j])
            if kind == 1:                              # ball: rotate by the quaternion about the anchor
                move = pose_to_transform(np.zeros(3), qpos[a:a + 4])
            elif kind == 2:                            # slide: translate along the axis
                move = transform(np.eye(3), m.jnt_axis[j] * (qpos[a] - m.qpos0[a]))
            else:                                      # hinge: rotate about the axis through the anchor
                move = transform(axis_angle_to_mat(m.jnt_axis[j], qpos[a] - m.qpos0[a]), np.zeros(3))
            Tb = Tb @ anchor @ move @ transform_inv(anchor)
        T.append(Tb)
    return T
