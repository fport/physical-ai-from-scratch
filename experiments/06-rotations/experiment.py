"""Lesson 06 · Rotations: matrices, Euler angles, axis-angle, quaternions.

Part A  Sanity checks: orthonormality, composition order, conversions with mujoco.mju_*.
Part B  Sweep pitch toward 90 deg -> condition number of the ZYX Euler-rate map.
Part C  Sweep start pitch -> deviation of naive Euler-angle interpolation from slerp.
Break   Euler rates needed for a 1 rad/s body rotation; slerp without the sign check.

Run:  python experiment.py      (pure NumPy + MuJoCo math, < 5 s)
"""
import platform

import mujoco
import numpy as np

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | Python {platform.python_version()} "
      f"| {platform.machine()}")
np.set_printoptions(precision=4, suppress=True)
DEG = np.pi / 180


# ---------- basic rotations (active: they rotate vectors, columns = rotated axes) ----------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyx_to_R(yaw, pitch, roll):
    """R = Rz(yaw) Ry(pitch) Rx(roll): intrinsic z-y'-x'' (aerospace yaw-pitch-roll)."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def R_to_euler_zyx(R):
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([yaw, pitch, roll])


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues(axis, angle):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = skew(k)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def angle_between(Ra, Rb):
    """Geodesic angle (rad) of the relative rotation Ra^T Rb."""
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.arccos(np.clip(c, -1, 1))


def quat_to_R(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def R_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def slerp(q0, q1, t, sign_check=True):
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if sign_check and dot < 0:          # q and -q are the same rotation: take the short arc
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1, 1)
    th = np.arccos(dot)
    if th < 1e-9:
        return q0
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


# ---------- Part A: sanity checks ----------
print("\n== A. Rotation matrix facts and conversions (MuJoCo quaternions are [w, x, y, z]) ==")
free = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body>"
                                      "</worldbody></mujoco>")
fd = mujoco.MjData(free)
print(f"one free body: nq = {free.nq} (position 3 + quaternion 4), nv = {free.nv} (linear 3 + angular 3); "
      f"default qpos quaternion = {fd.qpos[3:]}")
rng = np.random.default_rng(0)
errs_orth, errs_det = [], []
R = np.eye(3)
for _ in range(1000):                  # compose 1000 random rotations
    R = R @ euler_zyx_to_R(*rng.uniform(-np.pi, np.pi, 3))
    errs_orth.append(np.abs(R.T @ R - np.eye(3)).max())
    errs_det.append(abs(np.linalg.det(R) - 1))
print(f"after 1000 compositions: max |R^T R - I| = {max(errs_orth):.1e}, max |det R - 1| = {max(errs_det):.1e}")

A, B = Rz(90 * DEG), Rx(90 * DEG)
print(f"order matters: angle(Rz90*Rx90, Rx90*Rz90) = {angle_between(A @ B, B @ A) / DEG:.2f} deg")

p = np.array([1.0, 0.0, 0.0])
print(f"active  Rz(90) p          = {Rz(90 * DEG) @ p}   (the vector is rotated)")
print(f"passive Rz(90)^T p        = {Rz(90 * DEG).T @ p}   (same vector, coordinates in a frame rotated by +90)")

conv = {"euler->quat (mju_euler2Quat 'zyx') vs matrix": [],
        "axis-angle (mju_axisAngle2Quat) vs Rodrigues": [],
        "mju_mat2Quat -> mju_quat2Mat round trip": [],
        "q and -q give the same matrix": []}
for _ in range(1000):
    ypr = rng.uniform([-np.pi, -np.pi / 2, -np.pi], [np.pi, np.pi / 2, np.pi])
    Rm = euler_zyx_to_R(*ypr)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, ypr, "zyx")          # lowercase = intrinsic
    conv["euler->quat (mju_euler2Quat 'zyx') vs matrix"].append(np.abs(quat_to_R(q) - Rm).max())
    ax, ang = rng.normal(size=3), rng.uniform(0, np.pi)
    qa = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qa, ax / np.linalg.norm(ax), ang)
    conv["axis-angle (mju_axisAngle2Quat) vs Rodrigues"].append(np.abs(quat_to_R(qa) - rodrigues(ax, ang)).max())
    conv["mju_mat2Quat -> mju_quat2Mat round trip"].append(np.abs(quat_to_R(R_to_quat(Rm)) - Rm).max())
    conv["q and -q give the same matrix"].append(np.abs(quat_to_R(-q) - quat_to_R(q)).max())
print(f"{'check (1000 random rotations)':48s} {'max abs error':>14s}")
for k, v in conv.items():
    print(f"{k:48s} {max(v):14.1e}")


# ---------- Part B: Euler-rate map and gimbal lock ----------
def E_body(pitch, roll):
    """Body angular velocity w_b = E(pitch, roll) @ [roll_dot, pitch_dot, yaw_dot] for R = Rz Ry Rx."""
    sp, cp, sr, cr = np.sin(pitch), np.cos(pitch), np.sin(roll), np.cos(roll)
    return np.array([[1, 0, -sp],
                     [0, cr, sr * cp],
                     [0, -sr, cr * cp]])


def E_numeric(yaw, pitch, roll, h=1e-6):
    """Same map by central differences of R: w_b = vee(R^T dR/dt)."""
    ang = np.array([yaw, pitch, roll])
    cols = []
    for idx in (2, 1, 0):                      # roll, pitch, yaw columns
        d = np.zeros(3)
        d[idx] = h
        Rp, Rm = euler_zyx_to_R(*(ang + d)), euler_zyx_to_R(*(ang - d))
        W = euler_zyx_to_R(*ang).T @ (Rp - Rm) / (2 * h)
        cols.append([W[2, 1], W[0, 2], W[1, 0]])
    return np.array(cols).T


print("\n== B. Sweep pitch -> Euler-rate map w_b = E(pitch, roll) * [roll', pitch', yaw'] (roll = 30 deg) ==")
print(f"{'pitch (deg)':>11s} {'det E':>10s} {'sigma_min':>10s} {'cond(E)':>12s} {'|E - E_fd|':>11s}")
roll = 30 * DEG
for pd in [0, 30, 60, 80, 85, 89, 89.9, 89.99, 90]:
    E = E_body(pd * DEG, roll)
    s = np.linalg.svd(E, compute_uv=False)
    cond = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
    fd = np.abs(E - E_numeric(0.4, pd * DEG, roll)).max()
    print(f"{pd:11.2f} {np.linalg.det(E):10.2e} {s[-1]:10.2e} {cond:12.3e} {fd:11.1e}")


# ---------- Part C: interpolation, naive Euler vs slerp ----------
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interp_stats(pitch0, n_pairs=300, step_deg=60, n_t=101, seed=1):
    """Start orientations with a given pitch; goal = start rotated 60 deg about a random axis."""
    rg = np.random.default_rng(seed)
    ts = np.linspace(0, 1, n_t)
    max_dev, len_ratio = [], []
    for _ in range(n_pairs):
        R0 = euler_zyx_to_R(rg.uniform(-np.pi, np.pi), pitch0 * DEG, rg.uniform(-np.pi, np.pi))
        R1 = R0 @ rodrigues(rg.normal(size=3), step_deg * DEG)
        e0, e1 = R_to_euler_zyx(R0), R_to_euler_zyx(R1)
        de = wrap(e1 - e0)                         # naive but fair: shortest wrap per angle
        q0, q1 = R_to_quat(R0), R_to_quat(R1)
        dev, path = 0.0, 0.0
        prev = R0
        for t in ts:
            Re = euler_zyx_to_R(*(e0 + t * de))
            Rs = quat_to_R(slerp(q0, q1, t))
            dev = max(dev, angle_between(Re, Rs))
            path += angle_between(prev, Re)
            prev = Re
        max_dev.append(dev / DEG)
        len_ratio.append(path / (step_deg * DEG))
    return np.array(max_dev), np.array(len_ratio)


print("\n== C. Sweep start pitch -> naive Euler interpolation vs slerp (300 pairs, each a 60 deg rotation) ==")
print(f"{'start pitch':>11s} {'median dev':>11s} {'p95 dev':>9s} {'max dev':>9s} "
      f"{'median path/60':>15s} {'max path/60':>12s} {'dev>90':>7s}   (angles in deg)")
for p0 in [0, 30, 60, 75, 85, 89]:
    dev, ratio = interp_stats(p0)
    print(f"{p0:11d} {np.median(dev):11.2f} {np.percentile(dev, 95):9.2f} {dev.max():9.2f} "
          f"{np.median(ratio):15.3f} {ratio.max():12.3f} {int((dev > 90).sum()):7d}")


# ---------- Break it ----------
print("\n== Break it 1. Euler rates needed for a 1 rad/s body rotation about the 'lost' axis (roll = 30 deg) ==")
print(f"{'pitch (deg)':>11s} {'|euler rate| (rad/s)':>21s}")
for pd in [0, 60, 85, 89, 89.9, 89.99]:
    E = E_body(pd * DEG, roll)
    w = np.array([0, np.sin(roll), np.cos(roll)])    # direction E loses at pitch = 90
    rates = np.linalg.solve(E, w)
    print(f"{pd:11.2f} {np.linalg.norm(rates):21.1f}")

print("\n== Break it 2. slerp without the q.q1 < 0 sign check (300 pairs, 60 deg apart, random sign of q1) ==")
rg = np.random.default_rng(2)
arcs = {True: [], False: []}
flipped = 0
for _ in range(300):
    R0 = euler_zyx_to_R(*rg.uniform(-np.pi, np.pi, 3))
    R1 = R0 @ rodrigues(rg.normal(size=3), 60 * DEG)
    q0, q1 = R_to_quat(R0), R_to_quat(R1)
    if rg.random() < 0.5:
        q1 = -q1                                    # same rotation, other hemisphere
    flipped += np.dot(q0, q1) < 0
    for chk in (True, False):
        path, prev = 0.0, R0
        for t in np.linspace(0, 1, 101):
            Rt = quat_to_R(slerp(q0, q1, t, sign_check=chk))
            path += angle_between(prev, Rt)
            prev = Rt
        arcs[chk].append(path / DEG)
print(f"pairs with q0.q1 < 0: {flipped} of 300")
for chk in (True, False):
    a = np.array(arcs[chk])
    print(f"sign check {str(chk):5s}: rotation travelled min {a.min():6.1f} deg, max {a.max():6.1f} deg")
