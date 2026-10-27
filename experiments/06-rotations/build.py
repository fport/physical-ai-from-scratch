"""Lesson 06 · Build it: quaternions, Euler angles and slerp from scratch, checked against MuJoCo and SciPy.

Step 1  Quaternion product, conjugate and vector rotation vs mju_mulQuat, mju_negQuat, mju_rotVecQuat.
Step 2  Axis-angle and matrix -> quaternion vs mju_axisAngle2Quat, mju_mat2Quat and SciPy.
Step 3  Euler z-y'-x'' -> matrix vs mju_euler2Quat("zyx") and SciPy ("ZYX"), and back.
Step 4  The Euler-rate matrix E: vs finite differences of R, det E = cos(pitch), cond(E) near 90 deg.
Step 5  Slerp vs SciPy, and the sign check (60 deg vs 300 deg).

The functions live in physai/spatial.py (the lesson shows them step by step).
Run:  uv run python experiments/06-rotations/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.spatial import (axis_angle_to_quat, euler_rate_matrix, euler_zyx_to_mat, mat_to_euler_zyx,  # noqa: E402
                            mat_to_quat, quat_angle, quat_conj, quat_mul, quat_rotate, quat_to_mat, slerp)

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
rng = np.random.default_rng(0)
N = 1000
Q = rng.normal(size=(N, 4))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)          # random unit quaternions (uniform rotations)
Q2 = np.roll(Q, 1, axis=0)
V = rng.normal(size=(N, 3))


def mj(fn, *args, n=4):
    out = np.zeros(n)
    fn(out, *args)
    return out


print("\n== Step 1. Quaternion algebra vs MuJoCo (1000 random rotations) ==")
e_mul = max(np.abs(quat_mul(a, b) - mj(mujoco.mju_mulQuat, a, b)).max() for a, b in zip(Q, Q2))
e_conj = max(np.abs(quat_conj(a) - mj(mujoco.mju_negQuat, a)).max() for a in Q)
e_rot = max(np.abs(quat_rotate(a, v) - mj(mujoco.mju_rotVecQuat, v, a, n=3)).max() for a, v in zip(Q, V))
e_comp = max(np.abs(quat_to_mat(quat_mul(a, b)) - quat_to_mat(a) @ quat_to_mat(b)).max() for a, b in zip(Q, Q2))
e_neg = max(np.abs(quat_to_mat(-a) - quat_to_mat(a)).max() for a in Q)
print(f"quat_mul    vs mju_mulQuat     {e_mul:.1e}")
print(f"quat_conj   vs mju_negQuat     {e_conj:.1e}")
print(f"quat_rotate vs mju_rotVecQuat  {e_rot:.1e}")
print(f"R(a * b) vs R(a) @ R(b)        {e_comp:.1e}")
print(f"R(-q) vs R(q)                  {e_neg:.1e}")

print("\n== Step 2. Axis-angle and matrix -> quaternion vs MuJoCo and SciPy ==")
axes = V / np.linalg.norm(V, axis=1, keepdims=True)
angles = rng.uniform(-np.pi, np.pi, N)
e_aa = max(np.abs(axis_angle_to_quat(k, a) - mj(mujoco.mju_axisAngle2Quat, k, a)).max() for k, a in zip(axes, angles))
mats = [quat_to_mat(q) for q in Q]
sign = lambda q: q if q[0] >= 0 else -q                # noqa: E731  (q and -q: compare with w >= 0)
e_m2q = max(np.abs(mat_to_quat(R) - sign(mj(mujoco.mju_mat2Quat, R.flatten()))).max() for R in mats)
e_sp = max(np.abs(mat_to_quat(R) - sign(Rotation.from_matrix(R).as_quat(scalar_first=True))).max() for R in mats)
e_rt = max(np.abs(quat_to_mat(mat_to_quat(R)) - R).max() for R in mats)
print(f"axis_angle_to_quat vs mju_axisAngle2Quat       {e_aa:.1e}")
print(f"mat_to_quat vs mju_mat2Quat (w >= 0)           {e_m2q:.1e}")
print(f"mat_to_quat vs SciPy as_quat(scalar_first)     {e_sp:.1e}")
print(f"quat_to_mat(mat_to_quat(R)) round trip         {e_rt:.1e}")
print(f"180 deg about x: mat_to_quat gives {mat_to_quat(np.diag([1.0, -1, -1]))} (trace -1: the x branch)")

print("\n== Step 3. Euler z-y'-x'' vs mju_euler2Quat('zyx') and SciPy from_euler('ZYX') ==")
EUL = np.c_[rng.uniform(-np.pi, np.pi, N), rng.uniform(-np.pi / 2, np.pi / 2, N), rng.uniform(-np.pi, np.pi, N)]
e_mjz = max(np.abs(euler_zyx_to_mat(*e) - quat_to_mat(mj(mujoco.mju_euler2Quat, e, "zyx"))).max() for e in EUL)
e_spz = max(np.abs(euler_zyx_to_mat(*e) - Rotation.from_euler("ZYX", e).as_matrix()).max() for e in EUL)
e_lower = max(np.abs(euler_zyx_to_mat(*e) - Rotation.from_euler("zyx", e).as_matrix()).max() for e in EUL)
e_back = max(np.abs(np.array(mat_to_euler_zyx(euler_zyx_to_mat(*e))) - e).max() for e in EUL)
print(f"vs MuJoCo 'zyx' (lowercase = intrinsic there)  {e_mjz:.1e}")
print(f"vs SciPy 'ZYX' (uppercase = intrinsic there)   {e_spz:.1e}")
print(f"vs SciPy 'zyx' (extrinsic: a different rotation) {e_lower:.2f}")
print(f"mat_to_euler_zyx round trip, angles            {e_back:.1e} rad")
for p_deg in (89.0, 90.0):                            # gimbal lock: at 90 deg only yaw - roll matters
    a, b = euler_zyx_to_mat(0.7, np.radians(p_deg), 0.3), euler_zyx_to_mat(0.9, np.radians(p_deg), 0.5)
    print(f"pitch {p_deg:4.1f} deg: (yaw, roll) = (0.7, 0.3) and (0.9, 0.5) give matrices {np.abs(a - b).max():.1e} apart")

print("\n== Step 4. The Euler-rate matrix E (roll = 30 deg) ==")
roll = np.radians(30)
for p_deg in (0, 60, 80, 89, 89.9, 89.99):
    p = np.radians(p_deg)
    E = euler_rate_matrix(p, roll)
    eps, yaw = 1e-6, 0.4                              # w_b = vee(R^T dR/dt) for each unit rate, central differences
    E_fd = np.zeros((3, 3))
    for i, d_eul in enumerate(np.eye(3)):             # rates of (roll, pitch, yaw)
        ang = np.array([yaw, p, roll])
        step = np.array([d_eul[2], d_eul[1], d_eul[0]]) * eps
        R0, Rp, Rm = euler_zyx_to_mat(*ang), euler_zyx_to_mat(*(ang + step)), euler_zyx_to_mat(*(ang - step))
        W = R0.T @ (Rp - Rm) / (2 * eps)
        E_fd[:, i] = [W[2, 1], W[0, 2], W[1, 0]]
    s = np.linalg.svd(E, compute_uv=False)
    print(f"pitch {p_deg:5.2f} deg: det E {np.linalg.det(E):.3e} (cos {np.cos(p):.3e}) · cond {s[0] / s[-1]:9.2f} · "
          f"|E - E_fd| {np.abs(E - E_fd).max():.0e}")

print("\n== Step 5. Slerp vs SciPy, and the sign check (300 pairs, 60 deg apart) ==")
q0s = Q[:300]
q1s = np.array([quat_mul(q, axis_angle_to_quat(k, np.radians(60))) for q, k in zip(q0s, axes[:300])])
flip = rng.random(300) < 0.5
q1s[flip] *= -1                                       # same rotations, random sign
e_sl = 0.0
for a, b in zip(q0s, q1s):
    ref = Slerp([0, 1], Rotation.from_quat([a, b], scalar_first=True))(0.3).as_quat(scalar_first=True)
    e_sl = max(e_sl, np.abs(sign(slerp(a, b, 0.3)) - sign(ref)).max())
print(f"slerp(q0, q1, 0.3) vs SciPy Slerp (w >= 0): {e_sl:.1e} · pairs with q0.q1 < 0: {int(flip.sum())}")
ts = np.linspace(0, 1, 101)
for shortest in (True, False):
    travelled = [sum(quat_angle(slerp(a, b, t0, shortest), slerp(a, b, t1, shortest)) for t0, t1 in zip(ts, ts[1:]))
                 for a, b in zip(q0s, q1s)]
    print(f"sign check {str(shortest):5s}: rotation travelled min {np.degrees(min(travelled)):6.1f} deg, "
          f"max {np.degrees(max(travelled)):6.1f} deg")
