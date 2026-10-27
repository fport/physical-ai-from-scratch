"""Lesson 14 · Build it: impedance and operational-space control from scratch, on the Experiment's Panda.

Step 1  The orientation error vs a known rotation (SciPy builds it): e = sin(angle) axis.
Step 2  The hand's inertia Lambda from physai's own J (lesson 09) and M (lesson 11) vs MuJoCo's.
Step 3  Impedance vs operational-space wrench: the Experiment's 5 cm steps.
Step 4  The null-space projector: J M^-1 N = 0, and the Experiment's elbow push.
Step 5  Pressing on the surface: holding force = K x 5 cm.

The functions live in physai/control.py (the lesson shows them step by step). In the simulation loops
J and M come from MuJoCo (mj_jacSite, mj_fullM), which Step 2 shows equal to physai's to round-off.
Run:  uv run python experiments/14-impedance-control/build.py      (~4 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from experiment import P_START, PROBE_R, R_DES, build, ik, start  # noqa: E402  (the Experiment's model)
from physai.control import (impedance_wrench, null_space_projector, orientation_error, osc_wrench,  # noqa: E402
                            task_inertia, task_torque)
from physai.dynamics import mass_matrix, tree_from_mujoco  # noqa: E402
from physai.kinematics import chain_from_mujoco, jacobian  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

print("\n== Step 1. Orientation error for R_des = Rot(axis, angle) R ==")
R = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
k = np.array([0.6, 0.0, 0.8])                                  # unit rotation axis
for deg in (1, 10, 45, 90):
    Rd = Rotation.from_rotvec(np.deg2rad(deg) * k).as_matrix() @ R
    e = orientation_error(R, Rd)
    print(f"{deg:3d} deg about (0.6, 0, 0.8): |e| = {np.linalg.norm(e):.4f} (angle {np.deg2rad(deg):.4f} rad), "
          f"max |e - sin(angle) axis| = {np.abs(e - np.sin(np.deg2rad(deg)) * k).max():.1e}")

print("\n== Step 2. The hand's inertia at the start pose, from physai's J and M ==")
m = build()
d0 = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d0, m.key("home").id)
q0 = ik(m, P_START, R_DES, d0.qpos.copy())                     # the Experiment's start pose
J_ours = jacobian(chain_from_mujoco(m, "tcp"), q0[:7])
M_ours = mass_matrix(tree_from_mujoco(m), q0)[:7, :7]
d = start(m, q0)
site, lim = m.site("tcp").id, m.actuator_forcerange[:7, 1]
jp, jr, Mf = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((m.nv, m.nv))


def jac_mass(d):
    mujoco.mj_jacSite(m, d, jp, jr, site)
    mujoco.mj_fullM(m, d, Mf)
    return np.vstack([jp, jr])[:, :7], Mf[:7, :7]


J, M = jac_mass(d)
print(f"physai vs MuJoCo: max diff J {np.abs(J_ours - J).max():.1e}, M {np.abs(M_ours - M).max():.1e}")
Lam = task_inertia(J_ours, M_ours)
print(f"Lambda_xx, yy, zz = {Lam[0, 0]:.2f}, {Lam[1, 1]:.2f}, {Lam[2, 2]:.2f} kg; "
      f"Lambda from MuJoCo's J and M differs by {np.abs(Lam - task_inertia(J, M)).max():.1e}")


def tcp_error(d, target):
    R = d.site_xmat[site].reshape(3, 3)
    return np.r_[target - d.site_xpos[site], orientation_error(R, R_DES)]


print("\n== Step 3. 5 cm steps, 2 s each: impedance (K 600, D 50) vs operational space (kp 100, kd 14) ==")
print(f"{'controller':<10s} {'axis':>4s} {'overshoot %':>12s} {'t into 2 mm (s)':>16s} {'off-axis peak (mm)':>19s}")
for mode in ("impedance", "osc"):
    for ax in range(3):
        d = start(m, q0)
        target = P_START.copy()
        target[ax] += 0.05
        hist, off = [], 0.0
        while d.time < 2.0 - 1e-9:
            J, M = jac_mass(d)
            e, v = tcp_error(d, target), J @ d.qvel[:7]
            F = (impedance_wrench(e, v, 600.0, 50.0, 30.0, 3.0) if mode == "impedance"
                 else osc_wrench(task_inertia(J, M), e, v, 100.0, 14.0))
            d.qfrc_applied[:7] = task_torque(J, F, d.qfrc_bias[:7], lim)[0]
            mujoco.mj_step(m, d)
            hist.append((d.time, d.site_xpos[site][ax] - P_START[ax]))
            off = max(off, np.abs(np.delete(d.site_xpos[site] - P_START, ax)).max())
        t, x = np.array(hist).T
        outside = np.where(np.abs(x - 0.05) > 0.002)[0]
        print(f"{mode:<10s} {'xyz'[ax]:>4s} {max(0.0, (x.max() - 0.05) / 0.05 * 100):12.1f} "
              f"{t[outside[-1]]:16.3f} {off * 1000:19.2f}")

print("\n== Step 4. Null space: J M^-1 N, then the Experiment's 20 N elbow push ==")
J, M = jac_mass(start(m, q0))
N = null_space_projector(J, M, task_inertia(J, M))
print(f"max |J M^-1 N| = {np.abs(J @ np.linalg.solve(M, N)).max():.1e}  (a null-space torque never moves the hand)")
elbow = m.body("link4").id
for null in (False, True):
    d = start(m, q0)
    e0, emax = d.xpos[elbow].copy(), 0.0
    while d.time < 3.0 - 1e-9:
        d.xfrc_applied[elbow, :3] = [0, 20.0, 0] if 0.5 <= d.time < 0.8 else 0.0
        J, M = jac_mass(d)
        Lam = task_inertia(J, M)
        F = osc_wrench(Lam, tcp_error(d, P_START), J @ d.qvel[:7], 100.0, 14.0)
        if null:
            posture = 10.0 * (q0[:7] - d.qpos[:7]) - 2.0 * d.qvel[:7]
            tau = task_torque(J, F, d.qfrc_bias[:7], lim, null_space_projector(J, M, Lam), posture)[0]
        else:
            tau = task_torque(J, F, d.qfrc_bias[:7], lim)[0]
        d.qfrc_applied[:7] = tau
        mujoco.mj_step(m, d)
        emax = max(emax, np.linalg.norm(d.site_xpos[site] - P_START))
    print(f"{'with' if null else 'without':>7s} null-space posture: tcp error {emax * 1000:.2f} mm, "
          f"elbow moved {np.linalg.norm(d.xpos[elbow] - e0) * 100:.2f} cm, |q - q0| {np.linalg.norm(d.qpos[:7] - q0[:7]):.3f} rad")

print("\n== Step 5. Press 5 cm into the surface: holding force vs K x 5 cm ==")
wall_z = P_START[2] - PROBE_R * 2 - 0.03
mw = build(wall_z)
wall, f6 = mw.geom("wall").id, np.zeros(6)
site, lim = mw.site("tcp").id, mw.actuator_forcerange[:7, 1]
m = mw
for K in (300, 1000):
    d = start(m, q0)
    J, M = jac_mass(d)
    D = 2 * np.sqrt(K * task_inertia(J, M)[2, 2])               # critically damped along z
    goal, hold = wall_z - 0.05 + 2 * PROBE_R, []
    while d.time < 2.5 - 1e-9:
        u = min(1.0, d.time / 1.0)
        target = P_START.copy()
        target[2] += (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5) * (goal - P_START[2])
        J, M = jac_mass(d)
        F = impedance_wrench(tcp_error(d, target), J @ d.qvel[:7], K, D, 30.0, 3.0)
        d.qfrc_applied[:7] = task_torque(J, F, d.qfrc_bias[:7], lim)[0]
        mujoco.mj_step(m, d)
        fn = 0.0
        for k in range(d.ncon):
            if wall in (d.contact[k].geom1, d.contact[k].geom2):
                mujoco.mj_contactForce(m, d, k, f6)
                fn += f6[0]
        if d.time > 2.2:
            hold.append(fn)
    print(f"K = {K:5d} N/m: K x 5 cm = {K * 0.05:5.1f} N, holding force {np.mean(hold):6.2f} N")
