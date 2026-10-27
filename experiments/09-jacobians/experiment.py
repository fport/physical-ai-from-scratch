"""Lesson 09 · Jacobians and differential kinematics (SO-101, MuJoCo Menagerie).

Part A  Velocity map: J qdot vs the actual site velocity from a tiny finite step.
Part B  Sweep the finite-difference step eps: |J_fd - J_mujoco| (forward vs central differences).
Part C  Static force map tau = J^T F: push the gripper site with an external force in simulation,
        let the position servos settle, compare their torques with J^T F.
Break   How much tip force does each joint see? Max vertical tip load the model's servos can hold.

Run:  python experiment.py      (< 30 s)
"""
import os
import platform
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | Python {platform.python_version()} "
      f"| {platform.machine()}")

m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
d = mujoco.MjData(m)
SITE = m.site("gripperframe").id
NA = 5                                         # arm joints; dof 5 (gripper) does not move the site
LO, HI = m.jnt_range[:NA, 0], m.jnt_range[:NA, 1]
ARM_NAMES = [m.joint(i).name for i in range(NA)]


def site_pose(q):
    d.qpos[:NA] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[SITE].copy(), site_quat()


def site_quat():
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, d.site_xmat[SITE])
    return q


def jac_mj(q):
    d.qpos[:NA] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)                    # mj_jac* needs cdof/subtree data from mj_comPos
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, SITE)
    return jp[:, :NA], jr[:, :NA]


def jac_fd(q, eps, central):
    p0, q0 = site_pose(q)
    Jp, Jr = np.zeros((3, NA)), np.zeros((3, NA))
    for i in range(NA):
        dq = np.zeros(NA)
        dq[i] = eps
        pp, qp = site_pose(q + dq)
        w = np.zeros(3)
        if central:
            pm, qm = site_pose(q - dq)
            Jp[:, i] = (pp - pm) / (2 * eps)
            mujoco.mju_subQuat(w, qp, qm)     # rotation from qm to qp, in the qm (local) frame
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, qm)
            Jr[:, i] = R.reshape(3, 3) @ w / (2 * eps)
        else:
            Jp[:, i] = (pp - p0) / eps
            mujoco.mju_subQuat(w, qp, q0)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q0)
            Jr[:, i] = R.reshape(3, 3) @ w / eps
    return Jp, Jr


rng = np.random.default_rng(0)
QS = rng.uniform(LO, HI, size=(50, NA))

# ---------- Part A ----------
print("\n== A. Velocity map v = J_p qdot, w = J_r qdot (50 random poses, random qdot, |qdot| = 1 rad/s) ==")
ev, ew = [], []
for q in QS:
    qd = rng.normal(size=NA)
    qd /= np.linalg.norm(qd)
    Jp, Jr = jac_mj(q)
    dt = 1e-6
    p0, q0 = site_pose(q)
    p1, q1 = site_pose(q + qd * dt)
    w = np.zeros(3)
    mujoco.mju_subQuat(w, q1, q0)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q0)
    ev.append(np.linalg.norm((p1 - p0) / dt - Jp @ qd) / np.linalg.norm(Jp @ qd))
    ew.append(np.linalg.norm(R.reshape(3, 3) @ w / dt - Jr @ qd) / np.linalg.norm(Jr @ qd))
print(f"relative error, linear velocity: max {max(ev):.1e} | angular velocity: max {max(ew):.1e}")
print(f"typical site speed for |qdot| = 1 rad/s: median |v| = "
      f"{np.median([np.linalg.norm(jac_mj(q)[0] @ (np.ones(NA) / np.sqrt(NA))) for q in QS]):.3f} m/s")

# ---------- Part B ----------
print("\n== B. Sweep finite-difference step eps: median over 50 poses of max |J_fd - J_mj| ==")
print(f"{'eps (rad)':>9s} {'forward Jp':>11s} {'forward Jr':>11s} {'central Jp':>11s} {'central Jr':>11s}")
JM = [jac_mj(q) for q in QS]
best = {}
for k in range(1, 15):
    eps = 10.0 ** -k
    row = []
    for central in (False, True):
        ep, er = [], []
        for q, (Jp, Jr) in zip(QS, JM):
            fp, fr = jac_fd(q, eps, central)
            ep.append(np.abs(fp - Jp).max())
            er.append(np.abs(fr - Jr).max())
        row += [np.median(ep), np.median(er)]
        key = "central" if central else "forward"
        if key not in best or np.median(ep) < best[key][1]:
            best[key] = (eps, np.median(ep))
    print(f"{eps:9.0e} {row[0]:11.1e} {row[1]:11.1e} {row[2]:11.1e} {row[3]:11.1e}")
for k, (e, v) in best.items():
    print(f"best {k:7s} step for Jp: eps = {e:.0e} -> error {v:.1e}")
print(f"double precision machine epsilon: {np.finfo(float).eps:.1e}; "
      f"sqrt = {np.sqrt(np.finfo(float).eps):.1e}, cbrt = {np.cbrt(np.finfo(float).eps):.1e}")

# ---------- Part C ----------
print("\n== C. Static force map: external force F at the gripper site vs servo torques after settling ==")
mg = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
mg.opt.gravity[:] = 0                          # isolate the external force
mg.dof_frictionloss[:] = 0                     # Coulomb friction would hold part of the load
dg = mujoco.MjData(mg)
Q0 = np.array([0.0, -0.6, 0.9, 0.4, 0.3, 0.0])
F_LIST = [np.array([0, 0, -2.0]), np.array([2.0, 0, 0]), np.array([0, 1.5, 0]), np.array([1.0, -1.0, 1.0])]
body = mg.site_bodyid[SITE]
print(f"{'F (N)':>16s} {'J^T F (N m), joints 1-5':>40s} {'servo torque (N m)':>40s} {'max diff':>9s}")
for F in F_LIST:
    mujoco.mj_resetData(mg, dg)
    dg.qpos[:] = Q0
    dg.ctrl[:] = Q0
    for _ in range(4000):                      # 20 s at the model's 5 ms timestep
        mujoco.mj_kinematics(mg, dg)
        # xfrc_applied acts at the body's centre of mass: move the force to the site with r x F
        r = dg.site_xpos[SITE] - dg.xipos[body]
        dg.xfrc_applied[body, :3] = F
        dg.xfrc_applied[body, 3:] = np.cross(r, F)
        mujoco.mj_step(mg, dg)
    mujoco.mj_forward(mg, dg)
    jp = np.zeros((3, mg.nv))
    mujoco.mj_jacSite(mg, dg, jp, None, SITE)
    tau_pred = (jp.T @ F)[:NA]
    tau_servo = dg.qfrc_actuator[:NA]
    fmt = lambda v: " ".join(f"{a:7.4f}" for a in v)  # noqa: E731
    print(f"{np.array2string(F, precision=1):>16s} {fmt(tau_pred):>40s} {fmt(-tau_servo):>40s} "
          f"{np.abs(tau_pred + tau_servo).max():9.1e}")
print(f"joint speed at the end: max |qvel| = {np.abs(dg.qvel).max():.1e} rad/s")

# ---------- Break it ----------
print("\n== Break it. Vertical tip load the model's servos can hold (forcerange from so101.xml) ==")
fr = m.actuator_forcerange[:NA, 1]
print(f"servo force limit in the model: {fr[0]:.2f} N m per joint")
m2 = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
d2 = mujoco.MjData(m2)
print(f"{'pose (rad)':>30s} {'lever (m)':>9s} {'gravity tau (N m)':>32s} {'tau per N down (N m/N)':>32s} "
      f"{'max tip load (N)':>16s} {'limiting joint':>14s}")
fmt = lambda v: " ".join(f"{a:6.3f}" for a in v)  # noqa: E731
for name, q in [("near      (0,-1.5,1.5,0.5,0)", [0, -1.5, 1.5, 0.5, 0]),
                ("mid       (0,-0.6,0.9,0.4,0)", [0, -0.6, 0.9, 0.4, 0]),
                ("stretched (0,1.3,-1.3,0,0)", [0, 1.3, -1.3, 0.0, 0])]:
    d2.qpos[:] = 0
    d2.qpos[:NA] = q
    d2.qvel[:] = 0
    mujoco.mj_forward(m2, d2)
    g = d2.qfrc_bias[:NA].copy()               # at rest: torque the servos must supply against gravity
    jp = np.zeros((3, m2.nv))
    mujoco.mj_jacSite(m2, d2, jp, None, SITE)
    a = (jp.T @ np.array([0, 0, -1.0]))[:NA]   # generalized force per newton of downward tip force
    lever = np.linalg.norm((d2.site_xpos[SITE] - d2.xpos[m2.body("upper_arm").id])[[0, 2]])
    # servo torque with a tip load f: tau = g - f a; need |tau_i| <= limit for every joint
    fmax, lim = np.inf, "-"
    for i in range(NA):
        if abs(a[i]) < 1e-9:
            continue
        bounds = sorted([(g[i] - fr[i]) / a[i], (g[i] + fr[i]) / a[i]])
        if bounds[1] < fmax:
            fmax, lim = bounds[1], ARM_NAMES[i]
    print(f"{name:>30s} {lever:9.3f} {fmt(g):>32s} {fmt(a):>32s} {fmax:16.2f} {lim:>14s}")

print("\n== Break it (simulated). Stretched pose, gravity on, downward tip force as a fraction of the limit ==")
q_s = np.array([0, 1.3, -1.3, 0.0, 0, 0])
d2.qpos[:] = 0
d2.qpos[:NA] = q_s[:NA]
mujoco.mj_forward(m2, d2)
jp = np.zeros((3, m2.nv))
mujoco.mj_jacSite(m2, d2, jp, None, SITE)
a = (jp.T @ np.array([0, 0, -1.0]))[:NA]
g = d2.qfrc_bias[:NA].copy()
f_lim = min(sorted([(g[i] - fr[i]) / a[i], (g[i] + fr[i]) / a[i]])[1] for i in range(NA) if abs(a[i]) > 1e-9)
b2 = m2.site_bodyid[SITE]
print(f"{'tip force (N)':>13s} {'x limit':>8s} {'tip drop after 5 s (mm)':>24s} {'lift servo torque (N m)':>24s}")
for frac in [0.0, 0.5, 0.9, 1.1, 1.5]:
    mujoco.mj_resetData(m2, d2)
    d2.qpos[:] = q_s
    d2.ctrl[:] = q_s
    f = frac * f_lim
    mujoco.mj_forward(m2, d2)
    z0 = d2.site_xpos[SITE][2]
    for _ in range(1000):                      # 5 s
        mujoco.mj_kinematics(m2, d2)
        F = np.array([0, 0, -f])
        d2.xfrc_applied[b2, :3] = F
        d2.xfrc_applied[b2, 3:] = np.cross(d2.site_xpos[SITE] - d2.xipos[b2], F)
        mujoco.mj_step(m2, d2)
    mujoco.mj_forward(m2, d2)
    print(f"{f:13.2f} {frac:8.1f} {1000 * (z0 - d2.site_xpos[SITE][2]):24.1f} {d2.actuator_force[1]:24.3f}")
