"""Lesson 08 · Forward kinematics, degrees of freedom and workspace (SO-101, MuJoCo Menagerie).

Part A  Hand FK (product of transforms read from the MJCF chain) vs mj_kinematics.
Part B  DoF count; workspace by random sampling: sweep the number of samples -> voxel volume.
Part C  Why 5 arm joints cannot give arbitrary orientations: sweep how far the required
        approach direction leaves the arm's vertical plane -> best achievable orientation error.
        Then: random target orientations at one point -> reachable fraction.

Run:  python experiment.py      (~1 min on Apple M5)
"""
import os
import platform
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | Python {platform.python_version()} "
      f"| {platform.machine()}")

m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
d = mujoco.MjData(m)
SITE = m.site("gripperframe").id
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
LO, HI = m.jnt_range[:5, 0].copy(), m.jnt_range[:5, 1].copy()


def quat2R(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def axis_rot(axis, a):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def T(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


# ---- the chain, read once from the compiled model ----
chain = []                                    # bodies from base to gripper
b = m.site_bodyid[SITE]
while b != 0:
    chain.append(b)
    b = m.body_parentid[b]
chain = chain[::-1]
STATIC = [T(quat2R(m.body_quat[b]), m.body_pos[b]) for b in chain]
JOINT = [(m.body_jntadr[b], m.jnt_axis[m.body_jntadr[b]].copy()) if m.body_jntnum[b] else None for b in chain]
T_SITE = T(quat2R(m.site_quat[SITE]), m.site_pos[SITE])
assert np.allclose(m.jnt_pos[:5], 0)          # joints sit at their body origins in this model


def fk_hand(q):
    """T_world_site = prod_i [ T_static_i * Rot(axis_i, q_i) ] * T_site."""
    M = np.eye(4)
    for Ts, jt in zip(STATIC, JOINT):
        M = M @ Ts
        if jt is not None:
            j, ax = jt
            M = M @ T(axis_rot(ax, q[j]), np.zeros(3))
    return M @ T_SITE


def fk_mj(q):
    d.qpos[:5] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[SITE].copy(), d.site_xmat[SITE].reshape(3, 3).copy()


# ---------- Part A ----------
print("\n== A. Hand FK vs mj_kinematics, gripperframe site, 1000 random joint vectors within limits ==")
print("chain:", " -> ".join(m.body(b).name for b in chain), "-> site gripperframe")
rng = np.random.default_rng(0)
pe, re = [], []
for _ in range(1000):
    q = rng.uniform(LO, HI)
    Mh = fk_hand(q)
    p, R = fk_mj(q)
    pe.append(np.abs(Mh[:3, 3] - p).max())
    re.append(np.abs(Mh[:3, :3] - R).max())
print(f"max position error {max(pe):.1e} m, max rotation-matrix entry error {max(re):.1e}")
p_a = fk_hand(np.zeros(5))[:3, 3]
p_b = fk_hand(np.array([0, 0, 0, 0, np.pi]))[:3, 3]
print(f"site offset from the wrist_roll axis in the MJCF: {1000 * abs(m.site_pos[SITE][0]):.1f} mm; "
      f"site moves {1000 * np.linalg.norm(p_b - p_a):.1f} mm when wrist_roll goes 0 -> 180 deg")

# ---------- Part B ----------
print("\n== B. Degrees of freedom ==")
print(f"nq = {m.nq}, nv = {m.nv}, actuators = {m.nu}")
for j in range(m.njnt):
    lo, hi = np.rad2deg(m.jnt_range[j])
    print(f"  {m.joint(j).name:14s} hinge  range {lo:8.1f} .. {hi:6.1f} deg")
print("arm joints that move the gripper frame: 5; the 6th ('gripper') only opens the jaw")

print("\n== B. Workspace of the gripperframe site by uniform joint sampling (no collision filtering) ==")
d.qpos[:] = 0
mujoco.mj_kinematics(m, d)
shoulder = d.xpos[m.body("shoulder").id].copy()        # a point on the pan axis, 6.24 cm above the base
rg = np.random.default_rng(1)
Q = rg.uniform(LO, HI, size=(1_000_000, 5))
P = np.empty((len(Q), 3))
for i, q in enumerate(Q):
    P[i] = fk_mj(q)[0]
print(f"{'samples':>9s} {'vol 1 cm vox (L)':>17s} {'vol 2 cm vox (L)':>17s} {'hull vol (L)':>13s} "
      f"{'max reach (m)':>14s} {'below z=0':>10s}")
for n in [1_000, 10_000, 100_000, 1_000_000]:
    pts = P[:n]
    v1 = len(np.unique(np.floor(pts / 0.01).astype(int), axis=0)) * 1e-6 * 1e3
    v2 = len(np.unique(np.floor(pts / 0.02).astype(int), axis=0)) * 8e-6 * 1e3
    hull = ConvexHull(pts).volume * 1e3
    reach = np.linalg.norm(pts - shoulder, axis=1).max()
    print(f"{n:9d} {v1:17.1f} {v2:17.1f} {hull:13.1f} {reach:14.3f} {np.mean(pts[:, 2] < 0) * 100:9.1f}%")

# ---------- Part C ----------
TARGET = np.array([0.22, 0.0, 0.08])          # a point in front of the robot (base = world here)


def approach_of(R):
    return R[:, 0]                            # site x-axis points along the fingers


def solve(residual, seeds):
    best = None
    for s in seeds:
        r = least_squares(residual, s, bounds=(LO, HI), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    return best


seeds = np.random.default_rng(2).uniform(LO, HI, size=(12, 5))
# Reachable approach directions at a fixed point lie in the plane normal to the (parallel) pitch axes.
# Find that plane: solve position only, read the shoulder_lift axis n, build an in-plane direction a0.
W_POS = 1e4                                   # position residual weight: 1 um counts like 0.01 of a unit vector
r0 = solve(lambda q: W_POS * (fk_mj(q)[0] - TARGET), seeds)
fk_mj(r0.x)
n = d.xaxis[m.joint("shoulder_lift").id].copy()
pitch_axes = [d.xaxis[m.joint(j).id] for j in ("shoulder_lift", "elbow_flex", "wrist_flex")]
par = max(np.linalg.norm(np.cross(n, a)) for a in pitch_axes)
down = np.array([0, 0, -1.0])
fwd = np.cross(n, down)
fwd *= np.sign(fwd @ TARGET)
a0 = np.cos(np.pi / 4) * fwd / np.linalg.norm(fwd) + np.sin(np.pi / 4) * down     # 45 deg below horizontal
print(f"\npitch axes parallel: max |n x a| = {par:.1e}; n = {np.round(n, 4)}; a0 . n = {a0 @ n:.1e}")
print("\n== C. Sweep beta = angle of the required approach direction out of the arm's vertical plane ==")
print(f"target point {TARGET} m; a_req = cos(beta) a0 + sin(beta) n; 12 seeds, joint limits enforced")
print(f"{'beta (deg)':>10s} {'pos err (mm)':>13s} {'approach err (deg)':>19s}")
for beta in [0, 5, 10, 20, 30, 45, 60, 90]:
    a_req = np.cos(np.deg2rad(beta)) * a0 + np.sin(np.deg2rad(beta)) * n

    def res(q, a_req=a_req):
        p, R = fk_mj(q)
        return np.concatenate([W_POS * (p - TARGET), approach_of(R) - a_req])

    r = solve(res, seeds)
    p, R = fk_mj(r.x)
    ang = np.rad2deg(np.arccos(np.clip(approach_of(R) @ a_req, -1, 1)))
    print(f"{beta:10d} {1000 * np.linalg.norm(p - TARGET):13.3f} {ang:19.3f}")

print("\n== C. 200 uniformly random full target orientations at the same point (6-D task, 5 joints) ==")
rq = np.random.default_rng(3).normal(size=(200, 4))
rq /= np.linalg.norm(rq, axis=1, keepdims=True)
errs, perrs = [], []
for qt in rq:
    Rt = quat2R(qt)

    def res(q, Rt=Rt):
        p, R = fk_mj(q)
        return np.concatenate([W_POS * (p - TARGET), (R - Rt).ravel()])

    r = solve(res, seeds[:6])
    p, R = fk_mj(r.x)
    errs.append(np.rad2deg(np.arccos(np.clip((np.trace(R.T @ Rt) - 1) / 2, -1, 1))))
    perrs.append(1000 * np.linalg.norm(p - TARGET))
errs = np.array(errs)
print(f"orientation error: median {np.median(errs):.1f} deg, min {errs.min():.2f} deg, max {errs.max():.1f} deg; "
      f"median pos err {np.median(perrs):.3f} mm")
for tol in [1, 5, 10, 30]:
    print(f"  reachable within {tol:2d} deg: {np.sum(errs <= tol):3d} / 200")

print("\n== C. Same test on a 7-DoF arm: Franka Panda (Menagerie), body 'hand', point (0.5, 0, 0.4) m ==")
mp = mujoco.MjModel.from_xml_path(path("franka_emika_panda", "panda.xml"))
dp = mujoco.MjData(mp)
HAND = mp.body("hand").id
LO7, HI7 = mp.jnt_range[:7, 0].copy(), mp.jnt_range[:7, 1].copy()
P_TGT = np.array([0.5, 0.0, 0.4])
seeds7 = np.random.default_rng(4).uniform(LO7, HI7, size=(6, 7))


def fk7(q):
    dp.qpos[:7] = q
    mujoco.mj_kinematics(mp, dp)
    return dp.xpos[HAND].copy(), dp.xmat[HAND].reshape(3, 3).copy()


errs7 = []
for qt in rq:
    Rt = quat2R(qt)

    def res(q, Rt=Rt):
        p, R = fk7(q)
        return np.concatenate([W_POS * (p - P_TGT), (R - Rt).ravel()])

    best7 = None
    for s in seeds7:
        r = least_squares(res, s, bounds=(LO7, HI7), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        if best7 is None or r.cost < best7.cost:
            best7 = r
    p, R = fk7(best7.x)
    errs7.append(np.rad2deg(np.arccos(np.clip((np.trace(R.T @ Rt) - 1) / 2, -1, 1))))
errs7 = np.array(errs7)
print(f"orientation error: median {np.median(errs7):.2f} deg, max {errs7.max():.1f} deg")
for tol in [1, 5, 10, 30]:
    print(f"  reachable within {tol:2d} deg: {np.sum(errs7 <= tol):3d} / 200")
