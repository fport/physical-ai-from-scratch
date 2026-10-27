"""Lesson 10 · Build it: damped least-squares IK from scratch on physai.kinematics, checked at every step.

Step 1  One DLS step vs the SVD formula and the pseudo-inverse.
Step 2  The solver loop with joint-limit clipping; its answer checked with MuJoCo's FK.
Step 3  The Experiment's 500-target sweep at its best damping, lam = 0.1.
Step 4  Singular values, manipulability and joint speeds while the arm stretches.

The functions live in physai/kinematics.py (the lesson shows them step by step).
Run:  uv run python experiments/10-inverse-kinematics/build.py      (~3 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.kinematics import chain_from_mujoco, dls_step, fk, ik_solve, jacobian  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
d = mujoco.MjData(m)
SITE = m.site("gripperframe").id
NA = 5
LO, HI = m.jnt_range[:NA, 0], m.jnt_range[:NA, 1]
chain = chain_from_mujoco(m, "gripperframe")

print("\n== Step 1. One DLS step vs the SVD formula (q = 0, a 1 cm error) ==")
J = jacobian(chain, np.zeros(NA))[:3]
e = np.array([0.01, 0.0, -0.005]) / np.linalg.norm([0.01, 0.0, -0.005]) * 0.01
U, sig, Vt = np.linalg.svd(J, full_matrices=False)
print(f"singular values of J_p at q = 0: {' '.join(f'{s:.4f}' for s in sig)} m/rad")
for lam in (0.0, 0.01, 0.1):
    svd = Vt.T @ (sig / (sig ** 2 + lam ** 2) * (U.T @ e))
    print(f"lam = {lam:<5g} |dq| = {np.linalg.norm(dls_step(J, e, lam)):.4f} rad, "
          f"max diff to sum sigma/(sigma^2+lam^2) v u^T e: {np.abs(dls_step(J, e, lam) - svd).max():.1e}")
print(f"lam = 0 vs np.linalg.pinv(J) @ e: max diff {np.abs(dls_step(J, e, 0.0) - np.linalg.pinv(J) @ e).max():.1e}")

print("\n== Step 2. Solve for (0.20, 0.05, 0.05) m from q = 0, check with mj_kinematics ==")
target = np.array([0.20, 0.05, 0.05])
for lam in (0.0, 0.1):
    q, ok, it = ik_solve(chain, target, np.zeros(NA), LO, HI, lam=lam)
    d.qpos[:NA] = q
    mujoco.mj_kinematics(m, d)
    print(f"lam = {lam:<4g} success {ok}, {it:3d} iterations, q = {np.round(q, 3)} rad, "
          f"MuJoCo tool error {np.linalg.norm(d.site_xpos[SITE] - target) * 1000:.3f} mm")

print("\n== Step 3. The Experiment's sweep: 500 reachable targets, q0 = 0, tol 0.5 mm, 200 iterations ==")
targets = np.array([fk(chain, q)[2] for q in np.random.default_rng(0).uniform(LO, HI, size=(500, NA))])
res = [ik_solve(chain, t, np.zeros(NA), LO, HI, lam=0.1) for t in targets]
ok = np.array([r[1] for r in res])
at_lim = sum(1 for q, s, _ in res if not s and np.any((np.abs(q - LO) < 1e-9) | (np.abs(q - HI) < 1e-9)))
print(f"lam = 0.1: success {ok.mean() * 100:.1f}%, median {np.median([r[2] for r in res if r[1]]):.0f} iterations, "
      f"p95 {np.percentile([r[2] for r in res if r[1]], 95):.0f}; {at_lim} of {int((~ok).sum())} failures end at a joint limit")

print("\n== Step 4. Stretching the arm: singular values, manipulability, joint speed for 5 cm/s ==")
q_bent = np.array([0, 1.3, 0.0, 0.0, 0])
q_straight = np.array([0, 1.3, -1.287, 0.035, 0])               # the Experiment's grid-search optimum
print(f"{'s':>5s} {'sigma1':>7s} {'sigma2':>7s} {'sigma3':>9s} {'w':>10s} {'|qdot| pinv':>12s} {'|qdot| DLS 0.1':>15s}")
for s in (0.0, 0.5, 0.9, 0.99, 1.0):
    Jp = jacobian(chain, q_bent + s * (q_straight - q_bent))[:3]
    U, sig, Vt = np.linalg.svd(Jp)
    v = 0.05 * U[:, 2]                                         # 5 cm/s along the weakest direction
    print(f"{s:5.2f} {sig[0]:7.4f} {sig[1]:7.4f} {sig[2]:9.2e} {np.prod(sig):10.3e} "
          f"{np.linalg.norm(dls_step(Jp, v, 0.0)):12.2f} {np.linalg.norm(dls_step(Jp, v, 0.1)):15.3f}")
