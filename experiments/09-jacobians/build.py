"""Lesson 09 · Build it: the SO-101 Jacobian from scratch, checked against MuJoCo at every step.

Step 1  Read the hinge chain (static model numbers only).
Step 2  Forward kinematics by hand vs mj_kinematics.
Step 3  Hinge columns -> Jacobian vs mj_jacSite.
Step 4  Use it: tool velocity and joint torques for a push at the gripper.
Step 5  Finite differences of our own FK vs our analytic Jacobian.

The functions live in physai/kinematics.py (the lesson shows them step by step).
Run:  uv run python experiments/09-jacobians/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.kinematics import chain_from_mujoco, fd_jacobian, fk, jacobian  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
d = mujoco.MjData(m)
SITE = m.site("gripperframe").id
NA = 5
QS = np.random.default_rng(0).uniform(m.jnt_range[:NA, 0], m.jnt_range[:NA, 1], size=(50, NA))


def mj_site_and_jac(q):
    d.qpos[:NA] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, SITE)
    return d.site_xpos[SITE].copy(), np.r_[jp[:, :NA], jr[:, :NA]]


print("\n== Step 1. The chain: world -> gripperframe ==")
chain = chain_from_mujoco(m, "gripperframe")
print(f"{'link':>4s} {'offset in parent (m)':>26s} {'hinge axis (body frame)':>24s}")
for i, link in enumerate(chain.links):
    axis = "fixed" if link.axis is None else np.array2string(link.axis, precision=0)
    print(f"{i:4d} {np.array2string(link.pos, precision=4, suppress_small=True):>26s} {axis:>24s}")
print(f"tool point in the last body: {np.array2string(chain.tool_pos, precision=4, suppress_small=True)} m")

print("\n== Step 2. Forward kinematics by hand vs mj_kinematics (50 random poses) ==")
err = max(np.abs(fk(chain, q)[2] - mj_site_and_jac(q)[0]).max() for q in QS)
print(f"max |p_ours - site_xpos| = {err:.1e} m")

print("\n== Step 3. Jacobian from hinge columns vs mj_jacSite (50 poses) ==")
errs = np.array([np.abs(jacobian(chain, q) - mj_site_and_jac(q)[1]) for q in QS])
print(f"max entry error: linear rows {errs[:, :3].max():.1e}, angular rows {errs[:, 3:].max():.1e}")

print("\n== Step 4. Use it at q = (0, -0.6, 0.9, 0.4, 0.3) ==")
Q0 = np.array([0.0, -0.6, 0.9, 0.4, 0.3])
J = jacobian(chain, Q0)
qdot = np.ones(NA) / np.sqrt(NA)
print(f"tool speed for every joint at 1/sqrt(5) rad/s: {np.linalg.norm(J[:3] @ qdot):.3f} m/s")
for F in ([0, 0, -2.0], [2.0, 0, 0], [0, 1.5, 0]):
    tau = J[:3].T @ np.array(F)
    print(f"F = {str(F):>16s} N  ->  J^T F = {' '.join(f'{t:7.4f}' for t in tau)} N m")

print("\n== Step 5. Central differences of our FK vs our analytic Jacobian (eps = 1e-5) ==")
fd_err = np.median([np.abs(fd_jacobian(lambda x: fk(chain, x)[2], q) - jacobian(chain, q)[:3]).max() for q in QS])
print(f"median max entry error: {fd_err:.1e}")
