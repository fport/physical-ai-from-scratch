"""Lesson 08 · Build it: SO-101 forward kinematics from scratch, checked against MuJoCo at every step.

Step 1  Read the hinge chain (static model numbers only).
Step 2  Forward kinematics by hand vs mj_kinematics.
Step 3  Count the degrees of freedom and find the arm plane in your own FK output.
Step 4  Workspace by sampling, against the Experiment's table.

The functions live in physai/kinematics.py (the lesson shows them step by step).
Run:  uv run python experiments/08-forward-kinematics/build.py      (~2 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.kinematics import chain_from_mujoco, fk, sample_workspace, voxel_volume  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
d = mujoco.MjData(m)
SITE = m.site("gripperframe").id
NA = 5
LO, HI = m.jnt_range[:NA, 0], m.jnt_range[:NA, 1]

print("\n== Step 1. The chain: world -> gripperframe ==")
chain = chain_from_mujoco(m, "gripperframe")
print(f"{'link':>4s} {'offset in parent (m)':>26s} {'hinge axis (body frame)':>24s}")
for i, link in enumerate(chain.links):
    axis = "fixed" if link.axis is None else np.array2string(link.axis, precision=0)
    print(f"{i:4d} {np.array2string(link.pos, precision=4, suppress_small=True):>26s} {axis:>24s}")
print(f"tool point in the last body: {np.array2string(chain.tool_pos, precision=4, suppress_small=True)} m")

print("\n== Step 2. Forward kinematics by hand vs mj_kinematics (1000 random poses) ==")
QS = np.random.default_rng(0).uniform(LO, HI, size=(1000, NA))   # the Experiment's poses


def mj_site(q):
    d.qpos[:NA] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[SITE].copy()


err = max(np.abs(fk(chain, q)[2] - mj_site(q)).max() for q in QS)
print(f"max |p_ours - site_xpos| = {err:.1e} m")

print("\n== Step 3. Degrees of freedom and the arm plane, from your FK ==")
print(f"hinges in the chain: {sum(link.axis is not None for link in chain.links)}; "
      f"model: nq = {m.nq}, nv = {m.nv} (the 6th joint moves only the jaw)")
par = 0.0
for q in QS[:50]:
    axes, anchors, p = fk(chain, q)
    par = max(par, max(np.linalg.norm(np.cross(axes[1], axes[k])) for k in (2, 3)))
print(f"lift, elbow, wrist flex axes parallel: max |a_lift x a_k| over 50 poses = {par:.1e}")
axes, anchors, p0 = fk(chain, np.zeros(NA))
lever = np.linalg.norm(np.cross(axes[4], p0 - anchors[4]))     # distance of the tool from the roll axis
p180 = fk(chain, np.array([0, 0, 0, 0, np.pi]))[2]
print(f"tool point off the wrist_roll axis: {lever * 1000:.1f} mm; roll 0 -> 180 deg moves it "
      f"{np.linalg.norm(p180 - p0) * 1000:.1f} mm")

print("\n== Step 4. Workspace by sampling (seed 1, the Experiment's first samples) ==")
shoulder = fk(chain, np.zeros(NA))[1][0]                       # a point on the pan axis at shoulder height
print(f"{'samples':>8s} {'1 cm voxels (L)':>16s} {'2 cm voxels (L)':>16s} {'max reach (m)':>14s} {'below z = 0':>12s}")
P = sample_workspace(chain, LO, HI, 10_000)
for n in (1_000, 10_000):
    pts = P[:n]
    print(f"{n:8d} {voxel_volume(pts, 0.01) * 1e3:16.1f} {voxel_volume(pts, 0.02) * 1e3:16.1f} "
          f"{np.linalg.norm(pts - shoulder, axis=1).max():14.3f} {np.mean(pts[:, 2] < 0) * 100:11.1f}%")
