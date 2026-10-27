"""Lesson 11 · Build it: inverse dynamics, the mass matrix and a simulator step from scratch.

Step 1  Read the body tree (static model numbers) and place every centre of mass vs MuJoCo.
Step 2  Recursive Newton-Euler inverse dynamics vs mj_inverse and the hand formula (double pendulum).
Step 3  Bias and gravity torques vs qfrc_bias (double pendulum, SO-101).
Step 4  Mass matrix by composite rigid bodies vs mj_fullM (double pendulum, SO-101, Panda).
Step 5  Forward dynamics + semi-implicit Euler: the energy drift of the Experiment's Euler row.

The functions live in physai/dynamics.py (the lesson shows them step by step).
Run:  uv run python experiments/11-rigid-body-dynamics/build.py      (~3 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from experiment import G, analytic, xml  # noqa: E402  (the lesson's double pendulum)
from physai.dynamics import energy, forward_dynamics, mass_matrix, rnea, spatial_tree, tree_from_mujoco  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
m = mujoco.MjModel.from_xml_string(xml())
d = mujoco.MjData(m)
rng = np.random.default_rng(0)
STATES = [(rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5, 5, 2), rng.uniform(-20, 20, 2)) for _ in range(1000)]

print("\n== Step 1. The body tree of the double pendulum ==")
tree = tree_from_mujoco(m)
print(f"{'body':>4s} {'parent':>6s} {'joint':>6s} {'mass (kg)':>9s} {'com in body (m)':>18s} {'inertia diag (kg m^2)':>22s}")
for i, b in enumerate(tree.bodies):
    print(f"{i:4d} {b.parent:6d} {b.joint:>6s} {b.mass:9.3f} {np.array2string(b.com, precision=3):>18s} "
          f"{np.array2string(np.diag(b.inertia), precision=3):>22s}")
err = 0.0
for q, _, _ in STATES:
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    err = max(err, np.abs(np.array(spatial_tree(tree, q)[0]) - d.xipos[1:]).max())
print(f"centres of mass vs MuJoCo's xipos, 1000 random poses: max diff {err:.1e} m")

print("\n== Step 2. Inverse dynamics (RNEA) vs mj_inverse and the hand formula, 1000 random states ==")
e_mj, e_hand = 0.0, 0.0
for q, qd, qdd in STATES:
    d.qpos[:], d.qvel[:], d.qacc[:] = q, qd, qdd
    mujoco.mj_inverse(m, d)
    Ma, Ca, ga = analytic(q, qd)
    tau = rnea(tree, q, qd, qdd)
    e_mj, e_hand = max(e_mj, np.abs(tau - d.qfrc_inverse).max()), max(e_hand, np.abs(tau - (Ma @ qdd + Ca + ga)).max())
print(f"max |tau_ours - qfrc_inverse| = {e_mj:.1e} N m, max |tau_ours - (M qdd + C qd + g)| = {e_hand:.1e} N m")

print("\n== Step 3. Bias c = rnea(q, qd, 0) vs qfrc_bias; at rest it is gravity ==")
e_b = 0.0
for q, qd, _ in STATES:
    d.qpos[:], d.qvel[:] = q, qd
    mujoco.mj_forward(m, d)
    e_b = max(e_b, np.abs(rnea(tree, q, qd, np.zeros(2)) - d.qfrc_bias).max())
print(f"double pendulum, 1000 states: max |c_ours - qfrc_bias| = {e_b:.1e} N m")
so = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
so_d, so_tree = mujoco.MjData(so), tree_from_mujoco(so)
for name, q in (("stretched (all 0)", np.zeros(6)), ("folded (lift -1.57, elbow 1.57)", np.array([0, -1.57, 1.57, 0, 0, 0]))):
    g = rnea(so_tree, q, np.zeros(6), np.zeros(6))
    so_d.qpos[:] = q
    mujoco.mj_forward(so, so_d)
    print(f"SO-101 {name:<32s} g(q) = {' '.join(f'{x:6.3f}' for x in g[:5])} N m, "
          f"max diff to qfrc_bias {np.abs(g - so_d.qfrc_bias).max():.1e}")

print("\n== Step 4. Mass matrix by composite rigid bodies vs mj_fullM ==")
Mf = np.zeros((2, 2))
e_mj, e_hand = 0.0, 0.0
for q, _, _ in STATES:
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    mujoco.mj_fullM(m, d, Mf)
    M = mass_matrix(tree, q)
    e_mj, e_hand = max(e_mj, np.abs(M - Mf).max()), max(e_hand, np.abs(M - analytic(q, np.zeros(2))[0]).max())
print(f"double pendulum, 1000 poses: max diff {e_mj:.1e} kg m^2 to mj_fullM, {e_hand:.1e} to the formula")
for label, mm, key in (("SO-101", so, None), ("Panda", mujoco.MjModel.from_xml_path(path("franka_emika_panda")), "home")):
    dd, tr = mujoco.MjData(mm), tree_from_mujoco(mm)
    Mf = np.zeros((mm.nv, mm.nv))
    worst = 0.0
    for q in np.random.default_rng(3).uniform(mm.jnt_range[:, 0], mm.jnt_range[:, 1], size=(20, mm.nv)):
        dd.qpos[:] = q
        mujoco.mj_forward(mm, dd)
        mujoco.mj_fullM(mm, dd, Mf)
        worst = max(worst, np.abs(mass_matrix(tr, q) - Mf).max())
    print(f"{label}: {mm.nv} joints, 20 random poses, max diff {worst:.1e} kg m^2")
    if key:
        mujoco.mj_resetDataKeyframe(mm, dd, mm.key(key).id)
        M = mass_matrix(tr, dd.qpos.copy())
        print(f"Panda 'home': diag M = {' '.join(f'{x:.3f}' for x in np.diag(M)[:7])} kg m^2, "
              f"cond(M[:7, :7]) = {np.linalg.cond(M[:7, :7]):.1f}")

print("\n== Step 5. Forward dynamics + semi-implicit Euler, 10 s at dt = 2 ms (the Experiment's Euler row) ==")
scale = G * (1.0 * 0.2 + 0.5 * (0.4 + 0.15))      # potential energy released when both links hang down (J)
q, qd, dt = np.zeros(2), np.zeros(2), 0.002
e0, worst = energy(tree, q, qd), 0.0
for _ in range(5000):
    e_before = energy(tree, q, qd)                 # what d.energy holds after mj_step: the state before the step
    qd = qd + dt * forward_dynamics(tree, q, qd, np.zeros(2))   # new speed first...
    q = q + dt * qd                                # ...then the angle from the new speed
    worst = max(worst, abs(energy(tree, q, qd) - e0))
d = mujoco.MjData(m)
for _ in range(5000):
    mujoco.mj_step(m, d)
print(f"after 10 s: max |q_ours - q_mj_step| = {np.abs(d.qpos - q).max():.1e} rad")
print(f"energy error: max {100 * worst / scale:.3f}%, final {100 * (energy(tree, q, qd) - e0) / scale:+.3f}% "
      f"(one step earlier {100 * (e_before - e0) / scale:+.3f}%) of {scale:.3f} J")
