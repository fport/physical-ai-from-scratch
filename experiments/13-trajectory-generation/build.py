"""Lesson 13 · Build it: time scalings from boundary conditions, checked against the lesson's formulas.

Step 1  Cubic and quintic coefficients from boundary conditions (a small linear solve).
Step 2  Derivatives and peaks vs the closed-form values (6, 10 sqrt(3) / 3, 60).
Step 3  The trapezoid: continuity and its peak acceleration 4.5.
Step 4  Scale to the Panda move: peak acceleration and jerk vs T, exact and sampled at 2 ms.
Step 5  Track the quintic on the Panda, without and with M(q) qdd_ref from physai.dynamics.
Use it  SciPy's clamped cubic spline and quintic B-spline through two points are the same curves.

The functions live in physai/trajectory.py (the lesson shows them step by step).
Run:  uv run python experiments/13-trajectory-generation/build.py      (~3 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from experiment import DQ, KD, KP, Q0, load_panda  # noqa: E402  (the Experiment's Panda move and gains)
from physai.dynamics import mass_matrix, tree_from_mujoco  # noqa: E402
from physai.trajectory import (CUBIC, QUINTIC, REST, cubic, joint_reference, poly_eval,  # noqa: E402
                               poly_from_boundary, quintic, trapezoid)

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
U = np.linspace(0, 1, 100_001)

print("\n== Step 1. Coefficients from boundary conditions (c0 + c1 u + c2 u^2 + ...) ==")
print(f"cubic   (s, s' at both ends):        {' '.join(f'{c:6.2f}' for c in CUBIC)}")
print(f"quintic (s, s', s'' at both ends):   {' '.join(f'{c:6.2f}' for c in QUINTIC)}")
print(f"max diff to 3u^2 - 2u^3: {np.abs(CUBIC - [0, 0, 3, -2]).max():.1e}, "
      f"to 10u^3 - 15u^4 + 6u^5: {np.abs(QUINTIC - [0, 0, 0, 10, -15, 6]).max():.1e}")
septic = poly_from_boundary(REST + [(0, 2, 0.0), (1, 2, 0.0), (0, 3, 0.0), (1, 3, 0.0)], 7)
print(f"one more pair (zero jerk at both ends) gives degree 7: {' '.join(f'{c:g}' for c in np.round(septic, 9))}")

print("\n== Step 2. Peaks of the normalised shapes (100001 points on [0, 1]) ==")
print(f"{'shape':>8s} {'max ds/du':>10s} {'max d2s/du2':>12s} {'at u':>7s} {'max |d3s/du3| inside':>21s}")
for name, c in (("cubic", CUBIC), ("quintic", QUINTIC)):
    acc = poly_eval(c, U, 2)
    print(f"{name:>8s} {poly_eval(c, U, 1).max():10.4f} {acc.max():12.4f} {U[acc.argmax()]:7.4f} "
          f"{np.abs(poly_eval(c, U, 3)).max():21.4f}")
print(f"formulas: cubic 1.5, 6 at u = 0; quintic 1.875, 10 sqrt(3)/3 = {10 * np.sqrt(3) / 3:.4f} "
      f"at u = (3 - sqrt(3))/6 = {(3 - np.sqrt(3)) / 6:.4f}, jerk 60")

print("\n== Step 3. The trapezoid (ta = 1/3) ==")
tr = np.array([trapezoid(u) for u in U])
integral = np.concatenate([[0], np.cumsum((tr[1:, 1] + tr[:-1, 1]) / 2 * np.diff(U))])
print(f"s(1) = {tr[-1, 0]:.6f}, peak ds/du = {tr[:, 1].max():.4f}, peak |d2s/du2| = {np.abs(tr[:, 2]).max():.4f}")
print(f"max |s - integral of ds/du| = {np.abs(tr[:, 0] - integral).max():.1e} (s is continuous; d2s/du2 jumps 0 -> 4.5)")

print("\n== Step 4. Reference peaks of the Panda move (largest joint 0.6 rad) ==")
print(f"{'T (s)':>6s} {'shape':>9s} {'exact acc':>10s} {'2 ms acc':>9s} {'exact jerk':>11s} {'2 ms jerk':>10s}")
DT = 0.002
for T in (0.25, 0.5, 1.0, 2.0):
    for name, shape, peak_acc, peak_jerk in (("cubic", cubic, 6.0, np.inf), ("quintic", quintic, 10 * np.sqrt(3) / 3, 60.0)):
        ts = np.arange(0, T + DT / 2, DT)
        acc = np.array([joint_reference(shape, t, T, 0.0, 0.6)[2] for t in ts])
        jerk = np.abs(np.diff(acc) / DT).max() if name == "quintic" else np.inf
        print(f"{T:6.2f} {name:>9s} {0.6 * peak_acc / T ** 2:10.2f} {np.abs(acc).max():9.2f} "
              f"{0.6 * peak_jerk / T ** 3:11.2f} {jerk:10.2f}")
acc = np.array([joint_reference(trapezoid, t, 1.0, 0.0, 0.6)[2] for t in np.arange(0, 1.001, DT)])
print(f"  1.00 trapezoid {0.6 * 4.5:10.2f} {np.abs(acc).max():9.2f}")

print("\n== Step 5. Track the quintic on the Panda (T = 1 s, the Experiment's PD + qfrc_bias) ==")
m = load_panda()
tree = tree_from_mujoco(m)
lim = m.actuator_forcerange[:7, 1]
for ff in (False, True):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    err, jerk, prev = 0.0, 0.0, None
    while d.time < 1.5 - 1e-9:
        q_ref, qd_ref, qdd_ref = joint_reference(quintic, d.time, 1.0, Q0, DQ)
        tau = d.qfrc_bias[:7] + KP * (q_ref - d.qpos[:7]) + KD * (qd_ref - d.qvel[:7])
        if ff:                                     # lesson 11's mass matrix, minus the passive joint damping
            tau = tau + mass_matrix(tree, d.qpos)[:7, :7] @ qdd_ref - d.qfrc_passive[:7]
        d.qfrc_applied[:7] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        if prev is not None:
            jerk = max(jerk, np.abs(d.qacc[:7] - prev).max() / m.opt.timestep)
        prev = d.qacc[:7].copy()
        if d.time <= 1.0:
            err = max(err, np.abs(joint_reference(quintic, d.time, 1.0, Q0, DQ)[0] - d.qpos[:7]).max())
    print(f"{'PD + M qdd_ref - passive' if ff else 'PD + gravity':<25s} tracking error {np.degrees(err):.2f} deg, "
          f"robot peak jerk {jerk:.0f} rad/s^3")

print("\n== Use it. SciPy splines through the two end points (T = 1 s, 0 -> 0.6 rad) ==")
ts = np.linspace(0, 1, 501)
cs = CubicSpline([0, 1], [0, 0.6], bc_type="clamped")                     # zero speed at both ends
qs = make_interp_spline([0, 1], [0, 0.6], k=5, bc_type=([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)]))
print(f"CubicSpline(bc_type='clamped') vs cubic: max diff {max(abs(cs(t) - joint_reference(cubic, t, 1, 0, 0.6)[0]) for t in ts):.1e} rad")
print(f"make_interp_spline(k=5, zero speed and acceleration) vs quintic: max diff "
      f"{max(abs(qs(t) - joint_reference(quintic, t, 1, 0, 0.6)[0]) for t in ts):.1e} rad, peak jerk {np.abs(qs(ts, 3)).max():.2f} rad/s^3")
