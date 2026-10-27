"""Lesson 40 · Build it: the bookkeeping math of a real SO-101 deployment, checked in simulation.

Step 1  Encoder ticks <-> LeRobot degrees: the round trip and its one-tick error.
Step 2  Degrees -> model radians: fit sign and offset from two known poses.
Step 3  Gravity torque by hand along the preflight trajectory vs MuJoCo's qfrc_bias.
Step 4  The payload budget: torque per kg from the Jacobian, and the largest payload under 80% / 100% of 2.94 N m.
Step 5  The static preflight checks: range margin and largest goal step per tick, at several speeds.

The functions live in physai/sim2real.py; lesson 07's frame_tree and lesson 09's jacobian are reused.
Nothing here is measured on the real arm. Run:  uv run python experiments/40-real-so101/build.py  (~2 s)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.kinematics import chain_from_mujoco, jacobian  # noqa: E402
from physai.sim2real import deg_to_ticks, fit_calibration, gravity_torques, max_payload, ticks_to_deg  # noqa: E402
from physai.spatial import frame_tree  # noqa: E402

spec = importlib.util.spec_from_file_location("exp40", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)                       # trajectory() and build(payload) of the Experiment

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

print("\n== Step 1. Encoder ticks <-> degrees (12-bit encoder, calibrated range 1000-3000) ==")
lo_t, hi_t = 1000, 3000
raw = np.arange(lo_t, hi_t + 1)
deg = ticks_to_deg(raw, lo_t, hi_t)
print(f"middle of the range {ticks_to_deg(2000, lo_t, hi_t):.3f} deg, ends {deg[0]:.2f} / {deg[-1]:.2f} deg, "
      f"one tick = {360 / 4095:.4f} deg")
back = np.array([deg_to_ticks(v, lo_t, hi_t) for v in deg])
print(f"ticks -> deg -> ticks: {int((back == raw).sum())} of {len(raw)} positions come back exactly")
goal = np.linspace(-80, 80, 16001)                 # goals a policy might send, every 0.01 deg
sent = np.array([ticks_to_deg(deg_to_ticks(v, lo_t, hi_t), lo_t, hi_t) for v in goal])
err = sent - goal
print(f"deg -> ticks -> deg for 16,001 goals: error from {err.min():.4f} to {err.max():.4f} deg "
      f"(int() of a positive tick count rounds down: up to one tick, always low)")

print("\n== Step 2. Degrees -> model radians: sign and offset from two known poses ==")
true_a, true_b = -1.0, np.radians(7.0)             # an invented mounting: reversed direction, 7 deg offset
poses_model = np.array([-0.30, 0.90])              # two poses whose model angles you know (e.g. against a jig)
readings = np.array([ticks_to_deg(deg_to_ticks(np.degrees((q - true_b) / true_a), lo_t, hi_t), lo_t, hi_t)
                     for q in poses_model])        # what the bus reports, quantized to whole ticks
a, b = fit_calibration(readings, poses_model)
print(f"readings {readings.round(3)} deg -> fitted a = {a:.4f} (true {true_a}), b = {np.degrees(b):.3f} deg (true 7.000)")
test = np.radians(np.linspace(-40, 40, 9))
pred = a * np.radians([ticks_to_deg(deg_to_ticks(np.degrees((q - true_b) / true_a), lo_t, hi_t), lo_t, hi_t)
                       for q in test]) + b
print(f"9 test poses: max error {np.degrees(np.abs(pred - test)).max():.3f} deg (one tick is 0.088 deg)")

print("\n== Step 3. Gravity torque by hand along the 303 goals vs qfrc_bias ==")
m = exp.build(0.0)
d = mujoco.MjData(m)
goals = exp.trajectory(1.0)
worst, peak, peak_j = 0.0, 0.0, 0
tau_all = []
for g in goals:
    d.qpos[:6], d.qvel[:] = g, 0
    mujoco.mj_forward(m, d)
    tau = gravity_torques(m, d.qpos, frame_tree(m, d.qpos))[:5]
    tau_all.append(tau)
    worst = max(worst, np.abs(tau + d.qfrc_bias[:5]).max())
    if np.abs(tau).max() > peak:
        peak, peak_j = np.abs(tau).max(), int(np.abs(tau).argmax())
tau_all = np.array(tau_all)
print(f"goals {len(goals)}, max |ours + qfrc_bias| = {worst:.1e} N m")
print(f"largest static gravity torque {peak:.3f} N m at {NAMES[peak_j]} (Experiment: 0.446 N m)")

print("\n== Step 4. The payload budget at the gripper frame ==")
chain = chain_from_mujoco(m, "gripperframe")
per_kg = np.array([jacobian(chain, g[:5])[:3].T @ [0, 0, -9.81] for g in goals])   # torque per kg of payload
print(f"{'payload kg':>10s} {'ours N m':>9s} {'MuJoCo model with payload N m':>30s}")
for p in (0.2, 0.4, 0.6, 0.8):
    mp, dp = exp.build(p), None
    dp = mujoco.MjData(mp)
    ref = 0.0
    for g in goals:
        dp.qpos[:6], dp.qvel[:] = g, 0
        mujoco.mj_forward(mp, dp)
        ref = max(ref, np.abs(dp.qfrc_bias[:5]).max())
    print(f"{p:10.1f} {np.abs(tau_all + p * per_kg).max():9.3f} {ref:30.3f}")
for frac in (0.8, 1.0):
    print(f"largest payload with static torque <= {frac:.0%} of 2.94 N m: {max_payload(tau_all, per_kg, frac * 2.94):.3f} kg")

print("\n== Step 5. Static preflight: range margin and goal step per 30 fps tick ==")
lo, hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
for speed in (0.5, 1.0, 2.0, 4.0):
    gl = exp.trajectory(speed)
    margin = np.degrees(np.minimum(gl - lo, hi - gl).min())
    step = np.abs(np.diff(gl, axis=0)).max()
    print(f"x{speed:<4g} goals {len(gl):3d}, range margin {margin:.2f} deg, largest step {step:.3f} rad/tick -> "
          f"{'PASS' if step <= 0.10 and margin >= 5.0 else 'FAIL'}")
