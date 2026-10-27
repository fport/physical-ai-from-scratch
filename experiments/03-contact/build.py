"""Lesson 03 · Build it: MuJoCo's soft contact as a spring-damper, rebuilt by hand and checked against MuJoCo.

The test body is the Experiment's single-contact check: a 5 cm radius, 1 kg sphere on a plane, dt = 2 ms,
with condim 1 (frictionless) so that one contact is one constraint row.

Step 1  Impedance d(r) and the reference spring K, B from solimp/solref vs efc_KBIP.
Step 2  One contact by hand: a_ref vs efc_aref, then a 10 cm drop vs mj_step.
Step 3  Resting penetration from d k r = (1 - d) a_u vs the resting sphere, every timeconst.
Step 4  Stability: growth per step of the contact spring vs what the drop does.

The functions live in physai/sim.py (the lesson shows them step by step).
Run:  uv run python experiments/03-contact/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.sim import contact_accel, impedance, resting_penetration, semi_implicit_euler, solref_kb, step_growth  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
H, R, G = 0.002, 0.05, 9.81
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2)                   # MuJoCo's default


def sphere(tc, dr, z0, refsafe=True):
    m = mujoco.MjModel.from_xml_string(f"""<mujoco>
  <option timestep="{H}"><flag refsafe="{'enable' if refsafe else 'disable'}"/></option>
  <worldbody><geom type="plane" size="1 1 .1" solref="{tc} {dr}" condim="1"/>
  <body pos="0 0 {z0}"><freejoint/><geom type="sphere" size="{R}" mass="1" solref="{tc} {dr}" condim="1"/></body>
  </worldbody></mujoco>""")
    return m, mujoco.MjData(m)


print("\n== Step 1. Impedance and reference spring vs efc_KBIP (sphere placed at penetration r) ==")
print(f"{'timeconst':>9s} {'r (mm)':>7s} {'d ours':>8s} {'d MuJoCo':>9s} {'K ours':>10s} {'K MuJoCo':>10s} {'B ours':>8s} {'B MuJoCo':>9s}")
for tc, r in ((0.001, 0.0005), (0.004, 0.0005), (0.02, 0.0002), (0.02, 0.0005), (0.02, 0.002), (0.1, 0.0005)):
    m, d = sphere(tc, 1, R - r)
    mujoco.mj_forward(m, d)
    k, b = solref_kb((tc, 1), SOLIMP[1], H)
    K, B, imp, _ = d.efc_KBIP[0]
    print(f"{tc:9.3f} {r * 1000:7.1f} {impedance(r, SOLIMP):8.4f} {imp:9.4f} {k:10.1f} {K:10.1f} {b:8.2f} {B:9.2f}")

print("\n== Step 2. One contact by hand vs MuJoCo ==")
m, d = sphere(0.02, 1, R - 0.0005)
d.qvel[2] = -0.3
mujoco.mj_forward(m, d)
k, b = solref_kb((0.02, 1), SOLIMP[1], H)
dist, vel = d.efc_pos[0], d.efc_vel[0]
print(f"at r = 0.5 mm, v = -0.3 m/s: a_ref ours {-b * vel - k * impedance(-dist, SOLIMP) * dist:.6f} m/s^2 · "
      f"efc_aref {d.efc_aref[0]:.6f} m/s^2")
print("10 cm drop, 2 s:   peak penetration (mm)   rebound (mm)   max |z_ours - z_mujoco| (m)")


def drop(tc, dr, refsafe=True, steps=1000):
    """Our 1-D sphere and MuJoCo's, both dropped from 10 cm above the plane."""
    m, d = sphere(tc, dr, R + 0.10, refsafe)
    z, v, ours, mj = R + 0.10, 0.0, [], []
    for _ in range(steps):
        z, v = semi_implicit_euler(z, v, lambda x, u: contact_accel(x - R, u, -G, (tc, dr), SOLIMP, H, refsafe), H)
        mujoco.mj_step(m, d)
        ours.append(z)
        mj.append(d.qpos[2])
    return np.array(ours), np.array(mj)


for dr in (1.0, 0.3, 0.1):
    ours, mj = drop(0.02, dr)
    low = int(np.argmin(ours))
    print(f"dampratio {dr:3.1f}:       {(R - ours.min()) * 1000:9.3f} {max(0.0, ours[low:].max() - R) * 1000:15.3f} "
          f"{np.abs(ours - mj).max():18.1e}")

print("\n== Step 3. Resting penetration: bisection on d k r = (1 - d) a_u vs the resting sphere (mm) ==")
print(f"{'timeconst':>9s} {'ours':>8s} {'MuJoCo, 2 s at rest':>20s}")
for tc in (0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1):
    m, d = sphere(tc, 1, R)
    while d.time < 2.0 - 1e-9:
        mujoco.mj_step(m, d)
    print(f"{tc:9.3f} {resting_penetration(G, (tc, 1), SOLIMP, H) * 1000:8.4f} {(R - d.qpos[2]) * 1000:20.4f}")

print("\n== Step 4. Growth per step of the linearized contact spring (d = 0.95) vs the 10 cm drop ==")
print(f"{'timeconst':>9s} {'dampratio':>9s} {'refsafe':>7s} {'h*w':>5s} {'growth':>7s}   highest bottom (m, from 0.100): ours / MuJoCo")
for tc, dr, rs in ((0.0005, 1, False), (0.001, 1, False), (0.002, 1, False), (0.003, 1, False), (0.001, 1, True),
                   (0.02, 0.1, True), (0.02, 0.07, True), (0.02, 0.05, True)):
    k, b = solref_kb((tc, dr), SOLIMP[1], H, rs)
    dw = SOLIMP[1]
    growth = step_growth(semi_implicit_euler, H, dw * dw * k, dw * b)   # a = -d b v - d^2 k r near rest
    ours, mj = drop(tc, dr, rs)
    print(f"{tc:9.4f} {dr:9.2f} {'on' if rs else 'off':>7s} {H * dw * np.sqrt(k):5.2f} {growth:7.3f}   "
          f"{ours.max() - R:.3f} / {mj.max() - R:.3f}")
