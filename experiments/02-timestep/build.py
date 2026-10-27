"""Lesson 02 · Build it: three time steppers, their stability limits, and the Experiment's first-overlap column.

Step 1  Explicit Euler, semi-implicit Euler and RK4 on a spring vs mj_step (Euler and RK4 integrators).
Step 2  Energy after 1000 steps: explicit Euler grows by (1 + h^2 w^2) per step, the others do not.
Step 3  Growth per step vs h*w: the h*w < 2 limit, and MuJoCo crossing it.
Step 4  Free fall onto the thin tabletop: centre at the first overlapping step, ours vs MuJoCo, every dt.

The functions live in physai/sim.py (the lesson shows them step by step).
Run:  uv run python experiments/02-timestep/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.sim import explicit_euler, first_overlap, rk4, semi_implicit_euler, step_growth  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")


def spring(omega, h, integrator):
    """A 1 kg mass on a slide joint with stiffness omega^2, released from x = 0.1 m (no gravity, no contact)."""
    m = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{h}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody><body><joint type="slide" axis="1 0 0" stiffness="{omega**2}"/>
  <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qpos[0] = 0.1
    return m, d


def run_ours(step, omega, h, n):
    x, v, xs = 0.1, 0.0, [0.1]
    for _ in range(n):
        x, v = step(x, v, lambda q, u: -omega**2 * q, h)
        xs.append(x)
    return np.array(xs), x, v


def run_mj(omega, h, integrator, n):
    m, d = spring(omega, h, integrator)
    xs = [d.qpos[0]]
    for _ in range(n):
        mujoco.mj_step(m, d)
        xs.append(d.qpos[0])
    return np.array(xs)


OMEGA = 2 * np.pi                                      # 1 Hz spring
print("\n== Step 1. Our steppers vs mj_step on a 1 Hz spring (h = 10 ms, 500 steps) ==")
for name, step, integ in (("semi_implicit_euler", semi_implicit_euler, "Euler"), ("rk4", rk4, "RK4")):
    err = np.abs(run_ours(step, OMEGA, 0.01, 500)[0] - run_mj(OMEGA, 0.01, integ, 500)).max()
    print(f"{name:20s} vs integrator={integ:5s}: max |x_ours - qpos| = {err:.1e} m")
print("explicit_euler: no MuJoCo integrator moves with the old velocity; checked against its formula in Step 2")

print("\n== Step 2. Energy after 1000 steps, relative to the start (h = 10 ms, h*w = 0.063) ==")
h, n = 0.01, 1000
e0 = 0.5 * OMEGA**2 * 0.1**2
for name, step in (("explicit_euler", explicit_euler), ("semi_implicit_euler", semi_implicit_euler), ("rk4", rk4)):
    _, x, v = run_ours(step, OMEGA, h, n)
    print(f"{name:20s} E/E0 = {(0.5 * v * v + 0.5 * OMEGA**2 * x * x) / e0:.6f}")
print(f"formula for explicit Euler (1 + h^2 w^2)^n = {(1 + (h * OMEGA) ** 2) ** n:.6f}")

print("\n== Step 3. Growth per step on the undamped spring vs h*w ==")
print(f"{'h*w':>5s} {'explicit_euler':>15s} {'semi_implicit':>14s} {'rk4':>10s}   max |x| over 10 steps: ours (semi-implicit) / MuJoCo Euler")
for hw in (0.5, 1.0, 1.9, 2.1, 2.8, 2.9):
    g = [step_growth(s, hw / OMEGA, OMEGA**2) for s in (explicit_euler, semi_implicit_euler, rk4)]
    ours = np.abs(run_ours(semi_implicit_euler, OMEGA, hw / OMEGA, 10)[0]).max()
    mj = np.abs(run_mj(OMEGA, hw / OMEGA, "Euler", 10)).max()
    print(f"{hw:5.1f} {g[0]:15.6f} {g[1]:14.6f} {g[2]:10.6f}   {ours:.4g} m / {mj:.4g} m (started at 0.1)")

print("\n== Step 4. Drop onto the 1 cm tabletop: centre at the first overlapping step (cm) ==")
XML = """<mujoco><worldbody><geom type="plane" size="2 2 0.1"/>
  <geom name="table" type="box" size="0.12 0.12 0.005" pos="0 0 0.295"/>
  <body name="cube" pos="0 0 1.30"><freejoint/><geom type="box" size="0.025 0.025 0.025" mass="0.1"/></body>
</worldbody></mujoco>"""
print(f"{'dt (ms)':>7s} {'ours':>8s} {'MuJoCo':>8s}  side of the mid-plane (29.5 cm)")
for dt_ms in (0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50):
    ours = first_overlap(1.30, dt_ms / 1000, top=0.300, bottom=0.290, half=0.025)
    m = mujoco.MjModel.from_xml_string(XML)
    m.opt.timestep = dt_ms / 1000
    d, mj = mujoco.MjData(m), None
    while d.time < 1.5 and mj is None:
        mujoco.mj_step(m, d)
        z = d.body("cube").xpos[2]
        if z - 0.025 < 0.300 and z + 0.025 > 0.290:
            mj = z
    if ours is None:
        print(f"{dt_ms:7g} {'-':>8s} {'-' if mj is None else f'{mj * 100:.2f}':>8s}  no step overlaps: missed contact")
    else:
        side = "below: pushed out downward" if ours < 0.295 else "above"
        print(f"{dt_ms:7g} {ours * 100:8.2f} {mj * 100:8.2f}  {side}")
