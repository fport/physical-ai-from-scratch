"""Lesson 04 · Build it: Coulomb friction and MuJoCo's friction cones, rebuilt by hand and checked against MuJoCo.

Step 1  The slip test tan(theta) > mu, and the Experiment's slip criterion applied to an ideal Coulomb block.
Step 2  Sliding acceleration g (sin - mu cos) vs MuJoCo (8 cm, 1 kg cube; a ramp = gravity tilted by theta).
Step 3  Pyramid rows J_n +- mu J_t vs efc_J, and the contact force from the edge forces vs mj_contactForce.
Step 4  The pyramid's friction limit by direction: our formula vs a SciPy linear program, and MuJoCo's forces.

The functions live in physai/sim.py (the lesson shows them step by step).
Run:  uv run python experiments/04-friction/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.sim import pyramid_rows, pyramid_to_force, ramp_slip, tangential_limit  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
G = 9.81


def tilted(theta_deg, mu, cone, yaw_deg=0.0):
    """The cube on a flat floor with gravity tilted by theta: the same physics as a ramp, without an edge."""
    t, y = np.radians(theta_deg), np.radians(yaw_deg) / 2
    m = mujoco.MjModel.from_xml_string(f"""<mujoco>
  <option cone="{cone}" gravity="{G * np.sin(t)} 0 {-G * np.cos(t)}"/>
  <worldbody><geom type="plane" size="5 5 .1" friction="{mu} 0.005 0.0001"/>
  <body pos="0 0 .04" quat="{np.cos(y)} 0 0 {np.sin(y)}"><freejoint/>
  <geom type="box" size=".04 .04 .04" mass="1" friction="{mu} 0.005 0.0001"/></body></worldbody></mujoco>""")
    return m, mujoco.MjData(m)


print("\n== Step 1. Slip angle: tan(theta) > mu, then the Experiment's criterion on an ideal Coulomb block ==")
print(f"{'mu':>4s} {'arctan mu':>10s} {'first grid angle that slips':>28s} {'ideal block + criterion':>24s}")
for mu in (0.3, 0.5, 0.8, 1.0):
    ref = np.degrees(np.arctan(mu))
    grid = np.round(np.arange(round((ref - 3) / 0.05) * 0.05, ref + 1 + 1e-9, 0.05), 2)
    first = next(a for a in grid if ramp_slip(np.radians(a), mu)[0])
    accel = np.array([ramp_slip(np.radians(a), mu)[1] for a in grid])
    v1, v2 = 0.75 * accel * 100, 1.75 * accel * 100     # mean speed over 0.5-1.0 s and 1.5-2.0 s from rest, cm/s
    ok = (v2 > 2.0) & (v2 - v1 > 0.5)
    k = next(i for i in range(len(ok)) if ok[i:].all())
    print(f"{mu:4.1f} {ref:9.2f}° {first:27.2f}° {grid[k]:23.2f}°")

print("\n== Step 2. Sliding acceleration (mu = 0.5): g (sin - mu cos) vs MuJoCo, fit over 0.2-1.0 s ==")
print(f"{'theta':>6s} {'ours (m/s^2)':>13s} {'elliptic':>9s} {'pyramidal':>10s}")
for th in (28.0, 30.0, 35.0, 45.0):
    row = []
    for cone in ("elliptic", "pyramidal"):
        m, d = tilted(th, 0.5, cone)
        ts, xs = [], []
        while d.time < 1.0 - 1e-9:
            mujoco.mj_step(m, d)
            ts.append(d.time)
            xs.append(d.qpos[0])
        ts, xs = np.array(ts), np.array(xs)
        row.append(2 * np.polyfit(ts[ts > 0.2], xs[ts > 0.2], 2)[0])
    print(f"{th:5.1f}° {ramp_slip(np.radians(th), 0.5)[1]:13.4f} {row[0]:9.4f} {row[1]:10.4f}")

print("\n== Step 3. Pyramid rows and edge forces vs MuJoCo (cube holding at 20°, mu = 0.5, pyramidal) ==")
m, d = tilted(20.0, 0.5, "pyramidal")
for _ in range(250):
    mujoco.mj_step(m, d)
mujoco.mj_forward(m, d)
J_all = d.efc_J.reshape(d.nefc, m.nv)
row_err, force_err = 0.0, 0.0
for i in range(d.ncon):
    c = d.contact[i]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, c.pos, m.geom_bodyid[c.geom2])   # the floor has no dofs; the cube is geom2
    Jc = c.frame.reshape(3, 3) @ jp                              # contact frame rows: normal, tangent 1, tangent 2
    a = c.efc_address
    row_err = max(row_err, np.abs(pyramid_rows(Jc[0], Jc[1:], c.friction[:2]) - J_all[a:a + 4]).max())
    f = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f)
    force_err = max(force_err, np.abs(pyramid_to_force(d.efc_force[a:a + 4], c.friction[:2]) - f[:3]).max())
print(f"{d.ncon} contacts x 4 rows = {d.nefc} rows · max |our rows - efc_J| = {row_err:.1e}")
print(f"max |force from our edge sum - mj_contactForce| = {force_err:.1e} N")

print("\n== Step 4. Largest friction force per unit normal force, by sliding direction (mu = 0.5) ==")
print(f"{'phi':>5s} {'elliptic':>9s} {'pyramid (ours)':>15s} {'pyramid (linprog)':>18s}")
for phi in (0, 15, 30, 45):
    p = np.radians(phi)
    edges = np.array([[1, 0.5, 0], [1, -0.5, 0], [1, 0, 0.5], [1, 0, -0.5]])     # n +- mu t_i, as columns below
    # maximize the friction force along (cos phi, sin phi) with unit normal force and nothing sideways
    res = linprog(-(edges[:, 1] * np.cos(p) + edges[:, 2] * np.sin(p)), bounds=[(0, None)] * 4,
                  A_eq=np.array([edges[:, 0], -edges[:, 1] * np.sin(p) + edges[:, 2] * np.cos(p)]), b_eq=[1.0, 0.0])
    print(f"{phi:4d}° {tangential_limit(p, 0.5, 'elliptic'):9.4f} {tangential_limit(p, 0.5, 'pyramidal'):15.4f} "
          f"{-res.fun:18.4f}")
print("MuJoCo: friction impulse / normal impulse on the cube over 0.2-1.0 s (world frame)")
for th, label in ((20.0, "holding at 20° (tan 20° = 0.3640)"), (35.0, "sliding at 35° (mu = 0.5)")):
    for cone in ("elliptic", "pyramidal"):
        m, d = tilted(th, 0.5, cone)
        total = np.zeros(3)
        while d.time < 1.0 - 1e-9:
            mujoco.mj_step(m, d)
            if d.time <= 0.2:
                continue
            for i in range(d.ncon):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                total += d.contact[i].frame.reshape(3, 3).T @ f[:3]    # contact frame -> world
        print(f"  {label:34s} {cone:9s} {np.hypot(total[0], total[1]) / total[2]:.4f}")
