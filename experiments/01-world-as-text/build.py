"""Lesson 01 · Build it: what the compiler fills in, and one fall, rebuilt by hand and checked against MuJoCo.

Step 1  Mass and inertia of solid shapes vs body_mass / body_inertia.
Step 2  One body from two parts (parallel-axis theorem) vs body_ipos / body_inertia.
Step 3  Free fall with MuJoCo's Euler by hand vs mj_step (the Experiment's vacuum drop).
Step 4  Quadratic drag from the equivalent inertia box vs qfrc_fluid, then the air drop.
Step 5  The Experiment's mass sweep with air, from our own integrator.

The functions live in physai/sim.py (the lesson shows them step by step).
Run:  uv run python experiments/01-world-as-text/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.sim import combine_parts, inertia_box, quadratic_drag, semi_implicit_euler, solid_inertia  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

print("\n== Step 1. Solid shapes: our mass and inertia vs MuJoCo's compiler ==")
SHAPES = [("box", [0.1, 0.1, 0.1], None, 0.1), ("box", [0.1, 0.05, 0.02], 700, None),
          ("sphere", [0.04], 1000, None), ("cylinder", [0.03, 0.1], 2700, None), ("capsule", [0.03, 0.1], 1000, None)]
print(f"{'geom':9s} {'size (m)':>18s} {'mass (kg)':>10s} {'I_xx':>11s} {'I_yy':>11s} {'I_zz':>11s} {'max rel. err':>13s}")
for kind, size, rho, mass in SHAPES:
    attr = f'mass="{mass}"' if mass else f'density="{rho}"'
    m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><body><freejoint/><geom type="{kind}" '
                                       f'size="{" ".join(map(str, size))}" {attr}/></body></worldbody></mujoco>')
    if mass:                                           # a given mass fixes the density: rho = m / V
        rho = mass / solid_inertia(kind, size, 1.0)[0]
    m_ours, i_ours = solid_inertia(kind, size, rho)
    err = max(abs(m_ours / m.body_mass[1] - 1), np.abs(i_ours / m.body_inertia[1] - 1).max())
    print(f"{kind:9s} {str(size):>18s} {m_ours:10.4f} {i_ours[0]:11.4e} {i_ours[1]:11.4e} {i_ours[2]:11.4e} {err:13.1e}")

print("\n== Step 2. One body from two parts: a 1 kg box above a 0.5 kg sphere ==")
m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><freejoint/>
  <geom type="box" size=".05 .05 .05" pos="0 0 .1" mass="1"/>
  <geom type="sphere" size=".05" pos="0 0 -.1" mass="0.5"/></body></worldbody></mujoco>""")
parts = [solid_inertia("box", [0.05] * 3, 1 / 0.001), solid_inertia("sphere", [0.05], 0.5 / (4 / 3 * np.pi * 0.05**3))]
mass, com, inertia = combine_parts([p[0] for p in parts], [[0, 0, 0.1], [0, 0, -0.1]], [p[1] for p in parts])
print(f"mass {mass:.4f} kg (body_mass {m.body_mass[1]:.4f}) · COM z {com[2]:.5f} m (body_ipos z {m.body_ipos[1][2]:.5f})")
print(f"inertia about the COM, diagonal: {np.array2string(np.diag(inertia), precision=6)} kg m^2")
print(f"body_inertia:                    {np.array2string(m.body_inertia[1], precision=6)} kg m^2")
print(f"max rel. error {np.abs(np.diag(inertia) / m.body_inertia[1] - 1).max():.1e}; "
      f"each part alone about the COM would give I_xx {sum(p[1][0] for p in parts):.6f} (no parallel-axis term)")

XML = """<mujoco>
 <option gravity="0 0 -9.81"/>
 <worldbody>
  <geom type="plane" size="2 2 .1"/>
  <geom name="table" type="box" size=".35 .25 .4"/>
  <body name="light" pos="-.13 0 .9">
   <freejoint/>
   <geom type="box" size=".1 .1 .1" mass=".1"/>
  </body>
  <body name="heavy" pos=".13 0 .9">
   <freejoint/>
   <geom type="box" size=".1 .1 .1" mass=".1"/>
  </body>
 </worldbody>
</mujoco>"""                                           # the Experiment's 15-line file
TABLE_TOP, HALF, G = 0.40, 0.10, 9.81


def scene(light_kg, air):
    xml = XML.replace('mass=".1"', f'mass="{light_kg:g}"', 1).replace('mass=".1"', 'mass="1"', 1)
    return xml.replace('<option gravity="0 0 -9.81"/>', '<option gravity="0 0 -9.81" density="1.2"/>') if air else xml


def our_drop(mass, rho, z0=0.9, dt=0.002):
    """Our integrator: gravity plus drag on a 20 cm cube; heights until it crosses the tabletop."""
    box = inertia_box(mass, solid_inertia("box", [HALF] * 3, mass / 0.008)[1])
    accel = lambda z, v: -G + quadratic_drag(np.array([0, 0, v]), box, rho)[2] / mass  # noqa: E731
    zs, z, v = [z0], z0, 0.0
    while z - HALF > TABLE_TOP:
        z, v = semi_implicit_euler(z, v, accel, dt)
        zs.append(z)
    return np.array(zs)


def crossing_ms(zs, dt=0.002):
    """Interpolated time at which the bottom face reaches the tabletop (the Experiment's rule)."""
    i, b = len(zs) - 1, zs - HALF - TABLE_TOP
    return (i - 1 + b[i - 1] / (b[i - 1] - b[i])) * dt * 1000


def mj_heights(xml, body, n):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    adr = m.jnt_qposadr[m.body_jntadr[m.body(body).id]] + 2
    zs = [d.qpos[adr]]
    for _ in range(n):
        mujoco.mj_step(m, d)
        zs.append(d.qpos[adr])
    return np.array(zs)


print("\n== Step 3. Free fall by hand (velocity first, then position) vs mj_step, vacuum ==")
for name, kg in (("light", 0.1), ("heavy", 1.0)):
    ours = our_drop(kg, 0.0)
    mj = mj_heights(scene(0.1, False), name, len(ours) - 1)
    print(f"{name}: lands at step {len(ours) - 1} · max |z_ours - z_mujoco| over the fall {np.abs(ours - mj).max():.1e} m")
print(f"analytic sqrt(2 * 0.40 / 9.81) = {np.sqrt(0.8 / G):.4f} s = step {np.sqrt(0.8 / G) / 0.002:.1f}")

print("\n== Step 4. Drag from the equivalent inertia box vs MuJoCo's qfrc_fluid ==")
box = inertia_box(0.1, solid_inertia("box", [HALF] * 3, 0.1 / 0.008)[1])
m = mujoco.MjModel.from_xml_string(scene(0.1, True))
d = mujoco.MjData(m)
print(f"inertia box of the 100 g cube: {np.array2string(box, precision=4)} m (the geom is 0.2 m)")
dof = m.jnt_dofadr[m.body_jntadr[m.body("light").id]]
for vz in (-1.0, -2.8):
    d.qvel[:] = 0
    d.qvel[dof + 2] = vz
    mujoco.mj_forward(m, d)
    print(f"v_z = {vz:4.1f} m/s · our drag {quadratic_drag(np.array([0, 0, vz]), box, 1.2)[2]:.6f} N · "
          f"qfrc_fluid {d.qfrc_fluid[dof + 2]:.6f} N")
light, heavy = our_drop(0.1, 1.2), our_drop(1.0, 1.2)
k = len(heavy) - 1
print(f"air drop by hand: light lands at step {len(light) - 1}, heavy at step {k} · light is "
      f"{(light[k] - heavy[k]) * 100:.2f} cm higher when heavy lands · delay {crossing_ms(light) - crossing_ms(heavy):.2f} ms")
mj = mj_heights(scene(0.1, True), "light", len(light) - 1)
print(f"max |z_ours - z_mujoco| for the light cube over its fall: {np.abs(light - mj).max():.1e} m")

print("\n== Step 5. The mass sweep with air, from our own integrator (heavy = 1 kg) ==")
t_heavy = crossing_ms(our_drop(1.0, 1.2))
for kg in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    print(f"light {kg * 1000:5.0f} g · delay {crossing_ms(our_drop(kg, 1.2)) - t_heavy:6.2f} ms")
