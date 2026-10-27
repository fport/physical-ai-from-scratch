"""Lesson 02 · Timestep: why did the cube go through the table?

A 5 cm cube is dropped from 1 m onto a 1 cm thick tabletop.
Sweep: MuJoCo's timestep (default contact, then a stiffer contact).
Then the same drop with every integrator.

Run:  python experiment.py      (~1 s on Apple Silicon)
"""
import mujoco
import numpy as np

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <geom name="table" type="box" size="0.12 0.12 0.005" pos="0 0 0.295"/>
    <body name="cube" pos="0 0 1.30">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

TABLE_MID = 0.295                     # mid-plane of the tabletop (m)
TABLE_TOP, TABLE_BOTTOM = 0.300, 0.290
HALF = 0.025                          # cube half-size (m)
DT_MS = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
INTEGRATORS = ["EULER", "IMPLICITFAST", "RK4", "DISCRETE"]


def drop(dt_ms, integrator="EULER", solref=None, sim_sec=1.5, probe=None):
    xml = XML                              # contact solref = weighted mean of both geoms: set it on both
    if solref is not None:
        xml = xml.replace('<geom name="table"', f'<geom name="table" solref="{solref}"')
        xml = xml.replace('mass="0.1"', f'mass="0.1" solref="{solref}"')
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep = dt_ms / 1000          # <- the swept number
    m.opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + integrator)
    d = mujoco.MjData(m)
    table = m.geom("table").id
    z, contact_t = [], None
    while d.time < sim_sec - 1e-12:
        mujoco.mj_step(m, d)
        z.append(d.body("cube").xpos[2])
        hits = [c for c in d.contact[: d.ncon] if table in (c.geom1, c.geom2)]
        if hits and contact_t is None:
            contact_t = d.time
        if hits and probe is not None:
            c = min(hits, key=lambda c: c.dist)
            sign = 1.0 if c.geom1 == table else -1.0        # contact normal points from geom1 to geom2
            nz = sign * c.frame[2]
            if "t" not in probe:
                probe.update(t=d.time, depth_cm=-c.dist * 100, normal_z_on_cube=nz)
            if nz < 0 and "flip_t" not in probe:
                probe.update(flip_t=d.time, flip_centre_cm=d.body("cube").xpos[2] * 100)
    z = np.array(z)
    overlap = (z - HALF < TABLE_TOP) & (z + HALF > TABLE_BOTTOM)
    first = float(z[np.argmax(overlap)]) if overlap.any() else None
    passed = bool(z[-1] < 0.25)
    lowest = None if passed else float(z[np.argmax(z - HALF < TABLE_TOP):].min())
    return dict(first=first, contact=contact_t is not None, passed=passed, lowest=lowest)


def cm(x):
    return "   -  " if x is None else f"{x * 100:6.2f}"


def side(x):
    return "  -  " if x is None else ("below" if x < TABLE_MID else "above")


def main():
    print("mujoco", mujoco.__version__, "· numpy", np.__version__)
    v = np.sqrt(2 * 9.81 * (1.30 - HALF - TABLE_TOP))
    print(f"impact speed on the tabletop (free fall): {v:.2f} m/s · tabletop mid-plane {TABLE_MID * 100:.1f} cm\n")
    print("sweep: timestep (Euler)")
    print(" dt(ms) steps/s step(cm) | default solref 0.02 1: centre at 1st overlap(cm)  contact?  lowest centre(cm)"
          "  result       | solref 0.005 1: result")
    for dt in DT_MS:
        r = drop(dt)
        s = drop(dt, solref="0.005 1")
        print(f" {dt:5.1f} {1000 / dt:7.0f} {v * dt / 10:7.2f}  |  {cm(r['first'])} ({side(r['first'])})"
              f"            {'yes' if r['contact'] else 'no '}       {cm(r['lowest'])}"
              f"         {'WENT THROUGH' if r['passed'] else 'stayed      '} |  {'WENT THROUGH' if s['passed'] else 'stayed'}")

    for dt in (10, 1):
        probe = {}
        drop(dt, probe=probe)
        print(f"\ndt {dt:2d} ms: first table contact at t = {probe['t']:.3f} s · depth {probe['depth_cm']:.2f} cm · "
              f"z of the contact normal acting on the cube {probe['normal_z_on_cube']:+.2f} · "
              f"normal first points down at t = {probe['flip_t']:.3f} s, cube centre {probe['flip_centre_cm']:.2f} cm", end="")
    print("\n\nsame drop, every integrator (default solref): lowest centre if it stayed (cm) / result")
    for dt in [1, 2, 8, 10, 25]:
        cells = []
        for name in INTEGRATORS:
            r = drop(dt, integrator=name)
            cells.append(f"{name.lower():12s} {cm(r['lowest'])} {'THROUGH' if r['passed'] else 'stayed '}")
        print(f"  dt {dt:2d} ms · " + " · ".join(cells))


if __name__ == "__main__":
    main()
