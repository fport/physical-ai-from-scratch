"""Lesson 02 · Timestep: why did the cube go through the table?

A 5 cm cube is dropped from 1 m onto a 1 cm thick tabletop.
The only number that changes is MuJoCo's timestep.

Run:  python experiment.py
"""
import json
import sys

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
DT_MS = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]


def drop(dt_ms, sim_sec=1.5):
    m = mujoco.MjModel.from_xml_string(XML)
    m.opt.timestep = dt_ms / 1000          # <- the only number we change
    d = mujoco.MjData(m)
    table = m.geom("table").id
    z, contact_t = [], None
    while d.time < sim_sec - 1e-12:
        mujoco.mj_step(m, d)
        z.append(d.body("cube").xpos[2])
        if contact_t is None and any(table in (c.geom1, c.geom2) for c in d.contact[: d.ncon]):
            contact_t = d.time
    return np.array(z), contact_t


def first_overlap_center(z):
    """Height of the cube centre at the first step where it overlaps the tabletop."""
    overlap = (z - HALF < TABLE_TOP) & (z + HALF > TABLE_BOTTOM)
    return float(z[np.argmax(overlap)]) if overlap.any() else None


def main():
    print("mujoco", mujoco.__version__)
    v = np.sqrt(2 * 9.81 * (1.30 - HALF - TABLE_TOP))
    print(f"impact speed on the tabletop (free fall): {v:.2f} m/s\n")
    rows = []
    for dt in DT_MS:
        z, contact_t = drop(dt)
        passed = bool(z[-1] < 0.25)
        center = first_overlap_center(z)
        rows.append(dict(dt_ms=dt, steps_per_sec=1000 // dt, step_cm=round(v * dt / 10, 2),
                         final_z_cm=round(z[-1] * 100, 2), passed=passed,
                         first_overlap_center_cm=None if center is None else round(center * 100, 2),
                         first_table_contact_s=None if contact_t is None else round(contact_t, 3)))
        side = "-" if center is None else ("below" if center < TABLE_MID else "above")
        print(f"dt {dt:2d} ms · {1000 // dt:3d} steps/s · step {v * dt / 10:4.2f} cm · "
              f"centre at first overlap {side:5s} · contact {'no ' if contact_t is None else 'yes'} · "
              f"{'WENT THROUGH' if passed else 'stayed'}")
    if "--json" in sys.argv:
        print(json.dumps(dict(mujoco=mujoco.__version__, rows=rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
