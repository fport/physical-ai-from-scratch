"""Lesson 01 · The world is a text file: a 15-line scene.

In the same file we make one cube 10x heavier and drop both cubes at once.
Then we add air (density) to the file.

Run:  python experiment.py
"""
import mujoco
import numpy as np

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
</mujoco>"""
LINES = XML.split("\n")
TABLE_TOP, HALF = 0.40, 0.10


def make_xml(light_kg=0.1, heavy_kg=1.0, air=False):
    lines = list(LINES)
    lines[7] = lines[7].replace('mass=".1"', f'mass="{light_kg:g}"')
    lines[11] = lines[11].replace('mass=".1"', f'mass="{heavy_kg:g}"')
    if air:
        lines[1] = ' <option gravity="0 0 -9.81" density="1.2"/>'   # air density, kg/m³
    return "\n".join(lines)


def drop(xml, sim_sec=1.0):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    adr = {n: m.jnt_qposadr[m.body_jntadr[m.body(n).id]] for n in ("light", "heavy")}
    zs = {n: [d.qpos[a + 2]] for n, a in adr.items()}
    while d.time < sim_sec - 1e-9:
        mujoco.mj_step(m, d)
        for n, a in adr.items():
            zs[n].append(d.qpos[a + 2])
    out = {}
    for n, z in zs.items():
        bottom = np.array(z) - HALF
        i = int(np.argmax(bottom <= TABLE_TOP))          # step where the bottom face reaches the table
        t_cross = (i - 1 + (bottom[i - 1] - TABLE_TOP) / (bottom[i - 1] - bottom[i])) * dt
        out[n] = dict(step=i, cross_ms=t_cross * 1000, z=np.array(z))
    k = out["heavy"]["step"]
    out["gap_cm"] = (out["light"]["z"][k] - out["heavy"]["z"][k]) * 100
    out["diff_ms"] = out["light"]["cross_ms"] - out["heavy"]["cross_ms"]
    return out


def main():
    m = mujoco.MjModel.from_xml_string(XML)
    print("mujoco", mujoco.__version__, "·", len(LINES), "lines")
    print(f"defaults: timestep {m.opt.timestep} s · density {m.opt.density} · viscosity {m.opt.viscosity}\n")
    for name, xml in [("vacuum (100 g, 1 kg)", make_xml(0.1, 1.0)),
                      ("air    (100 g, 1 kg)", make_xml(0.1, 1.0, air=True))]:
        r = drop(xml)
        print(f"{name} · light step {r['light']['step']} · heavy step {r['heavy']['step']} · "
              f"diff {r['diff_ms']:.2f} ms · light is {r['gap_cm']:.2f} cm higher when heavy lands")
    print(f"analytic fall time sqrt(2*0.40/9.81) = {np.sqrt(2 * 0.40 / 9.81):.4f} s\n")
    print("sweep: mass of the light cube (heavy is always 1 kg)")
    for kg in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        v = drop(make_xml(kg, 1.0))["diff_ms"]
        a = drop(make_xml(kg, 1.0, air=True))["diff_ms"]
        print(f"  light {kg * 1000:6.0f} g · vacuum {v:6.2f} ms · air {a:6.2f} ms")


if __name__ == "__main__":
    main()
