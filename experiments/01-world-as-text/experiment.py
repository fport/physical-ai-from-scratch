"""Lesson 01 · The world is a text file: a 15-line scene, then a real robot file.

Part A: in the same file we make one cube 10x heavier and drop both cubes at once,
then add air (density) to the file and sweep the light cube's mass.
Part B: inspect the compiled SO-101 model from MuJoCo Menagerie.

Run:  python experiment.py      (~1 s on Apple Silicon)
"""
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

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


def inspect_so101():
    f = path("robotstudio_so101", "so101.xml")
    m = mujoco.MjModel.from_xml_path(f)
    tree = ET.parse(f)
    bodies = list(tree.getroot().iter("body"))
    n_inertial = sum(1 for b in bodies if b.find("inertial") is not None)
    inferred = [b.get("name") for b in bodies if b.find("inertial") is None]
    jtypes = [mujoco.mjtJoint(t).name for t in m.jnt_type]
    print("\nSO-101 (Menagerie robotstudio_so101/so101.xml), compiled")
    print(f"  bodies (incl. world) {m.nbody} · joints {m.njnt} ({', '.join(sorted(set(jtypes)))}) · dof {m.nv} · "
          f"actuators {m.nu} · geoms {m.ngeom} · sensors {m.nsensor}")
    print(f"  bodies with an explicit <inertial> in the file: {n_inertial} of {m.nbody - 1} · inferred from geoms: {', '.join(inferred)}")
    print(f"  total mass {sum(m.body_mass):.4f} kg · timestep {m.opt.timestep} s · "
          f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name} · cone {mujoco.mjtCone(m.opt.cone).name}")
    for a in range(m.nu):
        print(f"  actuator {m.actuator(a).name:13s} gain {mujoco.mjtGain(m.actuator_gaintype[a]).name:12s} "
              f"bias {mujoco.mjtBias(m.actuator_biastype[a]).name:12s} kp {m.actuator_gainprm[a, 0]:.2f} "
              f"kv {-m.actuator_biasprm[a, 2]:.3f} -> <position> servo")
    print("  body masses (kg):", " · ".join(f"{m.body(i).name} {m.body_mass[i]:.4f}" for i in range(1, m.nbody)))

    # Break it: ignore the explicit <inertial> data and infer everything from geoms (density 1000 kg/m³)
    spec = mujoco.MjSpec.from_file(f)
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    m2 = spec.compile()
    print("\nbreak it: same file compiled with inertiafromgeom='true'")
    print(f"  total mass {sum(m2.body_mass):.4f} kg (x{sum(m2.body_mass) / sum(m.body_mass):.2f})")
    for name in ("shoulder", "upper_arm", "wrist"):
        i = m.body(name).id
        print(f"  {name:9s} mass {m.body_mass[i]:.4f} -> {m2.body_mass[i]:.4f} kg · "
              f"largest principal inertia {m.body_inertia[i].max():.3e} -> {m2.body_inertia[i].max():.3e} kg·m²")


def main():
    m = mujoco.MjModel.from_xml_string(XML)
    print("mujoco", mujoco.__version__, "· numpy", np.__version__, "·", len(LINES), "lines")
    print(f"defaults: timestep {m.opt.timestep} s · density {m.opt.density} · viscosity {m.opt.viscosity} · "
          f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name}")
    b = m.body("light").id
    print(f"inertia of the 100 g, 20 cm cube inferred from its geom: {m.body_inertia[b]} kg·m² "
          f"(analytic m*(a^2+a^2)/12 = {0.1 * (0.2 ** 2 + 0.2 ** 2) / 12:.6f})")
    print(f"body quaternion of 'light' (w x y z): {m.body_quat[b]}\n")
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
    k_over_m = 0.5 * 1.2 * 0.2 ** 2 / np.array([0.1, 1.0])     # quadratic drag 0.5*rho*A*v^2, per unit mass
    T = np.sqrt(2 * 0.40 / 9.81)
    lost = k_over_m * 9.81 ** 2 * T ** 4 / 12 * 100
    print(f"hand estimate of the air gap: (k/m)*g^2*T^4/12 = {lost[0]:.2f} cm (light) - {lost[1]:.2f} cm (heavy)"
          f" = {lost[0] - lost[1]:.2f} cm")
    inspect_so101()


if __name__ == "__main__":
    main()
