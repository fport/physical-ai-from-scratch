"""Lesson 05 · Joints and actuators: a gravity-loaded link on a position servo.

A 20 cm, 0.3 kg link hangs on a hinge, held horizontal by a MuJoCo <position>
actuator (built-in PD). Sweep 1: kp -> steady sag and rise time.
Break it: the SO-101 Menagerie servo gains with less and less armature.
Reference: the actuator/joint parameters of the real SO-101 Menagerie model.

Run:  python experiment.py      (~1 s on Apple Silicon)
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

L, MASS, DT = 0.20, 0.30, 0.005      # link length (m), mass (kg), timestep (s) = SO-101 model timestep
TAU_G = MASS * 9.81 * L / 2          # gravity torque at horizontal (N·m)
I_LINK = MASS * L ** 2 / 3           # thin-rod inertia about the hinge (kg·m²)
SO101 = dict(kp=998.22, kv=2.731, damping=0.60, frictionloss=0.052, armature=0.028)   # Menagerie class "sts3215"


def xml(kp, kv=None, dampratio=None, armature=0.0, damping=0.0, frictionloss=0.0,
        integrator="implicitfast", forcerange=None):
    gain = f'kp="{kp}" ' + (f'kv="{kv}"' if kv is not None else f'dampratio="{dampratio}"')
    fr = f'forcerange="{-forcerange} {forcerange}"' if forcerange else ""
    return f"""
<mujoco>
  <option timestep="{DT}" integrator="{integrator}"/>
  <worldbody>
    <body name="link">
      <joint name="j" type="hinge" axis="0 -1 0" armature="{armature}" damping="{damping}" frictionloss="{frictionloss}"/>
      <geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.01" mass="{MASS}"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j" {gain} {fr}/>
  </actuator>
</mujoco>"""


def step_response(model_xml, q0=-0.5, target=0.0, T=3.0):
    """Start at q0 (rad), command target. Returns sag (rad), 10-90 % rise time (s), max |error| (rad), kv used."""
    m = mujoco.MjModel.from_xml_string(model_xml)
    d = mujoco.MjData(m)
    d.qpos[0], d.ctrl[0] = q0, target
    q = [q0]
    while d.time < T - 1e-9:
        mujoco.mj_step(m, d)
        q.append(d.qpos[0])
        if not np.isfinite(d.qpos[0]) or abs(d.qpos[0]) > 100:
            break
    q = np.array(q)
    err_max = float(np.max(np.abs(q - target)))
    if err_max > np.pi or len(q) < int(T / DT):
        return None, None, err_max, -m.actuator_biasprm[0, 2]
    sag = target - q[-int(0.5 / DT):].mean()
    frac = (q - q0) / ((target - sag) - q0)
    t = np.arange(len(q)) * DT

    def cross(level):                          # linear interpolation between steps
        i = int(np.argmax(frac >= level))
        return t[i - 1] + (level - frac[i - 1]) / (frac[i] - frac[i - 1]) * DT
    return sag, cross(0.9) - cross(0.1), err_max, -m.actuator_biasprm[0, 2]


def exact_sag(kp):
    s = TAU_G / kp
    for _ in range(100):                       # kp * s = tau_g * cos(s)
        s = TAU_G * np.cos(s) / kp
    return s


def so101_table():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    h = m.opt.timestep
    print(f"\nSO-101 Menagerie model (robotstudio_so101/so101.xml) · timestep {h} s · "
          f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name}")
    print(" actuator       kp      kv     forcerange      ctrlrange(rad)     armature damping frictionloss"
          "  M_ii(kg·m²)  link part  h·w  h·w(no armature)")
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        dof = m.jnt_dofadr[j]
        kp, kv = m.actuator_gainprm[a, 0], -m.actuator_biasprm[a, 2]
        mii, arm = M[dof, dof], m.dof_armature[dof]
        print(f" {m.actuator(a).name:13s} {kp:7.2f} {kv:6.3f}  [{m.actuator_forcerange[a, 0]:5.2f} {m.actuator_forcerange[a, 1]:4.2f}]"
              f"  [{m.actuator_ctrlrange[a, 0]:6.3f} {m.actuator_ctrlrange[a, 1]:6.3f}]  {arm:7.3f}  {m.dof_damping[dof]:6.2f}"
              f"  {m.dof_frictionloss[dof]:8.3f}    {mii:9.5f}  {mii - arm:9.5f}  {h * np.sqrt(kp / mii):4.2f}"
              f"  {h * np.sqrt(kp / (mii - arm)):5.2f}")
    print(" (M_ii at the home pose qpos0, includes armature; h·w = timestep * sqrt(kp / M_ii))")


def main():
    print("mujoco", mujoco.__version__, "· numpy", np.__version__)
    print(f"link {L * 100:.0f} cm, {MASS} kg · gravity torque at horizontal {TAU_G:.4f} N·m · "
          f"link inertia {I_LINK:.4f} kg·m² · timestep {DT} s · implicitfast\n")

    print("sweep 1: kp (armature 0.028, dampratio 1, no joint damping/frictionloss) · step -0.5 rad -> 0")
    print("   kp     kv(from dampratio)  sag(deg)  tau_g/kp(deg)  exact(deg)  rise 10-90 %(ms)")
    for kp in [1, 3, 10, 30, 100, 300, 1000]:
        sag, rise, _, kv = step_response(xml(kp, dampratio=1.0, armature=0.028))
        print(f" {kp:5d}   {kv:7.3f}             {np.degrees(sag):7.3f}   {np.degrees(TAU_G / kp):7.3f}"
              f"        {np.degrees(exact_sag(kp)):7.3f}     {rise * 1000:7.1f}")

    print("\nbreak it A: SO-101 servo (kp 998.22, kv 2.731, damping 0.6, frictionloss 0.052), "
          "sweep armature")
    print(" armature  I_eff(kg·m²)  h·w   " + "  ".join(f"{s:>22s}" for s in ("Euler", "implicitfast", "discrete")))
    for arm in [0.028, 0.01, 0.005, 0.002, 0.001, 0.0]:
        cells = []
        for integ in ("Euler", "implicitfast", "discrete"):
            sag, _, err, _ = step_response(xml(SO101["kp"], kv=SO101["kv"], armature=arm, damping=0.6,
                                               frictionloss=0.052, integrator=integ))
            cells.append("stable" if sag is not None else f"DIVERGED (|e| {err:.0f} rad)")
        print(f" {arm:7.3f}   {I_LINK + arm:9.4f}   {DT * np.sqrt(SO101['kp'] / (I_LINK + arm)):4.2f}   "
              + "  ".join(f"{c:>22s}" for c in cells))

    print("\nbreak it B: same, with the Menagerie forcerange ±2.94 N·m (Euler) -> steady chatter")
    for arm in [0.005, 0.001, 0.0]:
        m = mujoco.MjModel.from_xml_string(xml(SO101["kp"], kv=SO101["kv"], armature=arm, damping=0.6,
                                               frictionloss=0.052, integrator="Euler", forcerange=2.94))
        d = mujoco.MjData(m)
        d.qpos[0] = -0.5
        q = []
        while d.time < 3.0 - 1e-9:
            mujoco.mj_step(m, d)
            q.append(d.qpos[0])
        tail = np.degrees(np.array(q[-int(0.5 / DT):]))
        print(f" armature {arm:5.3f} · last 0.5 s: mean {tail.mean():7.3f} deg · std {tail.std():6.3f} deg · "
              f"peak-to-peak {np.ptp(tail):6.3f} deg")

    print("\nbreak it C: implicitfast, armature 0.028, kv 2.731, sweep kp")
    for kp in [998.22, 3000, 10000, 30000]:
        row = []
        for integ in ("implicitfast", "discrete"):
            sag, _, err, _ = step_response(xml(kp, kv=SO101["kv"], armature=0.028, damping=0.6,
                                               frictionloss=0.052, integrator=integ))
            row.append(f"{integ} " + ("stable" if sag is not None else f"DIVERGED (|e| {err:.0f} rad)"))
        print(f" kp {kp:8.2f} · h·w {DT * np.sqrt(kp / (I_LINK + 0.028)):4.2f} · " + " · ".join(row))

    so101_table()


if __name__ == "__main__":
    main()
