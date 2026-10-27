"""Lesson 12 · Joint control: PD, PID, gravity feedforward and position servos.

Four parts, each a sweep:
  A. PD on one joint without gravity: sweep the damping ratio zeta -> overshoot, settling time
     (compared with the second-order formula).
  B. The same joint under gravity: sweep Kp -> steady-state sag (compared with m g l cos(q) = Kp e);
     then PD vs PID (sweep Ki) vs PD + gravity feedforward from qfrc_bias.
  C. SO-101 (Menagerie): the STS3215 is modelled as a position actuator (a PD loop with a torque cap).
     Sweep a payload in the gripper -> shoulder sag and saturation.
  D. Unitree Go2 (Menagerie): stand up from lying with one PD rule on 12 joints; sweep Kp / Kd.

Run:  python experiment.py      (~30 s on Apple M5)
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

# ---------------------------------------------------------------- A/B: one joint
M_LINK, L_LINK = 1.0, 0.3          # 1 kg point-like mass 30 cm from the axis
I_LINK = M_LINK * L_LINK ** 2 + 1e-4   # inertia about the axis [kg m^2]
DT = 0.001


def joint_xml(gravity):
    g = "0 0 -9.81" if gravity else "0 0 0"
    return f"""
<mujoco>
  <option timestep="{DT}" gravity="{g}"/>
  <worldbody>
    <body name="link">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <inertial pos="{L_LINK} 0 0" mass="{M_LINK}" diaginertia="1e-4 1e-4 1e-4"/>
    </body>
  </worldbody>
  <actuator><motor joint="j"/></actuator>
</mujoco>"""


def simulate_joint(kp, kd, q0, q_target, gravity, ki=0.0, feedforward=False, T=4.0):
    """Torque control of the single joint. Returns time and angle arrays."""
    m = mujoco.MjModel.from_xml_string(joint_xml(gravity))
    d = mujoco.MjData(m)
    d.qpos[0] = q0
    mujoco.mj_forward(m, d)
    integ, ts, qs = 0.0, [], []
    while d.time < T - 1e-9:
        e = q_target - d.qpos[0]
        integ += e * DT
        tau = kp * e - kd * d.qvel[0] + ki * integ
        if feedforward:
            tau += d.qfrc_bias[0]          # gravity (+ Coriolis, zero for one joint) at the current q
        d.ctrl[0] = tau
        mujoco.mj_step(m, d)
        ts.append(d.time)
        qs.append(d.qpos[0])
    return np.array(ts), np.array(qs)


def step_metrics(t, q, q0, qt, band):
    """Overshoot (% of the step), settling time into +-band, final error."""
    sgn = np.sign(qt - q0)
    over = max(0.0, np.max(sgn * (q - qt))) / abs(qt - q0) * 100
    outside = np.where(np.abs(q - qt) > band)[0]
    ts = t[outside[-1]] if len(outside) else 0.0
    return over, ts, qt - q[-1]


def part_a():
    print("\n[A] PD on one joint, no gravity. I = %.4f kg m^2, Kp = 20 N m/rad, step 0 -> 0.5 rad, dt 1 ms" % I_LINK)
    print(f"{'zeta':>6}{'Kd':>8}{'overshoot %':>13}{'theory %':>10}{'t_settle 2% s':>15}")
    kp = 20.0
    for zeta in (0.1, 0.3, 0.5, 0.7, 1.0, 1.5):
        kd = 2 * zeta * np.sqrt(kp * I_LINK)
        t, q = simulate_joint(kp, kd, 0.0, 0.5, gravity=False)
        over, ts, _ = step_metrics(t, q, 0.0, 0.5, band=0.02 * 0.5)
        theory = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)) if zeta < 1 else 0.0
        print(f"{zeta:>6.1f}{kd:>8.3f}{over:>13.2f}{theory:>10.2f}{ts:>15.3f}")
    print("  Kp sweep at zeta = 0.7 (settling time should scale as 1/omega_n = sqrt(I/Kp)):")
    print(f"{'Kp':>6}{'omega_n rad/s':>15}{'t_settle 2% s':>15}")
    for kp in (5.0, 20.0, 80.0):
        kd = 2 * 0.7 * np.sqrt(kp * I_LINK)
        t, q = simulate_joint(kp, kd, 0.0, 0.5, gravity=False)
        _, ts, _ = step_metrics(t, q, 0.0, 0.5, band=0.01)
        print(f"{kp:>6.0f}{np.sqrt(kp / I_LINK):>15.2f}{ts:>15.3f}")


def part_b():
    mgl = M_LINK * 9.81 * L_LINK
    q0, qt = np.pi / 2, 0.0           # hanging straight down -> horizontal
    print(f"\n[B] Same joint under gravity (m g l = {mgl:.3f} N m). Step from hanging (90 deg) to horizontal (0 deg), zeta 0.7")
    print(f"{'Kp':>6}{'sag deg (sim)':>15}{'sag deg (Kp e = mgl cos e)':>28}")
    for kp in (5.0, 10.0, 20.0, 40.0, 80.0):
        kd = 2 * 0.7 * np.sqrt(kp * I_LINK)
        t, q = simulate_joint(kp, kd, q0, qt, gravity=True)
        e = 0.0
        for _ in range(200):               # fixed point of Kp e = mgl cos(e)
            e = mgl * np.cos(e) / kp
        print(f"{kp:>6.0f}{np.degrees(q[-1] - qt):>15.3f}{np.degrees(e):>28.3f}")
    kp = 20.0
    kd = 2 * 0.7 * np.sqrt(kp * I_LINK)
    print(f"  Kp = 20, Kd = {kd:.3f}: three ways to remove the sag (4 s simulated)")
    print(f"{'controller':<22}{'final err deg':>14}{'overshoot deg':>15}{'t into 0.5 deg s':>18}")
    cases = [("PD", 0.0, False), ("PID Ki=10", 10.0, False), ("PID Ki=40", 40.0, False),
             ("PID Ki=160", 160.0, False), ("PD + qfrc_bias", 0.0, True)]
    for name, ki, ff in cases:
        t, q = simulate_joint(kp, kd, q0, qt, gravity=True, ki=ki, feedforward=ff)
        over = max(0.0, -np.min(q - qt))   # going past horizontal means q < 0
        outside = np.where(np.abs(q - qt) > np.radians(0.5))[0]
        ts = t[outside[-1]] if len(outside) and outside[-1] < len(t) - 1 else float("nan")
        ts_txt = f"{ts:.3f}" if np.isfinite(ts) else "never"
        print(f"{name:<22}{np.degrees(q[-1] - qt):>14.3f}{np.degrees(over):>15.3f}{ts_txt:>18}")


# ---------------------------------------------------------------- C: SO-101 servo
def part_c():
    m0 = mujoco.MjModel.from_xml_path(path("robotstudio_so101"))
    a = m0.actuator("shoulder_lift").id
    kp, kv = m0.actuator_gainprm[a, 0], -m0.actuator_biasprm[a, 2]
    fr = m0.actuator_forcerange[a]
    print(f"\n[C] SO-101 (Menagerie) position actuator: force = kp (ctrl - q) - kv qdot, kp = {kp:.2f}, kv = {kv:.3f}, "
          f"forcerange = [{fr[0]:.2f}, {fr[1]:.2f}] N m")
    print("  Arm stretched forward (all joints 0, gripper ~39 cm from the base), payload at the gripper frame, 3 s hold")
    print(f"{'payload kg':>11}{'gravity torque N m':>20}{'shoulder sag deg':>18}{'actuator force N m':>20}{'gripper drop cm':>17}")
    for load in (0.0, 0.25, 0.5, 0.7, 0.8, 1.0):
        spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
        if load:
            b = spec.body("gripper")
            site = [s for s in spec.sites if s.name == "gripperframe"][0]
            b.add_body(name="payload", pos=site.pos).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01],
                                                              mass=load, contype=0, conaffinity=0)
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        tau_g = d.qfrc_bias[m.joint("shoulder_lift").dofadr[0]]
        z0 = d.site("gripperframe").xpos[2]
        d.ctrl[:] = 0.0
        while d.time < 3.0 - 1e-9:
            mujoco.mj_step(m, d)
        j = m.joint("shoulder_lift").qposadr[0]
        print(f"{load:>11.2f}{tau_g:>20.3f}{np.degrees(d.qpos[j]):>18.3f}"
              f"{d.actuator_force[m.actuator('shoulder_lift').id]:>20.3f}{(z0 - d.site('gripperframe').xpos[2]) * 100:>17.2f}")


# ---------------------------------------------------------------- D: Go2 case study
# unitree_rl_gym go2_config.py default_joint_angles, order FL FR RL RR x (hip, thigh, calf)
STAND = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
LIE = np.array([0.0, 1.25, -2.7] * 4)   # folded legs, belly near the ground
SIM_SEC = 3.0


def go2_run(kp, kd, load=0.0):
    m = mujoco.MjModel.from_xml_path(path("unitree_go2"))
    if load:
        m.body_mass[m.body("base").id] += load
    d = mujoco.MjData(m)
    d.qpos[7:] = LIE
    d.qpos[2] = 0.12
    mujoco.mj_forward(m, d)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    for _ in range(int(0.5 / m.opt.timestep)):          # settle the folded pose on the ground
        d.ctrl[:] = np.clip(40 * (LIE - d.qpos[7:]) - 1.0 * d.qvel[6:], lo, hi)
        mujoco.mj_step(m, d)
    feet = [m.geom(f).id for f in ("FL", "FR", "RL", "RR")]
    floor = m.geom("floor").id
    t0, log, i, n_sat, n_tot = d.time, [], 0, 0, 0
    every = max(1, int(round(1 / 120 / m.opt.timestep)))
    while d.time - t0 < SIM_SEC:
        tau = kp * (STAND - d.qpos[7:]) - kd * d.qvel[6:]   # the whole controller
        d.ctrl[:] = np.clip(tau, lo, hi)                    # clipped at the motor limit
        n_sat += int(np.any(np.abs(tau) >= hi))
        n_tot += 1
        mujoco.mj_step(m, d)
        if i % every == 0:
            touching = {c.geom1 if c.geom2 == floor else c.geom2 for c in d.contact[:d.ncon]
                        if floor in (c.geom1, c.geom2)}
            log.append([d.time - t0, float(d.qpos[2]), sum(f in touching for f in feet)])
        i += 1
    a = np.round(np.array(log), 4)
    t, h = a[:, 0], a[:, 1]
    start, final, peak = float(h[0]), float(h[t > SIM_SEC - 0.5].mean()), float(h.max())
    over = max(0.0, (peak - final) / (final - start) * 100) if final > start + 0.01 else 0.0
    k = int(np.argmax(h))
    return start, final, peak, over, float(t[k]), int(a[k, 2]), 100 * n_sat / n_tot, m.opt.timestep


def part_d():
    print("\n[D] Unitree Go2 (Menagerie): lie -> stand with tau = Kp (q* - q) - Kd qdot on all 12 joints, 3 s")
    print(f"{'case':<26}{'start cm':>9}{'peak cm':>9}{'t_peak s':>9}{'feet@peak':>10}{'final cm':>9}"
          f"{'overshoot %':>12}{'saturated %':>12}")
    cases = [("20 / 0.5 (unitree_rl_gym)", 20, 0.5, 0), ("5 / 0.5", 5, 0.5, 0), ("10 / 0.5", 10, 0.5, 0),
             ("40 / 0.5", 40, 0.5, 0), ("80 / 0.5", 80, 0.5, 0), ("20 / 0", 20, 0.0, 0), ("20 / 2", 20, 2.0, 0),
             ("60 / 5 (sdk2 stand ex.)", 60, 5.0, 0), ("20 / 0.5 + 5 kg on base", 20, 0.5, 5.0)]
    for name, kp, kd, load in cases:
        s, f, p, o, tp, nf, sat, dt = go2_run(kp, kd, load)
        print(f"{name:<26}{s * 100:>9.1f}{p * 100:>9.1f}{tp:>9.2f}{nf:>10d}{f * 100:>9.1f}{o:>12.1f}{sat:>12.1f}")
    print(f"  (Go2 timestep {dt * 1000:.0f} ms; final = mean height over 2.5-3 s; feet@peak = feet touching the floor;\n"
          "   saturated % = share of control steps where some joint hit its torque limit; overshoot = (peak-final)/(final-start))")
    # geometric height of the target stance: body level, joints exactly at STAND, foot sphere bottom -> base
    m = mujoco.MjModel.from_xml_path(path("unitree_go2"))
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = [0, 0, 1.0], [1, 0, 0, 0], STAND
    mujoco.mj_kinematics(m, d)
    hs = {}
    for foot in ("FL", "FR", "RL", "RR"):
        g = m.geom(foot).id
        hs[foot] = d.qpos[2] - (d.geom_xpos[g][2] - m.geom_size[g][0])
    print("  target stance, geometry only (cm): " + "  ".join(f"{k} {v * 100:.2f}" for k, v in hs.items()))


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__}")
    part_a()
    part_b()
    part_c()
    part_d()


if __name__ == "__main__":
    main()
