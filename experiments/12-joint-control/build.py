"""Lesson 12 · Build it: a joint controller from scratch, checked against MuJoCo at every step.

The joint is the Experiment's: a 1 kg point mass 30 cm from a hinge (I = 0.0901 kg m^2), dt = 1 ms.

Step 1  Gravity torque m g l cos(q) by hand vs qfrc_bias.
Step 2  A semi-implicit Euler step by hand vs mj_step (free swing).
Step 3  The PD law: overshoot vs MuJoCo and vs the zeta formula.
Step 4  Gravity sag vs MuJoCo and vs Kp e = m g l cos(e).
Step 5  Gravity feedforward from our own g(q), and why reading qfrc_bias after mj_step lags one step.
Step 6  Limits: integrator clamp (anti-windup) and the motor's torque cap.
Use it  MuJoCo's position actuator (kp, kv, dampratio) and gravcomp do the same inside mj_step.

The functions live in physai/control.py (the lesson shows them step by step).
Run:  uv run python experiments/12-joint-control/build.py      (~2 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.control import PID, gravity_torque, pd_sag, pd_torque, simulate_joint, step_joint  # noqa: E402

M_LINK, L_LINK, DT = 1.0, 0.3, 0.001
MGL = M_LINK * 9.81 * L_LINK


def joint_model(gravity=True, ctrlrange=None):
    """The Experiment's one-joint model (motor actuator, Euler integrator)."""
    limit = f'ctrlrange="{-ctrlrange} {ctrlrange}"' if ctrlrange else ""
    return mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <option timestep="{DT}" gravity="0 0 {-9.81 if gravity else 0}"/>
  <worldbody>
    <body name="link">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <inertial pos="{L_LINK} 0 0" mass="{M_LINK}" diaginertia="1e-4 1e-4 1e-4"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" {limit}/></actuator>
</mujoco>""")


def mujoco_run(torque_fn, q0, steps, gravity=True, ctrlrange=None, applied=False):
    """Same loop in MuJoCo: torque_fn(q, qd, d) -> torque, sent as ctrl (or as qfrc_applied)."""
    m = joint_model(gravity, ctrlrange)
    d = mujoco.MjData(m)
    d.qpos[0] = q0
    mujoco.mj_forward(m, d)
    qs = []
    for _ in range(steps):
        tau = torque_fn(d.qpos[0], d.qvel[0], d)
        if applied:
            d.qfrc_applied[0] = tau
        else:
            d.ctrl[0] = tau
        mujoco.mj_step(m, d)
        qs.append(d.qpos[0])
    return np.array(qs)


def metrics(q, q_target, band):
    """Overshoot past the target (moving down to it), final error, time of the last exit from +-band."""
    over = max(0.0, -np.min(q - q_target))
    outside = np.where(np.abs(q - q_target) > band)[0]
    t_in = (outside[-1] + 1) * DT if len(outside) and outside[-1] < len(q) - 1 else float("nan")
    return over, round(q[-1] - q_target, 9) + 0.0, t_in   # + 0.0 turns -0.0 into 0.0


print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
m = joint_model()
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
M = np.zeros((1, 1))
mujoco.mj_fullM(m, d, M)
I = M_LINK * L_LINK ** 2 + 1e-4
load = lambda q: gravity_torque(q, M_LINK, L_LINK)  # noqa: E731

print("\n== Step 1. Gravity torque m g l cos(q) by hand vs MuJoCo ==")
print(f"inertia: m l^2 + 1e-4 = {I:.4f} kg m^2, MuJoCo mass matrix = {M[0, 0]:.4f} kg m^2")
print(f"{'q (deg)':>8s} {'ours (N m)':>11s} {'-qfrc_bias (N m)':>17s}")
worst = 0.0
for deg in (0, 30, 60, 90, -45):
    d.qpos[0], d.qvel[0] = np.radians(deg), 0.0
    mujoco.mj_forward(m, d)
    ours = load(np.radians(deg))
    worst = max(worst, abs(ours + d.qfrc_bias[0]))
    print(f"{deg:8d} {ours:11.4f} {-d.qfrc_bias[0]:17.4f}")
print(f"max difference: {worst:.1e} N m")

print("\n== Step 2. Semi-implicit Euler by hand vs mj_step (free swing from horizontal, 1 s) ==")
q, qd, ours = 0.0, 0.0, []
for _ in range(1000):
    q, qd = step_joint(q, qd, 0.0, I, DT, load(q))
    ours.append(q)
theirs = mujoco_run(lambda q, qd, d: 0.0, 0.0, 1000)
print(f"angle after 1 s: ours {np.degrees(ours[-1]):.3f} deg, MuJoCo {np.degrees(theirs[-1]):.3f} deg")
print(f"max |q_ours - q_mujoco| over 1000 steps: {np.abs(np.array(ours) - theirs).max():.1e} rad")

print("\n== Step 3. PD step 0 -> 0.5 rad, Kp = 20, no gravity: overshoot ==")
print(f"{'zeta':>5s} {'Kd':>6s} {'ours %':>7s} {'MuJoCo %':>9s} {'formula %':>10s}")
worst = 0.0
for zeta in (0.1, 0.3, 0.5, 0.7, 1.0):
    kd = 2 * zeta * np.sqrt(20 * I)
    ours = simulate_joint(lambda q, qd: pd_torque(q, qd, 0.5, 20, kd), 0.0, I, DT, 4000)
    theirs = mujoco_run(lambda q, qd, d: 20 * (0.5 - q) - kd * qd, 0.0, 4000, gravity=False)
    worst = max(worst, np.abs(ours - theirs).max())
    formula = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)) if zeta < 1 else 0.0
    over_ours, over_mj = (max(0.0, (x.max() - 0.5) / 0.5 * 100) for x in (ours, theirs))
    print(f"{zeta:5.1f} {kd:6.3f} {over_ours:7.2f} {over_mj:9.2f} {formula:10.2f}")
print(f"max |q_ours - q_mujoco| over all runs: {worst:.1e} rad")
kd = 2 * 0.7 * np.sqrt(20 * I)
via_ctrl = mujoco_run(lambda q, qd, d: 20 * (0.5 - q) - kd * qd, 0.0, 4000, gravity=False)
via_applied = mujoco_run(lambda q, qd, d: 20 * (0.5 - q) - kd * qd, 0.0, 4000, gravity=False, applied=True)
print(f"same loop through qfrc_applied instead of the motor: max diff {np.abs(via_ctrl - via_applied).max():.1e} rad")

print("\n== Step 4. PD under gravity, hanging (90 deg) -> horizontal (0 deg), zeta 0.7: sag ==")
print(f"{'Kp':>4s} {'ours (deg)':>11s} {'MuJoCo (deg)':>13s} {'Kp e = mgl cos e (deg)':>23s}")
for kp in (5, 20, 80):
    kd = 2 * 0.7 * np.sqrt(kp * I)
    ours = simulate_joint(lambda q, qd: pd_torque(q, qd, 0.0, kp, kd), np.pi / 2, I, DT, 4000, load)
    theirs = mujoco_run(lambda q, qd, d: kp * (0.0 - q) - kd * qd, np.pi / 2, 4000)
    print(f"{kp:4d} {np.degrees(ours[-1]):11.3f} {np.degrees(theirs[-1]):13.3f} {np.degrees(pd_sag(kp, MGL)):23.3f}")

print("\n== Step 5. PD + gravity feedforward, Kp = 20, Kd = 1.879 ==")
KD = 2 * 0.7 * np.sqrt(20 * I)
ours = simulate_joint(lambda q, qd: pd_torque(q, qd, 0.0, 20, KD) - load(q), np.pi / 2, I, DT, 4000, load)


def with_fresh_bias(q, qd, d):
    mujoco.mj_forward(m, d)                        # recompute qfrc_bias at the current q
    return 20 * (0.0 - q) - KD * qd + d.qfrc_bias[0]


fresh = mujoco_run(with_fresh_bias, np.pi / 2, 4000)
stale = mujoco_run(lambda q, qd, d: 20 * (0.0 - q) - KD * qd + d.qfrc_bias[0], np.pi / 2, 4000)
print(f"{'feedforward':<38s} {'overshoot (deg)':>15s} {'final err (deg)':>16s} {'t into 0.5 deg (s)':>19s}")
for name, q in (("ours: -g(q) at the current q", ours), ("MuJoCo: qfrc_bias after mj_forward", fresh),
                ("MuJoCo: qfrc_bias left by mj_step", stale)):
    over, err, t_in = metrics(q, 0.0, np.radians(0.5))
    print(f"{name:<38s} {np.degrees(over):15.3f} {np.degrees(err):16.3f} {t_in:19.3f}")
print(f"max |q_ours - q_fresh|: {np.abs(ours - fresh).max():.1e} rad")

print("\n== Step 6. Limits: integrator clamp (anti-windup) and torque cap ==")
print(f"{'controller':<38s} {'overshoot (deg)':>15s} {'final err (deg)':>16s} {'t into 0.5 deg (s)':>19s}")
rows = [("PID Ki = 160", PID(20, KD, 160.0), False),
        ("PID Ki = 160, |Ki integ| <= 3 N m", PID(20, KD, 160.0, i_max=3.0), False),
        ("PD + feedforward", PID(20, KD), True),
        ("PD + feedforward, |tau| <= 3.5 N m", PID(20, KD, tau_max=3.5), True)]
for name, c, ff in rows:
    peak = [0.0]

    def ctrl(q, qd, c=c, ff=ff, peak=peak):
        tau = c(q, qd, 0.0, DT, tau_ff=-load(q) if ff else 0.0)
        peak[0] = max(peak[0], abs(tau))
        return tau
    ours = simulate_joint(ctrl, np.pi / 2, I, DT, 4000, load)
    over, err, t_in = metrics(ours, 0.0, np.radians(0.5))
    print(f"{name:<38s} {np.degrees(over):15.3f} {np.degrees(err):16.3f} {t_in:19.3f}   peak |tau| {peak[0]:.2f} N m")
c_ref = PID(20, KD, 160.0)
pid_mj = mujoco_run(lambda q, qd, d: c_ref(q, qd, 0.0, DT), np.pi / 2, 4000)
c_ours = PID(20, KD, 160.0)
pid_ours = simulate_joint(lambda q, qd: c_ours(q, qd, 0.0, DT), np.pi / 2, I, DT, 4000, load)
print(f"PID Ki = 160, ours vs MuJoCo: max diff {np.abs(pid_ours - pid_mj).max():.1e} rad")
c_cap = PID(20, KD, tau_max=3.5)
cap_ours = simulate_joint(lambda q, qd: c_cap(q, qd, 0.0, DT, tau_ff=-load(q)), np.pi / 2, I, DT, 4000, load)
cap_mj = mujoco_run(lambda q, qd, d: 20 * (0.0 - q) - KD * qd - gravity_torque(q, M_LINK, L_LINK), np.pi / 2, 4000,
                    ctrlrange=3.5)
print(f"torque cap, ours (clip in PID) vs MuJoCo (unclipped torque, ctrlrange = +-3.5): "
      f"max diff {np.abs(cap_ours - cap_mj).max():.1e} rad")

print("\n== Use it. MuJoCo's own position actuator and gravity compensation ==")


def position_run(q0, target, gravity, gains, body="", joint=""):
    """A MuJoCo position actuator holding `target`: the PD loop runs inside mj_step."""
    m = mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <option timestep="{DT}" gravity="0 0 {-9.81 if gravity else 0}"/>
  <worldbody>
    <body name="link" {body}>
      <joint name="j" type="hinge" axis="0 1 0" {joint}/>
      <inertial pos="{L_LINK} 0 0" mass="{M_LINK}" diaginertia="1e-4 1e-4 1e-4"/>
    </body>
  </worldbody>
  <actuator><position joint="j" {gains}/></actuator>
</mujoco>""")
    d = mujoco.MjData(m)
    d.qpos[0], d.ctrl[0] = q0, target
    qs = []
    for _ in range(4000):
        mujoco.mj_step(m, d)
        qs.append(d.qpos[0])
    return m, d, np.array(qs)


m_dr, _, _ = position_run(0.0, 0.5, False, 'kp="20" dampratio="0.7"')
print(f'dampratio="0.7" -> kv = {-m_dr.actuator_biasprm[0, 2]:.3f} (2 zeta sqrt(Kp I) = {KD:.3f})')
_, _, q_pos = position_run(0.0, 0.5, False, f'kp="20" kv="{KD}"')
c = PID(20, KD)
q_pd = simulate_joint(lambda q, qd: c(q, qd, 0.5, DT), 0.0, I, DT, 4000)
print(f"step 0 -> 0.5 rad: overshoot {(q_pos.max() - 0.5) / 0.5 * 100:.2f} %, "
      f"max diff to our PD loop {np.abs(q_pos - q_pd).max():.1e} rad")
print(f"{'under gravity, hanging -> horizontal':<40s} {'overshoot (deg)':>15s} {'final err (deg)':>16s} "
      f"{'qfrc_passive':>13s} {'qfrc_actuator':>14s}")
for name, body, joint in (("position actuator", "", ""), ('+ body gravcomp="1"', 'gravcomp="1"', ""),
                          ('+ joint actuatorgravcomp="true"', 'gravcomp="1"', 'actuatorgravcomp="true"')):
    _, d_g, q_g = position_run(np.pi / 2, 0.0, True, f'kp="20" kv="{KD}"', body, joint)
    over, err, _ = metrics(q_g, 0.0, np.radians(0.5))
    print(f"{name:<40s} {np.degrees(over):15.3f} {np.degrees(err):16.3f} {round(d_g.qfrc_passive[0], 9) + 0.0:13.3f} "
          f"{round(d_g.qfrc_actuator[0], 9) + 0.0:14.3f}")
