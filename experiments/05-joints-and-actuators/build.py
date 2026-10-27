"""Lesson 05 · Build it: actuator force laws and a position servo, rebuilt by hand and checked against MuJoCo.

Step 1  The affine actuator (motor, position, velocity) vs actuator_force and qfrc_actuator.
Step 2  The Experiment's link: inertia from lesson 01, gravity torque vs qfrc_bias, kv from dampratio vs biasprm.
Step 3  Our servo loop (implicit damping, as implicitfast) vs mj_step; sag and rise time for the kp sweep.
Step 4  Growth per step for Euler and implicitfast vs what MuJoCo does in Break it A and B.

The functions live in physai/sim.py (the lesson shows them step by step).
Run:  uv run python experiments/05-joints-and-actuators/build.py      (~1 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.sim import (actuator_force, kv_from_dampratio, semi_implicit_euler, servo_sag, solid_inertia,  # noqa: E402
                        step_growth)

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
L, MASS, DT, G = 0.20, 0.30, 0.005, 9.81
TAU_G = MASS * G * L / 2


def link(kp, kv=None, dampratio=None, armature=0.028, damping=0.0, frictionloss=0.0, integrator="implicitfast"):
    """The Experiment's 20 cm, 0.3 kg capsule on a hinge, horizontal at q = 0, with a <position> servo."""
    gain = f'kp="{kp}" ' + (f'kv="{kv}"' if kv is not None else f'dampratio="{dampratio}"')
    return mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{DT}" integrator="{integrator}"/>
  <worldbody><body name="link">
    <joint name="j" type="hinge" axis="0 -1 0" armature="{armature}" damping="{damping}" frictionloss="{frictionloss}"/>
    <geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.01" mass="{MASS}"/></body></worldbody>
  <actuator><position joint="j" {gain}/></actuator></mujoco>""")


print("\n== Step 1. Actuator force laws vs MuJoCo (100 random states and controls) ==")
m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
  <body><joint name="a" type="hinge"/><geom type="capsule" fromto="0 0 0 .2 0 0" size=".01"/></body>
  <body pos="0 .3 0"><joint name="b" type="hinge"/><geom type="capsule" fromto="0 0 0 .2 0 0" size=".01"/></body>
  <body pos="0 .6 0"><joint name="c" type="hinge"/><geom type="capsule" fromto="0 0 0 .2 0 0" size=".01"/></body>
  </worldbody><actuator>
  <motor joint="a" gear="3"/>
  <position joint="b" kp="40" kv="2" forcerange="-5 5"/>
  <velocity joint="c" kv="0.7"/>
  </actuator></mujoco>""")
d = mujoco.MjData(m)
rng = np.random.default_rng(0)
err = np.zeros(3)
for _ in range(100):
    d.qpos[:], d.qvel[:], d.ctrl[:] = rng.uniform(-1, 1, 3), rng.uniform(-3, 3, 3), rng.uniform(-1, 1, 3)
    mujoco.mj_forward(m, d)
    for a in range(m.nu):
        fr = m.actuator_forcerange[a] if m.actuator_forcelimited[a] else None
        f = actuator_force(d.ctrl[a], d.actuator_length[a], d.actuator_velocity[a],
                           m.actuator_gainprm[a], m.actuator_biasprm[a], fr)
        err[a] = max(err[a], abs(f - d.actuator_force[a]), abs(m.actuator_gear[a, 0] * f - d.qfrc_actuator[a]))
for a, name in enumerate(("motor, gear 3", "position kp 40 kv 2, forcerange ±5", "velocity kv 0.7")):
    print(f"{name:36s} gainprm[0] {m.actuator_gainprm[a, 0]:5.1f} · biasprm[:3] {np.array2string(m.actuator_biasprm[a, :3], precision=1)}"
          f" · max error {err[a]:.1e} N m")

print("\n== Step 2. The Experiment's link: inertia, gravity torque, kv from dampratio ==")
rho = MASS / solid_inertia("capsule", [0.01, L / 2], 1.0)[0]
m_cap, i_cap = solid_inertia("capsule", [0.01, L / 2], rho)
I_LINK = i_cap[0] + m_cap * (L / 2) ** 2               # capsule's transverse inertia, moved to the hinge
m = link(1, dampratio=1.0)
d = mujoco.MjData(m)
d.qpos[0] = 0.3
mujoco.mj_forward(m, d)
M = np.zeros((1, 1))
mujoco.mj_fullM(m, d, M)
print(f"link inertia about the hinge {I_LINK:.8f} kg m^2 · M_00 - armature {M[0, 0] - 0.028:.8f}")
print(f"gravity torque at q = 0.3: -tau_g cos q = {-TAU_G * np.cos(0.3):.6f} N m · -qfrc_bias {-d.qfrc_bias[0]:.6f} N m")
print(f"{'kp':>5s} {'kv ours':>8s} {'kv MuJoCo':>10s}")
for kp in (1, 10, 100, 1000):
    print(f"{kp:5d} {kv_from_dampratio(kp, 1.0, I_LINK + 0.028):8.3f} {-link(kp, dampratio=1.0).actuator_biasprm[0, 2]:10.3f}")


def our_servo(kp, kv, I, h=DT, c_implicit=None, q0=-0.5, T=3.0):
    """Joint angle under the position servo and gravity, our stepper; implicitfast treats kv implicitly."""
    accel = lambda q, v: (actuator_force(0.0, q, v, [kp], [0, -kp, -kv]) - TAU_G * np.cos(q)) / I  # noqa: E731
    c = kv / I if c_implicit is None else c_implicit
    q, v, qs = q0, 0.0, [q0]
    for _ in range(int(round(T / h))):
        q, v = semi_implicit_euler(q, v, accel, h, c)
        qs.append(q)
    return np.array(qs)


def sag_and_rise(qs, q0=-0.5):
    sag = -qs[-int(0.5 / DT):].mean()
    frac = (qs - q0) / (-sag - q0)
    t = np.arange(len(qs)) * DT

    def cross(level):
        i = int(np.argmax(frac >= level))
        return t[i - 1] + (level - frac[i - 1]) / (frac[i] - frac[i - 1]) * DT
    return sag, cross(0.9) - cross(0.1)


print("\n== Step 3. Our servo loop vs mj_step (implicitfast), step -0.5 rad -> 0, 3 s ==")
print(f"{'kp':>5s} {'max |q_ours - qpos|':>20s} {'sag (deg)':>10s} {'servo_sag (deg)':>16s} {'rise 10-90 % (ms)':>18s}")
for kp in (1, 3, 10, 30, 100, 300, 1000):
    kv = kv_from_dampratio(kp, 1.0, I_LINK + 0.028)
    ours = our_servo(kp, kv, I_LINK + 0.028)
    m = link(kp, dampratio=1.0)
    d = mujoco.MjData(m)
    d.qpos[0] = -0.5
    mj = [-0.5]
    for _ in range(len(ours) - 1):
        mujoco.mj_step(m, d)
        mj.append(d.qpos[0])
    sag, rise = sag_and_rise(ours)
    print(f"{kp:5d} {np.abs(ours - np.array(mj)).max():20.1e} {np.degrees(sag):10.3f} "
          f"{np.degrees(servo_sag(kp, TAU_G)):16.3f} {rise * 1000:18.1f}")

print("\n== Step 4. Growth per step: kv explicit (Euler) or implicit (implicitfast) vs MuJoCo ==")
KP, KV, DAMP = 998.22, 2.731, 0.6                       # the SO-101 servo


def mujoco_outcome(kp, armature, integrator):
    m = link(kp, kv=KV, armature=armature, damping=DAMP, frictionloss=0.052, integrator=integrator)
    d = mujoco.MjData(m)
    d.qpos[0] = -0.5
    while d.time < 3.0 - 1e-9:
        mujoco.mj_step(m, d)
        if not abs(d.qpos[0]) < np.pi:
            return "diverged"
    return "stable"


print(f"{'kp':>8s} {'armature':>8s} {'h*w':>5s} {'Euler growth':>13s} {'MuJoCo Euler':>13s} "
      f"{'implicitfast growth':>20s} {'MuJoCo implicitfast':>20s}")
for kp, arm in ((KP, 0.028), (KP, 0.010), (KP, 0.005), (KP, 0.002), (KP, 0.0), (3000, 0.028), (10000, 0.028),
                (30000, 0.028)):
    I = I_LINK + arm
    g_euler = step_growth(semi_implicit_euler, DT, kp / I, (KV + DAMP) / I, c_implicit=DAMP / I)
    g_fast = step_growth(semi_implicit_euler, DT, kp / I, (KV + DAMP) / I, c_implicit=(KV + DAMP) / I)
    print(f"{kp:8.0f} {arm:8.3f} {DT * np.sqrt(kp / I):5.2f} {g_euler:13.3f} {mujoco_outcome(kp, arm, 'Euler'):>13s} "
          f"{g_fast:20.3f} {mujoco_outcome(kp, arm, 'implicitfast'):>20s}")
