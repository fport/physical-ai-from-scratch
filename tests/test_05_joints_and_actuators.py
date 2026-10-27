"""Lesson 05: physai.sim actuator laws and the servo step against MuJoCo, plus the stability facts the lesson uses."""
import mujoco
import numpy as np

from conftest import assert_build_output
from physai.sim import actuator_force, kv_from_dampratio, semi_implicit_euler, servo_sag, step_growth

RNG = np.random.default_rng(5)


def hinge(actuator, integrator="implicitfast", armature=0.02, damping=0.0, h=0.005):
    m = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{h}" integrator="{integrator}" gravity="0 0 0"/>
      <worldbody><body><joint name="j" type="hinge" axis="0 1 0" armature="{armature}" damping="{damping}"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".01" mass=".3"/></body></worldbody>
      <actuator>{actuator}</actuator></mujoco>""")
    return m, mujoco.MjData(m)


def test_actuator_force_matches_mujoco():
    for act in ('<motor joint="j" gear="2.5"/>', '<position joint="j" kp="30" kv="1.5" forcerange="-2 2"/>',
                '<velocity joint="j" kv="0.8" gear="1.7"/>', '<general joint="j" gainprm="4" biastype="affine" biasprm="0.3 -2 -0.1"/>'):
        m, d = hinge(act)
        fr = m.actuator_forcerange[0] if m.actuator_forcelimited[0] else None
        for _ in range(20):
            d.qpos[0], d.qvel[0], d.ctrl[0] = RNG.uniform(-2, 2, 3)
            mujoco.mj_forward(m, d)
            f = actuator_force(d.ctrl[0], d.actuator_length[0], d.actuator_velocity[0],
                               m.actuator_gainprm[0], m.actuator_biasprm[0], fr)
            assert np.isclose(f, d.actuator_force[0], atol=1e-12)
            assert np.isclose(m.actuator_gear[0, 0] * f, d.qfrc_actuator[0], atol=1e-12)


def test_kv_from_dampratio_matches_compiler():
    for kp, arm in ((5, 0.0), (80, 0.01), (900, 0.03)):
        m, d = hinge(f'<position joint="j" kp="{kp}" dampratio="0.7"/>', armature=arm)
        mujoco.mj_forward(m, d)
        M = np.zeros((1, 1))
        mujoco.mj_fullM(m, d, M)
        assert np.isclose(kv_from_dampratio(kp, 0.7, M[0, 0]), -m.actuator_biasprm[0, 2], rtol=1e-10)


def test_implicit_damping_step_matches_mujoco_integrators():
    kp, kv, damp = 400.0, 3.0, 0.5
    for integ in ("Euler", "implicitfast"):
        m, d = hinge(f'<position joint="j" kp="{kp}" kv="{kv}"/>', integrator=integ, damping=damp)
        d.qpos[0] = 0.4
        mujoco.mj_forward(m, d)
        M = np.zeros((1, 1))
        mujoco.mj_fullM(m, d, M)
        I = M[0, 0]
        c = damp / I if integ == "Euler" else (kv + damp) / I     # what each integrator treats implicitly
        q, v = 0.4, 0.0
        for _ in range(300):
            q, v = semi_implicit_euler(q, v, lambda x, u: (-kp * x - (kv + damp) * u) / I, 0.005, c)
            mujoco.mj_step(m, d)
            assert abs(q - d.qpos[0]) < 1e-12


def test_servo_sag_solves_static_balance_and_implicit_damping_helps():
    for kp in RNG.uniform(0.5, 1000, 10):
        e = servo_sag(kp, 0.2943)
        assert abs(kp * e - 0.2943 * np.cos(e)) < 1e-12
    h, w2, c = 0.005, 2.6**2 / 0.005**2, 400.0                # h*w = 2.6 > 2, but h c = 2: 2.6^2 < 4 + 2 h c
    assert step_growth(semi_implicit_euler, h, w2, c, c_implicit=c) < 1 < step_growth(semi_implicit_euler, h, w2, c)


def test_build_script_reproduces_its_output():
    assert_build_output("05-joints-and-actuators")
