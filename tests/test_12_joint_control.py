"""Lesson 12: physai.control against MuJoCo, plus properties the lesson relies on."""
import mujoco
import numpy as np
import pytest

from conftest import assert_build_output
from physai.control import PID, gravity_torque, pd_sag, simulate_joint, step_joint

DT = 0.001


def link_model(m_link, l_link, gravity=True):
    return mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <option timestep="{DT}" gravity="0 0 {-9.81 if gravity else 0}"/>
  <worldbody>
    <body>
      <joint type="hinge" axis="0 1 0"/>
      <inertial pos="{l_link} 0 0" mass="{m_link}" diaginertia="1e-4 1e-4 1e-4"/>
    </body>
  </worldbody>
</mujoco>""")


@pytest.mark.parametrize("seed", range(3))
def test_gravity_torque_matches_qfrc_bias(seed):
    rng = np.random.default_rng(seed)
    m_link, l_link = rng.uniform(0.2, 3.0), rng.uniform(0.05, 0.5)
    m = link_model(m_link, l_link)
    d = mujoco.MjData(m)
    for q in rng.uniform(-np.pi, np.pi, 10):
        d.qpos[0] = q
        mujoco.mj_forward(m, d)
        assert abs(gravity_torque(q, m_link, l_link) + d.qfrc_bias[0]) < 1e-12


@pytest.mark.parametrize("seed", range(3))
def test_pd_loop_matches_mj_step(seed):
    rng = np.random.default_rng(10 + seed)
    m_link, l_link = rng.uniform(0.2, 3.0), rng.uniform(0.05, 0.5)
    kp, kd, target = rng.uniform(5, 100), rng.uniform(0.1, 5), rng.uniform(-1, 1)
    m = link_model(m_link, l_link)
    d = mujoco.MjData(m)
    inertia = m_link * l_link ** 2 + 1e-4
    theirs = []
    for _ in range(1500):
        d.qfrc_applied[0] = kp * (target - d.qpos[0]) - kd * d.qvel[0]
        mujoco.mj_step(m, d)
        theirs.append(d.qpos[0])
    c = PID(kp, kd)
    ours = simulate_joint(lambda q, qd: c(q, qd, target, DT), 0.0, inertia, DT, 1500,
                          lambda q: gravity_torque(q, m_link, l_link))
    assert np.abs(ours - np.array(theirs)).max() < 1e-10


@pytest.mark.parametrize("zeta", [0.2, 0.5, 0.8])
def test_overshoot_follows_damping_ratio_formula(zeta):
    inertia, kp = 0.0901, 20.0
    kd = 2 * zeta * np.sqrt(kp * inertia)
    c = PID(kp, kd)
    q = simulate_joint(lambda q, qd: c(q, qd, 0.5, DT), 0.0, inertia, DT, 4000)
    formula = np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2))
    assert abs((q.max() - 0.5) / 0.5 - formula) < 0.003      # within 0.3 percentage points at dt = 1 ms


@pytest.mark.parametrize("kp", [5.0, 20.0, 80.0])
def test_sag_satisfies_kp_e_equals_mgl_cos_e(kp):
    inertia, mgl = 0.0901, 1.0 * 9.81 * 0.3
    c = PID(kp, 2 * 0.7 * np.sqrt(kp * inertia))
    q = simulate_joint(lambda q, qd: c(q, qd, 0.0, DT), np.pi / 2, inertia, DT, 6000,
                       lambda q: gravity_torque(q, 1.0, 0.3))
    e = pd_sag(kp, mgl)
    assert abs(q[-1] - e) < 1e-6 and abs(kp * e - mgl * np.cos(e)) < 1e-12


def test_integrator_clamp_and_torque_cap():
    c = PID(20.0, 1.0, ki=100.0, i_max=2.0, tau_max=3.0)
    for _ in range(1000):                         # a large error that never goes away
        tau = c(0.0, 0.0, 1.0, DT)
    assert abs(c.ki * c.integ) <= 2.0 + 1e-12 and tau == 3.0


def test_step_joint_is_semi_implicit():
    q, qd = step_joint(0.0, 0.0, 1.0, 2.0, 0.1)   # qdd = 0.5: speed first, then angle from the new speed
    assert qd == pytest.approx(0.05) and q == pytest.approx(0.005)


def test_build_script_reproduces_its_output():
    assert_build_output("12-joint-control")
