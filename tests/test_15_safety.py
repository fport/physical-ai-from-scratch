"""Lesson 15: physai.safety against MuJoCo's position actuator, plus properties the lesson relies on."""
import mujoco
import numpy as np

from conftest import assert_build_output
from physai.safety import EStop, Watchdog, clamp_goal, servo_torque

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body>
      <joint name="j" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j" kp="998.22" kv="2.731" ctrlrange="-1 1" forcerange="-{cap} {cap}"/>
  </actuator>
</mujoco>"""


def test_servo_torque_matches_mujoco_position_actuator():
    rng = np.random.default_rng(0)
    for cap in (2.94, 0.5, 0.05):
        m = mujoco.MjModel.from_xml_string(XML.format(cap=cap))
        d = mujoco.MjData(m)
        for _ in range(300):
            d.qpos[0], d.qvel[0], d.ctrl[0] = rng.uniform(-1, 1), rng.uniform(-5, 5), rng.uniform(-2, 2)
            mujoco.mj_forward(m, d)
            goal = clamp_goal(d.ctrl[0], d.qpos[0], -1.0, 1.0)          # MuJoCo clamps ctrl to ctrlrange
            assert abs(servo_torque(goal, d.qpos[0], d.qvel[0], 998.22, 2.731, cap) - d.actuator_force[0]) < 1e-10


def test_clamp_goal_guarantees():
    rng = np.random.default_rng(1)
    target, q_now = rng.uniform(-5, 5, 10_000), rng.uniform(-1, 1, 10_000)
    g = clamp_goal(target, q_now, -1.0, 1.0, 0.05)
    assert np.all(np.abs(g - q_now) <= 0.05 + 1e-15)
    assert np.all(np.abs(g) <= 1.0)
    assert np.allclose(clamp_goal(g, q_now, -1.0, 1.0, 0.05), g)                 # idempotent
    inside = (np.abs(target - q_now) <= 0.05) & (np.abs(target) <= 1.0)
    assert np.array_equal(g[inside], target[inside])                              # a safe target passes unchanged
    assert clamp_goal(0.7, 0.0) == 0.7                                            # no limits: no change


def test_watchdog_and_estop_latch():
    wd = Watchdog(0.2)
    assert not wd.expired(10.0)                      # nothing sent yet: nothing to time out
    wd.feed(1.0)
    assert not wd.expired(1.2) and wd.expired(1.2001)
    es = EStop()
    assert [es.update(p) for p in (False, True, False)] == [False, True, True]
    assert es.reset(pressed=True) and not es.reset(pressed=False)


def test_build_script_reproduces_its_output():
    assert_build_output("15-safety")
