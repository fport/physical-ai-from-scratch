"""Lesson 34: action modes, the servo PD law against MuJoCo, discount per rate, smoothness and power."""
import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import assert_build_output
from physai.rl import action_to_ctrl, discount_for_rate, pd_torque, torque_rate_and_power


def test_action_modes():
    q0, lo, hi, prev = np.zeros(2), -np.ones(2), np.ones(2), np.array([0.5, 0.95])
    a = np.array([0.5, 2.0])
    assert np.allclose(action_to_ctrl(a, "position", q0, 1.0, prev, lo, hi), [0.5, 1.0])
    assert np.allclose(action_to_ctrl(a, "delta", q0, 0.1, prev, lo, hi), [0.55, 1.0])
    assert np.allclose(action_to_ctrl(a, "torque", q0, 1.0, prev, lo, hi), [1.47, 2.94])


def test_pd_matches_mujoco_position_servo():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101"))
    d = mujoco.MjData(m)
    kp, kv, tmax = m.actuator_gainprm[0, 0], -m.actuator_biasprm[0, 2], m.actuator_forcerange[0, 1]
    rng = np.random.default_rng(0)
    for _ in range(50):
        d.qpos[:5] = rng.uniform(-0.5, 0.5, 5)
        d.qvel[:5] = rng.normal(size=5)
        d.ctrl[:5] = d.qpos[:5] + rng.normal(0, 0.004, 5)
        mujoco.mj_forward(m, d)
        assert np.allclose(pd_torque(d.ctrl[:5], d.qpos[:5], d.qvel[:5], kp, kv, tmax), d.actuator_force[:5], atol=1e-9)


def test_discount_keeps_horizon_and_metrics():
    for dt in (0.005, 0.02, 0.08):
        assert dt / (1 - discount_for_rate(dt)) == pytest.approx(0.4)
    tau = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    rate, power = torque_rate_and_power(tau, np.array([[1.0, 1.0], [-1.0, 0.5]]), 0.5)
    assert rate == pytest.approx(np.sqrt((4 + 16) / 2)) and power == pytest.approx((1 + 2) / 2)


def test_build_script_reproduces_its_output():
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")                                 # build imports lesson 32's experiment.py
    assert_build_output("34-observation-action-design")
