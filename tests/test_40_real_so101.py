"""Lesson 40: calibration arithmetic and the gravity-torque budget against MuJoCo."""
import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import assert_build_output
from physai.sim2real import deg_to_ticks, fit_calibration, gravity_torques, max_payload, ticks_to_deg
from physai.spatial import frame_tree


def test_ticks_degrees_round_trip_and_truncation():
    raw = np.arange(500, 3600)
    assert all(deg_to_ticks(ticks_to_deg(r, 500, 3599), 500, 3599) == r for r in raw)
    goals = np.linspace(-60, 60, 2001)
    err = np.array([ticks_to_deg(deg_to_ticks(g, 500, 3599), 500, 3599) for g in goals]) - goals
    assert err.max() <= 1e-12 and err.min() >= -360 / 4095 - 1e-12


def test_fit_calibration_recovers_sign_and_offset():
    deg = np.array([-30.0, 10.0, 45.0])
    q = -1.0 * np.radians(deg) + 0.2
    a, b = fit_calibration(deg, q)
    assert a == pytest.approx(-1.0) and b == pytest.approx(0.2)


@pytest.mark.parametrize("seed", range(3))
def test_gravity_torques_match_qfrc_bias(seed):
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    d = mujoco.MjData(m)
    d.qpos[:6] = np.random.default_rng(seed).uniform(m.jnt_range[:6, 0], m.jnt_range[:6, 1])
    mujoco.mj_forward(m, d)
    assert np.abs(gravity_torques(m, d.qpos, frame_tree(m, d.qpos)) + d.qfrc_bias).max() < 1e-12


def test_max_payload_is_where_the_first_joint_reaches_the_limit():
    tau0 = np.array([[0.5, -1.0], [1.0, 0.2]])
    per_kg = np.array([[1.0, -2.0], [0.5, 0.0]])
    p = max_payload(tau0, per_kg, 2.0)
    assert p == pytest.approx(0.5)                       # pose 0, joint 1: -1 - 2 p = -2
    assert np.abs(tau0 + p * per_kg).max() == pytest.approx(2.0)


def test_build_script_reproduces_its_output():
    assert_build_output("40-real-so101")
