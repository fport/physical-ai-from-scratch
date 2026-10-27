"""Lesson 09: physai.kinematics against MuJoCo, plus properties the lesson relies on."""
import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import assert_build_output
from physai.kinematics import chain_from_mujoco, fd_jacobian, fk, hinge_column, jacobian


@pytest.fixture(scope="module")
def arm():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    qs = np.random.default_rng(1).uniform(m.jnt_range[:5, 0], m.jnt_range[:5, 1], size=(20, 5))
    return m, mujoco.MjData(m), chain_from_mujoco(m, "gripperframe"), qs


def mujoco_site_jac(m, d, q):
    d.qpos[:5] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    sid = m.site("gripperframe").id
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    return d.site_xpos[sid].copy(), np.r_[jp[:, :5], jr[:, :5]]


def test_fk_matches_mujoco(arm):
    m, d, chain, qs = arm
    for q in qs:
        assert np.allclose(fk(chain, q)[2], mujoco_site_jac(m, d, q)[0], atol=1e-12)


def test_jacobian_matches_mj_jacsite(arm):
    m, d, chain, qs = arm
    for q in qs:
        assert np.allclose(jacobian(chain, q), mujoco_site_jac(m, d, q)[1], atol=1e-12)


def test_central_differences_agree(arm):
    _, _, chain, qs = arm
    for q in qs:
        assert np.abs(fd_jacobian(lambda x: fk(chain, x)[2], q) - jacobian(chain, q)[:3]).max() < 1e-9


def test_hinge_column_is_perpendicular_to_axis_and_lever():
    a, o, p = np.array([0, 0, 1.0]), np.zeros(3), np.array([0.3, 0.1, 0.2])
    col = hinge_column(a, o, p)
    assert abs(col[:3] @ a) < 1e-15 and abs(col[:3] @ (p - o)) < 1e-15
    assert np.allclose(col[3:], a)


def test_vertical_push_loads_no_pan_torque(arm):
    _, _, chain, _ = arm
    J = jacobian(chain, np.array([0.0, -0.6, 0.9, 0.4, 0.3]))
    assert abs((J[:3].T @ np.array([0, 0, -2.0]))[0]) < 1e-12   # pan axis is vertical


def test_build_script_reproduces_its_output():
    assert_build_output("09-jacobians")
