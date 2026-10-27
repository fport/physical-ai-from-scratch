"""Lesson 14: physai.control task-space functions against MuJoCo and the properties the lesson relies on."""
import os
import sys

import mujoco
import numpy as np
import pytest

from conftest import ROOT, assert_build_output
from physai.control import (impedance_wrench, null_space_projector, orientation_error, osc_wrench, task_inertia,
                            task_torque)

sys.path.insert(0, os.path.join(ROOT, "experiments", "14-impedance-control"))


@pytest.fixture(scope="module")
def panda():
    from experiment import build
    m = build()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    jp, jr, M = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((m.nv, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, m.site("tcp").id)
    mujoco.mj_fullM(m, d, M)
    return np.vstack([jp, jr])[:, :7], M[:7, :7]


def test_null_space_torque_never_accelerates_the_tool(panda):
    J, M = panda
    N = null_space_projector(J, M, task_inertia(J, M))
    tau0 = np.random.default_rng(0).normal(size=7)
    assert np.abs(J @ np.linalg.solve(M, N @ tau0)).max() < 1e-12


def test_osc_gives_unit_mass_response(panda):
    J, M = panda
    Lam = task_inertia(J, M)
    e, v = np.array([0.01, -0.02, 0.005, 0.01, 0, -0.02]), np.zeros(6)
    F = osc_wrench(Lam, e, v, 100.0, 14.0)
    assert np.allclose(J @ np.linalg.solve(M, J.T @ F), 100.0 * e)   # tool acceleration (ignoring Jdot qdot)


def test_task_inertia_is_symmetric_positive_definite(panda):
    Lam = task_inertia(*panda)
    assert np.allclose(Lam, Lam.T, atol=1e-9) and np.linalg.eigvalsh(Lam).min() > 0


def test_orientation_error_is_sin_angle_times_axis():
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("zyx", [0.4, 0.1, -0.7]).as_matrix()
    for angle in (0.01, 0.5, 1.5):
        k = np.array([1.0, 2.0, 2.0]) / 3
        e = orientation_error(R, Rotation.from_rotvec(angle * k).as_matrix() @ R)
        assert np.allclose(e, np.sin(angle) * k, atol=1e-14)


def test_impedance_wrench_and_torque_clip():
    F = impedance_wrench(np.array([0.05, 0, 0, 0, 0, 0.1]), np.array([0.1, 0, 0, 0, 0, 0]), 600, 50, 30, 3)
    assert np.allclose(F, [600 * 0.05 - 50 * 0.1, 0, 0, 0, 0, 3.0])
    tau, clipped = task_torque(np.eye(6), F * 10, np.zeros(6), np.full(6, 12.0))
    assert clipped and np.abs(tau).max() == 12.0


def test_build_script_reproduces_its_output():
    assert_build_output("14-impedance-control")
