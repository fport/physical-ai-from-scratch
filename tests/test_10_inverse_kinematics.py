"""Lesson 10: physai.kinematics damped least-squares IK against MuJoCo's FK and the SVD formula."""
import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import assert_build_output
from physai.kinematics import chain_from_mujoco, dls_step, fk, ik_solve, jacobian


@pytest.fixture(scope="module")
def arm():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    return m, mujoco.MjData(m), chain_from_mujoco(m, "gripperframe"), m.jnt_range[:5, 0], m.jnt_range[:5, 1]


@pytest.mark.parametrize("lam", [0.0, 0.03, 0.3])
def test_dls_step_matches_svd_formula(lam):
    rng = np.random.default_rng(0)
    J, e = rng.normal(size=(3, 5)), rng.normal(size=3)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    assert np.allclose(dls_step(J, e, lam), Vt.T @ (s / (s ** 2 + lam ** 2) * (U.T @ e)), atol=1e-12)


def test_solutions_land_on_target_in_mujoco_and_respect_limits(arm):
    m, d, chain, lo, hi = arm
    sid = m.site("gripperframe").id
    for q_true in np.random.default_rng(4).uniform(lo, hi, size=(20, 5)):
        target = fk(chain, q_true)[2]
        q, ok, _ = ik_solve(chain, target, np.zeros(5), lo, hi, lam=0.1)
        assert np.all((q >= lo) & (q <= hi))
        if ok:
            d.qpos[:5] = q
            mujoco.mj_kinematics(m, d)
            assert np.linalg.norm(d.site_xpos[sid] - target) < 5e-4


def test_damping_bounds_the_step_near_the_stretched_singularity(arm):
    _, _, chain, _, _ = arm
    J = jacobian(chain, np.array([0, 1.3, -1.287, 0.035, 0]))[:3]
    u3 = np.linalg.svd(J)[0][:, 2]
    v = 0.05 * u3
    assert np.linalg.norm(dls_step(J, v, 0.0)) > 100          # pseudo-inverse: hundreds of rad/s
    assert np.linalg.norm(dls_step(J, v, 0.1)) < 0.01         # DLS: bounded, gives up on that direction


def test_build_script_reproduces_its_output():
    assert_build_output("10-inverse-kinematics")
