"""Lesson 11: physai.dynamics (RNEA, composite rigid bodies, forward dynamics) against MuJoCo."""
import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import assert_build_output
from physai.dynamics import energy, forward_dynamics, mass_matrix, rnea, tree_from_mujoco

PENDULUM = """
<mujoco>
  <option gravity="0 0 -9.81"><flag energy="enable"/></option>
  <worldbody>
    <body>
      <joint type="hinge" axis="0 1 0"/>
      <inertial pos="0.2 0 0" mass="1" diaginertia="0.02 0.02 0.02"/>
      <body pos="0.4 0 0">
        <joint type="slide" axis="1 0 0"/>
        <inertial pos="0.1 0.02 0" quat="0.9 0.1 0.3 0.2" mass="0.5" diaginertia="0.003 0.005 0.007"/>
        <body pos="0.2 0 0.05" euler="0.3 0.2 0.1">
          <joint type="hinge" axis="0 0.6 0.8" pos="0 0 0.02"/>
          <inertial pos="0.15 0 0.01" mass="0.3" diaginertia="0.002 0.003 0.004"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module", params=["xml", "so101", "panda"])
def model(request):
    if request.param == "xml":
        m = mujoco.MjModel.from_xml_string(PENDULUM)
    elif request.param == "so101":
        m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    else:
        m = mujoco.MjModel.from_xml_path(path("franka_emika_panda"))
    return m, mujoco.MjData(m), tree_from_mujoco(m)


def random_states(m, n=5, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.uniform(-1, 1, m.nv) * 0.3, rng.uniform(-3, 3, m.nv), rng.uniform(-10, 10, m.nv)


def test_mass_matrix_matches_mj_fullM(model):
    m, d, tree = model
    Mf = np.zeros((m.nv, m.nv))
    for q, _, _ in random_states(m):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, Mf)
        assert np.allclose(mass_matrix(tree, q), Mf, atol=1e-12)


def test_bias_matches_qfrc_bias(model):
    m, d, tree = model
    for q, qd, _ in random_states(m):
        d.qpos[:], d.qvel[:] = q, qd
        mujoco.mj_forward(m, d)
        assert np.allclose(rnea(tree, q, qd, np.zeros(m.nv)), d.qfrc_bias, atol=1e-11)


def test_rnea_matches_mj_inverse_without_passive_forces():
    m = mujoco.MjModel.from_xml_string(PENDULUM)
    d, tree = mujoco.MjData(m), tree_from_mujoco(m)
    for q, qd, qdd in random_states(m, 20):
        d.qpos[:], d.qvel[:], d.qacc[:] = q, qd, qdd
        mujoco.mj_inverse(m, d)
        assert np.allclose(rnea(tree, q, qd, qdd), d.qfrc_inverse, atol=1e-12)


def test_forward_dynamics_inverts_rnea(model):
    m, _, tree = model
    for q, qd, qdd in random_states(m, 3):
        assert np.allclose(forward_dynamics(tree, q, qd, rnea(tree, q, qd, qdd)), qdd, atol=1e-8)


def test_mass_matrix_is_symmetric_positive_definite(model):
    m, _, tree = model
    for q, _, _ in random_states(m, 3):
        M = mass_matrix(tree, q)
        assert np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > 0


def test_energy_matches_mujoco():
    m = mujoco.MjModel.from_xml_string(PENDULUM)
    d, tree = mujoco.MjData(m), tree_from_mujoco(m)
    for q, qd, _ in random_states(m):
        d.qpos[:], d.qvel[:] = q, qd
        mujoco.mj_forward(m, d)
        assert energy(tree, q, qd) == pytest.approx(d.energy.sum(), abs=1e-12)


def test_build_script_reproduces_its_output():
    assert_build_output("11-rigid-body-dynamics")
