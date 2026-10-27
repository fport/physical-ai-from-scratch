"""Lesson 43: the pendulum and the least-squares world model against MuJoCo, plus properties the lesson relies on."""
import mujoco
import numpy as np
import pytest

from conftest import assert_build_output
from physai.foundation import fit_world_model, pendulum_step, predict, rollout_errors, wrap

XML = """<mujoco><option timestep="0.01" gravity="0 0 -9.81"/><worldbody><body>
  <joint name="j" type="hinge" axis="0 1 0" damping="0.05"/>
  <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.01" mass="0"/>
  <geom type="sphere" pos="0 0 -0.5" size="0.03" mass="1"/></body></worldbody>
  <actuator><motor joint="j" ctrlrange="-2 2" ctrllimited="true"/></actuator></mujoco>"""


@pytest.mark.parametrize("seed", range(3))
def test_pendulum_step_matches_mujoco_over_a_trajectory(seed):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    s = rng.uniform([-np.pi, -6], [np.pi, 6])
    d.qpos[0], d.qvel[0] = s
    for u in rng.uniform(-2.5, 2.5, 40):
        d.ctrl[0] = u
        mujoco.mj_step(m, d, nstep=5)
        s = pendulum_step(s, u)
    assert np.abs(s - [d.qpos[0], d.qvel[0]]).max() < 1e-10


def test_batched_step_equals_single_steps():
    rng = np.random.default_rng(4)
    S, U = rng.uniform(-3, 3, (7, 2)), rng.uniform(-2, 2, 7)
    assert np.allclose(pendulum_step(S, U), np.array([pendulum_step(s, u) for s, u in zip(S, U)]), atol=1e-15)


def test_higher_degree_fits_better_and_exact_model_has_zero_rollout_error():
    rng = np.random.default_rng(0)
    S = np.stack([rng.uniform(-np.pi, np.pi, 3000), rng.uniform(-8, 8, 3000)], axis=-1)
    A = rng.uniform(-2, 2, 3000)
    S2 = pendulum_step(S, A)
    errs = [np.abs(predict(fit_world_model(S, A, S2, deg), S, A, deg) - S2).mean() for deg in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2]
    U = rng.uniform(-2, 2, (20, 30))
    res = rollout_errors(pendulum_step, pendulum_step, S[:20], U, (1, 30))
    assert res[30][0] == 0.0


def test_wrap_maps_into_minus_pi_pi():
    a = np.linspace(-20, 20, 1001)
    w = wrap(a)
    assert np.all((w >= -np.pi) & (w < np.pi)) and np.allclose(np.sin(w), np.sin(a))


def test_build_script_reproduces_its_output():
    assert_build_output("43-world-models")
