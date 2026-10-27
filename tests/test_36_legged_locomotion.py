"""Lesson 36: the vectorized environment against MuJoCo, SciPy and mujoco.rollout."""
import mujoco
import numpy as np
import pytest
from mujoco import rollout
from scipy.spatial.transform import Rotation

from common.menagerie import path
from conftest import assert_build_output
from physai.vecenv import VecEnv, projected_gravity, roll_pitch, samples_per_update

ARM = """<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/>
  <body name="box" pos="0 0 0.3"><freejoint/><geom type="box" size=".1 .05 .05" mass="1"/>
    <body pos="0.1 0 0"><joint name="j" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".01" mass=".1"/>
  </body></body></worldbody><actuator><position joint="j" kp="5"/></actuator></mujoco>"""


@pytest.mark.parametrize("seed", range(3))
def test_projected_gravity_and_roll_pitch(seed):
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    assert np.allclose(projected_gravity(q), R.inv().apply([0, 0, -1]), atol=1e-12)
    _, pitch, roll = R.as_euler("ZYX")
    assert np.allclose(roll_pitch(q), (roll, pitch), atol=1e-12)


def test_batch_equals_rollout_and_resets_in_the_same_step():
    m = mujoco.MjModel.from_xml_string(ARM)

    def reset_fn(m, d, rng):
        mujoco.mj_resetData(m, d)
        d.qpos[7] = rng.uniform(-0.3, 0.3)
        mujoco.mj_forward(m, d)

    env = VecEnv(m, 4, reset_fn, np.random.default_rng(0), decimation=3, max_steps=50, min_height=-1.0,
                 max_roll=10.0, max_pitch=10.0)
    env.reset()
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    x0 = np.array([np.zeros(mujoco.mj_stateSize(m, spec)) for _ in range(4)])
    for i, d in enumerate(env.datas):
        mujoco.mj_getState(m, d, x0[i], spec)
    acts = np.random.default_rng(1).uniform(-1, 1, (40, 4, 1))
    state, _ = rollout.rollout(m, mujoco.MjData(m), x0, np.repeat(acts.transpose(1, 0, 2), 3, axis=1))
    for a in acts:
        env.step(a)
    assert np.abs(state[:, -1, 1:1 + m.nq] - np.array([d.qpos for d in env.datas])).max() == 0.0
    for a in acts[:10]:
        _, fell, timed_out = env.step(a)
    assert timed_out.all() and env.episodes == 4 and np.all(env.t == 0)
    assert env.physics_steps == 4 * 50 * 3


def test_go2_falls_are_detected_and_reset():
    m = mujoco.MjModel.from_xml_path(path("unitree_go2", "scene.xml"))
    key = m.key("home").id

    def reset_fn(m, d, rng):
        mujoco.mj_resetDataKeyframe(m, d, key)
        mujoco.mj_forward(m, d)

    env = VecEnv(m, 2, reset_fn, np.random.default_rng(0))
    env.reset()
    env.datas[0].qpos[3:7] = [np.cos(0.6), np.sin(0.6), 0, 0]   # rolled by 1.2 rad
    _, fell, _ = env.step(np.tile(m.key_ctrl[key], (2, 1)))
    assert fell.tolist() == [True, False] and env.datas[0].qpos[3] == pytest.approx(1.0)


def test_samples_per_update():
    assert samples_per_update(4096, 24) == 98_304


def test_build_script_reproduces_its_output():
    assert_build_output("36-legged-locomotion")
