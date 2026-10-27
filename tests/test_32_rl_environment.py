"""Lesson 32: physai.rl.ReachTask against the lesson's ReachEnv, and the value target."""
import os
import sys

import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import ROOT, assert_build_output
from physai.rl import ReachTask, value_target


def test_seeding_and_flags():
    env = ReachTask(path("robotstudio_so101"), episode_seconds=0.2)
    o1, _ = env.reset(seed=3)
    t1 = env.target.copy()
    env.reset(seed=4)
    o2, _ = env.reset(seed=3)
    assert np.array_equal(t1, env.target) and np.array_equal(o1, o2) and o1.shape == (15,) and o1.dtype == np.float32
    flags = [env.step(np.zeros(4))[2:4] for _ in range(env.max_steps)]
    assert flags[-1] == (False, True) and all(f == (False, False) for f in flags[:-1])


def test_frame_skip_holds_the_command():
    env = ReachTask(path("robotstudio_so101"), frame_skip=3)
    env.reset(seed=0)
    d = mujoco.MjData(env.model)
    d.qpos[:], d.qvel[:], d.ctrl[:] = env.data.qpos, env.data.qvel, env.data.ctrl
    a = np.array([0.2, -0.1, 0.3, 0.0])
    env.step(a)
    d.ctrl[env.act] = env.Q_HOME + a
    for _ in range(3):
        mujoco.mj_step(env.model, d)
    assert np.array_equal(d.qpos, env.data.qpos) and env.dt == pytest.approx(0.015)


def test_matches_reach_env():
    pytest.importorskip("gymnasium")
    sys.path.insert(0, os.path.join(ROOT, "experiments", "32-rl-environment"))
    from reach_env import ReachEnv
    ours, ref = ReachTask(path("robotstudio_so101"), terminate_on_success=True), ReachEnv(terminate_on_success=True)
    rng = np.random.default_rng(0)
    for s in range(3):
        assert np.array_equal(ours.reset(seed=s)[0], ref.reset(seed=s)[0])
        for _ in range(80):
            a = rng.uniform(-1, 1, 4)
            r1, r2 = ours.step(a), ref.step(a)
            assert np.array_equal(r1[0], r2[0]) and r1[1:4] == r2[1:4]
            if r1[2] or r1[3]:
                break


def test_value_target():
    assert value_target(1.0, 10.0, 0.0, 0.9) == pytest.approx(10.0)
    assert value_target(1.0, 10.0, 1.0, 0.9) == 1.0
    V = -0.01 / (1 - 0.95)                                        # a stationary state: bootstrapping is consistent
    assert value_target(-0.01, V, 0.0, 0.95) == pytest.approx(V)


def test_build_script_reproduces_its_output():
    pytest.importorskip("gymnasium")
    assert_build_output("32-rl-environment")
