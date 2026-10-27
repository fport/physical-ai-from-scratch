"""Lesson 26: physai.data (frames, hold, re-targeting) against the saved dataset and the SO-101 kinematics."""
import os

import mujoco
import numpy as np

from common.menagerie import path
from conftest import ROOT, assert_build_output
from physai.data import hold, retarget, to_frames
from physai.kinematics import chain_from_mujoco, fk, jacobian


def test_frames_and_hold_round_trip():
    steps = np.arange(100.0)[:, None] * np.ones((1, 3))
    obs, act = to_frames(-np.ones(3), steps, steps + 0.5, 20)
    assert obs.shape == (5, 3) and act.shape == (5, 3)
    assert np.array_equal(obs[:, 0], [-1, 19, 39, 59, 79]) and np.array_equal(act[:, 0], [19.5, 39.5, 59.5, 79.5, 99.5])
    held = hold(act, 20)
    assert len(held) == 100 and np.array_equal(held[20:40], np.repeat(act[1:2], 20, axis=0))


def test_retarget_moves_the_tool_by_the_shift_to_first_order():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    chain = chain_from_mujoco(m, "gripperframe")
    q = np.array([0.1, -0.4, 0.8, 1.2, 1.6])
    errs = []
    for s in (0.004, 0.008, 0.016):
        shift = np.array([0.0, s, 0.0])
        q2 = retarget(q[None], lambda x: jacobian(chain, x)[:3], shift)[0]
        errs.append(np.linalg.norm(fk(chain, q2)[2] - fk(chain, q)[2] - shift))
    assert errs[0] < 1e-3 * 0.004 * 100 and 3.0 < errs[2] / errs[1] < 5.0     # second-order residual


def test_saved_demos_are_25_hz_frames():
    D = np.load(os.path.join(ROOT, "experiments", "26-teleop-replay-data", "demos.npz"))
    assert int(D["fps"]) == 25 and np.allclose(np.diff(D["timestamp"][:114]), 0.04, atol=1e-6)


def test_build_script_reproduces_its_output():
    assert_build_output("26-teleop-replay-data")
