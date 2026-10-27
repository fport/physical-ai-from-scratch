"""Lesson 08: physai.kinematics forward kinematics and workspace sampling against MuJoCo."""
import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import assert_build_output
from physai.kinematics import chain_from_mujoco, fk, sample_workspace, voxel_volume


@pytest.fixture(scope="module")
def arm():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    return m, mujoco.MjData(m), chain_from_mujoco(m, "gripperframe")


def test_fk_matches_mj_kinematics(arm):
    m, d, chain = arm
    sid = m.site("gripperframe").id
    for q in np.random.default_rng(1).uniform(m.jnt_range[:5, 0], m.jnt_range[:5, 1], size=(50, 5)):
        d.qpos[:5] = q
        mujoco.mj_kinematics(m, d)
        axes, anchors, p = fk(chain, q)
        assert np.allclose(p, d.site_xpos[sid], atol=1e-12)
        assert np.allclose(axes, d.xaxis[:5], atol=1e-12)      # hinge axes in the world, same order
        assert np.allclose(anchors, d.xanchor[:5], atol=1e-12)


def test_pitch_axes_stay_parallel(arm):
    _, _, chain = arm
    for q in np.random.default_rng(2).uniform(-1.5, 1.5, size=(20, 5)):
        axes = fk(chain, q)[0]
        assert max(np.linalg.norm(np.cross(axes[1], axes[k])) for k in (2, 3)) < 1e-12


def test_voxel_volume_counts_cells():
    pts = np.array([[0.001, 0.001, 0.001], [0.004, 0.002, 0.003], [0.015, 0.0, 0.0]])
    assert voxel_volume(pts, 0.01) == pytest.approx(2e-6)      # two occupied 1 cm cells
    assert voxel_volume(pts, 0.1) == pytest.approx(1e-3)


def test_workspace_volume_grows_with_samples(arm):
    m, _, chain = arm
    pts = sample_workspace(chain, m.jnt_range[:5, 0], m.jnt_range[:5, 1], 2000)
    assert voxel_volume(pts[:500], 0.01) < voxel_volume(pts, 0.01)
    assert np.all(np.linalg.norm(pts - fk(chain, np.zeros(5))[1][0], axis=1) < 0.48)   # reach below 0.473 m + margin


def test_build_script_reproduces_its_output():
    assert_build_output("08-forward-kinematics")
