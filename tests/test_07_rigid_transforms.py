"""Lesson 07: physai.spatial transforms and the frame-tree walk against MuJoCo, plus properties the lesson uses."""
import mujoco
import numpy as np

from common.menagerie import path
from conftest import assert_build_output
from physai.spatial import frame_tree, pose_to_transform, transform, transform_inv, transform_point

RNG = np.random.default_rng(7)


def rand_pose():
    q = RNG.normal(size=4)
    return RNG.uniform(-1, 1, 3), q / np.linalg.norm(q)


def mj_pose(fn, *args):
    pos, quat = np.zeros(3), np.zeros(4)
    fn(pos, quat, *args)
    return pose_to_transform(pos, quat)


def test_inverse_and_composition_match_mujoco():
    for _ in range(50):
        a, b = rand_pose(), rand_pose()
        Ta, Tb = pose_to_transform(*a), pose_to_transform(*b)
        assert np.allclose(transform_inv(Ta), mj_pose(mujoco.mju_negPose, *a), atol=1e-14)
        assert np.allclose(Ta @ Tb, mj_pose(mujoco.mju_mulPose, *a, *b), atol=1e-14)
        assert np.allclose(transform_inv(Ta) @ Ta, np.eye(4), atol=1e-14)


def test_frame_tree_matches_mujoco_on_so101():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    d = mujoco.MjData(m)
    for q in RNG.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1], size=(20, m.njnt)):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        T = frame_tree(m, d.qpos)
        for b in range(m.nbody):
            assert np.allclose(T[b][:3, 3], d.xpos[b], atol=1e-14)
            assert np.allclose(T[b][:3, :3], d.xmat[b].reshape(3, 3), atol=1e-14)


def test_frame_tree_handles_free_ball_and_slide_joints():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body pos="0.1 0.2 0.3"><freejoint/><geom size=".05"/>
        <body pos="0 0 .2" quat="0.9 0.1 0.3 0.2"><joint type="ball" pos="0.01 0 0.02"/><geom size=".03"/>
          <body pos=".1 0 0"><joint type="slide" axis="0.6 0 0.8" ref="0.05"/><joint type="hinge" axis="0 1 0"
            pos="0 0.02 0"/><geom size=".02"/></body></body></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    for _ in range(20):
        d.qpos[:] = RNG.normal(size=m.nq)
        for a in (3, 7):                                      # normalize the two quaternions
            d.qpos[a:a + 4] /= np.linalg.norm(d.qpos[a:a + 4])
        mujoco.mj_kinematics(m, d)
        T = frame_tree(m, d.qpos)
        for b in range(m.nbody):
            assert np.allclose(T[b][:3, 3], d.xpos[b], atol=1e-13)
            assert np.allclose(T[b][:3, :3], d.xmat[b].reshape(3, 3), atol=1e-13)


def test_rigid_transforms_preserve_distances():
    T = transform(pose_to_transform(*rand_pose())[:3, :3], RNG.normal(size=3))
    p, r = RNG.normal(size=(2, 3))
    assert np.isclose(np.linalg.norm(transform_point(T, p) - transform_point(T, r)), np.linalg.norm(p - r))


def test_build_script_reproduces_its_output():
    assert_build_output("07-rigid-transforms")
