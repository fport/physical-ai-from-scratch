"""Lesson 06: physai.spatial rotations against MuJoCo's mju_* functions and SciPy, plus properties the lesson uses."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from conftest import assert_build_output
from physai.spatial import (axis_angle_to_mat, axis_angle_to_quat, euler_rate_matrix, euler_zyx_to_mat,
                            mat_to_euler_zyx, mat_to_quat, quat_angle, quat_conj, quat_mul, quat_rotate, quat_to_mat,
                            slerp)

RNG = np.random.default_rng(6)
Q = RNG.normal(size=(50, 4))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)


def mj(fn, *args, n=4):
    out = np.zeros(n)
    fn(out, *args)
    return out


def pos_w(q):
    return q if q[0] >= 0 else -q


def test_quaternion_algebra_matches_mujoco():
    for a, b, v in zip(Q, np.roll(Q, 1, axis=0), RNG.normal(size=(50, 3))):
        assert np.allclose(quat_mul(a, b), mj(mujoco.mju_mulQuat, a, b), atol=1e-15)
        assert np.allclose(quat_conj(a), mj(mujoco.mju_negQuat, a))
        assert np.allclose(quat_rotate(a, v), mj(mujoco.mju_rotVecQuat, v, a, n=3), atol=1e-14)
        assert np.allclose(quat_to_mat(quat_mul(a, b)), quat_to_mat(a) @ quat_to_mat(b), atol=1e-14)


def test_conversions_match_mujoco_and_scipy():
    for q in Q:
        R = quat_to_mat(q)
        assert np.allclose(mat_to_quat(R), pos_w(q), atol=1e-14)
        assert np.allclose(mat_to_quat(R), pos_w(mj(mujoco.mju_mat2Quat, R.flatten())), atol=1e-14)
        k, a = q[1:] / np.linalg.norm(q[1:]), RNG.uniform(-np.pi, np.pi)
        assert np.allclose(axis_angle_to_quat(k, a), mj(mujoco.mju_axisAngle2Quat, k, a), atol=1e-15)
        assert np.allclose(quat_to_mat(axis_angle_to_quat(k, a)), axis_angle_to_mat(k, a), atol=1e-14)
    for R in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):   # 180 deg: w = 0 branches
        assert np.allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-15)


def test_euler_zyx_matches_mujoco_and_scipy_and_round_trips():
    for e in np.c_[RNG.uniform(-3, 3, 50), RNG.uniform(-1.5, 1.5, 50), RNG.uniform(-3, 3, 50)]:
        R = euler_zyx_to_mat(*e)
        assert np.allclose(R, quat_to_mat(mj(mujoco.mju_euler2Quat, e, "zyx")), atol=1e-14)
        assert np.allclose(R, Rotation.from_euler("ZYX", e).as_matrix(), atol=1e-14)
        assert np.allclose(mat_to_euler_zyx(R), e, atol=1e-9)


def test_euler_rate_matrix_determinant_and_finite_differences():
    for pitch, roll in RNG.uniform(-1.5, 1.5, (20, 2)):
        E = euler_rate_matrix(pitch, roll)
        assert np.isclose(np.linalg.det(E), np.cos(pitch), atol=1e-14)
        ang, eps = np.array([0.3, pitch, roll]), 1e-6          # (yaw, pitch, roll)
        for col, idx in enumerate((2, 1, 0)):                   # rates of roll, pitch, yaw
            d = np.zeros(3)
            d[idx] = eps
            W = euler_zyx_to_mat(*ang).T @ (euler_zyx_to_mat(*(ang + d)) - euler_zyx_to_mat(*(ang - d))) / (2 * eps)
            assert np.allclose([W[2, 1], W[0, 2], W[1, 0]], E[:, col], atol=1e-8)


def test_slerp_matches_scipy_and_needs_the_sign_check():
    for a, b in zip(Q[:25], Q[25:]):
        ref = Slerp([0, 1], Rotation.from_quat([a, b], scalar_first=True))(0.4).as_quat(scalar_first=True)
        assert np.allclose(pos_w(slerp(a, b, 0.4)), pos_w(ref), atol=1e-12)
        assert np.isclose(quat_angle(slerp(a, b, 0.4), a), 0.4 * quat_angle(a, b), atol=1e-9)
    a, b = Q[0], -quat_mul(Q[0], axis_angle_to_quat([0, 0, 1], np.radians(60)))   # 60 deg apart, negative dot
    assert np.isclose(np.degrees(quat_angle(slerp(a, b, 0.5), a)), 30)
    assert np.isclose(np.degrees(quat_angle(slerp(a, b, 0.5, shortest=False), a)), 150)   # the long way round


def test_build_script_reproduces_its_output():
    assert_build_output("06-rotations")
