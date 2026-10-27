"""Lesson 19: physai.vision camera model and calibration against SciPy, OpenCV and exact synthetic data."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import assert_build_output
from physai.vision import (hand_eye_park, homography_dlt, intrinsics_from_fovy, mat_to_rotvec,
                           pose_from_homography, project_points, refine_calibration, rotvec_to_mat,
                           zhang_intrinsics)

RNG = np.random.default_rng(19)
K = np.array([[600.0, 0.0, 330.0], [0.0, 590.0, 245.0], [0.0, 0.0, 1.0]])
OBJ = np.c_[np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025 - [0.1, 0.0625], np.zeros(54)]


def random_board_poses(n, tilt_deg=35.0):
    poses = []
    for _ in range(n):
        R = Rotation.from_euler("xyz", RNG.uniform(-np.radians(tilt_deg), np.radians(tilt_deg), 3)).as_matrix()
        poses.append((R, np.array([RNG.uniform(-0.05, 0.05), RNG.uniform(-0.05, 0.05), RNG.uniform(0.45, 0.65)])))
    return poses


def test_fovy_edge_lands_on_the_image_edge():
    for fovy, w, h in [(50, 640, 480), (61, 1280, 720), (90, 100, 80)]:
        Kf = intrinsics_from_fovy(fovy, w, h)
        top = np.array([[0.0, -np.tan(np.radians(fovy) / 2), 1.0]])     # a ray along the upper edge of the view
        assert np.isclose(project_points(top, np.eye(3), np.zeros(3), Kf)[0, 1], -0.5)  # edge of pixel row 0
        assert Kf[0, 0] == Kf[1, 1]


def test_rotvec_roundtrip_matches_scipy():
    for w in RNG.normal(0, 1.0, (20, 3)):
        assert np.allclose(rotvec_to_mat(w), Rotation.from_rotvec(w).as_matrix(), atol=1e-12)
        assert np.allclose(mat_to_rotvec(rotvec_to_mat(w)), Rotation.from_matrix(rotvec_to_mat(w)).as_rotvec(), atol=1e-9)


def test_homography_recovers_an_exact_map():
    for _ in range(10):
        H = np.eye(3) + RNG.normal(0, 0.1, (3, 3))
        src = RNG.uniform(-1, 1, (8, 2))
        q = np.c_[src, np.ones(8)] @ H.T
        assert np.allclose(homography_dlt(src, q[:, :2] / q[:, 2:]), H / H[2, 2], atol=1e-9)


def test_zhang_and_pose_from_homography_are_exact_on_exact_data():
    poses = random_board_poses(6)
    Hs = [K @ np.c_[R[:, 0], R[:, 1], t] for R, t in poses]
    Kz, _ = zhang_intrinsics(Hs)
    assert np.allclose(Kz, K, atol=1e-6)
    for H, (R, t) in zip(Hs, poses):
        Re, te = pose_from_homography(H, Kz)
        assert np.allclose(Re, R, atol=1e-8) and np.allclose(te, t, atol=1e-9)


def test_refine_recovers_intrinsics_and_distortion():
    poses = random_board_poses(8)
    dist = (-0.12, 0.05, 0.0, 0.0, 0.0)
    views = [project_points(OBJ, R, t, K, dist) for R, t in poses]
    starts = [(R @ rotvec_to_mat(RNG.normal(0, 0.01, 3)), t + RNG.normal(0, 0.002, 3)) for R, t in poses]
    Kr, dr, _, rms = refine_calibration(OBJ, views, K * [[1.02], [0.98], [1]], starts, iters=20)
    assert rms < 1e-6 and np.allclose(Kr, K, rtol=1e-6) and np.allclose(dr, dist, atol=1e-6)


def test_hand_eye_exact_data():
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = rotvec_to_mat([0.3, -0.2, 0.5]), [0.02, -0.07, 0.03]
    G = [np.eye(4) for _ in range(6)]
    for g in G:
        g[:3, :3], g[:3, 3] = rotvec_to_mat(RNG.normal(0, 0.8, 3)), RNG.normal(0, 0.2, 3)
    T_bt = np.eye(4)
    T_bt[:3, 3] = [0.25, 0.0, 0.0]
    C = [np.linalg.inv(g @ X) @ T_bt for g in G]
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    A = [np.linalg.inv(G[j]) @ G[i] for i, j in pairs]
    B = [C[j] @ np.linalg.inv(C[i]) for i, j in pairs]
    assert np.allclose(hand_eye_park(A, B), X, atol=1e-10)


def test_projection_and_calibration_match_opencv():
    cv2 = pytest.importorskip("cv2")
    poses = random_board_poses(10)
    dist = np.array([-0.1, 0.02, 0.001, -0.002, 0.005])
    for R, t in poses:
        p, _ = cv2.projectPoints(OBJ, cv2.Rodrigues(R)[0], t, K, dist)
        assert np.allclose(p.reshape(-1, 2), project_points(OBJ, R, t, K, dist), atol=1e-9)
    views = [project_points(OBJ, R, t, K) + RNG.normal(0, 0.1, (54, 2)) for R, t in poses]
    for v in views[:3]:
        Hc, _ = cv2.findHomography(OBJ[:, :2], v, 0)
        q = np.c_[OBJ[:, :2], np.ones(54)]
        a, b = q @ homography_dlt(OBJ[:, :2], v).T, q @ Hc.T
        assert np.abs(a[:, :2] / a[:, 2:] - b[:, :2] / b[:, 2:]).max() < 0.05
    Hs = [homography_dlt(OBJ[:, :2], v) for v in views]
    Kz, _ = zhang_intrinsics(Hs)
    Kr, dr, _, rms = refine_calibration(OBJ, views, Kz, [pose_from_homography(H, Kz) for H in Hs])
    flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    rms_cv, K_cv, d_cv, _, _ = cv2.calibrateCamera([OBJ.astype(np.float32)] * 10, [v.astype(np.float32) for v in views],
                                                   (640, 480), None, None, flags=flags)
    assert rms <= rms_cv + 1e-9                         # the same optimum, OpenCV stops a little earlier
    assert np.allclose(Kr, K_cv, atol=1e-3) and np.allclose(dr, d_cv.ravel(), atol=1e-4)


def test_build_script_reproduces_its_output():
    pytest.importorskip("cv2")
    assert_build_output("19-camera-calibration")
