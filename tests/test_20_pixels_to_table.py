"""Lesson 20: physai.vision rays, depth and homographies against SciPy, OpenCV and exact geometry."""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from conftest import assert_build_output
from physai.vision import (apply_homography, back_project, bilinear, homography_dlt, intrinsics_from_fovy,
                           linearize_depth, pixel_rays, plane_homography, project_points, ray_plane)

RNG = np.random.default_rng(20)
K = intrinsics_from_fovy(50, 640, 480)
R_WC = Rotation.from_euler("xyz", [np.pi - np.radians(30), 0.0, np.pi / 2]).as_matrix()   # looks down, tilted
C = np.array([-0.1, 0.0, 0.6])


def table_view(n):
    uv = np.c_[RNG.uniform(0, 639, n), RNG.uniform(0, 479, n)]
    return uv[pixel_rays(uv, K, R_WC)[:, 2] < 0]          # keep rays that go down to the table


def test_back_projection_inverts_projection():
    P = np.c_[RNG.uniform(0.1, 0.4, (50, 2)), RNG.uniform(0, 0.1, 50)]
    R, t = R_WC.T, -R_WC.T @ C
    uv = project_points(P, R, t, K)
    Z = (P - C) @ R_WC[:, 2]                              # depth along the optical axis
    assert np.allclose(back_project(uv, Z, K, R_WC, C), P, atol=1e-12)


def test_ray_plane_lands_on_the_plane_and_projects_back():
    uv = table_view(200)
    for z0 in (0.0, 0.05):
        P, s = ray_plane(uv, K, R_WC, C, z0)
        assert np.allclose(P[:, 2], z0, atol=1e-12) and np.all(s > 0)
        assert np.allclose(project_points(P, R_WC.T, -R_WC.T @ C, K), uv, atol=1e-8)


def test_linearize_depth_matches_opengl_reversed_z():
    near, far = 0.005, 24.9
    assert np.isclose(linearize_depth(1.0, near, far), near) and np.isclose(linearize_depth(0.0, near, far), far)
    c_coef = -0.5 * (-(far + near) / (far - near)) - 0.5      # the steps MuJoCo's Renderer applies
    d_coef = -0.5 * (-2 * far * near / (far - near))
    d = RNG.uniform(0, 1, 100)
    assert np.allclose(linearize_depth(d, near, far), d_coef / (d + c_coef), rtol=1e-12)


def test_bilinear_matches_scipy():
    img = RNG.normal(size=(48, 64))
    uv = np.c_[RNG.uniform(0, 63, 200), RNG.uniform(0, 47, 200)]
    assert np.allclose(bilinear(img, uv), map_coordinates(img, [uv[:, 1], uv[:, 0]], order=1), atol=1e-12)


def test_plane_homography_equals_fitted_and_ray_plane():
    uv = table_view(300)
    P, _ = ray_plane(uv, K, R_WC, C)
    H = plane_homography(K, R_WC, C)
    assert np.allclose(apply_homography(H, uv), P[:, :2], atol=1e-12)
    assert np.allclose(homography_dlt(uv[:4], P[:4, :2]), H, rtol=1e-8, atol=1e-10)   # 4 points fix all 8 dof


def test_four_point_homography_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    src, dst = RNG.uniform(0, 600, (4, 2)), RNG.uniform(-0.2, 0.2, (4, 2))
    Hc, _ = cv2.findHomography(src, dst, 0)
    assert np.allclose(homography_dlt(src, dst), Hc / Hc[2, 2], rtol=1e-6, atol=1e-9)


def test_build_script_reproduces_its_output():
    pytest.importorskip("cv2")
    assert_build_output("20-pixels-to-table")
