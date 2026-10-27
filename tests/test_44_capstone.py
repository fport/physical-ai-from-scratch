"""Lesson 44: the capstone's perception estimate, attribution and failure budget."""
import numpy as np
import pytest

from conftest import assert_build_output
from physai.capstone import attribute, estimate_xy, even_split, failure_budget, tilt
from physai.vision import intrinsics_from_fovy

K = intrinsics_from_fovy(50.0, 640, 480)
C = np.array([0.2, 0.0, 0.6])
R_DOWN = np.diag([1.0, -1.0, -1.0])                    # camera looking straight down, OpenCV axes


@pytest.mark.parametrize("seed", range(3))
def test_calibrated_estimate_is_within_a_pixel_footprint(seed):
    p = np.r_[np.random.default_rng(seed).uniform([0.1, -0.1], [0.3, 0.1]), 0.031]
    est = estimate_xy(p, K, R_DOWN, R_DOWN, C, 0.031)
    footprint = (C[2] - 0.031) / K[0, 0]                # metres per pixel on the plane
    assert np.linalg.norm(est - p[:2]) <= footprint * np.sqrt(2) / 2 + 1e-12


def test_unmodelled_tilt_shifts_the_estimate_by_about_h_alpha():
    p = np.array([0.2, 0.0, 0.031])                    # under the camera
    err = np.linalg.norm(estimate_xy(p, K, tilt(1.0) @ R_DOWN, R_DOWN, C, 0.031) - p[:2])
    assert abs(err - (C[2] - 0.031) * np.radians(1.0)) < 0.0015


def test_tilt_is_a_rotation_by_the_angle():
    R = tilt(2.0)
    assert np.allclose(R @ R.T, np.eye(3)) and np.degrees(np.arccos((np.trace(R) - 1) / 2)) == pytest.approx(2.0)


def test_attribution_takes_the_first_broken_layer():
    assert attribute({"place", "perception", "grasp"}) == "perception"
    assert attribute(set()) is None


def test_failure_budget_product_equals_end_to_end():
    firsts = [None] * 20 + ["IK/limits"] * 3 + ["grasp"] * 2 + ["language"]
    rows = failure_budget(firsts, len(firsts))
    assert np.prod([p / r for _, r, p in rows]) == pytest.approx(20 / 26)
    assert even_split(0.9, 5) ** 5 == pytest.approx(0.9)


def test_build_script_reproduces_its_output():
    assert_build_output("44-capstone")
