"""Lesson 21: physai.vision colour detection against SciPy and OpenCV, plus properties the lesson relies on."""
import numpy as np
import pytest
from scipy import ndimage

from conftest import assert_build_output
from physai.vision import (color_mask, detect_color, label_blobs, largest_blob, mask_centroid, mask_to_table_xy,
                           open3x3, rgb_to_hsv8)

RED = [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))]
EIGHT = np.ones((3, 3), bool)


def random_mask(seed, shape=(60, 80), p=0.45):
    return ndimage.binary_opening(np.random.default_rng(seed).random(shape) < p) | \
        (np.random.default_rng(seed + 100).random(shape) < 0.02)


def test_hsv_known_colours_and_brightness_invariance():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [0, 0, 0]]], np.uint8)
    assert rgb_to_hsv8(rgb)[0].tolist() == [[0, 255, 255], [60, 255, 255], [120, 255, 255], [0, 0, 128], [0, 0, 0]]
    bright, dim = rgb_to_hsv8(np.array([[[200, 20, 20]]], np.uint8)), rgb_to_hsv8(np.array([[[100, 10, 10]]], np.uint8))
    assert bright[0, 0, 0] == dim[0, 0, 0] and bright[0, 0, 1] == dim[0, 0, 1] and bright[0, 0, 2] == 2 * dim[0, 0, 2]
    x = np.random.default_rng(0).integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    hsv = rgb_to_hsv8(x)
    assert np.array_equal(hsv[..., 2], x.max(-1)) and hsv[..., 0].max() <= 179


def test_hsv_matches_opencv_to_one_unit():
    cv2 = pytest.importorskip("cv2")
    x = np.random.default_rng(1).integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
    ours, ref = rgb_to_hsv8(x).astype(int), cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(int)
    dh = np.abs(ours[..., 0] - ref[..., 0])
    assert np.minimum(dh, 180 - dh).max() <= 1 and np.abs(ours[..., 1:] - ref[..., 1:]).max() <= 1
    assert np.array_equal(color_mask(ref.astype(np.uint8), RED),
                          sum(cv2.inRange(ref.astype(np.uint8), lo, hi) for lo, hi in RED) > 0)


def test_red_needs_two_hue_boxes_and_bounds_are_inclusive():
    hsv = np.array([[[0, 200, 200], [10, 120, 70], [11, 200, 200], [90, 200, 200], [170, 200, 200], [179, 119, 200]]],
                   np.uint8)
    assert color_mask(hsv, RED)[0].tolist() == [True, True, False, False, True, False]


def test_opening_matches_scipy_with_opencv_borders():
    for seed in range(5):
        m = random_mask(seed)
        ref = ndimage.binary_dilation(ndimage.binary_erosion(m, EIGHT, border_value=1), EIGHT, border_value=0)
        assert np.array_equal(open3x3(m), ref)


def test_blobs_match_scipy_label():
    for seed in range(5):
        m = random_mask(seed)
        ref, n = ndimage.label(m, structure=EIGHT)
        lab = label_blobs(m)
        pairs = {(a, b) for a, b in zip(lab[m], ref[m])}
        assert len(np.unique(lab[m])) == n and len(pairs) == n          # same partition into blobs
        areas = np.bincount(ref.ravel())[1:]
        big = largest_blob(m, min_area=1)
        assert big.sum() == areas.max() and np.all(ref[big] == ref[big][0])
    assert largest_blob(np.zeros((5, 5), bool)) is None
    one = np.zeros((20, 20), bool)
    one[2:6, 2:6] = True
    assert largest_blob(one, min_area=30) is None and largest_blob(one, min_area=16).sum() == 16


def test_centroid_matches_center_of_mass():
    m = random_mask(7)
    r, c = ndimage.center_of_mass(m)
    assert np.allclose(mask_centroid(m), [c, r])


def test_back_projection_of_a_flat_face_under_the_camera():
    f, w, h = 514.7, 640, 480
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    R_wc, c = np.diag([1.0, -1.0, -1.0]), np.array([0.20, 0.0, 0.60])    # looking straight down
    depth = np.full((h, w), 0.57)                                         # a face 3 cm above the table
    mask = np.zeros((h, w), bool)
    mask[230:250, 310:330] = True                                         # centred on the principal point
    assert np.allclose(mask_to_table_xy(mask, depth, K, R_wc, c), [0.20, 0.0], atol=1e-12)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[mask] = [217, 26, 26]
    assert np.array_equal(detect_color(rgb, RED), mask)


def test_build_script_reproduces_its_output():
    pytest.importorskip("cv2")
    assert_build_output("21-object-perception")
