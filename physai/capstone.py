"""The capstone pipeline's contracts, composed from earlier modules (lesson 44 · Build it).

Perception uses physai.vision (lessons 19-21) and physai.spatial (lesson 06); reach uses physai.foundation's
validation layer (lesson 42) on physai.kinematics (lessons 09-10); the harness uses physai.evaluation's Wilson
interval (lesson 25). This module adds only what the capstone itself needs: the camera's estimate under an
unmodelled tilt, first-violation attribution and the failure budget.
"""
import numpy as np

from physai.spatial import axis_angle_to_mat
from physai.vision import project_points, ray_plane

ORDER = ("language", "perception", "IK/limits", "grasp", "place")


def tilt(deg, axis=(1.0, 1.0, 0.0)):
    """World-frame rotation of a camera that has tilted by `deg` about a horizontal axis."""
    a = np.asarray(axis, float)
    return axis_angle_to_mat(a / np.linalg.norm(a), np.radians(deg))


def estimate_xy(p_top, K, R_wc_true, R_wc_model, c, z_top):
    """Where perception puts a cube: the true camera images the top-face centre p_top at the nearest pixel, and
    the calibrated (model) camera casts that pixel's ray back onto the plane z = z_top."""
    uv = np.round(project_points(p_top[None], R_wc_true.T, -R_wc_true.T @ c, K))   # whole pixels
    return ray_plane(uv, K, R_wc_model, c, z_top)[0][0, :2]


def attribute(broken, order=ORDER):
    """First layer, in pipeline order, whose contract broke; None if all held. broken: set of layer names."""
    return next((layer for layer in order if layer in broken), None)


def failure_budget(first_failures, n, order=ORDER):
    """Episodes that reach each layer and pass it, from the first broken layer of every failed episode.
    Returns [(layer, reached, passed)]; the product of passed / reached is the end-to-end rate."""
    rows, reached = [], n
    for layer in order:
        failed = sum(f == layer for f in first_failures)
        rows.append((layer, reached, reached - failed))
        reached -= failed
    return rows


def even_split(P, L):
    """Pass rate every one of L layers in series needs for an end-to-end rate P."""
    return P ** (1 / L)
