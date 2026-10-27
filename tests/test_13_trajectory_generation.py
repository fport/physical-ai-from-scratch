"""Lesson 13: physai.trajectory time scalings against the closed-form formulas of the lesson."""
import numpy as np
import pytest

from conftest import assert_build_output
from physai.trajectory import CUBIC, QUINTIC, cubic, joint_reference, poly_eval, poly_from_boundary, quintic, trapezoid

U = np.linspace(0, 1, 2001)


def test_coefficients_are_the_textbook_polynomials():
    assert np.allclose(CUBIC, [0, 0, 3, -2], atol=1e-12)
    assert np.allclose(QUINTIC, [0, 0, 0, 10, -15, 6], atol=1e-12)


@pytest.mark.parametrize("shape", [cubic, quintic, trapezoid])
def test_shapes_start_and_end_at_rest(shape):
    s0, v0, _ = shape(0.0)
    s1, v1, _ = shape(1.0)
    assert (s0, s1) == pytest.approx((0.0, 1.0), abs=1e-12) and (v0, v1) == pytest.approx((0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize("shape", [cubic, quintic, trapezoid])
def test_derivatives_match_finite_differences(shape):
    s = np.array([shape(u) for u in U])
    du = U[1] - U[0]
    assert np.abs(np.gradient(s[:, 0], du) - s[:, 1])[1:-1].max() < 2e-3
    if shape is not trapezoid:                                 # the trapezoid's acceleration jumps at its corners
        assert np.abs(np.gradient(s[:, 1], du) - s[:, 2])[5:-5].max() < 0.05


def test_quintic_peaks_match_formulas():
    acc = poly_eval(QUINTIC, U, 2)
    assert acc.max() == pytest.approx(10 * np.sqrt(3) / 3, rel=1e-5)
    assert np.abs(poly_eval(QUINTIC, U, 3)).max() == pytest.approx(60)


def test_reference_scales_as_inverse_powers_of_duration():
    a = [abs(joint_reference(cubic, 1e-9, T, 0.0, 0.6)[2]) for T in (0.5, 1.0)]
    assert a[0] / a[1] == pytest.approx(4.0)                   # halve T, four times the acceleration


def test_boundary_solver_meets_every_condition():
    conds = [(0, 0, 0.0), (1, 0, 1.0), (0, 1, 0.5), (1, 1, 0.0), (0.5, 2, 0.0)]
    c = poly_from_boundary(conds, 4)
    assert all(poly_eval(c, u, r) == pytest.approx(v, abs=1e-12) for u, r, v in conds)


def test_build_script_reproduces_its_output():
    assert_build_output("13-trajectory-generation")
