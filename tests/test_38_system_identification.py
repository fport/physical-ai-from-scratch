"""Lesson 38: the hand-written Levenberg-Marquardt fit against NumPy and SciPy, plus properties the lesson relies on."""
import numpy as np
import pytest
from scipy.optimize import least_squares

from conftest import assert_build_output
from physai.sim2real import levenberg_marquardt, residual_jacobian


@pytest.mark.parametrize("seed", range(3))
def test_jacobian_of_a_linear_residual_is_its_matrix(seed):
    rng = np.random.default_rng(seed)
    A, b = rng.normal(size=(30, 4)), rng.normal(size=30)
    J = residual_jacobian(lambda x: A @ x - b, rng.normal(size=4))
    assert np.abs(J - A).max() < 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_linear_least_squares_matches_lstsq(seed):
    rng = np.random.default_rng(10 + seed)
    A, b = rng.normal(size=(50, 5)), rng.normal(size=50)
    x, r, _, _ = levenberg_marquardt(lambda x: A @ x - b, np.zeros(5), -np.inf, np.inf)
    assert np.abs(x - np.linalg.lstsq(A, b, rcond=None)[0]).max() < 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_nonlinear_fit_recovers_parameters_like_scipy(seed):
    """A damped oscillation y = a exp(-b t) sin(w t + p): the kind of black-box model sysid fits."""
    rng = np.random.default_rng(20 + seed)
    t = np.linspace(0, 3, 300)
    true = np.array([1.0, 0.8, 6.0, 0.3]) * rng.uniform(0.8, 1.2, 4)
    model = lambda x: x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t + x[3])  # noqa: E731
    y = model(true)
    lb, ub = np.array([0.1, 0.0, 3.0, -1.0]), np.array([3.0, 3.0, 9.0, 1.0])
    x0 = np.array([0.8, 0.5, 5.5, 0.0])
    x, _, _, _ = levenberg_marquardt(lambda x: model(x) - y, x0, lb, ub)
    s = least_squares(lambda x: model(x) - y, x0, bounds=(lb, ub))
    assert np.abs(x - true).max() < 1e-6 and np.abs(s.x - true).max() < 1e-6


def test_solution_stays_inside_the_bounds():
    x, _, _, _ = levenberg_marquardt(lambda x: x - 5.0, np.zeros(3), -1.0, 2.0)
    assert np.all(x <= 2.0) and np.allclose(x, 2.0)


def test_build_script_reproduces_its_output():
    assert_build_output("38-system-identification")
