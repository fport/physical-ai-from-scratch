"""Lesson 30: flow-matching pair, the exact velocity field of a dataset, Euler sampling."""
import numpy as np

from conftest import assert_build_output
from physai.imitation import empirical_velocity, euler_sample, flow_matching_pair


def test_pair_endpoints_and_target():
    rng = np.random.default_rng(0)
    a0, a1 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    at0, v = flow_matching_pair(a0, a1, np.zeros(5))
    at1, _ = flow_matching_pair(a0, a1, np.ones(5))
    assert np.allclose(at0, a0) and np.allclose(at1, a1) and np.allclose(v, a1 - a0)


def test_one_euler_step_is_the_mean():
    rng = np.random.default_rng(1)
    data, a0 = rng.normal(size=(40, 6)), rng.normal(size=(10, 6))
    assert np.allclose(empirical_velocity(a0, 0.0, data), data.mean(0) - a0)
    assert np.allclose(euler_sample(lambda a, t: empirical_velocity(a, t, data), a0, 1), data.mean(0))


def test_single_demonstration_is_reached_exactly():
    x, a0 = np.array([[0.7, -1.2]]), np.random.default_rng(2).normal(size=(4, 2))
    for n in (1, 3, 10):
        assert np.allclose(euler_sample(lambda a, t: empirical_velocity(a, t, x), a0, n), x)


def test_many_steps_land_on_a_mode():
    data = np.array([[-1.1], [1.1]])
    a = euler_sample(lambda a, t: empirical_velocity(a, t, data), np.random.default_rng(3).normal(size=(200, 1)), 64)
    assert np.all(np.abs(np.abs(a) - 1.1) < 0.05) and 50 < np.sum(a > 0) < 150


def test_build_script_reproduces_its_output():
    assert_build_output("30-diffusion-and-flow-policies")
