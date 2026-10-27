"""Lesson 39: the domain-randomization sampler and CEM against SciPy and known answers."""
import numpy as np
import pytest
from scipy import stats

from conftest import assert_build_output
from physai.sim2real import cem, dr_sample, expected_cost, inside_support


@pytest.mark.parametrize("w", [0.25, 0.75, 1.5])
def test_scales_are_log_uniform_and_delay_uniform(w):
    p = dr_sample(np.random.default_rng(1), 20000, w)
    for k in ("mass", "damping", "friction"):
        assert stats.kstest(p[k], stats.loguniform(2 ** -w, 2 ** w).cdf).pvalue > 1e-3
    assert stats.kstest(p["delay"], stats.uniform(0.005, 0.040 * w).cdf).pvalue > 1e-3


def test_width_zero_is_nominal_and_draws_nothing():
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    p = dr_sample(rng, 5, 0.0)
    assert rng.bit_generator.state == state
    assert np.all(p["mass"] == 1) and np.all(p["delay"] == 0.005)


def test_every_sample_is_inside_its_own_support_and_wider_supports_nest():
    p = dr_sample(np.random.default_rng(2), 1000, 0.6)
    assert all(v.all() for v in inside_support(p, 0.6).values())
    narrow, wide = inside_support(p, 0.3), inside_support(p, 0.9)
    assert all((wide[k] | ~narrow[k]).all() for k in p)


def test_expected_cost_averages_each_controller_over_the_same_plants():
    plants = {"mass": np.array([1.0, 2.0, 3.0]), "damping": np.ones(3), "friction": np.ones(3), "delay": np.zeros(3)}
    rollout = lambda g, pl: g[:, 0] * pl["mass"]  # noqa: E731
    assert np.allclose(expected_cost(rollout, np.array([[1.0], [10.0]]), plants), [2.0, 20.0])


@pytest.mark.parametrize("seed", range(3))
def test_cem_finds_the_minimum_of_a_quadratic(seed):
    rng = np.random.default_rng(seed)
    target = rng.uniform(-2, 2, 4)
    mu = cem(lambda x, r: ((x - target) ** 2).sum(axis=1), np.zeros(4), np.full(4, 2.0), -5.0, 5.0, rng, iters=30)
    assert np.abs(mu - target).max() < 0.05


def test_build_script_reproduces_its_output():
    assert_build_output("39-domain-randomization")
