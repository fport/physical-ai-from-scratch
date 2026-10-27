"""Lesson 25: physai.evaluation against SciPy and the properties the lesson relies on."""
import numpy as np
from scipy import stats

from conftest import assert_build_output
from physai.evaluation import binom_pmf, coverage, fisher_p, power, trials_needed, wald, wilson

Z = stats.norm.ppf(0.975)


def test_wilson_matches_scipy():
    for n in (1, 7, 30, 300):
        for k in range(0, n + 1, max(1, n // 7)):
            ci = stats.binomtest(k, n).proportion_ci(method="wilson")
            assert np.allclose(wilson(k, n, Z), (ci.low, ci.high), atol=1e-12)


def test_pmf_and_fisher_match_scipy():
    assert np.allclose([binom_pmf(k, 30, 0.3) for k in range(31)], stats.binom.pmf(range(31), 30, 0.3), atol=1e-15)
    rng = np.random.default_rng(0)
    for _ in range(50):
        n1, n2 = rng.integers(5, 60, 2)
        k1, k2 = rng.integers(0, n1 + 1), rng.integers(0, n2 + 1)
        assert abs(fisher_p(k1, n1, k2, n2) - stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]) < 1e-12


def test_wald_collapses_and_wilson_does_not():
    assert wald(10, 10) == (1.0, 1.0) and wald(0, 10) == (0.0, 0.0)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo < 0.75


def test_wilson_coverage_is_near_nominal_and_wald_is_not():
    assert abs(coverage(wilson, 50, 0.737) - 0.95) < 0.02
    assert coverage(wald, 5, 0.737) < 0.8


def test_half_width_shrinks_like_one_over_sqrt_n():
    assert 3.5 < trials_needed(0.7, 0.05) / trials_needed(0.7, 0.10) < 4.5


def test_power_grows_with_trials():
    p = [power(0.74, 0.88, n) for n in (10, 30, 60)]
    assert p[0] < p[1] < p[2] and power(0.8, 0.8, 30) < 0.05


def test_build_script_reproduces_its_output():
    assert_build_output("25-evaluating-policies")
