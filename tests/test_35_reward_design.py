"""Lesson 35: reward terms, the telescoping of potential-based shaping, and policy invariance on random MDPs."""
import numpy as np
import pytest

from conftest import assert_build_output
from physai.rl import (action_rate_penalty, closeness_reward, discounted_sum, potential_shaping, sparse_reward,
                       value_iteration)


def test_reward_terms():
    assert sparse_reward(0.019) == 1.0 and sparse_reward(0.02) == 0.0
    assert closeness_reward(0.0) == 1.0 and 0 < closeness_reward(0.05) < 0.25
    assert action_rate_penalty([1.0, 0.0], [0.0, 0.0], 0.3) == pytest.approx(-0.3)


def test_shaping_telescopes():
    rng = np.random.default_rng(0)
    phi, g = rng.normal(size=31), 0.93
    F = [potential_shaping(phi[t + 1], phi[t], g) for t in range(30)]
    assert discounted_sum(F, g) == pytest.approx(g ** 30 * phi[30] - phi[0])


def test_potential_shaping_keeps_the_optimal_policy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        nS, nA, g = 12, 3, 0.9
        nxt, R = rng.integers(0, nS, (nS, nA)), rng.normal(size=(nS, nA))
        term = np.zeros(nS, bool)
        term[0] = True
        phi = rng.normal(size=nS) * 3
        phi[term] = 0.0                                          # the potential of a terminal state is 0
        V, pi = value_iteration(nxt, R, term, g)
        V2, pi2 = value_iteration(nxt, R + potential_shaping(phi[nxt], phi[:, None], g), term, g)
        assert np.array_equal(pi[~term], pi2[~term]) and np.allclose(V2[~term], V[~term] - phi[~term])


def test_build_script_reproduces_its_output():
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")                                 # build imports the Experiment, which imports ppo.py
    assert_build_output("35-reward-design")
