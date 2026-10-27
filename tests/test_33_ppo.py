"""Lesson 33: GAE, Gaussian log-probability, the clipped surrogate, approximate KL and the horizon."""
import numpy as np
import pytest
from scipy.stats import norm

from conftest import assert_build_output
from physai.rl import approx_kl, clipped_surrogate, gae, gaussian_log_prob, horizon_seconds


def test_gae_matches_direct_sum_and_special_cases():
    rng = np.random.default_rng(0)
    T, N, g, lam = 20, 3, 0.9, 0.7
    R, V, L = rng.normal(size=(T, N)), rng.normal(size=(T, N)), rng.normal(size=N)
    adv, ret = gae(R, V, L, np.zeros((T, N)), g, lam)
    Vn = np.vstack([V[1:], L])
    delta = R + g * Vn - V
    direct = np.array([sum((g * lam) ** (k - t) * delta[k] for k in range(t, T)) for t in range(T)])
    assert np.allclose(adv, direct) and np.allclose(ret, adv + V)
    assert np.allclose(gae(R, V, L, np.zeros((T, N)), g, 0.0)[0], delta)
    mc = np.array([sum(g ** (k - t) * R[k] for k in range(t, T)) + g ** (T - t) * L for t in range(T)])
    assert np.allclose(gae(R, V, L, np.zeros((T, N)), g, 1.0)[0], mc - V)


def test_gae_does_not_leak_across_episode_ends():
    R, V, L = np.ones((6, 1)), np.zeros((6, 1)), np.array([100.0])
    dones = np.zeros((6, 1))
    dones[2] = 1
    adv, _ = gae(R, V, L, dones, 0.5, 1.0)
    assert np.allclose(adv[:3, 0], [1.75, 1.5, 1.0])             # first episode: 1 + 0.5 + 0.25 ...
    assert adv[3, 0] > 10                                        # second episode bootstraps from L = 100


def test_gaussian_log_prob_matches_scipy():
    rng = np.random.default_rng(1)
    mu, log_std = rng.normal(size=(50, 3)), rng.normal(size=3) * 0.3
    a = rng.normal(size=(50, 3))
    assert np.allclose(gaussian_log_prob(a, mu, log_std), norm.logpdf(a, mu, np.exp(log_std)).sum(-1))


def test_clipped_surrogate_values_and_gradients():
    torch = pytest.importorskip("torch")
    ratio = torch.tensor([0.7, 1.3, 0.7, 1.3], requires_grad=True)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss = clipped_surrogate(ratio, adv)
    loss.backward()
    assert loss.item() == pytest.approx(-(0.7 + 1.2 - 0.8 - 1.3) / 4)
    assert np.allclose(ratio.grad.numpy(), [-0.25, 0.0, 0.0, 0.25])  # no gain from moving past the clip


def test_kl_and_horizon():
    assert approx_kl(np.array([0.0, -1.0]), np.array([-0.5, -1.5])) == pytest.approx(0.5)
    assert horizon_seconds(0.02, 0.95) == pytest.approx(0.4) and horizon_seconds(0.02, 0.99) == pytest.approx(2.0)


def test_build_script_reproduces_its_output():
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    assert_build_output("33-ppo")
