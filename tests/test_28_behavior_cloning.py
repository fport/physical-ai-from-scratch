"""Lesson 28: physai.imitation's behavior cloning pieces against NumPy, torch and lesson 27's trainer."""
import importlib.util
import os

import numpy as np
import pytest

from conftest import ROOT, assert_build_output
from physai.imitation import bc_policy, fit_mean_std, mlp_forward, train_bc


def test_mean_std_matches_numpy():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(500, 7))
    mean, std = fit_mean_std(x)
    assert np.allclose(mean, x.mean(0), atol=1e-5) and np.allclose(std, x.std(0) + 1e-6, atol=1e-5)


def test_mlp_forward_is_relu_network():
    rng = np.random.default_rng(1)
    layers = [(rng.normal(size=(8, 3)), rng.normal(size=8)), (rng.normal(size=(2, 8)), rng.normal(size=2))]
    x = rng.normal(size=(20, 3))
    want = np.maximum(x @ layers[0][0].T + layers[0][1], 0) @ layers[1][0].T + layers[1][1]
    assert np.allclose(mlp_forward(layers, x), want, rtol=1e-12)


def test_policy_unnormalizes():
    # identity network in normalized units -> the policy maps a normalized input back onto the action scale
    layers = [(np.eye(2), np.zeros(2)), (np.eye(2), np.zeros(2))]
    pi = bc_policy(layers, (np.array([1.0, 2.0]), np.array([2.0, 4.0])), (np.array([10.0, 20.0]), np.array([3.0, 5.0])))
    assert np.allclose(pi(np.array([3.0, 6.0])), [[13.0, 25.0]])     # (3-1)/2 = 1 -> 10 + 3; (6-2)/4 = 1 -> 20 + 5


def test_train_bc_matches_lesson_27_trainer():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("l27_for_test", os.path.join(ROOT, "experiments", "27-robot-datasets", "experiment.py"))
    l27 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(l27)
    rng = np.random.default_rng(2)
    X, Y = rng.normal(size=(300, 5)), rng.normal(size=(300, 2))
    noise = np.full(5, 0.05)
    mine, xs, ys = train_bc(X, Y, hidden=32, steps=50, batch=64, seed=3, obs_noise=noise)
    ref = l27.train_mlp(X, Y, hidden=32, steps=50, batch=64, seed=3, obs_noise=noise)
    assert all(np.array_equal(w1, w2) and np.array_equal(b1, b2) for (w1, b1), (w2, b2) in zip(mine, ref.layers))
    assert np.allclose(bc_policy(mine, xs, ys)(X), ref(X))


def test_build_script_reproduces_its_output():
    pytest.importorskip("torch")
    assert_build_output("28-behavior-cloning")
