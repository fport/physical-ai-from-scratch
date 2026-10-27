"""Lesson 17: physai.estimation's encoder model, filters, velocity and delay against SciPy and NumPy."""
import numpy as np
from scipy import ndimage
from scipy.signal import group_delay, lfilter, lfilter_zi

from conftest import assert_build_output
from physai.estimation import central_difference, delay_samples, encoder_reading, low_pass, moving_average

DT = 0.002


def test_moving_average_matches_scipy_after_startup():
    x = np.random.default_rng(0).normal(size=500)
    for n in (3, 5, 17):
        y = moving_average(x, n)
        assert np.allclose(y[n - 1:], lfilter(np.ones(n) / n, 1.0, x)[n - 1:], atol=1e-12)
        causal = ndimage.uniform_filter1d(x, n, origin=(n - 1) // 2)    # origin shifts the window to the past
        assert np.allclose(y[n - 1:], causal[n - 1:], atol=1e-12)


def test_low_pass_matches_lfilter_with_steady_start():
    x = np.random.default_rng(1).normal(size=1000) + 3.0
    for fc in (100, 10, 0.5):
        a = DT / (1 / (2 * np.pi * fc) + DT)
        b_, a_ = [a], [1.0, a - 1.0]
        ref, _ = lfilter(b_, a_, x, zi=lfilter_zi(b_, a_) * x[0])
        assert np.allclose(low_pass(x, fc, DT), ref, atol=1e-12)


def test_delays_match_group_delay_at_low_frequency():
    for n in (5, 17, 65):
        _, gd = group_delay((np.ones(n) / n, [1.0]), w=[0.2], fs=1 / DT)
        assert abs(gd[0] - (n - 1) / 2) < 1e-9
    for fc in (50, 10, 2):
        tau = 1 / (2 * np.pi * fc)
        a = DT / (tau + DT)
        _, gd = group_delay(([a], [1.0, a - 1.0]), w=[1e-4], fs=1 / DT)
        assert abs(gd[0] - tau / DT) < 1e-4 * tau / DT          # (1 - a) / a = tau / dt


def test_encoder_rounding_is_bounded_and_central_difference_is_np_gradient():
    q = np.random.default_rng(2).uniform(-3, 3, 10000)
    step = 2 * np.pi / 4096
    err = encoder_reading(q, 4096, 0.0, np.random.default_rng(0)) - q
    assert np.abs(err).max() <= step / 2 + 1e-12
    assert abs(err.std() - step / np.sqrt(12)) < 0.02 * step / np.sqrt(12)
    y = np.cumsum(np.random.default_rng(3).normal(size=300))
    assert np.allclose(central_difference(y, DT), np.gradient(y, DT))


def test_delay_samples_finds_a_known_shift():
    t = np.arange(5000) * DT
    ref = np.sin(2 * np.pi * 0.3 * t)
    for lag in (0, 7, 42):
        assert delay_samples(np.concatenate([np.zeros(lag), ref[:len(ref) - lag]]), ref) == lag


def test_build_script_reproduces_its_output():
    assert_build_output("17-sensors-noise-filtering")
