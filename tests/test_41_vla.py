"""Lesson 41: action tokens and the flow-matching sampler, plus properties the lesson relies on."""
import numpy as np
import pytest

from conftest import assert_build_output
from physai.foundation import detokenize, flow_velocity, sample_chunk, tokenize


@pytest.mark.parametrize("n_bins", [4, 37, 256])
def test_error_bounded_by_half_a_bin_and_rms_near_delta_over_sqrt12(n_bins):
    rng = np.random.default_rng(n_bins)
    lo, hi = np.array([-1.0, 0.0]), np.array([2.0, 0.5])
    a = rng.uniform(lo, hi, (50000, 2))
    err = detokenize(tokenize(a, lo, hi, n_bins), lo, hi, n_bins) - a
    delta = (hi - lo) / n_bins
    assert np.all(np.abs(err) <= delta / 2 + 1e-12)
    assert np.allclose(np.sqrt((err ** 2).mean(axis=0)), delta / np.sqrt(12), rtol=0.02)


def test_tokens_cover_all_bins_and_clip_outside_the_range():
    k = tokenize(np.array([-5.0, 0.0, 0.999, 1.0, 7.0]), 0.0, 1.0, 10)
    assert k.tolist() == [0, 0, 9, 9, 9]
    assert np.array_equal(tokenize(detokenize(np.arange(10), 0.0, 1.0, 10), 0.0, 1.0, 10), np.arange(10))


def test_flow_velocity_at_tau_zero_points_to_the_data_mean():
    rng = np.random.default_rng(1)
    data, x = rng.normal(size=(30, 5)), rng.normal(size=5)
    assert np.allclose(flow_velocity(x, 0.0, data), data.mean(axis=0) - x)


def test_one_training_chunk_is_reached_exactly():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(1, 8))
    out = sample_chunk(lambda x, t: flow_velocity(x, t, a), rng.normal(size=8), 3)
    assert np.allclose(out, a[0])


def test_more_flow_steps_land_closer_to_a_training_chunk():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(40, 20))
    z = rng.normal(size=20)
    dist = [np.sqrt(((sample_chunk(lambda x, t: flow_velocity(x, t, data), z, n) - data) ** 2).sum(1)).min()
            for n in (1, 20)]
    assert dist[1] < 0.1 * dist[0]


def test_build_script_reproduces_its_output():
    assert_build_output("41-vla")
