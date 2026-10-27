"""Lesson 31: chunk targets and temporal ensembling (list formula vs running average)."""
import numpy as np

from conftest import assert_build_output
from physai.imitation import TemporalEnsembler, chunk_targets, ensembled_action


def test_chunk_targets_shift_and_pad():
    a = np.arange(10.0)[:, None]
    c = chunk_targets(a, 4)
    assert c.shape == (10, 4, 1)
    assert np.array_equal(c[2, :, 0], [2, 3, 4, 5]) and np.array_equal(c[8, :, 0], [8, 9, 9, 9])


def test_running_average_equals_list_formula():
    rng = np.random.default_rng(0)
    for k, m in ((1, 0.01), (5, 0.1), (16, -0.05)):
        preds = rng.normal(size=(40, k, 3))
        ens = TemporalEnsembler(k, m)
        for t in range(40):
            assert np.allclose(ens.update(preds[t]), ensembled_action(preds, t, k, m), atol=1e-12)


def test_m_zero_is_plain_mean_and_older_weighs_more():
    preds = np.random.default_rng(1).normal(size=(10, 4, 2))
    t = 7
    plain = np.mean([preds[s][t - s] for s in range(4, 8)], axis=0)
    assert np.allclose(ensembled_action(preds, t, 4, 0.0), plain)
    oldest, newest = np.zeros((10, 4, 1)), np.zeros((10, 4, 1))
    oldest[4, 3] = 1.0                                               # predicted for step 7 at step 4
    newest[7, 0] = 1.0                                               # predicted for step 7 at step 7
    assert ensembled_action(oldest, t, 4, 0.5)[0] > 0.25 > ensembled_action(newest, t, 4, 0.5)[0]


def test_build_script_reproduces_its_output():
    assert_build_output("31-act")
