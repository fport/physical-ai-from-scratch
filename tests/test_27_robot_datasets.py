"""Lesson 27: physai.data (episode bookkeeping, statistics, splits, normalization) and the split leak."""
import os

import numpy as np

from conftest import ROOT, assert_build_output
from physai.data import (episode_starts, feature_stats, global_index, merge_stats, nearest_neighbour, normalize,
                         split_episodes, split_frames, unnormalize)


def toy(rng, lengths):
    ep = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)])
    fr = np.concatenate([np.arange(n) for n in lengths])
    return ep, fr, rng.normal(size=(len(ep), 3)) * [1, 10, 0.01] + [0, 5, 1]


def test_starts_and_index_with_unequal_episodes():
    ep, fr, _ = toy(np.random.default_rng(0), [3, 5, 1, 4])
    assert episode_starts(ep).tolist() == [0, 3, 8, 9]
    assert np.array_equal(global_index(ep, fr), np.arange(len(ep)))


def test_merged_stats_equal_global_stats():
    ep, _, x = toy(np.random.default_rng(1), [30, 7, 50, 12])
    s = feature_stats(x[ep == 0])
    for e in range(1, 4):
        s = merge_stats(s, feature_stats(x[ep == e]))
    g = feature_stats(x)
    for k in ("mean", "std", "min", "max"):
        assert np.allclose(s[k], g[k], atol=1e-12)
    assert s["count"] == g["count"]


def test_normalization_round_trip_and_unit_invariance():
    _, _, x = toy(np.random.default_rng(2), [40, 40])
    for mode in ("mean_std", "min_max"):
        s = feature_stats(x)
        assert np.allclose(unnormalize(normalize(x, s, mode), s, mode), x, atol=1e-12)
        y = x * [1, 1, 1000.0]
        assert np.allclose(normalize(x, s, mode), normalize(y, feature_stats(y), mode), atol=1e-5)
    z = normalize(x, feature_stats(x), "min_max")
    assert np.isclose(z.min(), -1) and np.isclose(z.max(), 1)


def test_episode_split_never_shares_an_episode_but_frame_split_does():
    ep, _, _ = toy(np.random.default_rng(3), [50] * 10)
    tr, va = split_episodes(ep, [2, 7])
    assert not set(ep[tr]) & set(ep[va]) and set(ep[va]) == {2, 7}
    f_tr, f_va = split_frames(np.arange(len(ep)), 0.2, np.random.default_rng(0))
    assert len(f_va) == 100 and len(set(ep[f_tr]) & set(ep[f_va])) == 10


def test_nearest_neighbour_is_exact_on_training_rows():
    x = np.random.default_rng(4).normal(size=(300, 5))
    y = np.arange(300.0)[:, None]
    assert np.array_equal(nearest_neighbour(x, y, x, chunk=64), y)


def test_frame_split_leaks_on_the_lesson_26_demos():
    D = np.load(os.path.join(ROOT, "experiments", "26-teleop-replay-data", "demos.npz"))
    O, A, EP = D["obs"].astype(float), D["action"].astype(float), D["episode_index"].astype(int)
    sel = np.flatnonzero(EP < 10)
    f_tr, f_va = split_frames(sel, 0.2, np.random.default_rng(0))
    e_tr, e_va = split_episodes(EP, [8, 9])
    e_tr &= EP < 10
    err = []
    for a, b in ((f_tr, f_va), (e_tr, e_va)):
        s = feature_stats(O[a])
        p = nearest_neighbour(normalize(O[a], s), A[a], normalize(O[b], s))
        err.append(np.sqrt(np.mean((p - A[b]) ** 2)))
    assert err[0] < 0.5 * err[1]


def test_build_script_reproduces_its_output():
    assert_build_output("27-robot-datasets")
