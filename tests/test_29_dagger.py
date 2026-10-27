"""Lesson 29: the drift measure, the nearest-neighbour learner and the DAgger loop."""
import numpy as np

from conftest import assert_build_output
from physai.imitation import dagger, knn_policy, nearest_distance


def test_nearest_distance_is_min_of_max_abs():
    rng = np.random.default_rng(0)
    frames, pts = rng.normal(size=(300, 4)), rng.normal(size=(50, 4))
    brute = np.abs(pts[:, None] - frames[None]).max(-1).min(1)
    assert np.allclose(nearest_distance(pts, frames), brute, atol=0)
    assert np.all(nearest_distance(frames, frames) == 0)


def test_knn_policy_is_exact_on_its_data():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(100, 3)), rng.normal(size=100)
    assert np.array_equal(knn_policy(X, Y)(X), Y)


def test_dagger_aggregates_and_labels_with_the_expert():
    sizes = []

    def fit(X, Y):
        sizes.append(len(X))
        return knn_policy(X, Y)

    expert = lambda S: -S[:, 0]                                       # noqa: E731
    rollout = lambda pi, s0: np.array([s0 + i for i in range(5)], float)  # noqa: E731  visits 5 states
    X0 = np.zeros((3, 1))
    starts = [[np.array([1.0])], [np.array([2.0]), np.array([3.0])]]
    pols = dagger(fit, expert, rollout, X0, expert(X0), starts)
    assert len(pols) == 3 and sizes == [3, 8, 18]
    assert pols[-1](np.array([[6.0]]))[0] == -6.0                  # a learner state labelled by the expert
    sizes.clear()
    dagger(fit, expert, rollout, X0, expert(X0), starts, aggregate=False)
    assert sizes == [3, 5, 10]


def test_build_script_reproduces_its_output():
    assert_build_output("29-dagger")
