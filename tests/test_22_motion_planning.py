"""Lesson 22: physai.planning on a 2-D toy world and a MuJoCo scene, plus properties the lesson relies on."""
import mujoco
import numpy as np
from scipy.spatial import cKDTree

from common.menagerie import path
from conftest import assert_build_output
from physai.planning import Tree, edge_free, path_length, rrt_connect, shortcut, steer, worst_along


def wall_free(q):
    """A 2-D world: a wall at x in [0.4, 0.6] with a gap at y in [0.45, 0.55]."""
    return not (0.4 <= q[0] <= 0.6 and not 0.45 <= q[1] <= 0.55)


def test_nearest_matches_kdtree():
    rng = np.random.default_rng(0)
    t = Tree(np.zeros(4))
    for _ in range(499):
        t.add(rng.normal(size=4), 0)
    for x in rng.normal(size=(100, 4)):
        assert t.nearest(x) == cKDTree(t.q[:t.n]).query(x)[1]


def test_steer_caps_the_step():
    a, b = np.zeros(3), np.array([3.0, 4.0, 0.0])
    assert np.isclose(np.linalg.norm(steer(a, b, 0.5)), 0.5)
    assert np.array_equal(steer(a, b, 10.0), b)


def test_edge_check_spacing_and_coarse_miss():
    calls = []
    edge_free(lambda q: calls.append(q) or True, np.zeros(2), np.array([1.0, 0.0]), 0.1)
    assert len(calls) == 10 and np.allclose(calls[-1], [1.0, 0.0])
    a, b = np.array([0.3, 0.2]), np.array([0.7, 0.2])            # crosses the wall below the gap
    assert not edge_free(wall_free, a, b, 0.01)
    assert edge_free(wall_free, a, b, 0.5)                         # one check at b: the wall is skipped


def test_rrt_connect_finds_a_free_path_through_the_gap():
    lo, hi = np.zeros(2), np.ones(2)
    for s in range(5):
        p, _ = rrt_connect(wall_free, lo, hi, np.array([0.1, 0.1]), np.array([0.9, 0.9]),
                           np.random.default_rng(s), eps=0.1, res=0.005, budget=5000)
        assert p is not None and np.allclose(p[0], [0.1, 0.1]) and np.allclose(p[-1], [0.9, 0.9])
        assert worst_along(lambda q: float(not wall_free(q)), p, 0.001) == 0.0
        short = shortcut(wall_free, p, 0.005, np.random.default_rng(s))
        assert path_length(short) <= path_length(p) + 1e-12
        assert worst_along(lambda q: float(not wall_free(q)), short, 0.001) == 0.0


def test_mujoco_collision_oracle_sees_the_box():
    spec = mujoco.MjSpec.from_file(path("franka_emika_panda"))
    g = spec.worldbody.add_geom()
    g.type, g.size, g.pos = mujoco.mjtGeom.mjGEOM_BOX, [0.1, 0.1, 0.25], [0.55, 0.0, 0.25]
    m = spec.compile()
    d = mujoco.MjData(m)

    def depth(q):
        d.qpos[:7], d.qpos[7:] = q, 0.04
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        return 0.0 if d.ncon == 0 else -min(d.contact[i].dist for i in range(d.ncon))

    qs, qg = np.array([0.8, 0.6, 0, -1.8, 0, 2.4, 0.785]), np.array([-0.8, 0.6, 0, -1.8, 0, 2.4, 0.785])
    assert depth(qs) == 0.0 and depth(qg) == 0.0
    assert abs(worst_along(depth, [qs, qg], 0.005) - 0.139) < 5e-4


def test_build_script_reproduces_its_output():
    assert_build_output("22-motion-planning")
