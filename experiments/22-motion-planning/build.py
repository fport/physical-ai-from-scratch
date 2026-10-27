"""Lesson 22 · Build it: RRT-Connect in joint space from scratch, against the Experiment's numbers.

Step 1  The collision oracle and the edge check: the straight line start -> goal.
Step 2  Steer and a tree with nearest-neighbour search vs scipy's cKDTree.
Step 3  RRT-Connect on the box scene, 100 seeds at res 0.05 rad.
Step 4  Shortcutting the raw paths.
Step 5  Dense re-check at 0.005 rad: how many "successes" go through the box.

The functions live in physai/planning.py (the lesson shows them step by step).
Run:  uv run python experiments/22-motion-planning/build.py      (~3 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.planning import Tree, edge_free, path_length, rrt_connect, shortcut, steer, worst_along  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

# The scene of the Experiment: Panda + a 20 x 20 x 50 cm box at x = 0.55 m.
spec = mujoco.MjSpec.from_file(path("franka_emika_panda"))
g = spec.worldbody.add_geom()
g.name, g.type, g.size, g.pos = "box", mujoco.mjtGeom.mjGEOM_BOX, [0.1, 0.1, 0.25], [0.55, 0.0, 0.25]
m = spec.compile()
d = mujoco.MjData(m)
LO, HI = m.jnt_range[:7, 0].copy(), m.jnt_range[:7, 1].copy()
Q_START = np.array([0.8, 0.6, 0.0, -1.8, 0.0, 2.4, 0.785])
Q_GOAL = np.array([-0.8, 0.6, 0.0, -1.8, 0.0, 2.4, 0.785])
calls = [0]


def depth(q):
    """Deepest penetration (m) at pose q, 0 if collision-free. The only place MuJoCo is asked."""
    calls[0] += 1
    d.qpos[:7], d.qpos[7:] = q, 0.04             # fingers open, fixed
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    return 0.0 if d.ncon == 0 else -min(d.contact[i].dist for i in range(d.ncon))


def free(q):
    return depth(q) == 0.0


print("\n== Step 1. The edge check on the straight line start -> goal ==")
print(f"start free: {free(Q_START)} | goal free: {free(Q_GOAL)} | "
      f"length {np.linalg.norm(Q_GOAL - Q_START):.2f} rad")
print(f"{'res (rad)':>9} {'poses on edge':>14} {'tested until answer':>20} {'edge_free says':>15}")
for res in (0.005, 0.05, 0.25, 0.5, 1.0):
    calls[0] = 0
    ok = edge_free(free, Q_START, Q_GOAL, res)
    n = int(np.ceil(np.linalg.norm(Q_GOAL - Q_START) / res))
    print(f"{res:9.3f} {n:14d} {calls[0]:20d} {str(ok):>15}")
print(f"dense truth: worst penetration on the line = {1000 * worst_along(depth, [Q_START, Q_GOAL], 0.005):.1f} mm")

print("\n== Step 2. Steer and nearest neighbour vs scipy cKDTree (1000 random nodes, 200 queries) ==")
rng = np.random.default_rng(0)
tree = Tree(Q_START)
for _ in range(999):
    tree.add(rng.uniform(LO, HI), 0)
queries = rng.uniform(LO, HI, size=(200, 7))
same = sum(tree.nearest(x) == cKDTree(tree.q[:tree.n]).query(x)[1] for x in queries)
y = steer(Q_START, Q_GOAL, 0.5)
print(f"nearest index identical in {same}/200 queries")
print(f"steer 0.5 rad toward the goal: step length {np.linalg.norm(y - Q_START):.4f} rad; "
      f"with eps = 2 it returns the goal itself: {np.array_equal(steer(Q_START, Q_GOAL, 2.0), Q_GOAL)}")

print("\n== Step 3. RRT-Connect, res 0.05 rad, 100 seeds, budget 2000 iterations ==")
runs = []
for s in range(100):
    rng = np.random.default_rng(s)
    calls[0] = 0
    p, iters = rrt_connect(free, LO, HI, Q_START, Q_GOAL, rng, eps=0.5, res=0.05, budget=2000)
    runs.append(dict(path=p, iters=iters, checks=calls[0], rng=rng))
ok = [r for r in runs if r["path"] is not None]
print(f"success {len(ok)}/100 | median iterations {np.median([r['iters'] for r in ok]):.0f} | "
      f"median collision checks {np.median([r['checks'] for r in ok]):.0f}")

print("\n== Step 4. Shortcutting (200 tries, same res) ==")
for r in ok:
    r["raw"] = path_length(r["path"])
    r["path"] = shortcut(free, r["path"], 0.05, r["rng"], tries=200)
    r["short"] = path_length(r["path"])
print(f"median length raw {np.median([r['raw'] for r in ok]):.2f} rad -> shortcut "
      f"{np.median([r['short'] for r in ok]):.2f} rad | median waypoints after: "
      f"{np.median([len(r['path']) for r in ok]):.0f}")

print("\n== Step 5. Dense re-check of every returned path at 0.005 rad ==")
worst = np.array([worst_along(depth, r["path"], 0.005) for r in ok])
print(f"paths in collision: {int((worst > 0).sum())}/{len(ok)} | worst penetration {1000 * worst.max():.1f} mm")
