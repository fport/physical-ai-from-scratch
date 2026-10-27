"""Lesson 22 · Configuration space and sampling-based planning.

Franka Panda (MuJoCo Menagerie) next to a box. We plan in the 7-D joint space (C-space).
A configuration q is "in collision" if MuJoCo reports any contact after mj_kinematics +
mj_collision (exactly the contacts mj_forward would produce, without dynamics).

Part A  Sweep the edge-check resolution of RRT-Connect (one extend step = 0.5 rad).
        For each value: success rate over seeds, iterations, collision checks, path length,
        and how many returned paths actually hit the box when re-checked densely.
Part B  RRT vs RRT-Connect, and path shortcutting, at a safe resolution.
Break   Narrow passage: the arm must thread its forearm through a slot in a wall.
        Sweep the slot height -> success rate within a fixed iteration budget.

Run:  python experiment.py      (deterministic, fixed seeds)
"""
import os
import platform
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | Python {platform.python_version()} "
      f"| {platform.machine()}")

STEP = 0.5          # rad: max length of one tree edge (extend step), L2 norm in joint space
DENSE = 0.005       # rad: resolution used to re-check a finished path ("ground truth")


# ---------------------------------------------------------------- scene + collision
def build(boxes):
    """Panda scene with extra static boxes [(name, half_size, pos), ...]."""
    spec = mujoco.MjSpec.from_file(path("franka_emika_panda"))
    for name, size, pos in boxes:
        g = spec.worldbody.add_geom()
        g.name, g.type, g.size, g.pos = name, mujoco.mjtGeom.mjGEOM_BOX, size, pos
    return spec.compile()


class Checker:
    """Collision checking in C-space. Counts every call so we can report the cost."""

    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        self.lo, self.hi = model.jnt_range[:7, 0].copy(), model.jnt_range[:7, 1].copy()
        self.calls = 0

    def depth(self, q):
        """Deepest penetration (m, >0) at configuration q; 0 if collision-free."""
        self.calls += 1
        d = self.d
        d.qpos[:7] = q
        d.qpos[7:] = 0.04                       # fingers open, fixed
        mujoco.mj_kinematics(self.m, d)         # link poses
        mujoco.mj_collision(self.m, d)          # broad + narrow phase -> d.contact
        if d.ncon == 0:
            return 0.0
        return -min(d.contact[i].dist for i in range(d.ncon))

    def free(self, q):
        return self.depth(q) == 0.0

    def edge_free(self, a, b, res):
        """Check points along the straight joint-space segment a->b, spaced <= res."""
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / res)))
        for k in range(1, n + 1):
            if not self.free(a + (b - a) * (k / n)):
                return False
        return True


def path_length(p):
    return float(sum(np.linalg.norm(p[i + 1] - p[i]) for i in range(len(p) - 1)))


def dense_depth(chk, p):
    """Ground truth: max penetration along the whole path at DENSE resolution."""
    worst = 0.0
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / DENSE)))
        for k in range(n + 1):
            worst = max(worst, chk.depth(a + (b - a) * (k / n)))
    return worst


# ---------------------------------------------------------------- planners
class Tree:
    def __init__(self, root):
        self.q = np.zeros((20000 * 4, 7))
        self.parent = np.zeros(20000 * 4, dtype=int)
        self.q[0], self.parent[0], self.n = root, -1, 1

    def nearest(self, x):
        return int(np.argmin(np.sum((self.q[:self.n] - x) ** 2, axis=1)))

    def add(self, x, parent):
        self.q[self.n], self.parent[self.n] = x, parent
        self.n += 1
        return self.n - 1

    def branch(self, i):
        out = []
        while i != -1:
            out.append(self.q[i].copy())
            i = self.parent[i]
        return out          # node i ... root


def steer(a, b):
    v = b - a
    dist = np.linalg.norm(v)
    return b.copy() if dist <= STEP else a + v * (STEP / dist)


def rrt(chk, qs, qg, res, rng, max_iter, goal_bias=0.05):
    """Plain RRT with goal biasing (LaValle 1998)."""
    t = Tree(qs)
    for it in range(1, max_iter + 1):
        x = qg if rng.random() < goal_bias else rng.uniform(chk.lo, chk.hi)
        i = t.nearest(x)
        y = steer(t.q[i], x)
        if not chk.edge_free(t.q[i], y, res):
            continue
        j = t.add(y, i)
        if np.linalg.norm(y - qg) <= STEP and chk.edge_free(y, qg, res):
            return t.branch(j)[::-1] + [qg.copy()], it
    return None, max_iter


def rrt_connect(chk, qs, qg, res, rng, max_iter):
    """RRT-Connect (Kuffner & LaValle 2000): grow two trees, greedily connect them."""
    ta, tb = Tree(qs), Tree(qg)
    a_is_start = True
    for it in range(1, max_iter + 1):
        x = rng.uniform(chk.lo, chk.hi)
        i = ta.nearest(x)
        y = steer(ta.q[i], x)
        if chk.edge_free(ta.q[i], y, res):
            ja = ta.add(y, i)
            # CONNECT: step tree b toward y until it reaches y or hits an obstacle
            jb = tb.nearest(y)
            while True:
                z = steer(tb.q[jb], y)
                if not chk.edge_free(tb.q[jb], z, res):
                    break
                jb = tb.add(z, jb)
                if np.allclose(z, y):
                    pa, pb = ta.branch(ja)[::-1], tb.branch(jb)[1:]
                    p = pa + pb
                    return (p if a_is_start else p[::-1]), it
        ta, tb, a_is_start = tb, ta, not a_is_start
    return None, max_iter


def shortcut(chk, p, res, rng, tries=200):
    """Random shortcutting: replace a sub-path by a straight segment if it is collision-free."""
    p = [q.copy() for q in p]
    for _ in range(tries):
        if len(p) < 3:
            break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i < 2:
            continue
        if chk.edge_free(p[i], p[j], res):
            p = p[:i + 1] + p[j:]
    return p


def run(chk, planner, qs, qg, res, seeds, max_iter, do_shortcut=False):
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        chk.calls = 0
        p, iters = planner(chk, qs, qg, res, rng, max_iter)
        calls = chk.calls
        if p is None:
            rows.append(dict(ok=False, iters=iters, calls=calls))
            continue
        L_raw = path_length(p)
        if do_shortcut:
            p = shortcut(chk, p, res, rng)
        rows.append(dict(ok=True, iters=iters, calls=calls, L_raw=L_raw, L=path_length(p),
                         depth=dense_depth(chk, p)))
    return rows


def summarize(rows):
    ok = [r for r in rows if r["ok"]]
    hit = [r for r in ok if r["depth"] > 0]
    return dict(n=len(rows), succ=len(ok),
                iters=np.median([r["iters"] for r in ok]) if ok else float("nan"),
                calls=np.median([r["calls"] for r in ok]) if ok else float("nan"),
                L_raw=np.median([r["L_raw"] for r in ok]) if ok else float("nan"),
                L=np.median([r["L"] for r in ok]) if ok else float("nan"),
                hit=len(hit),
                depth=1000 * max([r["depth"] for r in hit]) if hit else 0.0)


# ---------------------------------------------------------------- Part A / B: box obstacle
BOX = [("box", [0.1, 0.1, 0.25], [0.55, 0.0, 0.25])]          # 20 x 20 x 50 cm box
q_start = np.array([0.8, 0.6, 0.0, -1.8, 0.0, 2.4, 0.785])    # hand left of the box
q_goal = np.array([-0.8, 0.6, 0.0, -1.8, 0.0, 2.4, 0.785])    # hand right of the box

chk = Checker(build(BOX))
print(f"\nScene A: Panda + box 0.20 x 0.20 x 0.50 m at x = 0.55 m. Extend step {STEP} rad, "
      f"dense re-check {DENSE} rad.")
print(f"start free: {chk.free(q_start)} | goal free: {chk.free(q_goal)} | "
      f"straight line free (dense): {dense_depth(chk, [q_start, q_goal]) == 0.0} | "
      f"straight-line max penetration {1000 * dense_depth(chk, [q_start, q_goal]):.1f} mm | "
      f"straight length {np.linalg.norm(q_goal - q_start):.2f} rad")

SEEDS = range(100)
print("\nPart A · RRT-Connect, sweep edge-check resolution (100 seeds, budget 2000 iterations, "
      "200 shortcut tries)")
print(f"{'res (rad)':>9} | {'success':>7} | {'med iters':>9} | {'med checks':>10} | "
      f"{'med length raw':>14} | {'med length short':>16} | {'paths in collision':>18} | {'worst pen. (mm)':>15}")
for res in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
    s = summarize(run(chk, rrt_connect, q_start, q_goal, res, SEEDS, 2000, do_shortcut=True))
    print(f"{res:9.2f} | {s['succ']:3d}/{s['n']:<3d} | {s['iters']:9.0f} | {s['calls']:10.0f} | "
          f"{s['L_raw']:14.2f} | {s['L']:16.2f} | {s['hit']:8d}/{s['succ']:<9d} | {s['depth']:15.1f}")

print("\nPart B · RRT vs RRT-Connect at res 0.05 rad (100 seeds, budget 2000 iterations), "
      "then shortcutting")
print(f"{'planner':>11} | {'success':>7} | {'med iters':>9} | {'med checks':>10} | "
      f"{'med length raw':>14} | {'med length short':>16} | {'paths in collision':>18}")
for name, planner in [("RRT", rrt), ("RRT-Connect", rrt_connect)]:
    s = summarize(run(chk, planner, q_start, q_goal, 0.05, SEEDS, 2000, do_shortcut=True))
    print(f"{name:>11} | {s['succ']:3d}/{s['n']:<3d} | {s['iters']:9.0f} | {s['calls']:10.0f} | "
          f"{s['L_raw']:14.2f} | {s['L']:16.2f} | {s['hit']:8d}/{s['succ']:<9d}")

# ---------------------------------------------------------------- Break it: narrow passage
WX, ZC = 0.40, 0.45                      # wall plane x, slot centre height
q_s2 = np.array([0.52, -0.02, 0.26, -2.65, 0.01, 2.63, 1.54])  # hand in front of the wall
q_g2 = np.array([0.0, 0.85, 0.0, -1.63, 0.0, 2.48, 0.785])     # hand behind the wall, forearm in slot


def slot_scene(gap):
    lo, hi = ZC - gap / 2, ZC + gap / 2
    return [("wall_low", [0.02, 1.0, lo / 2], [WX, 0.0, lo / 2]),
            ("wall_high", [0.02, 1.0, (1.4 - hi) / 2], [WX, 0.0, (1.4 + hi) / 2])]


print(f"\nBreak it · wall at x = {WX} m (2 m wide, 1.4 m high) with a horizontal slot centred at "
      f"z = {ZC} m. RRT-Connect, res 0.05 rad, 30 seeds, budget 3000 iterations")
print(f"{'slot (m)':>8} | {'start/goal free':>15} | {'success':>7} | {'med iters':>9} | "
      f"{'med checks':>10} | {'paths in collision':>18}")
for gap in [0.40, 0.30, 0.25, 0.22, 0.20]:
    c2 = Checker(build(slot_scene(gap)))
    free = c2.free(q_s2) and c2.free(q_g2)
    s = summarize(run(c2, rrt_connect, q_s2, q_g2, 0.05, range(30), 3000))
    it = f"{s['iters']:9.0f}" if s["succ"] else f"{'-':>9}"
    ca = f"{s['calls']:10.0f}" if s["succ"] else f"{'-':>10}"
    print(f"{gap:8.2f} | {str(free):>15} | {s['succ']:3d}/{s['n']:<3d} | {it} | {ca} | "
          f"{s['hit']:8d}/{s['succ']:<9d}")
