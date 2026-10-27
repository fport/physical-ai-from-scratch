"""Sampling-based motion planning in joint space (lesson 22 · Build it).

The planner never sees an obstacle: it only calls free(q) -> bool, a collision checker for one pose.
Everything here is plain NumPy; the lesson's free(q) wraps MuJoCo's mj_kinematics + mj_collision.
"""
import numpy as np


def edge_free(free, a, b, res):
    """True if free(q) holds at points spaced at most res (L2, rad) along the segment a -> b."""
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / res)))
    return all(free(a + (b - a) * (k / n)) for k in range(1, n + 1))


def worst_along(f, path, res):
    """Largest f(q) at points spaced at most res along a piecewise-linear path, both ends included."""
    worst = 0.0
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / res)))
        worst = max(worst, max(f(a + (b - a) * (k / n)) for k in range(n + 1)))
    return worst


def steer(a, b, eps):
    """Move from a toward b by at most eps (L2)."""
    v = b - a
    dist = np.linalg.norm(v)
    return b.copy() if dist <= eps else a + v * (eps / dist)


class Tree:
    """Poses in a preallocated array; parent[i] is the index of node i's parent (-1 for the root)."""

    def __init__(self, root, capacity=100_000):
        self.q = np.zeros((capacity, len(root)))
        self.parent = np.full(capacity, -1)
        self.q[0], self.n = root, 1

    def nearest(self, x):
        return int(np.argmin(np.sum((self.q[:self.n] - x) ** 2, axis=1)))

    def add(self, x, parent):
        self.q[self.n], self.parent[self.n] = x, parent
        self.n += 1
        return self.n - 1

    def branch(self, i):
        """Poses from node i back to the root."""
        out = []
        while i != -1:
            out.append(self.q[i].copy())
            i = self.parent[i]
        return out


def rrt_connect(free, lo, hi, q_start, q_goal, rng, eps=0.5, res=0.05, budget=2000):
    """RRT-Connect (Kuffner & LaValle 2000). Returns (path as a list of poses, or None; iterations used)."""
    ta, tb, a_is_start = Tree(q_start), Tree(q_goal), True
    for it in range(1, budget + 1):
        x = rng.uniform(lo, hi)                          # the dart
        i = ta.nearest(x)
        y = steer(ta.q[i], x, eps)
        if edge_free(free, ta.q[i], y, res):
            ja = ta.add(y, i)
            jb = tb.nearest(y)                           # CONNECT: tree b walks toward y
            while True:
                z = steer(tb.q[jb], y, eps)
                if not edge_free(free, tb.q[jb], z, res):
                    break                                # tree b is blocked
                jb = tb.add(z, jb)
                if np.allclose(z, y):                    # the trees touch
                    path = ta.branch(ja)[::-1] + tb.branch(jb)[1:]
                    return (path if a_is_start else path[::-1]), it
        ta, tb, a_is_start = tb, ta, not a_is_start      # alternate which tree grows
    return None, budget


def path_length(path):
    """Sum of the L2 lengths of the segments (rad)."""
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


def shortcut(free, path, res, rng, tries=200):
    """Random shortcutting: replace the poses between two waypoints by a straight edge if it is free."""
    p = [q.copy() for q in path]
    for _ in range(tries):
        if len(p) < 3:
            break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i >= 2 and edge_free(free, p[i], p[j], res):
            p = p[:i + 1] + p[j:]
    return p
