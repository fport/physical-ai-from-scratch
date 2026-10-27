"""Rigid-body dynamics of a tree of hinges and slides (lesson 11 · Build it).

Only static model numbers come from MuJoCo (body offsets, joint axes, masses, centres of mass,
inertias, armature, gravity). Everything is computed in world coordinates with 6-D spatial vectors
(angular part first, then linear, taken about the world origin), so composite inertias are plain sums.
"""
from dataclasses import dataclass

import numpy as np

from physai.spatial import axis_angle_to_mat, quat_to_mat


@dataclass
class Body:
    parent: int          # index of the parent body in the tree, -1 for the world
    pos: np.ndarray      # origin in the parent's frame (m)
    rot: np.ndarray      # orientation in the parent's frame at q = 0, 3x3
    joint: str           # "hinge", "slide" or "" (welded to its parent)
    axis: np.ndarray     # joint axis in this body's frame (unit)
    anchor: np.ndarray   # a point on the joint axis, in this body's frame (m)
    mass: float          # kg
    com: np.ndarray      # centre of mass in this body's frame (m)
    inertia: np.ndarray  # 3x3 rotational inertia about the centre of mass, body frame (kg m^2)


@dataclass
class Tree:
    bodies: list         # base to tips: a parent always comes before its children
    armature: np.ndarray  # rotor inertia added to each joint, one per degree of freedom (kg m^2)
    gravity: np.ndarray  # m/s^2, world frame


def tree_from_mujoco(m):
    """Every body of a model whose joints are single hinges or slides (static model data only)."""
    bodies = []
    for b in range(1, m.nbody):
        kind, axis, anchor = "", None, None
        if m.body_jntnum[b] > 1:
            raise ValueError("one joint per body only")
        if m.body_jntnum[b]:
            j = m.body_jntadr[b]
            kind = {3: "hinge", 2: "slide"}[int(m.jnt_type[j])]
            axis, anchor = m.jnt_axis[j].copy(), m.jnt_pos[j].copy()
        R = quat_to_mat(m.body_iquat[b])           # principal axes of inertia in the body frame
        bodies.append(Body(int(m.body_parentid[b]) - 1, m.body_pos[b].copy(), quat_to_mat(m.body_quat[b]),
                           kind, axis, anchor, float(m.body_mass[b]), m.body_ipos[b].copy(),
                           R @ np.diag(m.body_inertia[b]) @ R.T))
    return Tree(bodies, m.dof_armature.copy(), m.opt.gravity.copy())


def skew(v):
    """Matrix of the cross product: skew(a) @ b = a x b."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def spatial_inertia(mass, c, Ic):
    """6x6 inertia about the world origin of a body with centre of mass c and inertia Ic about c (world)."""
    C = skew(c)
    return np.block([[Ic + mass * C @ C.T, mass * C], [mass * C.T, mass * np.eye(3)]])


def spatial_tree(tree, q):
    """At joint positions q: each body's centre of mass, 6x6 inertia and joint motion vector, all in the world.

    A joint's motion vector S is the body's spatial velocity per unit joint speed: (a, o x a) for a hinge
    with axis a through o, (0, a) for a slide; None for a welded body.
    """
    R, p, coms, inertias, S, k = [], [], [], [], [], 0
    for b in tree.bodies:
        Rp, pp = (R[b.parent], p[b.parent]) if b.parent >= 0 else (np.eye(3), np.zeros(3))
        Rb, pb, s = Rp @ b.rot, pp + Rp @ b.pos, None
        if b.joint:
            a, o = Rb @ b.axis, pb + Rb @ b.anchor
            if b.joint == "hinge":
                turn = axis_angle_to_mat(a, q[k])
                Rb, pb, s = turn @ Rb, o + turn @ (pb - o), np.r_[a, np.cross(o, a)]
            else:
                pb, s = pb + a * q[k], np.r_[np.zeros(3), a]
            k += 1
        R.append(Rb)
        p.append(pb)
        coms.append(pb + Rb @ b.com)
        inertias.append(spatial_inertia(b.mass, coms[-1], Rb @ b.inertia @ Rb.T))
        S.append(s)
    return coms, inertias, S


def cross_motion(v):
    """6x6 matrix of v x (.) for motion vectors (velocities, accelerations)."""
    w, u = skew(v[:3]), skew(v[3:])
    return np.block([[w, np.zeros((3, 3))], [u, w]])


def rnea(tree, q, qd, qdd):
    """Inverse dynamics by recursive Newton-Euler: the joint forces that give accelerations qdd at (q, qd)."""
    _, inertias, S = spatial_tree(tree, q)
    V, A, f, dof, k = [], [], [], [], 0
    for i, b in enumerate(tree.bodies):            # outward: velocities and accelerations
        if b.parent >= 0:
            v, a = V[b.parent], A[b.parent]
        else:                                      # gravity enters as an upward acceleration of the base
            v, a = np.zeros(6), np.r_[np.zeros(3), -tree.gravity]
        dof.append(k if S[i] is not None else None)
        if S[i] is not None:
            v = v + S[i] * qd[k]
            a = a + S[i] * qdd[k] + cross_motion(v) @ S[i] * qd[k]
            k += 1
        V.append(v)
        A.append(a)
        f.append(inertias[i] @ a - cross_motion(v).T @ inertias[i] @ v)   # net force on the body
    tau = tree.armature * np.asarray(qdd, dtype=float)
    for i in reversed(range(len(tree.bodies))):    # inward: each joint carries its whole subtree
        if dof[i] is not None:
            tau[dof[i]] += S[i] @ f[i]
        if tree.bodies[i].parent >= 0:
            f[tree.bodies[i].parent] = f[tree.bodies[i].parent] + f[i]
    return tau


def mass_matrix(tree, q):
    """Joint-space inertia M(q) by composite rigid bodies: M_ij = S_i . (Ic_j S_j) for i an ancestor of j."""
    _, inertias, S = spatial_tree(tree, q)
    comp = [I.copy() for I in inertias]
    for i in reversed(range(len(tree.bodies))):    # composite inertia: a body plus everything beyond it
        if tree.bodies[i].parent >= 0:
            comp[tree.bodies[i].parent] += comp[i]
    dof = np.cumsum([s is not None for s in S]) - 1
    M = np.zeros((len(tree.armature), len(tree.armature)))
    for j in range(len(tree.bodies)):
        if S[j] is None:
            continue
        F, i = comp[j] @ S[j], j                   # force to accelerate body j's subtree at joint j's unit rate
        while i >= 0:                              # ...as felt by every joint between j and the base
            if S[i] is not None:
                M[dof[i], dof[j]] = M[dof[j], dof[i]] = S[i] @ F
            i = tree.bodies[i].parent
    return M + np.diag(tree.armature)


def forward_dynamics(tree, q, qd, tau):
    """Joint accelerations for applied joint forces tau: M(q) qdd = tau - c(q, qd)."""
    bias = rnea(tree, q, qd, np.zeros(len(q)))     # c = C(q, qd) qd + g(q): forces for zero acceleration
    return np.linalg.solve(mass_matrix(tree, q), tau - bias)


def energy(tree, q, qd):
    """Kinetic plus potential energy (J), potential measured from the world origin."""
    coms, _, _ = spatial_tree(tree, q)
    potential = -sum(b.mass * tree.gravity @ c for b, c in zip(tree.bodies, coms))
    return 0.5 * qd @ mass_matrix(tree, q) @ qd + potential
