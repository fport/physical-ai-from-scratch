"""Forward kinematics and Jacobians of a serial chain of hinges (lesson 09 · Build it).

Only the model's static numbers come from MuJoCo (body offsets, joint axes, the tool site);
all the kinematics is computed here, so it can be checked against mj_kinematics and mj_jacSite.
"""
from dataclasses import dataclass

import numpy as np

from physai.spatial import axis_angle_to_mat, quat_to_mat


@dataclass
class Link:
    pos: np.ndarray    # body origin in the parent body's frame (m)
    rot: np.ndarray    # body orientation in the parent frame, 3x3, at q = 0
    axis: np.ndarray   # hinge axis in this body's frame (unit)
    anchor: np.ndarray  # a point on the hinge axis, in this body's frame (m)


@dataclass
class Chain:
    links: list        # base to tip, one hinge each
    tool_pos: np.ndarray  # tool point in the last body's frame (m)


def chain_from_mujoco(m, site_name):
    """Read the hinge chain from the world down to a site (static model data only)."""
    site = m.site(site_name)
    bodies, b = [], int(site.bodyid[0])
    while b != 0:                                  # walk up to the world body
        bodies.append(b)
        b = int(m.body_parentid[b])
    links = []
    for b in reversed(bodies):
        hinges = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == 3]
        j = hinges[0] if hinges else None          # bodies without a hinge are fixed offsets
        links.append(Link(m.body_pos[b].copy(), quat_to_mat(m.body_quat[b]),
                          m.jnt_axis[j].copy() if j is not None else None,
                          m.jnt_pos[j].copy() if j is not None else None))
    return Chain(links, site.pos.copy())


def fk(chain, q):
    """Joint axes and anchor points in the world, and the tool position, at joint angles q."""
    R, p = np.eye(3), np.zeros(3)                  # pose of the current body in the world
    axes, anchors, i = [], [], 0
    for link in chain.links:
        p = p + R @ link.pos                       # step to this body's origin...
        R = R @ link.rot                           # ...and its orientation at q = 0
        if link.axis is not None:
            a, o = R @ link.axis, p + R @ link.anchor
            axes.append(a)
            anchors.append(o)
            turn = axis_angle_to_mat(a, q[i])      # turn everything beyond about the hinge line
            R, p = turn @ R, o + turn @ (p - o)
            i += 1
    return np.array(axes), np.array(anchors), p + R @ chain.tool_pos


def hinge_column(axis, o, p):
    """Tool twist per 1 rad/s of one hinge: linear part a x (p - o), angular part a."""
    return np.r_[np.cross(axis, p - o), axis]


def jacobian(chain, q):
    """Geometric Jacobian (6 x n) of the tool point: rows 0-2 linear, rows 3-5 angular, world frame."""
    axes, anchors, p = fk(chain, q)
    return np.column_stack([hinge_column(a, o, p) for a, o in zip(axes, anchors)])


def fd_jacobian(tool_pos, q, eps=1e-5):
    """Position rows of the Jacobian by central differences of any function q -> tool position."""
    cols = []
    for i in range(len(q)):
        dq = np.zeros(len(q))
        dq[i] = eps
        cols.append((tool_pos(q + dq) - tool_pos(q - dq)) / (2 * eps))
    return np.column_stack(cols)
