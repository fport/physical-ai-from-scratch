"""What a physics engine computes for you, rebuilt in plain NumPy (lessons 01-05).

Mass properties from shapes, time steppers, MuJoCo's soft contact, Coulomb friction cones and actuator
force laws. Each function is checked against MuJoCo in its lesson's Build it and in tests/.
"""
import numpy as np


# ---------- lesson 01: mass properties and drag ----------

def solid_inertia(kind, size, density):
    """Mass and principal inertia of a uniform solid, with MuJoCo's `size` convention (half-lengths)."""
    if kind == "box":                                  # size = half-sides (x, y, z)
        a, b, c = 2 * np.asarray(size, float)
        mass = density * a * b * c
        return mass, mass / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    if kind == "sphere":                               # size = (radius,)
        r = size[0]
        mass = density * 4 / 3 * np.pi * r**3
        return mass, np.full(3, 2 / 5 * mass * r * r)
    r, h = size[0], 2 * size[1]                        # cylinder/capsule: radius, half-length along z
    m_cyl = density * np.pi * r * r * h
    i_xx, i_zz = m_cyl * (3 * r * r + h * h) / 12, m_cyl * r * r / 2
    if kind == "cylinder":
        return m_cyl, np.array([i_xx, i_xx, i_zz])
    m_sph = density * 4 / 3 * np.pi * r**3             # capsule: cylinder plus two hemispherical caps
    i_xx += m_sph * (2 * r * r / 5 + h * h / 4 + 3 * h * r / 8)
    i_zz += m_sph * 2 * r * r / 5
    return m_cyl + m_sph, np.array([i_xx, i_xx, i_zz])


def combine_parts(masses, positions, inertias):
    """One rigid body from parts (axis-aligned principal inertias at given positions): mass, COM, 3x3 inertia."""
    masses, positions = np.asarray(masses, float), np.asarray(positions, float)
    mass = masses.sum()
    com = masses @ positions / mass
    inertia = np.zeros((3, 3))
    for m_i, p, i_diag in zip(masses, positions, inertias):
        r = p - com                                    # parallel-axis theorem: shift each part to the COM
        inertia += np.diag(i_diag) + m_i * (r @ r * np.eye(3) - np.outer(r, r))
    return mass, com, inertia


def inertia_box(mass, inertia):
    """Full side lengths of the uniform box with this mass and principal inertia (MuJoCo's drag shape)."""
    i_x, i_y, i_z = inertia
    return np.sqrt(6 / mass * np.array([i_y + i_z - i_x, i_x + i_z - i_y, i_x + i_y - i_z]))


def quadratic_drag(v, box, rho):
    """Drag force on a box moving at v (box axes): -1/2 rho * (area facing each axis) * |v| v."""
    area = np.array([box[1] * box[2], box[0] * box[2], box[0] * box[1]])
    return -0.5 * rho * area * np.abs(v) * v


# ---------- lesson 02: time steppers ----------

def explicit_euler(x, v, accel, h):
    """Move with the old velocity, then update it."""
    return x + h * v, v + h * accel(x, v)


def semi_implicit_euler(x, v, accel, h, c_implicit=0.0):
    """Update the velocity first, then move with the new one (MuJoCo's Euler).

    c_implicit: damping per unit inertia treated implicitly, v' = v + h a / (1 + h c), as MuJoCo's
    Euler does for joint damping and implicitfast for every velocity-dependent force (lesson 05).
    """
    v_new = v + h * accel(x, v) / (1 + h * c_implicit)
    return x + h * v_new, v_new


def rk4(x, v, accel, h):
    """Classic fourth-order Runge-Kutta on (x, v): four force evaluations per step."""
    k1x, k1v = v, accel(x, v)
    k2x, k2v = v + h / 2 * k1v, accel(x + h / 2 * k1x, v + h / 2 * k1v)
    k3x, k3v = v + h / 2 * k2v, accel(x + h / 2 * k2x, v + h / 2 * k2v)
    k4x, k4v = v + h * k3v, accel(x + h * k3x, v + h * k3v)
    return (x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
            v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def step_growth(step, h, k, c=0.0, **kw):
    """Largest |eigenvalue| of one step on a = -k x - c v: above 1, every step amplifies the motion."""
    accel = lambda x, v: -k * x - c * v                # noqa: E731
    cols = [step(x, v, accel, h, **kw) for x, v in ((1.0, 0.0), (0.0, 1.0))]
    return np.abs(np.linalg.eigvals(np.array(cols).T)).max()


def first_overlap(z0, h, top, bottom, half, g=9.81):
    """Free fall with MuJoCo's Euler from rest at z0: centre height at the first step a cube of half-size
    `half` overlaps a slab between `bottom` and `top`, or None if a step jumps over the slab."""
    z, v = z0, 0.0
    while z + half > bottom:
        z, v = semi_implicit_euler(z, v, lambda x, u: -g, h)
        if z - half < top and z + half > bottom:
            return z
    return None


# ---------- lesson 03: soft contact ----------

def impedance(r, solimp):
    """MuJoCo's impedance d(r) for penetration r >= 0: d0 at r = 0 rising to dwidth at r = width."""
    d0, dw, width, mid, power = solimp
    x = min(r / width, 1.0)
    if x >= 1.0:
        return dw
    if x <= mid:                                       # two power curves joined at the midpoint
        y = x**power / mid ** (power - 1)
    else:
        y = 1 - (1 - x) ** power / (1 - mid) ** (power - 1)
    return d0 + y * (dw - d0)


def solref_kb(solref, dmax, h, refsafe=True):
    """Stiffness K and damping B of the reference spring, as MuJoCo stores them (efc_KBIP[:, :2])."""
    timeconst, dampratio = solref
    if refsafe:
        timeconst = max(timeconst, 2 * h)              # the guard: never stiffer than two timesteps
    return 1 / (dmax**2 * timeconst**2 * dampratio**2), 2 / (dmax * timeconst)


def contact_accel(dist, vel, a0, solref, solimp, h, refsafe=True):
    """Normal acceleration of one frictionless contact (dist < 0 = penetrating, a0 = acceleration without it)."""
    if dist >= 0:
        return a0
    d = impedance(-dist, solimp)
    k, b = solref_kb(solref, solimp[1], h, refsafe)
    a_ref = -b * vel - k * d * dist                    # what the reference spring-damper asks for
    return max(a0, (1 - d) * a0 + d * a_ref)           # a contact can push, never pull


def resting_penetration(a_u, solref, solimp, h):
    """Penetration where the contact holds a body still: d k r = (1 - d) a_u, solved for r by bisection."""
    def excess(r):                                     # > 0 once the spring pushes more than needed
        d = impedance(r, solimp)
        k, _ = solref_kb(solref, solimp[1], h)
        return d * d * k * r - (1 - d) * a_u
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if excess(mid) > 0 else (mid, hi)
    return (lo + hi) / 2


# ---------- lesson 04: Coulomb friction ----------

def ramp_slip(theta, mu, g=9.81):
    """Does a block on a ramp at angle theta slip (tan theta > mu), and its downhill acceleration if so."""
    slips = np.tan(theta) > mu
    return slips, g * (np.sin(theta) - mu * np.cos(theta)) if slips else 0.0


def pyramid_rows(j_normal, j_tangent, mu):
    """Constraint rows of MuJoCo's pyramidal cone: J_n + mu J_t and J_n - mu J_t for each tangent direction."""
    return np.array([j_normal + s * m * jt for m, jt in zip(mu, j_tangent) for s in (1, -1)])


def pyramid_to_force(edge_forces, mu):
    """Contact-frame force (normal, tangential...) from the non-negative pyramid edge forces."""
    f = np.asarray(edge_forces, float).reshape(-1, 2)
    return np.r_[f.sum(), [m * (fp - fm) for m, (fp, fm) in zip(mu, f)]]


def tangential_limit(phi, mu, cone):
    """Largest friction force per unit normal force when sliding at angle phi to the first tangent axis."""
    if cone == "elliptic":
        return mu
    return mu / (abs(np.cos(phi)) + abs(np.sin(phi)))  # pyramid: |f1| + |f2| <= mu f_n, a diamond


# ---------- lesson 05: actuators ----------

def actuator_force(ctrl, length, velocity, gainprm, biasprm, forcerange=None):
    """MuJoCo's affine actuator: gain * ctrl + bias(length, velocity), then clamped to forcerange."""
    force = gainprm[0] * ctrl + biasprm[0] + biasprm[1] * length + biasprm[2] * velocity
    return force if forcerange is None else np.clip(force, *forcerange)


def kv_from_dampratio(kp, dampratio, inertia):
    """Damping gain that gives a position servo this damping ratio on this inertia: 2 zeta sqrt(kp I)."""
    return 2 * dampratio * np.sqrt(kp * inertia)


def servo_sag(kp, tau_g):
    """Steady error e of a PD servo holding a link against gravity: kp e = tau_g cos e (Newton's method)."""
    e = tau_g / kp
    for _ in range(50):
        e -= (kp * e - tau_g * np.cos(e)) / (kp + tau_g * np.sin(e))
    return e
