"""Foundation-model building blocks you can check without a GPU (lessons 41-43).

Plain NumPy. Lesson 41: OpenVLA-style action tokens and a flow-matching sampler. Lesson 42: the validation
layer between a language model's tool calls and the robot. Lesson 43: a pendulum, a least-squares world model,
open-loop rollouts and random-shooting MPC. Each piece is checked in experiments/<lesson>/build.py.
"""
import itertools
import json

import numpy as np

from physai.kinematics import fk, jacobian


# ---------------------------------------------------------------- lesson 41 · VLA action heads
def tokenize(a, lo, hi, n_bins=256):
    """OpenVLA-style action tokens: clip each dimension to [lo, hi], cut it into n_bins equal bins, return the index."""
    x = (np.clip(a, lo, hi) - lo) / (hi - lo)               # 0..1
    return np.minimum((x * n_bins).astype(int), n_bins - 1)


def detokenize(k, lo, hi, n_bins=256):
    """Back to an action: the centre of bin k."""
    return lo + (k + 0.5) / n_bins * (hi - lo)


def flow_velocity(x, tau, data):
    """The flow-matching velocity a perfectly trained network would output when the training set is `data` (n, d):
    for x_tau = tau a + (1 - tau) eps, E[a - eps | x_tau = x] = (weighted mean of the data - x) / (1 - tau)."""
    logw = -((x[None] - tau * data) ** 2).sum(axis=1) / (2 * (1 - tau) ** 2)   # posterior over training chunks
    w = np.exp(logw - logw.max())
    return (w @ data / w.sum() - x) / (1 - tau)


def sample_chunk(velocity, x0, n_steps):
    """Flow-matching inference: integrate dx/dtau = v(x, tau) from noise (tau = 0) to an action chunk (tau = 1)."""
    x = x0.copy()
    for i in range(n_steps):
        x = x + velocity(x, i / n_steps) / n_steps        # Euler step of size 1 / n_steps
    return x


# ---------------------------------------------------------------- lesson 42 · tool validation
JSON_TYPES = {"number": (int, float), "string": (str,), "boolean": (bool,), "object": (dict,)}


def check_args(schema, args):
    """Check tool-call arguments against the JSON-Schema subset tools use (object, required, typed properties,
    no unknown keys). Returns None if valid, else an error message for the model."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"missing argument(s): {', '.join(missing)}"
    for k, v in args.items():
        if k not in props:
            return f"unknown argument '{k}'"
        want = props[k].get("type")
        if want and (not isinstance(v, JSON_TYPES[want]) or (want == "number" and isinstance(v, bool))):
            return f"'{k}' must be a {want}, got {type(v).__name__}"   # True is an int in Python, not a number
    return None


def ik_position(chain, target, q0, lo, hi, iters=300, damping=0.03):
    """Damped-least-squares position IK on lesson 09's chain, joints clamped to [lo, hi] after every step.
    Returns the joint angles and the distance left between the tool point and the target (m)."""
    q = np.clip(np.asarray(q0, float), lo, hi)
    for _ in range(iters):
        e = target - fk(chain, q)[2]
        if np.linalg.norm(e) < 5e-4:
            break
        J = jacobian(chain, q)[:3]
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), e), lo, hi)
    return q, float(np.linalg.norm(target - fk(chain, q)[2]))


def validate_move(p_cm, reach, z_min_cm=2.0, tol_mm=5.0):
    """The move_to contract: above the table, and reachable within tol by IK inside the joint limits.
    reach(p_m) -> (q, error_m). Returns (q or None, reason or None)."""
    if p_cm[2] < z_min_cm:
        return None, f"z = {p_cm[2]:g} cm is below the minimum {z_min_cm} cm above the table"
    q, err = reach(np.asarray(p_cm, float) / 100)
    if err * 1000 > tol_mm:
        return None, f"target out of reach: closest IK solution within joint limits misses it by {err * 100:.1f} cm"
    return q, None


class ToolLayer:
    """The only path from a model's tool calls to the robot: unknown tools and bad arguments are answered with an
    error, rejected calls with the reason; both go back to the model as text, nothing raises."""

    def __init__(self, tools, schemas):
        self.tools, self.schemas, self.log = tools, schemas, []   # tools: name -> fn(**args) -> dict

    def call(self, name, args):
        if name not in self.tools:
            result = {"error": f"unknown tool '{name}'"}
        elif (problem := check_args(self.schemas[name], args)) is not None:
            result = {"error": problem}
        else:
            result = self.tools[name](**args)
        verdict = "error" if "error" in result else "rejected" if "rejected" in result else "ok"
        self.log.append((name, args, verdict))
        return json.dumps(result)


def claim_precision(episodes):
    """P(done | claimed success) from (claimed, done) pairs measured by the harness, with the counts."""
    claimed = [done for c, done in episodes if c]
    return (sum(claimed) / len(claimed) if claimed else float("nan")), len(claimed), len(claimed) - sum(claimed)


# ---------------------------------------------------------------- lesson 43 · world models
PEND = dict(inertia=0.25036, mgl=0.5 * 9.81, damping=0.05, u_max=2.0, h=0.01, substeps=5)   # the Experiment's


def pendulum_step(s, u, p=PEND):
    """One 0.05 s control step of the torque-limited pendulum, s = (theta, theta_dot), theta = 0 hanging.
    Semi-implicit Euler with implicit joint damping, as MuJoCo's Euler integrator; works on batches."""
    th, w = s[..., 0], s[..., 1]
    u = np.clip(u, -p["u_max"], p["u_max"])
    for _ in range(p["substeps"]):
        tau = u - p["mgl"] * np.sin(th) - p["damping"] * w
        w = w + p["h"] * tau / (p["inertia"] + p["h"] * p["damping"])   # damping taken at the new speed
        th = th + p["h"] * w
    return np.stack([th, w], axis=-1)


def poly_features(s, u, degree):
    """All monomials up to `degree` of (cos theta, sin theta, theta_dot / 8, u / 2), constant included."""
    base = np.stack([np.cos(s[..., 0]), np.sin(s[..., 0]), s[..., 1] / 8, u / 2], axis=-1)
    cols = [np.ones(base.shape[:-1])]
    for d in range(1, degree + 1):
        for idx in itertools.combinations_with_replacement(range(4), d):
            cols.append(np.prod(base[..., list(idx)], axis=-1))
    return np.stack(cols, axis=-1)


def fit_world_model(S, A, S_next, degree):
    """Least-squares fit of the change of state: S_next - S ~ poly_features(S, A) @ W. Returns W."""
    return np.linalg.lstsq(poly_features(S, A, degree), S_next - S, rcond=None)[0]


def predict(W, s, u, degree):
    """Learned dynamics: the next state is the current one plus the predicted change."""
    return s + poly_features(s, u, degree) @ W


def wrap(a):
    """Angle difference mapped into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def rollout_errors(step_true, step_model, s0, U, horizons):
    """Feed the same actions U (n, T) to the true system and to the model from the same states s0 (n, 2);
    return {H: (mean, median) |angle error|} after H steps of open-loop prediction."""
    s_true, s_model, out = s0.copy(), s0.copy(), {}
    for k in range(U.shape[1]):
        s_true, s_model = step_true(s_true, U[:, k]), step_model(s_model, U[:, k])
        if k + 1 in horizons:
            e = np.abs(wrap(s_model[:, 0] - s_true[:, 0]))
            out[k + 1] = (e.mean(), np.median(e))
    return out


def swing_up_cost(s, u):
    """Per-step cost: angle from upright, speed, torque."""
    return wrap(s[..., 0] - np.pi) ** 2 + 0.1 * s[..., 1] ** 2 + 0.001 * u ** 2


def shooting_mpc(step_plan, step_true, s0, H, K, rng, steps=100, hold=5, u_max=2.0):
    """Random-shooting MPC: K piecewise-constant torque sequences of length H, rolled out with step_plan;
    the first action of the cheapest one goes to step_true, then replan from the measured state.
    Returns success (within 0.2 rad of upright for the last 20 steps), mean tail error, first upright step."""
    s, angles = s0.copy(), []
    for _ in range(steps):
        U = np.repeat(rng.uniform(-u_max, u_max, (K, -(-H // hold))), hold, axis=1)[:, :H]
        x, cost = np.repeat(s[None], K, axis=0), np.zeros(K)
        for t in range(H):
            x = step_plan(x, U[:, t])
            cost += swing_up_cost(x, U[:, t])
        s = step_true(s, U[np.argmin(cost), 0])
        angles.append(abs(wrap(s[0] - np.pi)))
    a = np.array(angles)
    first = int(np.argmax(a < 0.2)) if np.any(a < 0.2) else -1
    return bool(np.all(a[-20:] < 0.2)), float(np.mean(a[-20:])), first
