"""Lesson 35 · Build it: reward terms, shaping and the termination hack, checked at every step.

Step 1  The reward terms vs the Experiment's reward functions.
Step 2  Potential-based shaping telescopes: on a real SO-101 episode and around a loop.
Step 3  A 1-D corridor solved exactly (value iteration): which rewards keep the optimal policy, which change it.
Step 4  The termination hack in numbers: the scripted controller's written return vs parking outside the ball.
Step 5  How large an action-rate penalty is compared with the task term.

Step 3 is a tabular toy, solved exactly; Steps 2, 4 and 5 use the lesson's reach task and scripted IK controller,
not trained policies. The functions live in physai/rl.py (the lesson shows them step by step).
Run:  uv run python experiments/35-reward-design/build.py      (~3 s, deterministic)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "32-rl-environment"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.rl import (action_rate_penalty, closeness_reward, discounted_sum, potential_shaping,  # noqa: E402
                       sparse_reward, value_iteration)
from reach_env import ReachEnv  # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp = load("rl35_experiment", "experiment.py")
exp32 = load("rl32_experiment", "../32-rl-environment/experiment.py")
print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
G = 0.95

print("\n== Step 1. Reward terms vs the Experiment's reward functions, 1000 random steps ==")
rng = np.random.default_rng(0)
worst = {"sparse": 0.0, "pbrs": 0.0, "closeness": 0.0, "action rate": 0.0}
for _ in range(1000):
    info = {"dist": rng.uniform(0, 0.3), "dist_prev": rng.uniform(0, 0.3),
            "action": rng.uniform(-1, 1, 4), "prev_action": rng.uniform(-1, 1, 4)}
    d, dp = info["dist"], info["dist_prev"]
    ours = {"sparse": sparse_reward(d), "pbrs": sparse_reward(d) + potential_shaping(-20 * d, -20 * dp, G),
            "closeness": closeness_reward(d), "action rate": -d + action_rate_penalty(info["action"], info["prev_action"], 0.3)}
    theirs = {"sparse": exp.r_sparse(info), "pbrs": exp.r_pbrs(info), "closeness": exp.r_closeness(info),
              "action rate": exp.r_action_rate(0.3)(info)}
    for k in worst:
        worst[k] = max(worst[k], abs(ours[k] - theirs[k]))
print(", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
print("closeness at d = 0, 2, 5, 30.8 mm: " + ", ".join(f"{closeness_reward(x):.3f}" for x in (0, 0.02, 0.05, 0.0308)))

print("\n== Step 2. Shaping telescopes: sum_t gamma^t F_t = gamma^T Phi(s_T) - Phi(s_0), Phi = -20 d ==")
env = ReachEnv(reward_fn=lambda info: potential_shaping(-20 * info["dist"], -20 * info["dist_prev"], G))
for name, policy in (("scripted IK", lambda e: exp32.scripted_ik(e, None)), ("random actions", lambda e: rng.uniform(-1, 1, 4))):
    _, info = env.reset(seed=10_000)
    d0, F, done = info["dist"], [], False
    while not done:
        _, r, term, trunc, info = env.step(policy(env))
        F.append(r)
        done = term or trunc
    lhs, rhs = discounted_sum(F, G), G ** len(F) * (-20 * info["dist"]) - (-20 * d0)
    print(f"{name:>14}: d {d0 * 1000:5.1f} -> {info['dist'] * 1000:5.1f} mm, sum gamma^t F_t = {lhs:+.6f}, "
          f"gamma^T Phi_T - Phi_0 = {rhs:+.6f}, diff {abs(lhs - rhs):.1e}")
phi_a, phi_b = -20 * 0.10, -20 * 0.05                       # step 10 cm -> 5 cm, then back
for g in (1.0, G):
    loop = potential_shaping(phi_b, phi_a, g) + g * potential_shaping(phi_a, phi_b, g)
    print(f"loop 10 cm -> 5 cm -> 10 cm, gamma {g:.2f}: shaping earned {loop:+.4f} (= (gamma^2 - 1) Phi = {(g * g - 1) * phi_a + 0.0:+.4f})")

print("\n== Step 3. A corridor of 11 cells, 1 cm apart; cell 10 is the target and ends the episode ==")
n = 11
S = np.arange(n)
nxt = np.stack([np.maximum(S - 1, 0), S, np.minimum(S + 1, n - 1)], 1)       # actions: left, stay, right
term = S == n - 1
dist = (n - 1 - nxt) * 0.01                                                 # distance after the step [m]
d_now = (n - 1 - S)[:, None] * 0.01
rewards = {
    "sparse: 1 on arrival": (nxt == n - 1).astype(float),
    "sparse + gamma Phi' - Phi": (nxt == n - 1) + potential_shaping(np.where(term[nxt], 0.0, -20 * dist), -20 * d_now, G),
    "sparse + 0.5 per step closer": (nxt == n - 1) + 0.5 * (dist < d_now),
    "closeness 1 - tanh(d/5cm)": closeness_reward(dist),
}
print(f"{'reward':<30} greedy action in cells 0..9   value of cell 0")
values = {}
for name, R in rewards.items():
    values[name], pi = value_iteration(nxt, R, term, G)
    print(f"{name:<30} {''.join('<o>'[a] for a in pi[:-1]):>12}   {values[name][0]:15.3f}")
print("(> right, < left, o stay)")
shift = values["sparse + gamma Phi' - Phi"] - values["sparse: 1 on arrival"]
print(f"shaped - sparse values, cells 0..9: max |difference - (-Phi(s))| = {np.abs(shift[:-1] - 20 * d_now[:-1, 0]).max():.1e}")

print("\n== Step 4. The termination hack in numbers ==")
env = ReachEnv(reward_fn=exp.r_closeness, terminate_on_success=True)
rets, lengths = [], []
for e in range(100):
    env.reset(seed=10_000 + e)
    R, done = 0.0, False
    while not done:
        _, r, term_, trunc, info = env.step(exp32.scripted_ik(env, None))
        R, done = R + r, term_ or trunc
    rets.append(R)
    lengths.append(env.t)
park = closeness_reward(0.0308)
print(f"scripted IK, closeness reward, ends on success: written return {np.mean(rets):.2f}, mean length {np.mean(lengths):.1f} steps")
print(f"parking at 30.8 mm for 80 steps: {park:.3f} per step, written return {80 * park:.2f}; "
      f"discounted value of parking forever r / (1 - gamma) = {park / (1 - G):.2f}")
print(f"entering the ball pays at most {closeness_reward(0.0):.2f} once, then nothing")

print("\n== Step 5. Action-rate penalty vs the task term, scripted IK, 100 episodes ==")
env = ReachEnv()
d_sum, da_sum, steps = 0.0, 0.0, 0
for e in range(100):
    env.reset(seed=10_000 + e)
    done, a_prev = False, np.zeros(4)
    while not done:
        a = exp32.scripted_ik(env, None)
        _, r, term_, trunc, info = env.step(a)
        d_sum, da_sum, steps = d_sum - r, da_sum - action_rate_penalty(a, a_prev, 1.0), steps + 1
        a_prev, done = env.prev_a, term_ or trunc
print(f"per step: mean d = {d_sum / steps * 1000:.1f} mm, mean ||a_t - a_t-1||^2 = {da_sum / steps:.4f}")
for w in (0.03, 0.1, 0.3, 1.0, 3.0):
    print(f"w = {w:<4}: penalty / task term = {w * da_sum / d_sum:.2f}")
print(f"Go2 config: action_rate = -0.01 times dt 0.02 s -> {0.01 * 0.02:.4f} per step")
