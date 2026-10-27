"""Compact PPO for continuous control (lesson 33; imported by lessons 34 and 35).

Gaussian policy with a state-independent log-std, separate value network, GAE(lambda),
clipped surrogate, value loss, entropy bonus. CPU, single thread, N environments stepped in a loop.
Time-limit truncation is bootstrapped with V(s_final); true termination is not.

Usage:  cfg = PPOConfig(seed=0); policy, log = train(make_env, cfg); evaluate(make_env, policy, 100)
Nothing runs on import.
"""
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "32-rl-environment"))
from reach_env import wilson  # noqa: E402,F401  (re-exported for the experiments)


@dataclass
class PPOConfig:
    seed: int = 0
    total_steps: int = 120_000     # environment steps (policy decisions) over all envs
    n_envs: int = 16
    n_steps: int = 64              # rollout length per env per iteration -> 1024 samples / iteration
    epochs: int = 10
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 64
    log_std_init: float = -0.5


def mlp(n_in, n_out, h):
    return nn.Sequential(nn.Linear(n_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_out))


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_act, h, log_std_init):
        super().__init__()
        self.mu = mlp(n_obs, n_act, h)
        self.v = mlp(n_obs, 1, h)
        self.log_std = nn.Parameter(torch.full((n_act,), log_std_init))
        with torch.no_grad():                                   # small initial actions
            self.mu[-1].weight.mul_(0.01)
            self.mu[-1].bias.zero_()

    def dist(self, obs):
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())

    def act(self, obs_np, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(obs_np, dtype=torch.float32)
            if deterministic:
                return self.mu(o).numpy()
            d = self.dist(o)
            a = d.sample()
            return a.numpy(), d.log_prob(a).sum(-1).numpy(), self.v(o).squeeze(-1).numpy()


def train(make_env, cfg, eval_every=0, eval_fn=None, verbose=False):
    """Train PPO. make_env(i) returns a fresh env. Returns (policy, log dict)."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    envs = [make_env(i) for i in range(cfg.n_envs)]
    obs = np.stack([e.reset(seed=int(cfg.seed * 1000 + i))[0] for i, e in enumerate(envs)])
    n_obs, n_act = obs.shape[1], envs[0].action_space.shape[0]
    pi = ActorCritic(n_obs, n_act, cfg.hidden, cfg.log_std_init)
    opt = torch.optim.Adam(pi.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.n_steps, cfg.n_envs
    buf_o = np.zeros((T, N, n_obs), np.float32)
    buf_a = np.zeros((T, N, n_act), np.float32)
    buf_lp = np.zeros((T, N), np.float32)
    buf_v = np.zeros((T, N), np.float32)
    buf_r = np.zeros((T, N), np.float32)
    buf_done = np.zeros((T, N), np.float32)
    ep_ret, ep_returns, ep_success = np.zeros(N), [], []
    log = {"iters": [], "t0": time.perf_counter()}
    n_iter = cfg.total_steps // (T * N)

    for it in range(n_iter):
        for t in range(T):
            a, lp, v = pi.act(obs)
            buf_o[t], buf_a[t], buf_lp[t], buf_v[t] = obs, a, lp, v
            for i, e in enumerate(envs):
                o2, r, term, trunc, info = e.step(a[i])
                ep_ret[i] += r
                if trunc and not term:                        # time limit: bootstrap the value of s_final
                    with torch.no_grad():
                        r += cfg.gamma * pi.v(torch.as_tensor(o2)).item()
                buf_r[t, i] = r
                buf_done[t, i] = float(term or trunc)
                if term or trunc:
                    ep_returns.append(ep_ret[i])
                    ep_success.append(float(info["is_success"]))
                    ep_ret[i] = 0.0
                    o2, _ = e.reset()
                obs[i] = o2
        with torch.no_grad():
            last_v = pi.v(torch.as_tensor(obs)).squeeze(-1).numpy()
        # GAE(lambda)
        adv = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_r[t] + cfg.gamma * nv * nonterm - buf_v[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_v

        b_o = torch.as_tensor(buf_o.reshape(T * N, -1))
        b_a = torch.as_tensor(buf_a.reshape(T * N, -1))
        b_lp = torch.as_tensor(buf_lp.reshape(-1))
        b_adv = torch.as_tensor(adv.reshape(-1))
        b_ret = torch.as_tensor(ret.reshape(-1))
        clip_frac, approx_kl = [], []
        for _ in range(cfg.epochs):
            idx = rng.permutation(T * N)
            for s in range(0, T * N, cfg.minibatch):
                mb = idx[s:s + cfg.minibatch]
                d = pi.dist(b_o[mb])
                lp = d.log_prob(b_a[mb]).sum(-1)
                ratio = (lp - b_lp[mb]).exp()
                A = b_adv[mb]
                A = (A - A.mean()) / (A.std() + 1e-8)
                pg = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                vl = ((pi.v(b_o[mb]).squeeze(-1) - b_ret[mb]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pi.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    clip_frac.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    approx_kl.append((b_lp[mb] - lp).mean().item())
        row = {"iter": it + 1, "steps": (it + 1) * T * N,
               "ret": float(np.mean(ep_returns[-50:])) if ep_returns else float("nan"),
               "succ": float(np.mean(ep_success[-50:])) if ep_success else float("nan"),
               "kl": float(np.mean(approx_kl)), "clipfrac": float(np.mean(clip_frac)),
               "std": float(pi.log_std.detach().exp().mean())}
        if eval_every and eval_fn and ((it + 1) % eval_every == 0 or it + 1 == n_iter):
            row["eval_succ"] = eval_fn(pi)
        log["iters"].append(row)
        if verbose:
            print(row, flush=True)
    log["seconds"] = time.perf_counter() - log["t0"]
    return pi, log


def evaluate(make_env, pi, n_episodes=100, seed0=10_000, deterministic=True):
    """Run n episodes with fixed seeds. Returns dict with success count, returns and env stats."""
    env = make_env(0)
    succ, rets, final_d, tau_rate, power, lengths = 0, [], [], [], [], []
    for k in range(n_episodes):
        o, _ = env.reset(seed=seed0 + k)
        R, done = 0.0, False
        while not done:
            a = pi.act(o[None], deterministic=True)[0] if deterministic else pi.act(o[None])[0][0]
            o, r, term, trunc, info = env.step(a)
            R += r
            done = term or trunc
        succ += int(info["is_success"])
        rets.append(R)
        final_d.append(info["dist"])
        lengths.append(env.t)
        st = getattr(env, "stats", None)
        if st and st["n"]:
            tau_rate.append(np.sqrt(st["tau_rate_sq"] / st["n"]))
            power.append(st["power"] / st["n"])
    out = {"k": succ, "n": n_episodes, "ret": float(np.mean(rets)), "final_d": float(np.median(final_d)),
           "len": float(np.mean(lengths))}
    if tau_rate:
        out["tau_rate"] = float(np.mean(tau_rate))
        out["power"] = float(np.mean(power))
    return out
