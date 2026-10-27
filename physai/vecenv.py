"""A vectorized environment: N copies of one MuJoCo model stepped as a batch (lesson 36 · Build it).

One MjModel, one MjData per robot. Every policy step applies a batch of actions, advances each robot by
`decimation` physics steps, builds a batch of observations, ends episodes that fell or timed out and resets
them in the same step (as legged_gym does), and counts what was simulated.
"""
import mujoco
import numpy as np

from physai.spatial import quat_to_mat


def projected_gravity(quat):
    """Gravity direction (0, 0, -1) seen in the base frame: R^T g for the base quaternion (w, x, y, z)."""
    return quat_to_mat(quat).T @ np.array([0.0, 0.0, -1.0])


def roll_pitch(quat):
    """Roll and pitch of the base (rad), from R = Rz(yaw) Ry(pitch) Rx(roll)."""
    R = quat_to_mat(quat)
    return np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))


class VecEnv:
    """N robots, one model. reset_fn(m, d, rng) puts one MjData into a start state."""

    def __init__(self, m, n, reset_fn, rng, decimation=4, max_steps=1000, max_roll=0.8, max_pitch=1.0,
                 min_height=0.15):
        self.m, self.n, self.reset_fn, self.rng = m, n, reset_fn, rng
        self.decimation, self.max_steps = decimation, max_steps
        self.max_roll, self.max_pitch, self.min_height = max_roll, max_pitch, min_height
        self.datas = [mujoco.MjData(m) for _ in range(n)]
        self.t = np.zeros(n, dtype=int)                # policy steps in the current episode, per robot
        self.policy_steps = self.physics_steps = self.episodes = 0

    def obs_one(self, d):
        """Base height, projected gravity, base velocity (6), joint angles and speeds."""
        return np.r_[d.qpos[2], projected_gravity(d.qpos[3:7]), d.qvel[:6], d.qpos[7:], d.qvel[6:]]

    def reset(self):
        for d in self.datas:
            self.reset_fn(self.m, d, self.rng)
        self.t[:] = 0
        return np.array([self.obs_one(d) for d in self.datas])

    def step(self, actions):
        """actions: (n, nu) control targets. Returns obs, fell, timed_out (one row / entry per robot)."""
        for d, a in zip(self.datas, actions):
            d.ctrl[:] = a
            mujoco.mj_step(self.m, d, nstep=self.decimation)
        self.t += 1
        self.policy_steps += self.n
        self.physics_steps += self.n * self.decimation
        fell, timed_out = np.zeros(self.n, bool), self.t >= self.max_steps
        for i, d in enumerate(self.datas):
            roll, pitch = roll_pitch(d.qpos[3:7])
            fell[i] = abs(roll) > self.max_roll or abs(pitch) > self.max_pitch or d.qpos[2] < self.min_height
        for i in np.flatnonzero(fell | timed_out):     # auto-reset in the same step
            self.reset_fn(self.m, self.datas[i], self.rng)
            self.t[i] = 0
            self.episodes += 1
        return np.array([self.obs_one(d) for d in self.datas]), fell, timed_out


def samples_per_update(n_envs, steps_per_env):
    """On-policy batch size: every robot contributes steps_per_env transitions to one PPO update."""
    return n_envs * steps_per_env
