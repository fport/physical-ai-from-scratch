"""SO-101 reach task as a Gymnasium environment (lesson 32; reused by lessons 33-35).

The arm (MuJoCo Menagerie `robotstudio_so101`) must bring its gripper site to a target point that is
sampled at every reset. Four joints are controlled (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex); wrist_roll and the gripper are held at 0 by their own servos.
Observation (15): q - q_home (4), 0.1 * qdot (4), target - gripper (3), previous action (4)
(+ servo target - q_home (4) in "delta" mode; obs_mode "no_vel" / "abs_goal" for lesson 34).

Action modes (lesson 34):
  "position"  absolute joint target  q* = q_home + scale * a        (servo PD inside MuJoCo)
  "delta"     incremental target     q* <- q* + scale * a
  "torque"    joint torque           tau = 2.94 N m * a             (servo PD removed)
Rewards and terminations are pluggable (lesson 35): reward_fn(info) -> float.
"""
import os
import sys

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
Q_HOME = np.array([0.0, -1.0, 1.0, 0.5])           # folded pose, gripper ~14 cm above the floor
TARGET_SPREAD = np.array([1.0, 0.6, 0.6, 0.6])     # targets = FK(q_home + U(-spread, spread))
TAU_MAX = 2.94                                      # STS3215 force range in the Menagerie model [N m]
DEFAULT_SCALE = {"position": 1.0, "delta": 0.1, "torque": 1.0}


def dense_reward(info):
    """Default reward: minus the gripper-to-target distance in metres, every policy step."""
    return -info["dist"]


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (95% by default)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


class ReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_skip=4, episode_seconds=1.6, action_mode="position", action_scale=None,
                 obs_mode="full", reward_fn=dense_reward, terminate_on_success=False,
                 success_radius=0.02, record=False):
        self.model = mujoco.MjModel.from_xml_path(path("robotstudio_so101"))
        self.data = mujoco.MjData(self.model)
        self._fk = mujoco.MjData(self.model)                     # scratch data for target sampling
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip            # policy period [s]
        self.max_steps = int(round(episode_seconds / self.dt))
        self.mode = action_mode
        self.scale = DEFAULT_SCALE[action_mode] if action_scale is None else action_scale
        self.obs_mode = obs_mode
        self.reward_fn = reward_fn
        self.terminate_on_success = terminate_on_success
        self.radius = success_radius
        self.record = record

        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.vadr = np.array([self.model.joint(j).dofadr[0] for j in JOINTS])
        self.act = np.array([self.model.actuator(j).id for j in JOINTS])
        self.site = self.model.site("gripperframe").id
        self.lo = self.model.actuator_ctrlrange[self.act, 0].copy()
        self.hi = self.model.actuator_ctrlrange[self.act, 1].copy()
        if action_mode == "torque":                               # turn the 4 servos into plain motors
            for i in self.act:
                self.model.actuator_gainprm[i, :3] = [1.0, 0.0, 0.0]
                self.model.actuator_biasprm[i, :3] = 0.0
                self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
                self.model.actuator_ctrlrange[i] = [-TAU_MAX, TAU_MAX]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        n_obs = len(self._obs_vector(np.zeros(3), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _ee(self):
        return self.data.site_xpos[self.site]

    def _sample_target(self):
        while True:
            q = np.clip(Q_HOME + self.np_random.uniform(-TARGET_SPREAD, TARGET_SPREAD), self.lo * 0.95, self.hi * 0.95)
            self._fk.qpos[:] = 0.0
            self._fk.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self._fk)
            p = self._fk.site_xpos[self.site].copy()
            if p[2] > 0.04 and self._fk.ncon == 0:               # above the floor, no self/floor contact
                return p

    def _obs_vector(self, goal_rel, q, qd, prev_a, ctrl_rel):
        parts = [q, ]
        if self.obs_mode != "no_vel":
            parts.append(0.1 * qd)
        if self.obs_mode == "abs_goal":
            parts.append(self.target if hasattr(self, "target") else np.zeros(3))
        else:
            parts.append(goal_rel)
        parts.append(prev_a)
        if self.mode == "delta":
            parts.append(ctrl_rel)
        return np.concatenate(parts).astype(np.float32)

    def _obs(self):
        q = self.data.qpos[self.qadr] - Q_HOME
        qd = self.data.qvel[self.vadr]
        return self._obs_vector(self.target - self._ee(), q, qd, self.prev_a,
                                self.data.ctrl[self.act] - Q_HOME)

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.target = self._sample_target()
        q0 = Q_HOME + self.np_random.uniform(-0.1, 0.1, size=4)
        self.data.qpos[self.qadr] = q0
        self.data.ctrl[:] = 0.0
        if self.mode != "torque":
            self.data.ctrl[self.act] = q0
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        self.prev_a = np.zeros(4)
        self.dist = float(np.linalg.norm(self.target - self._ee()))
        self.stats = {"tau_rate_sq": 0.0, "power": 0.0, "n": 0}
        return self._obs(), {"dist": self.dist}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.mode == "position":
            self.data.ctrl[self.act] = np.clip(Q_HOME + self.scale * a, self.lo, self.hi)
        elif self.mode == "delta":
            self.data.ctrl[self.act] = np.clip(self.data.ctrl[self.act] + self.scale * a, self.lo, self.hi)
        else:
            self.data.ctrl[self.act] = TAU_MAX * self.scale * a
        if self.record:                                          # physics-rate statistics for evaluation
            for _ in range(self.frame_skip):
                tau_before = self.data.actuator_force[self.act].copy()
                mujoco.mj_step(self.model, self.data)
                tau = self.data.actuator_force[self.act]
                self.stats["tau_rate_sq"] += float(np.sum(((tau - tau_before) / self.model.opt.timestep) ** 2))
                self.stats["power"] += float(np.sum(np.abs(tau * self.data.qvel[self.vadr])))
                self.stats["n"] += 1
        else:
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.t += 1
        d_prev, self.dist = self.dist, float(np.linalg.norm(self.target - self._ee()))
        success = self.dist < self.radius
        info = {"dist": self.dist, "dist_prev": d_prev, "action": a, "prev_action": self.prev_a,
                "success_now": success}
        reward = float(self.reward_fn(info))
        self.prev_a = a
        terminated = bool(self.terminate_on_success and success)
        truncated = (not terminated) and self.t >= self.max_steps
        if terminated or truncated:
            info["is_success"] = success                         # reached (terminal) or still there at the time limit
        return self._obs(), reward, terminated, truncated, info
