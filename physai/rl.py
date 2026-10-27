"""Reinforcement learning for robots (lessons 32-35 · Build it).

The SO-101 reach task behind the Gymnasium API (reset/step, frame skip, terminated vs truncated), the pieces
of PPO (GAE, Gaussian log-probability, the clipped surrogate, approximate KL), the policy-to-robot interface
(action modes, PD torque, discount per control rate, smoothness and power) and reward design (reward terms,
potential-based shaping, tabular value iteration). NumPy and MuJoCo; `clipped_surrogate` takes PyTorch
tensors so that it can be differentiated, without importing PyTorch itself.
Checked in experiments/{32-rl-environment,33-ppo,34-observation-action-design,35-reward-design}/build.py
and in tests/.
"""
import mujoco
import numpy as np


# ---------------------------------------------------------------- lesson 32 · the environment
class ReachTask:
    """SO-101 reach with the Gymnasium API: position actions around a home pose, frame skip, time limit.

    Observation (15): q - q_home, 0.1 * qdot, target - gripper, previous action. Reward: -distance [m].
    """
    JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
    Q_HOME = np.array([0.0, -1.0, 1.0, 0.5])
    SPREAD = np.array([1.0, 0.6, 0.6, 0.6])        # targets = FK(q_home + U(-spread, spread)), always reachable

    def __init__(self, model_path, frame_skip=4, episode_seconds=1.6, scale=1.0, radius=0.02,
                 terminate_on_success=False):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data, self._fk = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.frame_skip, self.scale, self.radius = frame_skip, scale, radius
        self.terminate_on_success = terminate_on_success
        self.dt = self.model.opt.timestep * frame_skip                 # policy period
        self.max_steps = int(round(episode_seconds / self.dt))
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in self.JOINTS])
        self.vadr = np.array([self.model.joint(j).dofadr[0] for j in self.JOINTS])
        self.act = np.array([self.model.actuator(j).id for j in self.JOINTS])
        self.site = self.model.site("gripperframe").id
        self.lo, self.hi = self.model.actuator_ctrlrange[self.act].T.copy()
        self.np_random = None

    def _sample_target(self):
        while True:                                                    # rejection sampling in joint space
            q = np.clip(self.Q_HOME + self.np_random.uniform(-self.SPREAD, self.SPREAD), self.lo * 0.95, self.hi * 0.95)
            self._fk.qpos[:] = 0.0
            self._fk.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self._fk)
            p = self._fk.site_xpos[self.site].copy()
            if p[2] > 0.04 and self._fk.ncon == 0:                     # above the floor, nothing in contact
                return p

    def _obs(self):
        q = self.data.qpos[self.qadr] - self.Q_HOME
        qd = self.data.qvel[self.vadr]
        ee = self.data.site_xpos[self.site]
        return np.concatenate([q, 0.1 * qd, self.target - ee, self.prev_a]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None or self.np_random is None:
            self.np_random = np.random.default_rng(seed)               # all randomness comes from this generator
        mujoco.mj_resetData(self.model, self.data)
        self.target = self._sample_target()
        q0 = self.Q_HOME + self.np_random.uniform(-0.1, 0.1, size=4)
        self.data.qpos[self.qadr] = q0
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.act] = q0                                  # the servos hold the start pose
        mujoco.mj_forward(self.model, self.data)
        self.t, self.prev_a = 0, np.zeros(4)
        self.dist = float(np.linalg.norm(self.target - self.data.site_xpos[self.site]))
        return self._obs(), {"dist": self.dist}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[self.act] = np.clip(self.Q_HOME + self.scale * a, self.lo, self.hi)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)  # hold the action for frame_skip steps
        self.t += 1
        self.dist = float(np.linalg.norm(self.target - self.data.site_xpos[self.site]))
        self.prev_a = a
        success = self.dist < self.radius
        terminated = bool(self.terminate_on_success and success)       # the task itself is over
        truncated = (not terminated) and self.t >= self.max_steps      # we stopped watching
        info = {"dist": self.dist}
        if terminated or truncated:
            info["is_success"] = success
        return self._obs(), -self.dist, terminated, truncated, info


def value_target(r, v_next, terminated, gamma):
    """One-step value target: bootstrap from the next state unless the episode terminated (truncation bootstraps)."""
    return r + gamma * (1.0 - terminated) * v_next


# ---------------------------------------------------------------- lesson 33 · PPO
def gae(rewards, values, last_value, dones, gamma, lam):
    """Generalized advantage estimation over a (T, N) rollout, backwards in time; returns (advantages, returns).

    dones[t] = 1 where an episode ended after step t (terminated or truncated); a truncated step's reward
    must already include gamma * V(s_final).
    """
    adv = np.zeros_like(rewards)
    run = np.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]   # TD error
        run = delta + gamma * lam * nonterminal * run
        adv[t] = run
    return adv, adv + values


def gaussian_log_prob(a, mu, log_std):
    """Log-density of a diagonal Gaussian policy, summed over the action dimensions."""
    z = (a - mu) / np.exp(log_std)
    return (-0.5 * z ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


def clipped_surrogate(ratio, adv, clip=0.2):
    """PPO's policy loss on PyTorch tensors: -mean(min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A))."""
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - clip, 1 + clip) * adv
    return -unclipped.minimum(clipped).mean()


def approx_kl(logp_old, logp_new):
    """The cheap KL estimate PPO logs per update: mean of log pi_old - log pi_new over the batch's actions."""
    return (logp_old - logp_new).mean()


def horizon_seconds(dt, gamma):
    """How far ahead a discount looks, in seconds: the weights gamma^l sum to 1 / (1 - gamma) steps."""
    return dt / (1.0 - gamma)


# ---------------------------------------------------------------- lesson 34 · observation and action design
def action_to_ctrl(a, mode, q_home, scale, ctrl_prev, lo, hi, tau_max=2.94):
    """Map a policy output in [-1, 1] to the actuator command: absolute target, delta target or torque."""
    a = np.clip(a, -1.0, 1.0)
    if mode == "position":
        return np.clip(q_home + scale * a, lo, hi)
    if mode == "delta":
        return np.clip(ctrl_prev + scale * a, lo, hi)
    return tau_max * scale * a                                         # "torque": the policy is the controller


def pd_torque(q_star, q, qd, kp, kd, tau_max):
    """What a position servo does at every physics step: a PD law towards the target, clipped to its torque limit."""
    return np.clip(kp * (q_star - q) - kd * qd, -tau_max, tau_max)


def discount_for_rate(dt, horizon_s=0.4):
    """The discount that keeps a fixed look-ahead in seconds when the policy period dt changes."""
    return 1.0 - dt / horizon_s


def torque_rate_and_power(tau, qd, h):
    """Smoothness and energy of one episode at the physics rate.

    tau: (N + 1, n) actuator torques, starting with the one before the first step; qd: (N, n) joint velocities
    after each step; h: physics time step. Returns (RMS of ||d tau / dt|| [N m/s], mean sum |tau qdot| [W]).
    """
    rate = np.diff(tau, axis=0) / h
    return np.sqrt(np.mean(np.sum(rate ** 2, axis=1))), np.mean(np.sum(np.abs(tau[1:] * qd), axis=1))


# ---------------------------------------------------------------- lesson 35 · reward design
def sparse_reward(d, radius=0.02):
    """1 inside the success ball, 0 elsewhere."""
    return float(d < radius)


def closeness_reward(d, scale=0.05):
    """Positive at every step, 1 at the target: the reward behind the termination hack."""
    return 1.0 - np.tanh(d / scale)


def potential_shaping(phi, phi_prev, gamma):
    """Potential-based shaping term F = gamma * Phi(s') - Phi(s) (Ng, Harada, Russell 1999)."""
    return gamma * phi - phi_prev


def action_rate_penalty(a, a_prev, w):
    """Smoothness regularizer: -w * ||a_t - a_(t-1)||^2."""
    return -w * float(np.sum((np.asarray(a) - np.asarray(a_prev)) ** 2))


def discounted_sum(rewards, gamma):
    """sum_t gamma^t r_t."""
    return float(np.sum(np.asarray(rewards) * gamma ** np.arange(len(rewards))))


def value_iteration(next_state, reward, terminal, gamma, iters=2000):
    """Optimal values and greedy actions of a deterministic tabular MDP.

    next_state[s, a] and reward[s, a] describe one step; entering a terminal state ends the episode (value 0 after).
    """
    V = np.zeros(next_state.shape[0])
    for _ in range(iters):
        Q = reward + gamma * np.where(terminal[next_state], 0.0, V[next_state])
        V = Q.max(1)
    return V, Q.argmax(1)
