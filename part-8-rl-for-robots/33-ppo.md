# 33 · PPO for continuous control

<!-- meta -->
| | |
|---|---|
| **Part** | [8 · Reinforcement learning for robots](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [32](32-rl-environment.md) |
| **Learn first** | [Policy Gradient — REINFORCE from Scratch](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/06-policy-gradients-reinforce) · [Actor-Critic — A2C and A3C](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/07-actor-critic-a2c-a3c) · [Proximal Policy Optimization (PPO)](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/08-ppo) |
| **Time** | ~75 min |
| **Code** | [`experiments/33-ppo/experiment.py`](../experiments/33-ppo/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> PPO itself is a few lines of math. On a robot, most of the work is choosing the time scales around it.

## The problem

In [lesson 32](32-rl-environment.md) a random policy reached the target in 1 of 200 episodes and a scripted IK controller reached all 200, but only because it read the Jacobian from the model. The question here: can a policy learn the same reach from reward alone, on a CPU, in about a minute per run? And when it does, how sure are we? One training run is not a result: RL outcomes change with the seed. So we report success over fixed evaluation episodes with a Wilson 95% interval (see [lesson 25](../part-7-learning-from-demonstration/25-evaluating-policies.md)), for three seeds.

## Core idea

The policy-gradient, actor-critic and PPO derivations are on AIEFS ([REINFORCE](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/06-policy-gradients-reinforce), [A2C](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/07-actor-critic-a2c-a3c), [PPO](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/08-ppo)). For continuous control you need four pieces.

**Gaussian policy.** The actor outputs the mean of a diagonal Gaussian, $a \sim \mathcal N(\mu_\theta(o), \mathrm{diag}\,\sigma^2)$, with $\log\sigma$ a learned vector that does not depend on the observation. The environment clips $a$ to $[-1, 1]$; the log-probability is computed for the unclipped sample.

**GAE** (Schulman et al., [arXiv 1506.02438](https://arxiv.org/abs/1506.02438)):

$$\delta_t = r_t + \gamma (1-d_t) V(s_{t+1}) - V(s_t), \qquad \hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}$$

**Clipped surrogate** (Schulman et al., [arXiv 1707.06347](https://arxiv.org/abs/1707.06347)), with $\rho_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$:

$$L = -\mathbb E\big[\min(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t)\big] + c_v (V_\phi(s_t) - \hat R_t)^2 - c_e\, \mathcal H[\pi_\theta]$$

**Truncation is not termination.** At the 1.6 s time limit the arm is not in a terminal state, so the last reward gets the bootstrap $r_T \leftarrow r_T + \gamma V(s_{T+1})$ and only then is the episode cut ($d_t = 1$). A true terminal state (lesson 35) gets no bootstrap.

**The robot-specific part: $\gamma$ is a time constant.** A discount of $\gamma$ at policy period $\Delta t$ weighs the future over roughly

$$\tau \approx \frac{\Delta t}{1-\gamma}$$

seconds. At 50 Hz, $\gamma = 0.99$ means 2 s and $\gamma = 0.95$ means 0.4 s. Change the control rate and you silently change the task's horizon. The GAE paper treats $\gamma$ explicitly as a variance-reduction parameter that introduces bias, not as part of the true objective.

| | This lesson | Go2 training config (`unitree_rl_gym`) |
|---|---|---|
| parallel envs × steps per iteration | 16 × 64 | 4096 × 24 |
| epochs / minibatches per iteration | 10 / 4 | 5 / 4 |
| clip $\epsilon$, $\gamma$, $\lambda$ | 0.2, 0.95, 0.95 | 0.2, 0.99, 0.95 |
| learning rate | 1e-3, fixed | 1e-3, `schedule = 'adaptive'`, `desired_kl = 0.01` |
| entropy coefficient | 0 | 0.01 |
| policy network | 64-64 tanh | 512-256-128 elu |

The right column is read from [`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py) (`LeggedRobotCfgPPO`) and [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py).

## Experiment

[`ppo.py`](../experiments/33-ppo/ppo.py) is the whole algorithm (about 200 lines, no RL library); [`experiment.py`](../experiments/33-ppo/experiment.py) runs it. The environment is the lesson-32 reach task unchanged: absolute joint targets, 50 Hz, reward $-d$, 300,000 policy steps per run. The core of the update:

```python
# GAE over a (T, N) rollout; buf_done = terminated OR truncated,
# truncated steps already got r += gamma * V(s_final)
for t in reversed(range(T)):
    nv = last_v if t == T - 1 else buf_v[t + 1]
    nonterm = 1.0 - buf_done[t]
    delta = buf_r[t] + gamma * nv * nonterm - buf_v[t]
    gae = delta + gamma * lam * nonterm * gae
    adv[t] = gae

# one minibatch step
ratio = (pi.dist(o).log_prob(a).sum(-1) - logp_old).exp()
A = (A - A.mean()) / (A.std() + 1e-8)
pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
loss = pg + 0.5 * ((pi.v(o).squeeze(-1) - ret) ** 2).mean() - ent_coef * entropy
```

**Predict** before you run: with $\gamma$ from 0.9 to 0.995, which horizon works best for a 1.6 s reach?

MuJoCo 3.17.0, NumPy 2.5.4, PyTorch 2.14.1 (CPU, one thread per run), Apple Silicon Mac (Apple M5). Each run is deterministic for its seed: a full rerun reproduced every success count, distance, return and KL value in [`output.txt`](../experiments/33-ppo/output.txt); only the timings changed. Evaluation uses the mean action on episodes with seeds 10000–10099.

**Reference config, three seeds:**

| seed | success | Wilson 95% | median final distance | train time |
|---|---|---|---|---|
| 0 | 100/100 | 96.3–100.0% | 6.6 mm | 61 s |
| 1 | 98/100 | 93.0–99.4% | 7.9 mm | 61 s |
| 2 | 100/100 | 96.3–100.0% | 8.0 mm | 60 s |
| pooled | 298/300 (99.3%) | 97.6–99.8% | | |

Training times are wall-clock with six runs in parallel on a Mac that was also running other jobs; treat them as approximate. The whole script (15 runs) took 222 s and prints its wall-clock time on the last line; under heavier load expect several minutes more. `python experiment.py --quick` trains seed 0 only (5 runs): it reproduces the seed-0 entries of every table, not the pooled numbers.

Training success (stochastic policy, last 50 episodes) by environment steps:

| seed | 50k | 100k | 150k | 200k | 250k | 299k |
|---|---|---|---|---|---|---|
| 0 | 0.20 | 0.14 | 0.18 | 0.68 | 0.96 | 1.00 |
| 1 | 0.16 | 0.22 | 0.28 | 0.40 | 0.92 | 0.96 |
| 2 | 0.16 | 0.20 | 0.28 | 0.88 | 0.94 | 0.98 |

For the first 150k steps training success stays between 0.14 and 0.28; then it jumps within about 100k steps. A run stopped at 150k would look like a failure.

**Sweep $\gamma$** (3 seeds each, 300 evaluation episodes pooled):

| $\gamma$ | horizon at 50 Hz | success per seed | pooled | Wilson 95% |
|---|---|---|---|---|
| 0.9 | 0.2 s | 77 / 97 / 98 | 272/300 (90.7%) | 86.8–93.5% |
| **0.95** | **0.4 s** | 100 / 98 / 100 | 298/300 (99.3%) | 97.6–99.8% |
| 0.99 | 2.0 s | 32 / 74 / 53 | 159/300 (53.0%) | 47.3–58.6% |
| 0.995 | 4.0 s | 25 / 31 / 79 | 135/300 (45.0%) | 39.5–50.7% |

The common default $\gamma = 0.99$ halves the success rate here, and the seed spread grows (25 to 79 out of 100 at 0.995). Within this budget, a reach that takes a few hundred milliseconds was best served by a horizon of the same order. We did not isolate the mechanism. Two candidates: longer horizons add the noise of distant rewards to every advantage estimate (the bias-variance trade the GAE paper describes), and the value targets grow like $1/(1-\gamma)$ while `ppo.py` neither normalizes returns nor separates the gradient-norm clip of actor and critic, so a larger value loss also shrinks the policy step. Two caveats: this is a budget-limited result (300k steps; we did not test whether $\gamma = 0.99$ catches up with more data), and the Wilson intervals count 300 evaluation episodes but only three trained policies, so they understate seed-to-seed variation. The 0.95 vs 0.99 gap holds in every seed; the 0.99 vs 0.995 difference does not.

## Break it

The clip is what keeps ten epochs on the same batch from walking the policy too far. Set $\epsilon = 100$ (no clipping in practice) and reuse each batch for 30 epochs:

| seed | success | Wilson 95% | median final distance | approx. KL per update, iter 1 / 10 / 50 / 100 / 292 |
|---|---|---|---|---|
| 0 | 0/100 | 0.0–3.7% | 196.5 mm | 1.49 / 3.1 / 8.2 / 435 / 3780 |
| 1 | 0/100 | 0.0–3.7% | 272.5 mm | 1.58 / 4.16 / 17.5 / 46.6 / 125 |
| 2 | 1/100 | 0.2–5.4% | 144.2 mm | 1.94 / 3.04 / 7.78 / 392 / 1070 |
| reference ($\epsilon$ 0.2, 10 epochs) | | | | 0.00759 / 0.0201 / 0.0211 / 0.0259 / 0.0657 (mean of 3) |

Mechanism: the surrogate $\rho \hat A$ is only a valid estimate of improvement near $\pi_{\theta_{\text{old}}}$. Without the clip, the optimizer keeps pushing $\rho$ for the samples with large advantages, the mean and $\log\sigma$ jump, and the approximate KL between the policy that collected the batch and the one being updated goes from about 0.01 to hundreds. The next batch is then collected by a policy that has little to do with the one the value function was fitted to, so the advantages are wrong too. The final distances (144.2–272.5 mm) are of the same order as the random policy in lesson 32 (185.7 mm median). Watching approximate KL and clip fraction per update is the cheapest early warning; Unitree's config goes one step further and adapts the learning rate to a KL target (`desired_kl = 0.01`).

## On the real robot

- The trained actor is a 15 → 64 → 64 → 4 MLP; deploying it means running $\mu_\theta(o)$ at 50 Hz and sending the four outputs as servo targets. The critic is not needed on the robot.
- The SO-101's STS3215 servos are modelled in the Menagerie file as position actuators with `kp = 998.22`, `kv = 2.731` and `forcerange = ±2.94` N·m ([`so101.xml`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101)). The real servo's internal loop, latency and backlash differ; the policy was trained on none of that (Part 9).
- Procedure for the real arm: fixed list of 20+ reachable targets, run the deterministic policy, measure final gripper position with the calibrated camera of [lesson 20](../part-5-perception-and-estimation/20-pixels-to-table.md), report $k/n$ with a Wilson interval. **Not yet measured on the real arm.**

## Check yourself

1. **Easy.** What horizon in seconds does the Go2 config's $\gamma = 0.99$ correspond to at its 50 Hz policy rate? What $\gamma$ gives the same horizon at 200 Hz?
2. **Medium.** In `ppo.py`, delete the truncation bootstrap (treat the time limit as a terminal state). Before running: which states get a wrong value, and is it too high or too low? Run three seeds and compare with the reference table.
3. **Hard.** Replace the fixed learning rate by the KL-adaptive rule (halve the lr when the mean KL of an iteration exceeds $2 \times 0.01$, grow it by 1.5× when below $0.01/2$). Does it rescue the Break-it setting? Report success with Wilson intervals over three seeds.

### Answers

1. $\tau = 0.02 / (1 - 0.99) = 2.0$ s. At 200 Hz ($\Delta t = 5$ ms): $\gamma = 1 - 0.005/2 = 0.9975$.
2. The states near the time limit. Every reward here is $-d \le 0$, so cutting off the future makes their value targets too high (closer to zero than the bootstrapped targets), and the same arm state gets different targets depending on the clock, which is not in the observation. Measure it; we did not run this variant.
3. Open; we did not run it. Report the KL per iteration together with success, otherwise you cannot tell whether the schedule or luck did it.

## Takeaways and sources

- A compact PPO (Gaussian policy, GAE, clipped surrogate) learns the SO-101 reach on a CPU: 298/300 evaluation episodes (Wilson 97.6–99.8%) over three seeds, about a minute per run.
- $\gamma$ is a horizon in seconds, $\Delta t/(1-\gamma)$; at 50 Hz, 0.95 (0.4 s) reached 99.3% and the default 0.99 (2 s) 53.0% within the same budget.
- Without the clip and with heavy data reuse, KL explodes and success falls to 0–1%; log KL and clip fraction every update.

Next: [lesson 34](34-observation-action-design.md), what the policy sees and what its outputs mean.

**Sources**

- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms", 2017. [arXiv 1707.06347](https://arxiv.org/abs/1707.06347)
- J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2015. [arXiv 1506.02438](https://arxiv.org/abs/1506.02438)
- [Gymnasium `Env` API](https://gymnasium.farama.org/api/env/) (terminated vs truncated).
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py) (BSD-3-Clause).
- [MuJoCo Menagerie · `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- Code: [`experiments/33-ppo/ppo.py`](../experiments/33-ppo/ppo.py), [`experiment.py`](../experiments/33-ppo/experiment.py), [`output.txt`](../experiments/33-ppo/output.txt).
