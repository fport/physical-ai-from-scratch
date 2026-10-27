# 32 · A MuJoCo RL environment

<!-- meta -->
| | |
|---|---|
| **Part** | [8 · Reinforcement learning for robots](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [12](../part-4-dynamics-and-control/12-joint-control.md), [25](../part-7-learning-from-demonstration/25-evaluating-policies.md) |
| **Learn first** | [MDPs, States, Actions & Rewards](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/01-mdps-states-actions-rewards) |
| **Time** | ~45 min |
| **Code** | [`experiments/32-rl-environment/experiment.py`](../experiments/32-rl-environment/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> Before any learning, write the contract: what the policy sees, what its numbers do to the motors, when an episode ends, and how often it decides.

## The problem

An RL algorithm only ever sees what the environment hands it: an observation, a reward, two end-of-episode flags. Mistakes in that interface do not crash anything; they make training quietly worse. The classic one: Gymnasium replaced the single `done` flag with `terminated` and `truncated` in v0.26 ([Env API](https://gymnasium.farama.org/api/env/)), because "the episode hit its time limit" and "the episode reached a terminal state" need different value targets. Unitree's locomotion training code carries the same distinction as a config switch, `send_timeouts = True # send time out information to the algorithm` ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py)). In this lesson we build the SO-101 reach environment used in lessons 33–35 and measure the baselines that tell us later what "learned" means.

## Core idea

The MDP itself is on AIEFS ([MDPs, States, Actions & Rewards](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/01-mdps-states-actions-rewards)). A Gymnasium environment implements it with two calls:

| Call | Returns | Robot meaning |
|---|---|---|
| `reset(seed=None)` | `obs, info` | put the robot in a start state, sample a new task (here: a target point) |
| `step(action)` | `obs, reward, terminated, truncated, info` | apply the action for one policy period, advance physics |
| `observation_space`, `action_space` | `Box` shapes and bounds | the policy's input and output sizes |

**Seeding.** `reset(seed=k)` re-seeds the environment's own generator `self.np_random`; all randomness (target, start pose) must come from it, so that `reset(seed=k)` reproduces the same episode. Evaluation then uses a fixed list of seeds.

**Terminated vs truncated.** `terminated` means the MDP reached a terminal state (goal reached, robot fell); `truncated` means something outside the MDP stopped it, usually a time limit ([Env API](https://gymnasium.farama.org/api/env/)). The learner's value target differs:

$$y_t = r_t + \gamma\,(1 - \text{terminated}_t)\, V(s_{t+1})$$

On truncation the future still exists and is bootstrapped; lesson 33 implements exactly that.

**Frame skip.** The policy acts every $k$ physics steps, so its period is $\Delta t = k\,h$. The Go2 training config uses $h = 5$ ms and `decimation = 4`, a 20 ms (50 Hz) policy ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)). The Menagerie SO-101 also uses $h = 5$ ms, so we take $k = 4$ as well.

**The reach task** ([`reach_env.py`](../experiments/32-rl-environment/reach_env.py)):

| Element | Definition |
|---|---|
| Robot | Menagerie `robotstudio_so101`; 4 controlled joints (shoulder pan, shoulder lift, elbow, wrist flex), wrist roll and gripper held at 0 |
| Action | $a \in [-1,1]^4$, joint target $q^* = q_{\text{home}} + 1.0\,\text{rad} \cdot a$, tracked by the model's servo PD (lesson 34 changes this) |
| Observation (15) | $q - q_{\text{home}}$, $0.1\,\dot q$, $p_{\text{target}} - p_{\text{gripper}}$, previous action |
| Target | per episode: forward kinematics of $q_{\text{home}} + U(\pm[1.0, 0.6, 0.6, 0.6])$ rad, rejected if below 4 cm or in contact, so every target is reachable |
| Start | $q_{\text{home}} + U(\pm 0.1)$ rad, at rest |
| Reward | $r_t = -\lVert p_{\text{target}} - p_{\text{gripper}} \rVert$ (metres) every step |
| Episode | 1.6 s = 80 policy steps, then `truncated` |
| Success | gripper within 2 cm of the target at the time limit |

Sampling targets in joint space and passing them through forward kinematics ([lesson 08](../part-3-kinematics/08-forward-kinematics.md)) is a cheap way to guarantee that a 0% success rate is the policy's fault, not the task's.

## Experiment

The two methods that matter:

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)                      # seeds self.np_random
    mujoco.mj_resetData(self.model, self.data)
    self.target = self._sample_target()           # FK of a random reachable joint vector
    q0 = Q_HOME + self.np_random.uniform(-0.1, 0.1, size=4)
    self.data.qpos[self.qadr] = q0
    self.data.ctrl[self.act] = q0                 # servos hold the start pose
    mujoco.mj_forward(self.model, self.data)
    ...
    return self._obs(), {"dist": self.dist}

def step(self, action):
    a = np.clip(action, -1.0, 1.0)
    self.data.ctrl[self.act] = np.clip(Q_HOME + self.scale * a, self.lo, self.hi)
    mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
    self.t += 1
    self.dist = float(np.linalg.norm(self.target - self._ee()))
    reward = self.reward_fn({"dist": self.dist, ...})
    terminated = bool(self.terminate_on_success and self.dist < self.radius)
    truncated = (not terminated) and self.t >= self.max_steps
    return self._obs(), reward, terminated, truncated, info
```

[`experiment.py`](../experiments/32-rl-environment/experiment.py) checks the API, runs three baseline policies for 200 episodes each (seeds 0–199) and sweeps `frame_skip`. The scripted controller is one damped least-squares IK step per policy step ([lesson 10](../part-3-kinematics/10-inverse-kinematics.md)): $\Delta q = J^\top (J J^\top + 0.05^2 I)^{-1} e$, sent as $q^* = q + \Delta q$. It reads the Jacobian from the model, which a policy cannot.

**Predict:** what success rate does a random policy get with a 2 cm radius?

MuJoCo 3.17.0, NumPy 2.5.4, Gymnasium 1.4.0, Apple Silicon Mac (Apple M5), one thread. The full script took 54 s wall-clock on a loaded machine (it prints this at the end); `python experiment.py --quick` (50 episodes per row, 1 s per throughput row) is a smoke test and does not reproduce the tables. Sections A, B and D are deterministic (a rerun gave identical numbers); section C is a timing and changes from run to run.

- `reset(seed=7)` twice (with `reset(seed=8)` in between) gives the same target: `True`.
- With `terminate_on_success=True`, the scripted controller's 50 episodes all ended `terminated` (50), none `truncated`.

**Baselines** (200 episodes each):

| policy | return (mean ± std) | median final distance | success | Wilson 95% |
|---|---|---|---|---|
| random | −14.37 ± 5.10 | 185.7 mm | 1/200 | 0.1–2.8% |
| zero action (hold home) | −12.87 ± 4.92 | 170.8 mm | 1/200 | 0.1–2.8% |
| scripted IK | −1.16 ± 0.68 | 0.2 mm | 200/200 | 98.1–100.0% |

Random actions are no better than standing still: they end 185.7 and 170.8 mm (median) from the target. The scripted IK sets the ceiling; its return of −1.16 is the distance accumulated while the arm moves, not a residual error (median final distance 0.2 mm). Any learned policy in this part will be placed between these two rows.

**Throughput vs frame_skip** (random actions, one env, one thread; measured in process CPU time so the other jobs on the machine distort it less; approximate):

| frame_skip | policy rate | env steps / CPU s | physics steps / CPU s | sim s / CPU s |
|---|---|---|---|---|
| 1 | 200 Hz | 22595 | 22595 | 113.0 |
| 2 | 100 Hz | 22316 | 44631 | 223.2 |
| 4 | 50 Hz | 14809 | 59235 | 296.2 |
| 8 | 25 Hz | 10900 | 87203 | 436.0 |
| 16 | 12.5 Hz | 6224 | 99585 | 497.9 |
| 32 | 6.25 Hz | 3357 | 107417 | 537.1 |

A linear fit gives **31.0 µs + 8.3 µs × frame_skip** of CPU time per `env.step()`. One SO-101 physics step costs about 8 µs; the Python side of the step (clipping, observation, reward, bookkeeping) costs about as much as four physics steps. At frame_skip 1 the env spends most of its time in Python, one reason to batch many environments per Python call (lesson 36). At frame_skip 4 the simulation runs 296.2 simulated seconds per CPU second.

## Break it

Truncation is part of the task definition. Shorten the time limit and run the scripted controller, which is known to be perfect at 1.6 s:

| time limit | steps | success | Wilson 95% | median final distance |
|---|---|---|---|---|
| 0.08 s | 4 | 11/200 | 3.1–9.6% | 120.1 mm |
| 0.16 s | 8 | 43/200 | 16.4–27.7% | 50.7 mm |
| 0.24 s | 12 | 81/200 | 33.9–47.4% | 23.1 mm |
| 0.32 s | 16 | 160/200 | 73.9–85.0% | 7.9 mm |
| 0.48 s | 24 | 199/200 | 97.2–99.9% | 1.0 mm |
| 1.60 s | 80 | 200/200 | 98.1–100.0% | 0.2 mm |

Below about 0.5 s even the model-based controller starts to miss: with torque-limited servos (±2.94 N·m) and one IK step per 20 ms, this controller needs between 0.32 and 0.48 s for almost every reach. It is not time-optimal (a policy can command larger target jumps), so 81/200 at 0.24 s is a reference, not a hard ceiling. Still, a learner given a 0.24 s episode is scored on a task that even a model-based controller mostly fails, and its low success would look like a learning problem. Measure a scripted or oracle baseline under the same time limit before tuning the learner.

## On the real robot

- On hardware `step()` has to wait for real time: one 20 ms policy step takes 20 ms. In simulation the same env ran 296.2 simulated seconds per CPU second (section C, frame_skip 4). That ratio is why policies are trained in simulation.
- `reset()` on a real arm is a motion, not an assignment: the arm has to be driven back to $q_{\text{home}}$ and the target placed or marked, which costs seconds per episode and needs a safe trajectory ([lesson 15](../part-4-dynamics-and-control/15-safety.md)).
- The observation must be built from what the robot measures: servo encoder positions, finite-difference velocities, and a target position from a camera ([lesson 20](../part-5-perception-and-estimation/20-pixels-to-table.md)). Each of these has noise and delay that this simulated env does not have (Part 9).
- Procedure for the SO-101: run the scripted IK baseline on 20 fixed reachable targets, measure the final gripper position with the camera, report $k/n$ with a Wilson interval. **Not yet measured on the real arm.**

## Check yourself

1. **Easy.** The scripted controller reaches the target and the env is configured with `terminate_on_success=True`. Which flag is set, and what does the learner bootstrap from that step?
2. **Medium.** Add a rule: if any arm geom touches the floor, the episode ends. Should that be `terminated` or `truncated`? Implement it in `step()` and count how often the random policy triggers it in 200 episodes.
3. **Hard.** Wrap 16 copies of the env in `gymnasium.vector.SyncVectorEnv` and `AsyncVectorEnv` and measure total env steps/s against one env. Where does the difference come from?

### Answers

1. `terminated = True`, `truncated = False`; the target is $y = r$ with no $\gamma V(s')$ term, because the goal state is terminal by definition of the task.
2. `terminated`: touching the floor is part of the task's dynamics (a failure state), not an external clock. The count is yours to measure; we did not run it.
3. Open; we did not run it. `SyncVectorEnv` still steps the 16 copies one after another in the same Python process, so expect roughly the single-env rate. `AsyncVectorEnv` runs each copy in its own subprocess, so the gain comes from extra cores, minus the inter-process cost per step, which matters most at small frame_skip where a step is short (section C).

## Takeaways and sources

- The environment is a contract: `reset(seed)` reproduces episodes, `step()` returns five values, and `terminated` (task over) must be kept apart from `truncated` (clock over).
- Baselines bracket the task: random and zero actions succeed 1/200 times, scripted IK 200/200; a learned policy must be read against both.
- Frame skip sets both the control rate and the simulator's speed; the episode length (truncation) defines what is achievable at all.

Next: [lesson 33](33-ppo.md), training a policy on this environment with PPO.

**Sources**

- [Gymnasium · `Env` API](https://gymnasium.farama.org/api/env/) (`reset`, `step`, terminated vs truncated, seeding).
- [MuJoCo documentation · Python bindings](https://mujoco.readthedocs.io/en/stable/python.html).
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py) (BSD-3-Clause).
- [MuJoCo Menagerie · `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- Code: [`reach_env.py`](../experiments/32-rl-environment/reach_env.py), [`experiment.py`](../experiments/32-rl-environment/experiment.py), [`output.txt`](../experiments/32-rl-environment/output.txt).
