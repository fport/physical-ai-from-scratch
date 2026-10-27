# 37 · Control rate and end-to-end latency

<!-- meta -->
| | |
|---|---|
| **Part** | [9 · Sim to real](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [12](../part-4-dynamics-and-control/12-joint-control.md), [17](../part-5-perception-and-estimation/17-sensors-noise-filtering.md) |
| **Learn first** | [Stochastic Processes](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/22-stochastic-processes) |
| **Time** | ~45 min |
| **Code** | [`experiments/37-latency/experiment.py`](../experiments/37-latency/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A feedback loop does not care where the delay comes from; it only sees the sum, and the sum has a hard limit set by the loop's own speed.

## The problem

A PD loop tuned in simulation ([lesson 12](../part-4-dynamics-and-control/12-joint-control.md)) assumes the controller sees the joint *now* and the torque acts *now*. On a real robot the reading is old by the time it arrives, the command is computed on a clock, sent over a bus and held until the next tick. LeRobot's recording examples run the robot loop at 30 frames per second ([LeRobot imitation-learning guide](https://huggingface.co/docs/lerobot/il_robots)), so a policy that outputs one action per frame is already 33 ms coarse before any compute time. How much delay can a loop take, why does it go unstable instead of just "slower", and what can you do about it?

## Core idea

**Where latency comes from.** End-to-end latency is the time from a physical event to the actuator's response to it:

| Stage | Source of delay |
|---|---|
| Sensing | exposure / encoder sampling, sensor-internal filtering |
| Filtering | moving average $\frac{N-1}{2}\Delta t$, low-pass $\approx\tau$ ([lesson 17](../part-5-perception-and-estimation/17-sensors-noise-filtering.md)) |
| Compute | state estimation, policy inference (a large model can take longer than one tick) |
| Communication | USB / serial bus round trips, OS scheduling |
| Sample and hold | a command computed at tick $k$ is held for a period $T$: on average $T/2$ old |
| Actuation | the servo's own response time, torque saturation |

**Why delay destabilizes.** A pure delay $d$ multiplies the loop transfer function by $e^{-sd}$: gain unchanged, phase reduced by $\omega d$. A loop is stable while it has phase margin left at its gain-crossover frequency $\omega_c$ (where $|L(j\omega_c)| = 1$), so

$$d_{\max} = \frac{\mathrm{PM}}{\omega_c}.$$

For PD on an inertia, $L(s) = \frac{K_p + K_d s}{I s^2}$, the crossover solves $I^2\omega^4 = K_p^2 + K_d^2\omega^2$ and $\mathrm{PM} = \arctan(K_d\omega_c/K_p)$. Two consequences: a faster loop (higher $\omega_c$) tolerates *less* delay, and at the limit the loop oscillates at $\omega_c$.

**Control rate vs latency.** A controller at rate $f$ adds about $T/2 = 1/(2f)$ of delay from the hold, plus one full $T$ if the result is only applied at the next tick. Rate and latency are different knobs: a 1 kHz loop with 50 ms of camera delay is still a 50 ms loop.

**Mitigations.** (1) Lower the loop gain (slower, but more margin). (2) Predict: integrate a model from the delayed state over the delay, using the commands already sent (Smith-predictor idea). (3) Plan ahead in open loop: action chunking predicts a sequence of future actions per inference ([lesson 31](../part-7-learning-from-demonstration/31-act.md)); LeRobot's ACT defaults to `chunk_size = 100` and `n_action_steps = 100` ([`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)), and its rollout tool offers `--inference.type=rtc` "for smooth execution with slow VLA models" ([LeRobot guide](https://huggingface.co/docs/lerobot/il_robots)). Chunking trades reaction time for smoothness: while a chunk plays, the policy does not react to new observations.

## Experiment

Full script: [`experiments/37-latency/experiment.py`](../experiments/37-latency/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple Silicon Mac (Apple M5), deterministic, well under a minute of CPU.

**A. Delay sweep.** One joint as in lesson 12 ($I$ = 0.0901 kg·m², no gravity), $K_p$ = 20, $K_d$ = 1.879 (ζ = 0.7), physics and control at 1 kHz, step 0 → 0.5 rad. The controller reads the joint $d$ seconds late:

```python
j = max(k - nd, 0)
q_m, v_m = hist_q[j], hist_v[j]          # what the controller sees
u = kp * (TARGET - q_m) - kd * v_m
```

*Predict* with the formula: $\omega_c$ = 22.99 rad/s, PM = 65.2°, so $d_{\max}$ = 49.5 ms.

| delay [ms] | 0 | 5 | 10 | 20 | 30 | 40 | 50 | 60–100 |
|---|---|---|---|---|---|---|---|---|
| overshoot % | 4.45 | 4.47 | 4.51 | 5.02 | 11.51 | 31.23 | 215.84 | diverges |
| settle 2% [s] | 0.398 | 0.376 | 0.351 | 0.291 | 0.331 | 0.926 | never | never |

Bisection on whole milliseconds: stable at 48 ms, unstable at 49 ms. The oscillation at 49 ms has a period of 272 ms; $2\pi/\omega_c$ = 273 ms. Up to about 20 ms the delay costs almost nothing (it even shortens the settling time); the last 20 ms before the limit cost everything.

**B. Control rate.** Same joint, no added delay, controller at $f$ Hz with zero-order hold (physics stays at 1 kHz). Second variant: the command is ready one tick later. Last two columns: part A's overshoot at the equivalent delays $T/2$ and $3T/2$.

| rate [Hz] | T [ms] | overshoot % | + 1 tick compute | A at T/2 | A at 3T/2 |
|---|---|---|---|---|---|
| 1000 | 1.0 | 4.45 | 4.46 | 4.45 | 4.46 |
| 100 | 10.0 | 4.48 | 4.61 | 4.47 | 4.62 |
| 50 | 20.0 | 4.55 | 10.31 | 4.51 | 11.51 |
| 30 | 33.3 | 4.75 | 78.07 (unstable) | 4.72 | 215.84 |
| 20 | 50.0 | 5.34 | diverges | 6.66 | diverges |
| 10 | 100.0 | diverges | diverges | 215.84 | diverges |

The $T/2$ and $3T/2$ rules predict the trend: 30 Hz with immediate application behaves like 17 ms of delay (fine); the same 30 Hz with one tick of compute latency behaves like 50 ms and is past the limit.

**C. Prediction.** At $d$ = 40 ms, predict the present state by integrating $I\ddot q = \tau$ over the delay from the delayed reading, with the torques already sent:

| controller | overshoot % | settle 2% [s] |
|---|---|---|
| no delay (reference) | 4.45 | 0.398 |
| delay, no prediction | 31.23 | 0.926 |
| delay + prediction, exact model | 4.45 | 0.398 |
| inertia +50% in the model | 0.00 | 0.223 |
| inertia −30% in the model | 9.44 | 0.491 |
| assumed delay 20 ms (true 40) | 4.70 | 0.300 |

With an exact model and no disturbances, prediction removes the delay completely. With a wrong model it still helps here, but it changes the closed-loop response: knowing the delay and the dynamics (lesson 38) is what makes prediction safe.

**D. SO-101 at 30 Hz.** Menagerie SO-101, `shoulder_pan` servo (2 ms step). An outer "policy" loop at 30 Hz nudges the servo target from delayed readings: `target += K*T*(goal - q_delayed)`, $K$ = 10 s⁻¹. For an ideal servo, an integrator with delay is unstable when $K(d + T/2) > \pi/2$, i.e. above 140 ms (exact discrete-time analysis: from 5 ticks = 167 ms). But the servo is not ideal: alone, it reaches 50% of a 0.02 / 0.10 / 0.50 rad step after 18 / 44 / 110 ms. Every one of these steps drives the servo into its 2.94 N·m torque limit (with $k_p$ = 998 N·m/rad that takes only 2.94/998 ≈ 3 mrad of error), so the move is acceleration-limited and a larger step takes longer.

| delay [ms] | 0 | 33 | 50 | 67 | 100 | 117 | 133 |
|---|---|---|---|---|---|---|---|
| goal 0.1 rad: overshoot % | 0.0 | 0.1 | 12.3 | 30.8 | 70.9 | 80.8 | 94.7 |
| goal 0.1 rad: late \|e\|/goal | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.004 | 0.267 |
| goal 0.5 rad: overshoot % | 36.3 | 69.1 | 146.9 | 290.3 | 290.0 | 289.7 | 289.9 |
| goal 0.5 rad: late \|e\|/goal | 0.000 | 0.000 | 1.744 | 2.903 | 3.391 | 3.493 | 2.899 |

For small corrections the loop keeps oscillating only at 133 ms, close to the ideal-servo estimate. For a 0.5 rad move it already fails at 50 ms: the saturated servo adds its own, amplitude-dependent delay.

## Break it

Fix the delay at 30 ms and make the PD loop stiffer at constant ζ = 0.7:

| $K_p$ | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|
| $\omega_c$ [rad/s] | 11.49 | 16.25 | 22.99 | 32.51 | 45.97 | 65.01 |
| $d_{\max}$ [ms] | 98.9 | 70.0 | 49.5 | 35.0 | 24.7 | 17.5 |
| overshoot % | 4.73 | 5.30 | 11.51 | 37.78 | diverges | diverges |

The damping ratio is the same in every row, yet the loop dies at $K_p$ = 80. Stiffness raises $\omega_c$, and the same 30 ms costs $\omega_c \cdot d$ of phase. The formula predicts the failing rows exactly (24.7 and 17.5 ms are below 30 ms). Gains tuned in a zero-delay simulation are tuned for a loop that does not exist.

## On the real robot

- **Measure the latency, do not guess it.** LeRobot's SO-101 follower times each `Present_Position` bus read and each camera read in `get_observation` and logs them at debug level ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)). With `max_relative_target` set, `send_action` adds another bus read, and a code comment warns of a slower loop. Procedure for the end-to-end number: command a step on one joint, log command time and `Present_Position` time stamps, and take the time until the reading starts to move. Not yet measured on the real arm.
- **The servo runs its own inner loop.** As in part D, the policy closes a slow outer loop around a fast position servo. Latency in that outer loop costs reaction time for absolute-position actions, and stability when the policy reacts to its own delayed observations (visual servoing, corrections).
- **Simulate the latency you will have.** Tan et al. simulate latency, together with system identification and an actuator model, to transfer quadruped gaits from simulation ([arXiv 1804.10332](https://arxiv.org/abs/1804.10332)). Put the measured delay into the training simulator (lesson 39) rather than tuning at zero delay.

## Check yourself

1. **Easy.** A camera pipeline adds 40 ms, the policy runs at 20 Hz and its output is applied one tick after the frame. Estimate the total delay with the $T/2$ rule.
2. **Medium.** Using $d_{\max} = \mathrm{PM}/\omega_c$, explain why part B's 30 Hz loop with one tick of compute fails while part A at 40 ms survives.
3. **Hard.** In part C, replace the exact predictor's model with one that ignores the motor command (pure extrapolation $q + \dot q\,d$). Predict, then run: overshoot at 40 ms?

### Answers

1. 40 ms + 50 ms (one tick of compute) + 25 ms (hold) = 115 ms.
2. One tick of compute plus hold at 30 Hz is about $3T/2$ = 50 ms, past the 49.5 ms margin; part A's 40 ms is inside it (31.23% overshoot, stable).
3. 15.15% overshoot, between the exact predictor (4.45%) and no prediction (31.23%), but settling takes 1.902 s, longer than with no prediction (0.926 s): extrapolating with the delayed velocity ignores the acceleration the controller itself commanded during the delay. Row "extrapolation" in part C of `output.txt`.

## Takeaways and sources

- Delay removes phase $\omega d$; the loop survives up to $d_{\max} = \mathrm{PM}/\omega_c$ (predicted 49.5 ms, measured 48–49 ms, oscillating at the predicted 273 ms period).
- Control rate adds about $T/2$ of delay, plus $T$ for each tick of compute latency: 30 Hz is fine with immediate application and unstable with one tick of compute in this loop.
- Prediction with a correct model removes the delay (31.23% → 4.45% overshoot); stiffer loops and saturating actuators tolerate less delay.

Next: [lesson 38 · System identification](38-system-identification.md).

Sources:
- [LeRobot · Imitation learning on real-world robots](https://huggingface.co/docs/lerobot/il_robots) (30 fps examples, `lerobot-rollout`, `--inference.type=rtc`; read 2026-10-27).
- LeRobot [`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py), [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py) (Apache-2.0).
- J. Tan et al., [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332), 2018.
- T. Zhao et al., [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (ACT), 2023.
- [MuJoCo Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), ch. 11.
- All measured numbers: [`experiments/37-latency/output.txt`](../experiments/37-latency/output.txt).
