# 05 · Joints and actuators

<!-- meta -->
| | |
|---|---|
| **Part** | [1 · Simulation](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [01](01-world-as-text.md), [02](02-timestep.md) |
| **Learn first** | [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml) |
| **Time** | ~45 min |
| **Code** | [`experiments/05-joints-and-actuators/experiment.py`](../experiments/05-joints-and-actuators/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A `<position>` actuator is a PD controller living inside the physics step: its gain sets the sag, and the armature decides whether the step can integrate it.

## The problem

The Menagerie SO-101 gives every joint the same servo: `<position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>` on a joint with `armature="0.028"`, and a comment says these gains are not a one-to-one copy of the real servo's gains ([so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). Why is kp almost 1000, what does armature do on a hobby servo, and what happens if you copy the gains into your own model but drop the armature? This lesson builds one gravity-loaded link, measures how kp sets steady-state sag and rise time, and then finds the point where the same gains make the simulation diverge.

## Core idea

### Joints

| `joint type` | dof | Typical use |
|---|---|---|
| `hinge` | 1 rotation about `axis` | every SO-101 joint |
| `slide` | 1 translation along `axis` | linear stages, prismatic grippers |
| `ball` | 3 rotations | rarely in arms |
| `free` | 6 | floating bodies, robot bases |

| Joint attribute | Effect | Source note |
|---|---|---|
| `range` | position limit (a constraint, soft like contact) | enabled by `limited` or `autolimits` |
| `damping` | $\tau = -\text{damping}\cdot\dot q$ | the Euler integrator treats it implicitly |
| `armature` | extra inertia added to the joint's diagonal of $M$ | reflected rotor inertia: rotor inertia × gear ratio² |
| `frictionloss` | load-independent dry friction (unlike Coulomb, it does not scale with a normal force) | handled as a constraint type, not a force law |

Sources: [XML reference · joint](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-armature), [Computation · Friction loss](https://mujoco.readthedocs.io/en/stable/computation/index.html#cofriction).

### Actuators

Every actuator is `general`: force $= \text{gain}\cdot\text{ctrl} + \text{bias}$, then clamped to `forcerange`, then mapped to the joint through `gear` (joint torque = gear × force). The shortcuts set gain and bias ([XML reference · actuator](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position), [Modeling · Actuators](https://mujoco.readthedocs.io/en/stable/modeling.html#cactuators)):

| Shortcut | Force | Notes |
|---|---|---|
| `motor` | $u$ | torque command; the policy is the controller |
| `position` | $k_p(u - q) - k_v\,\dot q$ | built-in PD; `kp` default 1, `kv` default 0, or `dampratio` |
| `velocity` | $k_v(u - \dot q)$ | the docs recommend `implicitfast`/`implicit` with it |

`ctrlrange` clamps $u$; `forcerange` clamps the force. `dampratio` sets $k_v = 2\zeta\sqrt{k_p\,m}$ with $m$ the reflected inertia at `qpos0`.

### What the gains do

For a link with gravity torque $\tau_g\cos q$ and effective inertia $I = I_{\text{link}} + \text{armature}$:

$$e_{\text{ss}}:\ k_p\,e_{\text{ss}} = \tau_g \cos e_{\text{ss}}\ \ (\approx \tau_g/k_p), \qquad \omega = \sqrt{k_p / I}$$

Sag falls as $1/k_p$, rise time as $1/\sqrt{k_p}$. But the position term is integrated explicitly by `Euler`, `implicit` and `implicitfast`; the docs give the explicit stability limit as $h \lesssim 2/\omega_{\max}$ and note that only `discrete` is implicit in position ([Computation · Integrators](https://mujoco.readthedocs.io/en/stable/computation/index.html#geintegrators)). Armature lowers $\omega$; that is its numerical role. (Linearization around the setpoint: see AIEFS: [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml).)

## Experiment

A 20 cm, 0.3 kg capsule on a hinge, horizontal at $q = 0$. $\tau_g$ = 0.2943 N·m, $I_{\text{link}}$ = 0.0040 kg·m², `dt` = 5 ms (the SO-101 model timestep), `implicitfast`.

```python
<option timestep="0.005" integrator="implicitfast"/>
<body name="link">
  <joint name="j" type="hinge" axis="0 -1 0" armature="0.028"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.01" mass="0.3"/>
</body>
<actuator><position joint="j" kp="{kp}" dampratio="1"/></actuator>
```

Start at $q = -0.5$ rad, command $u = 0$, run 3 s. Sag = mean error over the last 0.5 s; rise = 10–90 % time.

**Predict.** kp × 10: what happens to sag, and to rise time?

**Run.** [`experiment.py`](../experiments/05-joints-and-actuators/experiment.py), MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministic; output in [`output.txt`](../experiments/05-joints-and-actuators/output.txt).

| kp (N·m/rad) | kv from dampratio 1 | sag (°) | $\tau_g/k_p$ (°) | exact (°) | rise 10–90 % (ms) |
|---|---|---|---|---|---|
| 1 | 0.359 | 16.193 | 16.862 | 16.193 | 533.0 |
| 3 | 0.621 | 5.594 | 5.621 | 5.594 | 335.6 |
| 10 | 1.134 | 1.685 | 1.686 | 1.685 | 185.6 |
| 30 | 1.964 | 0.562 | 0.562 | 0.562 | 106.1 |
| 100 | 3.586 | 0.169 | 0.169 | 0.169 | 56.7 |
| 300 | 6.211 | 0.056 | 0.056 | 0.056 | 31.6 |
| 1000 | 11.340 | 0.017 | 0.017 | 0.017 | 16.5 |

**Observe.** The sag equals the static balance $k_p e = \tau_g\cos e$ to three decimals in every row; the linear $\tau_g/k_p$ is off only when the sag is large (kp = 1). A position servo with no integral term always sags under gravity: 0.017° at kp 1000 here, 16.193° at kp 1. Rise time falls as $1/\sqrt{k_p}$ (kp × 10 from 10 to 100: 185.6 → 56.7 ms). Removing that sag without huge gains is what gravity feedforward does in [lesson 12](../part-4-dynamics-and-control/12-joint-control.md).

## Break it

**A · keep the SO-101 servo, drop the armature.** kp 998.22, kv 2.731, joint damping 0.6, frictionloss 0.052, `dt` = 5 ms; sweep armature:

| armature | $I$ (kg·m²) | $h\omega$ | Euler | implicitfast | discrete |
|---|---|---|---|---|---|
| 0.028 | 0.0320 | 0.88 | stable | stable | stable |
| 0.010 | 0.0140 | 1.34 | stable | stable | stable |
| 0.005 | 0.0090 | 1.67 | **diverged** | stable | stable |
| 0.002 | 0.0060 | 2.04 | **diverged** | stable | stable |
| 0.001 | 0.0050 | 2.23 | **diverged** | stable | stable |
| 0 | 0.0040 | 2.50 | **diverged** | stable | stable |

Euler fails first, already at $h\omega$ = 1.67: it treats joint damping implicitly but integrates the actuator's `kv` term explicitly, which is why the docs recommend `implicitfast` or `implicit` whenever `kv` is used ([actuator/position/kv](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position)). With the Menagerie `forcerange` ±2.94 N·m added, Euler no longer explodes; it chatters instead: peak-to-peak 0.629° at armature 0.001 and 0.798° at 0, a failure that looks like servo noise rather than a numerical problem.

**B · raise kp** (implicitfast, armature 0.028): stable at kp 3000 ($h\omega$ = 1.53), **diverged** at kp 10000 ($h\omega$ = 2.80) and 30000 ($h\omega$ = 4.84). `discrete` stayed stable at all four. `implicitfast` treats damping implicitly but `kp` explicitly. A linear analysis of that update gives the bound $h^2\omega^2 < 4 + 2hc$, with $c = (k_v + \text{damping})/I$; damping integrated explicitly (Euler's `kv`) enters with a minus sign instead, and the docs' $h\omega \lesssim 2$ is the undamped case. With armature 0.028, $hc$ is small and the limit sits just above 2: stable at 1.53, diverged at 2.80. In A, removing the armature made $I$ tiny and $hc$ large, which is why `implicitfast` survived $h\omega$ = 2.50 while Euler, with `kv` explicit, failed at 1.67. It predicts every cell in A and B.

## On the real robot

The script reads the compiled SO-101 model ([Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101)), home pose `qpos0`, `dt` = 5 ms:

| joint | kp | kv | forcerange (N·m) | armature | $M_{ii}$ (kg·m²) | link part | $h\omega$ | $h\omega$ without armature |
|---|---|---|---|---|---|---|---|---|
| shoulder_pan | 998.22 | 2.731 | ±2.94 | 0.028 | 0.04446 | 0.01646 | 0.75 | 1.23 |
| shoulder_lift | 998.22 | 2.731 | ±2.94 | 0.028 | 0.04542 | 0.01742 | 0.74 | 1.20 |
| elbow_flex | 998.22 | 2.731 | ±2.94 | 0.028 | 0.03734 | 0.00934 | 0.82 | 1.63 |
| wrist_flex | 998.22 | 2.731 | ±2.94 | 0.028 | 0.02913 | 0.00113 | 0.93 | 4.71 |
| wrist_roll | 998.22 | 2.731 | ±2.94 | 0.028 | 0.02808 | 0.00008 | 0.94 | 17.62 |
| gripper | 998.22 | 2.731 | ±2.94 | 0.028 | 0.02802 | 0.00002 | 0.94 | 39.33 |

For the wrist and gripper the armature is almost the whole effective inertia; without it those joints would sit far past $h\omega \approx 2$. Copying kp and kv from this model without its armature into an `Euler` model reproduces Break it A. Whether the SO-101 itself stays stable under its own `implicitfast` without armature was not tested here: its large implicit `kv` term raises the limit (Break it B), so do not assume either way.

Real hardware facts, from the [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101): the follower arm uses six STS3215 servos with 1/345 gearing; the leader mixes 1/191, 1/345 and 1/147 gearing so it can be moved by hand. The servos share one bus, each with a unique ID (shoulder pan 1 … gripper 6) stored in the servo's EEPROM. The Menagerie comment states that its kp, kv and forcerange were derived assuming a servo proportional gain of 16, and are not a one-to-one mapping of the gains LeRobot uses.

Procedure to compare: command a 0.5 rad step on the real elbow, log the joint position through LeRobot, compare rise time and steady sag with the simulated joint under the same load. **Not yet measured on the real arm.**

## Check yourself

1. **Easy.** Replace `<position>` by `<motor>` and send `ctrl = 0`. What does the link do? What `ctrl` holds it horizontal?
2. **Medium.** Add `gear="2"` to the position actuator, keep kp = 100. Does the sag change? Explain with the gear's effect on length and force.
3. **Hard.** For Break it A, find the largest armature at which Euler diverges with kv = 0 (keep joint damping 0.6). Compare with the $h\omega$ = 2 prediction.

### Answers

1. It falls under gravity (zero torque). Holding it needs $u = \tau_g$ = 0.2943 N·m at $q = 0$.
2. `gear` scales the actuator length and the force mapping, so the effective joint stiffness becomes gear² × kp: expect sag reduced about four times. Measure it.
3. Undamped prediction: $I = k_p (h/2)^2$, i.e. $h\omega = 2$. The implicit joint damping widens the bound to $h^2\omega^2 < 4 + 2hc$ with $c$ = damping/$I$, so expect divergence to start at a somewhat smaller armature. Run the sweep to find it.

## Takeaways and sources

- `<position>` = PD with gains kp, kv; steady sag under gravity is $\tau_g\cos e/k_p$ (16.193° at kp 1, 0.017° at kp 1000 here), rise time ∝ $1/\sqrt{k_p}$.
- Armature lowers $\omega = \sqrt{k_p/I}$. The SO-101's 0.028 is most of the wrist and gripper inertia and keeps $h\omega$ below 1 at 5 ms.
- `Euler` breaks first (explicit `kv`), `implicitfast` breaks once $h^2\omega^2 > 4 + 2hc$ (explicit `kp`, implicit damping; ≈ 2 when damping is small), `discrete` held in all tested cases.

Next: [lesson 06 · Rotations](../part-2-spatial-math/06-rotations.md). Control on top of these actuators: [lesson 12](../part-4-dynamics-and-control/12-joint-control.md).

**Sources**
- [MuJoCo XML reference · joint, actuator (motor, position, velocity, general)](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position).
- [MuJoCo Modeling · Actuators](https://mujoco.readthedocs.io/en/stable/modeling.html#cactuators); [Computation · Integrators](https://mujoco.readthedocs.io/en/stable/computation/index.html#geintegrators).
- [Menagerie · robotstudio_so101/so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml) (class `sts3215`).
- [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101).
- Our numbers: [`experiments/05-joints-and-actuators/output.txt`](../experiments/05-joints-and-actuators/output.txt).
