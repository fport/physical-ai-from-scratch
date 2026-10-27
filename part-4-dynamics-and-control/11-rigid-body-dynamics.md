# 11 · Rigid-body dynamics

<!-- meta -->
| | |
|---|---|
| **Part** | [4 · Dynamics and control](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [09](../part-3-kinematics/09-jacobians.md) |
| **Learn first** | [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml) · [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) |
| **Time** | ~60 min |
| **Code** | [`experiments/11-rigid-body-dynamics/experiment.py`](../experiments/11-rigid-body-dynamics/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> Every controller in this part is built from three terms: an inertia matrix, a bias force and the torque you send. MuJoCo computes all three for you; this lesson checks that it computes them correctly.

## The problem

To hold a Franka Panda still in its `home` pose, joint 2 needs about 29 N·m and joint 4 about 22 N·m just to fight gravity (measured on the Menagerie model below). To move it along a path you also need the torque that accelerates its links, and that torque depends on where the other links are. If a controller ignores any of this, the arm sags, lags or overshoots, and the gains that fix it in one pose break it in another. Lessons 12–16 all start from the same equation of motion. Before using it, we verify MuJoCo's version of it against a system small enough to derive by hand, and we measure how the choice of integrator changes the physics it produces.

## Core idea

For a rigid open chain with joint positions $q \in \mathbb{R}^n$ (Lynch & Park, *Modern Robotics*, ch. 8):

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = \tau + J^\top(q)\, f$$

| Term | Meaning | Where MuJoCo stores it |
|---|---|---|
| $M(q)$ | joint-space inertia matrix, symmetric positive definite, depends on the pose | `mjData.M` (sparse; `qM` in older versions), dense copy via `mj_fullM` |
| $C(q,\dot q)\dot q + g(q)$ | Coriolis/centrifugal and gravity forces: the **bias** force $c$ | `mjData.qfrc_bias` |
| $\tau$ | forces you apply: actuators and user forces | `qfrc_actuator`, `qfrc_applied` |
| $J^\top f$ | contact and other constraint forces mapped to the joints | `qfrc_constraint` |

MuJoCo writes the same equation as $M\dot v + c = \tau + J^\top f$ and computes $c$ with the recursive Newton–Euler algorithm at zero acceleration ([Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)). Two calls give you both directions:

- **Forward dynamics** `mj_forward`: given $q, \dot q, \tau$, compute $\ddot q$ (`qacc`). This is what `mj_step` does before integrating.
- **Inverse dynamics** `mj_inverse`: given $q, \dot q, \ddot q$ ("qacc must be set before calling"), compute the force that produces it (`qfrc_inverse`) ([API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)).

With $\dot q = 0$ the bias is pure gravity, so `qfrc_bias` at rest is the gravity-compensation torque you will feed forward in lesson 12. The matrix $M$ is what turns a desired acceleration into torque in lessons 13 and 14.

The second half of the lesson is about time. MuJoCo integrates the equation with one of several schemes: `Euler` (semi-implicit: new velocity first, then position with the new velocity), `RK4`, `implicit` (includes the derivatives of velocity-dependent forces) and `implicitfast` (the same but without the Coriolis/centripetal derivatives; recommended by the docs for most models, although the default is still `Euler`) ([Computation · Numerical integration](https://mujoco.readthedocs.io/en/stable/computation/index.html)). A frictionless pendulum must conserve energy, so energy drift is a clean measure of integration error.

## Experiment

**Predict first.** For a two-link pendulum derived by hand, how close will MuJoCo's `M`, `qfrc_bias` and `mj_inverse` be to the formulas: 1e-3, 1e-8 or machine precision? And which integrator will keep the energy of a swinging double pendulum constant at the default 2 ms step?

**Setup.** Two links swing in the x–z plane on hinges about $y$. Link 1: 1 kg, 40 cm, centre of mass at 20 cm, inertia 0.02 kg·m². Link 2: 0.5 kg, 30 cm, centre of mass at 15 cm, inertia 0.005 kg·m². No joint damping, no friction, no contacts. With $a = m_2 l_1 c_2$:

$$M = \begin{bmatrix} m_1c_1^2 + I_1 + m_2(l_1^2 + c_2^2 + 2 l_1 c_2\cos q_2) + I_2 & m_2(c_2^2 + l_1 c_2 \cos q_2) + I_2 \\ \cdot & m_2 c_2^2 + I_2 \end{bmatrix},\quad C\dot q = a \sin q_2 \begin{bmatrix} -(2\dot q_1 \dot q_2 + \dot q_2^2) \\ \dot q_1^2 \end{bmatrix}$$

```python
d.qpos[:], d.qvel[:], d.qacc[:] = q, qd, qdd
mujoco.mj_inverse(m, d)              # -> d.qfrc_inverse, also fills d.M and d.qfrc_bias
mujoco.mj_fullM(m, d, Mf)            # dense M (MuJoCo 3.17 signature: model, data, dst)
tau = Ma @ qdd + Ca + ga             # the hand-derived torque
d.qfrc_applied[:] = tau
mujoco.mj_forward(m, d)              # must give back qdd
```

1000 random states (seed 0, $q \in [-\pi, \pi]$, $\dot q \in [-5, 5]$ rad/s, $\ddot q \in [-20, 20]$ rad/s²). Then the pendulum is released from horizontal and swings passively for 10 s, for 4 integrators × 6 timesteps. Full script: [`experiments/11-rigid-body-dynamics/experiment.py`](../experiments/11-rigid-body-dynamics/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministic, ~3 s.

**Result 1: MuJoCo matches the formulas to machine precision.**

| Check (max over 1000 states) | max abs. difference |
|---|---|
| `mj_fullM` vs $M(q)$ | 1.39e-16 kg·m² |
| `qfrc_bias` vs $C\dot q + g$ | 1.78e-15 N·m |
| `mj_inverse` vs $\tau$ (largest $|\tau|$ = 9.0 N·m) | 3.55e-15 N·m |
| `mj_forward` acceleration vs $\ddot q$ | 1.88e-13 rad/s² |

**Result 2: the integrator changes the physics.** Energy error over 10 s, in % of the 4.660 J the pendulum can release; each cell is max |E(t) − E₀| / final E − E₀.

| integrator | 0.5 ms | 1 ms | 2 ms | 5 ms | 10 ms | 20 ms |
|---|---|---|---|---|---|---|
| `Euler` | 3.467 / +3.467 | 6.984 / +6.984 | 14.357 / +14.355 | 41.123 / +39.316 | 89.992 / +56.191 | 140.049 / +124.182 |
| `RK4` | 0.000 / −0.000 | 0.000 / −0.000 | 0.000 / −0.000 | 0.000 / −0.000 | 0.003 / −0.001 | 0.037 / −0.014 |
| `implicit` | 3.093 / −3.093 | 6.007 / −6.007 | 11.352 / −11.352 | 24.317 / −24.317 | 39.105 / −39.105 | 55.292 / −54.794 |
| `implicitfast` | identical to `Euler` in every cell | | | | | |

What the table says:

- **Euler gains energy, implicit loses it**, both roughly in proportion to the timestep (halve dt, halve the error: 6.984 → 3.467). At the default 2 ms, the frictionless double pendulum gains 14.357% energy in 10 s.
- **`implicitfast` = `Euler` here.** This model has no joint damping, so the only velocity-dependent force is Coriolis/centripetal, which `implicitfast` drops and `Euler` never had. The two updates become the same.
- **`implicit` dissipates.** Linearising the Coriolis forces adds numerical damping. That is often welcome in a robot with contacts; it is wrong if you are studying energy.
- **RK4 is in another class**: 0.000% (to three decimals) up to 5 ms, at four force evaluations per step.

**Result 3: the same quantities on robots.** With $\dot q = 0$:

| Model, pose | $g(q)$ from `qfrc_bias` [N·m] |
|---|---|
| Panda `home`, joints 1–7 | 0.00, −29.36, 0.00, 22.05, 0.67, 2.31, −0.00 |
| SO-101 stretched forward (all 0), joints 1–5 | 0.000, −0.573, −0.483, −0.130, −0.007 |
| SO-101 folded (lift −1.57, elbow 1.57) | 0.000, −0.066, −0.483, −0.130, −0.007 |

The Panda's diagonal of $M$ at `home` is 1.509, 2.795, 1.471, 1.058, 0.144, 0.154, 0.107 kg·m², of which 0.100 on every joint is the model's `armature` (reflected rotor inertia added to the diagonal). Condition number of $M$: 32.6. A shoulder that feels 2.795 kg·m² and a wrist that feels 0.107 kg·m² will not behave alike under the same gains; that is the subject of lessons 12 and 14.

## Break it

Push the timestep of the semi-implicit Euler run: at 20 ms the passive pendulum ends 10 s with **+124.182%** of its releasable energy and peaks at +140.049%. A frictionless pendulum swinging higher than it was released is not physics; it is integration error. Nothing in the simulation warns you: there is no NaN, no contact, no exception. The only defence is a check like this one: energy (enable it with `<flag energy="enable"/>`, read `d.energy`), or comparing against a smaller step.

The mechanism: semi-implicit Euler is a first-order method that uses the inertia and the Coriolis force from the start of each step. For a system with constant inertia and no velocity-dependent forces it is symplectic and its energy error stays bounded; here $M$ depends on $q_2$ and the Coriolis force is evaluated explicitly, so that structure is lost and the per-step error accumulates instead of oscillating. Its size scales with dt, as the first row shows.

## On the real robot

- **SO-101.** LeRobot drives the STS3215 servos in position mode ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), `set_operating_mode("position", motor)`); there is no torque input, so $M$ and $g$ enter only through the servo's own loop. The $g(q)$ numbers above use the Menagerie model's masses and inertias; they are model outputs, not measured on the real arm.
- **Panda.** libfranka exposes the dynamic model to user code; its Cartesian impedance example reads `model.coriolis(robot_state)` and the Jacobian every cycle and adds the Coriolis torque to its command ([`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp)). The same structure appears in lesson 14.
- **Model error.** Real arms carry cables, gearbox friction and payloads that the URDF/MJCF does not contain. An inverse-dynamics term is only as good as the inertial parameters behind it; identifying them is the subject of [lesson 38 · System identification](../part-9-sim-to-real/38-system-identification.md).

## Check yourself

1. **Easy.** For the double pendulum above, by how much does $M_{11}$ change between $q_2 = 0$ (arm straight) and $q_2 = \pi$ (folded back)?
2. **Medium.** Call `mj_inverse` on the Panda at `home` with `qvel = 0` and `qacc = 0`. What should `qfrc_inverse` equal, and why?
3. **Hard.** Add `damping="0.05"` to both pendulum joints and rerun part B. Predict before running: does `implicitfast` still equal `Euler`? Does the energy now decrease for every integrator?

### Answers

1. $\Delta M_{11} = 4 m_2 l_1 c_2 = 4 \cdot 0.5 \cdot 0.4 \cdot 0.15 = 0.12$ kg·m² (larger when straight).
2. The gravity torque, i.e. `qfrc_bias` from Result 3 (−29.36 N·m on joint 2, 22.05 N·m on joint 4): with zero acceleration, zero velocity and no contacts, the only force needed is the one that cancels gravity; passive damping is zero at zero velocity.
3. Not part of the saved run; run it. Hint from the docs: `Euler` treats joint damping implicitly, and `implicitfast` adds the derivatives of every velocity-dependent force except Coriolis/centripetal. With joint damping as the only such force, both build the same implicit update, so expect them to stay identical. The damping now removes energy; check whether Euler's numerical gain still shows on top of it.

## Takeaways and sources

- $M(q)\ddot q + c(q,\dot q) = \tau + J^\top f$; in MuJoCo, `mj_fullM` gives $M$, `qfrc_bias` gives $c$ (pure gravity at rest), `mj_inverse` gives the torque for a desired acceleration. On a hand-derived double pendulum they match to ~1e-15.
- The integrator is part of the physics: at 2 ms, semi-implicit Euler adds 14.357% energy to a frictionless double pendulum in 10 s, `implicit` removes 11.352%, RK4 stays at 0.000% (three decimals).
- Robot inertia is pose-dependent and very uneven across joints (Panda diagonal 2.795 vs 0.107 kg·m²), which is why one set of gains rarely fits all joints.

Next: [lesson 12 · Joint control](12-joint-control.md).

Sources:
- Lynch, K. M., Park, F. C. [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Cambridge University Press, 2017, ch. 8 (Dynamics of Open Chains).
- MuJoCo documentation: [Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html) (equations of motion, integrators), [API functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) (`mj_inverse`, `mj_forward`, `mj_fullM`), [`mjdata.h`](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h) (`M`, `qfrc_bias`, `qfrc_inverse`).
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): `franka_emika_panda`, `robotstudio_so101`.
- LeRobot [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py); libfranka [`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp) (both read 2026-10-27).
- All measured numbers: [`experiments/11-rigid-body-dynamics/output.txt`](../experiments/11-rigid-body-dynamics/output.txt).
