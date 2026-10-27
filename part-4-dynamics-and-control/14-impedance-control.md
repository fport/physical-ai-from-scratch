# 14 · Operational-space and impedance control

<!-- meta -->
| | |
|---|---|
| **Part** | [4 · Dynamics and control](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [11](11-rigid-body-dynamics.md), [12](12-joint-control.md) |
| **Learn first** | [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) |
| **Time** | ~60 min |
| **Code** | [`experiments/14-impedance-control/experiment.py`](../experiments/14-impedance-control/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> Position control asks "where is the hand?". Impedance control asks "how hard should the hand push back when it is not there?"

## The problem

A stiff position controller that meets an unexpected table keeps pushing until something gives: the motor torque limit, the part, or the table. Tasks such as wiping, inserting a peg or pressing a button need the opposite: a hand that behaves like a spring of chosen stiffness. libfranka's Cartesian impedance example does exactly this on a Panda with a translational stiffness of 150 N/m and a rotational stiffness of 10 N·m/rad, computing joint torques from the Jacobian every cycle ([`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp)). This lesson builds that controller and its operational-space relative on a torque-controlled Panda in MuJoCo, then pushes the hand into a rigid surface to see what stiffness really buys.

## Core idea

Let $x$ be the pose of the tool centre point (TCP), $\dot x = J(q)\dot q$ its 6-D velocity (lesson 09), $e$ the pose error (position, and orientation as $\tfrac12\sum_i R_i \times R_{d,i}$).

**Task-space PD / impedance.** Pick a desired wrench from a virtual spring–damper and map it to joints with $J^\top$ (the principle of virtual work):

$$\tau = J^\top \big(K e - D\dot x\big) + g(q) \;(+\, C\dot q)$$

At rest against an obstacle, the hand pushes with $F = K e$: you set the force by choosing $K$ and how far the target sits beyond the surface (Hogan 1985).

**Operational space.** The arm's inertia as felt at the hand is

$$\Lambda(q) = \big(J M^{-1} J^\top\big)^{-1}$$

and it differs by direction and pose. Commanding $F = \Lambda\,(k_p e - k_d \dot x)$ makes the hand behave like a unit mass in every direction, so one pair of gains gives the same response along x, y and z (Khatib 1987; Lynch & Park ch. 11). A full implementation also cancels the task-space Coriolis term; here $C\dot q + g$ is cancelled in joint space with `qfrc_bias`.

**Null space.** A 7-joint arm controlling a 6-D pose has one joint-space direction that does not move the hand. Torques projected with

$$N = I - J^\top \bar J^\top, \qquad \bar J = M^{-1}J^\top\Lambda,$$

act only in that direction; a posture term $N\,(k_0 (q_0 - q) - d_0\dot q)$ keeps the elbow where you want it without disturbing the hand.

**What we changed in the model.** Menagerie's Panda has position-servo actuators. We set their `gainprm` and `biasprm` to zero (they produce no force), write our torques to `qfrc_applied`, and clip them at the model's `forcerange` (87 N·m joints 1–4, 12 N·m joints 5–7). With `MjSpec` we add a `tcp` site 10.34 cm in front of the flange and a probe sphere of 1 cm radius just ahead of it; for part B we add a rigid block under the hand, and only the probe can collide with it (`contype = conaffinity = 2`).

## Experiment

Full script: [`experiments/14-impedance-control/experiment.py`](../experiments/14-impedance-control/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministic, ~3 s. Start pose: IK from `home` to TCP (50, 0, 30) cm, hand pointing down (q = 0.0, −0.11, −0.0, −2.273, −0.0, 2.163, −0.785).

```python
Lam = np.linalg.inv(J @ Minv @ J.T)
if mode == "impedance":
    F = np.concatenate([K * e[:3] - D * v[:3], Kr * e[3:] - Dr * v[3:]])
else:                                            # operational space
    F = Lam @ (kp * e - kd * v)
tau = J.T @ F + d.qfrc_bias[:7]
tau = np.clip(tau, -lim, lim)
```

**A. 5 cm steps along x, y, z** (2 s each). Impedance: K = 600 N/m, D = 50 N·s/m. Operational space: $k_p$ = 100 s⁻², $k_d$ = 14 s⁻¹. *Predict:* which controller gives the same response in all three directions?

| controller | axis | $\Lambda_{ii}$ | overshoot | time into ±2 mm | off-axis peak |
|---|---|---|---|---|---|
| impedance | x | 11.87 kg | 13.0% | 0.854 s | 5.43 mm |
| impedance | y | 5.10 kg | 9.8% | 0.532 s | 1.02 mm |
| impedance | z | 6.88 kg | 17.0% | 0.572 s | 5.61 mm |
| operational space | x | 11.87 kg | 3.2% | 0.322 s | 1.70 mm |
| operational space | y | 5.10 kg | 3.8% | 0.304 s | 0.68 mm |
| operational space | z | 6.88 kg | 4.6% | 0.500 s | 2.97 mm |

The hand feels 11.87 kg along x but 5.10 kg along y. With one K and one D, impedance control is a different spring–mass system per direction (settling 0.532–0.854 s) and couples the axes (5.61 mm off-axis). Scaling by Λ makes the responses close (3.2–4.6% overshoot) and the off-axis motion smaller. The residual differences come from what this controller leaves out (task-space Coriolis, joint damping, torque clipping).

**A2. Null space.** Operational-space control holds the TCP still; a 20 N push along +y hits the elbow (`link4`) for 0.3 s.

| controller | TCP error (max) | elbow displacement at the end | $\|q - q_0\|$ at the end |
|---|---|---|---|
| no null-space term | 14.13 mm | 0.30 cm | 0.124 rad |
| + null-space posture | 14.14 mm | 0.00 cm | 0.000 rad |

Without the posture term nothing restores the redundant direction: the arm ends in a different configuration with the hand in the same place. The projected term brings it back without changing the hand error.

**B. Push into a surface.** The target ramps (quintic, 1 s) from 3 cm above the surface to 5 cm below it and holds for 1.5 s. Impedance control, $D = 2\sqrt{K\Lambda_{zz}}$, $K_r$ = 30 N·m/rad. *Predict:* is the holding force $K \times 5$ cm for every K?

| K [N/m] | K × 5 cm | force, holding | peak force | penetration | steps saturated | contact lost | max tilt |
|---|---|---|---|---|---|---|---|
| 100 | 5.0 N | 5.00 N | 54.7 N | 0.545 mm | 0.0% | 0 | 0.46° |
| 300 | 15.0 | 14.98 | 74.7 | 0.708 | 0.0% | 0 | 0.56° |
| 1000 | 50.0 | 49.77 | 89.5 | 0.823 | 0.0% | 0 | 0.59° |
| 3000 | 150.0 | 148.45 | 148.5 | 0.888 | 0.0% | 0 | 0.60° |
| 10000 | 500.0 | 483.60 | 1028.1 | 1.823 | 77.9% | 0 | 56.91° |
| 30000 | 1500.0 | 395.95 | 1132.7 | 19.180 | 80.8% | 6 | 96.27° |
| 100000 | 5000.0 | 205.03 | 1245.0 | 21.679 | 81.8% | 10 | 111.22° |

Up to K = 3000 the holding force matches K × 5 cm (148.45 N against 150 N at K = 3000): **the force is designed, not measured.** Penetration stays below 1 mm and is set by MuJoCo's soft-contact parameters, not by the stiffness you chose: it grows only from 0.545 to 0.888 mm while the force grows about 30 times. The peak force at low K (54.7 N at K = 100, against a 5.00 N holding force) is the impact: a hand with $\Lambda_{zz}$ = 6.88 kg arriving at speed is stopped by the stiff contact, whatever spring you designed behind it.

## Break it

Raise K past what the joints can deliver. At K = 10000 the commanded wrench needs more torque than some joint can give in 77.9% of the steps. Per-joint clipping keeps each torque legal but changes the direction of the wrench actually applied, and the orientation loop loses: the hand tilts 56.91°. At K = 30000 and 100000 the tilt passes 90°, the probe bounces on and off the surface (6 and 10 contact losses during the hold), the holding force falls to 395.95 and 205.03 N, and the measured penetration (19.180 and 21.679 mm, along the now-tilted probe) is no longer a contact property at all. The lesson: a stiffness is only real while the torque limits are not reached. When they are, scale the whole command or limit the wrench in task space, rather than clipping joints independently; lesson 15 returns to torque limits as a safety tool.

## On the real robot

- **Panda.** libfranka's example uses $K$ = 150 N/m and 10 N·m/rad with damping $2\sqrt{K}$ (unit mass), and sends $\tau_d = J^\top(-K e - D J\dot q) + $ `coriolis`; it adds no gravity term to the command ([`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp)). It also sets the robot's collision thresholds before starting (`setCollisionBehavior`). Our K = 100–300 rows are in that range; the numbers above are simulation results, not Panda measurements.
- **SO-101.** LeRobot puts the STS3215 servos in position mode ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)); there is no joint-torque command, so the controller of this lesson cannot run on it as written. Compliance there comes from the servo's own gains and torque limit (lesson 12, part C).
- **Contact models differ.** Real contact stiffness, friction and the robot's own joint elasticity shape the impact peak; the 54.7 N spike is a property of MuJoCo's soft contact at a 2 ms step and should not be read as a real force.

## Check yourself

1. **Easy.** With K = 1000 N/m, how far behind a surface must the target sit for a 20 N press?
2. **Medium.** In part A2 the TCP error is 14.13 mm without and 14.14 mm with the null-space term. Why can the null-space term not reduce it?
3. **Hard.** In part B, replace per-joint clipping by a uniform scale $\tau \leftarrow \tau \cdot \min(1, \min_i \tau_{\max,i}/|\tau_i|)$ and rerun K = 10000. Predict before running: does the tilt disappear? What happens to the force?

### Answers

1. $e = F/K$ = 20 / 1000 = 2 cm.
2. With the dynamically consistent projection $N = I - J^\top \bar J^\top$, null-space torques produce no acceleration of the hand, so they can neither cause nor correct a hand error. The 14 mm comes from the elbow push itself and is resisted only by the task-space part of the controller.
3. Not computed here; run it.

## Takeaways and sources

- $\tau = J^\top F + g$ turns a designed task-space spring into joint torques; against a surface the force is $K e$ (5.00 / 14.98 / 49.77 / 148.45 N for K = 100–3000 at 5 cm).
- The hand's inertia $\Lambda$ depends on direction (11.87 vs 5.10 kg here); scaling by $\Lambda$ (operational space) gives uniform responses; null-space projection controls the redundant elbow without moving the hand.
- Stiffness ends at the torque limits: past them per-joint clipping tilts the hand (56.91° at K = 10000) and the force no longer follows K.

Next: [lesson 15 · Safety](15-safety.md).

Sources:
- O. Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation", *IEEE Journal on Robotics and Automation* 3(1):43–53, 1987, [doi:10.1109/JRA.1987.1087068](https://doi.org/10.1109/JRA.1987.1087068).
- N. Hogan, "Impedance Control: An Approach to Manipulation: Part I—Theory", *Journal of Dynamic Systems, Measurement, and Control* 107(1):1–7, 1985, [doi:10.1115/1.3140702](https://doi.org/10.1115/1.3140702).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), ch. 11 (Robot Control: task-space motion, force and impedance control).
- libfranka [`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp); LeRobot [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py) (read 2026-10-27).
- MuJoCo: [`mj_jacSite`, `mj_fullM`, `mj_contactForce`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html); [Menagerie `franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/franka_emika_panda).
- All measured numbers: [`experiments/14-impedance-control/output.txt`](../experiments/14-impedance-control/output.txt).
