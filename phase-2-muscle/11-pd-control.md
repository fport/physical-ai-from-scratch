# 11 · 12 joints, two numbers: 20 and 0.5

> The AI says "leg to this angle". What pushes it there is a two-number rule.

> **Status: draft.** The experiment has been run, the notebook is ready and the numbers are verified. The text is not yet finished in book format; the I (integral) term is missing from this page.

| | |
|---|---|
| **Phase** | [2 · Muscle](README.md) |
| **Type** | Experiment |
| **Prerequisites** | [10 · Position servo](10-position-servo.md) |
| **Time** | ~45 min |
| **Tools** | MuJoCo 3.17 · Menagerie `unitree_go2` · Colab or Mac |
| **Code** | [notebook](../notebooks/11-go2-two-numbers.ipynb) |
| **Status** | 📝 draft · notebook ready |

## Why it matters

The Unitree Go2 has 12 joints (hip, thigh and knee per leg). In Unitree's open training code, every joint in the training simulation is driven by the same two numbers: `stiffness = 20` (Kp, N·m/rad) and `damping = 0.5` (Kd, N·m·s/rad), `control_type = 'P'` ([`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)). The torque is computed with this line ([`legged_robot.py` · `_compute_torques`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot.py)):

```python
torques = p_gains * (actions_scaled + default_dof_pos - dof_pos) - d_gains * dof_vel
```

The AI policy states the target angle at 50 Hz (`sim.dt = 0.005` × `decimation = 4` = 20 ms), and the PD rule turns it into torque every 5 ms.

**Note:** these numbers are for the **training simulation**. Unitree's repo has no deploy config for the Go2 on the real robot (only G1, H1, H1_2). Unitree's low-level stand-up example for the real Go2 uses **Kp = 60, Kd = 5** ([`go2_stand_example.py`](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/low_level/go2_stand_example.py)).

## Concept

```python
tau = Kp * (q_target - q) - Kd * q_vel
```

- **Kp** turns the error into torque: like a spring. The larger it is, the stiffer the joint is pulled toward the target.
- **Kd** brakes the velocity: like a damper. It damps oscillation and overshoot.
- **The I term** (accumulated error) is absent here. I is the classic way to close the steady sag under load; in Go2 training the policy closes this gap by shifting the target (see Results).

## Experiment (in the notebook)

The Go2 lies on the ground with its legs folded. All 12 joints get the same PD rule for 3 s; torque is clipped at the motor's own limit. The target angles are `default_joint_angles` from Unitree's config. Measurement is via body height. **Overshoot** is measured not against the target but against the height the robot itself settles at by second 3: (peak − final) / (final − start).

## Results in numbers

MuJoCo 3.17.0, 2 ms step, Menagerie `unitree_go2`. The numbers were recomputed from the raw logs and rerun with the same protocol (identical).

| Kp / Kd | Start | Peak | Final (2.5–3 s avg.) | Overshoot | Note |
|---|---|---|---|---|---|
| **20 / 0.5** (Unitree training) | 10.0 cm | 25.6 | **25.2** | 3.0% | All four feet stay on the ground |
| 5 / 0.5 | 9.9 | 11.1 | **11.1** | — | Could not stand up |
| 80 / 0.5 | 10.0 | **50.8** | 29.9 | 104.5% | At the peak (t = 0.33 s) all four feet are in the air; torque saturated |
| 20 / 0 | 10.0 | 26.3 | 24.7 | **11.0%** | Bounces more without the brake |
| 20 / 2 | 10.0 | 25.3 | 25.2 | 1.2% | |
| 60 / 5 (real Go2 example) | | | 29.9 | 0.2% | |

**A 7-centimetre gap.** If the target angles were held exactly, the body would be at about **32 cm** (front leg 32.35, rear 31.39 cm; geometry only). With 20 / 0.5 the body stays at 25.2 cm: **~7 cm too low**. PD alone does not close this gap. In training, the policy closes it by shifting the target (`0.25 · action`).

**What is robust and what is not.** "Kp 5 can't stand up", "Kp 80 launches" and "~7 cm sag at 20 / 0.5" came out independent of the starting pose, the measurement window and the model's passive damping. The overshoot **ratio** between Kd 0 and 0.5, however, is not robust (between 2.2× and 3.7× depending on window and model). Only the qualitative "remove Kd and it bounces more" is robust.

## Limits

- The Menagerie Go2 model gives the joints passive damping (`damping="2"`) and friction loss (`frictionloss="0.2"`); the training URDF has neither. So even in the "Kd = 0" run the joint has a damping of 2, and all oscillations are softer than in the training simulation.
- The timestep is 2 ms (5 ms in training); the knee torque limit is 45.43 N·m (35.55 in the training URDF).
- This is a physics model experiment: no AI, just PD on fixed target angles.
- The Go2's `base_height_target = 0.25` reward is not used in Go2 training (its weight is 0). That 25 cm coincides with the measured 25.2 cm is a coincidence.

## Exercises (with the notebook's "BURAYI DEĞİŞTİR" ("change this") cell)

1. **Easy.** Run with `KP = 60, KD = 5`. Where does the body settle?
2. **Medium.** Put a 5 kg load on the back (`LOAD_KG = 5`). With 20 / 0.5, how many cm does the sag increase?
3. **Medium.** Read [`go2_stand_example.cpp`](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp): are the target angles given all at once, or shifted step by step? If they are shifted, why does a large Kp launch the robot less? Apply this in the experiment.
4. **Hard.** Add an I term to the rule (`Ki * ∫(q_target − q) dt`). Does the 7 cm gap close? How long does it take? What happens to overshoot?

## Sources

- [`unitree_rl_gym` · `go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py), [`legged_robot.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot.py), [`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py) (BSD-3-Clause, © Unitree Robotics).
- [`unitree_rl_gym/deploy`](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy): no Go2 deploy config.
- [`unitree_sdk2` · `go2_stand_example.cpp`](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp): Kp 60, Kd 5.
- [MuJoCo Menagerie · `unitree_go2`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2): derived from Unitree's URDF, folder license BSD-3-Clause (© Unitree Robotics).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Chapter 11 (robot control).
- Numbers from the logs of the Koddan Kasa 11 v3 experiment and the internal verification note (27.10.2026).

> Go2 model: MuJoCo Menagerie (Google DeepMind), from the Unitree Robotics URDF, BSD-3-Clause. The Unitree and Go2 names are used factually; Unitree does not sponsor or endorse this work.
