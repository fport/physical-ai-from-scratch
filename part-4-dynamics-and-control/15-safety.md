# 15 · Safety: limits, torque caps, e-stop and watchdogs

<!-- meta -->
| | |
|---|---|
| **Part** | [4 · Dynamics and control](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [12](12-joint-control.md) |
| **Learn first** | none |
| **Time** | ~40 min |
| **Code** | [`experiments/15-safety/experiment.py`](../experiments/15-safety/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A learned policy will one day send a wrong number. Safety is the layer between that number and the motor that makes the wrong number cheap.

## The problem

LeRobot's SO-101 follower has a setting whose own comment says it exists "for safety purposes": `max_relative_target` limits how far a goal position may be from the present position, and it is off (`None`) by default ([`config_so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py)). The same codebase sets the gripper servo's torque limit to 50% "to avoid burnout". Neither number says what happens when a policy outputs a bad target next to a person or an obstacle. This lesson measures it in simulation: one bad command, several safeguards, and the impact force, speed and torque with and without each.

## Core idea

Safety layers sit between the command and the actuator, from the outside in:

| Layer | What it limits | Typical implementation |
|---|---|---|
| Joint position limit | where the joint may go | clamp the goal to a software range inside the mechanical range |
| Rate / relative-target limit | how fast the goal may change | clamp goal to present ± δ per tick, or limit $|\Delta q_{\text{cmd}}|/\Delta t$ |
| Velocity clamp | joint speed | same as above for position-controlled joints |
| Torque cap | force at contact | actuator force limit; for a servo: its torque-limit register |
| Watchdog | stale commands | if no fresh message within $t_w$: stop and hold |
| E-stop | everything | remove power or command a stop; must not depend on the software it protects |

Two rules shape the design. First, **a limit acts on a variable, not on harm**: a torque cap bounds the steady push force ($F \approx \tau_{\max}/r$ at lever $r$) but not directly the impact peak, which depends on speed and contact stiffness. Second, **what a stopped command means depends on the interface**: a position servo with no new goal holds the last one; a velocity interface with no new command keeps moving.

## Experiment

**Setup.** SO-101 (Menagerie `scene.xml`), each joint the model's STS3215 position servo ($k_p$ = 998.22, $k_v$ = 2.731, cap ±2.94 N·m; lesson 12). Timestep set to 2 ms; the command loop runs at 50 Hz. Arm pose: shoulder_lift −0.5, elbow 0.5, wrist 0.8 rad; gripper frame at (29.4, −0.1, 12.8) cm. A fixed post (2 × 2 × 20 cm) stands in the gripper's swing path at pan ≈ 0.6 rad.

```python
goal = target
if pos_limit is not None:
    goal = np.clip(goal, -pos_limit, pos_limit)                       # software joint limit
if rel_limit is not None:
    goal = np.clip(goal, d.qpos[0] - rel_limit, d.qpos[0] + rel_limit)  # LeRobot-style relative target
d.ctrl[pan] = goal                                  # torque cap: m.actuator_forcerange
```

Full script: [`experiments/15-safety/experiment.py`](../experiments/15-safety/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministic, ~3 s.

**A. One bad command.** At t = 0.1 s the shoulder_pan target jumps 0 → 1.2 rad (a corrupted action). *Predict:* which single safeguard prevents the impact, which only softens it?

| safeguard | peak force on post | steady push force | peak pan speed | peak torque |
|---|---|---|---|---|
| none | 125.98 N | 14.33 N | 4.49 rad/s | 2.94 N·m |
| joint limit \|pan\| ≤ 0.3 rad | 0.00 | 0.00 | 4.02 | 2.94 |
| relative target ≤ 0.02 rad/tick | 45.82 | 14.48 | 1.82 | 2.94 |
| torque cap 0.5 N·m | 21.34 | 2.12 | 0.75 | 0.50 |
| all three | 0.00 | 0.00 | 0.75 | 0.50 |

Only the position limit prevents contact, and only because we knew where the obstacle was. The rate limit cuts the impact (45.82 vs 125.98 N) but not the steady push. The torque cap cuts both.

**B. Sweep the torque cap** (same bad command, nothing else):

| cap [N·m] | 2.94 | 2.00 | 1.00 | 0.50 | 0.25 | 0.10 |
|---|---|---|---|---|---|---|
| peak force [N] | 125.98 | 69.79 | 37.96 | 21.34 | 8.57 | 0.00 |
| steady push [N] | 14.33 | 8.73 | 4.34 | 2.12 | 1.48 | 0.00 |
| peak speed [rad/s] | 4.49 | 3.16 | 1.58 | 0.75 | 0.33 | 0.08 |

The cap also limits speed, because the joint accelerates more slowly; at 0.10 N·m the arm does not reach the post within 2 s. Between 0.5 and 2 N·m the steady push is nearly proportional to the cap (2.12, 4.34, 8.73 N): $F \approx \tau_{\max}/r$ with a lever of about 23 cm.

**C. Sweep the relative-target limit δ** (goal clamped to present ± δ every 20 ms):

| δ [rad] | none | 0.2 | 0.05 | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|---|---|
| speed at impact [rad/s] | 4.494 | 4.494 | 2.866 | 1.513 | 0.929 | 0.359 |
| peak force [N] | 125.98 | 125.98 | 72.94 | 45.82 | 21.39 | 17.17 |
| steady push [N] | 14.33 | 14.33 | 13.54 | 14.48 | 14.67 | 13.54 |

δ = 0.2 rad per tick is no limit at all for this servo. Smaller δ lowers the impact speed and the peak, but the steady push stays near 14 N: once the arm stalls on the post, the goal is still present + δ, and $k_p\,\delta$ ≥ 2.94 N·m for every δ here (998.22 × 0.005 ≈ 5 N·m), so the servo pushes at its cap.

**D. Watchdog.** The operator jogs shoulder_pan at 1 rad/s with velocity commands (the goal advances by $v\,\Delta t$ every tick). The last message arrives at t = 0.18 s with pan at 0.185 rad; the controller keeps using the last velocity unless a watchdog fires and holds the current position.

| timeout | stopped after the last message | max travel after it | peak force on post |
|---|---|---|---|
| none | 0.350 s (stopped by the post) | 19.57° | 24.48 N |
| 1 s | 0.350 s | 19.57° | 24.48 N |
| 0.5 s | 0.350 s | 19.57° | 24.48 N |
| 0.2 s | 0.230 s | 11.61° | 0.00 |
| 0.1 s | 0.130 s | 5.88° | 0.00 |
| 0.05 s | 0.090 s | 3.58° | 0.00 |

A watchdog only helps if it fires before the arm reaches the obstacle: here, timeouts of 0.5 s and 1 s are too slow. Travel after the last message grows with the timeout at roughly the jog speed.

**E. Two ways to stop** from the same 1 rad/s jog at t = 0.2 s:

| stop | max pan travel after the stop | pan stopped after | gripper drop |
|---|---|---|---|
| hold position (goal := present) | 0.15° | 0.030 s | 0.02 cm |
| torque off (actuation disabled) | 1.53° | 0.124 s | 11.74 cm |

Cutting torque stops the pan joint by friction alone and lets gravity take the shoulder: the gripper falls 11.74 cm, onto the floor from 12.8 cm. Holding position stops faster and keeps the arm up, but depends on the controller still working.

## Break it

Use the rate limiter as the only protection and make it generous: δ = 0.2 rad per 20 ms tick equals 10 rad/s, more than the servo ever reaches (4.494 rad/s), so the run is identical to having no limiter (125.98 N). Make it tight instead: δ = 0.005 rad cuts the impact to 17.17 N, yet the arm still pushes on the post with 13.54 N for as long as the bad command lasts. A limiter tuned for normal motion is not a contact-force limit; that job belongs to the torque cap (B) or to a model of where the obstacles are (A).

## On the real robot

- **SO-101 / LeRobot.** `max_relative_target` (default `None`) clamps each goal to present ± cap via `ensure_safe_goal_position` ([`robots/utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/utils.py)), the rule used in part C. At connect, LeRobot writes position-mode P/I/D coefficients 16/0/32 to every STS3215 and, for the gripper only, `Max_Torque_Limit` 500 ("50% of max torque"), `Protection_Current` 250 and `Overload_Torque` 25 ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [`config_so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py)). `disable_torque_on_disconnect = True` means a disconnect behaves like part E's "torque off": expect the arm to drop.
- **Panda.** libfranka's impedance example sets collision thresholds before moving (`setCollisionBehavior`) and its `rate_limiting.h` defines joint acceleration, jerk and torque-rate limits (lesson 13) ([libfranka](https://github.com/frankarobotics/libfranka)).
- **Not measured on the real arm.** Impact forces here depend on MuJoCo's contact model and the 2 ms step. Procedure for the SO-101: put a kitchen scale or a force sensor where the post is, command a slow jog (not a step) into it at reduced torque limit, and record the peak; start with the lowest limit and never put a hand in the path. A hardware e-stop that cuts motor power is separate from all software layers above.

## Check yourself

1. **Easy.** With the torque cap at 1.0 N·m and a lever of 23 cm, what steady push force do you expect? Compare with part B.
2. **Medium.** Why does δ = 0.02 rad/tick give a peak pan speed of 1.82 rad/s in part A, above the 1 rad/s that 0.02 rad per 20 ms suggests?
3. **Hard.** Add a velocity-dependent torque cap (full cap below 0.2 rad/s, 0.5 N·m above) and rerun part A. Predict before running: impact peak and steady push.

### Answers

1. $1.0/0.23 \approx 4.3$ N; part B measures 4.34 N.
2. The goal is updated only every 20 ms and is relative to the *present* position. Within a tick the servo sees an error of δ, and $k_p\delta$ ≈ 20 N·m exceeds its 2.94 N·m cap, so it accelerates at full torque, passes the goal and brakes; the speed oscillates around δ/Δt = 1 rad/s with peaks above it.
3. Not computed here; run it.

## Takeaways and sources

- Each safeguard bounds one variable: position limits prevent known collisions (0 N), torque caps bound the push (14.33 → 2.12 N at 0.5 N·m), rate limits bound the speed at impact (4.494 → 0.359 rad/s) but not the push.
- A watchdog must fire before the arm can reach anything: 0.05–0.2 s avoided the post here, 0.5 s did not.
- "Stop" is a design choice: holding position stops in 0.030 s; torque off drops the gripper 11.74 cm.

Next: [lesson 16 · Grasping and scripted pick-and-place](16-grasping-and-pick-place.md).

Sources:
- LeRobot [`config_so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py), [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [`robots/utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/utils.py) (Apache-2.0, read 2026-10-27).
- libfranka [`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp), [`rate_limiting.h`](https://github.com/frankarobotics/libfranka/blob/main/include/franka/rate_limiting.h).
- MuJoCo [Modeling · actuators and `forcerange`](https://mujoco.readthedocs.io/en/stable/modeling.html), [`mj_contactForce`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html); [Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), ch. 11.
- All measured numbers: [`experiments/15-safety/output.txt`](../experiments/15-safety/output.txt).
