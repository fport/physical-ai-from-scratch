# The physical AI stack

> A robot is a stack of loops: fast ones that hold a joint where it should be, slower ones that decide where it should be.

A robot that picks up a cube or walks across a room runs several layers of software at once. Each layer has its own job, its own rate and its own failure modes. This book teaches them bottom-up, in the order you need them.

| Layer | Job | Typical tools | Book |
|---|---|---|---|
| Physics and simulation | Predict how bodies move and collide | MuJoCo, MJCF/URDF models | [Part 1](../part-1-simulation/README.md) |
| Geometry | Describe where things are and how they are oriented | Rotation matrices, quaternions, transforms | [Part 2](../part-2-spatial-math/README.md) |
| Kinematics | Map joint angles to the gripper pose and back | FK, Jacobians, IK | [Part 3](../part-3-kinematics/README.md) |
| Low-level control | Turn a target into motor torques, safely | PD/PID, servos, impedance control, limits | [Part 4](../part-4-dynamics-and-control/README.md) |
| State estimation and perception | Turn noisy sensors and pixels into a state | Filters, Kalman filter, camera calibration, detection | [Part 5](../part-5-perception-and-estimation/README.md) |
| Planning and optimal control | Find feasible motions, optimize controls | RRT, trajectory optimization, LQR, MPC | [Part 6](../part-6-planning-and-optimal-control/README.md) |
| Learned policies | Map observations to actions from data or reward | Behavior cloning, ACT, diffusion policies, PPO | [Parts 7–8](../part-7-learning-from-demonstration/README.md) |
| Deployment | Make sim-trained behavior work on hardware | Latency handling, system identification, domain randomization, LeRobot | [Part 9](../part-9-sim-to-real/README.md) |
| Task layer | Decide what to do from language and context | VLAs, LLM agents with robot tools, world models | [Part 10](../part-10-foundation-models-and-agents/README.md) |

## Loops run at different rates

Lower layers run faster than higher ones. Two concrete examples used in this book:

- In Unitree's open training code for the Go2, the learned policy picks new joint targets every 20 ms (50 Hz) while a PD rule turns them into torques every 5 ms (`sim.dt = 0.005`, `decimation = 4` in [`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py)). Lesson [12](../part-4-dynamics-and-control/12-joint-control.md) uses these numbers.
- LeRobot records demonstrations on the SO-101 at about 30 frames per second ([LeRobot docs](https://huggingface.co/docs/lerobot/il_robots)); the servos close their own position loop inside the motor (lesson [12](../part-4-dynamics-and-control/12-joint-control.md)).

## Where ROS 2 fits

On real robots the layers usually run as separate processes that exchange messages; [ROS 2](https://docs.ros.org/) is the most common middleware for that (nodes, topics, services, actions). This book keeps every experiment in a single Python process with MuJoCo so the algorithms stay visible; the same algorithms are what you would put inside ROS 2 nodes. LeRobot, used in Parts 7 and 9, talks to the SO-101 directly over its serial bus.

## The robots in this book

| Robot | What it is | Model |
|---|---|---|
| SO-101 | Low-cost 6-motor arm (5 joints + gripper) used with LeRobot | Menagerie `robotstudio_so101` |
| Franka Panda | 7-DoF research arm, used where redundancy or torque control matters | Menagerie `franka_emika_panda` |
| Unitree Go2 | 12-joint quadruped, used for control and locomotion | Menagerie `unitree_go2` |

All three models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and are fetched by `experiments/common/menagerie.py`.
