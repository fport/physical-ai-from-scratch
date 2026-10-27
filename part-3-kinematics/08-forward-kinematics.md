# 08 · Forward kinematics, degrees of freedom and workspace

<!-- meta -->
| | |
|---|---|
| **Part** | [3 · Kinematics](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [07](../part-2-spatial-math/07-rigid-transforms.md) |
| **Learn first** | [Matrix Transformations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/03-matrix-transformations) |
| **Time** | ~45 min |
| **Code** | [`experiments/08-forward-kinematics/experiment.py`](../experiments/08-forward-kinematics/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> Forward kinematics is a product of transforms along the tree; the number of joints in that product decides which poses exist at all.

## The problem

The SO-101 follower arm uses six STS3215 servos ([LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/so101)). One of them only opens and closes the jaw. That leaves **five** joints to place a gripper whose pose has **six** numbers (3 position, 3 orientation). A policy or a teleoperator can ask for a gripper pose that no joint setting produces, and nothing in the arm will say so.

Before computing joint angles for a target (lesson 10), you need the forward map: joint angles → gripper pose. This lesson builds it by hand from the MJCF file, checks it against MuJoCo, measures the arm's workspace, and shows exactly which orientations the five joints cannot give.

## Core idea

**Forward kinematics (FK)** composes the frame tree of lesson 07 from the base to the tool. In MJCF every body has a fixed offset from its parent (`pos`, `quat`) and, for a hinge, a rotation about an axis given in the body frame. For a serial chain of hinges:

$$T_{W,\text{site}}(q) = \prod_{i=1}^{n}\Big(T^{\text{static}}_{i}\ \mathrm{Rot}(\hat a_i, q_i)\Big)\ T_{\text{site}} ,$$

with $T^{\text{static}}_i$ from `body_pos`/`body_quat`, $\hat a_i$ = `jnt_axis` and the joint at the body origin (`jnt_pos = 0` in this model). Lynch & Park write the same map as a product of matrix exponentials, $T(q) = e^{[\mathcal S_1]q_1}\cdots e^{[\mathcal S_n]q_n}M$ (*Modern Robotics*, Ch. 4); the factors differ, the result does not.

**Degrees of freedom.** For an open chain of 1-DoF joints, the mobility equals the number of joints. A gripper pose lives in a 6-D space, so:

| Arm | Joints moving the tool | Task dimension | Consequence |
|---|---|---|---|
| SO-101 | 5 (pan, lift, elbow, wrist flex, wrist roll) | 6 | at a given point, only a 2-D family of orientations exists |
| Franka Panda | 7 | 6 | one redundant DoF: generically many solutions |

On the SO-101, lift, elbow and wrist flex have **parallel** axes, so they move the tool in one vertical plane whose direction is set by the pan joint; wrist roll spins the tool about its own approach axis. At a fixed position the approach axis must therefore lie in that plane.

**Workspace.** The set of tool positions over all joint values within limits. There is no closed form for real arms; you sample it. Matrix products are covered in AIEFS: [Matrix Transformations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/03-matrix-transformations).

## Experiment

Script: [`experiments/08-forward-kinematics/experiment.py`](../experiments/08-forward-kinematics/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, SciPy, Apple Silicon Mac (Apple M5), deterministic seeds, about 80 s of CPU time.

**Predict first.** How many 1 cm voxels does the SO-101's reachable space occupy, and do 10,000 random joint samples find most of them?

**Step 1 · FK by hand.** The chain is read from the compiled model: base → shoulder → upper_arm → lower_arm → wrist → gripper → site `gripperframe`.

```python
def fk_hand(q):
    M = np.eye(4)
    for Ts, jt in zip(STATIC, JOINT):          # T_static from body_pos / body_quat
        M = M @ Ts
        if jt is not None:
            j, ax = jt
            M = M @ T(axis_rot(ax, q[j]), np.zeros(3))
    return M @ T_SITE                          # site_pos / site_quat
```

Over 1000 random joint vectors within limits, hand FK and `mj_kinematics` agree to **3.3e-16 m** in position and **1.2e-15** per rotation-matrix entry. The tool site is 12.0 mm off the wrist-roll axis, so rolling 0 → 180° moves it 24.0 mm: a detail that matters later for IK.

**Step 2 · DoF.** The model has `nq = 6`, `nv = 6`, 6 actuators:

| Joint | Range (deg) |
|---|---|
| shoulder_pan | −110.0 .. 110.0 |
| shoulder_lift | −100.0 .. 100.0 |
| elbow_flex | −96.8 .. 96.8 |
| wrist_flex | −95.0 .. 95.0 |
| wrist_roll | −157.2 .. 162.8 |
| gripper | −10.0 .. 100.0 |

Five of them move `gripperframe`; the sixth moves only the jaw.

**Step 3 · sweep the number of samples → workspace estimate.** Joint vectors uniform within limits, no collision filtering.

| Samples | volume, 1 cm voxels | volume, 2 cm voxels | convex hull | max reach | site below z = 0 |
|---|---|---|---|---|---|
| 1,000 | 1.0 L | 7.6 L | 234.5 L | 0.471 m | 25.4% |
| 10,000 | 9.5 L | 60.9 L | 269.7 L | 0.473 m | 25.2% |
| 100,000 | 71.8 L | 222.2 L | 283.6 L | 0.473 m | 24.9% |
| 1,000,000 | 230.4 L | 282.9 L | 289.6 L | 0.473 m | 25.1% |

Max reach (measured from the point on the pan axis at shoulder height) converges after 10,000 samples, the volume does not: with 1 cm voxels it is still growing at one million samples, because uniform joint sampling visits the workspace very unevenly. The convex hull converges fastest but overestimates, since it fills every concave region. A quarter of all joint vectors put the tool below the table plane the base stands on.

## Break it

**Required orientation out of the arm plane.** Fix the target point (0.22, 0, 0.08) m. Find the arm plane (normal $n$ = shoulder-lift axis; the three pitch axes are parallel to 4.6e-16). Ask for an approach direction $a = \cos\beta\,a_0 + \sin\beta\,n$, where $a_0$ is in the plane, 45° below horizontal. Solve with a bounded least-squares solver from 12 seeds (lesson 10 builds such solvers):

| β (out of plane) | 0° | 5° | 10° | 20° | 30° | 45° | 60° | 90° |
|---|---|---|---|---|---|---|---|---|
| best approach error | 0.098° | 3.086° | 8.083° | 18.077° | 28.070° | 43.058° | 58.036° | 87.682° |
| position error | 0.000 mm | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The position is always reached; the orientation error grows one-for-one with β, staying slightly below it (3.086° at 5°, 87.682° at 90°). That small gain comes from the 12 mm site offset: rolling the wrist moves the site out of the plane, which lets pan turn slightly. At β = 0 the solver ends with wrist_roll at its joint limit, leaving 0.098°.

**Random orientations.** 200 uniformly random full orientations at the same point (6-D task):

| Arm | median error | within 1° | within 5° | within 10° | within 30° |
|---|---|---|---|---|---|
| SO-101 (5 joints), point (0.22, 0, 0.08) m | 72.0° | 2 / 200 | 5 / 200 | 8 / 200 | 33 / 200 |
| Franka Panda (7 joints), point (0.5, 0, 0.4) m | 0.00° | 181 / 200 | 187 / 200 | 189 / 200 | 199 / 200 |

For the SO-101, an arbitrary gripper orientation is the exception, not the rule. The Panda still misses some targets (max error 33.6°); with joint limits active and only 6 seeds, this run does not separate joint limits from local minima.

## On the real robot

- The LeRobot guide names the six motors shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll and gripper ([LeRobot SO-101](https://huggingface.co/docs/lerobot/so101)), the same six joints as the Menagerie model.
- Real FK is only as good as the link lengths and the joint zero offsets. LeRobot's calibration puts every joint at the middle of its range and then moves each through its full range; the Menagerie model notes a 5-degree calibration offset on the elbow range in its comments ([so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). How far the real gripper lands from the model's FK has not yet been measured on the real arm. Procedure: command 20 poses, measure the gripper tip with a ruler or a camera marker, compare with `mj_kinematics`.
- Datasets that store gripper poses as 6-D actions for a 5-DoF arm contain targets the robot cannot reach exactly. Store joint positions, or check reachability before training.

## Check yourself

1. **Easy.** Why is the `gripper` joint not part of the FK chain to `gripperframe`?
2. **Medium.** Using `fk_hand`, find the joint vector in the 1,000,000-sample run that gives the maximum reach. Is the arm straight?
3. **Hard.** Which single extra hinge, placed where, would let the SO-101 reach any orientation at the test point? Argue with the plane picture, then test it by adding the joint with `MjSpec`.

### Answers

1. The site `gripperframe` is attached to the `gripper` body; the `gripper` joint moves the child body `moving_jaw_so101_v1`, which is not an ancestor of the site.
2. Not computed in this lesson's script. Compare what you find with the stretched pose of lesson 10.
3. A hinge whose axis is not parallel to the pitch axes and not along the approach axis, for example a wrist yaw between wrist flex and wrist roll. It lets the approach axis leave the plane. Not computed here.

## Takeaways and sources

- FK = product of static offsets and joint rotations read from the MJCF; hand version matched `mj_kinematics` to 3.3e-16 m.
- Sampled workspace volume depends on sample count and voxel size (230.4 L vs 282.9 L at one million samples); quote both.
- 5 joints cannot realize a 6-D pose: the out-of-plane orientation error grows with β (87.682° at 90°), and only 2 of 200 random orientations were reachable within 1°, vs 181 of 200 for the 7-joint Panda.

Next: [lesson 09 · Jacobians and differential kinematics](09-jacobians.md).

**Sources**

- K. M. Lynch, F. C. Park, *Modern Robotics*, Ch. 2 "Configuration Space" and Ch. 4 "Forward Kinematics". [Book page](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).
- LeRobot, [SO-101 assembly and setup guide](https://huggingface.co/docs/lerobot/so101).
- MuJoCo Menagerie: [`robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml) (Apache-2.0), [`franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/franka_emika_panda).
- MuJoCo [API functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) (`mj_kinematics`).
