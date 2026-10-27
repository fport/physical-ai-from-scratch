# 07 · Rigid transforms and frame trees

<!-- meta -->
| | |
|---|---|
| **Part** | [2 · Spatial math](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [06](06-rotations.md) |
| **Learn first** | [Vectors, Matrices & Operations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/02-vectors-matrices-operations) · [Matrix Transformations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/03-matrix-transformations) |
| **Time** | ~45 min |
| **Code** | [`experiments/07-rigid-transforms/experiment.py`](../experiments/07-rigid-transforms/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> Every point is a number triple only together with the name of its frame; a transform is how you change that name.

## The problem

The Menagerie SO-101 model carries a wrist camera (`wrist_cam`, vertical field of view 61°, defined on the gripper in [`so101.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). A detector finds a cube in that image. To grasp it, the arm needs the cube in its **base** frame; to close the jaws, in the **gripper** frame; to log it with other robots, in the **world** frame. Each of these is a different triple of numbers for the same physical point, and they change every time a joint moves.

Mixing them up does not crash anything. It silently sends the gripper 10 to 42 cm away from the cube (measured below). This lesson builds the four transforms by hand and checks them against MuJoCo.

## Core idea

A **rigid transform** (pose) of frame $B$ in frame $A$ is a rotation $R_{AB}$ (lesson 06) plus the position $t_{AB}$ of $B$'s origin in $A$. Pack them in a 4×4 **homogeneous transform**:

$$T_{AB} = \begin{bmatrix} R_{AB} & t_{AB} \\ 0 & 1\end{bmatrix},\qquad p_A = T_{AB}\,p_B\ \ (\text{with } p = [x\ y\ z\ 1]^\top).$$

| Operation | Formula | Rule of thumb |
|---|---|---|
| Compose | $T_{AC} = T_{AB}\,T_{BC}$ | inner subscripts cancel |
| Invert | $T_{AB}^{-1} = T_{BA} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1\end{bmatrix}$ | never call a general matrix inverse |
| Change frame of a point | $p_B = T_{AB}^{-1}\,p_A$ | the subscript you want goes on the right |

A robot is a **frame tree**: world → base → shoulder → … → gripper → camera. MuJoCo computes the world pose of every node and stores it in `mjData` ([API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)):

| Object | Position | Orientation | Filled by |
|---|---|---|---|
| body (regular frame) | `xpos` | `xmat`, `xquat` | `mj_kinematics` |
| body inertial frame | `xipos` | `ximat` | `mj_kinematics` |
| site | `site_xpos` | `site_xmat` | `mj_kinematics` |
| camera | `cam_xpos` | `cam_xmat` | `mj_camlight` (part of `mj_forward`) |

Two conventions to memorize. A MuJoCo camera looks along the **−z** axis of its frame, with +x right and +y up in the image ([XML reference: camera](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)). In a `framepos` sensor, `objtype/reftype="body"` means the body's **inertial** frame, while `"xbody"` means its regular frame ([XML reference: framepos](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-framepos)). Vectors and matrix products are covered in AIEFS: [Vectors, Matrices & Operations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/02-vectors-matrices-operations).

## Experiment

Script: [`experiments/07-rigid-transforms/experiment.py`](../experiments/07-rigid-transforms/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Apple Silicon Mac (Apple M5), deterministic, about 2 s.

**Setup.** The SO-101 base is moved with `MjSpec` to (0.10, −0.05, 0) m and yawed 30°, so world ≠ base. A 3 cm cube sits on the table at (0.2905, 0.0600, 0.015) m in world. The arm is in a pose where the wrist camera sees the cube. Four `framepos` sensors report the cube in base (`xbody`), base (`body`), gripper site and camera.

**Predict first.** When you rotate `shoulder_pan`, which of the four coordinate triples of the (unmoving) cube change?

```python
def T_inv(M):
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)

F = {"B": T(d.xmat[base], d.xpos[base]),
     "G": T(d.site_xmat[site], d.site_xpos[site]),
     "C": T(d.cam_xmat[cam], d.cam_xpos[cam])}
cube_in = {k: (T_inv(Tw) @ np.append(p_world, 1))[:3] for k, Tw in F.items()}
```

**Result 1 · hand transforms vs MuJoCo.**

| Frame | cube by hand (m) | max abs diff vs sensor |
|---|---|---|
| world W | (0.2905, 0.0600, 0.0150) | – |
| base B | (0.2200, 0.0000, 0.0150) | 2.8e-17 |
| gripper G (site `gripperframe`) | (0.0806, −0.0290, −0.0205) | 1.4e-17 |
| camera C (`wrist_cam`) | (−0.0064, −0.0011, −0.1508) | 4.1e-17 |

The closed-form inverse matches `np.linalg.inv` to 1.1e-15, and $T_{WB}T_{BG}$ reproduces $T_{WG}$ to 5.6e-17. The camera triple has $z = -0.1508$ m: the cube is 15 cm in front of the lens, almost on the optical axis.

**Result 2 · from camera frame to pixels.** With $f = (H/2)/\tan(\text{fovy}/2) = 407.4$ px at 640×480, the cube centre projects to $u = 302.7$, $v = 242.8$ px ($u = W/2 + f x/(-z)$, $v = H/2 - f y/(-z)$). Projecting the 8 cube corners and filling their convex hull gives a silhouette with IoU 0.979 against MuJoCo's rendered segmentation mask (9643 vs 9454 px), centroid offset 0.74 px. The silhouette, not the centroid, is the right check (see On the real robot).

**Result 3 · sweep `shoulder_pan`.** Other joints fixed; coordinates in cm.

| pan | cube in B | cube in G | cube in C | gripper site in W |
|---|---|---|---|---|
| −40° | 22.00, 0.00, 1.50 | 5.30, 2.62, 11.15 | 12.62, 2.90, −9.83 | 18.88, 9.21, 8.26 |
| 0° | 22.00, 0.00, 1.50 | 8.16, −1.46, −0.19 | 1.23, 1.14, −14.39 | 25.47, 2.79, 8.26 |
| 40° | 22.00, 0.00, 1.50 | 5.30, −10.68, −7.96 | −6.62, −8.08, −17.01 | 26.40, −6.36, 8.26 |

Then every arm joint is swept ±30° around the pose (kinematics only, limits not enforced):

| Joint swept | max change in W | in B | in G | in C | drift of $T_{GC}$ |
|---|---|---|---|---|---|
| shoulder_pan | 0.00 mm | 0.00 mm | 93.77 mm | 93.77 mm | 7.8e-16 |
| shoulder_lift | 0.00 | 0.00 | 93.44 | 93.44 | 7.8e-16 |
| elbow_flex | 0.00 | 0.00 | 150.07 | 150.07 | 7.8e-16 |
| wrist_flex | 0.00 | 0.00 | 124.53 | 124.53 | 5.6e-16 |
| wrist_roll | 0.00 | 0.00 | 15.74 | 15.74 | 5.6e-16 |

Coordinates in a frame change only when a joint lies **between** that frame and the world in the tree. Base coordinates never change; gripper and camera coordinates change by exactly the same distance, because $T_{GC}$ is constant (drift ≤ 7.8e-16) and a rigid transform preserves distances. `wrist_roll` changes the gripper-frame coordinates least (15.74 mm): turning by up to 30° about the roll axis moves them by $2r\sin 15°$, with $r$ the cube's distance from that axis, so the cube is only about 30 mm off it.

## Break it

Five ways to compute the cube in the gripper frame, scored against the sensor:

| Computation | error |
|---|---|
| $T_{WG}^{-1}\,p$ (correct) | 0.0 mm |
| $R\,(p - t)$: R instead of Rᵀ (active/passive confusion) | 101.0 mm |
| $(T_{BG}\,T_{WB})^{-1}\,p$: wrong composition order | 103.2 mm |
| $R^\top p$: forgot the translation | 274.3 mm |
| $T_{WG}\,p$: used the transform instead of its inverse | 424.4 mm |

And the subtle one: a `framepos` sensor with `reftype="body"` reads (−0.0044, 0.0397, 0.2032) m, 295.5 mm away from the true base-frame value, because it uses the base's inertial frame, whose origin is at the centre of mass and whose axes are the principal axes of inertia. None of these errors throws an exception.

## On the real robot

- On hardware there is no `framepos` sensor; $T_{WB}$ (where the base is bolted) and $T_{GC}$ (where the camera sits on the gripper) must be **measured** by calibration (see [lesson 19](../part-5-perception-and-estimation/19-camera-calibration.md)). The model's $T_{GC}$ is a CAD value. Not yet measured on the real arm.
- OpenCV's pinhole model projects $u = f_x X_c/Z_c + c_x$, $v = f_y Y_c/Z_c + c_y$ with pixel rows growing downward, i.e. +z forward and +y down ([calib3d.hpp](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)). A MuJoCo camera-frame point becomes an OpenCV one by a 180° rotation about x: $(x, y, z) \to (x, -y, -z)$. Getting this wrong flips the image.
- Do not use blob centroids as projected centres. In Result 2 the rendered mask centroid is $u = 299.3$, $v = 240.3$ px, while the centre projects to $u = 302.7$, $v = 242.8$ px: under perspective, the centroid of a solid's image is not the image of its centre. For calibration use corners or keypoints.

## Check yourself

1. **Easy.** In Result 3, why is the gripper site's world z constant (8.26 cm) for every pan angle?
2. **Medium.** Write $T_{BC}$ in terms of the `T_inv` and the dictionary `F`. Which joints does it depend on?
3. **Hard.** Add a `framequat` sensor for the cube with `reftype="camera"` and verify $R_{CW}R_{W,\text{cube}}$ by hand. What is the max error?

### Answers

1. The pan axis is vertical (base is only yawed), so rotating about it keeps every point at its height.
2. $T_{BC}$ = `T_inv(F["B"]) @ F["C"]`. It depends on all five arm joints, and on none of the base placement.
3. Not computed in this lesson's script; compare your error with the position errors of Result 1.

## Takeaways and sources

- $p_B = T_{AB}^{-1}p_A$ with the closed-form inverse; in this setup hand transforms matched MuJoCo to ≤ 4.1e-17 m.
- A joint changes coordinates only in frames downstream of it in the tree; base coordinates of the cube stayed at 0.00 mm change while gripper/camera coordinates moved up to 150.07 mm.
- Know the frame conventions: camera −z forward (MuJoCo), `body` = inertial frame vs `xbody` = regular frame (295.5 mm trap).

Next: [lesson 08 · Forward kinematics, degrees of freedom and workspace](../part-3-kinematics/08-forward-kinematics.md).

**Sources**

- K. M. Lynch, F. C. Park, *Modern Robotics*, Ch. 3 "Rigid-Body Motions" (homogeneous transforms). [Book page](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).
- MuJoCo: [XML reference · camera](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera), [XML reference · framepos](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-framepos), [API functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).
- SO-101 model: [MuJoCo Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml) (Apache-2.0).
