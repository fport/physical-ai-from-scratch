# 19 · Camera model, calibration and hand-eye calibration

<!-- meta -->
| | |
|---|---|
| **Part** | [5 · Perception and estimation](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [07](../part-2-spatial-math/07-rigid-transforms.md) |
| **Learn first** | [Image Fundamentals — Pixels, Channels, Color Spaces](https://aiengineeringfromscratch.com/lesson?path=phases/04-computer-vision/01-image-fundamentals) · [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) |
| **Time** | ~60 min |
| **Code** | [`experiments/19-camera-calibration/experiment.py`](../experiments/19-camera-calibration/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A camera is a 3×3 matrix, a few distortion numbers and a pose; calibration recovers them only if your data makes them observable.

## The problem

Every pixel-to-table computation (lesson 20) and every grasp from vision (lesson 21) multiplies by the camera matrix $K$ and the camera pose. A focal length that is 1 % off scales every lateral offset computed from the image by 1 %; a wrong camera-to-gripper transform shifts every grasp by the same error. OpenCV's `calibrateCamera` returns a small reprojection error almost always, including when $K$ is wrong. This lesson calibrates a MuJoCo camera whose true $K$ is known, so you can see when the error number lies.

## Core idea

**Pinhole model** (see Hartley & Zisserman, *Multiple View Geometry*, ch. on camera models). A point $X_c = (x, y, z)$ in the camera frame (OpenCV convention: $x$ right, $y$ down, $z$ forward) projects to

$$\begin{bmatrix}u\\v\\1\end{bmatrix} \sim K\begin{bmatrix}x/z\\y/z\\1\end{bmatrix},\qquad K = \begin{bmatrix}f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0&0&1\end{bmatrix},\qquad X_c = R\,X_w + t$$

$K$ are the **intrinsics**, $(R, t)$ the **extrinsics**. Lens **distortion** is applied to the normalized coordinates before $K$; the radial part is $x_d = x_n(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$, plus tangential terms $p_1, p_2$ ([OpenCV calib3d](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)).

**K from MuJoCo.** A camera's `fovy` is the vertical field of view in degrees; the horizontal one follows from the image size ([MuJoCo XML · camera/fovy](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera-fovy)). So pixels are square and

$$f_x = f_y = \frac{H/2}{\tan(\mathrm{fovy}/2)},\qquad c_x = \frac{W-1}{2},\quad c_y = \frac{H-1}{2}$$

(the $-1$ is OpenCV's convention that pixel centres sit on integer coordinates). MuJoCo cameras look along their $-z$ axis with $y$ up, so $R_{\text{OpenCV}} = R_{\text{MuJoCo}}\,\mathrm{diag}(1,-1,-1)$.

**Calibration.** `calibrateCamera` follows Zhang's planar method ([Zhang 2000](https://doi.org/10.1109/34.888718)): initial intrinsics from the plane-to-image homographies, initial board poses with `solvePnP`, then Levenberg–Marquardt on the total reprojection error ([OpenCV calib3d docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)).

**Hand-eye.** For a camera on the gripper, the unknown is $X$, the camera pose in the gripper frame. With gripper poses $G_i$ (forward kinematics) and board poses $C_i$ in the camera (PnP), the board is fixed in the base, $G_i X C_i = G_j X C_j$, which gives

$$A_{ij}X = XB_{ij},\qquad A_{ij} = G_j^{-1}G_i,\quad B_{ij} = C_jC_i^{-1}$$

Park & Martin's closed form ([Park & Martin 1994](https://doi.org/10.1109/70.326576)): with $\alpha = \log R_A$, $\beta = \log R_B$, $M = \sum\beta\alpha^\top$, $R_X = (M^\top M)^{-1/2}M^\top$; then $t_X$ from the stacked linear system $(R_A - I)\,t_X = R_X t_B - t_A$. OpenCV's `calibrateHandEye` offers this and four other methods, and needs at least two motions with non-parallel rotation axes.

## Experiment

**Setup.** A 9×6 inner-corner checkerboard with 25 mm squares, built from box geoms (exact geometry, no texture mapping), moved as a mocap body to random poses (±35° tilt about x and y, ±20° about z) 0.5–0.65 m in front of a 640×480 camera with `fovy = 50`. Ground truth: $f_x = f_y = 514.68$ px, $c_x = 319.5$, $c_y = 239.5$. Corners from `findChessboardCorners` + `cornerSubPix`. 60 training views, 15 held-out views. Held-out error = reproject with `solvePnP` under the estimated $K$ and distortion.

```python
r.update_scene(d, camera="cam"); img = r.render()
ok, c = cv2.findChessboardCorners(gray, (9, 6))
c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), crit)
rms, K, dist, _, _ = cv2.calibrateCamera([objp] * n, views, (640, 480), None, None)
```

**Predict.** Few views should give a small training error but a wrong $K$.

**Run** `python experiments/19-camera-calibration/experiment.py` (MuJoCo 3.17.0, OpenCV 5.0.0, NumPy 2.5.4, Apple M5, deterministic: two runs gave identical output).

The renderer is a pinhole: detected corners sit 0.220 px (mean), 0.525 px (max) from the projection with the ground-truth $K$. With all 60 views: $f_x = f_y = 514.58$, $c_x = 319.57$, $c_y = 239.33$, $k_1 = +0.0005$, training RMS 0.091 px.

| Views | Train RMS (px) | Held-out RMS (px) | $f_x$ error (%) | $f_y$ error (%) | $c_x$ error (px) | $c_y$ error (px) |
|---|---|---|---|---|---|---|
| 3 | 0.088 | 0.103 | 0.292 | 0.230 | 1.17 | 0.89 |
| 5 | 0.089 | 0.101 | 0.145 | 0.146 | 0.93 | 0.54 |
| 10 | 0.093 | 0.101 | 0.066 | 0.064 | 0.41 | 0.30 |
| 20 | 0.093 | 0.101 | 0.043 | 0.046 | 0.40 | 0.24 |
| 40 | 0.091 | 0.101 | 0.021 | 0.025 | 0.23 | 0.18 |

(median of 10 random subsets per row)

**Observe.** The training RMS is flat (0.088–0.093 px) while the focal error drops about 14-fold from 3 to 40 views. The reprojection error measures how well the model fits these corners, not how right $K$ is. The held-out RMS is just as flat (0.101–0.103 px): `solvePnP` re-fits each held-out pose and absorbs a small focal error, so it only flags gross failures (2.441 px in the tilt test below). Ground truth is the only honest check; on a real camera, add the parameter standard deviations that `calibrateCameraExtended` returns.

**Distortion.** We warp 20 views with a known $k_1 = -0.15$ (`cv2.remap`). Calibration recovers $k_1 = -0.1558$ with 0.096 % focal error. Forcing distortion to zero on the same views raises the RMS to 0.252 px and the focal error to 1.24 %: an unmodeled lens bends $f$.

**Hand-eye.** SO-101 gripper poses from forward kinematics at random joint angles (60 % of each range); $X$ is the Menagerie `wrist_cam` pose in the gripper frame (offset 76.0 mm). Noise is added to the board-in-camera poses, 15 poses per solve, 20 trials. `cv2.calibrateHandEye` is not exposed in the OpenCV 5.0.0 Python build used here, so the script implements Park & Martin directly (`hand_eye_park`).

| Rotation noise (deg) | Translation noise (mm) | $X$ rotation error (deg) | $X$ translation error (mm) |
|---|---|---|---|
| 0 | 0 | 0.000 | 0.00 |
| 0.1 | 0.5 | 0.050 | 0.31 |
| 0.2 | 1 | 0.105 | 0.68 |
| 0.5 | 2 | 0.177 | 1.55 |
| 1 | 5 | 0.389 | 2.84 |
| 2 | 10 | 0.905 | 6.87 |

Exact data gives exact $X$; with 105 motion pairs the error stays below the per-pose noise.

## Break it

**Remove the tilt.** 10 views, tilt limited to ±t:

| Tilt (deg) | Train RMS (px) | Held-out RMS (px) | $f_x$ error (%) |
|---|---|---|---|
| 0 | 0.088 | 2.441 | 1193 |
| 2 | 0.087 | 0.179 | 5.61 |
| 5 | 0.083 | 0.114 | 1.93 |
| 15 | 0.092 | 0.102 | 0.48 |
| 35 | 0.091 | 0.101 | 0.02 |

With fronto-parallel boards the training RMS is still 0.088 px, and the focal length is off by 1193 %. Mechanism: for a plane parallel to the image, $u = f_x X/Z + c_x$ only constrains $f_x/Z$; a longer focal length and a farther board explain the same corners. Tilt creates perspective foreshortening across the board, which separates $f$ from $Z$. OpenCV's own `calibrateCamera` docs ask for "multiple board poses with significant tilt" ([calib3d.hpp](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)).

**Remove the rotation diversity** (hand-eye, noise 0.5° / 2 mm, joints sampled in a fraction of their range; a fresh random pose pool per row):

| Range fraction | Median rotation between poses (deg) | $X$ rotation error (deg) | $X$ translation error (mm) |
|---|---|---|---|
| 0.02 | 3.6 | 4.818 | 27.18 |
| 0.05 | 9.2 | 2.200 | 13.68 |
| 0.1 | 18.8 | 0.977 | 4.89 |
| 0.3 | 54.8 | 0.348 | 1.85 |
| 0.6 | 116.4 | 0.274 | 1.33 |

Translation enters through $(R_A - I)\,t_X$; when the arm barely rotates, $R_A - I \approx 0$ and noise is divided by a near-zero matrix.

## On the real robot

- The Menagerie SO-101 has a `wrist_cam` with `fovy = 61` in its model file ([robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101)); a real webcam has its own $K$ and distortion. Not yet measured on the real arm.
- Intrinsics procedure: print a checkerboard, measure the square size with calipers (scale error goes straight into $t$), glue it flat, take 20+ views with strong tilt and different distances, check held-out error and parameter standard deviations, not only the RMS.
- Hand-eye procedure: fix the board on the table, move the arm through 15+ poses with large wrist rotations, record joint angles (forward kinematics, [lesson 08](../part-3-kinematics/08-forward-kinematics.md)) and the PnP board pose for each, then solve $AX = XB$. Joint-angle errors and servo backlash enter $A$ directly.
- Autofocus and auto-zoom change $K$; lock them before calibrating.

## Check yourself

1. **Easy.** Compute $f_y$ for a 1280×720 image and `fovy = 61`.
2. **Medium.** Why is a 0.09 px training RMS no evidence of a good calibration? Point to the two tables.
3. **Hard.** Set `CALIB_FIX_ASPECT_RATIO` and `CALIB_FIX_PRINCIPAL_POINT` and repeat the 0° tilt row. Does fixing parameters rescue the degenerate case?

### Answers

1. $f_y = 360/\tan(30.5°)$; $= 611.2$ px.
2. The views sweep shows flat training RMS while $f_x$ error changes 14-fold; the tilt sweep shows 0.088 px with a 1193 % focal error. RMS measures fit, not observability.
3. Not computed here. Fixing $c_x, c_y$ and the aspect ratio does not remove the $f/Z$ ambiguity of parallel planes; run it to see.

## Takeaways and sources

- In MuJoCo, $K$ follows from `fovy` and the image size; use it as ground truth to test your calibration pipeline.
- Calibration accuracy comes from view diversity (tilt, distance, rotation), not from a small reprojection error.
- Hand-eye needs large, non-parallel rotations; its translation error grows roughly like $1/\|R_A - I\|$.

Next: [lesson 20](20-pixels-to-table.md), from pixels to the table.

Sources:
- Z. Zhang, "A flexible new technique for camera calibration", *IEEE TPAMI* 22(11):1330–1334, 2000. [doi:10.1109/34.888718](https://doi.org/10.1109/34.888718)
- F. C. Park, B. J. Martin, "Robot sensor calibration: solving AX=XB on the Euclidean group", *IEEE Trans. Robotics and Automation* 10(5):717–721, 1994. [doi:10.1109/70.326576](https://doi.org/10.1109/70.326576)
- [OpenCV calib3d docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) and [source header](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp): `calibrateCamera` algorithm, distortion model, `calibrateHandEye` methods and minimum motions.
- [MuJoCo XML reference · camera](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera-fovy): `fovy` semantics.
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004 ([book page](https://www.robots.ox.ac.uk/~vgg/hzbook/)).
- All numbers above: [`experiment.py`](../experiments/19-camera-calibration/experiment.py), output in [`output.txt`](../experiments/19-camera-calibration/output.txt).
