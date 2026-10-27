# 20 · From pixels to the table: homography and depth back-projection

<!-- meta -->
| | |
|---|---|
| **Part** | [5 · Perception and estimation](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [19](19-camera-calibration.md) |
| **Learn first** | [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) · [Monocular Depth & Geometry Estimation](https://aiengineeringfromscratch.com/lesson?path=phases/04-computer-vision/26-monocular-depth) |
| **Time** | ~45 min |
| **Code** | [`experiments/20-pixels-to-table/experiment.py`](../experiments/20-pixels-to-table/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A pixel is a ray; you need one more fact to turn it into a point: a known plane (homography) or a measured depth.

## The problem

The detector in lesson 21 will say "the cube is at pixel (u, v)". The arm needs (x, y, z) in its base frame. A pixel only fixes a ray from the camera centre; every point on that ray lands on the same pixel. Two standard ways to pick the point: assume the object lies on the table plane and use a homography, or read the depth image and back-project. Both are easy to get subtly wrong: a homography is silently wrong for anything above the plane, and a depth image is silently wrong if you misread what its values mean.

## Core idea

**Back-projection** (with $K$ from [lesson 19](19-camera-calibration.md)). For pixel $(u,v)$ with depth $Z$ measured **along the optical axis**:

$$X_c = Z\,K^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix},\qquad X_w = R_{wc}X_c + c$$

where $R_{wc}$ has the OpenCV camera axes as columns and $c$ is the camera centre. If $Z$ were the Euclidean distance along the ray instead, you would have to divide by $\|K^{-1}[u,v,1]^\top\|$ first.

**Depth from MuJoCo.** The GPU stores a non-linear z-buffer value $d\in[0,1]$, not metres. With near/far clipping distances $n = \texttt{znear}\cdot\text{extent}$ and $f = \texttt{zfar}\cdot\text{extent}$ (defaults 0.01 and 50; [MuJoCo XML · visual/map](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-map-znear)), a conventional OpenGL buffer gives $Z = \dfrac{nf}{f - d\,(f-n)}$. MuJoCo 3.17's Python `Renderer` reads a reversed buffer (1 at near, 0 at far) and applies the equivalent $Z = \dfrac{nf}{n + d\,(f-n)}$ in float64, then returns metres as float32 ([`rendering/classic/renderer.py`, tag 3.17.0](https://github.com/google-deepmind/mujoco/blob/3.17.0/python/mujoco/rendering/classic/renderer.py)). So `Renderer.enable_depth_rendering()` already returns metric depth along the optical axis; if you read the raw buffer yourself (C API `mjr_readPixels`), you must linearize it. The docs warn that a near plane too close costs depth resolution.

**Homography.** Points on a plane map to the image by a 3×3 projective transform. For the table $z=0$: $s\,[u,v,1]^\top = K[r_1\ r_2\ t]\,[x,y,1]^\top$, so $[x,y,1]^\top \sim H\,[u,v,1]^\top$. $H$ has 8 degrees of freedom; 4 point pairs (no 3 collinear) determine it, more pairs give a least-squares fit (`cv2.findHomography`, method 0; RANSAC/LMEDS/RHO for outliers, [OpenCV calib3d](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)). No $K$, no depth needed, only known table points. The price: the formula is only true for $z = 0$.

| Method | Needs | Valid for | Fails when |
|---|---|---|---|
| Homography | 4+ known table points | points on the plane | object has height, table not flat |
| Depth back-projection | $K$, camera pose, depth image | any visible surface | depth misread, depth noise, missing depth |

## Experiment

**Setup.** MuJoCo scene: grey table plane, four boxes 10/20/50/100 mm tall, a 640×480 camera (`fovy = 50`) tilted 30° from vertical, aimed at (0.25, 0, 0). Ground truth for a pixel is the exact intersection of its ray with the surface it sees. Homography markers: 4 or 12 known table points inside a 20 × 20 cm patch, their pixels perturbed by Gaussian noise $\sigma$; test pixels are random table pixels across the whole view, also perturbed. Depth back-projection uses the rendered depth with bilinear lookup.

```python
r.enable_depth_rendering(); depth = r.render()            # metres along the optical axis
P = c + depth_at(uv)[:, None] * (np.c_[uv, ones] @ np.linalg.inv(K).T @ R_wc.T)
H, _ = cv2.findHomography(marker_px, marker_xy, 0)       # pixel -> table (x, y)
```

**Predict.** Depth is exact in simulation, so its error should be pixel noise times the pixel footprint. The homography should add its own fitting error and break above the plane.

**Run** `python experiments/20-pixels-to-table/experiment.py` (MuJoCo 3.17.0, OpenCV 5.0.0, NumPy 2.5.4, Apple M5, deterministic). On this Mac MuJoCo prints a stderr warning that `ARB_clip_control` is unavailable and depth accuracy will be limited; the measurements below are what you get anyway.

**What is in the depth image** (camera at 0.6 m; extent 0.497 m, so near 5.0 mm, far 24.9 m; 300 table pixels):

| Comparison | Median | Max |
|---|---|---|
| rendered depth vs z-depth along optical axis | 0.001 mm | 0.008 mm |
| rendered depth vs Euclidean ray length | 45.8 mm | 138.0 mm |
| back-projection, depth used as $Z$ | 0.001 mm | 0.009 mm |
| back-projection, depth misused as ray length | 45.8 mm | 138.0 mm |

The returned values are z-depth. Treating them as ray length costs up to 138.0 mm at the image corners.

**Pixel noise** (camera 0.6 m; 200 test pixels × 50 trials, median error in mm):

| Noise (px) | H from 4 points | H from 12 points | Depth back-projection |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 0.5 | 2.15 | 1.81 | 0.81 |
| 1 | 4.33 | 3.27 | 1.60 |
| 2 | 8.90 | 7.67 | 3.21 |
| 4 | 17.08 | 14.17 | 6.37 |

**Camera height** (1 px noise, 12-point H):

| Height (m) | mm per px at target | Homography (mm) | Depth back-projection (mm) |
|---|---|---|---|
| 0.3 | 0.67 | 1.13 | 0.88 |
| 0.5 | 1.12 | 2.76 | 1.43 |
| 0.8 | 1.79 | 5.13 | 2.18 |
| 1.2 | 2.69 | 9.56 | 3.28 |
| 2.0 | 4.49 | 13.92 | 5.72 |

**Observe.**

1. All errors are linear in pixel noise. In the height sweep the depth back-projection error is 1.2–1.3 times the pixel footprint at the target: with perfect depth, pixel noise is the whole error budget.
2. The homography is up to about 3 times worse at equal noise: its 8 parameters are fitted from noisy markers that cover only a 20 × 20 cm patch, and test pixels extrapolate beyond it. 12 markers beat 4 (3.27 vs 4.33 mm at 1 px).
3. With height the depth error grows like the footprint (0.88 → 5.72 mm, footprint 0.67 → 4.49 mm), the homography error faster (1.13 → 13.92 mm): the fixed marker patch covers fewer pixels as the camera rises.
4. Simulated depth is noise-free; real depth sensors add their own noise, so the depth column is a lower bound.

## Break it

Query the pixel that sees the centre of each box's top face (no pixel noise):

| Box height (mm) | Homography xy error (mm) | Depth back-projection error (mm) |
|---|---|---|
| 10 | 4.7 | 0.00 |
| 20 | 10.8 | 0.00 |
| 50 | 36.7 | 0.00 |
| 100 | 88.5 | 0.01 |

The homography maps every pixel to where its ray hits $z = 0$, so a point at height $h$ slides away from the camera along the ray: the error grows with $h$ and with how oblique the ray is. A 10 mm object already costs 4.7 mm. Fix: if the object height $h$ is known, intersect the ray with $z = h$ instead (this needs $K$ and the pose, not just $H$), or use depth.

## On the real robot

- For an SO-101 tabletop setup, a homography from 4–12 markers taped at measured positions is a quick calibration and is enough for flat, thin objects. Measure the marker positions in the robot base frame, not by eye. Not yet measured on the real arm.
- Real depth cameras return pixels without valid depth; check how your driver marks them and mask them before averaging. Check your sensor's documentation for whether it reports z-depth or range; we did not test any real depth camera.
- The camera pose $(R_{wc}, c)$ comes from extrinsic calibration (a board at a known base position) or hand-eye calibration for a wrist camera ([lesson 19](19-camera-calibration.md)); its error adds directly to every back-projected point.

## Check yourself

1. **Easy.** Why does $H$ need at least 4 point pairs?
2. **Medium.** At 0.6 m and 1 px noise, depth back-projection gives 1.60 mm. Name two error sources on a real robot that this simulation leaves out.
3. **Hard.** Implement "known height" back-projection: intersect the pixel ray with $z = h$ and rerun the break-it table. What error do you get if you assume $h = 30$ mm for the 50 mm box?

### Answers

1. $H$ is defined up to scale: 9 entries minus 1 = 8 unknowns, and each point pair gives 2 equations.
2. Depth sensor noise and calibration error in $K$ and the camera pose (also: rolling-shutter and timing offsets on a moving robot).
3. With the correct $h$ the ray hits the true point. Along one fixed ray the horizontal offset is linear in height, and the homography row is the special case $h = 0$; so assuming 30 mm for the 50 mm box gives $36.7 \times 20/50 \approx 14.7$ mm (derived from the table, not run).

## Takeaways and sources

- `mujoco.Renderer` depth is metric z-depth along the optical axis (max 0.008 mm off here); misusing it as ray length costs up to 138.0 mm.
- Homography: no intrinsics needed, but valid only on the plane; 10 mm of object height already costs 4.7 mm.
- Both methods scale linearly with pixel noise and with the pixel footprint; aim cameras close and use more than 4 markers.

Next: [lesson 21](21-object-perception.md), object perception for manipulation.

Sources:
- [MuJoCo XML reference · visual/map znear, zfar](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-map-znear): clipping planes as fractions of the model extent; resolution warning for a close near plane.
- [MuJoCo `rendering/classic/renderer.py` at 3.17.0](https://github.com/google-deepmind/mujoco/blob/3.17.0/python/mujoco/rendering/classic/renderer.py): reversed-Z read and conversion to metres.
- [OpenCV calib3d](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp): `findHomography` methods.
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004 ([book page](https://www.robots.ox.ac.uk/~vgg/hzbook/)): plane-induced homographies and back-projection.
- All numbers above: [`experiment.py`](../experiments/20-pixels-to-table/experiment.py), output in [`output.txt`](../experiments/20-pixels-to-table/output.txt).
