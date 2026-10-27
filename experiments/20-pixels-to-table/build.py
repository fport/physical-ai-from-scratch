"""Lesson 20 · Build it: from pixels to table points with your own rays, depth and homographies.

Step 1  Pixel rays, ray-plane intersection and depth back-projection vs MuJoCo's rendered depth.
Step 2  The raw depth buffer: linearize it yourself and compare with mujoco.Renderer's metres.
Step 3  The table homography from the camera model, and fitted from markers (lesson 19's DLT) vs cv2.findHomography.
Step 4  The Experiment's pixel-noise sweep with your functions.
Step 5  Break it: raised points, and intersecting the ray with an assumed height instead.

The functions live in physai/vision.py (the lesson shows them step by step).
The scene, markers and random pixels come from experiment.py (same seed), so both tell one story.
Run:  uv run python experiments/20-pixels-to-table/build.py      (~1 s, deterministic)
"""
import importlib.util
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.vision import (apply_homography, back_project, bilinear, homography_dlt, linearize_depth,  # noqa: E402
                           pixel_rays, plane_homography, ray_plane)

spec = importlib.util.spec_from_file_location("table_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

print(f"MuJoCo {mujoco.__version__} | OpenCV {cv2.__version__} | NumPy {np.__version__}")


def mm(x):
    return 1000 * np.asarray(x)


# ---------------------------------------------------------------- Step 1
print("\n== Step 1. Rays, the table plane and depth back-projection (camera 0.6 m, 300 table pixels) ==")
rng = np.random.default_rng(exp.SEED)
m, depth, K, R_wc, c = exp.setup(0.6)                       # rendered metric depth, K, camera pose
uv = exp.table_pixels(depth, K, R_wc, c, rng, 300)          # the Experiment's 300 pixels
P_true, s = ray_plane(uv, K, R_wc, c)                       # truth: where each ray meets z = 0
z_img = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
ray_len = s * np.linalg.norm(pixel_rays(uv, K, R_wc), axis=1)
e_bp = mm(np.linalg.norm(back_project(uv, z_img, K, R_wc, c) - P_true, axis=1))
dirs = pixel_rays(uv, K, R_wc)
e_wrong = mm(np.linalg.norm(c + (z_img / np.linalg.norm(dirs, axis=1))[:, None] * dirs - P_true, axis=1))
print(f"max |ray_plane - Experiment's hit_plane| = {np.abs(P_true - exp.hit_plane(uv, K, R_wc, c)[0]).max():.1e} m")
print(f"rendered depth vs z-depth s:           median {np.median(mm(np.abs(z_img - s))):.3f} mm, max {mm(np.abs(z_img - s)).max():.3f} mm")
print(f"rendered depth vs ray length:          median {np.median(mm(np.abs(z_img - ray_len))):.1f} mm, max {mm(np.abs(z_img - ray_len)).max():.1f} mm")
print(f"back_project, depth used as Z:         median {np.median(e_bp):.3f} mm, max {e_bp.max():.3f} mm")
print(f"back_project, depth misused as length: median {np.median(e_wrong):.1f} mm, max {e_wrong.max():.1f} mm")
q = uv + rng.normal(0, 1.0, uv.shape)
print(f"sub-pixel lookup: max |bilinear - cv2.remap| = {np.abs(bilinear(depth, q) - exp.bilinear(depth, q)).max():.1e} m")

# ---------------------------------------------------------------- Step 2
print("\n== Step 2. The raw depth buffer ==")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, exp.H, exp.W)
r.enable_depth_rendering()
r.update_scene(d, camera="cam")
metres = r.render().copy()                                  # what the Renderer returns
raw = np.zeros((exp.H, exp.W), np.float32)
mujoco.mjr_readPixels(None, raw, r._rect, r._mjr_context)   # the same frame, unconverted (OpenGL: bottom row first)
r.close()
raw = np.flipud(raw).astype(np.float64)
near, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
Z = linearize_depth(raw, near, far)
conventional = near * far / (far - raw * (far - near))      # the textbook formula, wrong for a reversed buffer
i, j = exp.H // 2, exp.W // 2
print(f"near {mm(near):.1f} mm, far {far:.1f} m; raw values {raw.min():.6f} .. {raw.max():.6f} (1 = near, 0 = far)")
print(f"max |linearize_depth(raw) - Renderer| = {mm(np.abs(Z - metres).max()):.4f} mm")
print(f"image centre: Renderer {metres[i, j]:.4f} m, ours {Z[i, j]:.4f} m, conventional formula {conventional[i, j]:.4f} m")

# ---------------------------------------------------------------- Step 3
print("\n== Step 3. The table homography: from the camera model and from markers ==")
H_cam = plane_homography(K, R_wc, c)
print(f"plane_homography vs ray_plane on the 300 pixels: max {np.abs(apply_homography(H_cam, uv) - P_true[:, :2]).max():.1e} m")
for name, marks in [("4 markers", exp.MARK4), ("12 markers", exp.MARK12)]:
    px = exp.project(np.c_[marks, np.zeros(len(marks))], K, R_wc, c)
    H_fit = homography_dlt(px, marks)                       # lesson 19's DLT: pixels -> table (x, y)
    noisy = px + rng.normal(0, 1.0, px.shape)
    H_noisy = homography_dlt(noisy, marks)
    H_cv, _ = cv2.findHomography(noisy.astype(np.float32), marks.astype(np.float32), 0)
    gap = np.linalg.norm(apply_homography(H_noisy, uv) - apply_homography(H_cv, uv), axis=1)
    print(f"{name:>10}: exact pixels, max |H_fit - H_cam| on the table {np.abs(apply_homography(H_fit, uv) - P_true[:, :2]).max():.1e} m;"
          f" 1 px noise, ours vs cv2: median {np.median(mm(gap)):.3f} mm, max {mm(gap).max():.3f} mm")

# ---------------------------------------------------------------- Step 4
print("\n== Step 4. The pixel-noise sweep with your functions (Experiment seed and loop, 200 pixels x 50 trials) ==")
rng = np.random.default_rng(exp.SEED)
uv = exp.table_pixels(depth, K, R_wc, c, rng, 300)
print(" noise px  H from 4 pts  H from 12 pts  depth back-proj   (median mm)")
for sigma in [0, 0.5, 1, 2, 4]:
    e4, e12, ed = [], [], []
    for _ in range(50):
        q = uv[rng.choice(len(uv), 200, replace=False)]
        Pq, _ = ray_plane(q, K, R_wc, c)
        qn = q + rng.normal(0, sigma, q.shape)
        for marks, acc in [(exp.MARK4, e4), (exp.MARK12, e12)]:
            px = exp.project(np.c_[marks, np.zeros(len(marks))], K, R_wc, c) + rng.normal(0, sigma, (len(marks), 2))
            xy = apply_homography(homography_dlt(px, marks), qn)
            acc.append(np.median(mm(np.linalg.norm(xy - Pq[:, :2], axis=1))))
        ed.append(np.median(mm(np.linalg.norm(back_project(qn, bilinear(depth, qn), K, R_wc, c) - Pq, axis=1))))
    print(f"{sigma:>9} {np.median(e4):>13.2f} {np.median(e12):>14.2f} {np.median(ed):>16.2f}")

# ---------------------------------------------------------------- Step 5
print("\n== Step 5. Break it: the top face of each box (no pixel noise) ==")
print(" box height mm  homography mm  depth back-proj mm  ray_plane at z = 0.6 h mm")
for bh, (bx, by) in zip(exp.BOX_H, exp.BOX_XY):
    px = np.round(exp.project(np.array([[bx, by, bh]]), K, R_wc, c))
    P_top, _ = ray_plane(px, K, R_wc, c, bh)               # what this pixel really sees
    e_h = np.linalg.norm(apply_homography(H_cam, px)[0] - P_top[0, :2])
    e_d = np.linalg.norm(back_project(px, bilinear(depth, px), K, R_wc, c)[0] - P_top[0])
    e_g = np.linalg.norm(ray_plane(px, K, R_wc, c, 0.6 * bh)[0][0, :2] - P_top[0, :2])
    print(f"{mm(bh):>14.0f} {mm(e_h):>14.1f} {mm(e_d):>19.2f} {mm(e_g):>26.1f}")
