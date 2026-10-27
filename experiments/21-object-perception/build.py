"""Lesson 21 · Build it: a colour detector from scratch, checked against OpenCV at every step.

Step 1  RGB -> HSV in OpenCV's 8-bit units vs cv2.cvtColor (all pixels of a frame, 1M random colours).
Step 2  The red threshold vs cv2.inRange.
Step 3  Opening and the largest blob vs cv2.morphologyEx and cv2.connectedComponentsWithStats.
Step 4  Centroid from moments vs cv2.moments; pixels -> table (x, y) vs the truth and the Experiment.
Step 5  Close the loop: our detector + the Experiment's pick routine vs the Experiment's table.

The functions live in physai/vision.py (the lesson shows them step by step).
The scene, camera, lighting and pick routine are taken from experiment.py, so both tell one story.
Run:  uv run python experiments/21-object-perception/build.py      (~5 s, deterministic)
"""
import importlib.util
import os
import sys

import cv2
import mujoco
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.vision import (color_mask, detect_color, label_blobs, largest_blob, mask_centroid,  # noqa: E402
                           mask_to_table_xy, open3x3, rgb_to_hsv8)

spec = importlib.util.spec_from_file_location("perception_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | OpenCV {cv2.__version__}")

RED = [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))]   # the Experiment's thresholds
m = exp.build()
d = mujoco.MjData(m)
rng = np.random.default_rng(exp.SEED)
positions = [np.array([rng.uniform(0.15, 0.24), rng.uniform(-0.07, 0.07)]) for _ in range(exp.N_POS)]
exp.set_cube(m, d, positions[0])
cam = exp.Camera(m, d)


def render(xy, scale=1.0):
    exp.set_light(m, scale=scale)
    exp.set_cube(m, d, xy)
    cam.rgb.update_scene(d, "top")
    return cam.rgb.render()


def render_depth():
    cam.dep.update_scene(d, "top")
    return cam.dep.render()


def top_pixel(xy):
    """Pixel that sees the centre of the cube's top face (as in the Experiment's 'top HSV')."""
    pc = (np.r_[xy, 2 * exp.HALF] - cam.c) @ cam.R_wc
    u, v = np.round((cam.K @ (pc / pc[2]))[:2]).astype(int)
    return v, u


def hsv_str(p):
    return "(" + ",".join(str(int(x)) for x in p) + ")"


def hue_diff(a, b):
    dh = np.abs(a[..., 0].astype(int) - b[..., 0].astype(int))
    return np.minimum(dh, 180 - dh)


print("\n== Step 1. RGB -> HSV (OpenCV 8-bit units) vs cv2.cvtColor ==")
colors = np.random.default_rng(1).integers(0, 256, size=(1000, 1000, 3), dtype=np.uint8)
ours, ref = rgb_to_hsv8(colors), cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)
dh, ds, dv = hue_diff(ours, ref), np.abs(ours[..., 1].astype(int) - ref[..., 1]), np.abs(ours[..., 2].astype(int) - ref[..., 2])
print(f"1,000,000 random colours: H differs on {np.mean(dh > 0) * 100:.2f} % (max {dh.max()}), "
      f"S on {np.mean(ds > 0) * 100:.2f} % (max {ds.max()}), V on {np.mean(dv > 0) * 100:.2f} %")
rgb = render(positions[0])
ours, ref = rgb_to_hsv8(rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
off = (hue_diff(ours, ref) > 0) | np.any(ours[..., 1:] != ref[..., 1:], -1)
reddish = (ref[..., 0] <= 10) | (ref[..., 0] >= 170)
print(f"rendered frame (position 1, light scale 1.0, {rgb.shape[1]} x {rgb.shape[0]}): {off.sum()} pixels off by 1, "
      f"median OpenCV hue of those {np.median(ref[off][:, 0]):.0f}, {np.sum(off & reddish)} of them with a red hue")
print(" scale  top HSV ours  top HSV OpenCV")
for s in (0.1, 1.0, 3.0):
    img = render(positions[0], s)
    v, u = top_pixel(positions[0])
    print(f"{s:6.1f} {hsv_str(rgb_to_hsv8(img)[v, u]):>13} {hsv_str(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[v, u]):>15}")

print("\n== Step 2. The red threshold vs cv2.inRange ==")
rgb = render(positions[0])
hsv_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
ref_mask = sum(cv2.inRange(hsv_cv, lo, hi) for lo, hi in RED) > 0
print(f"same HSV input: our mask differs from inRange on {np.sum(color_mask(hsv_cv, RED) != ref_mask)} pixels")
raw = color_mask(rgb_to_hsv8(rgb), RED)
print(f"our HSV input:  {raw.sum()} red pixels, differs on {np.sum(raw != ref_mask)} pixels")
print(f"red pixels in the hue box near 0: {color_mask(rgb_to_hsv8(rgb), RED[:1]).sum()}, "
      f"near 179: {color_mask(rgb_to_hsv8(rgb), RED[1:]).sum()}")

print("\n== Step 3. Opening and the largest blob vs OpenCV and SciPy ==")
speck = raw.copy()
speck[[40, 41, 300], [50, 50, 600]] = True                 # three stray pixels: a 2-pixel and a 1-pixel speck
opened = open3x3(speck)
cv_open = cv2.morphologyEx(speck.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
print(f"opening: {speck.sum()} -> {opened.sum()} pixels, differs from cv2.morphologyEx on {np.sum(opened != cv_open)}")
lab = label_blobs(speck)
n_cv = cv2.connectedComponentsWithStats(speck.astype(np.uint8))[0] - 1
n_sp = ndimage.label(speck, structure=np.ones((3, 3)))[1]
print(f"blobs before opening: ours {len(np.unique(lab[lab > 0]))}, OpenCV {n_cv}, scipy.ndimage.label {n_sp}")
blob = largest_blob(opened)
ref_blob = exp.detect(rgb)
print(f"largest blob: {blob.sum()} pixels, differs from the Experiment's detect() on {np.sum(blob != ref_blob)}")

print("\n== Step 4. Centroid and table position ==")
mo = cv2.moments(blob.astype(np.uint8), binaryImage=True)
uv = mask_centroid(blob)
print(f"centroid (u, v) = ({uv[0]:.3f}, {uv[1]:.3f}) px, max |ours - cv2.moments| = "
      f"{np.abs(uv - [mo['m10'] / mo['m00'], mo['m01'] / mo['m00']]).max():.1e}")
err_c, err_all, err_top, diff_exp, est_nominal = [], [], [], [], []
for xy in positions:
    rgb, depth = render(xy), render_depth()
    blob = detect_color(rgb, RED)
    u, v = np.round(mask_centroid(blob)).astype(int)
    p = cam.c + depth[v, u] * (cam.R_wc @ np.linalg.inv(cam.K) @ [u, v, 1.0])
    err_c.append(np.linalg.norm(p[:2] - xy))
    err_all.append(np.linalg.norm(mask_to_table_xy(blob, depth, cam.K, cam.R_wc, cam.c, z_min=-1.0) - xy))
    est = mask_to_table_xy(blob, depth, cam.K, cam.R_wc, cam.c)
    err_top.append(np.linalg.norm(est - xy))
    est_nominal.append(est)
    diff_exp.append(np.linalg.norm(est - exp.locate(exp.detect(rgb), depth, cam.K, cam.R_wc, cam.c)))
print("30 positions, light scale 1.0              median mm  max mm")
for name, e in [("centroid pixel, back-projected", err_c), ("mean of all blob points", err_all),
                ("mean of top-face points (z > 20 mm)", err_top)]:
    print(f"  {name:<38} {np.median(e) * 1000:7.2f} {np.max(e) * 1000:7.2f}")
print(f"max |ours - Experiment's locate()| = {np.max(diff_exp):.1e} m")
print(f"one pixel at the top face ({cam.c[2] - 2 * exp.HALF:.2f} m from the camera) spans "
      f"{(cam.c[2] - 2 * exp.HALF) / cam.K[0, 0] * 1000:.2f} mm")

print("\n== Step 5. Close the loop: our detector + the Experiment's pick ==")
print(" scale  detected  median err mm  max err mm  picks ok")
for s in (0.1, 1.0, 3.0):
    det, errs, ok = 0, [], 0
    for i, xy in enumerate(positions):
        if s == 1.0:
            est = est_nominal[i]                             # the frames of Step 4
            exp.set_cube(m, d, xy)
        else:
            blob = detect_color(render(xy, s), RED)
            if blob is None:
                continue
            est = mask_to_table_xy(blob, render_depth(), cam.K, cam.R_wc, cam.c)
        det, errs = det + 1, errs + [1000 * np.linalg.norm(est - xy)]
        ok += exp.pick(m, d, est)
    med, mx = (f"{np.median(errs):.1f}", f"{np.max(errs):.1f}") if errs else ("-", "-")
    print(f"{s:6.1f} {det:6d}/30 {med:>14} {mx:>11} {ok:6d}/30")
