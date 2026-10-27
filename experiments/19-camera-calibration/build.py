"""Lesson 19 · Build it: the camera model, Zhang's calibration and hand-eye calibration from scratch.

Step 1  K from fovy and the pinhole projection, against MuJoCo-rendered corners and cv2.projectPoints.
Step 2  The board-to-image homography (normalized DLT), against cv2.findHomography and the true homography.
Step 3  Zhang's closed-form intrinsics from homographies; what a flat board does to the linear system.
Step 4  Board poses from the homographies, then Levenberg-Marquardt on the reprojection error vs cv2.calibrateCamera.
Step 5  Hand-eye AX = XB (Park & Martin) against the Experiment's solver and table.

The functions live in physai/vision.py (the lesson shows them step by step).
The rendered views come from experiment.py (same scene, same seed), so both tell one story.
Run:  uv run python experiments/19-camera-calibration/build.py      (~2 s, deterministic)
"""
import importlib.util
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.vision import (hand_eye_park, homography_dlt, intrinsics_from_fovy, mat_to_rotvec,  # noqa: E402
                           pose_from_homography, project_points, refine_calibration, zhang_intrinsics)

spec = importlib.util.spec_from_file_location("calib_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

print(f"MuJoCo {mujoco.__version__} | OpenCV {cv2.__version__} | NumPy {np.__version__}")

# The Experiment's render_views() draws a random board pose per attempt; record them to know the truth.
drawn = []
_random_pose = exp.random_pose


def recording_pose(rng, tilt):
    drawn.append(_random_pose(rng, tilt))
    return drawn[-1]


exp.random_pose = recording_pose
OBJ = exp.object_points().astype(float)              # 54 corners on the board plane z = 0 (m)
CAM = np.array([0.0, 0.0, exp.CAM_Z])


def board_pose(pos, quat):
    """Board frame -> OpenCV camera frame for a mocap pose (camera at CAM looking down world -z)."""
    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, quat)
    Rb = Rb.reshape(3, 3)
    return exp.FLIP @ Rb, exp.FLIP @ (Rb @ [0.0, 0.0, exp.TOP] + pos - CAM)


def views_with_poses(n, tilt, rng, K):
    """Rendered + detected corners (n views) and the true board pose of each view."""
    drawn.clear()
    views, truth_uv = exp.render_views(n, tilt, rng)
    candidates = [board_pose(*p) for p in drawn]
    poses = [next((R, t) for R, t in candidates if np.allclose(project_points(OBJ, R, t, K), uv, atol=1e-6))
             for uv in truth_uv]
    return [v.reshape(-1, 2).astype(float) for v in views], poses


def apply_h(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:]


def rot_deg(R1, R2):
    return np.degrees(np.linalg.norm(mat_to_rotvec(R1.T @ R2)))


# ---------------------------------------------------------------- Step 1
print("\n== Step 1. K from fovy and the pinhole projection ==")
K_true = intrinsics_from_fovy(exp.FOVY, exp.W, exp.H)
print(f"fovy {exp.FOVY:g} deg, {exp.W}x{exp.H}: fx = fy = {K_true[0, 0]:.2f} px, cx = {K_true[0, 2]:.1f}, "
      f"cy = {K_true[1, 2]:.1f}   max |K - Experiment's k_true()| = {np.abs(K_true - exp.k_true()).max():.1e}")
rng = np.random.default_rng(exp.SEED)
views, poses = views_with_poses(60, 35, rng, K_true)
err = np.concatenate([np.linalg.norm(v - project_points(OBJ, R, t, K_true), axis=1) for v, (R, t) in zip(views, poses)])
print(f"60 rendered views: detected corners vs our projection, mean {err.mean():.3f} px, max {err.max():.3f} px")
dist = np.array([-0.15, 0.03, 0.001, -0.002, 0.01])
diff = 0.0
for R, t in poses:
    p_cv, _ = cv2.projectPoints(OBJ, cv2.Rodrigues(R)[0], t, K_true, dist)
    diff = max(diff, np.abs(p_cv.reshape(-1, 2) - project_points(OBJ, R, t, K_true, dist)).max())
print(f"with dist (k1, k2, p1, p2, k3) = {tuple(dist.tolist())}: max |ours - cv2.projectPoints| = {diff:.1e} px")

# ---------------------------------------------------------------- Step 2
print("\n== Step 2. One homography per view (board plane -> pixels) ==")
Hs = [homography_dlt(OBJ[:, :2], v) for v in views]
H_true = [K_true @ np.c_[R[:, 0], R[:, 1], t] for R, t in poses]
fit = np.sqrt(np.mean([np.mean(np.sum((apply_h(H, OBJ[:, :2]) - v) ** 2, axis=1)) for H, v in zip(Hs, views)]))
vs_cv = max(np.abs(apply_h(H, OBJ[:, :2]) - apply_h(cv2.findHomography(OBJ[:, :2], v, 0)[0], OBJ[:, :2])).max()
            for H, v in zip(Hs, views))
vs_true = max(np.abs(apply_h(H, OBJ[:, :2]) - apply_h(G, OBJ[:, :2])).max() for H, G in zip(Hs, H_true))
exact = max(np.abs(apply_h(homography_dlt(OBJ[:, :2], apply_h(G, OBJ[:, :2])), OBJ[:, :2]) - apply_h(G, OBJ[:, :2])).max()
            for G in H_true)
print(f"RMS distance of the 54 corners from H(board point): {fit:.3f} px")
print(f"max |ours - cv2.findHomography| = {vs_cv:.4f} px, max |ours - true K[r1 r2 t]| = {vs_true:.3f} px")
print(f"on exact projected corners: max |ours - true| = {exact:.1e} px")

# ---------------------------------------------------------------- Step 3
print("\n== Step 3. Zhang's closed form: K from the homographies ==")
K_exact, _ = zhang_intrinsics(H_true)
print(f"from the 60 true homographies: max |K - K_true| = {np.abs(K_exact - K_true).max():.1e} px")
print(" views      fx      fy      cx      cy   skew  fx err %")
for n in (3, 5, 10, 20, 60):
    Kz, sv = zhang_intrinsics(Hs[:n])
    print(f"{n:>6} {Kz[0, 0]:7.2f} {Kz[1, 1]:7.2f} {Kz[0, 2]:7.2f} {Kz[1, 2]:7.2f} {Kz[0, 1]:6.3f} "
          f"{100 * abs(Kz[0, 0] / K_true[0, 0] - 1):9.3f}")
K_zhang, sv35 = zhang_intrinsics(Hs[:10])
flat, _ = views_with_poses(10, 0, rng, K_true)
K_flat, sv0 = zhang_intrinsics([homography_dlt(OBJ[:, :2], v) for v in flat])
print("singular values of V / largest, 10 views (the last one is the solution, the one before decides):")
print("  tilt 35 deg: " + " ".join(f"{s:.1e}" for s in sv35 / sv35[0]))
print("  tilt  0 deg: " + " ".join(f"{s:.1e}" for s in sv0 / sv0[0]) + f"   -> fx = {K_flat[0, 0]:.0f} px")

# ---------------------------------------------------------------- Step 4
print("\n== Step 4. Poses from the homographies, then Levenberg-Marquardt on all 60 views ==")
K_zhang, _ = zhang_intrinsics(Hs)
starts = [pose_from_homography(H, K_zhang) for H in Hs]
print(f"start poses vs truth: median {np.median([rot_deg(R, Rt) for (R, _), (Rt, _) in zip(starts, poses)]):.3f} deg, "
      f"{np.median([1000 * np.linalg.norm(t - tt) for (_, t), (_, tt) in zip(starts, poses)]):.2f} mm")
K_lm, d_lm, fitted, rms = refine_calibration(OBJ, views, K_zhang, starts)
obj32, v32 = [OBJ.astype(np.float32)] * len(views), [v.astype(np.float32) for v in views]
flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
rms_cv, K_cv, d_cv, _, _ = cv2.calibrateCamera(obj32, v32, (exp.W, exp.H), None, None, flags=flags)
rms_5, K_5, d_5, _, _ = cv2.calibrateCamera(obj32, v32, (exp.W, exp.H), None, None)
print("                                       fx      fy      cx      cy       k1       k2  RMS px")
for name, Kx, dx, r in [("ours (k1, k2)", K_lm, d_lm.ravel(), rms), ("cv2.calibrateCamera, k1 k2 only", K_cv, d_cv.ravel(), rms_cv),
                        ("cv2.calibrateCamera, default 5 coef", K_5, d_5.ravel(), rms_5)]:
    print(f"{name:<36} {Kx[0, 0]:7.2f} {Kx[1, 1]:7.2f} {Kx[0, 2]:7.2f} {Kx[1, 2]:7.2f} {dx[0]:+8.4f} {dx[1]:+8.4f} {r:7.3f}")
print(f"max |K ours - K cv2 (same model)| = {np.abs(K_lm - K_cv).max():.1e} px, fx error vs truth {100 * abs(K_lm[0, 0] / K_true[0, 0] - 1):.3f} %")
print(f"fitted poses vs truth: median {np.median([rot_deg(R, Rt) for (R, _), (Rt, _) in zip(fitted, poses)]):.3f} deg, "
      f"{np.median([1000 * np.linalg.norm(t - tt) for (_, t), (_, tt) in zip(fitted, poses)]):.2f} mm")

# ---------------------------------------------------------------- Step 5
print("\n== Step 5. Hand-eye AX = XB on the Experiment's SO-101 poses (same seed, 15 poses, 20 trials) ==")
hrng = np.random.default_rng(exp.SEED)
gp, X = exp.so101_gripper_poses(400, hrng)
T_bt = exp.T(np.eye(3), np.array([0.25, 0.0, 0.0]))       # board on the table, base frame
print(f"true camera offset in the gripper frame: {1000 * np.linalg.norm(X[:3, 3]):.1f} mm")
print(" rot noise deg  trans noise mm  X rot err deg  X trans err mm  max |X ours - X Exp.|")
for rd, tm in [(0, 0), (0.1, 0.5), (0.2, 1), (0.5, 2), (1, 5), (2, 10)]:
    errs, gap = [], 0.0
    for _ in range(20):
        idx = hrng.choice(len(gp), 15, replace=False)
        G = [gp[i] for i in idx]
        C = [exp.noisy(np.linalg.inv(g @ X) @ T_bt, hrng, rd, tm) for g in G]     # board in camera, with noise
        pairs = [(i, j) for i in range(15) for j in range(i + 1, 15)]
        A = [np.linalg.inv(G[j]) @ G[i] for i, j in pairs]
        B = [C[j] @ np.linalg.inv(C[i]) for i, j in pairs]
        Xe = hand_eye_park(A, B)
        gap = max(gap, np.abs(Xe - exp.hand_eye_park(A, B)).max())
        errs.append([rot_deg(Xe[:3, :3], X[:3, :3]), 1000 * np.linalg.norm(Xe[:3, 3] - X[:3, 3])])
    e = np.median(errs, axis=0)
    print(f"{rd:>14} {tm:>15} {e[0]:>14.3f} {e[1]:>15.2f} {gap:>21.1e}")
