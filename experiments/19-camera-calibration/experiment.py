"""Lesson 19 · Camera model, calibration and hand-eye calibration.

Part A. A checkerboard (built from box geoms, exact geometry) is moved to random
poses in front of a MuJoCo camera with known fovy and image size. OpenCV finds the
corners and cv2.calibrateCamera estimates K. Ground truth K comes from fovy.
  Sweep: number of views -> reprojection error (train / held-out), focal error.
  Break it: limit the board tilt -> focal length becomes unobservable.
  Extra: a synthetic radial distortion k1 is applied with cv2.remap and recovered.

Part B. Hand-eye calibration AX = XB for the SO-101 wrist camera. Gripper poses
come from SO-101 forward kinematics at random joint angles; the camera-in-gripper
transform X is the wrist_cam of the Menagerie model. Solved with Park & Martin's
closed form (cv2.calibrateHandEye is not exposed in the OpenCV 5.0.0 Python build).
  Sweep: noise on the target-in-camera poses -> rotation / translation error of X.

Run:  python experiment.py
"""
import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

W, H, FOVY = 640, 480, 50.0
NX, NY, SQ = 9, 6, 0.025           # inner corners and square size (m)
CAM_Z = 0.6
TOP = 0.003                        # top surface of the black squares above the board origin (m)
SEED = 0
FLIP = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera (x right, y up, look -z) -> OpenCV (x right, y down, look +z)


def board_xml():
    sq = []
    for i in range(NX + 1):
        for j in range(NY + 1):
            if (i + j) % 2 == 0:
                x = (i - (NX + 1) / 2 + 0.5) * SQ
                y = (j - (NY + 1) / 2 + 0.5) * SQ
                sq.append(f'<geom type="box" size="{SQ/2} {SQ/2} 0.0005" pos="{x:.5f} {y:.5f} 0.0025" '
                          f'rgba="0 0 0 1" contype="0" conaffinity="0"/>')
    return f"""
<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/>
    <headlight diffuse="0.9 0.9 0.9" ambient="0.4 0.4 0.4" specular="0 0 0"/></visual>
  <worldbody>
    <camera name="cam" pos="0 0 {CAM_Z}" fovy="{FOVY}"/>
    <body name="board" mocap="true">
      <geom type="box" size="0.16 0.12 0.002" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      {''.join(sq)}
    </body>
  </worldbody>
</mujoco>"""


def k_true():
    f = (H / 2) / np.tan(np.radians(FOVY) / 2)
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])


def object_points():
    # corner (i, j) in the board frame; the first corner is at the -x, -y side
    g = np.zeros((NY * NX, 3), np.float32)
    g[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2) * SQ
    g[:, 0] -= (NX - 1) / 2 * SQ
    g[:, 1] -= (NY - 1) / 2 * SQ
    # z = 0: OpenCV needs a planar rig at z = 0; the board's top surface sits TOP above its origin
    return g


def random_pose(rng, tilt_deg):
    ax, ay = np.radians(rng.uniform(-tilt_deg, tilt_deg, 2))
    az = np.radians(rng.uniform(-20, 20))
    rot = np.zeros(4)
    mujoco.mju_euler2Quat(rot, np.array([ax, ay, az]), "xyz")
    pos = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.10)])
    return pos, rot


def render_views(n, tilt_deg, rng, distort=None):
    m = mujoco.MjModel.from_xml_string(board_xml())
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    objp = object_points()
    views, poses = [], []
    while len(views) < n:
        pos, quat = random_pose(rng, tilt_deg)
        d.mocap_pos[0], d.mocap_quat[0] = pos, quat
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="cam")
        img = r.render()
        if distort is not None:
            img = cv2.remap(img, distort[0], distort[1], cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ok, c = cv2.findChessboardCorners(gray, (NX, NY))
        if not ok:
            continue
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        c = np.asarray(c, np.float32).reshape(-1, 1, 2)
        # Label each detected corner with its board index (OpenCV's row order can start from
        # any corner). We use the ground-truth projection only for this labelling.
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        board = objp + np.array([0, 0, TOP], np.float32)
        pc = (FLIP @ (R.reshape(3, 3) @ board.T + pos[:, None] - np.array([[0], [0], [CAM_Z]]))).T
        uv = (k_true() @ (pc / pc[:, 2:3]).T).T[:, :2]
        nn = np.argmin(np.linalg.norm(uv[:, None, :] - c[None, :, 0, :], axis=2), axis=1)
        if len(set(nn)) != len(nn):
            continue
        c = c[nn].copy()
        views.append(c)
        poses.append(uv)
    r.close()
    return views, poses


def calibrate(views, flags=0):
    objp = object_points()
    rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(views), views, (W, H), None, None, flags=flags)
    return rms, K, dist


def heldout_rms(K, dist, views):
    objp = object_points()
    errs = []
    for c in views:
        _, rv, tv = cv2.solvePnP(objp, c, K, dist)
        p, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        errs.append(np.sum((p - c) ** 2, axis=2).ravel())
    return np.sqrt(np.mean(np.concatenate(errs)))


# ---------------- hand-eye (Park & Martin 1994) ----------------
def T(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def so3_log(R):
    w, _ = cv2.Rodrigues(R)
    return w.ravel()


def hand_eye_park(A_list, B_list):
    """Solve AX = XB. A: gripper motions, B: camera motions (4x4)."""
    M = sum(np.outer(so3_log(B[:3, :3]), so3_log(A[:3, :3])) for A, B in zip(A_list, B_list))
    w, V = np.linalg.eigh(M.T @ M)
    Rx = V @ np.diag(w ** -0.5) @ V.T @ M.T
    C = np.vstack([A[:3, :3] - np.eye(3) for A in A_list])
    d = np.concatenate([Rx @ B[:3, 3] - A[:3, 3] for A, B in zip(A_list, B_list)])
    tx = np.linalg.lstsq(C, d, rcond=None)[0]
    return T(Rx, tx)


def so101_gripper_poses(n, rng, frac=0.6):
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101"))
    d = mujoco.MjData(m)
    gid, cid = m.body("gripper").id, m.camera("wrist_cam").id
    lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
    poses = []
    for _ in range(n):
        d.qpos[:5] = rng.uniform(frac * lo, frac * hi)
        mujoco.mj_fwdPosition(m, d)       # also updates camera poses (mj_camlight)
        poses.append(T(d.xmat[gid].reshape(3, 3), d.xpos[gid].copy()))
    # ground-truth X = camera (OpenCV convention) in the gripper frame
    Rg, tg = poses[-1][:3, :3], poses[-1][:3, 3]
    Rc, tc = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    X = T(Rg.T @ Rc @ FLIP, Rg.T @ (tc - tg))
    return poses, X


def noisy(Tm, rng, rot_deg, trans_mm):
    dR, _ = cv2.Rodrigues(rng.normal(0, np.radians(rot_deg), 3))
    return T(dR @ Tm[:3, :3], Tm[:3, 3] + rng.normal(0, trans_mm / 1000, 3))


def rot_err_deg(R1, R2):
    return np.degrees(np.linalg.norm(so3_log(R1.T @ R2)))


def hand_eye_trials(gp, X, T_bt, rng, rd, tm, n_poses=15, trials=20):
    errs = []
    for _ in range(trials):
        idx = rng.choice(len(gp), n_poses, replace=False)
        G = [gp[i] for i in idx]
        Ct = [noisy(np.linalg.inv(g @ X) @ T_bt, rng, rd, tm) for g in G]   # target in camera
        pairs = [(i, j) for i in range(n_poses) for j in range(i + 1, n_poses)]
        A = [np.linalg.inv(G[j]) @ G[i] for i, j in pairs]
        B = [Ct[j] @ np.linalg.inv(Ct[i]) for i, j in pairs]
        Xe = hand_eye_park(A, B)
        errs.append([rot_err_deg(Xe[:3, :3], X[:3, :3]), 1000 * np.linalg.norm(Xe[:3, 3] - X[:3, 3])])
    return np.median(np.array(errs), axis=0)


def main():
    print(f"mujoco {mujoco.__version__} · opencv {cv2.__version__} · numpy {np.__version__} · seed {SEED}")
    Kt = k_true()
    print(f"image {W}x{H}, fovy {FOVY} deg -> ground truth fx = fy = {Kt[0,0]:.2f} px, "
          f"cx = {Kt[0,2]:.1f}, cy = {Kt[1,2]:.1f}")
    rng = np.random.default_rng(SEED)

    # A1. render a pool of views
    pool, proj = render_views(60, 35, rng)
    test, _ = render_views(15, 35, rng)
    det_err = np.concatenate([np.linalg.norm(c[:, 0] - p, axis=1) for c, p in zip(pool, proj)])
    print(f"\n[A1] {len(pool)} train + {len(test)} test views rendered; detected corners vs pinhole projection "
          f"with ground-truth K: mean {det_err.mean():.3f} px, max {det_err.max():.3f} px")

    # A2. sweep the number of views
    print("\n[A2] sweep number of views (10 random subsets each, median)")
    print(f"{'views':>6} {'train RMS px':>13} {'held-out RMS px':>16} {'fx err %':>9} {'fy err %':>9} "
          f"{'cx err px':>10} {'cy err px':>10}")
    for n in [3, 5, 10, 20, 40]:
        rows = []
        for _ in range(10):
            idx = rng.choice(len(pool), n, replace=False)
            rms, K, dist = calibrate([pool[i] for i in idx])
            rows.append([rms, heldout_rms(K, dist, test), 100 * abs(K[0, 0] / Kt[0, 0] - 1),
                         100 * abs(K[1, 1] / Kt[1, 1] - 1), abs(K[0, 2] - Kt[0, 2]), abs(K[1, 2] - Kt[1, 2])])
        r = np.median(np.array(rows), axis=0)
        print(f"{n:>6} {r[0]:>13.3f} {r[1]:>16.3f} {r[2]:>9.3f} {r[3]:>9.3f} {r[4]:>10.2f} {r[5]:>10.2f}")
    rms, K, dist = calibrate(pool)
    print(f"    all 60 views: fx {K[0,0]:.2f}, fy {K[1,1]:.2f}, cx {K[0,2]:.2f}, cy {K[1,2]:.2f}, "
          f"k1 {dist[0,0]:+.4f}, train RMS {rms:.3f} px")

    # A3. break it: limit the tilt
    print("\n[A3] break it: 10 views, board tilt limited to +-t deg (3 trials, median)")
    print(f"{'tilt deg':>9} {'train RMS px':>13} {'held-out RMS px':>16} {'fx err %':>9}")
    for tilt in [0, 2, 5, 15, 35]:
        rows = []
        for _ in range(3):
            v, _ = render_views(10, tilt, rng)
            rms, K, dist = calibrate(v)
            rows.append([rms, heldout_rms(K, dist, test), 100 * abs(K[0, 0] / Kt[0, 0] - 1)])
        r = np.median(np.array(rows), axis=0)
        print(f"{tilt:>9} {r[0]:>13.3f} {r[1]:>16.3f} {r[2]:>9.0f}" if r[2] > 100 else
              f"{tilt:>9} {r[0]:>13.3f} {r[1]:>16.3f} {r[2]:>9.2f}")

    # A4. synthetic radial distortion
    k1 = -0.15
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pts = np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, Kt, np.array([k1, 0, 0, 0, 0.0]), P=Kt).reshape(H, W, 2)
    maps = (und[..., 0].astype(np.float32), und[..., 1].astype(np.float32))
    dv, _ = render_views(20, 35, np.random.default_rng(SEED + 1), distort=maps)
    rms0, K0, d0 = calibrate(dv, flags=cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    print(f"\n[A4] synthetic k1 = {k1} applied to 20 views: estimated k1 = {d0[0,0]:+.4f}, "
          f"fx err {100*abs(K0[0,0]/Kt[0,0]-1):.3f} %, RMS {rms0:.3f} px")
    rms_nd, K_nd, _ = calibrate(dv, flags=cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                                | cv2.CALIB_ZERO_TANGENT_DIST)
    print(f"     same views, distortion forced to zero: RMS {rms_nd:.3f} px, fx err "
          f"{100*abs(K_nd[0,0]/Kt[0,0]-1):.2f} %")

    # B. hand-eye
    print(f"\n[B] hand-eye AX = XB, SO-101 wrist camera, 15 poses, 20 trials per row (median)")
    print(f"    cv2.calibrateHandEye available in this OpenCV build: {hasattr(cv2, 'calibrateHandEye')} "
          f"-> Park & Martin closed form implemented here")
    hrng = np.random.default_rng(SEED)
    gp, X = so101_gripper_poses(400, hrng)
    print(f"    ground-truth camera offset in gripper frame: |t| = {1000*np.linalg.norm(X[:3,3]):.1f} mm")
    T_bt = T(np.eye(3), np.array([0.25, 0.0, 0.0]))     # target on the table, base frame
    print(f"{'rot noise deg':>14} {'trans noise mm':>15} {'X rot err deg':>14} {'X trans err mm':>15}")
    for rd, tm in [(0, 0), (0.1, 0.5), (0.2, 1), (0.5, 2), (1, 5), (2, 10)]:
        e = hand_eye_trials(gp, X, T_bt, hrng, rd, tm)
        print(f"{rd:>14} {tm:>15} {e[0]:>14.3f} {e[1]:>15.2f}")

    print("\n[B2] break it: joints sampled in a fraction of their range (noise 0.5 deg / 2 mm)")
    print(f"{'range frac':>11} {'median rot between poses deg':>29} {'X rot err deg':>14} {'X trans err mm':>15}")
    for frac in [0.02, 0.05, 0.1, 0.3, 0.6]:
        gpf, Xf = so101_gripper_poses(400, hrng, frac)
        spread = np.median([rot_err_deg(gpf[i][:3, :3], gpf[i + 1][:3, :3]) for i in range(0, 398, 2)])
        e = hand_eye_trials(gpf, Xf, T_bt, hrng, 0.5, 2)
        print(f"{frac:>11} {spread:>29.1f} {e[0]:>14.3f} {e[1]:>15.2f}")


if __name__ == "__main__":
    main()
