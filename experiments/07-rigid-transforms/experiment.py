"""Lesson 07 · Rigid transforms and frame trees (SO-101, MuJoCo Menagerie).

The SO-101 base is bolted to the table at an offset and a 30 deg yaw, so world != base.
A 3 cm cube sits on the table. We express the cube's position in four frames:
world W, base B (body "base"), gripper G (site "gripperframe"), wrist camera C (camera "wrist_cam"),
by hand with 4x4 homogeneous transforms in NumPy, and compare with MuJoCo framepos sensors.

Part A  Hand transforms vs MuJoCo sensors, inverse check, camera projection vs the renderer.
Part B  Sweep shoulder_pan: which frame coordinates of the (unmoving) cube change?
Part C  Every arm joint +-30 deg: max change of the cube's coordinates per frame; T_GC stays constant.
Break   Wrong composition order / R instead of R^T: how wrong is the cube position?

Run:  python experiment.py      (< 10 s)
"""
import os
import platform
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | Python {platform.python_version()} "
      f"| {platform.machine()}")
np.set_printoptions(precision=4, suppress=True)

BASE_POS = np.array([0.10, -0.05, 0.0])      # where the base is bolted (m)
BASE_YAW = np.deg2rad(30)                    # and how it is turned
CUBE_W = np.array([0.2905, 0.0600, 0.015])   # cube centre on the table, world frame (m)
Q_ARM = np.array([0.13, -1.65, 1.15, 1.33, -2.13])   # a pose where the wrist camera sees the cube
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def build():
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101", "so101.xml"))
    base = spec.body("base")
    base.pos = BASE_POS
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0, 0, BASE_YAW]), "xyz")
    base.quat = q
    cube = spec.worldbody.add_body(name="cube")
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015] * 3, rgba=[0, 1, 0, 1])
    # reftype "xbody" = the body's regular frame; "body" would be its INERTIAL frame (a trap, see Part A)
    for name, rt, rn in [("cube_in_base", mujoco.mjtObj.mjOBJ_XBODY, "base"),
                         ("cube_in_base_inertial", mujoco.mjtObj.mjOBJ_BODY, "base"),
                         ("cube_in_gripperframe", mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
                         ("cube_in_wrist_cam", mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")]:
        spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_XBODY, objname="cube", reftype=rt, refname=rn)
    return spec.compile()


m = build()
d = mujoco.MjData(m)
SITE = m.site("gripperframe").id
CAM = m.camera("wrist_cam").id
BASE = m.body("base").id
CUBE = m.body("cube").id
qadr = {j: m.jnt_qposadr[m.joint(j).id] for j in ARM}
cube_qadr = m.jnt_qposadr[m.body_jntadr[CUBE]]


def set_pose(q_arm):
    for j, v in zip(ARM, q_arm):
        d.qpos[qadr[j]] = v
    d.qpos[cube_qadr:cube_qadr + 3] = CUBE_W
    d.qpos[cube_qadr + 3:cube_qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)      # mj_kinematics alone does NOT fill cam_xpos/cam_xmat (mj_camlight does)


def T(R, t):
    """4x4 homogeneous transform from rotation R (3x3) and translation t (3,)."""
    M = np.eye(4)
    M[:3, :3] = np.asarray(R).reshape(3, 3)
    M[:3, 3] = t
    return M


def T_inv(M):
    """Closed-form inverse: [R t; 0 1]^-1 = [R^T  -R^T t; 0 1]."""
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)


def frames():
    """World poses of the frames, read from MuJoCo's frame tree."""
    return {"W": np.eye(4),
            "B": T(d.xmat[BASE], d.xpos[BASE]),
            "G": T(d.site_xmat[SITE], d.site_xpos[SITE]),
            "C": T(d.cam_xmat[CAM], d.cam_xpos[CAM])}


def cube_in_all():
    pw = np.append(d.xpos[CUBE], 1.0)
    return {k: (T_inv(Tw) @ pw)[:3] for k, Tw in frames().items()}


SENSOR = {"B": "cube_in_base", "G": "cube_in_gripperframe", "C": "cube_in_wrist_cam"}

# ---------- Part A ----------
set_pose(Q_ARM)
print(f"\nsetup: base at {BASE_POS} m, yaw {np.rad2deg(BASE_YAW):.0f} deg; cube at {CUBE_W} m; arm q = {Q_ARM} rad")
print("\n== A. Cube position in each frame: hand (NumPy 4x4) vs MuJoCo framepos sensor ==")
print(f"{'frame':28s} {'hand x, y, z (m)':>30s} {'max |hand - sensor|':>20s}")
hand = cube_in_all()
names = {"W": "world W", "B": "base B (body base)", "G": "gripper G (site gripperframe)",
         "C": "camera C (camera wrist_cam)"}
for k in "WBGC":
    err = np.abs(hand[k] - d.sensor(SENSOR[k]).data).max() if k in SENSOR else 0.0
    print(f"{names[k]:28s} {np.array2string(hand[k], precision=4, floatmode='fixed'):>30s} {err:20.1e}")
trap = d.sensor("cube_in_base_inertial").data
print(f"trap: sensor with reftype='body' (base INERTIAL frame) reads {np.array2string(trap, precision=4)}, "
      f"{1000 * np.linalg.norm(trap - hand['B']):.1f} mm away from the base-frame value")

F = frames()
inv_err = max(np.abs(T_inv(F[k]) - np.linalg.inv(F[k])).max() for k in "BGC")
print(f"closed-form T^-1 vs np.linalg.inv, max abs diff: {inv_err:.1e}")
T_BG = T_inv(F["B"]) @ F["G"]
chain_err = np.abs(F["B"] @ T_BG - F["G"]).max()
print(f"T_WG == T_WB @ T_BG, max abs diff: {chain_err:.1e}")

# camera projection: MuJoCo camera looks along -z, x right, y up; fovy is vertical FOV
W_PX, H_PX = 640, 480
fovy = m.cam_fovy[CAM]
f_px = (H_PX / 2) / np.tan(np.deg2rad(fovy) / 2)
x, y, z = hand["C"]
u_hand = W_PX / 2 + f_px * x / (-z)
v_hand = H_PX / 2 - f_px * y / (-z)          # image rows grow downward
def project(p_c):
    """Pinhole projection of camera-frame points (MuJoCo convention) to pixels (u right, v down)."""
    p_c = np.atleast_2d(p_c)
    return np.stack([W_PX / 2 + f_px * p_c[:, 0] / (-p_c[:, 2]),
                     H_PX / 2 - f_px * p_c[:, 1] / (-p_c[:, 2])], axis=1)


try:
    import cv2
    r = mujoco.Renderer(m, height=H_PX, width=W_PX)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="wrist_cam")
    seg = r.render()
    r.close()
    cube_geom = m.body_geomadr[CUBE]
    mask = (seg[..., 0] == cube_geom) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    # hand silhouette: 8 cube corners world -> camera -> pixels, filled convex hull
    corners_w = CUBE_W + 0.015 * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    T_CW = T_inv(frames()["C"])
    corners_c = (T_CW[:3, :3] @ corners_w.T).T + T_CW[:3, 3]
    uv = project(corners_c)
    hull = cv2.convexHull(np.round(uv * 16).astype(np.int32))
    sil = np.zeros((H_PX, W_PX), np.uint8)
    cv2.fillConvexPoly(sil, hull, 1, lineType=cv2.LINE_8, shift=4)
    sil = sil.astype(bool)
    iou = (mask & sil).sum() / (mask | sil).sum()
    vv, uu = np.nonzero(mask)
    vs, us = np.nonzero(sil)
    print(f"camera fovy {fovy:.0f} deg -> f = {f_px:.1f} px at {W_PX}x{H_PX}; "
          f"cube centre projects to u = {u_hand:.1f}, v = {v_hand:.1f} px")
    print(f"rendered cube mask centroid: u = {uu.mean() + 0.5:.1f}, v = {vv.mean() + 0.5:.1f} px "
          f"(a blob centroid is not the projected centre: perspective)")
    print(f"silhouette: rendered {mask.sum()} px, hand-projected {sil.sum()} px, IoU = {iou:.3f}, "
          f"centroid offset = {np.hypot(uu.mean() - us.mean(), vv.mean() - vs.mean()):.2f} px")
except Exception as e:  # headless machines without an OpenGL context
    print(f"renderer unavailable ({type(e).__name__}); projected cube centre u = {u_hand:.1f}, v = {v_hand:.1f} px")

# ---------- Part B ----------
print("\n== B. Sweep shoulder_pan (other joints fixed): cube coordinates in each frame (cm) ==")
print(f"{'pan (deg)':>9s} {'in B: x y z':>22s} {'in G: x y z':>22s} {'in C: x y z':>22s} {'gripper site in W':>22s}")
for pan_deg in [-40, -20, 0, 20, 40]:
    q = Q_ARM.copy()
    q[0] = np.deg2rad(pan_deg)
    set_pose(q)
    c = cube_in_all()
    fmt = lambda v: " ".join(f"{100 * a:6.2f}" for a in v)  # noqa: E731
    print(f"{pan_deg:9.1f} {fmt(c['B']):>22s} {fmt(c['G']):>22s} {fmt(c['C']):>22s} {fmt(d.site_xpos[SITE]):>22s}")

# ---------- Part C ----------
print("\n== C. Each arm joint swept +-30 deg around the pose: max change of cube coordinates (mm) ==")
print(f"{'joint':>14s} {'in W':>8s} {'in B':>8s} {'in G':>8s} {'in C':>8s} {'T_GC drift':>11s}")
set_pose(Q_ARM)
ref = cube_in_all()
T_GC_ref = T_inv(frames()["G"]) @ frames()["C"]
for i, j in enumerate(ARM):
    dmax = {k: 0.0 for k in "WBGC"}
    drift = 0.0
    for delta in np.deg2rad(np.linspace(-30, 30, 13)):
        q = Q_ARM.copy()
        q[i] += delta
        set_pose(q)
        c = cube_in_all()
        for k in "WBGC":
            dmax[k] = max(dmax[k], np.linalg.norm(c[k] - ref[k]))
        Fr = frames()
        drift = max(drift, np.abs(T_inv(Fr["G"]) @ Fr["C"] - T_GC_ref).max())
    print(f"{j:>14s} {1000 * dmax['W']:8.2f} {1000 * dmax['B']:8.2f} {1000 * dmax['G']:8.2f} "
          f"{1000 * dmax['C']:8.2f} {drift:11.1e}")

# ---------- Break it ----------
print("\n== Break it. Common frame mistakes, cube position in the gripper frame (error vs sensor, mm) ==")
set_pose(Q_ARM)
F = frames()
pw = np.append(d.xpos[CUBE], 1.0)
truth = d.sensor("cube_in_gripperframe").data.copy()
R, t = F["G"][:3, :3], F["G"][:3, 3]
mistakes = {
    "correct: T_WG^-1 p": (T_inv(F["G"]) @ pw)[:3],
    "used T_WG instead of its inverse": (F["G"] @ pw)[:3],
    "R instead of R^T: R (p - t)": R @ (pw[:3] - t),
    "forgot the translation: R^T p": R.T @ pw[:3],
    "wrong order: (T_BG T_WB)^-1 p": (T_inv(T_BG @ F["B"]) @ pw)[:3],
}
for k, v in mistakes.items():
    print(f"{k:36s} {1000 * np.linalg.norm(v - truth):10.1f}")
