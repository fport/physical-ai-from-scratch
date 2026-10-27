"""Camera models, calibration, pixels to the table and color detection (lessons 19-21 · Build it).

Plain NumPy. Each function is checked against OpenCV and MuJoCo's own renderer in
experiments/{19-camera-calibration,20-pixels-to-table,21-object-perception}/build.py (and in tests/).
"""
import numpy as np


# --- lesson 19: camera model and calibration ---


def intrinsics_from_fovy(fovy, width, height):
    """Camera matrix K of a MuJoCo camera: square pixels, vertical field of view fovy (degrees)."""
    f = (height / 2) / np.tan(np.radians(fovy) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def project_points(X, R, t, K, dist=(0.0, 0.0, 0.0, 0.0, 0.0)):
    """World points X (N, 3) to pixels (N, 2). R, t: world -> camera; dist = (k1, k2, p1, p2, k3), OpenCV's order."""
    Xc = X @ R.T + t                                    # camera frame: x right, y down, z forward
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]     # divide by depth
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)     # tangential terms p1, p2
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.c_[K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]]


def homography_dlt(src, dst):
    """3x3 H with dst ~ H src from N >= 4 point pairs (N, 2 arrays): the normalized direct linear transform."""
    def normalizer(p):                                   # centroid to 0, mean distance to sqrt(2)
        c = p.mean(axis=0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    Ts, Td = normalizer(src), normalizer(dst)
    a = np.c_[src, np.ones(len(src))] @ Ts.T
    b = np.c_[dst, np.ones(len(dst))] @ Td.T
    A = []
    for (x, y, _), (u, v, _) in zip(a, b):               # u = (h1 . a) / (h3 . a), same for v: two rows
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    Hn = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)  # least-squares null vector of A
    H = np.linalg.inv(Td) @ Hn @ Ts                      # undo the normalization
    return H / H[2, 2]


def zhang_intrinsics(Hs):
    """Closed-form K from >= 3 board-to-image homographies (Zhang 2000); also returns the singular values of V."""
    def v(H, i, j):                                      # h_i^T B h_j as a linear function of b
        a, c = H[:, i], H[:, j]
        return np.array([a[0] * c[0], a[0] * c[1] + a[1] * c[0], a[1] * c[1],
                         a[2] * c[0] + a[0] * c[2], a[2] * c[1] + a[1] * c[2], a[2] * c[2]])
    V = np.array([row for H in Hs for row in (v(H, 0, 1), v(H, 0, 0) - v(H, 1, 1))])
    _, sv, Vt = np.linalg.svd(V)
    b11, b12, b22, b13, b23, b33 = Vt[-1] * np.sign(Vt[-1][0])  # B = K^-T K^-1, up to scale
    cy = (b12 * b13 - b11 * b23) / (b11 * b22 - b12**2)
    lam = b33 - (b13**2 + cy * (b12 * b13 - b11 * b23)) / b11
    fx = np.sqrt(lam / b11)
    fy = np.sqrt(lam * b11 / (b11 * b22 - b12**2))
    skew = -b12 * fx**2 * fy / lam
    cx = skew * cy / fy - b13 * fx**2 / lam
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), sv


def pose_from_homography(H, K):
    """Board pose (R, t) in the camera from its homography: K^-1 H = scale * [r1 r2 t]."""
    A = np.linalg.inv(K) @ H
    s = np.sign(A[2, 2]) / np.linalg.norm(A[:, 0])       # the board lies in front: t_z > 0
    r1, r2, t = s * A[:, 0], s * A[:, 1], s * A[:, 2]
    U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    return U @ Vt, t                                     # nearest rotation matrix


def rotvec_to_mat(w):
    """Rotation vector (axis * angle, rad) to a 3x3 matrix (Rodrigues' formula, safe at w = 0)."""
    a = np.linalg.norm(w)
    k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if a < 1e-12:
        return np.eye(3) + k
    return np.eye(3) + np.sin(a) / a * k + (1 - np.cos(a)) / a**2 * (k @ k)


def mat_to_rotvec(R):
    """3x3 rotation matrix to its rotation vector (angle below pi)."""
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / 2 if a < 1e-12 else a / (2 * np.sin(a)) * v


def refine_calibration(obj, views, K, poses, iters=10):
    """Levenberg-Marquardt on the total reprojection error over fx, fy, cx, cy, k1, k2 and every board pose.

    obj: (N, 3) board corners; views: list of (N, 2) detected pixels; poses: list of (R, t) starting guesses.
    Pose i moves by a small correction c_i = (w, dt): R @ rotvec_to_mat(w), t + dt. Returns K, dist, poses, RMS.
    """
    def view_residual(q, pose, c, uv):                   # q = (fx, fy, cx, cy, k1, k2)
        Kq = np.array([[q[0], 0.0, q[2]], [0.0, q[1], q[3]], [0.0, 0.0, 1.0]])
        R, t = pose[0] @ rotvec_to_mat(c[:3]), pose[1] + c[3:]
        return (project_points(obj, R, t, Kq, (q[4], q[5], 0.0, 0.0, 0.0)) - uv).ravel()

    def residual(q, c):
        return np.concatenate([view_residual(q, p, ci, uv) for p, ci, uv in zip(poses, c, views)])

    n, m = len(views), 2 * len(obj)
    q, c = np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0], np.zeros((n, 6))
    r, damping = residual(q, c), 1e-3
    for _ in range(iters):
        J = np.zeros((n * m, 6 + 6 * n))                 # pose i only moves the pixels of view i
        for i, (p, uv) in enumerate(zip(poses, views)):
            rows, x = slice(i * m, (i + 1) * m), np.r_[q, c[i]]
            for j in range(12):                          # forward differences: 6 shared + 6 own parameters
                h = 1e-6 * max(1.0, abs(x[j]))
                xh = x.copy()
                xh[j] += h
                J[rows, j if j < 6 else 6 * i + j] = (view_residual(xh[:6], p, xh[6:], uv) - r[rows]) / h
        JtJ = J.T @ J
        step = np.linalg.solve(JtJ + damping * np.diag(np.diag(JtJ)), -J.T @ r)
        q_new, c_new = q + step[:6], c + step[6:].reshape(n, 6)
        r_new = residual(q_new, c_new)
        if r_new @ r_new < r @ r:                        # better: accept, trust the linear model more
            q, c, r, damping = q_new, c_new, r_new, damping / 10
        else:                                            # worse: reject, take a shorter step
            damping *= 10
    Kq = np.array([[q[0], 0.0, q[2]], [0.0, q[1], q[3]], [0.0, 0.0, 1.0]])
    new = [(R @ rotvec_to_mat(ci[:3]), t + ci[3:]) for (R, t), ci in zip(poses, c)]
    return Kq, np.array([q[4], q[5], 0.0, 0.0, 0.0]), new, np.sqrt(2 * np.mean(r**2))


def hand_eye_park(As, Bs):
    """Solve A X = X B (Park & Martin 1994). As: gripper motions, Bs: camera motions (4x4); X: camera in gripper."""
    M = sum(np.outer(mat_to_rotvec(B[:3, :3]), mat_to_rotvec(A[:3, :3])) for A, B in zip(As, Bs))
    w, V = np.linalg.eigh(M.T @ M)
    Rx = V @ np.diag(w ** -0.5) @ V.T @ M.T              # rotation: (M^T M)^(-1/2) M^T
    C = np.vstack([A[:3, :3] - np.eye(3) for A, _ in zip(As, Bs)])
    d = np.concatenate([Rx @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = Rx, np.linalg.lstsq(C, d, rcond=None)[0]   # translation: (R_A - I) t = R_X t_B - t_A
    return X


# --- lesson 20: pixels to the table ---


def pixel_rays(uv, K, R_wc):
    """World-frame directions of the rays through pixels uv (N, 2), scaled so each has camera z = 1."""
    p = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T      # [x/z, y/z, 1] in the camera frame
    return p @ R_wc.T                                         # rotate into the world


def back_project(uv, Z, K, R_wc, c):
    """World points seen at pixels uv with depth Z (N,) measured along the optical axis, not along the ray."""
    return c + Z[:, None] * pixel_rays(uv, K, R_wc)


def ray_plane(uv, K, R_wc, c, z0=0.0):
    """Where the pixel rays meet the horizontal plane z = z0; also returns the z-depth of each hit."""
    r = pixel_rays(uv, K, R_wc)
    s = (z0 - c[2]) / r[:, 2]                                 # how far to walk along each ray
    return c + s[:, None] * r, s


def linearize_depth(d, near, far):
    """Reversed-Z depth-buffer value d (1 at near, 0 at far) to metres along the optical axis."""
    return near * far / (near + d * (far - near))


def bilinear(img, uv):
    """Sample img at sub-pixel positions uv (N, 2) from the four nearest pixels (pixel centres on integers)."""
    u0 = np.clip(np.floor(uv[:, 0]).astype(int), 0, img.shape[1] - 2)
    v0 = np.clip(np.floor(uv[:, 1]).astype(int), 0, img.shape[0] - 2)
    a, b = uv[:, 0] - u0, uv[:, 1] - v0                       # weights inside the pixel square
    return ((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u0 + 1]
            + (1 - a) * b * img[v0 + 1, u0] + a * b * img[v0 + 1, u0 + 1])


def plane_homography(K, R_wc, c):
    """Pixel -> table (x, y) homography of the plane z = 0, from the camera model: (K [r1 r2 t])^-1."""
    R = R_wc.T                                                # world -> camera, as in lesson 19
    t = -R @ c
    H = np.linalg.inv(K @ np.c_[R[:, 0], R[:, 1], t])
    return H / H[2, 2]


def apply_homography(H, p):
    """Map points p (N, 2) with H and divide out the scale."""
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


# --- lesson 21: color detection ---


def rgb_to_hsv8(rgb):
    """8-bit RGB image -> HSV in OpenCV's 8-bit units: H = degrees / 2 in 0..179, S and V in 0..255."""
    x = rgb.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(axis=-1)                                   # value: the brightest channel
    diff = v - x.min(axis=-1)                            # 0 for a grey pixel
    s = np.where(v > 0, 255 * diff / np.maximum(v, 1), 0)  # saturation: colourful vs grey
    dd = np.maximum(diff, 1)
    h = np.where(v == r, 30 * (g - b) / dd,              # hue in half-degrees, sector by sector
                 np.where(v == g, 60 + 30 * (b - r) / dd, 120 + 30 * (r - g) / dd))
    h = np.floor(np.where(diff > 0, h, 0) + 0.5)         # round first ...
    h = np.where(h < 0, h + 180, h)                      # ... then wrap: red sits at both ends
    return np.stack([h, np.floor(s + 0.5), v], axis=-1).astype(np.uint8)


def color_mask(hsv, ranges):
    """Pixels inside any of the (lo, hi) HSV boxes, bounds inclusive; red needs two boxes (hue wraps)."""
    mask = np.zeros(hsv.shape[:2], bool)
    for lo, hi in ranges:
        mask |= np.all((hsv >= np.asarray(lo)) & (hsv <= np.asarray(hi)), axis=-1)
    return mask


def _shifts(a, fill):
    """The 9 copies of a 2D array shifted by -1, 0, +1 pixels in each direction; outside = fill."""
    p = np.pad(a, 1, constant_values=fill)
    h, w = a.shape
    return [p[i:i + h, j:j + w] for i in range(3) for j in range(3)]


def open3x3(mask):
    """Morphological opening with a 3x3 square: erode (keep a pixel if all 9 are set), then dilate."""
    eroded = np.logical_and.reduce(_shifts(mask, True))   # outside the image counts as set, as in OpenCV
    return np.logical_or.reduce(_shifts(eroded, False))


def label_blobs(mask):
    """Label 8-connected blobs: every pixel takes the largest id among its neighbours until nothing changes."""
    lab = np.where(mask, np.arange(1, mask.size + 1).reshape(mask.shape), 0)
    while True:
        new = np.where(mask, np.maximum.reduce(_shifts(lab, 0)), 0)
        if np.array_equal(new, lab):
            return lab                                    # one id per blob (not numbered 1, 2, 3 ...)
        lab = new


def largest_blob(mask, min_area=30):
    """The biggest blob as a boolean mask, or None if there is none of at least min_area pixels."""
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))   # label only the bounding box
    lab = label_blobs(mask[box])
    ids, area = np.unique(lab[lab > 0], return_counts=True)
    if area.max() < min_area:
        return None
    out = np.zeros_like(mask)
    out[box] = lab == ids[np.argmax(area)]
    return out


def mask_centroid(mask):
    """Pixel centroid (u, v) from the image moments m00, m10, m01 of a binary mask."""
    v, u = np.nonzero(mask)
    m00, m10, m01 = len(u), u.sum(), v.sum()
    return np.array([m10 / m00, m01 / m00])


def mask_to_table_xy(mask, depth, K, R_wc, c, z_min=0.02):
    """Back-project every mask pixel with its depth (lesson 20) and average (x, y) of the points above z_min."""
    v, u = np.nonzero(mask)
    rays = np.c_[u, v, np.ones(len(u))] @ np.linalg.inv(K).T @ R_wc.T   # camera ray of each pixel, world frame
    P = c + depth[v, u][:, None] * rays                                   # depth is the distance along the axis
    top = P[P[:, 2] > z_min]                                              # the top face of the object
    return (top if len(top) else P)[:, :2].mean(axis=0)


def detect_color(rgb, ranges, min_area=30):
    """HSV threshold -> opening -> largest blob: the object's mask, or None."""
    return largest_blob(open3x3(color_mask(rgb_to_hsv8(rgb), ranges)), min_area)
