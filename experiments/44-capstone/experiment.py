"""Lesson 44 · Capstone: say, see, plan, grasp. One integrated pipeline, measured end to end in simulation.

A spoken-style command ("put the red cube in the box") goes to a local language model (Ollama HTTP API,
/api/chat with tools, temperature 0, fixed seed). The model only chooses tools and object names:
    look()                -> names of the objects perception currently sees
    pick(object)          -> perception -> validation -> IK -> min-jerk trajectory -> physical grasp
    place(target)         -> validation -> IK -> min-jerk trajectory -> open the jaw over the box
    report_result(...)    -> the model's claim (recorded, never used to decide success)
Geometry never passes through the model. Perception sees a rendered RGB + depth image from a fixed overhead
camera (lessons 19-21); no ground-truth position reaches the agent or the tool layer's planner.

Layers and the lessons they come from:
    language     Ollama tool calling                         lesson 42
    perception   HSV mask + depth back-projection            lessons 20, 21
    validation   workspace, joint margins, IK residual       lessons 10, 15
    motion       DLS IK + min-jerk joint trajectories        lessons 10, 13, 16
    grasp        servo squeeze, no weld (scripted_pick)      lesson 16
    place        IK over the box (ik_solve), open the jaw    lesson 10
    harness      success from the simulator state only       lesson 25 (Wilson interval)

Experiment:
  A. Fixed command set x random cube layouts, two models -> end-to-end success with a Wilson 95% interval,
     failures attributed to the first failing layer (language, perception, IK/limits, grasp, place).
  B. Robustness sweep: an unmodelled camera tilt (extrinsic calibration error, lessons 19/38) -> end-to-end
     success and where the failures land.
LLM replies are memoised within one run by the exact request (messages + tools + options): with temperature 0
and a fixed seed the same request gave the same reply in lesson 42, and identical conversations recur
across layouts because no coordinates reach the model. Every run starts with an empty cache.

Needs `ollama serve` with qwen2.5:7b and qwen3:1.7b pulled.
Run:  python experiment.py      (a few minutes; time depends on machine load and the LLM)
"""
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import urllib.request

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
from common.menagerie import path  # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g16 = load("lesson16", "16-grasping-and-pick-place/experiment.py")   # scripted_pick, plan, ik, minjerk
g10 = load("lesson10", "10-inverse-kinematics/experiment.py")        # ik_solve
g21 = load("lesson21", "21-object-perception/experiment.py")         # camera, intrinsics, locate, light

# ------------------------------------------------------------------ scene
HALF = g16.CUBE_HALF                       # 3 cm cubes
TABLE_TOP = 0.001                          # 2 mm table slab on the floor, top at z = 1 mm
COLORS = {"red": (0.85, 0.1, 0.1), "green": (0.1, 0.7, 0.15), "blue": (0.05, 0.15, 0.9)}
BOX_XY = np.array([0.12, -0.18])           # open box: 8 x 8 cm inside, 3 cm walls, purple
BOX_IN, WALL_T, WALL_H = 0.04, 0.004, 0.03
BOX_RGB = (0.6, 0.15, 0.75)
CUBE_REGION = ((0.16, 0.25), (-0.08, 0.10))  # x, y range [m] where cubes are dropped
MIN_GAP = 0.07                             # centre spacing between cubes [m]
Q_PARK = g21.Q_PARK                        # arm folded back, out of the camera's view (lesson 21)
PLACE_Z = 0.075                            # gripper site height above the box when the jaw opens [m]
PLACE_AXIS_W = 0.003                       # soft "jaw points down" term for the place IK (a released cube needs
                                           # no exact orientation; a hard term pins wrist_flex at its limit)

# ------------------------------------------------------------------ validation (lesson 15 style limits)
R_MIN, R_MAX = 0.12, 0.30                  # radial distance of a grasp target from the pan axis [m]
IK_TOL_MM = 5.0                            # every planned pose must be reached within 5 mm
JOINT_MARGIN = 0.05                        # planned joint goals at least 0.05 rad inside the model range
PAN_AXIS = np.array([0.0388, 0.0])
PERC_TOL_MM = 8.0                          # perception contract: target estimate within 8 mm (lesson 21: 8 mm aim
                                           # error already cost 9 of 30 picks)

# ------------------------------------------------------------------ perception thresholds (OpenCV HSV)
HSV = {"red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
       "green": [((45, 120, 70), (85, 255, 255))],
       "blue": [((100, 120, 70), (130, 255, 255))],
       "box": [((135, 120, 70), (160, 255, 255))]}

# ------------------------------------------------------------------ language
MODELS = ("qwen2.5:7b", "qwen3:1.7b")
URL = "http://localhost:11434/api/chat"
OPTIONS = {"temperature": 0, "seed": 0}
MAX_LLM_CALLS = 6
SYSTEM = ("You control a robot arm (SO-101) on a table through tools. The user's request comes from speech "
          "recognition and may contain recognition errors. Use look() to see which objects are on the table, "
          "pick(object) to grasp a cube and place(target) to put the cube you hold into a container. Use object "
          "names exactly as look() returns them. When you are finished, call report_result exactly once: "
          "success=true only if the task was completed.")
TOOLS = [
    {"type": "function", "function": {"name": "look", "description": "List the objects the camera sees on the table.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "pick", "description": "Grasp one cube from above and lift it.",
                                      "parameters": {"type": "object", "properties": {
                                          "object": {"type": "string", "description": "e.g. 'red cube'"}},
                                          "required": ["object"]}}},
    {"type": "function", "function": {"name": "place", "description": "Put the cube you hold into a container.",
                                      "parameters": {"type": "object", "properties": {
                                          "target": {"type": "string", "description": "e.g. 'box'"}},
                                          "required": ["target"]}}},
    {"type": "function", "function": {"name": "report_result", "description": "Finish and report the outcome.",
                                      "parameters": {"type": "object", "properties": {
                                          "success": {"type": "boolean"}, "message": {"type": "string"}},
                                          "required": ["success", "message"]}}},
]
# (id, spoken-style command, cube the user means)
COMMANDS = [
    ("plain", "put the red cube in the box", "red"),
    ("filler", "um, could you grab the green block and drop it in the bin", "green"),
    ("terse", "blue one, box, please", "blue"),
    ("asr", "put the read cube in the box", "red"),                       # speech-recognition homophone
    ("negation", "the cube that is neither red nor green goes in the box", "blue"),
    ("turkish", "kırmızı küpü kutuya koy", "red"),                        # Turkish: "put the red cube in the box"
]
N_LAYOUTS = 5
LAYOUT_SEED = 0
TILTS_DEG = (0.0, 0.5, 1.0, 1.5, 2.0)
SWEEP_MODEL = "qwen2.5:7b"
SWEEP_COMMANDS = ("plain", "filler", "terse")   # one command per cube colour keeps the sweep's LLM time short


# ================================================================== scene
def build(layout):
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
    spec.option.timestep = g16.DT
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.45, 0.45, TABLE_TOP],
                            pos=[0.15, 0, 0], rgba=[0.45, 0.45, 0.45, 1])
    for c, xy in layout.items():
        b = spec.worldbody.add_body(name=f"{c}_cube", pos=[xy[0], xy[1], TABLE_TOP + HALF])
        b.add_freejoint()
        b.add_geom(name=f"{c}_cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, mass=0.03, rgba=[*COLORS[c], 1],
                   friction=[1.0, 0.03, 0.003], condim=4, solref=[0.01, 1], priority=2)
    r = BOX_IN + WALL_T
    for k, (dx, dy, sx, sy) in enumerate([(r, 0, WALL_T, r + WALL_T), (-r, 0, WALL_T, r + WALL_T),
                                          (0, r, r + WALL_T, WALL_T), (0, -r, r + WALL_T, WALL_T)]):
        spec.worldbody.add_geom(name=f"box_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, WALL_H / 2],
                                pos=[BOX_XY[0] + dx, BOX_XY[1] + dy, TABLE_TOP + WALL_H / 2], rgba=[*BOX_RGB, 1])
    spec.worldbody.add_geom(name="box_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[r + WALL_T, r + WALL_T, 0.001],
                            pos=[BOX_XY[0], BOX_XY[1], TABLE_TOP + 0.001], rgba=[*BOX_RGB, 1])
    spec.worldbody.add_camera(name="top", pos=g21.CAM_POS.tolist(), fovy=g21.FOVY)
    m = spec.compile()
    g21.set_light(m, 1.0)                  # lesson 21's nominal light
    return m, mujoco.MjData(m)


def make_layouts(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        pts = []
        while len(pts) < 3:
            p = np.array([rng.uniform(*CUBE_REGION[0]), rng.uniform(*CUBE_REGION[1])])
            if all(np.linalg.norm(p - q) >= MIN_GAP for q in pts):
                pts.append(p)
        out.append(dict(zip(rng.permutation(list(COLORS)), pts)))
    return out


def arm_ids(m, cube_color):
    """Lesson 16's index dict, with `cube` = the cube the grasp aims at (scripted_pick watches its jaw contacts
    to decide when the squeeze has settled; that is the role of a jaw-stall sensor on a real gripper)."""
    jf, jm = set(), set()
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        b = m.body(m.geom_bodyid[g]).name
        if b == "gripper":
            jf.add(g)
        elif b == "moving_jaw_so101_v1":
            jm.add(g)
    return dict(cube=m.geom(f"{cube_color}_cube").id, cube_body=m.body(f"{cube_color}_cube").id,
                site=m.site("gripperframe").id, qadr=[m.joint(j).qposadr[0] for j in g16.ARM],
                dadr=[m.joint(j).dofadr[0] for j in g16.ARM], act=[m.actuator(j).id for j in g16.ARM],
                grip=m.actuator("gripper").id, grip_q=m.joint("gripper").qposadr[0], fixed=jf, moving=jm)


# ================================================================== perception
class Perception:
    """Fixed overhead camera. Extrinsics are 'calibrated' once on the nominal model; tilt_deg rotates the
    real camera afterwards (an extrinsic error the perception code does not know about)."""

    def __init__(self, m, d, tilt_deg=0.0):
        cid = m.camera("top").id
        mujoco.mj_forward(m, d)
        self.K = g21.intrinsics()
        self.R_wc = d.cam_xmat[cid].reshape(3, 3) @ g21.FLIP
        self.c = d.cam_xpos[cid].copy()
        if tilt_deg:
            axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
            a = math.radians(tilt_deg) / 2
            dq = np.r_[math.cos(a), math.sin(a) * axis]
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, dq, m.cam_quat[cid].copy())
            m.cam_quat[cid] = q
        self.ren = mujoco.Renderer(m, g21.H, g21.W)

    def close(self):
        self.ren.close()

    def mask(self, hsv, name):
        mk = sum(cv2.inRange(hsv, lo, hi) for lo, hi in HSV[name]).clip(0, 255).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mk)
        if n < 2:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return lab == k if stats[k, cv2.CC_STAT_AREA] >= 30 else None

    def see(self, d):
        """-> {name: (x, y) in the base frame [m]} for every object found in one RGB-D frame."""
        self.ren.update_scene(d, "top")
        img = self.ren.render().copy()
        self.ren.enable_depth_rendering()
        depth = self.ren.render().copy()
        self.ren.disable_depth_rendering()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        out = {}
        for name in HSV:
            mk = self.mask(hsv, name)
            if mk is None:
                continue
            if name == "box":                       # whole box (walls + floor): centre of all its points
                v, u = np.nonzero(mk)
                rays = np.c_[u, v, np.ones(len(u))] @ np.linalg.inv(self.K).T @ self.R_wc.T
                out[name] = (self.c + depth[v, u][:, None] * rays)[:, :2].mean(axis=0)
            else:                                   # cube: top-face points (lesson 21)
                out[f"{name} cube"] = g21.locate(mk, depth, self.K, self.R_wc, self.c)
        return out


# ================================================================== tool layer
def color_of(name):
    s = str(name).lower()
    return next((c for c in COLORS if c in s), None)


class Robot:
    """SO-101 + table + three cubes + box. The tool methods return short JSON for the model;
    `log` keeps per-layer measurements for the harness (never shown to the model)."""

    def __init__(self, layout, tilt_deg=0.0):
        self.m, self.d = build(layout)
        self.I = arm_ids(self.m, "red")
        self.d.qpos[self.I["qadr"]] = Q_PARK[:5]
        self.d.qpos[self.I["grip_q"]] = g16.G_OPEN
        self.d.ctrl[self.I["act"]] = Q_PARK[:5]
        self.d.ctrl[self.I["grip"]] = g16.G_OPEN
        self.grip_cmd = g16.G_OPEN
        self.cam = Perception(self.m, self.d, tilt_deg)
        for _ in range(int(0.3 / g16.DT)):          # let the cubes settle on the table
            mujoco.mj_step(self.m, self.d)
        self.holding = None
        self.scene = {}                              # last perception result
        self.log = dict(picks=[], places=[], rejections=[], errors=0)

    # ---------- motion helpers (lesson 13: min-jerk in joint space; lesson 16: the servos track it)
    def arm_q(self):
        return self.d.qpos[self.I["qadr"]].copy()

    def drive(self, q_goal, T=1.5, settle=0.3, grip=None):
        q0, t0 = self.arm_q(), self.d.time
        grip = self.grip_cmd if grip is None else grip
        while self.d.time - t0 < T + settle:
            s = g16.minjerk((self.d.time - t0) / T)
            self.d.ctrl[self.I["act"]] = q0 + s * (np.asarray(q_goal) - q0)
            self.d.ctrl[self.I["grip"]] = grip
            mujoco.mj_step(self.m, self.d)

    def park(self):
        if np.max(np.abs(self.arm_q() - Q_PARK[:5])) > 0.05:
            self.drive(Q_PARK[:5])

    def perceive(self):
        self.park()
        self.scene = self.cam.see(self.d)
        return self.scene

    def visible(self):
        return sorted(self.scene)

    def margin(self, q):
        """Smallest distance [rad] of the arm joint goals q from the model's joint range."""
        lo, hi = self.m.jnt_range[:5, 0], self.m.jnt_range[:5, 1]
        return float(np.min(np.minimum(q - lo, hi - q)))

    def reject(self, tool, why):
        self.log["rejections"].append((tool, why))
        return json.dumps({"rejected": why})

    # ---------- tools
    def look(self):
        self.perceive()
        return json.dumps({"visible_objects": self.visible()})

    def pick(self, object=""):
        if self.holding:
            return json.dumps({"error": f"already holding the {self.holding} cube; place it first"})
        self.perceive()
        c = color_of(object)
        name = f"{c} cube" if c else str(object)
        if c is None or name not in self.scene:
            self.log["errors"] += 1
            return json.dumps({"error": f"no '{object}' visible; visible objects: {self.visible()}"})
        xy = self.scene[name]
        rec = dict(color=c, est=xy.copy(), rejected=None, lifted=False,
                   true=self.d.xpos[self.m.body(f"{c}_cube").id][:2].copy())   # true: harness log only
        self.log["picks"].append(rec)
        # validation: workspace, then IK residual and joint margins of the three grasp poses (lessons 10, 15)
        r = float(np.linalg.norm(xy - PAN_AXIS))
        if not (R_MIN <= r <= R_MAX):
            rec["rejected"] = "workspace"
            return self.reject("pick", f"{name} is outside the workspace ({R_MIN*100:.0f}-{R_MAX*100:.0f} cm "
                                       "from the base)")
        self.I = arm_ids(self.m, c)
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = self.d.qpos
        scratch.qpos[self.I["qadr"]] = g16.HOME_Q
        poses, err = g16.plan(self.m, scratch, self.I, np.r_[xy, HALF])
        rec["ik_mm"] = err * 1000
        if err * 1000 > IK_TOL_MM:
            rec["rejected"] = "ik"
            return self.reject("pick", f"grasp pose for {name} is not reachable (IK error above {IK_TOL_MM:.0f} mm)")
        rec["approach_margin"] = min(self.margin(poses["pre"]), self.margin(poses["lift"]))
        if self.margin(poses["down"]) < JOINT_MARGIN:
            rec["rejected"] = "joint margin"
            return self.reject("pick", f"grasp pose for {name} is within {JOINT_MARGIN} rad of a joint limit")
        # execute: home (jaw open) -> lesson 16's physical scripted pick, aimed at the PERCEIVED position
        self.grip_cmd = g16.G_OPEN
        self.drive(g16.HOME_Q, T=1.0, settle=0.3)
        g16.ids = lambda _m, _I=self.I: _I          # scripted_pick looks up its indices through lesson 16's ids()
        g16.scripted_pick(self.m, self.d, np.r_[xy, HALF])
        self.grip_cmd = g16.G_CLOSE
        rec["lifted"] = self.cube_lifted(c)          # harness-only ground truth, for attribution
        # postcondition from proprioception: the jaw stopped on something instead of closing fully
        jaw = float(self.d.qpos[self.I["grip_q"]])
        rec["jaw"] = jaw
        if jaw > g16.G_CLOSE + 0.15:
            self.holding = c
            return json.dumps({"status": f"holding the {name}"})
        self.grip_cmd = g16.G_OPEN
        self.drive(Q_PARK[:5], grip=g16.G_OPEN)
        return json.dumps({"status": "gripper closed on nothing; the cube was not grasped"})

    def place(self, target=""):
        if not self.holding:
            return json.dumps({"error": "not holding anything"})
        if "box" not in str(target).lower():
            self.log["errors"] += 1
            return json.dumps({"error": f"no '{target}' visible; containers: ['box']"})
        if "box" not in self.scene:
            self.log["errors"] += 1
            return json.dumps({"error": "the box has not been seen; call look() first"})
        bx = self.scene["box"]
        rec = dict(color=self.holding, rejected=None)
        self.log["places"].append(rec)
        if not (R_MIN <= np.linalg.norm(bx - PAN_AXIS) <= R_MAX):
            rec["rejected"] = "workspace"
            return self.reject("place", "box outside the workspace")
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = self.d.qpos
        res = g10.ik_solve(self.m, scratch, np.r_[bx, PLACE_Z], joints=g16.ARM, q0=self.arm_q(),
                           target_axis=np.array([0.0, 0.0, -1.0]), axis_index=0, axis_weight=PLACE_AXIS_W,
                           max_iters=400)
        mujoco.mj_kinematics(self.m, scratch)
        pos_err = float(np.linalg.norm(scratch.site_xpos[self.I["site"]] - np.r_[bx, PLACE_Z]))
        rec["ik_mm"] = pos_err * 1000
        if pos_err * 1000 > IK_TOL_MM:
            rec["rejected"] = "ik"
            return self.reject("place", f"pose over the box is not reachable (IK error above {IK_TOL_MM:.0f} mm)")
        if self.margin(res.q) < JOINT_MARGIN:
            rec["rejected"] = "joint margin"
            return self.reject("place", "pose over the box is too close to a joint limit")
        self.drive(res.q, T=1.5, settle=0.3, grip=g16.G_CLOSE)
        self.grip_cmd = g16.G_OPEN
        self.drive(res.q, T=0.3, settle=0.5, grip=g16.G_OPEN)   # open the jaw, let the cube drop
        self.holding = None
        self.drive(Q_PARK[:5], T=1.5, settle=0.5, grip=g16.G_OPEN)
        return json.dumps({"status": "released over the box"})

    def call(self, name, args):
        fn = {"look": self.look, "pick": self.pick, "place": self.place}.get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool '{name}'"})
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return json.dumps({"error": f"bad arguments: {e}"})

    # ---------- world measurements (harness only)
    def cube_pos(self, c):
        return self.d.xpos[self.m.body(f"{c}_cube").id].copy()

    def cube_lifted(self, c):
        return bool(self.cube_pos(c)[2] - HALF >= g16.SUCCESS_LIFT_M)

    def in_box(self, c):
        p = self.cube_pos(c)
        return bool(np.all(np.abs(p[:2] - BOX_XY) <= BOX_IN) and p[2] < TABLE_TOP + WALL_H)


# ================================================================== agent
CACHE, STATS = {}, dict(requests=0, llm_calls=0, llm_s=0.0)


def chat(model, messages):
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": False, "think": False, "options": OPTIONS}
    key = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
    STATS["requests"] += 1
    if key not in CACHE:
        t0 = time.time()
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            CACHE[key] = json.load(r)["message"]
        STATS["llm_calls"] += 1
        STATS["llm_s"] += time.time() - t0
    return CACHE[key]


def run_episode(model, cmd, layout, tilt_deg=0.0):
    cid, text, want = cmd
    robot = Robot(layout, tilt_deg)
    start = {c: robot.cube_pos(c) for c in COLORS}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    calls, report, n_llm = [], None, 0
    while n_llm < MAX_LLM_CALLS and report is None:
        msg = chat(model, messages)
        n_llm += 1
        messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})
        tcs = msg.get("tool_calls") or []
        if not tcs:
            break
        for tc in tcs:
            name, args = tc["function"]["name"], tc["function"].get("arguments") or {}
            if name == "report_result":
                report = args
                break
            res = robot.call(name, args)
            calls.append((name, args, res))
            messages.append({"role": "tool", "tool_name": name, "content": res})
    if robot.holding:                                   # the episode ends: put the arm back, jaw open
        robot.grip_cmd = g16.G_OPEN
        robot.drive(Q_PARK[:5], grip=g16.G_OPEN)
    for _ in range(int(0.5 / g16.DT)):
        mujoco.mj_step(robot.m, robot.d)
    robot.cam.close()
    # ---- success: from the simulator state only
    others_in = [c for c in COLORS if c != want and robot.in_box(c)]
    success = robot.in_box(want) and not others_in
    # ---- attribution: first failing layer, in pipeline order (last pick attempt of the wanted cube)
    picks = robot.log["picks"]
    pick_args = [color_of(a.get("object")) for n, a, _ in calls if n == "pick"]
    wanted = [p for p in picks if p["color"] == want]
    layer = None
    if not success:
        if want not in pick_args:
            layer = "language"                           # never asked for the right cube
        elif not wanted:
            layer = "perception"                         # asked, but perception did not see it
        else:
            p = wanted[-1]
            err = 1000 * np.linalg.norm(p["est"] - p["true"])
            placed = [q for q in robot.log["places"] if q["color"] == want]
            if err > PERC_TOL_MM:
                layer = "perception"
            elif p["rejected"]:
                layer = "IK/limits"
            elif not p["lifted"]:
                layer = "grasp"
            elif not placed:
                layer = "language"                       # held the right cube, never asked to place it
            elif placed[-1]["rejected"]:
                layer = "IK/limits"
            else:
                layer = "place"
    claim = None if report is None else bool(report.get("success") in (True, "true", "True"))
    perc_err = [1000 * np.linalg.norm(p["est"] - p["true"]) for p in wanted]
    appr = [p["approach_margin"] for p in picks if "approach_margin" in p]
    sig = hashlib.sha1(json.dumps([(n, a) for n, a, _ in calls] + [report], sort_keys=True).encode()).hexdigest()[:8]
    moved_other = max(np.linalg.norm(robot.cube_pos(c)[:2] - start[c][:2]) for c in COLORS if c != want)
    return dict(model=model, cid=cid, success=success, layer=layer, claim=claim, calls=calls, n_llm=n_llm, sig=sig,
                perc_err=perc_err[-1] if perc_err else float("nan"), others_in=others_in,
                moved_other_cm=moved_other * 100, rejections=len(robot.log["rejections"]),
                appr_margin=appr[-1] if appr else float("nan"))


# ================================================================== stats + printing
def wilson(k, n, z=1.96):
    """Wilson score interval (same formula as lesson 25)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


LAYERS = ("language", "perception", "IK/limits", "grasp", "place")


def fmt_call(name, args, res):
    arg = ",".join(str(v) for v in args.values())
    if "rejected" in res:
        tag = "[REJ:" + ("workspace" if "workspace" in res else "IK" if "IK" in res else "limit") + "]"
    else:
        tag = "[ERR]" if "error" in res else ("[EMPTY]" if "nothing" in res else "")
    return f"{name}({arg}){tag}"


def summary_row(label, rs):
    k, n = sum(r["success"] for r in rs), len(rs)
    lo, hi = wilson(k, n)
    fails = [sum(r["layer"] == L for r in rs) for L in LAYERS]
    false_claims = sum(r["claim"] is True and not r["success"] for r in rs)
    return (f"{label:<22}{f'{k}/{n}':>7}{k/n:>7.2f}{f'[{lo:.2f}, {hi:.2f}]':>15}"
            + "".join(f"{f:>12}" for f in fails) + f"{false_claims:>8}")


HEADER = (f"{'':<22}{'success':>7}{'rate':>7}{'Wilson 95%':>15}" + "".join(f"{L:>12}" for L in LAYERS)
          + f"{'false':>8}\n{'':<22}{'':>29}{'failures attributed to the first failing layer':>60}{'claims':>8}")


def main():
    with urllib.request.urlopen("http://localhost:11434/api/version") as r:
        ollama_v = json.load(r)["version"]
    t_start = time.time()
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · opencv {cv2.__version__} · python "
          f"{platform.python_version()} · ollama {ollama_v} · options {OPTIONS} · think off")
    layouts = make_layouts(N_LAYOUTS, LAYOUT_SEED)
    print(f"{len(COMMANDS)} commands x {N_LAYOUTS} cube layouts (seed {LAYOUT_SEED}); cubes x in "
          f"{CUBE_REGION[0]} m, y in {CUBE_REGION[1]} m, >= {MIN_GAP*100:.0f} cm apart; box centre "
          f"({BOX_XY[0]*100:.0f}, {BOX_XY[1]*100:.0f}) cm")
    print("success (simulator): the requested cube rests inside the box and no other cube is in it")
    print(f"validation: grasp target {R_MIN*100:.0f}-{R_MAX*100:.0f} cm from the pan axis; every planned pose reached "
          f"within {IK_TOL_MM} mm; grasp-pose and place-pose joint goals >= {JOINT_MARGIN} rad inside the model range")
    print(f"attribution (failed episodes only), first match in pipeline order: language (no pick of the requested "
          f"cube, or no place after holding it) -> perception (not detected, or estimate > {PERC_TOL_MM:.0f} mm off) "
          f"-> IK/limits (rejected by validation) -> grasp (cube not lifted >= 4 cm) -> place (not in the box)")
    print("false claims: report_result(success=true) while the simulator says not done\n")
    for i, lay in enumerate(layouts):
        print(f"layout {i}: " + ", ".join(f"{c} ({p[0]*100:.1f}, {p[1]*100:.1f})" for c, p in sorted(lay.items())))

    # ---------------- A. end-to-end evaluation
    rows = []
    print("\n[A] end-to-end, camera as calibrated (tilt 0)")
    for model in MODELS:
        t0 = time.time()
        for cmd in COMMANDS:
            for li, lay in enumerate(layouts):
                r = run_episode(model, cmd, lay)
                r.update(layout=li, tilt=0.0)
                rows.append(r)
                trace = " -> ".join(fmt_call(*c) for c in r["calls"]) or "(no tool call)"
                print(f"{model:<11}{cmd[0]:<9}L{li} sig {r['sig']}  {trace}  => claim {r['claim']}, "
                      f"success {r['success']}{'' if r['success'] else ', layer ' + str(r['layer'])}"
                      f", perc err {r['perc_err']:.1f} mm")
        print(f"  ({model}: ~{(time.time() - t0) / 60:.1f} min wall, approximate)")
        print(f"  {model} done", file=sys.stderr, flush=True)

    print("\nper command (n = layouts)")
    print(f"{'model':<11}{'command':<10}{'text':<60}{'success':>8}")
    for model in MODELS:
        for cmd in COMMANDS:
            rs = [r for r in rows if r["model"] == model and r["cid"] == cmd[0]]
            print(f"{model:<11}{cmd[0]:<10}{cmd[1]:<60}{sum(r['success'] for r in rs):>6}/{len(rs)}")
    print("\nTotals\n" + HEADER)
    for model in MODELS:
        print(summary_row(model, [r for r in rows if r["model"] == model]))

    # per-layer conditional success (failure budget) for the sweep model
    rs = [r for r in rows if r["model"] == SWEEP_MODEL]
    n = len(rs)
    print(f"\nfailure budget, {SWEEP_MODEL}: episodes that reach each layer and pass it")
    reached = n
    prod = 1.0
    for L in LAYERS:
        failed = sum(r["layer"] == L for r in rs)
        passed = reached - failed
        p = passed / reached if reached else float("nan")
        prod *= p
        print(f"  {L:<11} reached {reached:>3}  passed {passed:>3}  p = {p:.3f}")
        reached = passed
    print(f"  product of layer pass rates = {prod:.3f} (= end-to-end {sum(r['success'] for r in rs)}/{n})")
    lo30, _ = wilson(n, n)
    print(f"  budget: 0.90 end-to-end over {len(LAYERS)} layers in series needs each layer >= "
          f"{0.9 ** (1 / len(LAYERS)):.3f}; a layer that passed {n}/{n} has Wilson 95% lower bound {lo30:.3f}")
    perr = [r["perc_err"] for r in rows if np.isfinite(r["perc_err"])]
    print(f"perception error of the requested cube, tilt 0, last pick per episode: median {np.median(perr):.2f} mm, "
          f"max {np.max(perr):.2f} mm (n = {len(perr)})")
    am = [r["appr_margin"] for r in rows if np.isfinite(r["appr_margin"])]
    print(f"joint margin of the planned approach/lift poses (validation checks only the grasp pose): median "
          f"{np.median(am):.3f} rad, min {np.min(am):.3f} rad (wrist_flex at its 1.658 rad limit); n = {len(am)}")
    print(f"other cubes pushed > 1 cm: {sum(r['moved_other_cm'] > 1 for r in rows)} of {len(rows)} episodes")

    # ---------------- B. robustness sweep: camera tilt
    sub = [c for c in COMMANDS if c[0] in SWEEP_COMMANDS]
    print(f"\n[B] unmodelled camera tilt (rotation about the horizontal axis (1, 1, 0)/sqrt 2), {SWEEP_MODEL}, "
          f"commands {', '.join(SWEEP_COMMANDS)} x {N_LAYOUTS} layouts = {len(sub) * N_LAYOUTS} episodes per value")
    sweep = {0.0: [r for r in rs if r["cid"] in SWEEP_COMMANDS]}
    for tilt in TILTS_DEG[1:]:
        out = []
        for cmd in sub:
            for li, lay in enumerate(layouts):
                r = run_episode(SWEEP_MODEL, cmd, lay, tilt)
                r.update(layout=li, tilt=tilt)
                out.append(r)
                if not r["success"]:
                    trace = " -> ".join(fmt_call(*c) for c in r["calls"]) or "(no tool call)"
                    print(f"  tilt {tilt:.1f} {cmd[0]:<7}L{li} {trace}  => claim {r['claim']}, layer {r['layer']}, "
                          f"perc err {r['perc_err']:.1f} mm")
        sweep[tilt] = out
        print(f"  tilt {tilt:.1f} done, {STATS['llm_calls']} distinct LLM calls so far", file=sys.stderr, flush=True)
    print(f"{'tilt deg':<10}{'median perc err mm':>20} " + HEADER.split("\n")[0][22:])
    for tilt in TILTS_DEG:
        e = [r["perc_err"] for r in sweep[tilt] if np.isfinite(r["perc_err"])]
        print(f"{tilt:<10.1f}{np.median(e) if e else float('nan'):>20.1f} " + summary_row("", sweep[tilt])[22:])

    allrows = rows + [r for t in TILTS_DEG[1:] for r in sweep[t]]
    print(f"\nLLM: {STATS['requests']} requests, {STATS['llm_calls']} distinct (sent to Ollama), "
          f"{STATS['requests'] - STATS['llm_calls']} answered from the in-run cache; "
          f"~{STATS['llm_s'] / 60:.1f} min in LLM calls")
    print("run signature (tool calls + reports of every episode; compare across reruns): "
          + hashlib.sha1("".join(r["sig"] for r in allrows).encode()).hexdigest()[:12])
    print(f"total wall time ~{(time.time() - t_start) / 60:.1f} min (approximate: heavily loaded machine, not a benchmark)")


if __name__ == "__main__":
    main()
