"""Lesson 42 · Build it: the validation layer between a language model's tool calls and the robot, no network.

Step 1  The argument check: the JSON-Schema subset the five tool definitions use.
Step 2  Reachability: damped-least-squares position IK on lesson 09's chain, inside the joint limits.
Step 3  The move_to and pick contracts on every target the models sent in the Experiment.
Step 4  Replay the Experiment's 28 validation-ON transcripts through our layer: same rejections?
Step 5  The harness: what a success claim is worth, from the Experiment's episode records.

The functions live in physai/foundation.py (IK uses physai/kinematics.py from lesson 09).
Run:  uv run python experiments/42-llm-agents/build.py      (~3 s, deterministic)
"""
import importlib.util
import os
import re
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.foundation import ToolLayer, check_args, claim_precision, ik_position, validate_move  # noqa: E402
from physai.kinematics import chain_from_mujoco  # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g10 = load("lesson10", "10-inverse-kinematics/experiment.py")      # reference IK (MuJoCo, lesson 10)
exp_src = open(os.path.join(HERE, "experiment.py"), encoding="utf-8").read()
TOOLS = eval(re.search(r"^TOOLS = (\[.*?^\])", exp_src, re.M | re.S).group(1))   # the Experiment's tool definitions
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
d = mujoco.MjData(m)
chain = chain_from_mujoco(m, "gripperframe")
LO, HI = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
HOME = np.array([0.0, -1.2, 0.5, 1.2, 1.68])                       # lesson 16's home pose


def reach(p):
    """Best of two seeds, like the Experiment's tool layer (which also tries the current pose)."""
    return min((ik_position(chain, p, q0, LO, HI) for q0 in (HOME, np.zeros(5))), key=lambda r: r[1])


print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

print("\n== Step 1. Argument checks against the Experiment's tool schemas ==")
cases = [("move_to", {"x_cm": 20, "y_cm": 5, "z_cm": 15}), ("move_to", {"x_cm": 20, "y_cm": 5}),
         ("move_to", {"x_cm": "20", "y_cm": 5, "z_cm": 15}), ("move_to", {"x_cm": True, "y_cm": 5, "z_cm": 15}),
         ("move_to", {"x_cm": 20, "y_cm": 5, "z_cm": 15, "speed": 2}), ("pick", {"name": "red cube"}),
         ("report_result", {"success": "yes", "message": "done"}), ("get_state", {})]
for name, args in cases:
    print(f"{name}({args}) -> {check_args(SCHEMAS[name], args) or 'ok'}")

print("\n== Step 2. Position IK on lesson 09's chain vs lesson 10's MuJoCo ik_solve ==")
print(f"{'target cm':<18s} {'ours: miss mm':>14s} {'lesson 10: miss mm':>19s}")
for p in ((20, 5, 15), (20, 0, 2), (22, 3, 1.1), (70, 0, 10), (50, 0, 1.1)):
    _, err = reach(np.array(p) / 100)
    ref = min(g10.ik_solve(m, d, np.array(p) / 100, q0=q0, max_iters=300).err for q0 in (HOME, np.zeros(5)))
    print(f"{str(p):<18s} {err * 1000:14.2f} {ref * 1000:19.2f}")

print("\n== Step 3. The move_to and pick contracts on every target the models sent ==")
for p in ((20, 5, 15), (20, 0, 2), (0, 0, 0), (20, 0, -5), (70, 0, 10), (0.2, 0.05, 0.15)):
    q, why = validate_move(p, reach)
    print(f"move_to{p}: {'accept' if why is None else 'reject: ' + why}")
for cube in ((22, 3), (50, 0)):                                     # pick: the three grasp heights of lesson 16
    worst = max(reach(np.array([cube[0], cube[1], z]) / 100)[1] for z in (1.1, 7.0, 8.5))
    print(f"pick, cube at {cube} cm: grasp points missed by up to {worst * 1000:.1f} mm -> "
          f"{'accept' if worst * 1000 <= 5.0 else 'reject'}")

print("\n== Step 4. The Experiment's validation-ON transcripts through our layer ==")
out = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
CUBE = {c[0]: c[2] for c in load("exp42", "42-llm-agents/experiment.py").COMMANDS}
CALL = re.compile(r"(\w+)\(([^)]*)\)(?:\[(\w+)\])?")
agree = total = 0
marks = {"ERR": "error", "REJ": "rejected", None: "ok"}
for line in re.findall(r"^(\S+) \[ON \] (\S+)\s+rep \d sig \w+\s+(.*?)  =>", out, re.M):
    model, cmd, calls = line
    cube_cm = np.array(CUBE[cmd]) * 100

    def move_to(x_cm, y_cm, z_cm):
        _, why = validate_move((x_cm, y_cm, z_cm), reach)
        return {"rejected": why} if why else {"status": "moved"}

    def pick(name):
        if "red" not in name:
            return {"error": f"no object named '{name}' found"}
        worst = max(reach(np.array([cube_cm[0], cube_cm[1], z]) / 100)[1] for z in (1.1, 7.0, 8.5))
        return {"rejected": "object out of reach"} if worst * 1000 > 5.0 else {"status": "pick executed"}

    def find_object(name):
        return {"name": "red cube"} if "red" in name else {"error": f"no object named '{name}' found"}

    layer = ToolLayer({"move_to": move_to, "pick": pick, "find_object": find_object,
                       "get_state": lambda: {"status": "ok"}}, SCHEMAS)
    for name, raw, mark in CALL.findall(calls):
        args = ({k: float(v) for k, v in zip(("x_cm", "y_cm", "z_cm"), raw.split(","))} if name == "move_to"
                else {"name": raw} if raw else {})
        layer.call(name, args)
        total += 1
        agree += layer.log[-1][2] == marks[mark or None]
print(f"episodes {len(re.findall(r'^\S+ \[ON \]', out, re.M))}, tool calls {total}, "
      f"our verdict = the Experiment's marker on {agree}/{total}")

print("\n== Step 5. The harness: precision of the success claim, P(done | claimed) ==")
rec = re.findall(r"^(\S+) \[(ON |OFF)\] \S+\s+rep \d .*=> claim (\w+), done (\w+)", out, re.M)
for model in ("qwen3:1.7b", "qwen2.5:7b"):
    for mode in ("OFF", "ON "):
        eps = [(c == "DONE", dn == "True") for mo, md, c, dn in rec if mo == model and md == mode]
        prec, n_claim, n_false = claim_precision(eps)
        done = sum(dn for _, dn in eps)
        print(f"{model} {mode.strip():<3s}: episodes {len(eps)}, done {done}, claimed {n_claim}, false {n_false}, "
              f"P(done | claimed) = {prec:.2f}")
