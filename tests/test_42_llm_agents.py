"""Lesson 42: the tool validation layer and the IK behind it, against MuJoCo and known answers."""
import json

import mujoco
import numpy as np
import pytest

from common.menagerie import path
from conftest import assert_build_output
from physai.foundation import ToolLayer, check_args, claim_precision, ik_position, validate_move
from physai.kinematics import chain_from_mujoco

MOVE = {"type": "object", "properties": {"x_cm": {"type": "number"}, "y_cm": {"type": "number"},
                                         "z_cm": {"type": "number"}}, "required": ["x_cm", "y_cm", "z_cm"]}


@pytest.fixture(scope="module")
def arm():
    m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
    return m, mujoco.MjData(m), chain_from_mujoco(m, "gripperframe")


@pytest.mark.parametrize("args, ok", [({"x_cm": 1, "y_cm": 2.5, "z_cm": 3}, True), ({"x_cm": 1, "y_cm": 2}, False),
                                      ({"x_cm": "1", "y_cm": 2, "z_cm": 3}, False),
                                      ({"x_cm": False, "y_cm": 2, "z_cm": 3}, False),
                                      ({"x_cm": 1, "y_cm": 2, "z_cm": 3, "w": 0}, False), ([1, 2, 3], False)])
def test_check_args(args, ok):
    assert (check_args(MOVE, args) is None) == ok


@pytest.mark.parametrize("seed", range(3))
def test_ik_reaches_a_reachable_point_inside_the_limits_and_matches_mujoco_site(arm, seed):
    m, d, chain = arm
    lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
    q_true = np.random.default_rng(seed).uniform(lo * 0.6, hi * 0.6)
    d.qpos[:5] = q_true
    mujoco.mj_kinematics(m, d)
    target = d.site_xpos[m.site("gripperframe").id].copy()
    q, err = ik_position(chain, target, np.zeros(5), lo, hi, iters=1000)
    assert err < 1e-3 and np.all(q >= lo) and np.all(q <= hi)
    d.qpos[:5] = q
    mujoco.mj_kinematics(m, d)
    assert np.linalg.norm(d.site_xpos[m.site("gripperframe").id] - target) < 1e-3


def test_validate_move_checks_height_before_reach():
    calls = []
    q, why = validate_move((20, 0, -5), lambda p: calls.append(p) or (None, 0.0))
    assert q is None and "below" in why and not calls
    q, why = validate_move((20, 0, 5), lambda p: (np.zeros(5), 0.01))
    assert q is None and "out of reach" in why


def test_tool_layer_answers_errors_as_data():
    layer = ToolLayer({"move_to": lambda x_cm, y_cm, z_cm: {"status": "moved"}}, {"move_to": MOVE})
    assert "unknown tool" in json.loads(layer.call("fly", {}))["error"]
    assert "missing" in json.loads(layer.call("move_to", {"x_cm": 1}))["error"]
    assert json.loads(layer.call("move_to", {"x_cm": 1, "y_cm": 2, "z_cm": 3})) == {"status": "moved"}
    assert [v for _, _, v in layer.log] == ["error", "error", "ok"]


def test_claim_precision():
    assert claim_precision([(True, True), (True, False), (False, True), (True, True)])[:2] == (2 / 3, 3)


def test_build_script_reproduces_its_output():
    assert_build_output("42-llm-agents")
