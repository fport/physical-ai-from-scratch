# 02 · Timestep: why did the cube go through the table?

> A simulation does not let time flow, it jumps step by step. And it only looks for collisions at those moments.

| | |
|---|---|
| **Phase** | [0 · World, frame by frame](README.md) |
| **Type** | Experiment |
| **Prerequisites** | [01 · The world is a text file](01-world-as-text.md) (not required, this lesson can be read on its own) |
| **Time** | ~30 min |
| **Tools** | MuJoCo 3.17 · Python · Mac or Colab CPU |
| **Code** | [`experiments/02-timestep/experiment.py`](../experiments/02-timestep/experiment.py) · [notebook](../notebooks/02-timestep.ipynb) |
| **Status** | ✅ done |

## Why it matters

Robot dogs learn to walk in simulation, not in the real world. Unitree's open training code [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) runs the Go2 simulation in **5 ms** steps (`sim.dt = 0.005`). The AI makes a decision every 4 steps (`decimation = 4`): 50 decisions per second. At every step in between, torque is applied to the joints ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)).

This number is not picked at random. The MuJoCo documentation describes the timestep as perhaps the most important parameter the user can set ([Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)). The larger the step, the faster the simulation, because you cover the same second with less computation. But past a certain point the physics starts giving wrong answers. In this lesson you will find that point with your own eyes: a cube dropped from 1 metre will pass through a 1 cm thick tabletop.

## Concept

### Time is not continuous, it is frame by frame

A physics engine works like a film projector. It computes the state of the world (positions, velocities) at one instant, then moves time forward by `dt` and computes it again. In MuJoCo this number is called `timestep`, and its default is **0.002 s** (2 ms), i.e. 500 steps per second ([XML reference · option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)).

At every step, MuJoCo's default integrator (semi-implicit Euler) does two things ([Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)):

```
v ← v + a · dt      # first the velocity is updated with the acceleration
x ← x + v · dt      # then the position is updated with the NEW velocity
```

So in one step the cube jumps by **velocity × dt**. Where it is in between is never computed.

### Collisions are only checked at step instants

This is the real issue. MuJoCo finds contacts at every step from the positions at that instant. It does not check the path the cube swept between two steps. If, as the cube approaches the tabletop, it is above the tabletop at one step and more than halfway inside it at the next, the engine does find a contact, but it picks the shortest way to push the cube out. If that shortest way points down, the physics pushes the cube **below** the tabletop.

With even larger steps, the cube jumps completely below the tabletop in a single step and the contact is never seen. In the game and simulation world this is called **tunneling**.

## Build it

### Step 1 · Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # mujoco==3.17.0, numpy
```

### Step 2 · Scene: floor, 1 cm tabletop, 5 cm cube

```python
XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <geom name="table" type="box" size="0.12 0.12 0.005" pos="0 0 0.295"/>
    <body name="cube" pos="0 0 1.30">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
```

In MJCF, `size` means half-length. The tabletop is 24 × 24 cm, 1 cm thick, with its top face at 30 cm. The cube is 5 cm, its centre at 1.30 m, so its bottom face is 97.5 cm above the tabletop.

### Step 3 · Change one line: `timestep`

```python
import mujoco, numpy as np

def drop(dt_ms, sim_sec=1.5):
    m = mujoco.MjModel.from_xml_string(XML)
    m.opt.timestep = dt_ms / 1000          # <- the only number we change
    d = mujoco.MjData(m)
    z = []
    while d.time < sim_sec - 1e-12:
        mujoco.mj_step(m, d)
        z.append(d.body("cube").xpos[2])
    return np.array(z)

for dt in [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
    z = drop(dt)
    print(f"dt {dt:2d} ms · final height {z[-1]*100:5.1f} cm ·", "WENT THROUGH" if z[-1] < 0.25 else "stayed")
```

To cover the same 1.5 seconds, the number of steps changes too (750 at 2 ms, 150 at 10 ms). This is a direct consequence of `dt`, not a separate variable.

### Step 4 · Why? Look at the first overlap

For every run, the full script ([`experiment.py`](../experiments/02-timestep/experiment.py)) also prints: whether, at the step where the cube first overlaps the tabletop, its centre is above or below the tabletop's mid-plane (29.5 cm), and whether the contact was seen at all.

```bash
python experiments/02-timestep/experiment.py
```

## Results in numbers

MuJoCo 3.17.0, Apple M5, deterministic. The same script was run twice and gave the same output. The numbers also match the independent run in the [verification note](#sources) exactly.

The cube hits the tabletop at **4.37 m/s** (√(2 · 9.81 · 0.975)). At 2 ms one step is **0.87 cm**, at 10 ms **4.37 cm**.

| dt (ms) | steps/s | distance per step (cm) | cube centre at first overlap (cm) | contact seen? | result |
|---|---|---|---|---|---|
| 2 | 500 | 0.87 | 31.99 | yes | stayed |
| 4 | 250 | 1.75 | 32.43 | yes | stayed |
| 6 | 166 | 2.62 | 32.00 | yes | stayed |
| 8 | 125 | 3.50 | 29.80 | yes | stayed (barely) |
| **10** | 100 | 4.37 | **28.47** | yes | **went through** |
| 12 | 83 | 5.25 | 30.69 | yes | stayed |
| **15** | 66 | 6.56 | **27.36** | yes | **went through** |
| 20 | 50 | 8.75 | 30.72 | yes | stayed |
| 25 | 40 | 10.93 | — | **no** | went through |
| 30 | 33 | 13.12 | — | no | went through |
| 40 | 25 | 17.49 | — | no | went through |
| 50 | 20 | 21.87 | — | no | went through |

The tabletop's mid-plane is at 29.5 cm. The rule holds across the whole table: **if, at the step where the cube first overlaps the tabletop, its centre is below the mid-plane, it goes through; if it is above, it stays.**

Three observations:

1. **The rule "it is safe if the step is thinner than the tabletop" is wrong.** At 4, 6 and 8 ms one step (1.75 / 2.6 / 3.5 cm) is larger than the tabletop's 1 cm thickness, yet the cube stayed. It also stayed at 12 and 20 ms.
2. **At 10 and 15 ms the contact is caught, but resolved in the wrong direction.** In the 10 ms run, at t = 0.46 s the cube is 1.97 cm inside the tabletop and the z component of the contact normal is −1.00: MuJoCo pushes the cube down. The contact force is computed correctly, just towards the wrong side.
3. **25 ms and above is real tunneling.** The cube jumps completely below the tabletop in a single step, and the engine never sees the contact.

The result does not grow smoothly with `dt` (12 and 20 stay), because which side the cube ends up on depends on how close to the tabletop the last contact-free step lands. Even 2 ms does not catch it "comfortably": soft contact sinks the cube in deep, the centre goes down to at most 29.60 cm, only 0.1 cm above the mid-plane, and comes back up. (Soft contact is the topic of [lesson 03](03-soft-contact.md).)

## Limits

- The scene is simplified: no robot arm, only the floor, the tabletop and the cube. The scene in the video also had an arm, but its control was zero; the results are the same.
- One run per `dt`. Because MuJoCo is deterministic, running again gives the same result. This does not mean "every time", it means "for every value I tried".
- Nothing above 50 ms was tried.
- The rule was measured for this geometry (1 cm tabletop, 5 cm cube, 97.5 cm drop). With a different thickness or speed the thresholds change.

## Exercises

1. **Easy.** Make the tabletop 4 cm thick (`size="0.12 0.12 0.02"`, and adjust `pos` so the top face stays at 30 cm). At which `dt` values does the cube go through? Check the rule (is the centre below the mid-plane at first overlap) against the new mid-plane.
2. **Medium.** Put the cube 3 m up. What is the impact speed in m/s? Does the threshold `dt` change? Estimate first, then run it.
3. **Medium.** Set `m.opt.integrator` to `mujoco.mjtIntegrator.mjINT_RK4`. Does the table change? If not, why (hint: is the problem in the integrator, or in the timing of the collision check)?
4. **Hard.** Write a function that, for a given drop height, finds the **smallest** `dt` at which the cube goes through, at 0.1 ms resolution. Plot this threshold as the height goes from 0.25 m to 3 m. Why is the threshold not a smooth curve?

## Key terms

| Term | What people say | What it actually means |
|---|---|---|
| `timestep` (dt) | "The speed of the simulation" | How far the physics engine moves time forward in one step. A smaller dt is more accurate, but means more computation for the same second |
| Semi-implicit Euler | "A simple integrator" | The method that updates the velocity first, then the position with the new velocity. MuJoCo's default |
| Tunneling | "A physics bug" | The contact is never seen because a fast body gets to the other side of a thin obstacle in a single step |
| Contact normal | "The collision direction" | The direction the engine picks to separate two bodies. If the body has gone in deep enough, it can point the opposite way |
| Decimation | — | In RL training, how many physics steps pass between the policy's decisions. 4 in `unitree_rl_gym` |

## Sources

- [MuJoCo XML reference · `option/timestep`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option): default 0.002 s.
- [MuJoCo · Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html): the step loop and semi-implicit Euler; the importance of the timestep.
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py): `sim.dt = 0.005`, `decimation = 4`. The Go2 config does not change `dt` and sets `decimation` to 4 again (read on 2026-10-27).
- Every number on this page comes from a run of `experiment.py` (MuJoCo 3.17.0, 2026-10-27). The contact penetration (1.97 cm), the contact normal (−1.00) and the lowest centre at 2 ms (29.60 cm) were read in the same scene with a separate probe script. The numbers also match the Koddan Kasa video team's internal verification note dated 2026-10-25 exactly.
