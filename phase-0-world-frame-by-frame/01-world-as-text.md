# 01 · The world is a text file

> A table, two cubes and gravity: 15 lines of text. The physics engine works out the rest, step by step.

| | |
|---|---|
| **Phase** | [0 · World, frame by frame](README.md) |
| **Type** | Experiment |
| **Prerequisites** | None. Being able to read Python is enough |
| **Time** | ~30 min |
| **Tools** | MuJoCo 3.17 · Python · Mac or Colab CPU |
| **Code** | [`experiments/01-world-as-text/experiment.py`](../experiments/01-world-as-text/experiment.py) · [notebook](../notebooks/01-world-as-text.ipynb) |
| **Status** | ✅ done |

## Why it matters

The simulation models of robot dogs like Boston Dynamics Spot and Unitree Go2 are public text files. Google DeepMind's [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) collection has `boston_dynamics_spot` and `unitree_go2` folders. Inside are XML files that describe the body, the legs, the joints and the motors. The Go2 model, for example, is derived from Unitree's open URDF and has a total mass of 15.21 kg ([Menagerie · unitree_go2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2)).

These files are not toys. According to the [MuJoCo Playground](https://playground.mujoco.org/) paper, policies trained on models like these learn within minutes on a single GPU, and some of them transfer to the real robot zero-shot, that is, without any extra tuning ([Zakka et al. 2025, arXiv 2502.08844](https://arxiv.org/abs/2502.08844)).

So the first layer of physical AI is a text file. In this lesson you will write your own 15-line world, break it, and measure how the physics engine "plays" that text.

## Concept

### The recipe and the cook

A recipe is not a meal. Someone has to carry out its lines. In simulation, the recipe is **MJCF** (MuJoCo's XML format) and the cook is the **physics engine**. The engine does not interpret the recipe. At every step it solves the same equations of motion.

### Anatomy of the 15 lines

```xml
<mujoco>
 <option gravity="0 0 -9.81"/>
 <worldbody>
  <geom type="plane" size="2 2 .1"/>
  <geom name="table" type="box" size=".35 .25 .4"/>
  <body name="light" pos="-.13 0 .9">
   <freejoint/>
   <geom type="box" size=".1 .1 .1" mass=".1"/>
  </body>
  <body name="heavy" pos=".13 0 .9">
   <freejoint/>
   <geom type="box" size=".1 .1 .1" mass=".1"/>
  </body>
 </worldbody>
</mujoco>
```

| Line | What it describes |
|---|---|
| 2 | `option`: gravity −9.81 m/s² (same as the default, written out explicitly) |
| 4 | `plane`: a ground that looks infinite |
| 5 | `table`: a 70 × 50 cm box, top at 40 cm (`size` is a half-length; the lower half is below the ground) |
| 6–9, 10–13 | Two `body` elements (`light` and `heavy`), each with a 20 cm cube. `freejoint` sets the body free: 3 translations + 3 rotations |
| 8, 12 | `mass`: mass (kg). The only thing we will change in the experiment |

### What you don't write also says something

Every setting that is not in the file takes its default value ([XML reference · option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)):

- No `timestep` → **0.002 s**, 500 steps per second. At every step the speed of a free-falling cube grows by 9.81 × 0.002 = **1.96 cm/s** (semi-implicit Euler, [Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)).
- No `density` and no `viscosity` → both are **0**. MuJoCo computes no fluid forces at all: there is **no air** in this world ([Fluid forces](https://mujoco.readthedocs.io/en/stable/computation/fluid.html)).

### Why doesn't the heavier one fall faster?

Gravity pulls the heavy cube 10 times harder (F = m·g), but it is also 10 times harder to accelerate (a = F/m = g). Mass cancels out. In *Two New Sciences* (*Discorsi*, 1638) Galileo argued that in a medium without resistance, bodies of different weights fall at the same speed ([Stanford Encyclopedia of Philosophy · Galileo](https://plato.stanford.edu/entries/galileo/)). Our file has no resistance either, so the comparison is made under the right conditions.

> The story of dropping balls from the Tower of Pisa first appears in the biography by Galileo's student Viviani, published after his death. It is not in Galileo's own writings, and historians consider it unverified. A documented early drop experiment is the one Stevin and de Groot carried out in Delft in 1586 ([Britannica · Galileo](https://www.britannica.com/biography/Galileo-Galilei), [Wikipedia · Galileo's Leaning Tower of Pisa experiment](https://en.wikipedia.org/wiki/Galileo%27s_Leaning_Tower_of_Pisa_experiment)).

## Build it

### Step 1 · Setup

```bash
pip install -r requirements.txt     # mujoco==3.17.0, numpy
```

### Step 2 · Read the file, count the lines

```python
import mujoco
LINES = XML.split("\n")
print(len(LINES), "lines")                    # 15
m = mujoco.MjModel.from_xml_string(XML)
print(m.opt.timestep, m.opt.density, m.opt.viscosity)   # 0.002 0.0 0.0
```

### Step 3 · Change the code live

The experiment edits the file line by line. The masses on lines 8 and 12 change; to add air, a single attribute goes onto line 2:

```python
def make_xml(light_kg=0.1, heavy_kg=1.0, air=False):
    lines = list(LINES)
    lines[7] = lines[7].replace('mass=".1"', f'mass="{light_kg:g}"')
    lines[11] = lines[11].replace('mass=".1"', f'mass="{heavy_kg:g}"')
    if air:
        lines[1] = ' <option gravity="0 0 -9.81" density="1.2"/>'   # air density, kg/m³
    return "\n".join(lines)
```

1.2 kg/m³ is close to the density of dry air at sea level at 20 °C (1.204) ([Engineering ToolBox · air density](https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html)). The line count does not change.

### Step 4 · Drop and measure

`drop()` releases both cubes at the same moment and finds the **physics step** at which each cube's bottom face comes down to the tabletop (40 cm). Full code: [`experiment.py`](../experiments/01-world-as-text/experiment.py).

```bash
python experiments/01-world-as-text/experiment.py
```

## Results in numbers

MuJoCo 3.17.0, Apple M5, deterministic (no randomness, two runs give the same output).

**Vacuum (the file as written), light 100 g, heavy 1 kg:** both land on the table **at step 143** (0.286 s). Their height sequences before contact are bit-for-bit identical. The analytic fall time is √(2 · 0.40 / 9.81) = 0.2856 s.

**With air added (`density="1.2"`):** the heavy cube still lands at step 143, the light cube **at step 145**. When the heavy cube touches the table, the light cube is **1.10 cm** higher. With interpolation, the time difference is **4.08 ms**.

**Sweep: mass of the light cube** (heavy always 1 kg, 20 cm cubes, 40 cm drop):

| light cube | difference in vacuum | difference in air |
|---|---|---|
| 5 g | 0.00 ms | 93.44 ms |
| 10 g | 0.00 ms | 46.25 ms |
| 20 g | 0.00 ms | 22.59 ms |
| 50 g | 0.00 ms | 8.65 ms |
| 100 g | 0.00 ms | 4.08 ms |
| 200 g | 0.00 ms | 1.81 ms |
| 500 g | 0.00 ms | 0.45 ms |
| 1 kg | 0.00 ms | 0.00 ms |

In vacuum the difference is zero for all eight masses. In air the difference scales roughly with **1/mass**: halve the mass and the delay roughly doubles. That is what we expect, because the deceleration caused by air drag (k·v²/m) is divided by the mass.

**Check by hand.** For a box, MuJoCo's default fluid model uses quadratic drag, ½·ρ·A·v² (effective coefficient ≈ 1). For a 100 g, 20 cm cube, k/m = 0.5 · 1.2 · 0.04 / 0.1 = 0.24 1/m. The distance lost during the fall is about (k/m)·g²·T⁴/12 ≈ 1.28 cm (light) − 0.13 cm (heavy) ≈ **1.15 cm**. This agrees with the simulation's 1.10 cm.

## Limits

- **The air result comes from a model, not a real measurement.** The real drag coefficient of a cube facing forward is ≈ 1.05 (Hoerner, *Fluid-Dynamic Drag*, 1965); MuJoCo's effective value is close to 1. If the cube does not rotate, the real difference is of the same order.
- A real paper cube wobbles and spins. The simulated cube is rigid and falls with a fixed orientation.
- The density of a 20 cm, 100 g cube is that of an empty cardboard box (0.0125 g/cm³). The light cube was chosen to be light on purpose, so that the effect of air would be visible.
- After the cube touches the table, soft contact sinks it in by a few mm. [Lesson 03](03-soft-contact.md) will measure that.

## Exercises

1. **Easy.** Put the cubes 2 m up (`pos="... 2.4"`). In vacuum, do they still land at the same step? In air, how many cm is the difference?
2. **Easy.** Set `gravity="0 0 -1.62"` (the Moon). In vacuum, at which step do the two cubes land? Compare with the analytic time.
3. **Medium.** `density="1000"` (water). What do the cubes do? Is the density of a 100 g, 20 cm cube larger or smaller than that of water?
4. **Medium.** How many ms late is a 20 g paper cube? First estimate it from the table with the 1/mass rule, then run it. (20 cm paper cube: 6 faces × 0.04 m² × 80 g/m² ≈ 19 g.)
5. **Hard.** Add a third cube to the file: same mass, but 10 cm. Which one lands first in air? Derive the answer from k/m = ½·ρ·A/m, then measure.

## Key terms

| Term | What people say | What it actually means |
|---|---|---|
| MJCF | "MuJoCo's 3D model file" | The XML format that describes the world (bodies, joints, motors, physics settings). A description of physics, not of appearance |
| URDF | "Robot file" | The robot description format of the ROS world. Most Menagerie models were converted to MJCF from the manufacturer's URDF |
| `body` / `geom` | "Object" | `body` is the moving frame, `geom` is the collision/mass shape attached to it |
| `freejoint` | — | The joint that sets a body free with 6 degrees of freedom (3 translations, 3 rotations) |
| Default value | "No setting" | Every attribute you don't write still gets a value. `density = 0` means "no air" |

## Sources

- [MuJoCo XML reference · `option`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option): the defaults `timestep` 0.002, `density` 0, `viscosity` 0.
- [MuJoCo · Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html) and [Fluid forces](https://mujoco.readthedocs.io/en/stable/computation/fluid.html): the integrator and the fluid model.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): `boston_dynamics_spot`, `unitree_go2` and other models.
- [Zakka et al., *MuJoCo Playground*, arXiv 2502.08844](https://arxiv.org/abs/2502.08844).
- [Stanford Encyclopedia of Philosophy · Galileo](https://plato.stanford.edu/entries/galileo/) · [Britannica · Galileo](https://www.britannica.com/biography/Galileo-Galilei).
- Every number on this page comes from a run of `experiment.py` (MuJoCo 3.17.0, 2026-10-27). They match the Koddan Kasa video team's internal verification note exactly.
