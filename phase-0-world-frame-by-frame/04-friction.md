# 04 · Friction: does a heavier cube slip later?

> The heavier one is held more tightly, but it is also pulled just as much harder. Mass cancels out: tan θ = μ.

| | |
|---|---|
| **Phase** | [0 · World, frame by frame](README.md) |
| **Type** | Experiment |
| **Prerequisites** | [01](01-world-as-text.md), [02](02-timestep.md); [03](03-soft-contact.md) helps but is not required |
| **Time** | ~40 min |
| **Tools** | MuJoCo 3.17 · Python · Mac or Colab CPU |
| **Code** | [`experiments/04-friction/experiment.py`](../experiments/04-friction/experiment.py) · [notebook](../notebooks/04-friction.ipynb) |
| **Status** | ✅ done |

## Why it matters

How many different floors does a robot dog see while it learns to walk? Unitree's open training code (`unitree_rl_gym`, built on ETH Zurich's `legged_gym`) spells out the answer: **4,096** parallel copies are trained at the same time (`num_envs = 4096`), and the floor friction of each copy is picked at random between **0.5 and 1.25** (`randomize_friction = True`, `friction_range = [0.5, 1.25]`). The Go2 config does not change these values ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [the same lines in legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py); read on 2026-10-27).

Why randomize the friction coefficient and not the weight? Because the number that really decides whether a foot slips is μ. A robot hand holding a box without dropping it relies on the same limit: the contact force must stay inside the **friction cone** (Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Chapter 12). In this lesson you will measure that limit on a ramp and see a place where your intuition will mislead you: a cube 100 times heavier slips at the same angle as the light one.

## Concept

### Friction has a ceiling

In the Coulomb model, the static friction force between two surfaces cannot exceed a ceiling proportional to the pressing force N: **f ≤ μ·N** ([OpenStax *University Physics* 1, §6.2](https://openstax.org/books/university-physics-volume-1/pages/6-2-friction)). When the pull exceeds this ceiling, the body slips.

### Two forces on a ramp

For a cube of mass m on a ramp at angle θ:

| | formula |
|---|---|
| What pulls the cube downhill (the component of its weight along the ramp) | m · g · sin θ |
| The friction ceiling (μ × pressing force) | μ · m · g · cos θ |

At the slip threshold the two are equal:

```
m · g · sin θ = μ · m · g · cos θ   ⇒   tan θ = μ
```

m appears on both sides. **Mass cancels out.** The heavy cube is held more tightly, but it is pulled harder in exactly the same proportion. The slip angle depends only on μ.

### How MuJoCo computes friction

- The `friction` attribute of every `geom` is three numbers: sliding, torsional, rolling. The first is μ ([XML reference · geom](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom)).
- When two geoms touch, unless `priority` is given, MuJoCo uses the **larger** of the two μ values. We gave both the same μ.
- The contact force is limited by a friction cone. The default `cone="pyramidal"` approximates the cone with a pyramid; `elliptic` is closer to the real cone ([XML reference · option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option), [Computation · Contact](https://mujoco.readthedocs.io/en/stable/computation/index.html)).
- MuJoCo has a single μ. The real-world distinction "static friction is larger than kinetic friction" does not exist in this model.

## Build it

### Step 1 · Scene: tilted ramp + 8 cm cube

```python
HALF = 0.04                    # cube half-size -> 8 cm cube
RAMP = (0.40, 0.15, 0.01)      # ramp half-sizes -> 80 × 30 × 2 cm

def xml(theta_deg, mu, mass, cone="pyramidal"):
    return f"""
<mujoco>
  <option cone="{cone}"/>
  <worldbody>
    <body name="ramp" euler="0 {-theta_deg} 0">
      <geom type="box" size="{RAMP[0]} {RAMP[1]} {RAMP[2]}" friction="{mu} 0.005 0.0001"/>
    </body>
    <body name="cube">
      <freejoint/>
      <geom type="box" size="{HALF} {HALF} {HALF}" mass="{mass}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""
```

The cube is placed **at rest**, touching the ramp surface and with the same orientation as the ramp. Then 2 s of simulation run, and the cube's downhill speed along the ramp is recorded.

### Step 2 · What does "slipping" mean?

Here is the first trap. MuJoCo's contact is soft (see [lesson 03](03-soft-contact.md)): even below the threshold, the cube creeps very slowly at a **constant speed**. In this experiment, the largest creep speeds measured at least 1° below the threshold are: μ 0.3 → 0.3 cm/s, μ 0.5 → 0.6 cm/s, μ 0.8 → 1.42 cm/s.

Real slipping, on the other hand, **accelerates**: a = g·(sin θ − μ·cos θ). So two conditions are required at once:

```python
V_TH, A_TH = 2.0, 0.5          # cm/s: slip criterion

def slips(theta_deg, mu, mass, cone="pyramidal"):
    """Slipping <=> both fast (v2 > 2 cm/s) and accelerating (v2 - v1 > 0.5 cm/s)."""
    v1, v2 = speeds(theta_deg, mu, mass, cone)   # mean downhill speed over 0.5–1.0 s and 1.5–2.0 s
    return v2 > V_TH and (v2 - v1) > A_TH
```

With only the speed threshold (v2 > 2 cm/s), the slip angle on the pyramidal cone would come out as 16.35° at μ 0.3 and 26.25° at μ 0.5: ~0.3° below theory, because just below arctan μ the cube glides at a constant 2–3 cm/s without accelerating.

### Step 3 · Sweep

**Slip angle:** the smallest angle that passes the criterion and for which every angle above it also passes. Grid 0.05°, from 3° below to 1° above the theoretical value for each μ (~80 angles). 3 μ × 3 masses = 9 sweeps.

```bash
python experiments/04-friction/experiment.py      # ~10 s on Apple M5
python experiments/04-friction/experiment.py --quick   # μ = 0.5 only
```

## Results in numbers

MuJoCo 3.17.0, 2 ms timestep, default pyramidal cone, deterministic.

| μ | theory arctan μ | 0.1 kg | 1 kg | 10 kg |
|---|---|---|---|---|
| 0.3 | 16.70° | 16.75° | 16.75° | 16.75° |
| 0.5 | 26.57° | 26.60° | 26.60° | 26.60° |
| 0.8 | 38.66° | 38.70° | 38.70° | 38.70° |

- In every row the three masses give the **same** angle (spread 0.000°). A cube 100 times heavier (0.1 → 10 kg) slips at the same angle.
- The measured value deviates from theory by at most **+0.05°**, i.e. one grid step.
- Only μ changes the angle.

In the static sweep, the difference between the speed sequences of the 0.1 kg and 10 kg cubes is of the order of 1.2·10⁻¹⁵ cm/s: numerical rounding level.

## Pitfall: the friction cone can depend on direction

In the video and in all the runs above, the cube is aligned with the ramp, not rotated. When the cube is **rotated 45°** about the ramp normal, slipping on the pyramidal cone comes early and irregularly: μ 0.5 → 26.25° (−0.32°), μ 0.8 → 37.70° (−0.96°), constant slides of 4–16 cm/s show up below the threshold, and the criterion is not monotonic. On the elliptic cone the unrotated cube gives the same angles.

The mechanism behind this difference (the direction dependence of the pyramid, or the box-box contact points) was not investigated. **Not verified**; it is written here only as an observation. In section 5 of the notebook you can run the pyramidal and elliptic cones side by side.

## Limits

- Single-μ model: on real surfaces static friction is larger than kinetic friction, and part of the "sudden slip" feeling comes from that. This simulation does not have that effect.
- The μ values of real materials (wood on wood, rubber on concrete, etc.) are deliberately not given in this lesson; sources differ widely.
- For an 8 cm cube the tipping limit is also tan θ = 1 (45°). For μ ≥ 1, whether the cube slips or tips over was not measured separately.

## Exercises

1. **Easy.** If μ = 1, at how many degrees does the cube slip? Estimate first, then run it in section 6 of the notebook with `MU = 1.0`. (The answer is at the end of the page.)
2. **Easy.** Set `MASS = 1000`. Does the table change?
3. **Medium.** Reduce the criterion to the speed threshold alone (v2 > 2 cm/s) and regenerate the table. Which rows slip, and why?
4. **Medium.** Give the ramp and the cube different μ values (e.g. ramp 0.3, cube 0.8). At what angle does it slip? Explain with MuJoCo's "pick the larger" rule. Then add `priority="1"` to one geom and measure again.
5. **Hard.** Unitree's training range is μ ∈ [0.5, 1.25]. Compute the slip angles for the most slippery and the grippiest floor in this range. If a floor below the range (e.g. μ = 0.2) was never seen in training, what would you expect from the policy on such a floor? Note this question down for [lesson 37 · domain randomization](../phase-7-sim-to-real/37-domain-randomization.md).

## Key terms

| Term | What people say | What it actually means |
|---|---|---|
| μ (friction coefficient) | "How rough the surface is" | The ratio of the friction ceiling to the pressing force. The tangent of the slip angle |
| Friction cone | — | The set of contact force directions that friction can support. Once the force leaves the cone, slipping starts |
| Pyramidal / elliptic cone | — | The two ways MuJoCo approximates the cone. Pyramidal is the default |
| Creep | "A simulation bug" | The constant, slow slide seen below the threshold because of soft contact |
| `friction_range` | — | In RL training, the random range from which each environment's friction is picked (domain randomization) |

## Sources

- [OpenStax, *University Physics* Volume 1, §6.2 Friction](https://openstax.org/books/university-physics-volume-1/pages/6-2-friction): Coulomb friction and the inclined plane.
- [MuJoCo XML reference · `geom/friction`, `option/cone`](https://mujoco.readthedocs.io/en/stable/XMLreference.html) and [Computation · Contact](https://mujoco.readthedocs.io/en/stable/computation/index.html).
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py) and [`legged_gym`](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py): `num_envs = 4096`, `randomize_friction = True`, `friction_range = [0.5, 1.25]`.
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Chapter 12 (Grasping and Manipulation): the friction cone and grasping.
- Every number on this page comes from runs of `experiment.py` and the notebook (MuJoCo 3.17.0, 2026-10-27). They match the Koddan Kasa video team's internal verification note exactly.

## Exercise answers

1. Theory: arctan 1 = **45°**. The notebook measures **44.90°** (−0.10°). In this lesson the cube slips rather than tipping over; but since the tipping limit for an 8 cm cube is also 45°, this was not measured separately.
