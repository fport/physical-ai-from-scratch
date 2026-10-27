"""Lesson 41 · Vision-language-action models: what goes in, what comes out, and what tokenizing actions costs.

A. Artifacts (ART): download small public files from the Hugging Face Hub, never the weights:
   config.json and the JSON header of each .safetensors file (an 8-byte length + a JSON table of tensor
   names, dtypes and shapes, fetched with an HTTP Range request). From them: inputs, action chunk length,
   flow-matching steps / action bins, and parameter counts per component.
   Models: lerobot/smolvla_base, lerobot/pi0_base, openvla/openvla-7b. Files are cached under <repo>/.cache/hf41.
B. Sweep: OpenVLA-style action tokenization (per-dimension uniform bins between the 1st and 99th percentile of
   the training actions) applied to lesson 16's scripted SO-101 pick. Bins per dimension -> joint error,
   gripper-frame position error, and whether the de-tokenized actions still lift the cube when replayed
   in MuJoCo.

Run:  python experiment.py      (needs internet for part A on the first run; ~1 min)
"""
import importlib.util
import json
import os
import platform
import struct
import urllib.request

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(REPO, ".cache", "hf41")
HUB = "https://huggingface.co"

spec = importlib.util.spec_from_file_location("lesson16", os.path.join(REPO, "experiments",
                                              "16-grasping-and-pick-place", "experiment.py"))
g16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g16)


# ---------------------------------------------------------------- A. artifacts
def fetch(repo, fn, rng=None):
    """Download a file (or a byte range of it) once and cache it."""
    os.makedirs(CACHE, exist_ok=True)
    tag = "" if rng is None else f".{rng[0]}-{rng[1]}"
    local = os.path.join(CACHE, repo.replace("/", "__") + "__" + fn + tag)
    if not os.path.exists(local):
        req = urllib.request.Request(f"{HUB}/{repo}/resolve/main/{fn}")
        if rng is not None:
            req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
        with urllib.request.urlopen(req, timeout=60) as r, open(local, "wb") as f:
            f.write(r.read())
    with open(local, "rb") as f:
        return f.read()


def safetensors_header(repo, fn):
    n = struct.unpack("<Q", fetch(repo, fn, (0, 7)))[0]
    h = json.loads(fetch(repo, fn, (8, 7 + n)))
    h.pop("__metadata__", None)
    return n, h


GROUPS = [("LM output head(s)", ("lm_head",)), ("action expert", ("lm_expert", "gemma_expert")),
          ("vision encoder", ("vision_model", "vision_tower", "vision_backbone")),
          ("vision-language connector", ("connector", "multi_modal_projector", "projector")),
          ("language model", ("text_model", "language_model")),
          ("action / state / time projections", ("action_", "state_proj", "time_mlp"))]


def group_of(key):
    for name, subs in GROUPS:
        if any(s in key for s in subs):
            return name
    return "other"


def inspect(repo, files):
    counts, header_bytes, n_tensors, dtypes, heads = {}, 0, 0, set(), {}
    for fn in files:
        n, h = safetensors_header(repo, fn)
        header_bytes += n
        n_tensors += len(h)
        for k, v in h.items():
            counts[group_of(k)] = counts.get(group_of(k), 0) + int(np.prod(v["shape"]))
            dtypes.add(v["dtype"])
            if "lm_head" in k:
                heads[k] = int(np.prod(v["shape"]))
    return counts, header_bytes, n_tensors, dtypes, heads


def part_a():
    print("[A] Hub artifacts (config.json + safetensors headers only; no weights downloaded)")
    smol = json.loads(fetch("lerobot/smolvla_base", "config.json"))
    pi0 = json.loads(fetch("lerobot/pi0_base", "config.json"))
    ovla = json.loads(fetch("openvla/openvla-7b", "config.json"))
    idx = json.loads(fetch("openvla/openvla-7b", "model.safetensors.index.json"))
    ovla_files = sorted(set(idx["weight_map"].values()))

    def feats(cfg):
        return ", ".join(f"{k.replace('observation.', '')} {v['shape']}" for k, v in cfg["input_features"].items())

    print(f"\n  lerobot/smolvla_base config: inputs {feats(smol)} + language (max {smol['tokenizer_max_length']} "
          f"tokens)")
    print(f"    output action {smol['output_features']['action']['shape']} (padded to max_action_dim "
          f"{smol['max_action_dim']}); chunk_size {smol['chunk_size']}, n_action_steps {smol['n_action_steps']}, "
          f"flow-matching num_steps {smol['num_steps']}")
    print(f"    VLM {smol['vlm_model_name']}, num_vlm_layers {smol['num_vlm_layers']}, attention_mode "
          f"{smol['attention_mode']}, self_attn_every_n_layers {smol['self_attn_every_n_layers']}, "
          f"expert_width_multiplier {smol['expert_width_multiplier']}, freeze_vision_encoder "
          f"{smol['freeze_vision_encoder']}, train_expert_only {smol['train_expert_only']}")
    print(f"  lerobot/pi0_base config: inputs {feats(pi0)} + language (max {pi0['tokenizer_max_length']} tokens)")
    print(f"    output action {pi0['output_features']['action']['shape']}; chunk_size {pi0['chunk_size']}, "
          f"num_inference_steps {pi0['num_inference_steps']}; paligemma_variant {pi0['paligemma_variant']}, "
          f"action_expert_variant {pi0['action_expert_variant']}, dtype {pi0['dtype']}")
    print(f"  openvla/openvla-7b config: image_sizes {ovla['image_sizes']}, n_action_bins {ovla['n_action_bins']}, "
          f"LLM {ovla['hf_llm_id']}, {len(ovla['norm_stats'])} datasets in norm_stats, action dims "
          f"{len(next(iter(ovla['norm_stats'].values()))['action']['q01'])}")

    print(f"\n  {'parameters per component (from tensor shapes)':<46}{'smolvla_base':>15}{'pi0_base':>15}"
          f"{'openvla-7b':>15}")
    res = [inspect("lerobot/smolvla_base", ["model.safetensors"]), inspect("lerobot/pi0_base", ["model.safetensors"]),
           inspect("openvla/openvla-7b", ovla_files)]
    for name in [g[0] for g in GROUPS] + ["other"]:
        vals = [r[0].get(name, 0) for r in res]
        print(f"  {name:<46}" + "".join(f"{v / 1e6:>14.1f}M" for v in vals))
    print(f"  {'total':<46}" + "".join(f"{sum(r[0].values()) / 1e6:>14.1f}M" for r in res))
    print(f"  {'tensors / header bytes fetched':<46}" + "".join(f"{f'{r[2]} / {r[1]}':>15}" for r in res))
    print(f"  {'dtypes':<46}" + "".join(f"{','.join(sorted(r[3])):>15}" for r in res))
    for (repo, r) in zip(("smolvla_base", "pi0_base", "openvla-7b"), res):
        print(f"  LM-head tensors in {repo}: " + ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in r[4].items()))
    print("\n  values per inference call (from the configs):")
    print(f"    smolvla_base: one chunk of {smol['chunk_size']} x {smol['max_action_dim']} continuous values "
          f"({smol['num_steps']} flow steps); executes n_action_steps = {smol['n_action_steps']} before re-planning")
    print(f"    pi0_base:     one chunk of {pi0['chunk_size']} x {pi0['max_action_dim']} continuous values "
          f"({pi0['num_inference_steps']} flow steps)")
    print(f"    openvla-7b:   one action = 7 tokens, one of {ovla['n_action_bins']} bins each, decoded "
          f"autoregressively; 7 decoder passes for one control step")


# ---------------------------------------------------------------- B. tokenization sweep
CTRL_EVERY = 10                                  # 2 ms sim step x 10 = 50 Hz control, zero-order hold


def demo():
    m, d = g16.build(g16.CUBE_HOME)
    g16.reset(m, d)
    r = g16.scripted_pick(m, d, np.array([*g16.CUBE_HOME, g16.CUBE_HALF]), record=True)
    ctrl = np.array([step[3] for step in r["trajectory"]])[::CTRL_EVERY]
    return r, ctrl


def tokenize(a, lo, hi, n_bins):
    """OpenVLA-style: clip to [q01, q99], uniform bins per dimension, return bin index."""
    x = (np.clip(a, lo, hi) - lo) / np.maximum(hi - lo, 1e-9)
    return np.minimum((x * n_bins).astype(int), n_bins - 1)


def detokenize(tok, lo, hi, n_bins):
    return lo + (tok + 0.5) / n_bins * (hi - lo)


def replay(actions):
    """Run de-tokenized joint targets open loop on a fresh scene; physical success test of lesson 16."""
    m, d = g16.build(g16.CUBE_HOME)
    g16.reset(m, d)
    I = g16.ids(m)
    for a in actions:
        d.ctrl[I["act"]] = a[:5]
        d.ctrl[I["grip"]] = a[5]
        mujoco.mj_step(m, d, nstep=CTRL_EVERY)
    nf, nm = g16.jaw_contacts(m, d, I)
    bottom = d.xpos[I["cube_body"]][2] - g16.CUBE_HALF
    return bool(bottom >= g16.SUCCESS_LIFT_M and nf > 0 and nm > 0), bottom * 100


def fk_tip(m, d, site, q):
    d.qpos[:6] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[site].copy()


def part_b():
    print("\n[B] Action tokenization on lesson 16's scripted SO-101 pick (6-D joint-position targets, 50 Hz)")
    r, A = demo()
    print(f"  demonstration: {len(A)} actions x {A.shape[1]} dims ({len(A) / 50:.2f} s), original pick success "
          f"{r['success']}, cube bottom {r['lift_cm']:.2f} cm")
    lo, hi = np.percentile(A, 1, axis=0), np.percentile(A, 99, axis=0)
    m = mujoco.MjModel.from_xml_path(g16.path("robotstudio_so101"))
    d = mujoco.MjData(m)
    site = m.site("gripperframe").id
    tips = np.array([fk_tip(m, d, site, a) for a in A])
    ok, lift = replay(A)
    print(f"  replay of the un-tokenized 50 Hz actions: success {ok}, cube bottom {lift:.2f} cm")
    clipped = np.mean((A < lo) | (A > hi)) * 100
    print(f"  bins span the 1st..99th percentile per dimension; {clipped:.2f}% of action values fall outside "
          f"and are clipped")
    print(f"  {'bins':>6}{'bits/dim':>9}{'RMS joint err deg':>19}{'max joint err deg':>19}"
          f"{'mean tip err mm':>17}{'max tip err mm':>16}{'replay success':>16}{'cube bottom cm':>16}")
    for n_bins in (4, 8, 16, 32, 64, 128, 256, 1024):
        Ad = detokenize(tokenize(A, lo, hi, n_bins), lo, hi, n_bins)
        err = np.degrees(np.abs(Ad[:, :5] - A[:, :5]))
        tip_err = np.linalg.norm(np.array([fk_tip(m, d, site, a) for a in Ad]) - tips, axis=1) * 1000
        ok, lift = replay(Ad)
        print(f"  {n_bins:>6}{np.log2(n_bins):>9.0f}{np.sqrt(np.mean(err ** 2)):>19.3f}{err.max():>19.3f}"
              f"{tip_err.mean():>17.2f}{tip_err.max():>16.2f}{str(ok):>16}{lift:>16.2f}")


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · python {platform.python_version()} · "
          f"{platform.machine()}")
    try:
        part_a()
    except Exception as e:                       # offline: part B still runs
        print(f"[A] skipped, Hub not reachable: {type(e).__name__}: {e}")
    part_b()


if __name__ == "__main__":
    main()
