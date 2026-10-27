"""Locate (and if needed fetch) MuJoCo Menagerie models used by the experiments.

Models are cloned once into .cache/mujoco_menagerie at the repo root (sparse checkout).
Pinned commit so every experiment sees the same model files.
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, ".cache", "mujoco_menagerie")
COMMIT = "f3b6013"
MODELS = ["robotstudio_so101", "unitree_go2", "franka_emika_panda"]


def path(model, file="scene.xml"):
    """Absolute path to a Menagerie model file, e.g. path("robotstudio_so101")."""
    if not os.path.isdir(os.path.join(CACHE, model)):
        if not os.path.isdir(CACHE):
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse",
                            "https://github.com/google-deepmind/mujoco_menagerie", CACHE], check=True)
            subprocess.run(["git", "-C", CACHE, "checkout", "-q", COMMIT], check=True)
        subprocess.run(["git", "-C", CACHE, "sparse-checkout", "set", *MODELS], check=True)
    return os.path.join(CACHE, model, file)
