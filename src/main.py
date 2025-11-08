"""src/main.py
Orchestrator that spawns `src.train` as a subprocess.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    original_cwd = Path(get_original_cwd())

    # ------------------------- Mode logic -------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.num_epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # ------------------------- Spawn training subprocess ----------------------------
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run_id}",
        f"mode={cfg.mode}",
        f"results_dir={cfg.results_dir}",
    ]
    subprocess.run(cmd, cwd=original_cwd, check=True)


if __name__ == "__main__":
    main()