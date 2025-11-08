"""src/train.py
Complete training script.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------------------------------------------------------------
# Make the project root importable no matter where Hydra changes cwd to ----------------
# -------------------------------------------------------------------------------------
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model  # noqa: E402  pylint: disable=wrong-import-position
from src.preprocess import build_dataloaders, compute_accuracy  # noqa: E402  pylint: disable=wrong-import-position


# -------------------------------------------------------------------------------------
# Utilities ---------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: DictConfig) -> torch.device:
    if torch.cuda.is_available() and cfg.hardware.device == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------------------------------------
# Epoch runner ------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def run_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    is_train: bool,
    cfg: DictConfig,
    epoch_idx: int,
    enable_wandb: bool,
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    """Runs one epoch and returns aggregated metrics and the updated global_step."""
    mode = "train" if is_train else "val"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_kl = 0.0
    preds: List[int] = []
    labels: List[int] = []

    step_limit = 2 if cfg.mode == "trial" else None

    loop = tqdm(enumerate(dataloader), total=len(dataloader), disable=not enable_wandb)
    for step, batch in loop:
        if step_limit is not None and step >= step_limit:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        labels_batch = batch.pop("labels")

        with torch.cuda.amp.autocast(enabled=cfg.hardware.fp16):
            outputs = model(**batch, labels=labels_batch)
            loss = outputs["loss"]
            logits = outputs["logits"]
            kl_value = outputs.get("kl", torch.tensor(0.0, device=device))

        if is_train:
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        batch_size = labels_batch.size(0)
        total_loss += loss.item() * batch_size
        total_kl += kl_value.item() * batch_size
        preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        labels.extend(labels_batch.detach().cpu().tolist())

        if enable_wandb:
            wandb.log({f"{mode}_loss_step": loss.item()}, step=global_step)
        global_step += 1

    n_samples = len(preds)
    metrics = {
        f"{mode}_loss": total_loss / n_samples,
        f"{mode}_acc": compute_accuracy(preds, labels),
        f"{mode}_kl": total_kl / n_samples,
    }
    return metrics, global_step


# -------------------------------------------------------------------------------------
# Evaluation helper -------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def evaluate_model(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    labels_list: List[int] = []
    total_loss = 0.0
    total_kl = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop("labels")
            with torch.cuda.amp.autocast(enabled=cfg.hardware.fp16):
                outputs = model(**batch, labels=labels_batch)
                loss = outputs["loss"]
                logits = outputs["logits"]
                kl_value = outputs.get("kl", torch.tensor(0.0, device=device))
            batch_size = labels_batch.size(0)
            total_loss += loss.item() * batch_size
            total_kl += kl_value.item() * batch_size
            preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            labels_list.extend(labels_batch.detach().cpu().tolist())

    metrics = {
        "test_loss": total_loss / len(preds),
        "test_acc": compute_accuracy(preds, labels_list),
        "test_kl": total_kl / len(preds),
    }
    return metrics, preds, labels_list


# -------------------------------------------------------------------------------------
# Training routine --------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def train_once(cfg: DictConfig, *, enable_wandb: bool = True) -> Tuple[float, Dict[str, float]]:
    """Trains the model once using the hyper-parameters inside cfg.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (best validation accuracy, dictionary with best metrics across train/val/test)
    """

    device = get_device(cfg)
    set_seed(cfg.seed)

    # ----------------------------------------- Data
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ----------------------------------------- Model
    model = build_model(cfg)
    model.to(device)

    # ----------------------------------------- Optimiser & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
    )
    total_train_steps = max(1, len(train_loader) * cfg.training.num_epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_train_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.hardware.fp16)

    # ----------------------------------------- WandB initialisation
    wandb_run = None
    if enable_wandb and cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"WandB URL: {wandb_run.url}")

    # ----------------------------------------- Training loop
    best_val_acc = -math.inf
    best_metrics: Dict[str, float] = {}
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        train_metrics, global_step = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            is_train=True,
            cfg=cfg,
            epoch_idx=epoch,
            enable_wandb=enable_wandb and wandb_run is not None,
            global_step=global_step,
        )

        with torch.no_grad():
            val_metrics, global_step = run_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=device,
                is_train=False,
                cfg=cfg,
                epoch_idx=epoch,
                enable_wandb=enable_wandb and wandb_run is not None,
                global_step=global_step,
            )

        epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        if wandb_run is not None:
            wandb.log(epoch_metrics, step=global_step)

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_metrics = epoch_metrics

    # ----------------------------------------- Final evaluation on test-set
    test_metrics, test_preds, test_labels = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        cfg=cfg,
    )

    best_metrics.update(test_metrics)

    # Confusion matrix for later visualisation
    cm = confusion_matrix(test_labels, test_preds).tolist()
    best_metrics["test_confusion_matrix"] = cm

    # ----------------------------------------- Write to WandB summary
    if wandb_run is not None:
        for k, v in best_metrics.items():
            wandb_run.summary[k] = v
        wandb_run.finish()

    return best_val_acc, best_metrics


# -------------------------------------------------------------------------------------
# Optuna objective --------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def objective(trial, *, base_cfg: DictConfig) -> float:
    # Deep-copy cfg via OmegaConf container --------------------------------
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    # Sample hyper-parameters ----------------------------------------------
    for hp_name, hp_cfg in cfg.optuna.search_space.items():
        if hp_cfg["type"] == "loguniform":
            sampled = trial.suggest_float(hp_name, hp_cfg["low"], hp_cfg["high"], log=True)
        elif hp_cfg["type"] == "categorical":
            sampled = trial.suggest_categorical(hp_name, hp_cfg["choices"])
        else:
            raise ValueError(f"Unknown hp type: {hp_cfg['type']}")

        # Write back to corresponding section ------------------------------
        if hp_name in cfg.training:
            cfg.training[hp_name] = sampled
        elif hp_name in cfg.model:
            cfg.model[hp_name] = sampled
        else:
            cfg[hp_name] = sampled

    # Disable WandB inside Optuna trials -----------------------------------
    cfg.wandb.mode = "disabled"

    val_acc, _ = train_once(cfg, enable_wandb=False)
    return val_acc


# -------------------------------------------------------------------------------------
# Hydra entry-point -------------------------------------------------------------------
# -------------------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):  # noqa: C901, pylint: disable=too-many-branches
    # Restore original working directory to keep paths consistent ----------
    original_cwd = Path(get_original_cwd())

    # Ensure HF caches go to .cache/ ---------------------------------------
    os.environ["TRANSFORMERS_CACHE"] = str(original_cwd / ".cache")
    os.environ["HF_DATASETS_CACHE"] = str(original_cwd / ".cache")

    # --------------------------- Mode adjustments -------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.num_epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # --------------------------- Optuna search (if any) -------------------
    best_cfg = cfg
    if cfg.optuna.n_trials > 0:
        import optuna

        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(lambda t: objective(t, base_cfg=cfg), n_trials=cfg.optuna.n_trials)

        # Merge best params back into cfg ----------------------------------
        for k, v in study.best_params.items():
            if k in best_cfg.training:
                best_cfg.training[k] = v
            elif k in best_cfg.model:
                best_cfg.model[k] = v
            else:
                best_cfg[k] = v
        print("Optuna finished. Best val_acc =", study.best_value)
        print("Best params:", study.best_params)

    # --------------------------- Final training ---------------------------
    _, best_metrics = train_once(best_cfg, enable_wandb=(best_cfg.wandb.mode != "disabled"))

    # --------------------------- Save metrics locally ---------------------
    results_dir = original_cwd / Path(cfg.results_dir) / cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as fp:
        json.dump(best_metrics, fp, indent=2)

    print("Training finished. Metrics written to", results_dir / "metrics.json")


if __name__ == "__main__":
    main()