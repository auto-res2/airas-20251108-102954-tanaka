"""src/preprocess.py
Data loading & preprocessing utilities.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# -------------------------------------------------------------------------------------
# Generic helpers ---------------------------------------------------------------------
# -------------------------------------------------------------------------------------

def compute_accuracy(preds, labels):
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    return float((preds_arr == labels_arr).mean())


# -------------------------------------------------------------------------------------
# GLUE-specific processing -------------------------------------------------------------
# -------------------------------------------------------------------------------------

def _glue_tokenise(examples, tokenizer, max_length):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


# -------------------------------------------------------------------------------------
# Main dataloader builder --------------------------------------------------------------
# -------------------------------------------------------------------------------------

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train_loader, val_loader, test_loader"""

    # ------------------------- Dataset loading ----------------------------------------
    if cfg.dataset.name != "glue":
        raise ValueError("Only GLUE datasets are supported in this template")
    ds: DatasetDict = load_dataset(
        "glue",
        cfg.dataset.subset,
        cache_dir=".cache/",
    )

    # ------------------------- Few-shot / full setup ----------------------------------
    rng = random.Random(cfg.seed)
    if cfg.dataset.setup == "32-shot":
        shots = 32
        label_list = list(set(ds["train"]["label"]))
        per_label = shots // len(label_list)
        sel_indices = []
        for lab in label_list:
            idxs = [i for i, y in enumerate(ds["train"]["label"]) if y == lab]
            sel_indices.extend(rng.sample(idxs, per_label))
        ds["train"] = ds["train"].select(sel_indices)
    # For trial mode, down-sample heavily to make CI fast ------------------------------
    if cfg.mode == "trial":
        ds["train"] = ds["train"].select(range(min(64, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(64, len(ds["validation"]))))
        ds["test"] = ds["validation"]  # GLUE has no test labels; use validation

    # ------------------------- Train/val split ----------------------------------------
    train_valid = ds["train"].train_test_split(test_size=0.1, seed=cfg.seed)
    ds = DatasetDict(
        {
            "train": train_valid["train"],
            "validation": train_valid["test"],
            "test": ds["validation"],  # GLUE's official validation split used as test here
        }
    )

    # ------------------------- Tokenisation -------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache/")

    def tokenise_fn(examples):
        out = _glue_tokenise(examples, tokenizer, cfg.dataset.max_length)
        out["labels"] = examples["label"]
        return out

    ds = ds.map(tokenise_fn, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch")

    # ------------------------- DataLoaders -------------------------------------------
    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    train_loader = DataLoader(ds["train"], batch_size=cfg.training.batch_size, shuffle=cfg.dataset.shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(ds["validation"], batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ds["test"], batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader