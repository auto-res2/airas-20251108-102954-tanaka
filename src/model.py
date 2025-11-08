"""src/model.py
Model architectures for VIBERT and Sparse-VIBERT classifier.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class VIBClassifier(nn.Module):
    """Variational Information Bottleneck classifier with optional L1 sparsity."""

    def __init__(
        self,
        base_model_name: str,
        num_labels: int = 2,
        ib: bool = True,
        ib_dim: int = 128,
        beta: float = 1.0,
        alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, cache_dir=".cache/")
        self.hidden_size = self.encoder.config.hidden_size

        self.ib = ib
        self.ib_dim = ib_dim
        self.beta = beta
        self.alpha = alpha
        self.num_labels = num_labels

        if self.ib:
            self.fc_mu = nn.Linear(self.hidden_size, ib_dim)
            self.fc_logvar = nn.Linear(self.hidden_size, ib_dim)
            classifier_in = ib_dim
        else:
            classifier_in = self.hidden_size
        self.classifier = nn.Linear(classifier_in, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):  # noqa: D401
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = enc_out.pooler_output if hasattr(enc_out, "pooler_output") else enc_out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        kl_val = torch.tensor(0.0, device=pooled.device)
        l1_pen = torch.tensor(0.0, device=pooled.device)

        if self.ib:
            mu = self.fc_mu(pooled)
            logvar = self.fc_logvar(pooled)
            z = self._reparameterise(mu, logvar)
            kl_val = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=1).mean()
            l1_pen = mu.abs().mean()
        else:
            z = pooled

        logits = self.classifier(z)
        output: Dict[str, torch.Tensor] = {"logits": logits, "kl": kl_val}

        if labels is not None:
            ce = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
            loss = ce + self.beta * kl_val + self.alpha * l1_pen
            output["loss"] = loss
        return output


def build_model(cfg):
    return VIBClassifier(
        base_model_name=cfg.model.name,
        num_labels=2,
        ib=cfg.model.ib,
        ib_dim=cfg.model.ib_dim,
        beta=cfg.model.beta,
        alpha=cfg.model.get("alpha", 0.0),
    )