"""损失函数集合：Focal / Class-Balanced / Margin Softmax。"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        ce_loss = F.nll_loss(log_prob, targets, reduction="none")
        focal = (1 - prob.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma
        loss = self.alpha * focal * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    def __init__(
        self,
        class_counts: Sequence[int],
        beta: float = 0.999,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-6)
        self.register_buffer("weights", weights / weights.sum() * len(counts))
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = F.cross_entropy(logits, targets, weight=self.weights, reduction=self.reduction)
        return loss


class MarginSoftmaxLoss(nn.Module):
    def __init__(self, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits = logits.clone()
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        logits[batch_indices, targets] -= self.margin
        logits = logits * self.scale
        return self.ce(logits, targets)


def build_loss(
    name: str,
    num_classes: int = 2,
    class_counts: Optional[Sequence[int]] = None,
) -> nn.Module:
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "focal":
        return FocalLoss()
    if name in {"class_balanced", "cb"}:
        if class_counts is None:
            raise ValueError("ClassBalancedLoss 需要提供 class_counts")
        return ClassBalancedLoss(class_counts)
    if name in {"margin", "aam", "margin_softmax"}:
        return MarginSoftmaxLoss()
    raise ValueError(f"未知损失函数: {name}")
