"""多分支 logit 融合模块。"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LogitFusionHead(nn.Module):
    """对各前端分支的 logits 进行学习型加权融合。"""

    def __init__(
        self,
        branch_names: Sequence[str],
        logit_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branch_names = list(branch_names)
        self.logit_dim = logit_dim

        layers: list[nn.Module] = []
        input_dim = logit_dim * len(self.branch_names)
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, logit_dim))
        self.network = nn.Sequential(*layers)

        self.register_parameter(
            "fusion_temperature",
            nn.Parameter(torch.ones(1)),
        )

    def forward(self, logits_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [logits_dict[name] for name in self.branch_names]
        if not features:
            raise ValueError("logits_dict 为空，无法进行融合")
        concat = torch.cat(features, dim=-1)
        fused = self.network(concat)
        return fused / self.fusion_temperature.clamp_min(1e-3)

    def extra_repr(self) -> str:
        return f"branches={self.branch_names}, logit_dim={self.logit_dim}"
