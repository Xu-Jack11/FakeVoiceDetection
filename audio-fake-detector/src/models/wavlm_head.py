"""WavLM-based classifier head."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel


class WavLMClassifier(nn.Module):
    """
    Tiny classification head on top of WavLM transformer features.
    """

    def __init__(
        self,
        pretrained_name: str = "microsoft/wavlm-base-plus",
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.freeze_encoder = freeze_encoder
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
            feat_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths)
            feat_lengths = feat_lengths.to(device=hidden_states.device, dtype=torch.long)
            max_len = hidden_states.size(1)
            frame_mask = (
                torch.arange(max_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(hidden_states.size(0), -1)
                < feat_lengths.unsqueeze(1)
            )
            mask = frame_mask.unsqueeze(-1).type_as(hidden_states)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden_states * mask).sum(dim=1) / denom
        else:
            pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def parameter_groups(
        self,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
    ) -> List[dict]:
        """
        Create optimizer parameter groups with separate learning rates.
        """
        groups: List[dict] = []
        if not self.freeze_encoder:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            if encoder_params:
                groups.append(
                    {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay}
                )
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups
