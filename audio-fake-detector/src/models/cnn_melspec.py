"""Lightweight CNN classifier fed by Mel-spectrogram features."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchaudio


class MelSpecCNN(nn.Module):
    """
    Mel-spectrogram + CNN head as a fallback model.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)
        specs = self.melspec(input_values)
        specs = self.amp_to_db(specs + 1e-10)
        specs = specs.unsqueeze(1) if specs.dim() == 3 else specs
        logits = self.classifier(self.backbone(specs))
        return logits

    def parameter_groups(
        self,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
    ) -> List[dict]:
        return [{"params": self.parameters(), "lr": head_lr, "weight_decay": weight_decay}]
