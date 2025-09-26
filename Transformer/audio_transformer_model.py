"""
Transformer-based audio classification models.
"""

import math
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

class PositionalEncoding(nn.Module):
    """Sine/cosine positional encoding with dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


class AudioTransformerClassifier(nn.Module):
    """Transformer encoder for audio spectrogram classification."""

    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")

        self.input_dim = input_dim
        self.pooling = pooling

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected input of shape (B, C, M, T)")
        bsz, _, n_mels, _ = x.shape
        if n_mels != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} mel bins, got {n_mels}")

        x = x.squeeze(1)  # (B, M, T)
        x = x.transpose(1, 2).contiguous()  # (B, T, M)
        x = self.input_proj(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if self.pooling == "mean":
            pooled = features.mean(dim=1)
        else:
            pooled = features[:, 0]
        return self.head(pooled)

class AudioPreprocessor:
    """Audio preprocessing with optional GPU acceleration."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_len: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = max_len
        self.target_length = int(self.sample_rate * self.max_len / self.hop_length)
        self.device = (
            torch.device(device)
            if device is not None and not isinstance(device, torch.device)
            else (device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        ).to(self.device)
        self.db_transform = torchaudio.transforms.AmplitudeToDB().to(self.device)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self._load_with_torchcodec = getattr(torchaudio, "load_with_torchcodec", None)
        self._torchcodec_warned = False

    def load_audio(self, file_path: str) -> Optional[torch.Tensor]:
        """Load audio and resample to the configured sample rate."""
        try:
            if self._load_with_torchcodec is not None:
                try:
                    waveform, sr = self._load_with_torchcodec(file_path)
                except Exception as exc:
                    if not self._torchcodec_warned:
                        warnings.warn(
                            "torchaudio.load_with_torchcodec failed; falling back to torchaudio.load."
                            " Install torchcodec for best performance.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._torchcodec_warned = True
                    waveform, sr = torchaudio.load(file_path)
            else:
                waveform, sr = torchaudio.load(file_path)
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            return None
        try:
            waveform = waveform.to(self.device)
            if sr != self.sample_rate:
                resampler = self._resamplers.get(sr)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
                    self._resamplers[sr] = resampler
                waveform = resampler(waveform)
            audio = waveform.mean(dim=0)
            return audio
        except Exception as exc:  # pragma: no cover - logging only
            print(f"Error loading {file_path}: {exc}")
            return None

    def extract_mel_spectrogram(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert waveform tensor to log-mel spectrogram."""
        if audio is None:
            return None
        waveform = audio.unsqueeze(0) if audio.dim() == 1 else audio
        if waveform.size(1) < self.n_fft:
            pad_size = self.n_fft - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_size))
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.db_transform(mel_spec)
        return mel_spec_db.squeeze(0)

    def pad_or_trim(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Normalize temporal dimension by padding or trimming."""
        if mel_spec.size(1) > self.target_length:
            mel_spec = mel_spec[:, : self.target_length]
        elif mel_spec.size(1) < self.target_length:
            pad_width = self.target_length - mel_spec.size(1)
            mel_spec = F.pad(mel_spec, (0, pad_width))
        return mel_spec

    def normalize(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Standardize spectrogram values."""
        mean = mel_spec.mean()
        std = mel_spec.std(unbiased=False)
        return (mel_spec - mean) / (std + 1e-8)

    def process_audio(self, file_path: str) -> Optional[torch.Tensor]:
        """Run the full preprocessing pipeline for an audio file."""
        audio = self.load_audio(file_path)
        if audio is None:
            return None
        mel_spec = self.extract_mel_spectrogram(audio)
        if mel_spec is None:
            return None
        mel_spec = self.pad_or_trim(mel_spec)
        mel_spec = self.normalize(mel_spec).unsqueeze(0).float()
        if self.device.type != "cpu":
            mel_spec = mel_spec.cpu()
        return mel_spec


class AudioDataset(Dataset):
    """Dataset that loads mel-spectrograms from audio files."""

    def __init__(
        self,
        csv_file: str,
        audio_dir: str,
        preprocessor: AudioPreprocessor,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        audio_name = self.data.iloc[idx]["audio_name"]
        audio_path = os.path.join(self.audio_dir, audio_name)
        mel_spec = self.preprocessor.process_audio(audio_path)
        if mel_spec is None:
            mel_spec = torch.zeros(
                1, self.preprocessor.n_mels, self.preprocessor.target_length
            )
        if "target" in self.data.columns:
            label = int(self.data.iloc[idx]["target"])
            return mel_spec, torch.tensor(label, dtype=torch.long)
        return mel_spec, audio_name
