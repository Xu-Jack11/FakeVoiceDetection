"""
Transformer-based audio classification models.
"""

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
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


CACHE_META_FILENAME = "metadata.json"
CACHE_EXT = ".pt"


@dataclass(frozen=True)
class MelCacheConfig:
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    max_len: int

    @classmethod
    def from_preprocessor(cls, preprocessor: "AudioPreprocessor") -> "MelCacheConfig":
        return cls(
            sample_rate=preprocessor.sample_rate,
            n_mels=preprocessor.n_mels,
            n_fft=preprocessor.n_fft,
            hop_length=preprocessor.hop_length,
            max_len=preprocessor.max_len,
        )

    @classmethod
    def load(cls, path: Path) -> "MelCacheConfig":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)


def _validate_or_write_metadata(cache_dir: Path, config: MelCacheConfig, strict: bool) -> None:
    meta_path = cache_dir / CACHE_META_FILENAME
    if meta_path.exists():
        try:
            existing = MelCacheConfig.load(meta_path)
        except Exception as exc:  # pragma: no cover - metadata read failure
            message = f"Failed to read cache metadata ({meta_path}): {exc}."
            if strict:
                raise RuntimeError(message) from exc
            warnings.warn(message, RuntimeWarning)
            return

        if existing != config:
            message = (
                "Cache configuration mismatch detected. "
                "Please rebuild caches or remove the existing cache directory."
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
    else:
        config.dump(meta_path)


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

    def load_audio(self, file_path: str) -> Optional[torch.Tensor]:
        """Load audio and resample to the configured sample rate."""
        try:
            waveform, sr = torchaudio.load_with_torchcodec(file_path)
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
    """Dataset that loads mel-spectrograms from audio files with optional caching."""

    def __init__(
        self,
        csv_file: str,
        audio_dir: str,
        preprocessor: AudioPreprocessor,
        cache_dir: Optional[str | os.PathLike] = None,
        auto_save_cache: bool = True,
        strict_cache: bool = True,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.preprocessor = preprocessor
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.auto_save_cache = auto_save_cache
        self.strict_cache = strict_cache
        self.cache_config: Optional[MelCacheConfig] = None

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_config = MelCacheConfig.from_preprocessor(self.preprocessor)
            _validate_or_write_metadata(self.cache_dir, self.cache_config, strict=self.strict_cache)

    def __len__(self) -> int:
        return len(self.data)

    def _load_from_cache(self, cache_path: Path) -> Optional[torch.Tensor]:
        try:
            return torch.load(cache_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - logging only
            warnings.warn(f"Failed to load cache file {cache_path}: {exc}", RuntimeWarning)
            return None

    def _write_to_cache(self, cache_path: Path, mel_spec: torch.Tensor) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mel_spec, cache_path)
        except Exception as exc:  # pragma: no cover - logging only
            warnings.warn(f"Failed to write cache file {cache_path}: {exc}", RuntimeWarning)

    def __getitem__(self, idx: int):
        audio_name = self.data.iloc[idx]["audio_name"]
        audio_path = os.path.join(self.audio_dir, audio_name)

        mel_spec: Optional[torch.Tensor] = None
        cache_path: Optional[Path] = None

        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{audio_name}{CACHE_EXT}"
            if cache_path.exists():
                mel_spec = self._load_from_cache(cache_path)

        if mel_spec is None:
            mel_spec = self.preprocessor.process_audio(audio_path)
            if mel_spec is not None and cache_path is not None and self.auto_save_cache:
                self._write_to_cache(cache_path, mel_spec)

        if mel_spec is None:
            mel_spec = torch.zeros(
                1,
                self.preprocessor.n_mels,
                self.preprocessor.target_length,
            )

        if "target" in self.data.columns:
            label = int(self.data.iloc[idx]["target"])
            return mel_spec, torch.tensor(label, dtype=torch.long)
        return mel_spec, audio_name
