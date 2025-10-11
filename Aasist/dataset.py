"""本地数据集适配工具，提供 VAD、电话增广与 RawBoost。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import soundfile as sf

from .data.telephony_augs import RandomTelephonyAugmenter


def _resample_linear(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    if waveform.size == 0:
        return waveform
    duration = waveform.shape[0] / float(orig_sr)
    target_len = int(round(duration * target_sr))
    if target_len <= 0:
        target_len = 1
    orig_indices = np.linspace(0.0, waveform.shape[0] - 1, num=waveform.shape[0])
    target_indices = np.linspace(0.0, waveform.shape[0] - 1, num=target_len)
    return np.interp(target_indices, orig_indices, waveform).astype(np.float32)


def _pad_or_crop(
    waveform: Tensor,
    target_len: int,
    training: bool,
) -> Tensor:
    current_len = waveform.shape[-1]
    if current_len >= target_len:
        if training:
            start = random.randint(0, max(current_len - target_len, 1))
        else:
            start = max((current_len - target_len) // 2, 0)
        return waveform[..., start : start + target_len]
    pad = target_len - current_len
    return F.pad(waveform, (0, pad))


class EnergyVAD:
    def __init__(
        self,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        energy_threshold: float = 0.6,
        min_speech_seconds: float = 0.8,
    ) -> None:
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.energy_threshold = energy_threshold
        self.min_speech_seconds = min_speech_seconds

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        frame_len = int(sample_rate * self.frame_ms / 1000)
        hop = int(sample_rate * self.hop_ms / 1000)
        if frame_len <= 0 or hop <= 0 or waveform.numel() < frame_len:
            return waveform
        unfolded = waveform.unfold(-1, frame_len, hop)
        energies = unfolded.pow(2).mean(dim=-1)
        threshold = energies.mean() * self.energy_threshold
        mask = energies > threshold
        if not mask.any():
            return waveform
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        start = int(indices[0].item() * hop)
        end = int(min(waveform.shape[-1], indices[-1].item() * hop + frame_len))
        trimmed = waveform[..., start:end]
        if trimmed.numel() < self.min_speech_seconds * sample_rate:
            return waveform
        return trimmed


@dataclass(slots=True)
class WaveDatasetConfig:
    csv_path: Path
    audio_dir: Path
    sample_rate: int = 16000
    max_len: int = 64600
    training: bool = True
    min_chunk_seconds: float = 2.0
    max_chunk_seconds: float = 6.0
    eval_chunk_seconds: float = 4.0
    apply_vad: bool = True
    telephony_aug: bool = True


class FakeVoiceWaveDataset(Dataset):
    """读取本地 fake voice 数据集的波形样本，支持电话链路增广。"""

    def __init__(self, config: WaveDatasetConfig) -> None:
        self.config = config
        self.data = pd.read_csv(config.csv_path)
        self.audio_dir = config.audio_dir
        self.sample_rate = config.sample_rate
        self.max_len = config.max_len
        self.training = config.training
        self.vad = EnergyVAD() if config.apply_vad else None
        self.telephony_aug = (
            RandomTelephonyAugmenter() if (self.training and config.telephony_aug) else None
        )

    def __len__(self) -> int:
        return len(self.data)

    def _load_waveform(self, audio_path: Path) -> Tensor:
        waveform, sr = sf.read(audio_path)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = _resample_linear(waveform, sr, self.sample_rate)
        tensor = torch.from_numpy(waveform).float()
        return tensor

    def _apply_vad(self, waveform: Tensor) -> Tensor:
        if self.vad is None:
            return waveform
        return self.vad(waveform, self.sample_rate)

    def _random_chunk(self, waveform: Tensor) -> Tensor:
        if self.training:
            chunk_seconds = random.uniform(
                self.config.min_chunk_seconds, self.config.max_chunk_seconds
            )
        else:
            chunk_seconds = self.config.eval_chunk_seconds
            if chunk_seconds <= 0:
                return waveform
        chunk_samples = int(chunk_seconds * self.sample_rate)
        chunk_samples = min(chunk_samples, self.max_len)
        return _pad_or_crop(waveform, chunk_samples, training=self.training)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | str | int]:
        row = self.data.iloc[idx]
        audio_name: str = row["audio_name"]
        audio_path = self.audio_dir / audio_name

        waveform = self._load_waveform(audio_path)
        waveform = self._apply_vad(waveform)
        waveform = self._random_chunk(waveform)

        if self.telephony_aug is not None:
            waveform = self.telephony_aug(waveform.unsqueeze(0), self.sample_rate).squeeze(0)

        sample: Dict[str, Tensor | str | int] = {
            "waveform": waveform.float(),
            "utt_id": audio_name,
            "length": torch.tensor(int(waveform.numel()), dtype=torch.long),
        }

        if "target" in row:
            sample["target"] = torch.tensor(int(row["target"]), dtype=torch.long)

        return sample
