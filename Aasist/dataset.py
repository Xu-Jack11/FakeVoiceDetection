"""本地数据集适配工具，供 AASIST 模型训练与推理使用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
import soundfile as sf


def _resample_linear(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """使用线性插值进行简单重采样，避免额外依赖。"""
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
    waveform: np.ndarray,
    max_len: int,
    training: bool,
) -> np.ndarray:
    """裁剪或补零到固定长度。"""
    current_len = waveform.shape[0]
    if current_len > max_len:
        if training:
            start = np.random.randint(0, current_len - max_len + 1)
        else:
            start = (current_len - max_len) // 2
        waveform = waveform[start : start + max_len]
    elif current_len < max_len:
        pad_width = max_len - current_len
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    return waveform


@dataclass(slots=True)
class WaveDatasetConfig:
    csv_path: Path
    audio_dir: Path
    sample_rate: int = 16000
    max_len: int = 64600
    training: bool = True


class FakeVoiceWaveDataset(Dataset):
    """读取本地 fake voice 数据集的波形样本。"""

    def __init__(self, config: WaveDatasetConfig) -> None:
        self.config = config
        self.data = pd.read_csv(config.csv_path)
        self.audio_dir = config.audio_dir
        self.sample_rate = config.sample_rate
        self.max_len = config.max_len
        self.training = config.training

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor | str]:
        row = self.data.iloc[idx]
        audio_name: str = row["audio_name"]
        audio_path = self.audio_dir / audio_name

        try:
            waveform, sr = sf.read(audio_path)
        except Exception as exc:
            raise RuntimeError(f"读取音频失败: {audio_path}") from exc

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = _resample_linear(waveform, sr, self.sample_rate)
        waveform = _pad_or_crop(waveform, self.max_len, training=self.training)

        tensor = torch.from_numpy(waveform).float()

        if "target" in row:
            target = torch.tensor(int(row["target"]), dtype=torch.long)
            return tensor, target
        return tensor, audio_name
