"""电话链路与 RawBoost 风格的数据增强。"""

from __future__ import annotations

import random
from typing import Iterable, Optional

import torch

try:  # pragma: no cover
    import torchaudio
    from torchaudio import functional as AF
except Exception:  # pragma: no cover
    torchaudio = None
    AF = None  # type: ignore


def _ensure_torchaudio() -> None:
    if torchaudio is None or AF is None:
        raise ImportError("telephony_augs 依赖 torchaudio，请先安装。")


def simulate_codec(
    waveform: torch.Tensor,
    sample_rate: int,
    target_rate: int,
) -> torch.Tensor:
    _ensure_torchaudio()
    resampled = AF.resample(waveform, sample_rate, target_rate)
    filtered = AF.bandpass_biquad(resampled, target_rate, 1700.0, Q=0.707)
    companded = AF.contrast(filtered, enhancement_amount=25.0)
    restored = AF.resample(companded, target_rate, sample_rate)
    return restored


def bandwidth_jitter(
    waveform: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    _ensure_torchaudio()
    target = sample_rate // random.choice([1, 2])
    return AF.resample(AF.resample(waveform, sample_rate, target), target, sample_rate)


def dynamic_range_compress(waveform: torch.Tensor) -> torch.Tensor:
    return torch.tanh(1.5 * waveform)


def subtle_distortion(waveform: torch.Tensor) -> torch.Tensor:
    return waveform + 0.0005 * torch.randn_like(waveform)


def apply_rawboost(waveform: torch.Tensor) -> torch.Tensor:
    gain = random.uniform(0.9, 1.1)
    waveform = waveform * gain
    waveform = waveform + random.uniform(-0.01, 0.01)
    if random.random() < 0.5:
        waveform = torch.roll(waveform, shifts=random.randint(-10, 10), dims=-1)
    if random.random() < 0.3:
        waveform = torch.clamp(waveform, -1.0, 1.0)
    return waveform


class RandomTelephonyAugmenter:
    """结合电话链路与 RawBoost 的随机增广器。"""

    def __init__(
        self,
        codecs: Optional[Iterable[int]] = None,
        p_codec: float = 0.6,
        p_bandwidth: float = 0.5,
        p_compand: float = 0.5,
        p_rawboost: float = 0.5,
    ) -> None:
        self.codecs = list(codecs or (8000, 12000, 16000))
        self.p_codec = p_codec
        self.p_bandwidth = p_bandwidth
        self.p_compand = p_compand
        self.p_rawboost = p_rawboost

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        out = waveform
        if torch.is_tensor(out):
            out = out.clone()
        if random.random() < self.p_codec:
            target_sr = random.choice(self.codecs)
            out = simulate_codec(out, sample_rate, target_sr)
        if random.random() < self.p_bandwidth:
            out = bandwidth_jitter(out, sample_rate)
        if random.random() < self.p_compand:
            out = dynamic_range_compress(out)
            out = subtle_distortion(out)
        if random.random() < self.p_rawboost:
            out = apply_rawboost(out)
        return out
