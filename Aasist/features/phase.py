"""相位相关特征（RPS、群时延等）前端实现。"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import FeatureExtractor


class PhaseFrontend(FeatureExtractor):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__("phase", cache_dir=cache_dir, device=device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

    def _extract(
        self,
        waveform: Tensor,
        sample_rate: int,
        metadata: Optional[dict] = None,
    ) -> Tensor:
        window = self.window.to(waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        magnitude = stft.abs().clamp_min(1e-6)
        phase = torch.angle(stft)

        # RPS (Relative Phase Shift)
        diff_time = torch.diff(phase, dim=-1, prepend=phase[..., :1])
        rps = torch.sin(diff_time)

        # Group delay (approx.)
        diff_freq = torch.diff(phase, dim=-2, prepend=phase[:, :1, :])
        group_delay = -diff_freq

        # Modified group delay using magnitude to suppress spikes
        mgd = group_delay[:, 1:, :] * magnitude[:, 1:, :]
        mgd = torch.nn.functional.pad(mgd, (0, 0, 1, 0))

        features = torch.stack([rps, group_delay, mgd], dim=1)
        return features
