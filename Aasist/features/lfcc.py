"""LFCC (Linear Frequency Cepstral Coefficient) 前端实现。"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .base import FeatureExtractor


class LFCCFrontend(FeatureExtractor):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        n_lfcc: int = 60,
        n_filter: int = 80,
        lifter: float = 0.0,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__("lfcc", cache_dir=cache_dir, device=device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_lfcc = n_lfcc
        self.n_filter = n_filter
        self.lifter_factor = lifter
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )
        self._filterbank_cache: dict[int, Tensor] = {}
        self.register_buffer(
            "dct_matrix",
            self._create_dct(n_filter, n_lfcc),
            persistent=False,
        )
        if lifter > 0:
            lifter_coeff = 1 + 0.5 * lifter * torch.sin(
                math.pi * torch.arange(n_lfcc) / lifter
            )
            self.register_buffer("lifter", lifter_coeff, persistent=False)
        else:
            self.register_buffer("lifter", torch.ones(n_lfcc), persistent=False)

    @staticmethod
    def _create_dct(n_mels: int, n_lfcc: int) -> Tensor:
        basis = torch.empty(n_lfcc, n_mels)
        basis[0] = math.sqrt(1 / n_mels)
        for k in range(1, n_lfcc):
            n = torch.arange(n_mels)
            basis[k] = math.sqrt(2 / n_mels) * torch.cos(
                math.pi / n_mels * (n + 0.5) * k
            )
        return basis

    def _get_filterbank(self, sample_rate: int) -> Tensor:
        cached = self._filterbank_cache.get(sample_rate)
        if cached is not None:
            return cached
        n_freqs = self.n_fft // 2 + 1
        edges = torch.linspace(0, sample_rate / 2, self.n_filter + 2)
        freqs = torch.linspace(0, sample_rate / 2, n_freqs)
        fbanks = torch.zeros(self.n_filter, n_freqs)
        for i in range(self.n_filter):
            left, center, right = edges[i], edges[i + 1], edges[i + 2]
            left_slope = (freqs - left) / max(center - left, 1e-6)
            right_slope = (right - freqs) / max(right - center, 1e-6)
            fbanks[i] = torch.clamp(torch.min(left_slope, right_slope), min=0.0)
        fbanks = fbanks / torch.clamp(fbanks.sum(dim=1, keepdim=True), min=1e-6)
        self._filterbank_cache[sample_rate] = fbanks
        return fbanks

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
        power = stft.abs().pow(2)
        fbanks = self._get_filterbank(sample_rate).to(waveform.device)
        spec = torch.matmul(power.transpose(1, 2), fbanks.t())
        spec = torch.log(spec + 1e-6)
        lfcc = torch.matmul(spec, self.dct_matrix.t().to(spec.device))
        lfcc = lfcc.transpose(1, 2)
        if self.lifter is not None:
            lfcc = lfcc * self.lifter.view(1, -1, 1)
        return lfcc
