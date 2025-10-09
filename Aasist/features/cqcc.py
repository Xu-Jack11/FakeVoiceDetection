"""Constant-Q Cepstral Coefficient (CQCC) 前端实现。"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .base import FeatureExtractor

try:
    import torchaudio
    from torchaudio.transforms import CQT
except Exception:  # pragma: no cover - 运行环境可能无 torchaudio
    torchaudio = None
    CQT = None  # type: ignore


class CQCCFrontend(FeatureExtractor):
    def __init__(
        self,
        fmin: float = 20.0,
        n_bins: int = 96,
        bins_per_octave: int = 24,
        hop_length: int = 256,
        n_cqcc: int = 60,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__("cqcc", cache_dir=cache_dir, device=device)
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.n_cqcc = n_cqcc

        if CQT is not None:
            self.cqt = CQT(
                sample_rate=16000,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                hop_length=hop_length,
                fmin=fmin,
                trainable=False,
            )
        else:
            self.cqt = None

        self.register_buffer(
            "dct_matrix",
            self._create_dct(n_bins, n_cqcc),
            persistent=False,
        )

    @staticmethod
    def _create_dct(n_input: int, n_output: int) -> Tensor:
        basis = torch.empty(n_output, n_input)
        basis[0] = math.sqrt(1 / n_input)
        n = torch.arange(n_input)
        for k in range(1, n_output):
            basis[k] = math.sqrt(2 / n_input) * torch.cos(
                math.pi / n_input * (n + 0.5) * k
            )
        return basis

    def _compute_cqt_torchaudio(
        self, waveform: Tensor, sample_rate: int
    ) -> Tensor:
        assert self.cqt is not None
        target_sr = getattr(self.cqt, "sample_rate", getattr(self.cqt, "sr", 16000))
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sr
            )
            sample_rate = target_sr
        transform = self.cqt.to(waveform.device)
        spec = transform(waveform)
        return spec.abs().clamp_min(1e-6)

    def _compute_cqt_fallback(
        self, waveform: Tensor, sample_rate: int
    ) -> Tensor:
        # 退化实现：使用对数频率间隔的滤波器组近似 CQT
        n_fft = 2048
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(n_fft, device=waveform.device),
            return_complex=True,
        )
        magnitude = stft.abs()
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=waveform.device)
        centers = self.fmin * (2 ** (torch.arange(self.n_bins, device=waveform.device) / self.bins_per_octave))
        bandwidth = centers * (2 ** (1 / self.bins_per_octave) - 1)
        filters = []
        for c, b in zip(centers, bandwidth):
            response = torch.exp(-0.5 * ((freqs - c) / (b + 1e-6)) ** 2)
            filters.append(response)
        filterbank = torch.stack(filters)
        filterbank = filterbank / torch.clamp(filterbank.sum(dim=1, keepdim=True), min=1e-6)
        proj = torch.matmul(filterbank, magnitude.transpose(1, 2)).transpose(1, 2)
        return proj.clamp_min(1e-6)

    def _extract(
        self,
        waveform: Tensor,
        sample_rate: int,
        metadata: Optional[dict] = None,
    ) -> Tensor:
        if CQT is not None:
            spec = self._compute_cqt_torchaudio(waveform, sample_rate)
        else:
            spec = self._compute_cqt_fallback(waveform, sample_rate)

        log_spec = torch.log(spec + 1e-6)
        cqcc = torch.matmul(log_spec, self.dct_matrix.t().to(log_spec.device))
        return cqcc.transpose(1, 2)
