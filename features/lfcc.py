"""LFCC (Linear Frequency Cepstral Coefficients) 特征提取

LFCC 使用线性频率尺度,与 MFCC 的 Mel 尺度互补,对合成伪迹更敏感
支持离线缓存和高效批量处理
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
from scipy.fftpack import dct


class LFCCExtractor:
    """LFCC 特征提取器,支持缓存和批量处理"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_lfcc: int = 60,
        n_filters: int = 70,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            sample_rate: 采样率
            n_fft: FFT 点数
            win_length: 窗长(样本数)
            hop_length: 帧移(样本数)
            n_lfcc: LFCC 系数数量
            n_filters: 线性滤波器组数量
            cache_dir: 特征缓存目录
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_lfcc = n_lfcc
        self.n_filters = n_filters
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建线性滤波器组
        self.filterbank = self._create_linear_filterbank()
    
    def _create_linear_filterbank(self) -> torch.Tensor:
        """创建线性频率滤波器组"""
        freq_bins = self.n_fft // 2 + 1
        freqs = torch.linspace(0, self.sample_rate / 2, freq_bins)
        
        # 线性间隔的中心频率
        center_freqs = torch.linspace(0, self.sample_rate / 2, self.n_filters + 2)
        
        filterbank = torch.zeros(self.n_filters, freq_bins)
        
        for i in range(self.n_filters):
            left = center_freqs[i]
            center = center_freqs[i + 1]
            right = center_freqs[i + 2]
            
            # 三角滤波器
            for j, freq in enumerate(freqs):
                if left <= freq <= center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center < freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)
        
        return filterbank
    
    def _get_cache_path(self, audio_path: str) -> Optional[Path]:
        """生成缓存文件路径"""
        if not self.cache_dir:
            return None
        
        # 使用音频路径和参数生成唯一hash
        cache_key = f"{audio_path}_{self.sample_rate}_{self.n_fft}_{self.hop_length}_{self.n_lfcc}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"lfcc_{cache_hash}.npy"
    
    def extract(self, waveform: torch.Tensor, audio_path: Optional[str] = None) -> torch.Tensor:
        """
        提取 LFCC 特征
        
        Args:
            waveform: 音频波形 (samples,) 或 (batch, samples)
            audio_path: 音频文件路径,用于缓存
            
        Returns:
            LFCC 特征 (n_lfcc, time) 或 (batch, n_lfcc, time)
        """
        # 检查缓存
        if audio_path:
            cache_path = self._get_cache_path(audio_path)
            if cache_path and cache_path.exists():
                lfcc = np.load(cache_path)
                return torch.from_numpy(lfcc)
        
        # 处理批量输入
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # STFT
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=waveform.device),
            return_complex=True,
        )
        
        # 幅度谱
        mag_spec = torch.abs(spec)  # (batch, freq_bins, time)
        
        # 应用线性滤波器组
        filterbank = self.filterbank.to(mag_spec.device)
        filtered = torch.matmul(filterbank, mag_spec)  # (batch, n_filters, time)
        
        # 对数
        log_filtered = torch.log(filtered + 1e-8)
        
        # DCT
        lfcc = self._apply_dct(log_filtered.cpu().numpy())
        lfcc = torch.from_numpy(lfcc).to(mag_spec.device)
        
        # 取前 n_lfcc 个系数
        lfcc = lfcc[:, :self.n_lfcc, :]
        
        if squeeze_output:
            lfcc = lfcc.squeeze(0)
        
        # 保存缓存
        if audio_path and not squeeze_output:
            cache_path = self._get_cache_path(audio_path)
            if cache_path:
                np.save(cache_path, lfcc[0].cpu().numpy())
        
        return lfcc
    
    def _apply_dct(self, x: np.ndarray) -> np.ndarray:
        """应用 DCT-II"""
        return dct(x, type=2, axis=1, norm='ortho')


def extract_lfcc(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_lfcc: int = 60,
    audio_path: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    便捷函数:提取 LFCC 特征
    
    Args:
        waveform: 音频波形
        sample_rate: 采样率
        n_lfcc: LFCC 系数数量
        audio_path: 音频文件路径
        cache_dir: 缓存目录
        
    Returns:
        LFCC 特征 (n_lfcc, time) 或 (batch, n_lfcc, time)
    """
    extractor = LFCCExtractor(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        cache_dir=cache_dir,
    )
    return extractor.extract(waveform, audio_path)
