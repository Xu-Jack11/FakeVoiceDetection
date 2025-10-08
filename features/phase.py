"""相位特征提取 (RPS/MGD/GD)

相位信息能捕捉合成语音的伪迹,与幅度谱特征互补
包含:
- RPS (Relative Phase Shift): 相对相位偏移
- MGD (Modified Group Delay): 修正群时延
- Instantaneous Frequency Deviation: 瞬时频率偏差
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch


class PhaseFeatureExtractor:
    """相位特征提取器"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_phase_features: int = 60,
        feature_type: Literal["rps", "mgd", "ifd", "all"] = "all",
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            sample_rate: 采样率
            n_fft: FFT 点数
            win_length: 窗长
            hop_length: 帧移
            n_phase_features: 输出特征维度
            feature_type: 特征类型 ("rps", "mgd", "ifd", "all")
            cache_dir: 缓存目录
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_phase_features = n_phase_features
        self.feature_type = feature_type
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, audio_path: str) -> Optional[Path]:
        """生成缓存文件路径"""
        if not self.cache_dir:
            return None
        
        cache_key = f"{audio_path}_{self.sample_rate}_{self.n_fft}_{self.feature_type}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"phase_{cache_hash}.npy"
    
    def extract(self, waveform: torch.Tensor, audio_path: Optional[str] = None) -> torch.Tensor:
        """
        提取相位特征
        
        Args:
            waveform: 音频波形 (samples,) 或 (batch, samples)
            audio_path: 音频文件路径
            
        Returns:
            相位特征 (n_phase_features, time) 或 (batch, n_phase_features, time)
        """
        # 检查缓存
        if audio_path:
            cache_path = self._get_cache_path(audio_path)
            if cache_path and cache_path.exists():
                phase_feat = np.load(cache_path)
                return torch.from_numpy(phase_feat)
        
        # 处理批量输入
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        device = waveform.device
        
        # STFT
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=device),
            return_complex=True,
        )
        
        # 幅度和相位
        mag = torch.abs(spec)  # (batch, freq, time)
        phase = torch.angle(spec)  # (batch, freq, time)
        
        features = []
        
        # 1. Relative Phase Shift (RPS)
        if self.feature_type in ["rps", "all"]:
            rps = self._compute_rps(phase)
            features.append(rps)
        
        # 2. Modified Group Delay (MGD)
        if self.feature_type in ["mgd", "all"]:
            mgd = self._compute_mgd(mag, phase)
            features.append(mgd)
        
        # 3. Instantaneous Frequency Deviation (IFD)
        if self.feature_type in ["ifd", "all"]:
            ifd = self._compute_ifd(phase)
            features.append(ifd)
        
        # 合并特征
        if len(features) == 1:
            phase_feat = features[0]
        else:
            phase_feat = torch.cat(features, dim=1)
        
        # 降维到指定维度
        if phase_feat.size(1) > self.n_phase_features:
            # 平均池化降维
            pool_factor = phase_feat.size(1) // self.n_phase_features
            phase_feat = torch.nn.functional.avg_pool1d(
                phase_feat.transpose(1, 2),
                kernel_size=pool_factor,
                stride=pool_factor,
            ).transpose(1, 2)
            phase_feat = phase_feat[:, :self.n_phase_features, :]
        elif phase_feat.size(1) < self.n_phase_features:
            # 上采样
            phase_feat = torch.nn.functional.interpolate(
                phase_feat,
                size=(self.n_phase_features, phase_feat.size(2)),
                mode='bilinear',
                align_corners=False,
            )
        
        if squeeze_output:
            phase_feat = phase_feat.squeeze(0)
            
            # 保存缓存
            if audio_path:
                cache_path = self._get_cache_path(audio_path)
                if cache_path:
                    np.save(cache_path, phase_feat.cpu().numpy())
        
        return phase_feat
    
    def _compute_rps(self, phase: torch.Tensor) -> torch.Tensor:
        """计算相对相位偏移 (Relative Phase Shift)"""
        # 时间差分
        phase_diff = torch.diff(phase, dim=2, prepend=phase[:, :, :1])
        
        # 归一化到 [-pi, pi]
        phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
        
        return phase_diff
    
    def _compute_mgd(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """计算修正群时延 (Modified Group Delay)"""
        # 群时延 = -d(phase)/d(freq)
        phase_grad_freq = torch.diff(phase, dim=1, prepend=phase[:, :1, :])
        
        # 使用幅度加权
        eps = 1e-8
        mgd = (mag + eps) * phase_grad_freq
        
        # Cepstral smoothing (简化版)
        mgd = torch.sign(mgd) * torch.log(torch.abs(mgd) + eps)
        
        return mgd
    
    def _compute_ifd(self, phase: torch.Tensor) -> torch.Tensor:
        """计算瞬时频率偏差 (Instantaneous Frequency Deviation)"""
        # 瞬时频率 = d(phase)/dt
        inst_freq = torch.diff(phase, dim=2, prepend=phase[:, :, :1])
        
        # 归一化
        inst_freq = torch.remainder(inst_freq + torch.pi, 2 * torch.pi) - torch.pi
        
        # 计算偏差 (相对于线性相位)
        freq_bins = torch.arange(phase.size(1), device=phase.device).view(1, -1, 1)
        expected_freq = 2 * torch.pi * freq_bins / self.n_fft
        
        ifd = inst_freq - expected_freq
        
        return ifd


def extract_phase_features(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_phase_features: int = 60,
    feature_type: Literal["rps", "mgd", "ifd", "all"] = "all",
    audio_path: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    便捷函数:提取相位特征
    
    Args:
        waveform: 音频波形
        sample_rate: 采样率
        n_phase_features: 特征维度
        feature_type: 特征类型
        audio_path: 音频文件路径
        cache_dir: 缓存目录
        
    Returns:
        相位特征 (n_phase_features, time) 或 (batch, n_phase_features, time)
    """
    extractor = PhaseFeatureExtractor(
        sample_rate=sample_rate,
        n_phase_features=n_phase_features,
        feature_type=feature_type,
        cache_dir=cache_dir,
    )
    return extractor.extract(waveform, audio_path)
