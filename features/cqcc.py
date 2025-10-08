"""CQCC (Constant-Q Cepstral Coefficients) 特征提取

CQCC 使用对数频率尺度的 Constant-Q 变换,对音频压缩伪迹敏感
是反伪造任务中的经典强基线
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa
from scipy.fftpack import dct


class CQCCExtractor:
    """CQCC 特征提取器,支持缓存"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_bins: int = 96,
        bins_per_octave: int = 24,
        n_cqcc: int = 60,
        fmin: float = 20.0,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            sample_rate: 采样率
            hop_length: 帧移
            n_bins: CQT 频率bins数量
            bins_per_octave: 每个八度的bins数
            n_cqcc: CQCC 系数数量
            fmin: 最低频率 (Hz)
            cache_dir: 缓存目录
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.n_cqcc = n_cqcc
        self.fmin = fmin
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, audio_path: str) -> Optional[Path]:
        """生成缓存文件路径"""
        if not self.cache_dir:
            return None
        
        cache_key = f"{audio_path}_{self.sample_rate}_{self.n_bins}_{self.n_cqcc}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"cqcc_{cache_hash}.npy"
    
    def extract(self, waveform: torch.Tensor, audio_path: Optional[str] = None) -> torch.Tensor:
        """
        提取 CQCC 特征
        
        Args:
            waveform: 音频波形 (samples,) 或 (batch, samples)
            audio_path: 音频文件路径
            
        Returns:
            CQCC 特征 (n_cqcc, time) 或 (batch, n_cqcc, time)
        """
        # 检查缓存
        if audio_path:
            cache_path = self._get_cache_path(audio_path)
            if cache_path and cache_path.exists():
                cqcc = np.load(cache_path)
                return torch.from_numpy(cqcc)
        
        # 处理批量输入
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        device = waveform.device
        waveform_np = waveform.cpu().numpy()
        
        batch_cqcc = []
        for wav in waveform_np:
            # 使用 librosa 计算 CQT
            cqt = librosa.cqt(
                wav,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                fmin=self.fmin,
            )
            
            # 幅度谱
            cqt_mag = np.abs(cqt)
            
            # 对数
            log_cqt = np.log(cqt_mag + 1e-8)
            
            # DCT
            cqcc = dct(log_cqt, type=2, axis=0, norm='ortho')
            
            # 取前 n_cqcc 个系数
            cqcc = cqcc[:self.n_cqcc, :]
            
            batch_cqcc.append(cqcc)
        
        cqcc_tensor = torch.from_numpy(np.stack(batch_cqcc, axis=0)).float().to(device)
        
        if squeeze_output:
            cqcc_tensor = cqcc_tensor.squeeze(0)
            
            # 保存缓存
            if audio_path:
                cache_path = self._get_cache_path(audio_path)
                if cache_path:
                    np.save(cache_path, cqcc_tensor.cpu().numpy())
        
        return cqcc_tensor


def extract_cqcc(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_cqcc: int = 60,
    audio_path: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    便捷函数:提取 CQCC 特征
    
    Args:
        waveform: 音频波形
        sample_rate: 采样率
        n_cqcc: CQCC 系数数量
        audio_path: 音频文件路径
        cache_dir: 缓存目录
        
    Returns:
        CQCC 特征 (n_cqcc, time) 或 (batch, n_cqcc, time)
    """
    extractor = CQCCExtractor(
        sample_rate=sample_rate,
        n_cqcc=n_cqcc,
        cache_dir=cache_dir,
    )
    return extractor.extract(waveform, audio_path)
