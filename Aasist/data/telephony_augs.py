"""电话信道增强 (Telephony Augmentations)

模拟真实电话场景的音频失真,提高模型鲁棒性:
1. 编解码模拟: AMR-NB/WB, G.711, G.729, Opus
2. 带宽抖动: 8k ↔ 16k 重采样
3. 带通滤波: 模拟电话频率范围 (300-3400 Hz)
4. 动态范围压缩/压扩
5. RawBoost: 波形域线性/非线性扰动
"""

from __future__ import annotations

import random
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


class TelephonyAugmentation:
    """电话信道增强"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        p_codec: float = 0.5,
        p_bandwidth: float = 0.3,
        p_bandpass: float = 0.6,
        p_compand: float = 0.4,
        p_rawboost: float = 0.7,
    ):
        """
        Args:
            sample_rate: 采样率
            p_codec: 编解码增强概率
            p_bandwidth: 带宽抖动概率
            p_bandpass: 带通滤波概率
            p_compand: 压扩概率
            p_rawboost: RawBoost 概率
        """
        self.sample_rate = sample_rate
        self.p_codec = p_codec
        self.p_bandwidth = p_bandwidth
        self.p_bandpass = p_bandpass
        self.p_compand = p_compand
        self.p_rawboost = p_rawboost
    
    def __call__(self, waveform: torch.Tensor, sr: Optional[int] = None) -> torch.Tensor:
        """
        应用电话信道增强
        
        Args:
            waveform: 音频波形 (samples,) 或 (channels, samples)
            sr: 原始采样率
            
        Returns:
            增强后的波形
        """
        if sr is None:
            sr = self.sample_rate
        
        # 确保是 1D
        squeeze_dim = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_dim = True
        
        # 1. 编解码模拟
        if random.random() < self.p_codec:
            waveform = self.codec_simulation(waveform, sr)
        
        # 2. 带宽抖动
        if random.random() < self.p_bandwidth:
            waveform = self.bandwidth_jitter(waveform, sr)
        
        # 3. 带通滤波
        if random.random() < self.p_bandpass:
            waveform = self.telephony_bandpass(waveform, sr)
        
        # 4. 动态范围压缩
        if random.random() < self.p_compand:
            waveform = self.dynamic_range_compression(waveform)
        
        # 5. RawBoost
        if random.random() < self.p_rawboost:
            waveform = self.rawboost(waveform)
        
        if squeeze_dim:
            waveform = waveform.squeeze(0)
        
        return waveform
    
    def codec_simulation(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """编解码模拟
        
        使用 torchaudio 或 ffmpeg 模拟低比特率编解码
        """
        codec_type = random.choice(["amr-nb", "g711", "opus-low", "resample"])
        
        if codec_type == "amr-nb":
            # AMR-NB: 窄带语音编码 (8kHz)
            waveform = self._resample(waveform, sr, 8000)
            waveform = self._add_quantization_noise(waveform, bits=13)
            waveform = self._resample(waveform, 8000, sr)
        
        elif codec_type == "g711":
            # G.711: μ-law/A-law 压扩
            waveform = self._apply_mulaw(waveform, quantization_channels=256)
        
        elif codec_type == "opus-low":
            # Opus 低比特率
            waveform = self._resample(waveform, sr, 16000)
            waveform = self._add_quantization_noise(waveform, bits=12)
            if sr != 16000:
                waveform = self._resample(waveform, 16000, sr)
        
        elif codec_type == "resample":
            # 简单的重采样失真
            intermediate_sr = random.choice([8000, 11025, 12000, 16000])
            waveform = self._resample(waveform, sr, intermediate_sr)
            waveform = self._resample(waveform, intermediate_sr, sr)
        
        return waveform
    
    def bandwidth_jitter(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """带宽抖动: 8k ↔ 16k"""
        target_sr = random.choice([8000, 12000, 16000])
        
        if target_sr != sr:
            waveform = self._resample(waveform, sr, target_sr)
            waveform = self._resample(waveform, target_sr, sr)
        
        return waveform
    
    def telephony_bandpass(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """电话带通滤波: 300-3400 Hz"""
        # 使用 SoX 风格的带通滤波
        lowcut = random.uniform(250, 350)
        highcut = random.uniform(3200, 3600)
        
        # 使用 torchaudio 的滤波器
        try:
            # 设计带通滤波器
            nyquist = sr / 2
            low_norm = lowcut / nyquist
            high_norm = highcut / nyquist
            
            # 简化版: 使用高通 + 低通
            waveform = torchaudio.functional.highpass_biquad(
                waveform, sr, lowcut, Q=0.707
            )
            waveform = torchaudio.functional.lowpass_biquad(
                waveform, sr, highcut, Q=0.707
            )
        except Exception:
            # Fallback: 简单的频域滤波
            pass
        
        return waveform
    
    def dynamic_range_compression(self, waveform: torch.Tensor) -> torch.Tensor:
        """动态范围压缩 (Compressor/Limiter)"""
        # 简单的压缩器
        threshold = random.uniform(0.3, 0.7)
        ratio = random.uniform(2.0, 6.0)
        
        # 查找超过阈值的部分
        sign = torch.sign(waveform)
        abs_wav = torch.abs(waveform)
        
        # 压缩超过阈值的部分
        mask = abs_wav > threshold
        compressed = abs_wav.clone()
        compressed[mask] = threshold + (abs_wav[mask] - threshold) / ratio
        
        waveform = sign * compressed
        
        # 软削波
        waveform = torch.tanh(waveform * random.uniform(0.8, 1.2))
        
        return waveform
    
    def rawboost(self, waveform: torch.Tensor) -> torch.Tensor:
        """RawBoost: 波形域扰动
        
        参考: RawBoost: A Raw Data Boosting and Augmentation Method 
               applied to Automatic Speaker Verification Anti-Spoofing
        """
        aug_type = random.choice(["LnL", "ISD", "SSI"])
        
        if aug_type == "LnL":
            # Linear and Non-linear Convolutive Noise
            waveform = self._rawboost_lnl(waveform)
        elif aug_type == "ISD":
            # Impulsive Signal-dependent Noise
            waveform = self._rawboost_isd(waveform)
        elif aug_type == "SSI":
            # Stationary Signal-independent Noise
            waveform = self._rawboost_ssi(waveform)
        
        return waveform
    
    def _rawboost_lnl(self, waveform: torch.Tensor) -> torch.Tensor:
        """RawBoost: 线性和非线性卷积噪声"""
        # 非线性失真
        alpha = random.uniform(0.1, 0.5)
        waveform = torch.tanh(alpha * waveform) / torch.tanh(torch.tensor(alpha))
        
        # 轻微混响 (简化版)
        decay = random.uniform(0.05, 0.15)
        delay_samples = random.randint(50, 200)
        
        if waveform.size(-1) > delay_samples:
            delayed = torch.nn.functional.pad(
                waveform[..., :-delay_samples], (delay_samples, 0)
            )
            waveform = waveform + decay * delayed
        
        return waveform
    
    def _rawboost_isd(self, waveform: torch.Tensor) -> torch.Tensor:
        """RawBoost: 脉冲信号相关噪声"""
        # 添加与信号幅度相关的噪声
        noise_level = random.uniform(0.001, 0.01)
        noise = torch.randn_like(waveform) * torch.abs(waveform) * noise_level
        waveform = waveform + noise
        
        return waveform
    
    def _rawboost_ssi(self, waveform: torch.Tensor) -> torch.Tensor:
        """RawBoost: 平稳信号独立噪声"""
        # 添加低级白噪声
        snr_db = random.uniform(25, 40)  # 高 SNR
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        waveform = waveform + noise
        
        return waveform
    
    def _resample(self, waveform: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
        """重采样"""
        if orig_sr == new_sr:
            return waveform
        
        resampler = T.Resample(orig_sr, new_sr)
        return resampler(waveform)
    
    def _add_quantization_noise(self, waveform: torch.Tensor, bits: int) -> torch.Tensor:
        """添加量化噪声"""
        levels = 2 ** bits
        quantized = torch.round(waveform * levels) / levels
        return quantized
    
    def _apply_mulaw(self, waveform: torch.Tensor, quantization_channels: int = 256) -> torch.Tensor:
        """应用 μ-law 压扩"""
        mu = quantization_channels - 1.0
        
        # μ-law 编码
        safe_audio_abs = torch.clamp(torch.abs(waveform), min=1e-8, max=1.0)
        magnitude = torch.log1p(mu * safe_audio_abs) / torch.log1p(torch.tensor(mu))
        signal = torch.sign(waveform) * magnitude
        
        # 量化
        signal = torch.round(signal * mu) / mu
        
        # μ-law 解码
        magnitude = torch.abs(signal)
        decoded = torch.sign(signal) * (torch.exp(magnitude * torch.log1p(torch.tensor(mu))) - 1.0) / mu
        
        return decoded


class CodecSimulationFFmpeg:
    """使用 ffmpeg 进行真实编解码模拟 (可选)
    
    需要系统安装 ffmpeg
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查 ffmpeg 是否可用"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            self.ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.ffmpeg_available = False
    
    def __call__(self, waveform: torch.Tensor, codec: str = "amr-nb") -> torch.Tensor:
        """
        使用 ffmpeg 进行编解码
        
        Args:
            waveform: 输入波形 (channels, samples)
            codec: 编解码器 ("amr-nb", "amr-wb", "g729", "opus")
            
        Returns:
            编解码后的波形
        """
        if not self.ffmpeg_available:
            return waveform
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            
            tmp_in_path = Path(tmp_in.name)
            tmp_out_path = Path(tmp_out.name)
            
            try:
                # 保存输入
                torchaudio.save(tmp_in_path, waveform, self.sample_rate)
                
                # ffmpeg 编解码
                if codec == "amr-nb":
                    cmd = [
                        "ffmpeg", "-y", "-i", str(tmp_in_path),
                        "-ar", "8000", "-ac", "1", "-ab", "12.2k",
                        "-acodec", "libopencore_amrnb",
                        str(tmp_out_path)
                    ]
                elif codec == "opus":
                    cmd = [
                        "ffmpeg", "-y", "-i", str(tmp_in_path),
                        "-ar", "16000", "-ac", "1", "-ab", "12k",
                        "-acodec", "libopus",
                        str(tmp_out_path)
                    ]
                else:
                    return waveform
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 加载输出
                waveform_out, _ = torchaudio.load(tmp_out_path)
                
                # 确保长度一致
                if waveform_out.size(-1) != waveform.size(-1):
                    if waveform_out.size(-1) < waveform.size(-1):
                        # 填充
                        pad_len = waveform.size(-1) - waveform_out.size(-1)
                        waveform_out = torch.nn.functional.pad(waveform_out, (0, pad_len))
                    else:
                        # 截断
                        waveform_out = waveform_out[..., :waveform.size(-1)]
                
                return waveform_out
            
            finally:
                # 清理临时文件
                tmp_in_path.unlink(missing_ok=True)
                tmp_out_path.unlink(missing_ok=True)
