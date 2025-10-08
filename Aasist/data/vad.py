"""VAD (Voice Activity Detection) 语音活动检测

用于裁剪静音段,提高训练效果
支持多种 VAD 方法:
1. 能量门限法 (快速,无依赖)
2. py-webrtcvad (准确,需要安装)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class EnergyVAD:
    """基于能量的VAD (快速,无外部依赖)"""
    
    def __init__(
        self,
        frame_length: int = 400,  # 25ms @ 16kHz
        hop_length: int = 160,    # 10ms @ 16kHz
        energy_threshold: float = 0.02,
        min_speech_duration: float = 0.5,  # 最小语音段长度(秒)
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration
    
    def __call__(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测语音活动段
        
        Args:
            waveform: 音频波形 (samples,)
            sample_rate: 采样率
            
        Returns:
            speech_waveform: 去除静音后的波形
            vad_mask: VAD mask (1 = 语音, 0 = 静音)
        """
        if len(waveform) == 0:
            return waveform, np.array([], dtype=bool)
        
        # 计算帧能量
        num_frames = 1 + (len(waveform) - self.frame_length) // self.hop_length
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            if end > len(waveform):
                frame = waveform[start:]
            else:
                frame = waveform[start:end]
            energy[i] = np.mean(frame ** 2)
        
        # 归一化能量
        if energy.max() > 0:
            energy = energy / energy.max()
        
        # 应用阈值
        vad_frames = energy > self.energy_threshold
        
        # 形态学操作: 去除短暂的语音/静音段
        min_frames = int(self.min_speech_duration * sample_rate / self.hop_length)
        vad_frames = self._smooth_vad(vad_frames, min_frames)
        
        # 创建采样级别的 mask
        vad_mask = np.zeros(len(waveform), dtype=bool)
        for i, is_speech in enumerate(vad_frames):
            if is_speech:
                start = i * self.hop_length
                end = min(start + self.hop_length, len(waveform))
                vad_mask[start:end] = True
        
        # 提取语音段
        if vad_mask.any():
            # 找到第一个和最后一个语音帧
            speech_indices = np.where(vad_mask)[0]
            start = speech_indices[0]
            end = speech_indices[-1] + 1
            speech_waveform = waveform[start:end]
        else:
            # 如果没有检测到语音,返回原始波形
            speech_waveform = waveform
            vad_mask = np.ones(len(waveform), dtype=bool)
        
        return speech_waveform, vad_mask
    
    def _smooth_vad(self, vad_frames: np.ndarray, min_frames: int) -> np.ndarray:
        """平滑 VAD 结果,去除短暂的语音/静音段"""
        # 闭运算: 填充短暂的静音
        vad_frames = self._morphology_close(vad_frames, min_frames)
        # 开运算: 去除短暂的语音
        vad_frames = self._morphology_open(vad_frames, min_frames)
        return vad_frames
    
    def _morphology_close(self, mask: np.ndarray, window: int) -> np.ndarray:
        """形态学闭运算"""
        # 膨胀
        dilated = self._dilate(mask, window)
        # 腐蚀
        closed = self._erode(dilated, window)
        return closed
    
    def _morphology_open(self, mask: np.ndarray, window: int) -> np.ndarray:
        """形态学开运算"""
        # 腐蚀
        eroded = self._erode(mask, window)
        # 膨胀
        opened = self._dilate(eroded, window)
        return opened
    
    def _dilate(self, mask: np.ndarray, window: int) -> np.ndarray:
        """膨胀"""
        result = mask.copy()
        for i in range(len(mask)):
            if mask[i]:
                start = max(0, i - window // 2)
                end = min(len(mask), i + window // 2 + 1)
                result[start:end] = True
        return result
    
    def _erode(self, mask: np.ndarray, window: int) -> np.ndarray:
        """腐蚀"""
        result = np.zeros_like(mask)
        for i in range(len(mask)):
            start = max(0, i - window // 2)
            end = min(len(mask), i + window // 2 + 1)
            if mask[start:end].all():
                result[i] = True
        return result


class WebRTCVAD:
    """基于 WebRTC 的 VAD (更准确,需要 py-webrtcvad)"""
    
    def __init__(
        self,
        aggressiveness: int = 2,  # 0-3, 越高越激进
        frame_duration: int = 30,  # ms, 10/20/30
        min_speech_duration: float = 0.5,
    ):
        """
        Args:
            aggressiveness: 0-3, 越高过滤越多
            frame_duration: 帧长(ms), 只支持 10/20/30
            min_speech_duration: 最小语音段长度(秒)
        """
        self.aggressiveness = aggressiveness
        self.frame_duration = frame_duration
        self.min_speech_duration = min_speech_duration
        
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.available = True
        except ImportError:
            print("Warning: webrtcvad not installed. Falling back to EnergyVAD.")
            self.available = False
            self.fallback = EnergyVAD(min_speech_duration=min_speech_duration)
    
    def __call__(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测语音活动段
        
        Args:
            waveform: 音频波形 (samples,)
            sample_rate: 采样率 (WebRTC VAD 只支持 8k/16k/32k/48k)
            
        Returns:
            speech_waveform: 去除静音后的波形
            vad_mask: VAD mask
        """
        if not self.available:
            return self.fallback(waveform, sample_rate)
        
        if sample_rate not in [8000, 16000, 32000, 48000]:
            # 重采样到 16kHz
            from scipy import signal
            waveform = signal.resample(
                waveform,
                int(len(waveform) * 16000 / sample_rate)
            )
            sample_rate = 16000
        
        # 转换为 16-bit PCM
        pcm = (waveform * 32767).astype(np.int16).tobytes()
        
        # 计算帧大小
        frame_size = int(sample_rate * self.frame_duration / 1000)
        
        # 分帧并检测
        vad_results = []
        for i in range(0, len(waveform), frame_size):
            frame = waveform[i:i+frame_size]
            if len(frame) < frame_size:
                # 填充最后一帧
                frame = np.pad(frame, (0, frame_size - len(frame)))
            
            frame_pcm = (frame * 32767).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(frame_pcm, sample_rate)
            vad_results.extend([is_speech] * frame_size)
        
        # 截断到原长度
        vad_results = vad_results[:len(waveform)]
        vad_mask = np.array(vad_results, dtype=bool)
        
        # 提取语音段
        if vad_mask.any():
            speech_indices = np.where(vad_mask)[0]
            start = speech_indices[0]
            end = speech_indices[-1] + 1
            speech_waveform = waveform[start:end]
        else:
            speech_waveform = waveform
            vad_mask = np.ones(len(waveform), dtype=bool)
        
        return speech_waveform, vad_mask


def apply_vad(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    vad_type: str = "energy",
    min_speech_ratio: float = 0.5,
    **vad_kwargs,
) -> np.ndarray:
    """
    便捷函数: 应用 VAD 并确保语音占比
    
    Args:
        waveform: 音频波形
        sample_rate: 采样率
        vad_type: VAD 类型 ("energy" 或 "webrtc")
        min_speech_ratio: 最小语音占比
        **vad_kwargs: VAD 参数
        
    Returns:
        处理后的波形
    """
    if vad_type == "energy":
        vad = EnergyVAD(**vad_kwargs)
    elif vad_type == "webrtc":
        vad = WebRTCVAD(**vad_kwargs)
    else:
        raise ValueError(f"Unknown VAD type: {vad_type}")
    
    speech_wav, vad_mask = vad(waveform, sample_rate)
    
    # 检查语音占比
    if len(speech_wav) > 0:
        speech_ratio = len(speech_wav) / len(waveform)
        if speech_ratio >= min_speech_ratio:
            return speech_wav
    
    # 如果语音占比太低,返回原始波形
    return waveform
