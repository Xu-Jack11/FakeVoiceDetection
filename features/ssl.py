"""SSL (Self-Supervised Learning) 特征提取

从预训练的自监督模型提取表征:
- wav2vec2: Facebook 的语音预训练模型
- WavLM: Microsoft 的增强语音模型
- Whisper: OpenAI 的多语言语音模型

这些表征对未知伪造/信道/语言有更强的泛化能力
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn


class SSLFeatureExtractor:
    """SSL 特征提取器,支持多种预训练模型"""
    
    def __init__(
        self,
        model_type: Literal["wav2vec2", "wavlm", "whisper"] = "wav2vec2",
        model_name: str = "facebook/wav2vec2-base",
        layer: int = -1,
        pooling: Literal["mean", "max", "last"] = "mean",
        cache_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_type: 模型类型
            model_name: HuggingFace 模型名称
            layer: 提取哪一层的特征 (-1 表示最后一层)
            pooling: 时间维度的池化方式
            cache_dir: 缓存目录
            device: 计算设备
        """
        self.model_type = model_type
        self.model_name = model_name
        self.layer = layer
        self.pooling = pooling
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 延迟加载模型
        self.model = None
        self.processor = None
    
    def _load_model(self):
        """延迟加载预训练模型"""
        if self.model is not None:
            return
        
        try:
            if self.model_type == "wav2vec2":
                from transformers import Wav2Vec2Model, Wav2Vec2Processor
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            elif self.model_type == "wavlm":
                from transformers import WavLMModel, Wav2Vec2Processor
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = WavLMModel.from_pretrained(self.model_name)
            elif self.model_type == "whisper":
                from transformers import WhisperModel, WhisperProcessor
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperModel.from_pretrained(self.model_name).encoder
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            print("Warning: transformers not installed. Using fallback feature extractor.")
            self.model = None
            self.processor = None
    
    def _get_cache_path(self, audio_path: str) -> Optional[Path]:
        """生成缓存文件路径"""
        if not self.cache_dir:
            return None
        
        cache_key = f"{audio_path}_{self.model_type}_{self.model_name}_{self.layer}_{self.pooling}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"ssl_{cache_hash}.npy"
    
    def extract(self, waveform: torch.Tensor, audio_path: Optional[str] = None) -> torch.Tensor:
        """
        提取 SSL 特征
        
        Args:
            waveform: 音频波形 (samples,) 或 (batch, samples)
            audio_path: 音频文件路径
            
        Returns:
            SSL 特征 (feature_dim, time) 或 (batch, feature_dim, time)
        """
        # 检查缓存
        if audio_path:
            cache_path = self._get_cache_path(audio_path)
            if cache_path and cache_path.exists():
                ssl_feat = np.load(cache_path)
                return torch.from_numpy(ssl_feat)
        
        # 加载模型
        self._load_model()
        
        if self.model is None:
            # Fallback: 使用简单的学习型特征提取
            return self._extract_fallback(waveform)
        
        # 处理批量输入
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        original_device = waveform.device
        waveform = waveform.cpu()
        
        with torch.no_grad():
            # 预处理
            if self.model_type in ["wav2vec2", "wavlm"]:
                inputs = self.processor(
                    waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.layer]  # (batch, time, dim)
                
            elif self.model_type == "whisper":
                # Whisper 需要 log-mel 输入
                from transformers import WhisperFeatureExtractor
                feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
                inputs = feature_extractor(
                    waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(inputs["input_features"])
                hidden_states = outputs.last_hidden_state  # (batch, time, dim)
            
            # 池化
            if self.pooling == "mean":
                features = hidden_states.mean(dim=1, keepdim=True)  # (batch, 1, dim)
            elif self.pooling == "max":
                features = hidden_states.max(dim=1, keepdim=True)[0]
            elif self.pooling == "last":
                features = hidden_states[:, -1:, :]
            else:
                # 保持时间维度
                features = hidden_states
            
            # 转置为 (batch, dim, time)
            features = features.transpose(1, 2)
        
        features = features.to(original_device)
        
        if squeeze_output:
            features = features.squeeze(0)
            
            # 保存缓存
            if audio_path:
                cache_path = self._get_cache_path(audio_path)
                if cache_path:
                    np.save(cache_path, features.cpu().numpy())
        
        return features
    
    def _extract_fallback(self, waveform: torch.Tensor) -> torch.Tensor:
        """Fallback: 使用简单的卷积网络提取特征"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 简单的 1D CNN
        # (batch, samples) -> (batch, 1, samples)
        x = waveform.unsqueeze(1)
        
        # 卷积层
        conv1 = nn.Conv1d(1, 64, kernel_size=10, stride=5).to(x.device)
        conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2).to(x.device)
        conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2).to(x.device)
        
        x = torch.relu(conv1(x))
        x = torch.relu(conv2(x))
        x = torch.relu(conv3(x))  # (batch, 256, time)
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x


def extract_ssl_features(
    waveform: torch.Tensor,
    model_type: Literal["wav2vec2", "wavlm", "whisper"] = "wav2vec2",
    model_name: str = "facebook/wav2vec2-base",
    audio_path: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    便捷函数:提取 SSL 特征
    
    Args:
        waveform: 音频波形
        model_type: 模型类型
        model_name: 模型名称
        audio_path: 音频文件路径
        cache_dir: 缓存目录
        
    Returns:
        SSL 特征 (feature_dim, time) 或 (batch, feature_dim, time)
    """
    extractor = SSLFeatureExtractor(
        model_type=model_type,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    return extractor.extract(waveform, audio_path)
