"""AASIST3 数据集加载器

支持:
1. 多路特征提取 (LFCC + CQCC + Phase + SSL)
2. VAD 语音活动检测
3. 电话信道增强
4. 随机切片 (2-6s 训练, 固定长度推理)
5. 离线特征缓存
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

# 导入特征提取器
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from features import extract_lfcc, extract_cqcc, extract_phase_features

# 导入增强和 VAD
from .telephony_augs import TelephonyAugmentation
from .vad import apply_vad


def _resample_linear(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """线性插值重采样"""
    if orig_sr == target_sr:
        return waveform
    if waveform.size == 0:
        return waveform
    duration = waveform.shape[0] / float(orig_sr)
    target_len = int(round(duration * target_sr))
    if target_len <= 0:
        target_len = 1
    orig_indices = np.linspace(0.0, waveform.shape[0] - 1, num=waveform.shape[0])
    target_indices = np.linspace(0.0, waveform.shape[0] - 1, num=target_len)
    return np.interp(target_indices, orig_indices, waveform).astype(np.float32)


@dataclass(slots=True)
class AASIST3DatasetConfig:
    """AASIST3 数据集配置"""
    csv_path: Path
    audio_dir: Path
    sample_rate: int = 16000
    
    # 切片配置
    training: bool = True
    min_chunk_duration: float = 2.0  # 训练时最小切片长度(秒)
    max_chunk_duration: float = 6.0  # 训练时最大切片长度(秒)
    fixed_chunk_duration: Optional[float] = None  # 推理时固定长度
    
    # 特征配置
    feature_types: List[str] = None  # ["lfcc", "cqcc", "phase", "ssl"]
    feature_cache_dir: Optional[Path] = None
    
    # VAD 配置
    use_vad: bool = True
    vad_type: str = "energy"  # "energy" or "webrtc"
    min_speech_ratio: float = 0.5
    
    # 增强配置
    use_augmentation: bool = True
    aug_prob: float = 0.8
    
    def __post_init__(self):
        if self.feature_types is None:
            self.feature_types = ["lfcc", "cqcc", "phase"]


class AASIST3Dataset(Dataset):
    """AASIST3 多路特征数据集"""
    
    def __init__(self, config: AASIST3DatasetConfig):
        self.config = config
        self.data = pd.read_csv(config.csv_path)
        self.audio_dir = config.audio_dir
        self.sample_rate = config.sample_rate
        
        # 初始化增强器
        if config.use_augmentation and config.training:
            self.augmentation = TelephonyAugmentation(sample_rate=config.sample_rate)
        else:
            self.augmentation = None
        
        # 特征缓存
        self.feature_cache_dir = config.feature_cache_dir
        if self.feature_cache_dir:
            self.feature_cache_dir = Path(self.feature_cache_dir)
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor | str]:
        """
        返回多路特征和标签
        
        Returns:
            features: {feature_type: tensor}
            label: 标签或音频名
        """
        row = self.data.iloc[idx]
        audio_name: str = row["audio_name"]
        audio_path = self.audio_dir / audio_name
        
        # 加载音频
        try:
            waveform, sr = sf.read(audio_path)
        except Exception as exc:
            raise RuntimeError(f"读取音频失败: {audio_path}") from exc
        
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        # 重采样
        waveform = _resample_linear(waveform, sr, self.sample_rate)
        
        # VAD
        if self.config.use_vad:
            waveform = apply_vad(
                waveform,
                sample_rate=self.sample_rate,
                vad_type=self.config.vad_type,
                min_speech_ratio=self.config.min_speech_ratio,
            )
        
        # 切片
        waveform = self._chunk_waveform(waveform)
        
        # 数据增强
        waveform_tensor = torch.from_numpy(waveform).float()
        if self.augmentation and np.random.random() < self.config.aug_prob:
            waveform_tensor = self.augmentation(waveform_tensor, self.sample_rate)
        
        # 提取多路特征
        features = self._extract_features(waveform_tensor, audio_name)
        
        # 返回标签
        if "target" in row:
            target = torch.tensor(int(row["target"]), dtype=torch.long)
            return features, target
        return features, audio_name
    
    def _chunk_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """切片或填充波形"""
        if self.config.training:
            # 训练: 随机切片 2-6s
            min_len = int(self.config.min_chunk_duration * self.sample_rate)
            max_len = int(self.config.max_chunk_duration * self.sample_rate)
            target_len = np.random.randint(min_len, max_len + 1)
        else:
            # 推理: 固定长度或全长
            if self.config.fixed_chunk_duration:
                target_len = int(self.config.fixed_chunk_duration * self.sample_rate)
            else:
                return waveform
        
        current_len = len(waveform)
        
        if current_len > target_len:
            # 随机裁剪
            if self.config.training:
                start = np.random.randint(0, current_len - target_len + 1)
            else:
                start = (current_len - target_len) // 2
            waveform = waveform[start:start + target_len]
        elif current_len < target_len:
            # 填充
            pad_width = target_len - current_len
            waveform = np.pad(waveform, (0, pad_width), mode="constant")
        
        return waveform
    
    def _extract_features(
        self,
        waveform: Tensor,
        audio_name: str,
    ) -> Dict[str, Tensor]:
        """提取多路特征"""
        features = {}
        
        for feature_type in self.config.feature_types:
            # 检查缓存
            cache_key = self._get_cache_key(audio_name, feature_type)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path and cache_path.exists() and not self.config.training:
                # 加载缓存 (仅在推理时)
                feat = torch.load(cache_path)
            else:
                # 提取特征
                feat = self._extract_single_feature(waveform, feature_type)
                
                # 保存缓存 (仅在推理时)
                if cache_path and not self.config.training:
                    torch.save(feat, cache_path)
            
            features[feature_type] = feat
        
        return features
    
    def _extract_single_feature(self, waveform: Tensor, feature_type: str) -> Tensor:
        """提取单个特征"""
        if feature_type == "lfcc":
            feat = extract_lfcc(
                waveform,
                sample_rate=self.sample_rate,
                n_lfcc=60,
            )
        elif feature_type == "cqcc":
            feat = extract_cqcc(
                waveform,
                sample_rate=self.sample_rate,
                n_cqcc=60,
            )
        elif feature_type == "phase":
            feat = extract_phase_features(
                waveform,
                sample_rate=self.sample_rate,
                n_phase_features=60,
                feature_type="all",
            )
        elif feature_type == "ssl":
            # SSL 特征需要特殊处理,这里用占位符
            # 实际使用时需要加载 SSL 模型
            feat = torch.randn(256, waveform.size(-1) // 320)  # 占位符
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # 确保格式: (channels, freq, time) or (channels, time)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)  # (freq, time) -> (1, freq, time)
        
        return feat
    
    def _get_cache_key(self, audio_name: str, feature_type: str) -> str:
        """生成缓存键"""
        key_str = f"{audio_name}_{feature_type}_{self.sample_rate}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Optional[Path]:
        """获取缓存路径"""
        if not self.feature_cache_dir:
            return None
        return self.feature_cache_dir / f"{cache_key}.pt"


def collate_fn_multipath(batch: List[Tuple[Dict[str, Tensor], Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
    """
    多路特征的 collate 函数
    
    Args:
        batch: [(features_dict, label), ...]
        
    Returns:
        batched_features: {feature_type: (batch, ...)}
        batched_labels: (batch,)
    """
    features_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 获取所有特征类型
    feature_types = list(features_list[0].keys())
    
    # 批处理各路特征
    batched_features = {}
    for feat_type in feature_types:
        feats = [f[feat_type] for f in features_list]
        
        # 找到最大时间长度
        max_time = max(f.size(-1) for f in feats)
        
        # 填充到相同长度
        padded_feats = []
        for feat in feats:
            if feat.size(-1) < max_time:
                pad_width = max_time - feat.size(-1)
                feat = torch.nn.functional.pad(feat, (0, pad_width))
            padded_feats.append(feat)
        
        batched_features[feat_type] = torch.stack(padded_feats, dim=0)
    
    # 批处理标签
    if isinstance(labels[0], torch.Tensor):
        batched_labels = torch.stack(labels)
    else:
        batched_labels = labels
    
    return batched_features, batched_labels
