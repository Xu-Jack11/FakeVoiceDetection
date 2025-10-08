"""AASIST3 推理脚本

支持:
1. 滑窗推理 (3-5s, 50% 重叠)
2. 多chunk集成 (mean/top-k pooling)
3. 温度缩放校准
4. 批量预测
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .data.dataset_multipath import AASIST3Dataset, AASIST3DatasetConfig
from .models.AASIST3 import create_aasist3_model


class SlidingWindowDataset(Dataset):
    """滑窗推理数据集
    
    将长音频切分为多个重叠的 chunk
    """
    
    def __init__(
        self,
        audio_paths: List[Path],
        sample_rate: int = 16000,
        window_size: float = 4.0,  # 秒
        hop_size: float = 2.0,  # 50% 重叠
        feature_types: List[str] = ["lfcc", "cqcc", "phase"],
    ):
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        self.window_samples = int(window_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)
        self.feature_types = feature_types
        
        # 预计算每个音频的 chunk 信息
        self.chunks_info = []
        for audio_path in audio_paths:
            info, _ = sf.info(audio_path)
            num_samples = info.frames
            
            # 计算 chunk 数量
            if num_samples <= self.window_samples:
                # 短音频,作为单个 chunk
                self.chunks_info.append((audio_path, 0, num_samples))
            else:
                # 长音频,滑窗切分
                for start in range(0, num_samples - self.window_samples + 1, self.hop_samples):
                    end = start + self.window_samples
                    self.chunks_info.append((audio_path, start, end))
                
                # 确保覆盖到音频末尾
                if self.chunks_info[-1][2] < num_samples:
                    self.chunks_info.append((audio_path, num_samples - self.window_samples, num_samples))
    
    def __len__(self) -> int:
        return len(self.chunks_info)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], str, int]:
        audio_path, start, end = self.chunks_info[idx]
        
        # 加载音频片段
        waveform, sr = sf.read(audio_path, start=start, stop=end)
        waveform = np.asarray(waveform, dtype=np.float32)
        
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        
        # 重采样 (如果需要)
        if sr != self.sample_rate:
            from scipy import signal
            waveform = signal.resample(waveform, int(len(waveform) * self.sample_rate / sr))
        
        # 填充到窗口大小
        if len(waveform) < self.window_samples:
            waveform = np.pad(waveform, (0, self.window_samples - len(waveform)))
        
        waveform = torch.from_numpy(waveform).float()
        
        # 提取特征 (简化版,实际应使用 dataset_multipath 的方法)
        features = self._extract_features(waveform)
        
        return features, str(audio_path.name), idx
    
    def _extract_features(self, waveform: Tensor) -> Dict[str, Tensor]:
        """提取特征 (占位符)"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from features import extract_lfcc, extract_cqcc, extract_phase_features
        
        features = {}
        for ft in self.feature_types:
            if ft == "lfcc":
                feat = extract_lfcc(waveform, sample_rate=self.sample_rate, n_lfcc=60)
            elif ft == "cqcc":
                feat = extract_cqcc(waveform, sample_rate=self.sample_rate, n_cqcc=60)
            elif ft == "phase":
                feat = extract_phase_features(waveform, sample_rate=self.sample_rate, n_phase_features=60)
            else:
                feat = torch.randn(60, 100)  # 占位符
            
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            features[ft] = feat
        
        return features


class TemperatureScaling(nn.Module):
    """温度缩放校准
    
    用于校准模型输出的置信度
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ):
        """在验证集上校准温度"""
        from torch.optim import LBFGS
        
        # 收集验证集的 logits 和 labels
        all_logits = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for features_dict, labels in val_loader:
                features_dict = {k: v.to(device) for k, v in features_dict.items()}
                logits = model(features_dict)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # 优化温度
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            loss = nll_criterion(self(all_logits), all_labels)
            loss.backward()
            return loss
        
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)
        optimizer.step(eval_loss)
        
        print(f"Calibrated temperature: {self.temperature.item():.4f}")


@torch.no_grad()
def predict_sliding_window(
    model: nn.Module,
    audio_paths: List[Path],
    device: torch.device,
    batch_size: int = 16,
    window_size: float = 4.0,
    hop_size: float = 2.0,
    feature_types: List[str] = ["lfcc", "cqcc", "phase"],
    pooling: str = "mean",  # "mean", "max", "top-k"
    temperature_scaling: Optional[TemperatureScaling] = None,
) -> Dict[str, float]:
    """
    滑窗推理
    
    Args:
        model: AASIST3 模型
        audio_paths: 音频文件路径列表
        device: 计算设备
        batch_size: 批大小
        window_size: 窗口大小(秒)
        hop_size: 跳跃大小(秒)
        feature_types: 特征类型
        pooling: chunk 聚合方式 ("mean", "max", "top-k")
        temperature_scaling: 温度缩放模块
        
    Returns:
        predictions: {audio_name: probability}
    """
    model.eval()
    
    # 创建滑窗数据集
    dataset = SlidingWindowDataset(
        audio_paths,
        window_size=window_size,
        hop_size=hop_size,
        feature_types=feature_types,
    )
    
    # 数据加载器
    from .data.dataset_multipath import collate_fn_multipath
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn_multipath,
    )
    
    # 收集每个音频的所有 chunk 预测
    audio_chunk_probs = {}
    
    for features_dict, audio_names, _ in tqdm(loader, desc="Inference"):
        features_dict = {k: v.to(device) for k, v in features_dict.items()}
        
        # 模型推理
        logits = model(features_dict)
        
        # 温度缩放
        if temperature_scaling:
            logits = temperature_scaling(logits)
        
        # 计算概率
        probs = torch.softmax(logits, dim=1)[:, 1]  # 假阳性概率
        probs = probs.cpu().numpy()
        
        # 按音频名分组
        for audio_name, prob in zip(audio_names, probs):
            if audio_name not in audio_chunk_probs:
                audio_chunk_probs[audio_name] = []
            audio_chunk_probs[audio_name].append(prob)
    
    # 聚合各音频的 chunk 预测
    predictions = {}
    for audio_name, chunk_probs in audio_chunk_probs.items():
        chunk_probs = np.array(chunk_probs)
        
        if pooling == "mean":
            final_prob = chunk_probs.mean()
        elif pooling == "max":
            final_prob = chunk_probs.max()
        elif pooling == "top-k":
            k = max(1, len(chunk_probs) // 2)  # top 50%
            final_prob = np.partition(chunk_probs, -k)[-k:].mean()
        else:
            final_prob = chunk_probs.mean()
        
        predictions[audio_name] = float(final_prob)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="AASIST3 Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_dir", type=str, required=True, help="测试音频目录")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="输出CSV路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--window_size", type=float, default=4.0, help="滑窗大小(秒)")
    parser.add_argument("--hop_size", type=float, default=2.0, help="跳跃大小(秒)")
    parser.add_argument("--pooling", type=str, default="mean", help="聚合方式")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint["model_config"]
    
    model = create_aasist3_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 获取音频文件列表
    test_dir = Path(args.test_dir)
    audio_paths = sorted(test_dir.glob("*.wav"))
    print(f"Found {len(audio_paths)} audio files")
    
    # 推理
    print("Running inference...")
    feature_types = list(model_config["feature_configs"].keys())
    predictions = predict_sliding_window(
        model=model,
        audio_paths=audio_paths,
        device=device,
        batch_size=args.batch_size,
        window_size=args.window_size,
        hop_size=args.hop_size,
        feature_types=feature_types,
        pooling=args.pooling,
    )
    
    # 应用阈值
    results = []
    for audio_name, prob in predictions.items():
        pred_label = 1 if prob >= args.threshold else 0
        results.append({
            "audio_name": audio_name,
            "probability": prob,
            "prediction": pred_label,
        })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")
    
    # 统计
    fake_count = (df["prediction"] == 1).sum()
    real_count = (df["prediction"] == 0).sum()
    print(f"\nPrediction Summary:")
    print(f"Fake: {fake_count} ({fake_count/len(df)*100:.1f}%)")
    print(f"Real: {real_count} ({real_count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
