"""AASIST3 训练脚本

支持:
1. 多路特征训练
2. 高级损失函数  
3. 验证集阈值优化
4. 混合精度训练
5. 学习率调度
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data.dataset_multipath import (
    AASIST3Dataset,
    AASIST3DatasetConfig,
    collate_fn_multipath,
)
from .losses import create_loss_function
from .models.AASIST3 import create_aasist3_model


def create_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    feature_types: list[str],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    csv_path = data_root / "train.csv"
    train_audio_dir = data_root / "train"
    
    # 划分训练/验证集
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=df["target"],
    )
    
    # 保存临时CSV
    tmp_dir = data_root.parent
    train_csv = tmp_dir / "aasist3_temp_train.csv"
    val_csv = tmp_dir / "aasist3_temp_val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # 创建数据集
    train_dataset = AASIST3Dataset(
        AASIST3DatasetConfig(
            csv_path=train_csv,
            audio_dir=train_audio_dir,
            sample_rate=16000,
            training=True,
            min_chunk_duration=2.0,
            max_chunk_duration=6.0,
            feature_types=feature_types,
            feature_cache_dir=tmp_dir / "feature_cache",
            use_vad=True,
            use_augmentation=True,
            aug_prob=0.8,
        )
    )
    
    val_dataset = AASIST3Dataset(
        AASIST3DatasetConfig(
            csv_path=val_csv,
            audio_dir=train_audio_dir,
            sample_rate=16000,
            training=False,
            fixed_chunk_duration=4.0,
            feature_types=feature_types,
            feature_cache_dir=tmp_dir / "feature_cache",
            use_vad=True,
            use_augmentation=False,
        )
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_multipath,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_multipath,
    )
    
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> float:
    """训练一个 epoch"""
    
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for features_dict, targets in pbar:
        # 移动到设备
        features_dict = {k: v.to(device) for k, v in features_dict.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if scaler:
            with autocast():
                logits = model(features_dict)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features_dict)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """验证模型"""
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    for features_dict, targets in tqdm(loader, desc="Validation"):
        features_dict = {k: v.to(device) for k, v in features_dict.items()}
        targets = targets.to(device)
        
        logits = model(features_dict)
        loss = criterion(logits, targets)
        
        probs = torch.softmax(logits, dim=1)[:, 1]  # 假阳性概率
        preds = (probs > 0.5).long()
        
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # 计算指标
    val_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return val_loss, f1, all_probs, all_targets


def find_optimal_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, float]:
    """在验证集上找到最优阈值"""
    
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    print(f"\nOptimal Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(description="Train AASIST3 model")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--feature_types", nargs="+", default=["lfcc", "cqcc", "phase"], help="特征类型")
    parser.add_argument("--loss_type", type=str, default="focal", help="损失函数类型")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据加载器
    print("Creating dataloaders...")
    data_root = Path(args.data_root)
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_types=args.feature_types,
        seed=args.seed,
    )
    
    # 创建模型
    print("Creating model...")
    feature_configs = {ft: 60 for ft in args.feature_types}
    if "ssl" in feature_configs:
        feature_configs["ssl"] = 256  # SSL 特征维度更高
    
    model_config = {
        "feature_configs": feature_configs,
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0],
        "num_classes": 2,
        "fusion_type": "mlp",
        "fusion_config": {
            "hidden_dims": [128, 64],
            "dropout": 0.3,
        },
    }
    
    model = create_aasist3_model(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    criterion = create_loss_function(
        loss_type=args.loss_type,
        num_classes=2,
        alpha=0.25,
        gamma=2.0,
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = GradScaler() if args.use_amp else None
    
    # 训练循环
    best_f1 = 0.0
    best_threshold = 0.5
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        history["train_loss"].append(train_loss)
        
        # 验证
        val_loss, val_f1, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        
        # 找到最优阈值
        threshold, f1_at_threshold = find_optimal_threshold(val_probs, val_targets)
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 保存最佳模型
        if f1_at_threshold > best_f1:
            best_f1 = f1_at_threshold
            best_threshold = threshold
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "best_threshold": best_threshold,
                    "model_config": model_config,
                },
                output_dir / "best_aasist3.pth",
            )
            print(f"Saved best model with F1={best_f1:.4f}, threshold={best_threshold:.4f}")
    
    # 保存训练历史
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    main()
