"""
Transformer-based training pipeline for audio deepfake detection.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from trainer import (
    AudioClassificationTrainer,
    evaluate_model,
    plot_confusion_matrix,
    predict_test_set,
)
from audio_transformer_model import AudioTransformerClassifier,AudioPreprocessor, AudioDataset


def main() -> None:
    """Entry point for Transformer-based training."""
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_csv = "dataset/train.csv"
    test_csv = "dataset/test.csv"
    train_audio_dir = "dataset/train"
    test_audio_dir = "dataset/test"

    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        max_len=5,
        device=device,
    )

    full_df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df["target"],
    )

    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"类别分布 - 训练集: {train_df['target'].value_counts().to_dict()}")
    print(f"类别分布 - 验证集: {val_df['target'].value_counts().to_dict()}")

    train_df.to_csv("temp_train.csv", index=False)
    val_df.to_csv("temp_val.csv", index=False)

    cache_root = Path("cache") / "mels"
    train_cache_dir = cache_root / "train"
    test_cache_dir = cache_root / "test"

    train_dataset = AudioDataset(
        "temp_train.csv",
        train_audio_dir,
        preprocessor,
        cache_dir=train_cache_dir,
    )
    val_dataset = AudioDataset(
        "temp_val.csv",
        train_audio_dir,
        preprocessor,
        cache_dir=train_cache_dir,
    )

    batch_size = 64
    num_workers = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=3,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    model = AudioTransformerClassifier(
        input_dim=preprocessor.n_mels,
        num_classes=2,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=768,
        dropout=0.1,
        pooling="mean",
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    trainer = AudioClassificationTrainer(
        model=model,
        device=device,
        learning_rate=0.01,
        weight_decay=1e-4,
    )

    best_val_acc, best_val_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        save_path="best_audio_transformer.pth",
    )

    trainer.plot_training_history("transformer_training_history.png")

    checkpoint = torch.load("best_audio_transformer.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n=== 验证集详细评估 ===")
    val_preds, val_targets, val_probs = evaluate_model(model, val_loader, device)

    accuracy = accuracy_score(val_targets, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_targets, val_preds, average="weighted"
    )

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    print("\n分类报告:")
    print(classification_report(val_targets, val_preds, target_names=["AI Generated", "Real Human"]))

    plot_confusion_matrix(val_targets, val_preds, "transformer_confusion_matrix.png")

    print("\n=== 测试集预测 ===")
    submission = predict_test_set(
        model=model,
        test_csv=test_csv,
        test_audio_dir=test_audio_dir,
        preprocessor=preprocessor,
        device=device,
        batch_size=batch_size,
        cache_dir=test_cache_dir,
    )

    submission.to_csv("transformer_submission.csv", index=False)
    print("预测结果已保存到 'transformer_submission.csv'")

    os.remove("temp_train.csv")
    os.remove("temp_val.csv")
    print("\nTransformer 模型训练和评估完毕")


if __name__ == "__main__":
    main()
