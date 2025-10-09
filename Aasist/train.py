"""AASIST 模型在本地数据集上的训练脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.amp as amp
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import DEFAULT_MODEL_CONFIG
from .dataset import FakeVoiceWaveDataset, WaveDatasetConfig
from .models import Model
from .predict import run_inference
from .losses import build_loss


class MaxLenWaveformCollate:
    """Pad variable-length waveforms so DataLoader workers can stack batches."""

    def __init__(self, max_len: int) -> None:
        self.max_len = int(max_len)

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        if not batch:
            raise ValueError("Empty batch received by the DataLoader.")

        batch_waveforms: List[Tensor] = []
        lengths: List[int] = []

        for sample in batch:
            waveform = sample["waveform"]
            if not isinstance(waveform, Tensor):
                raise TypeError("Expected waveform tensors in the batch.")

            copy_len = min(int(waveform.shape[-1]), self.max_len)
            padded = waveform.new_zeros(self.max_len)
            padded[:copy_len] = waveform[..., :copy_len]

            batch_waveforms.append(padded)
            lengths.append(copy_len)

        batch_dict: Dict[str, object] = {
            "waveform": torch.stack(batch_waveforms, dim=0),
            "length": torch.tensor(lengths, dtype=torch.long),
        }

        if all(sample.get("target") is not None for sample in batch):
            batch_dict["target"] = torch.stack(
                [sample["target"] for sample in batch],
                dim=0,
            )

        if all(sample.get("utt_id") is not None for sample in batch):
            batch_dict["utt_id"] = [str(sample["utt_id"]) for sample in batch]

        return batch_dict


def compute_best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    if thresholds.size == 0:
        return 0.5
    f1_scores = 2 * precision * recall / np.clip(precision + recall, a_min=1e-8, a_max=None)
    best_idx = int(np.argmax(f1_scores))
    best_idx = min(best_idx, thresholds.size - 1)
    return float(thresholds[best_idx])


def build_model(device: torch.device, config: Dict[str, object]) -> Model:
    model = Model(config).to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {nb_params:,}")
    return model


def create_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    sample_rate: int,
    max_len: int,
    model_config: Dict[str, object],
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    sample_rate = int(sample_rate)
    max_len = int(max_len)
    import pandas as pd
    from sklearn.model_selection import train_test_split

    csv_path = data_root / "train.csv"
    train_audio_dir = data_root / "train"

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=df["target"],
    )

    tmp_dir = data_root.parent
    train_split_csv = tmp_dir / "aasist_temp_train.csv"
    val_split_csv = tmp_dir / "aasist_temp_val.csv"
    train_df.to_csv(train_split_csv, index=False)
    val_df.to_csv(val_split_csv, index=False)

    train_dataset = FakeVoiceWaveDataset(
        WaveDatasetConfig(
            csv_path=train_split_csv,
            audio_dir=train_audio_dir,
            sample_rate=sample_rate,
            max_len=max_len,
            training=True,
            min_chunk_seconds=float(model_config.get("train_min_chunk_seconds", 2.0)),
            max_chunk_seconds=float(model_config.get("train_max_chunk_seconds", 6.0)),
            eval_chunk_seconds=float(model_config.get("eval_chunk_seconds", 4.0)),
            telephony_aug=bool(model_config.get("telephony_aug", True)),
        )
    )
    val_dataset = FakeVoiceWaveDataset(
        WaveDatasetConfig(
            csv_path=val_split_csv,
            audio_dir=train_audio_dir,
            sample_rate=sample_rate,
            max_len=max_len,
            training=False,
            min_chunk_seconds=float(model_config.get("train_min_chunk_seconds", 2.0)),
            max_chunk_seconds=float(model_config.get("train_max_chunk_seconds", 6.0)),
            eval_chunk_seconds=float(model_config.get("eval_chunk_seconds", 4.0)),
            telephony_aug=False,
        )
    )

    collate_fn = MaxLenWaveformCollate(max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: Model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    branch_loss_weight: float,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    progress = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", leave=False)
    for batch in progress:
        batch_wave = batch["waveform"].to(device)
        batch_target = batch["target"].to(device)
        utt_field = batch.get("utt_id", [])
        if isinstance(utt_field, (list, tuple)):
            utt_ids = [str(u) for u in utt_field]
        elif utt_field is None:
            utt_ids = []
        else:
            utt_ids = [str(utt_field)]

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast('cuda',enabled=use_amp):
            branch_logits, logits = model(batch_wave, utt_ids=utt_ids, training=True)
            fused_loss = criterion(logits, batch_target)
            if branch_loss_weight > 0:
                aux_losses = [criterion(bl, batch_target) for bl in branch_logits.values()]
                aux_loss = torch.stack(aux_losses).mean()
                loss = fused_loss + branch_loss_weight * aux_loss
            else:
                loss = fused_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_wave.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_target).sum().item()
        seen += batch_target.size(0)

        avg_loss_so_far = total_loss / max(seen, 1)
        acc_so_far = correct / max(seen, 1)
        progress.set_postfix({"loss": f"{avg_loss_so_far:.4f}", "acc": f"{acc_so_far * 100:.2f}%"})

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


def evaluate(
    model: Model,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    temperature: float,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float, float, str, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []
    correct = 0
    seen = 0
    all_logits: list[Tensor] = []

    with torch.no_grad():
        progress = tqdm(loader, desc=f"Validate [{epoch}/{total_epochs}]", leave=False)
        for batch in progress:
            batch_wave = batch["waveform"].to(device)
            batch_target = batch["target"].to(device)
            utt_field = batch.get("utt_id", [])
            if isinstance(utt_field, (list, tuple)):
                utt_ids = [str(u) for u in utt_field]
            elif utt_field is None:
                utt_ids = []
            else:
                utt_ids = [str(utt_field)]
            with amp.autocast('cuda',enabled=use_amp):
                _, logits = model(batch_wave, utt_ids=utt_ids, training=False)
                scaled_logits = logits / max(temperature, 1e-6)
                loss = criterion(scaled_logits, batch_target)
            total_loss += loss.item() * batch_wave.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(batch_target.detach().cpu().numpy())
            correct += (preds == batch_target).sum().item()
            seen += batch_target.size(0)
            all_logits.append(logits.detach().cpu())

            avg_loss_so_far = total_loss / max(seen, 1)
            acc_so_far = correct / max(seen, 1)
            progress.set_postfix({"loss": f"{avg_loss_so_far:.4f}", "acc": f"{acc_so_far * 100:.2f}%"})

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    report = classification_report(
        all_targets,
        all_preds,
        target_names=["AI Generated", "Real Human"],
        digits=4,
    )
    logits_tensor = torch.cat(all_logits, dim=0) / max(temperature, 1e-6)
    probs = torch.softmax(logits_tensor, dim=1)[:, 1].numpy()
    return avg_loss, acc, f1, report, probs, np.asarray(all_targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AASIST on local dataset")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument(
        "--class-counts",
        type=int,
        nargs="*",
        default=None,
        help="各类别样本数量 (用于 Class-Balanced Loss)",
    )
    parser.add_argument(
        "--branch-loss-weight",
        type=float,
        default=0.2,
        help="多分支辅助 loss 的权重 (0 表示关闭)",
    )
    parser.add_argument(
        "--temperature-scale",
        type=float,
        default=1.0,
        help="验证阶段 logits 温度缩放因子",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("Aasist/best_model.pth"))
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备 (例如 cuda 或 cpu)，默认自动检测",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选的 JSON 配置文件，用于覆盖默认的模型超参数",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs="*",
        choices=["lfcc", "cqcc", "phase", "ssl", "aasist"],
        default=None,
        help="手动指定需要启用的特征分支（默认启用全部）",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="可选测试集 CSV 路径（默认使用 data-root/test.csv）",
    )
    parser.add_argument(
        "--test-audio-dir",
        type=Path,
        default=None,
        help="可选测试集音频目录（默认使用 data-root/test）",
    )
    parser.add_argument(
        "--predict-output",
        type=Path,
        default=Path("Aasist/predictions_after_train.csv"),
        help="自动预测结果输出路径",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=64,
        help="自动预测时的数据批大小",
    )
    parser.add_argument(
        "--no-auto-predict",
        action="store_true",
        help="训练结束后跳过自动预测",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_config = dict(DEFAULT_MODEL_CONFIG)
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            user_conf = json.load(f)
        model_config.update(user_conf)

    if args.feature_types is not None:
        enabled = {name.lower() for name in args.feature_types}
        if not enabled:
            raise ValueError("--feature-types 至少需要指定一个分支")
        for name in ("lfcc", "cqcc", "phase", "ssl", "aasist"):
            model_config[f"enable_{name}"] = name in enabled

    model = build_model(device, model_config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = build_loss(
        args.loss,
        num_classes=2,
        class_counts=args.class_counts,
    )

    use_amp = device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)

    data_root = args.data_root
    test_csv_path = args.test_csv or (data_root / "test.csv")
    test_audio_dir = args.test_audio_dir or (data_root / "test")
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config.get("sample_rate", 16000),
        max_len=model_config.get("nb_samp", 64600),
        model_config=model_config,
        val_split=args.val_split,
        seed=args.seed,
    )

    best_f1 = 0.0
    best_state = None
    best_threshold = 0.5

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            branch_loss_weight=args.branch_loss_weight,
            epoch=epoch + 1,
            total_epochs=args.epochs,
        )
        (
            val_loss,
            val_acc,
            val_f1,
            report,
            val_probs,
            val_targets,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            temperature=args.temperature_scale,
            epoch=epoch + 1,
            total_epochs=args.epochs,
        )
        val_threshold = compute_best_threshold(val_probs, val_targets)
        thresh_preds = (val_probs >= val_threshold).astype(int)
        thresh_f1 = f1_score(val_targets, thresh_preds)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%"
        )
        print(
            "  Val Loss: {:.4f} | Val Acc: {:.2f}% | Val F1: {:.4f} | Val F1@thr: {:.4f} | Thr: {:.3f}".format(
                val_loss,
                val_acc * 100,
                val_f1,
                thresh_f1,
                val_threshold,
            )
        )
        print(report)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "metrics": {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                },
                "threshold": val_threshold,
                "temperature": args.temperature_scale,
            }
            print(f"  -> 新的最佳模型 (F1={val_f1:.4f})")
            best_threshold = val_threshold

    if best_state is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.output)
        print(f"最佳模型已保存到 {args.output}")
        if "threshold" in best_state:
            print(
                f"验证集最优阈值: {best_state['threshold']:.3f} (温度缩放 {best_state.get('temperature', 1.0):.2f})"
            )

        if args.no_auto_predict:
            print("已根据 --no-auto-predict 参数跳过自动预测。")
        else:
            if test_csv_path.exists() and test_audio_dir.exists():
                try:
                    print("开始使用最佳模型进行自动预测…")
                    run_inference(
                        checkpoint=args.output,
                        test_csv=test_csv_path,
                        audio_dir=test_audio_dir,
                        output=args.predict_output,
                        batch_size=args.predict_batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        config_path=args.config,
                        verbose=False,
                    )
                    print(f"自动预测完成，结果保存至 {args.predict_output}")
                except Exception as exc:
                    print(f"自动预测失败: {exc}")
            else:
                print("测试集 CSV 或音频目录不存在，自动预测已跳过。")
    else:
        print("未找到可保存的模型")


if __name__ == "__main__":
    main()
