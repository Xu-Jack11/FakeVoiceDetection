"""Training utilities for Transformer audio classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import pandas as pd


@dataclass
class TrainingHistory:
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]


class AudioClassificationTrainer:
    """Generic trainer for audio classification models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.history = TrainingHistory([], [], [], [])

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(loader, desc="Training", leave=False)
        for data, target in progress:
            non_blocking = self.device.type == "cuda"
            data = data.to(self.device, non_blocking=non_blocking)
            target = target.to(self.device, non_blocking=non_blocking)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, accuracy

    def validate_epoch(self, loader: DataLoader) -> Tuple[float, float, List[int], List[int]]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            progress = tqdm(loader, desc="Validation", leave=False)
            for data, target in progress:
                non_blocking = self.device.type == "cuda"
                data = data.to(self.device, non_blocking=non_blocking)
                target = target.to(self.device, non_blocking=non_blocking)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(target.cpu().tolist())

                progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, accuracy, all_preds, all_targets

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: str,
    ) -> Tuple[float, float]:
        best_val_acc = 0.0
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 10

        print(f"Starting training for {epochs} epochs on {self.device}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader)

            self.history.train_losses.append(train_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.val_losses.append(val_loss)
            self.history.val_accuracies.append(val_acc)

            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_preds": val_preds,
                        "val_targets": val_targets,
                    },
                    save_path,
                )
                print("New best model saved.")
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {max_patience} epochs without improvement.")
                break

        print("\nTraining complete.")
        print(f"Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        return best_val_acc, best_val_loss

    def plot_training_history(self, save_path: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.train_losses, label="Train Loss", color="blue")
        ax1.plot(self.history.val_losses, label="Val Loss", color="red")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history.train_accuracies, label="Train Acc", color="blue")
        ax2.plot(self.history.val_accuracies, label="Val Acc", color="red")
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds: List[int] = []
    targets: List[int] = []
    probs: List[List[float]] = []

    with torch.no_grad():
        for data, target in loader:
            non_blocking = device.type == "cuda"
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)

            preds.extend(output.argmax(dim=1).cpu().tolist())
            targets.extend(target.cpu().tolist())
            probs.extend(probabilities.cpu().tolist())

    return preds, targets, probs


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["AI Generated", "Real Human"],
        yticklabels=["AI Generated", "Real Human"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def predict_test_set(
    model: nn.Module,
    test_csv: str,
    test_audio_dir: str,
    preprocessor,
    device: torch.device,
    batch_size: int,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    from audio_transformer_model import AudioDataset  # local import to avoid cycles

    dataset = AudioDataset(
        test_csv,
        test_audio_dir,
        preprocessor,
        cache_dir=cache_dir,
    )
    cpu_count = os.cpu_count() or 1
    num_workers = min(4, max(1, cpu_count // 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    model.eval()
    predictions: List[int] = []
    audio_names: List[str] = []

    with torch.no_grad():
        for data, names in loader:
            data = data.to(device, non_blocking=device.type == "cuda")
            output = model(data)
            preds = output.argmax(dim=1)

            predictions.extend(preds.cpu().tolist())
            audio_names.extend(list(names))

    submission = pd.DataFrame({"audio_name": audio_names, "target": predictions})
    return submission
