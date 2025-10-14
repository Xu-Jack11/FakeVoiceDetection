"""Training entry-point for the audio fake detector."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from src.dataio.dataset import AudioDeepfakeDataset, collate_audio_samples
from src.models import create_model
from src.predict import run_prediction
from src.utils.metrics import compute_f1_scores, format_f1_log


def resolve_optional_path(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
    if path_value is None:
        return None
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        path_obj = path_obj.resolve()
    return path_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train audio deepfake detector.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_datasets(cfg: Dict, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg["data"]
    train_df = pd.read_csv(data_cfg["train_csv"])
    if data_cfg.get("limit_samples"):
        train_df = train_df.sample(n=int(data_cfg["limit_samples"]), random_state=seed)
    val_csv = data_cfg.get("val_csv")
    if val_csv:
        val_df = pd.read_csv(val_csv)
    else:
        split_cfg = data_cfg.get("stratified_split", {})
        if not split_cfg.get("enable", False):
            raise ValueError("Validation CSV missing and stratified split disabled.")
        val_ratio = float(split_cfg.get("val_ratio", 0.1))
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1.")
        stratify_labels = train_df[data_cfg.get("target_column", "target")]
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify_labels,
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_dataloaders(cfg: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    audio_dir = data_cfg.get("audio_dir")
    train_audio_dir = data_cfg.get("train_audio_dir", audio_dir)
    val_audio_dir = data_cfg.get("val_audio_dir", audio_dir)
    base_kwargs = {
        "sample_rate": sample_rate,
        "audio_column": data_cfg.get("audio_column", "audio_name"),
        "target_column": data_cfg.get("target_column", "target"),
    }
    crop_cfg = data_cfg.get("crop", {})
    train_kwargs = dict(base_kwargs)
    val_kwargs = dict(base_kwargs)
    if crop_cfg.get("enable", False):
        min_sec = crop_cfg.get("min_sec", 3.0)
        max_sec = crop_cfg.get("max_sec", 5.0)
        train_kwargs.update(
            {
                "crop_enable": True,
                "crop_min_sec": min_sec,
                "crop_max_sec": max_sec,
                "crop_mode": crop_cfg.get("train_mode", "random"),
            }
        )
        if crop_cfg.get("apply_to_validation", True):
            val_kwargs.update(
                {
                    "crop_enable": True,
                    "crop_min_sec": min_sec,
                    "crop_max_sec": max_sec,
                    "crop_mode": crop_cfg.get("eval_mode", "center"),
                }
            )
    rawboost_cfg = data_cfg.get("rawboost")
    if rawboost_cfg and rawboost_cfg.get("enable", False):
        train_kwargs["rawboost_cfg"] = rawboost_cfg
        if rawboost_cfg.get("apply_to_validation", False):
            val_kwargs["rawboost_cfg"] = rawboost_cfg
    train_dataset = AudioDeepfakeDataset(
        data=train_df,
        audio_dir=train_audio_dir,
        **train_kwargs,
    )
    val_dataset = AudioDeepfakeDataset(
        data=val_df,
        audio_dir=val_audio_dir,
        **val_kwargs,
    )
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_audio_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_audio_samples,
    )
    return train_loader, val_loader


def create_optimizer(model: torch.nn.Module, cfg: Dict) -> AdamW:
    train_cfg = cfg["training"]
    encoder_lr = float(train_cfg.get("encoder_lr", 3e-5))
    head_lr = float(train_cfg.get("head_lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    if hasattr(model, "parameter_groups"):
        param_groups = model.parameter_groups(encoder_lr, head_lr, weight_decay)
    else:
        param_groups = [{"params": model.parameters(), "lr": head_lr, "weight_decay": weight_decay}]
    return AdamW(param_groups)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    accumulation_steps: int,
    grad_clip: float,
) -> float:
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    step = 0
    amp_enabled = scaler.is_enabled()
    amp_device_type = "cuda" if device.type == "cuda" else device.type
    pbar = tqdm(
        enumerate(dataloader, start=1),
        total=len(dataloader),
        desc="Train",
        leave=True,
        dynamic_ncols=True,
    )
    for step_idx, batch in pbar:
        labels = batch["labels"].to(device)
        inputs = batch["input_values"].to(device)
        attn = batch["attention_mask"].to(device)
        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits = model(inputs, attention_mask=attn)
            loss = criterion(logits, labels)
        loss_value = loss.detach().item()
        running_loss += loss_value
        loss = loss / accumulation_steps
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        step += 1
        avg_loss = running_loss / step_idx
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"}, refresh=False)
        if step % accumulation_steps == 0:
            if grad_clip > 0:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
    # Flush remainder if accumulation didn't align exactly
    if step % accumulation_steps != 0:
        if grad_clip > 0:
            if amp_enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses = []
    targets, preds = [], []
    pbar = tqdm(dataloader, desc="Validate", leave=True, dynamic_ncols=True)
    for batch in pbar:
        labels = batch["labels"].to(device)
        inputs = batch["input_values"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(inputs, attention_mask=attn)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
        if losses:
            pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"}, refresh=False)
    macro_f1, per_class = compute_f1_scores(targets, preds)
    return {
        "loss": float(np.mean(losses) if losses else math.nan),
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_metric: float,
    cfg: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "model_cfg": cfg["model"],
            "data_cfg": {
                k: cfg["data"].get(k)
                for k in ("sample_rate", "audio_column", "target_column")
            },
        },
        path,
    )


def load_checkpoint_if_available(
    path: Optional[Path],
    model: torch.nn.Module,
    optimizer: AdamW,
) -> Tuple[int, float]:
    if path is None or not path.exists():
        return 0, -math.inf
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", -math.inf))
    print(f"Resumed training from {path} at epoch {start_epoch}.")
    return start_epoch, best_metric


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    train_df, val_df = prepare_datasets(cfg, seed)
    train_loader, val_loader = build_dataloaders(cfg, train_df, val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg["model"])
    model.to(device)

    optimizer = create_optimizer(model, cfg)

    train_cfg = cfg["training"]
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    checkpoint_name = train_cfg.get("checkpoint_name", "best.ckpt")
    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    amp_enabled = train_cfg.get("amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    resume_path = resolve_optional_path(train_cfg.get("resume_from"))
    start_epoch, best_metric = load_checkpoint_if_available(resume_path, model, optimizer)

    epochs = int(train_cfg.get("epochs", 10))
    min_epochs = int(train_cfg.get("min_epochs", 1))
    patience = int(train_cfg.get("patience", 3))
    no_improve_epochs = 0
    best_checkpoint_path: Optional[Path] = resume_path if resume_path and resume_path.exists() else None
    print(f"Training on device: {device} | Model: {cfg['model']['type']}")

    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip,
        )
        print(f"Train loss: {train_loss:.4f}")
        metrics = evaluate(model, val_loader, device=device)
        print(f"Validation loss: {metrics['loss']:.4f}")
        print(format_f1_log(metrics["macro_f1"], metrics["per_class"]))

        if metrics["macro_f1"] > best_metric:
            best_metric = metrics["macro_f1"]
            no_improve_epochs = 0
            ckpt_path = output_dir / checkpoint_name
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_metric, cfg)
            best_checkpoint_path = ckpt_path.resolve()
            print(f"Saved new best checkpoint to {ckpt_path} (macro-F1={best_metric:.4f}).")
        else:
            no_improve_epochs += 1
            if epoch >= min_epochs and no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_checkpoint_path is None:
        candidate = output_dir / checkpoint_name
        if candidate.exists():
            best_checkpoint_path = candidate.resolve()

    print("Training finished.")

    post_cfg = cfg.get("post_training", {})
    if post_cfg.get("run_prediction", False):
        predict_config_value = post_cfg.get("predict_config")
        if not predict_config_value:
            print("Post-training prediction skipped: predict_config not set.")
        else:
            predict_config_path = resolve_optional_path(predict_config_value)
            csv_override = resolve_optional_path(post_cfg.get("csv"))
            out_override = resolve_optional_path(post_cfg.get("out"))
            checkpoint_override = resolve_optional_path(post_cfg.get("checkpoint"))
            if checkpoint_override is None:
                checkpoint_override = best_checkpoint_path
            if checkpoint_override is None:
                print("Post-training prediction skipped: checkpoint file not available.")
            else:
                print("\nStarting post-training prediction...")
                run_prediction(
                    config_path=predict_config_path,
                    csv_path=csv_override,
                    out_path=out_override,
                    checkpoint_path=checkpoint_override,
                )


if __name__ == "__main__":
    main()
