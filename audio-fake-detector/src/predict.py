"""Prediction script for the audio fake detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from src.dataio.dataset import AudioDeepfakeDataset, collate_audio_samples
from src.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--csv", type=str, help="CSV file listing audio files.")
    parser.add_argument("--out", type=str, help="Path to save predictions CSV.")
    parser.add_argument("--model", type=str, help="Override checkpoint path.")
    return parser.parse_args()


def ensure_path(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
    if path_value is None:
        return None
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        path_obj = path_obj.resolve()
    return path_obj


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_prediction(
    config_path: Union[str, Path],
    csv_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    config_path = ensure_path(config_path)
    if config_path is None:
        raise ValueError("Config path must be provided.")
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    output_cfg = cfg.get("output", {})

    csv_candidate = csv_path if csv_path is not None else data_cfg.get("csv")
    csv_resolved = ensure_path(csv_candidate)
    if csv_resolved is None:
        raise ValueError("Input CSV must be provided via argument or config.")
    df = pd.read_csv(csv_resolved)

    checkpoint_candidate = (
        checkpoint_path
        or model_cfg.get("checkpoint")
        or model_cfg.get("checkpoint_path")
        or output_cfg.get("checkpoint")
    )
    checkpoint_resolved = ensure_path(checkpoint_candidate)
    if checkpoint_resolved is None:
        raise ValueError("Checkpoint path not found in arguments or config.")
    if not checkpoint_resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_resolved}")

    out_candidate = out_path if out_path is not None else output_cfg.get("path", "predictions.csv")
    out_resolved = ensure_path(out_candidate)
    if out_resolved is None:
        out_resolved = Path("predictions.csv").resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_resolved, map_location=device)
    model_section = checkpoint.get("model_cfg", model_cfg)
    model = create_model(model_section)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    data_meta = checkpoint.get("data_cfg", {})
    sample_rate = data_cfg.get("sample_rate", data_meta.get("sample_rate", 16000))
    audio_dir = ensure_path(data_cfg.get("audio_dir"))
    crop_cfg = data_cfg.get("crop", {})
    dataset_kwargs = {
        "sample_rate": sample_rate,
        "audio_column": data_cfg.get("audio_column", "audio_name"),
        "target_column": data_cfg.get("target_column", "target"),
    }
    if crop_cfg.get("enable", False):
        dataset_kwargs.update(
            {
                "crop_enable": True,
                "crop_min_sec": crop_cfg.get("min_sec", 3.0),
                "crop_max_sec": crop_cfg.get("max_sec", 5.0),
                "crop_mode": crop_cfg.get("mode", "center"),
            }
        )
    dataset = AudioDeepfakeDataset(
        data=df,
        audio_dir=audio_dir,
        **dataset_kwargs,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=True,
        collate_fn=collate_audio_samples,
    )

    print(
        f"Running prediction on {len(dataset)} files "
        f"(CSV={csv_resolved}) using checkpoint {checkpoint_resolved}"
    )

    audio_names, scores, preds = [], [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Predict", leave=True, dynamic_ncols=True)
        for batch in pbar:
            inputs = batch["input_values"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(inputs, attention_mask=attn)
            probabilities = torch.softmax(logits, dim=1)
            score_ai = probabilities[:, 0]
            prediction = (score_ai < 0.5).long()
            audio_names.extend(batch["audio_names"])
            scores.extend(score_ai.cpu().tolist())
            preds.extend(prediction.cpu().tolist())
            pbar.set_postfix({"processed": len(audio_names)}, refresh=False)

    output_df = pd.DataFrame(
        {
            "audio_name": audio_names,
            "score_ai": scores,
            "pred": preds,
        }
    )
    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_resolved, index=False)
    print(f"Saved predictions to {out_resolved}")
    return output_df


def main() -> None:
    args = parse_args()
    run_prediction(
        config_path=args.config,
        csv_path=args.csv,
        out_path=args.out,
        checkpoint_path=args.model,
    )


if __name__ == "__main__":
    main()
