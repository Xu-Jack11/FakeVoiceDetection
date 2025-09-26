"""Prediction script for the ResNet-based audio classifier."""

import argparse
import os
from typing import Dict, Type

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from audio_resnet_model import (
    AudioPreprocessor,
    AudioDataset,
    AudioResNet18,
    AudioResNet34,
    AudioResNet50,
)
from torch.cuda.amp import autocast

MODEL_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "resnet18": AudioResNet18,
    "resnet34": AudioResNet34,
    "resnet50": AudioResNet50,
}


def build_model(name: str, device: torch.device, input_channels: int) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](num_classes=2, input_channels=input_channels)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device,weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    predictions = []
    audio_names = []

    with torch.no_grad():
        for data, names in tqdm(loader, desc="Predicting"):
            data = data.to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                output = model(data)
            pred = output.argmax(dim=1)

            predictions.extend(pred.cpu().tolist())
            if isinstance(names, (list, tuple)):
                audio_names.extend(list(names))
            else:
                audio_names.extend(names.tolist())

    return pd.DataFrame({"audio_name": audio_names, "target": predictions})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions with a ResNet audio model.")
    parser.add_argument("--checkpoint", default="best_audio_model.pth", help="Path to model checkpoint")
    parser.add_argument("--model", default="resnet18", choices=list(MODEL_REGISTRY), help="Model architecture")
    parser.add_argument("--test-csv", default="dataset/test.csv", help="CSV file listing test audio")
    parser.add_argument("--audio-dir", default="dataset/test", help="Directory containing test audio files")
    parser.add_argument("--output", default="resnet_predictions.csv", help="Where to save predictions")
    parser.add_argument("--batch-size", type=int, default=32, help="Prediction batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--device", default=None, help="Device to run on (e.g., cuda, cpu)")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=5, help="Audio clip length in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    preprocessor = AudioPreprocessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_len=args.max_len,
        device=device,
    )

    dataset = AudioDataset(args.test_csv, args.audio_dir, preprocessor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.model, device, preprocessor.feature_channels)
    load_checkpoint(model, args.checkpoint, device)

    submission = predict(model, loader, device)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
