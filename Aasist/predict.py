"""使用 AASIST 模型进行批量预测的脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_MODEL_CONFIG
from .dataset import FakeVoiceWaveDataset, WaveDatasetConfig
from .models import Model

LABEL_MAP = {0: "伪造", 1: "真人"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AASIST inference on local audio files")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("Aasist/best_aasist_local.pth"),
        help="模型权重文件路径 (可为训练脚本导出的字典或纯 state_dict)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("dataset/test.csv"),
        help="包含 audio_name 列的 CSV",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("dataset/test"),
        help="实际 WAV 文件所在目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aasist_predictions.csv"),
        help="预测结果保存路径",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备 (默认自动选择 cuda 若可用)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="可选 JSON 文件覆盖模型默认超参数 (若 checkpoint 内含 config 则优先使用)",
    )
    return parser.parse_args()


def _load_model_config(state: dict | torch.Tensor, override: Path | None) -> Dict[str, object]:
    if isinstance(state, dict) and "config" in state and isinstance(state["config"], dict):
        base_config = state["config"]
    else:
        base_config = dict(DEFAULT_MODEL_CONFIG)

    if override is not None:
        with override.open("r", encoding="utf-8") as f:
            user_conf = json.load(f)
        base_config.update(user_conf)

    return base_config


def _build_model(
    device: torch.device,
    checkpoint_path: Path,
    config_path: Path | None,
) -> tuple[Model, Dict[str, object], dict | torch.Tensor]:
    state = torch.load(checkpoint_path, map_location=device)
    model_config = _load_model_config(state, config_path)

    model = Model(model_config).to(device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, model_config, state


def _create_dataloader(
    csv_path: Path,
    audio_dir: Path,
    model_config: Dict[str, object],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = FakeVoiceWaveDataset(
        WaveDatasetConfig(
            csv_path=csv_path,
            audio_dir=audio_dir,
            sample_rate=int(model_config.get("sample_rate", 16000)),
            max_len=int(model_config.get("nb_samp", 64600)),
            training=False,
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _collect_predictions(
    model: Model,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[dict] = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_wave, batch_names in loader:
            batch_wave = batch_wave.to(device)
            _, logits = model(batch_wave, Freq_aug=False)
            probs = softmax(logits)
            preds = torch.argmax(probs, dim=1)

            batch_names_list: Iterable[str]
            if isinstance(batch_names, torch.Tensor):
                batch_names_list = batch_names.cpu().tolist()
            else:
                batch_names_list = list(batch_names)

            for name, pred, prob_vector in zip(
                batch_names_list,
                preds.cpu().tolist(),
                probs.cpu().tolist(),
            ):
                pred_int = int(pred)
                rows.append(
                    {
                        "audio_name": str(name),
                        "target": pred_int,
                        "label": LABEL_MAP.get(pred_int, "未知"),
                        "confidence": float(prob_vector[pred_int]),
                    }
                )

    return pd.DataFrame(rows)


def run_inference(
    checkpoint: Path,
    test_csv: Path,
    audio_dir: Path,
    output: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    config_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """执行推理并可选地保存预测结果。"""

    inference_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config, _ = _build_model(inference_device, checkpoint, config_path)
    loader = _create_dataloader(
        csv_path=test_csv,
        audio_dir=audio_dir,
        model_config=model_config,
        batch_size=batch_size,
        num_workers=num_workers,
        device=inference_device,
    )
    df = _collect_predictions(model, loader, inference_device)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        if verbose:
            print(f"预测结果已保存到 {output}")

    if verbose and output is None:
        print(f"预测完成，共生成 {len(df)} 条结果")

    return df


def main() -> None:
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    df = run_inference(
        checkpoint=args.checkpoint,
        test_csv=args.test_csv,
        audio_dir=args.audio_dir,
        output=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        config_path=args.config,
        verbose=True,
    )

    if args.output is not None:
        print(f"共生成 {len(df)} 条预测，已保存到 {args.output}")


if __name__ == "__main__":
    main()
