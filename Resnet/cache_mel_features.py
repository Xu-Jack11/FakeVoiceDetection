"""Offline caching utilities for Mel-spectrogram features."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from audio_resnet_model import AudioPreprocessor

CACHE_META_FILENAME = "metadata.json"
CACHE_EXT = ".pt"


@dataclass(frozen=True)
class MelCacheConfig:
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    max_len: int

    @classmethod
    def from_preprocessor(cls, preprocessor: AudioPreprocessor) -> "MelCacheConfig":
        return cls(
            sample_rate=preprocessor.sample_rate,
            n_mels=preprocessor.n_mels,
            n_fft=preprocessor.n_fft,
            hop_length=preprocessor.hop_length,
            max_len=preprocessor.max_len,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "MelCacheConfig":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)


def _validate_or_write_metadata(cache_dir: Path, config: MelCacheConfig, overwrite: bool) -> None:
    meta_path = cache_dir / CACHE_META_FILENAME
    if meta_path.exists():
        existing = MelCacheConfig.load(meta_path)
        if existing != config:
            if overwrite:
                config.dump(meta_path)
            else:
                raise ValueError(
                    "Cache configuration mismatch. Existing metadata differs from current preprocessor settings. "
                    "Set --overwrite to rebuild the cache or remove the old directory."
                )
    else:
        config.dump(meta_path)


def _cache_single_file(
    audio_path: Path,
    dest_path: Path,
    preprocessor: AudioPreprocessor,
    overwrite: bool,
) -> bool:
    if dest_path.exists() and not overwrite:
        return False

    mel_spec = preprocessor.process_audio(str(audio_path))
    if mel_spec is None:
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mel_spec, dest_path)
    return True


def cache_mel_features(
    csv_path: Path,
    audio_dir: Path,
    cache_dir: Path,
    preprocessor: AudioPreprocessor,
    overwrite: bool = False,
    skip_existing: bool = True,
) -> None:
    import pandas as pd

    csv_path = csv_path.resolve()
    audio_dir = audio_dir.resolve()
    cache_dir = cache_dir.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    config = MelCacheConfig.from_preprocessor(preprocessor)
    _validate_or_write_metadata(cache_dir, config, overwrite=overwrite)

    df = pd.read_csv(csv_path)
    if "audio_name" not in df.columns:
        raise ValueError("CSV must contain an 'audio_name' column.")

    created = 0
    skipped = 0
    missing = 0

    for audio_name in tqdm(df["audio_name"], desc=f"Caching -> {cache_dir.name}"):
        audio_path = audio_dir / audio_name
        if not audio_path.exists():
            missing += 1
            continue

        dest_path = cache_dir / f"{audio_name}{CACHE_EXT}"
        if dest_path.exists() and skip_existing and not overwrite:
            skipped += 1
            continue

        if _cache_single_file(audio_path, dest_path, preprocessor, overwrite=overwrite):
            created += 1
        else:
            skipped += 1

    if missing:
        print(f"[cache_mel_features] WARNING: {missing} files listed in {csv_path} were missing from {audio_dir}.")
    print(
        f"[cache_mel_features] Completed for {csv_path.name}: created {created}, "
        f"skipped {skipped}, missing {missing}."
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Mel-spectrogram features for audio datasets.")
    parser.add_argument("csv", help="Path to CSV file containing audio_name column")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("cache_dir", help="Destination directory to store cached features")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="Recompute all cached files even if they exist")
    parser.add_argument(
        "--no-skip",
        dest="skip",
        action="store_false",
        help="Recompute files even if cache already exists (unless --overwrite).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    preprocessor = AudioPreprocessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_len=args.max_len,
    )

    cache_mel_features(
        csv_path=Path(args.csv),
        audio_dir=Path(args.audio_dir),
        cache_dir=Path(args.cache_dir),
        preprocessor=preprocessor,
        overwrite=args.overwrite,
        skip_existing=args.skip,
    )


if __name__ == "__main__":
    main()
