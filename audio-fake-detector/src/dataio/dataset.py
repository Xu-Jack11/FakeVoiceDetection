"""Dataset loading utilities with optional random cropping."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils.audio import load_audio, resolve_audio_path


@dataclass
class Sample:
    """Container for a single dataset sample."""

    waveform: torch.Tensor
    length: int
    label: Optional[int]
    name: str


class AudioDeepfakeDataset(Dataset):
    """
    Dataset that reads items from a CSV with columns `audio_name` and optional `target`.
    Supports simple cropping to fixed-duration segments.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        audio_dir: Optional[Path],
        sample_rate: int,
        audio_column: str = "audio_name",
        target_column: str = "target",
        crop_enable: bool = False,
        crop_min_sec: float = 3.0,
        crop_max_sec: float = 5.0,
        crop_mode: str = "random",  # options: random, center
    ) -> None:
        self.df = data.reset_index(drop=True)
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self.sample_rate = sample_rate
        self.audio_column = audio_column
        self.target_column = target_column
        self.has_labels = target_column in self.df.columns and self.df[target_column].notna().all()

        self.crop_enable = crop_enable
        self.crop_min_sec = float(crop_min_sec)
        self.crop_max_sec = float(crop_max_sec)
        if self.crop_max_sec < self.crop_min_sec:
            raise ValueError("crop_max_sec must be >= crop_min_sec.")
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        audio_name = row[self.audio_column]
        audio_path = resolve_audio_path(audio_name, self.audio_dir)
        waveform = load_audio(audio_path, sample_rate=self.sample_rate)

        waveform, length = self._crop_waveform(waveform)

        label = int(row[self.target_column]) if self.has_labels else None
        return Sample(waveform=waveform, length=length, label=label, name=str(audio_name))

    def _crop_waveform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        if not self.crop_enable:
            return waveform, waveform.size(0)

        min_len = int(self.crop_min_sec * self.sample_rate)
        max_len = int(self.crop_max_sec * self.sample_rate)
        if max_len <= 0:
            raise ValueError("crop_max_sec results in non-positive number of samples.")

        target_len = max_len
        if self.crop_mode == "random":
            target_len = random.randint(min_len, max_len)
        elif self.crop_mode == "center":
            target_len = min(max_len, max(min_len, waveform.size(0)))
        else:
            raise ValueError(f"Unsupported crop_mode '{self.crop_mode}'.")

        if waveform.size(0) >= target_len:
            if self.crop_mode == "center":
                start = max((waveform.size(0) - target_len) // 2, 0)
            else:
                max_start = waveform.size(0) - target_len
                start = random.randint(0, max_start) if max_start > 0 else 0
            cropped = waveform[start : start + target_len]
            length = target_len
        else:
            # Pad waveform to target length.
            pad_amount = target_len - waveform.size(0)
            cropped = F.pad(waveform, (0, pad_amount), mode="constant", value=0.0)
            length = waveform.size(0)

        return cropped, length


def collate_audio_samples(batch: Iterable[Sample]) -> dict:
    """
    Collate a batch of Samples into padded tensors.
    """
    batch_list: List[Sample] = list(batch)
    if not batch_list:
        raise ValueError("Empty batch encountered during collation.")
    waveforms = [item.waveform for item in batch_list]
    lengths = torch.tensor([item.length for item in batch_list], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True)
    attention_mask = torch.zeros_like(padded, dtype=torch.bool)
    for idx, length in enumerate(lengths.tolist()):
        attention_mask[idx, :length] = True
    batch_dict = {
        "input_values": padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "audio_names": [item.name for item in batch_list],
    }
    if batch_list[0].label is not None:
        labels = torch.tensor([item.label for item in batch_list], dtype=torch.long)
        batch_dict["labels"] = labels
    return batch_dict
