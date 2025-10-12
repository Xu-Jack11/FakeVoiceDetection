"""Audio loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Tuple

import torch
import torchaudio

_RESAMPLERS: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}


def resolve_audio_path(
    audio_name: Union[str, Path],
    root_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve an audio file path given a potential root directory.

    Parameters
    ----------
    audio_name:
        Path or string from the CSV.
    root_dir:
        Optional directory that stores the audio files.

    Returns
    -------
    Path
        Resolved path that must exist on disk.
    """
    audio_path = Path(audio_name)
    if not audio_path.is_absolute():
        base = Path(root_dir) if root_dir is not None else Path(".")
        audio_path = base / audio_path
    audio_path = audio_path.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def _get_resampler(orig_sr: int, target_sr: int) -> torchaudio.transforms.Resample:
    key = (orig_sr, target_sr)
    resampler = _RESAMPLERS.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        _RESAMPLERS[key] = resampler
    return resampler


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load an audio file, convert to mono, resample, and normalise.

    Returns
    -------
    torch.Tensor
        Float tensor with shape (time,) and values in [-1, 1].
    """
    load_fn = getattr(torchaudio, "load_with_torchcodec", torchaudio.load)
    waveform, sr = load_fn(str(audio_path))
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = _get_resampler(sr, sample_rate)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0).float()
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform
