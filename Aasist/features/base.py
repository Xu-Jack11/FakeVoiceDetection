"""通用特征提取基类，支持离线缓存。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


class FeatureExtractor(torch.nn.Module):
    """为多分支前端提供统一的缓存与调度逻辑。"""

    def __init__(
        self,
        name: str,
        cache_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_device = device

    def extra_repr(self) -> str:
        return f"name={self.name}, cache_dir={self.cache_dir}"

    def forward(
        self,
        waveform: Tensor,
        sample_rate: int,
        cache_key: Optional[str] = None,
        training: bool = True,
        metadata: Optional[dict] = None,
    ) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError(
                f"期望输入为二维张量 (batch, time)，但得到 {waveform.shape}"
            )

        use_cache = (
            not training
            and cache_key is not None
            and self.cache_dir is not None
        )

        cache_path: Optional[Path] = None
        if use_cache:
            cache_hash = self._build_cache_hash(cache_key, sample_rate, metadata)
            cache_path = self.cache_dir / f"{cache_hash}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path, map_location=waveform.device)
                return cached

        waveform = waveform.to(self.runtime_device or waveform.device)
        features = self._extract(waveform, sample_rate=sample_rate, metadata=metadata)
        features = features.to(waveform.device)

        if use_cache and cache_path is not None:
            torch.save(features.cpu(), cache_path)

        return features

    def _build_cache_hash(
        self,
        cache_key: str,
        sample_rate: int,
        metadata: Optional[dict],
    ) -> str:
        payload = {
            "key": cache_key,
            "sr": sample_rate,
            "name": self.name,
            "meta": metadata or {},
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def _extract(
        self,
        waveform: Tensor,
        sample_rate: int,
        metadata: Optional[dict] = None,
    ) -> Tensor:
        raise NotImplementedError
