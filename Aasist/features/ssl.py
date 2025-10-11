"""自监督语音模型隐藏层特征抽取。"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import FeatureExtractor

try:  # pragma: no cover
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None


_SSL_BUNDLES: dict[str, object] = {}
if torchaudio is not None:
    _SSL_BUNDLES = {
        "wav2vec2_base": torchaudio.pipelines.WAV2VEC2_BASE,
        "wav2vec2_large": torchaudio.pipelines.WAV2VEC2_LARGE,
        "wavlm_base": getattr(torchaudio.pipelines, "WAVLM_BASE", None),
    }


class SSLFrontend(FeatureExtractor):
    def __init__(
        self,
        model_name: str = "wav2vec2_base",
        layer: int = -1,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(f"ssl_{model_name}", cache_dir=cache_dir, device=device)
        if torchaudio is None:
            raise ImportError("torchaudio 未安装，无法启用 SSL 前端")
        if model_name not in _SSL_BUNDLES or _SSL_BUNDLES[model_name] is None:
            raise ValueError(f"未支持的 SSL 模型: {model_name}")
        bundle = _SSL_BUNDLES[model_name]
        self.ssl_model = bundle.get_model().to(device or torch.device("cpu"))
        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad_(False)
        self.layer = layer
        self.target_sample_rate = int(bundle.sample_rate)

    def _extract(
        self,
        waveform: Tensor,
        sample_rate: int,
        metadata: Optional[dict] = None,
    ) -> Tensor:
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.target_sample_rate,
            )
        with torch.no_grad():
            features, _ = self.ssl_model.extract_features(waveform)
        chosen_layer = features[self.layer]
        return chosen_layer
