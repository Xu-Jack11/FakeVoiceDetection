"""AASIST 通用配置常量。"""

from __future__ import annotations

from typing import Dict

DEFAULT_MODEL_CONFIG: Dict[str, object] = {
    "architecture": "AASIST3",
    "nb_samp": 64600,
    "sample_rate": 16000,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0],
    "feature_cache_root": "Aasist/cache",
    "ssl_model": "wav2vec2_base",
    "fusion_hidden": (192, 96),
    "fusion_dropout": 0.1,
    "inference_chunk_seconds": 4.0,
    "inference_hop_ratio": 0.5,
    "inference_topk_ratio": 0.5,
    "enable_lfcc": True,
    "enable_cqcc": True,
    "enable_phase": True,
    "enable_ssl": True,
    "enable_aasist": True,
}
