"""AASIST 通用配置常量。"""

from __future__ import annotations

from typing import Dict

DEFAULT_MODEL_CONFIG: Dict[str, object] = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0],
}
