"""特征提取前端模块集合。"""

from __future__ import annotations

from .lfcc import LFCCFrontend
from .cqcc import CQCCFrontend
from .phase import PhaseFrontend
from .ssl import SSLFrontend

__all__ = [
    "LFCCFrontend",
    "CQCCFrontend",
    "PhaseFrontend",
    "SSLFrontend",
]
