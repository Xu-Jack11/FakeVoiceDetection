"""多路并联特征提取模块

支持以下特征类型:
- LFCC (Linear Frequency Cepstral Coefficients)
- CQCC (Constant-Q Cepstral Coefficients)
- Phase (RPS/MGD 相位/群时延特征)
- SSL (Self-Supervised Learning embeddings from wav2vec2/WavLM/Whisper)
"""

from .lfcc import extract_lfcc
from .cqcc import extract_cqcc
from .phase import extract_phase_features
from .ssl import extract_ssl_features

__all__ = [
    "extract_lfcc",
    "extract_cqcc",
    "extract_phase_features",
    "extract_ssl_features",
]
