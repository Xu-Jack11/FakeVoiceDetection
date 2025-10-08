"""AASIST3 模块初始化"""

# 模型
from .models.AASIST3 import AASIST3, create_aasist3_model
from .models.fusion_head import MultiPathFusionHead

# 数据集
from .data.dataset_multipath import (
    AASIST3Dataset,
    AASIST3DatasetConfig,
    collate_fn_multipath,
)

# 增强和预处理
from .data.vad import EnergyVAD, WebRTCVAD, apply_vad
from .data.telephony_augs import TelephonyAugmentation

# 损失函数
from .losses import (
    FocalLoss,
    ClassBalancedLoss,
    AAMSoftmax,
    LabelSmoothingLoss,
    CombinedLoss,
    create_loss_function,
)

# 配置
from .config_aasist3 import (
    get_train_config,
    get_inference_config,
    get_t4_config,
)

__version__ = "3.0.0"

__all__ = [
    # 模型
    "AASIST3",
    "create_aasist3_model",
    "MultiPathFusionHead",
    # 数据集
    "AASIST3Dataset",
    "AASIST3DatasetConfig",
    "collate_fn_multipath",
    # 增强
    "EnergyVAD",
    "WebRTCVAD",
    "apply_vad",
    "TelephonyAugmentation",
    # 损失
    "FocalLoss",
    "ClassBalancedLoss",
    "AAMSoftmax",
    "LabelSmoothingLoss",
    "CombinedLoss",
    "create_loss_function",
    # 配置
    "get_train_config",
    "get_inference_config",
    "get_t4_config",
]
