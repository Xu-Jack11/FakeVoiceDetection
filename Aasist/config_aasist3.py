"""AASIST3 配置文件

集中管理所有超参数和配置
"""

from pathlib import Path
from typing import Dict, List

# ==================== 数据配置 ====================
DATA_CONFIG = {
    "data_root": Path("./dataset"),
    "sample_rate": 16000,
    "val_split": 0.2,
    "seed": 42,
}

# ==================== 特征配置 ====================
FEATURE_CONFIG = {
    # 特征类型: ["lfcc", "cqcc", "phase", "ssl"]
    "feature_types": ["lfcc", "cqcc", "phase"],
    
    # 各特征维度
    "feature_dims": {
        "lfcc": 60,
        "cqcc": 60,
        "phase": 60,
        "ssl": 256,  # SSL 特征维度更高
    },
    
    # 特征缓存
    "feature_cache_dir": Path("./feature_cache"),
    "use_cache": True,
    
    # 切片配置
    "min_chunk_duration": 2.0,  # 训练时最小切片(秒)
    "max_chunk_duration": 6.0,  # 训练时最大切片(秒)
    "fixed_chunk_duration": 4.0,  # 推理时固定长度(秒)
}

# ==================== VAD 配置 ====================
VAD_CONFIG = {
    "use_vad": True,
    "vad_type": "energy",  # "energy" or "webrtc"
    "min_speech_ratio": 0.5,
    "energy_threshold": 0.02,
    "min_speech_duration": 0.5,
}

# ==================== 数据增强配置 ====================
AUGMENTATION_CONFIG = {
    "use_augmentation": True,
    "aug_prob": 0.8,
    
    # 电话信道增强概率
    "p_codec": 0.5,
    "p_bandwidth": 0.3,
    "p_bandpass": 0.6,
    "p_compand": 0.4,
    "p_rawboost": 0.7,
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    # AASIST3 架构参数
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0],
    "num_classes": 2,
    
    # 融合配置
    "fusion_type": "mlp",  # "logit", "mlp", "attention"
    "fusion_config": {
        "hidden_dims": [128, 64],
        "dropout": 0.3,
    },
}

# ==================== 训练配置 ====================
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    
    # 学习率调度
    "scheduler": "cosine",  # "cosine", "step", "plateau"
    "scheduler_params": {
        "T_max": 50,
        "eta_min": 1e-6,
    },
    
    # 混合精度训练
    "use_amp": True,
    
    # 早停
    "early_stopping": True,
    "patience": 10,
}

# ==================== 损失函数配置 ====================
LOSS_CONFIG = {
    "loss_type": "focal",  # "focal", "ce", "class_balanced", "combined"
    
    # Focal Loss 参数
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    
    # Class-Balanced Loss 参数
    "cb_beta": 0.9999,
    
    # 组合损失权重
    "combined_weights": {
        "focal": 0.6,
        "ce": 0.2,
        "label_smoothing": 0.2,
    },
}

# ==================== 推理配置 ====================
INFERENCE_CONFIG = {
    "batch_size": 16,
    
    # 滑窗参数
    "window_size": 4.0,  # 秒
    "hop_size": 2.0,  # 秒 (50% 重叠)
    
    # 聚合方式
    "pooling": "mean",  # "mean", "max", "top-k"
    
    # 阈值
    "threshold": 0.5,  # 将被训练时优化的阈值覆盖
    
    # 温度缩放
    "use_temperature_scaling": False,
}

# ==================== 输出配置 ====================
OUTPUT_CONFIG = {
    "output_dir": Path("./output_aasist3"),
    "save_checkpoint_every": 5,  # 每 N 个 epoch 保存一次
    "save_best_only": True,
    "log_interval": 10,  # 每 N 个 batch 打印一次
}

# ==================== 便捷函数 ====================

def get_train_config() -> Dict:
    """获取训练配置"""
    return {
        **DATA_CONFIG,
        **FEATURE_CONFIG,
        **VAD_CONFIG,
        **AUGMENTATION_CONFIG,
        **MODEL_CONFIG,
        **TRAINING_CONFIG,
        **LOSS_CONFIG,
        **OUTPUT_CONFIG,
    }


def get_inference_config() -> Dict:
    """获取推理配置"""
    return {
        **DATA_CONFIG,
        **FEATURE_CONFIG,
        **MODEL_CONFIG,
        **INFERENCE_CONFIG,
        **OUTPUT_CONFIG,
    }


def print_config(config: Dict, title: str = "Configuration"):
    """打印配置"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}\n")
    
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n{'='*50}\n")


# ==================== T4 优化配置 ====================
T4_OPTIMIZED_CONFIG = {
    "batch_size": 24,  # T4 16GB 显存
    "use_amp": True,  # 必须开启
    "num_workers": 8,  # 16核 CPU
    "feature_types": ["lfcc", "cqcc", "phase"],  # 不用 SSL
    "gradient_accumulation_steps": 2,  # 梯度累积
}


def get_t4_config() -> Dict:
    """获取 T4 优化配置"""
    config = get_train_config()
    config.update(T4_OPTIMIZED_CONFIG)
    return config
