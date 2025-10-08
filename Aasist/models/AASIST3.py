"""AASIST3 - 多路并联特征的升级版 AASIST

相比原版 AASIST 的改进:
1. 支持多路并联输入: LFCC + CQCC + Phase + SSL
2. 每路有独立的前端特征提取
3. 使用 Fusion Head 进行多路融合
4. 更强的跨域泛化能力
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .AASIST import (
    GraphAttentionLayer,
    HtrgGraphAttentionLayer,
    GraphPool,
    ResidualBlock,
)
from .fusion_head import MultiPathFusionHead


class MultiPathFrontEnd(nn.Module):
    """多路并联前端
    
    为不同类型的特征提供独立的编码器
    """
    
    def __init__(
        self,
        feature_type: str,
        in_channels: int,
        out_channels: int = 64,
    ):
        """
        Args:
            feature_type: 特征类型 ("lfcc", "cqcc", "phase", "ssl")
            in_channels: 输入通道数(特征维度)
            out_channels: 输出通道数
        """
        super().__init__()
        self.feature_type = feature_type
        
        # 根据特征类型使用不同的编码器
        if feature_type in ["lfcc", "cqcc"]:
            # LFCC/CQCC: 类似 MFCC 的频谱特征，使用 CNN
            self.encoder = nn.Sequential(
                ResidualBlock(nb_filts=[in_channels, 32], first=True),
                ResidualBlock(nb_filts=[32, 32]),
                ResidualBlock(nb_filts=[32, out_channels]),
                ResidualBlock(nb_filts=[out_channels, out_channels]),
            )
        elif feature_type == "phase":
            # 相位特征: 更细粒度的结构
            self.encoder = nn.Sequential(
                ResidualBlock(nb_filts=[in_channels, 32], first=True),
                ResidualBlock(nb_filts=[32, 48]),
                ResidualBlock(nb_filts=[48, out_channels]),
                ResidualBlock(nb_filts=[out_channels, out_channels]),
            )
        elif feature_type == "ssl":
            # SSL 特征: 已经是高级表征，使用轻量级编码器
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.SELU(inplace=True),
                nn.MaxPool2d((1, 2)),
                nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SELU(inplace=True),
                nn.MaxPool2d((1, 2)),
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, channels, freq, time) 或 (batch, channels, time)
            
        Returns:
            编码后的特征 (batch, out_channels, freq', time')
        """
        # 确保输入是 4D
        if x.dim() == 3:
            x = x.unsqueeze(2)  # (batch, channels, 1, time)
        
        return self.encoder(x)


class AASIST3Branch(nn.Module):
    """AASIST3 的单个分支
    
    包含: 前端编码 -> GAT -> 图池化 -> 分类头
    """
    
    def __init__(
        self,
        feature_type: str,
        in_channels: int,
        gat_dims: List[int] = [64, 32],
        pool_ratios: List[float] = [0.5, 0.7, 0.5],
        temperatures: List[float] = [2.0, 2.0, 100.0],
        num_classes: int = 2,
    ):
        super().__init__()
        self.feature_type = feature_type
        
        # 前端编码器
        self.frontend = MultiPathFrontEnd(
            feature_type=feature_type,
            in_channels=in_channels,
            out_channels=gat_dims[0],
        )
        
        # GAT 层
        self.GAT_layer_S = GraphAttentionLayer(
            gat_dims[0], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            gat_dims[0], gat_dims[0], temperature=temperatures[1]
        )
        
        # 异构图注意力
        self.HtrgGAT_layer_ST1 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST2 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        
        # 图池化
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        # Master node
        self.master = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # 分类头
        self.drop = nn.Dropout(0.5)
        self.out_layer = nn.Linear(5 * gat_dims[1], num_classes)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: 输入特征 (batch, channels, freq, time)
            
        Returns:
            embedding: 特征嵌入 (batch, feature_dim)
            logits: 分类 logits (batch, num_classes)
        """
        # 前端编码
        e = self.frontend(x)  # (batch, channels, freq', time')
        
        # 频率维度最大池化 -> Spectral graph
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2)  # (batch, time', channels)
        
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        
        # 时间维度最大池化 -> Temporal graph
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)  # (batch, freq', channels)
        
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # 异构图融合
        master = self.master.expand(x.size(0), -1, -1)
        out_T, out_S, master = self.HtrgGAT_layer_ST1(out_T, out_S, master=self.master)
        
        out_S = self.pool_hS(out_S)
        out_T = self.pool_hT(out_T)
        
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST2(out_T, out_S, master=master)
        out_T = out_T + out_T_aug
        out_S = out_S + out_S_aug
        master = master + master_aug
        
        # 聚合
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        embedding = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        embedding = self.drop(embedding)
        
        logits = self.out_layer(embedding)
        
        return embedding, logits


class AASIST3(nn.Module):
    """AASIST3 主模型
    
    多路并联架构:
    - Branch 1: LFCC
    - Branch 2: CQCC
    - Branch 3: Phase
    - Branch 4: SSL (可选)
    """
    
    def __init__(
        self,
        feature_configs: Dict[str, int],
        gat_dims: List[int] = [64, 32],
        pool_ratios: List[float] = [0.5, 0.7, 0.5],
        temperatures: List[float] = [2.0, 2.0, 100.0],
        num_classes: int = 2,
        fusion_type: str = "mlp",
        fusion_config: Optional[Dict] = None,
    ):
        """
        Args:
            feature_configs: 特征配置 {feature_type: in_channels}
                例如: {"lfcc": 60, "cqcc": 60, "phase": 60, "ssl": 256}
            gat_dims: GAT 维度
            pool_ratios: 池化比例
            temperatures: 温度参数
            num_classes: 类别数
            fusion_type: 融合类型 ("logit", "mlp", "attention")
            fusion_config: 融合模块配置
        """
        super().__init__()
        self.feature_configs = feature_configs
        self.num_classes = num_classes
        
        # 创建各分支
        self.branches = nn.ModuleDict()
        for feature_type, in_channels in feature_configs.items():
            self.branches[feature_type] = AASIST3Branch(
                feature_type=feature_type,
                in_channels=in_channels,
                gat_dims=gat_dims,
                pool_ratios=pool_ratios,
                temperatures=temperatures,
                num_classes=num_classes,
            )
        
        # 融合模块
        self.fusion = MultiPathFusionHead(
            num_branches=len(feature_configs),
            num_classes=num_classes,
            fusion_type=fusion_type,
            fusion_config=fusion_config or {},
        )
    
    def forward(
        self,
        features: Dict[str, Tensor],
        return_embeddings: bool = False,
    ) -> Tensor | Tuple[Dict[str, Tensor], Tensor]:
        """
        Args:
            features: 各路特征 {feature_type: (batch, channels, freq, time)}
            return_embeddings: 是否返回各分支的嵌入
            
        Returns:
            如果 return_embeddings=False: 融合后的 logits (batch, num_classes)
            如果 return_embeddings=True: (embeddings_dict, fused_logits)
        """
        branch_logits = []
        branch_embeddings = {}
        
        # 各分支前向传播
        for feature_type in self.feature_configs.keys():
            if feature_type not in features:
                raise ValueError(f"Missing feature: {feature_type}")
            
            embedding, logits = self.branches[feature_type](features[feature_type])
            branch_logits.append(logits)
            branch_embeddings[feature_type] = embedding
        
        # 融合
        fused_logits = self.fusion(branch_logits)
        
        if return_embeddings:
            return branch_embeddings, fused_logits
        else:
            return fused_logits


def create_aasist3_model(
    config: Optional[Dict] = None,
) -> AASIST3:
    """
    便捷函数: 创建 AASIST3 模型
    
    Args:
        config: 模型配置
        
    Returns:
        AASIST3 模型实例
    """
    default_config = {
        "feature_configs": {
            "lfcc": 60,
            "cqcc": 60,
            "phase": 60,
        },
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0],
        "num_classes": 2,
        "fusion_type": "mlp",
        "fusion_config": {
            "hidden_dims": [128, 64],
            "dropout": 0.3,
        },
    }
    
    if config:
        default_config.update(config)
    
    return AASIST3(**default_config)
