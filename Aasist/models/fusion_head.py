"""多路特征融合模块

实现多种融合策略:
1. Logit-level 加权融合
2. 学习型 MLP 融合
3. Attention-based 融合
"""

from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitFusion(nn.Module):
    """Logit 级加权融合"""
    
    def __init__(
        self,
        num_branches: int,
        num_classes: int = 2,
        learnable_weights: bool = True,
    ):
        """
        Args:
            num_branches: 分支数量
            num_classes: 类别数
            learnable_weights: 是否学习权重
        """
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        
        if learnable_weights:
            # 可学习的权重参数
            self.weights = nn.Parameter(torch.ones(num_branches) / num_branches)
        else:
            # 固定权重
            self.register_buffer("weights", torch.ones(num_branches) / num_branches)
    
    def forward(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            logits_list: 各分支的 logits [(batch, num_classes), ...]
            
        Returns:
            融合后的 logits (batch, num_classes)
        """
        # Softmax 归一化权重
        weights = F.softmax(self.weights, dim=0)
        
        # 加权求和
        fused_logits = sum(w * logits for w, logits in zip(weights, logits_list))
        
        return fused_logits


class MLPFusion(nn.Module):
    """基于 MLP 的学习型融合"""
    
    def __init__(
        self,
        num_branches: int,
        num_classes: int = 2,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        """
        Args:
            num_branches: 分支数量
            num_classes: 类别数
            hidden_dims: MLP 隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        
        # 构建 MLP
        input_dim = num_branches * num_classes
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            logits_list: 各分支的 logits [(batch, num_classes), ...]
            
        Returns:
            融合后的 logits (batch, num_classes)
        """
        # 拼接所有 logits
        concat_logits = torch.cat(logits_list, dim=1)  # (batch, num_branches * num_classes)
        
        # MLP 融合
        fused_logits = self.mlp(concat_logits)
        
        return fused_logits


class AttentionFusion(nn.Module):
    """基于注意力的融合"""
    
    def __init__(
        self,
        num_branches: int,
        num_classes: int = 2,
        attention_dim: int = 64,
    ):
        """
        Args:
            num_branches: 分支数量
            num_classes: 类别数
            attention_dim: 注意力维度
        """
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        
        # 注意力网络
        self.attention = nn.Sequential(
            nn.Linear(num_classes, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
    
    def forward(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            logits_list: 各分支的 logits [(batch, num_classes), ...]
            
        Returns:
            融合后的 logits (batch, num_classes)
        """
        # 堆叠所有 logits
        stacked_logits = torch.stack(logits_list, dim=1)  # (batch, num_branches, num_classes)
        
        # 计算注意力权重
        attention_scores = self.attention(stacked_logits)  # (batch, num_branches, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, num_branches, 1)
        
        # 加权求和
        fused_logits = (stacked_logits * attention_weights).sum(dim=1)  # (batch, num_classes)
        
        return fused_logits


class MultiPathFusionHead(nn.Module):
    """多路特征融合头
    
    支持多种融合策略,适配 AASIST3 的多路并联架构
    """
    
    def __init__(
        self,
        num_branches: int,
        num_classes: int = 2,
        fusion_type: Literal["logit", "mlp", "attention"] = "mlp",
        fusion_config: Optional[dict] = None,
    ):
        """
        Args:
            num_branches: 分支数量 (例如: LFCC + CQCC + Phase + SSL = 4)
            num_classes: 类别数
            fusion_type: 融合类型 ("logit", "mlp", "attention")
            fusion_config: 融合模块的配置参数
        """
        super().__init__()
        self.num_branches = num_branches
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        if fusion_config is None:
            fusion_config = {}
        
        # 创建融合模块
        if fusion_type == "logit":
            self.fusion = LogitFusion(num_branches, num_classes, **fusion_config)
        elif fusion_type == "mlp":
            self.fusion = MLPFusion(num_branches, num_classes, **fusion_config)
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(num_branches, num_classes, **fusion_config)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            branch_outputs: 各分支的输出 logits [(batch, num_classes), ...]
            
        Returns:
            融合后的 logits (batch, num_classes)
        """
        if len(branch_outputs) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} branches, got {len(branch_outputs)}"
            )
        
        return self.fusion(branch_outputs)


class FeatureEmbeddingFusion(nn.Module):
    """特征嵌入级融合
    
    在特征嵌入层面进行融合,而不是在 logit 层面
    适合不同模态特征的深度融合
    """
    
    def __init__(
        self,
        input_dims: List[int],
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dims: 各分支的特征维度
            fusion_dim: 融合后的特征维度
            num_classes: 类别数
            dropout: Dropout 比例
        """
        super().__init__()
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        
        # 各分支的投影层
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for dim in input_dims
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(input_dims), fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_dim),
            nn.Linear(fusion_dim, num_classes),
        )
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: 各分支的特征嵌入 [(batch, dim), ...]
            
        Returns:
            分类 logits (batch, num_classes)
        """
        # 投影各分支特征
        projected = [proj(feat) for proj, feat in zip(self.projections, features_list)]
        
        # 拼接
        concat_features = torch.cat(projected, dim=1)
        
        # 融合
        logits = self.fusion(concat_features)
        
        return logits
