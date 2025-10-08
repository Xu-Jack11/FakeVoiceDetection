"""高级损失函数

支持:
1. FocalLoss: 解决类别不均衡
2. Class-Balanced Loss: 基于有效样本数的权重
3. AAM-Softmax (Additive Angular Margin): 增强域泛化
4. Label Smoothing: 防止过拟合
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss
    
    论文: Focal Loss for Dense Object Detection (Lin et al., 2017)
    解决类别不均衡问题,降低易分类样本的权重
    """
    
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: 类别权重,None 表示不使用
            gamma: 聚焦参数,越大越关注难样本
            reduction: "mean" or "sum" or "none"
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: 预测 logits (batch, num_classes)
            targets: 真实标签 (batch,) 或 (batch, num_classes)
            
        Returns:
            loss: 标量或 (batch,) 根据 reduction
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # 计算概率
        p = torch.exp(-ce_loss)
        
        # Focal loss = (1 - p)^gamma * CE
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        # 类别权重
        if self.alpha is not None:
            if targets.dim() == 1:
                # 类别索引
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            else:
                # One-hot 编码
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss
    
    论文: Class-Balanced Loss Based on Effective Number of Samples (Cui et al., 2019)
    根据有效样本数计算类别权重
    """
    
    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = "focal",
    ):
        """
        Args:
            samples_per_class: 每个类别的样本数 [N_class_0, N_class_1, ...]
            beta: 平衡参数,0.9999 for 长尾分布
            gamma: Focal loss 参数(如果使用)
            loss_type: "focal" or "ce"
        """
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # 计算有效样本数
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)
        
        self.register_buffer("weights", weights)
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: 预测 logits (batch, num_classes)
            targets: 真实标签 (batch,)
            
        Returns:
            loss: 标量
        """
        # 获取类别权重
        weights = self.weights.to(inputs.device)
        
        if self.loss_type == "focal":
            # Focal loss with class weights
            ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction="none")
            p = torch.exp(-ce_loss)
            focal_loss = (1 - p) ** self.gamma * ce_loss
            return focal_loss.mean()
        else:
            # Weighted CE loss
            return F.cross_entropy(inputs, targets, weight=weights)


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (AAM-Softmax / ArcFace)
    
    论文: ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., 2019)
    增强特征可分性,提高域泛化能力
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
            margin: 角度 margin (弧度)
            scale: 缩放因子
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # 权重矩阵 (num_classes, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            features: 特征嵌入 (batch, in_features)
            targets: 真实标签 (batch,)
            
        Returns:
            loss: AAM-Softmax loss
        """
        # L2 归一化
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(features, weight)  # (batch, num_classes)
        
        # 计算角度
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # 添加 margin
        target_theta = theta.clone()
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        target_theta = target_theta + one_hot * self.margin
        
        # 转回余弦
        target_cosine = torch.cos(target_theta)
        
        # 缩放并计算 loss
        logits = target_cosine * self.scale
        loss = F.cross_entropy(logits, targets)
        
        return loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss
    
    防止模型过度自信,提高泛化能力
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
    ):
        """
        Args:
            num_classes: 类别数
            smoothing: 平滑系数 (0.0 - 1.0)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: 预测 logits (batch, num_classes)
            targets: 真实标签 (batch,)
            
        Returns:
            loss: 标量
        """
        # Log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot 编码
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Label smoothing
        smooth_targets = targets_one_hot * self.confidence + self.smoothing / self.num_classes
        
        # 计算 KL 散度
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """组合损失函数
    
    支持多种损失的加权组合
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        loss_config: Optional[dict] = None,
    ):
        """
        Args:
            num_classes: 类别数
            loss_config: 损失配置
                {
                    "focal": {"weight": 0.5, "alpha": 0.25, "gamma": 2.0},
                    "ce": {"weight": 0.3},
                    "label_smoothing": {"weight": 0.2, "smoothing": 0.1},
                }
        """
        super().__init__()
        self.num_classes = num_classes
        
        if loss_config is None:
            # 默认配置: Focal Loss
            loss_config = {
                "focal": {"weight": 1.0, "alpha": 0.25, "gamma": 2.0}
            }
        
        self.loss_config = loss_config
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        # 创建损失函数
        for loss_name, config in loss_config.items():
            weight = config.pop("weight", 1.0)
            self.weights[loss_name] = weight
            
            if loss_name == "focal":
                self.losses[loss_name] = FocalLoss(**config)
            elif loss_name == "ce":
                self.losses[loss_name] = nn.CrossEntropyLoss(**config)
            elif loss_name == "label_smoothing":
                self.losses[loss_name] = LabelSmoothingLoss(num_classes, **config)
            elif loss_name == "class_balanced":
                self.losses[loss_name] = ClassBalancedLoss(**config)
            else:
                raise ValueError(f"Unknown loss: {loss_name}")
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: 预测 logits (batch, num_classes)
            targets: 真实标签 (batch,)
            
        Returns:
            total_loss: 组合损失
        """
        total_loss = 0.0
        
        for loss_name, loss_fn in self.losses.items():
            loss_val = loss_fn(inputs, targets)
            total_loss += self.weights[loss_name] * loss_val
        
        return total_loss


def create_loss_function(
    loss_type: str = "focal",
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """
    便捷函数: 创建损失函数
    
    Args:
        loss_type: 损失类型 ("focal", "ce", "class_balanced", "aam", "combined")
        num_classes: 类别数
        **kwargs: 损失函数参数
        
    Returns:
        损失函数
    """
    if loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "class_balanced":
        return ClassBalancedLoss(**kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(num_classes, **kwargs)
    elif loss_type == "combined":
        return CombinedLoss(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
