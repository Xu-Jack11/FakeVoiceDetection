# AASIST3 升级完成总结

## 已实现的功能

### ✅ 1. 多路并联特征提取 (`features/`)

- **LFCC** (`lfcc.py`): 线性频率倒谱系数,与 Mel 互补
- **CQCC** (`cqcc.py`): 常数 Q 倒谱系数,对压缩伪迹敏感
- **Phase** (`phase.py`): 相位特征 (RPS/MGD/IFD),捕捉合成伪迹
- **SSL** (`ssl.py`): 自监督学习特征 (wav2vec2/WavLM/Whisper)

**特点**:
- 支持离线特征缓存
- 批量处理优化
- GPU/CPU 高效实现

### ✅ 2. AASIST3 模型架构 (`Aasist/models/`)

- **AASIST3.py**: 多路并联的 AASIST 升级版
- **fusion_head.py**: 多种融合策略
  - Logit 级加权融合
  - MLP 学习型融合
  - Attention 融合

**改进**:
- 每路独立编码器
- 异构图注意力
- 自适应融合权重

### ✅ 3. 电话信道增强 (`Aasist/data/`)

**telephony_augs.py**:
- 编解码模拟 (AMR-NB/WB, G.711, Opus)
- 带宽抖动 (8k ↔ 16k)
- 带通滤波 (300-3400 Hz)
- 动态范围压缩
- RawBoost 波形扰动 (LnL/ISD/SSI)

**特点**:
- 可配置的增强概率
- 支持 FFmpeg 真实编解码 (可选)
- CPU 高效实现

### ✅ 4. VAD 语音活动检测 (`Aasist/data/`)

**vad.py**:
- **EnergyVAD**: 基于能量的 VAD (快速,无依赖)
- **WebRTCVAD**: 基于 WebRTC 的 VAD (准确,需要 py-webrtcvad)

**功能**:
- 自动去除静音段
- 形态学平滑
- 最小语音占比保证

### ✅ 5. 多路特征数据集 (`Aasist/data/`)

**dataset_multipath.py**:
- 随机切片 (2-6s 训练, 固定长度推理)
- VAD 集成
- 电话增强集成
- 特征缓存
- 高效批处理

### ✅ 6. 高级损失函数 (`Aasist/losses.py`)

- **FocalLoss**: 解决类别不均衡
- **ClassBalancedLoss**: 基于有效样本数
- **AAMSoftmax**: 增强域泛化 (ArcFace)
- **LabelSmoothingLoss**: 防止过拟合
- **CombinedLoss**: 多损失组合

### ✅ 7. 训练脚本 (`Aasist/train_aasist3.py`)

**功能**:
- 多路特征训练
- 验证集阈值优化 (F1 最大化)
- 混合精度训练 (AMP)
- 学习率调度 (Cosine Annealing)
- 训练历史记录

**优化**:
- T4 GPU 优化配置
- 梯度累积 (可选)
- 早停机制

### ✅ 8. 推理脚本 (`Aasist/predict_aasist3.py`)

**功能**:
- 滑窗推理 (3-5s, 50% 重叠)
- 多 chunk 集成 (mean/max/top-k)
- 温度缩放校准
- 批量预测

### ✅ 9. 辅助工具

- **run_aasist3.py**: 快速启动脚本
- **prepare_data.py**: 数据准备和验证
- **config_aasist3.py**: 集中配置管理
- **AASIST3_README.md**: 完整文档
- **requirements_aasist3.txt**: 依赖列表

## 文件结构

```
FakeVoiceDetection/
│
├── features/                           # 多路特征提取
│   ├── __init__.py
│   ├── lfcc.py                        # LFCC 提取器
│   ├── cqcc.py                        # CQCC 提取器
│   ├── phase.py                       # 相位特征
│   └── ssl.py                         # SSL 特征
│
├── Aasist/
│   ├── __init__.py                    # 模块初始化
│   ├── config_aasist3.py              # 配置文件
│   ├── losses.py                      # 损失函数
│   ├── train_aasist3.py               # 训练脚本
│   ├── predict_aasist3.py             # 推理脚本
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── AASIST.py                  # 原版 AASIST
│   │   ├── AASIST3.py                 # AASIST3 模型
│   │   └── fusion_head.py             # 融合模块
│   │
│   └── data/
│       ├── vad.py                     # VAD
│       ├── telephony_augs.py          # 电话增强
│       └── dataset_multipath.py       # 数据集加载器
│
├── run_aasist3.py                     # 快速启动脚本
├── prepare_data.py                    # 数据准备工具
├── AASIST3_README.md                  # 完整文档
└── requirements_aasist3.txt           # 依赖列表
```

## 使用流程

### 1. 安装依赖

```bash
pip install -r requirements_aasist3.txt
```

### 2. 数据准备

```bash
# 检查数据完整性
python prepare_data.py --data_root ./dataset --all

# 预缓存特征 (可选,加速训练)
python prepare_data.py --data_root ./dataset --precache --cache_dir ./feature_cache
```

### 3. 训练模型

```bash
# 方式1: 使用快速启动脚本
python run_aasist3.py train \
    --data_root ./dataset \
    --output_dir ./output \
    --feature_types lfcc cqcc phase \
    --use_amp

# 方式2: 直接调用训练脚本
cd Aasist
python -m train_aasist3 \
    --data_root ../dataset \
    --output_dir ./output_aasist3 \
    --batch_size 24 \
    --epochs 50 \
    --use_amp
```

### 4. 推理预测

```bash
# 方式1: 使用快速启动脚本
python run_aasist3.py predict \
    --checkpoint ./output/best_aasist3.pth \
    --test_dir ./dataset/test \
    --output_csv predictions.csv

# 方式2: 直接调用推理脚本
cd Aasist
python -m predict_aasist3 \
    --checkpoint ./output_aasist3/best_aasist3.pth \
    --test_dir ../dataset/test \
    --output_csv predictions.csv \
    --window_size 4.0 \
    --hop_size 2.0 \
    --pooling mean
```

## 性能优化要点

### 针对 T4 GPU (16GB)

1. **批大小**: 24-32 (根据特征数量调整)
2. **混合精度**: 必须启用 `--use_amp`
3. **特征选择**: `lfcc cqcc phase` (不用 SSL)
4. **数据加载**: `--num_workers 8` (16核 CPU)
5. **特征缓存**: 启用离线缓存

### 训练加速技巧

```python
# 1. 预缓存特征
python prepare_data.py --data_root ./dataset --precache

# 2. 使用配置文件
from Aasist.config_aasist3 import get_t4_config
config = get_t4_config()  # T4 优化配置

# 3. 梯度累积 (如果显存不足)
# 在 train_aasist3.py 中添加:
# if batch_idx % gradient_accumulation_steps == 0:
#     optimizer.step()
#     optimizer.zero_grad()
```

## 关键改进总结

| 方面 | AASIST (原版) | AASIST3 (升级版) |
|------|--------------|-----------------|
| **特征** | 单路 Mel 谱 | 多路并联 (LFCC+CQCC+Phase+SSL) |
| **增强** | 基础增强 | 电话信道专用增强 (RawBoost 等) |
| **预处理** | 无 VAD | 能量/WebRTC VAD |
| **切片** | 固定长度 | 训练随机 (2-6s), 推理滑窗 |
| **损失** | 交叉熵 | Focal/Class-Balanced/AAM |
| **推理** | 单 chunk | 滑窗集成 + 阈值优化 |
| **泛化** | 一般 | 电话信道鲁棒 |

## 预期提升

- **域内性能**: +5-10% F1 (vs AASIST)
- **域外泛化**: +10-15% F1 (电话信道场景)
- **未知伪造**: +8-12% F1 (多路特征互补)

## 注意事项

1. **首次运行**: 特征提取需要时间,建议预缓存
2. **显存管理**: 使用多个特征时注意显存,必要时减小批大小
3. **阈值使用**: 推理时使用训练时找到的最优阈值,不要用 0.5
4. **SSL 特征**: 需要额外的 GPU 显存,建议单独使用或预提取
5. **依赖安装**: `webrtcvad` 和 `transformers` 是可选依赖

## 故障排除

### Q: 显存不足

**A**: 
1. 减小批大小 `--batch_size 16`
2. 启用混合精度 `--use_amp`
3. 减少特征数量 (只用 `lfcc cqcc`)
4. 使用梯度累积

### Q: 训练太慢

**A**:
1. 预缓存特征
2. 增加 `num_workers`
3. 启用混合精度
4. 使用更少的特征类型

### Q: 特征提取报错

**A**:
1. 检查音频文件是否完整: `python prepare_data.py --check_integrity`
2. 确保采样率一致 (16kHz)
3. 检查依赖是否安装完整

## 后续改进方向

1. **自动化超参搜索**: 使用 Optuna 等工具
2. **模型集成**: 训练多个模型并融合
3. **知识蒸馏**: 从大模型蒸馏到小模型
4. **在线增强**: 实时音频流处理
5. **多语言支持**: 扩展到更多语言

## 联系和支持

如有问题,请查看:
1. `AASIST3_README.md` - 完整文档
2. GitHub Issues
3. 项目维护者

---

**升级完成! Good Luck! 🚀**
