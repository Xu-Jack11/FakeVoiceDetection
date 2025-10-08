# AASIST3 - 电话信道鲁棒的多路并联反伪造系统

## 概述

AASIST3 是 AASIST 模型的升级版本,专为电话信道场景下的音频反伪造任务设计。主要改进包括:

### 核心改进

1. **多路并联特征** (vs 单路 Mel 谱)
   - LFCC (Linear Frequency Cepstral Coefficients)
   - CQCC (Constant-Q Cepstral Coefficients)
   - Phase Features (RPS/MGD/IFD)
   - SSL Features (wav2vec2/WavLM/Whisper,可选)

2. **电话信道增强**
   - 编解码模拟 (AMR-NB/WB, G.711, Opus)
   - 带宽抖动 (8k ↔ 16k)
   - 带通滤波 (300-3400 Hz)
   - RawBoost 波形域扰动

3. **高级训练策略**
   - VAD 语音活动检测
   - 随机切片 (2-6s 训练)
   - Focal Loss / Class-Balanced Loss
   - 混合精度训练

4. **智能推理**
   - 滑窗推理 (3-5s, 50% 重叠)
   - 多chunk集成 (mean/top-k pooling)
   - 验证集阈值优化
   - 温度缩放校准

## 项目结构

```
FakeVoiceDetection/
├── features/                    # 多路特征提取
│   ├── __init__.py
│   ├── lfcc.py                 # LFCC 提取器
│   ├── cqcc.py                 # CQCC 提取器
│   ├── phase.py                # 相位特征提取器
│   └── ssl.py                  # SSL 特征提取器
│
├── Aasist/
│   ├── models/
│   │   ├── AASIST.py          # 原版 AASIST
│   │   ├── AASIST3.py         # 升级的 AASIST3
│   │   └── fusion_head.py     # 多路融合模块
│   │
│   ├── data/
│   │   ├── vad.py             # VAD 语音活动检测
│   │   ├── telephony_augs.py  # 电话信道增强
│   │   └── dataset_multipath.py  # 多路特征数据集
│   │
│   ├── losses.py               # 高级损失函数
│   ├── train_aasist3.py       # AASIST3 训练脚本
│   └── predict_aasist3.py     # AASIST3 推理脚本
│
└── dataset/
    ├── train/                  # 训练音频
    ├── test/                   # 测试音频
    └── train.csv               # 训练标签
```

## 依赖安装

### 核心依赖
```bash
pip install torch torchaudio
pip install numpy pandas scikit-learn
pip install soundfile librosa
pip install scipy
pip install tqdm
```

### 可选依赖
```bash
# SSL 特征 (wav2vec2/WavLM/Whisper)
pip install transformers

# WebRTC VAD (更准确的 VAD)
pip install webrtcvad

# FFmpeg 编解码模拟 (需要系统安装 ffmpeg)
# Windows: choco install ffmpeg
# Linux: sudo apt-get install ffmpeg
```

## 快速开始

### 1. 训练 AASIST3 模型

```bash
cd Aasist

# 基础训练 (LFCC + CQCC + Phase)
python -m train_aasist3 \
    --data_root ../dataset \
    --output_dir ./output_aasist3 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --feature_types lfcc cqcc phase \
    --loss_type focal \
    --use_amp

# 使用 SSL 特征 (需要更大的 GPU 内存)
python -m train_aasist3 \
    --data_root ../dataset \
    --output_dir ./output_aasist3_ssl \
    --batch_size 16 \
    --epochs 50 \
    --feature_types lfcc cqcc phase ssl \
    --use_amp
```

### 2. 推理预测

```bash
# 滑窗推理
python -m predict_aasist3 \
    --checkpoint ./output_aasist3/best_aasist3.pth \
    --test_dir ../dataset/test \
    --output_csv predictions.csv \
    --batch_size 16 \
    --window_size 4.0 \
    --hop_size 2.0 \
    --pooling mean \
    --threshold 0.5
```

## 配置说明

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 32 | 批大小 (T4 GPU 建议 16-32) |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--feature_types` | lfcc cqcc phase | 特征类型 |
| `--loss_type` | focal | 损失函数 (focal/ce/class_balanced/combined) |
| `--use_amp` | False | 是否使用混合精度训练 |
| `--num_workers` | 4 | 数据加载线程数 |

### 推理配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window_size` | 4.0 | 滑窗大小 (秒) |
| `--hop_size` | 2.0 | 跳跃大小 (秒,50% 重叠) |
| `--pooling` | mean | chunk 聚合方式 (mean/max/top-k) |
| `--threshold` | 0.5 | 分类阈值 |

## 性能优化建议

### 单卡 T4 训练优化

1. **批大小**: 16-32 (根据特征数量调整)
2. **混合精度**: 启用 `--use_amp` 可节省约 40% 显存
3. **数据加载**: `--num_workers 4-8` 充分利用 16 核 CPU
4. **特征缓存**: 启用离线缓存,训练时避免重复计算

### 离线特征缓存

```python
# 在 dataset_multipath.py 中配置
AASIST3DatasetConfig(
    feature_cache_dir=Path("./feature_cache"),  # 启用缓存
    ...
)
```

训练前可以先运行一次前向传播,缓存所有特征,后续训练会快很多。

### CPU/GPU 效率

- **特征提取**: LFCC/CQCC/Phase 在 CPU 上高效
- **SSL 特征**: 需要 GPU,建议预先提取并缓存
- **数据增强**: TelephonyAugmentation 在 CPU 上运行
- **模型训练**: 在 GPU 上进行

## 关键特性详解

### 1. 多路并联特征

不同特征捕捉音频的不同方面:

- **LFCC**: 线性频率尺度,对低频细节敏感
- **CQCC**: 对数频率尺度,对压缩伪迹敏感
- **Phase**: 相位信息,捕捉合成伪迹
- **SSL**: 高级语义表征,泛化能力强

### 2. 电话信道增强

模拟真实电话场景的失真:

```python
from Aasist.data.telephony_augs import TelephonyAugmentation

aug = TelephonyAugmentation(
    sample_rate=16000,
    p_codec=0.5,        # 编解码概率
    p_bandwidth=0.3,    # 带宽抖动概率
    p_bandpass=0.6,     # 带通滤波概率
    p_rawboost=0.7,     # RawBoost 概率
)

augmented_waveform = aug(waveform)
```

### 3. VAD 语音活动检测

去除静音段,提高训练效果:

```python
from Aasist.data.vad import apply_vad

speech_waveform = apply_vad(
    waveform,
    sample_rate=16000,
    vad_type="energy",  # "energy" 或 "webrtc"
    min_speech_ratio=0.5,
)
```

### 4. 滑窗推理

长音频切分为多个 chunk,提高鲁棒性:

```
音频: |----------60s----------|
窗口:  |--4s--|
         |--4s--|
           |--4s--|
             ...
             
最终预测 = mean(chunk_1, chunk_2, ..., chunk_N)
```

## 实验结果 (预期)

| 模型 | 特征 | Val F1 | 测试集 AUC |
|------|------|--------|------------|
| AASIST (baseline) | Mel | 0.85 | 0.90 |
| AASIST3 | LFCC + CQCC | 0.88 | 0.92 |
| AASIST3 | LFCC + CQCC + Phase | 0.90 | 0.94 |
| AASIST3 + SSL | 全部特征 | 0.92 | 0.95 |

*注: 以上为估算值,实际效果取决于数据集*

## 常见问题

### Q: 训练很慢怎么办?

A: 
1. 启用混合精度训练 `--use_amp`
2. 启用特征缓存
3. 减少特征数量 (只用 LFCC + CQCC)
4. 增加 `num_workers`

### Q: 显存不足怎么办?

A:
1. 减小批大小 `--batch_size 16`
2. 启用混合精度 `--use_amp`
3. 减少特征数量
4. 减小模型参数 (修改 `gat_dims`)

### Q: 如何调整阈值?

A: 训练脚本会自动在验证集上寻找最优阈值 (F1 最大)。
推理时使用训练时找到的 `best_threshold`。

### Q: 如何添加自定义特征?

A: 在 `features/` 下创建新的特征提取器,并在 `dataset_multipath.py` 中集成。

## 引用

如果您使用本代码,请引用:

```bibtex
@inproceedings{jung2021aasist,
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  author={Jung, Jee-weon and Heo, Hee-soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon son and Lee, Bong-jin and Yu, Ha-jin and Evans, Nicholas},
  booktitle={ICASSP 2022},
  year={2022}
}
```

## License

MIT License

## 联系方式

如有问题,请提 issue 或联系维护者。

---

**Good Luck!** 🚀
