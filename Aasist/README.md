# AASIST3 集成说明

本目录在 [clovaai/aasist](https://github.com/clovaai/aasist) 的基础上进行了全面升级，默认启用多分支的 **AASIST3** 结构：

- **幅度支路**：联合 LFCC、CQCC 倒谱特征；
- **相位支路**：引入 RPS/MGD 等相位/群时延提示；
- **SSL 支路**：抽取 wav2vec2/WavLM 等自监督模型隐藏层；
- **融合头**：`models/fusion_head.py` 中 2–3 层 MLP 做 logit 学习式融合；
- **数据增强**：电话链路编解码、带宽抖动、动态压缩 + RawBoost；
- **流程优化**：能量门限 VAD、滑窗集成推理、自适应阈值与温度缩放。

> ✅ **与旧版命令兼容**：训练/推理入口和参数与旧版保持一致，只是默认多了更强的特征与鲁棒性模块。

## 目录结构

- `models/aasist3.py`：AASIST3 主体，联合各分支并做融合。
- `models/fusion_head.py`：logit 级融合小型 MLP。
- `features/`：幅度（LFCC/CQCC）、相位、SSL 前端，支持离线缓存。
- `data/telephony_augs.py`：电话链路 + RawBoost 波形增广。
- `dataset.py`：VAD、随机切片、增广与数据缓存逻辑。
- `train.py`：训练脚本，支持多种损失与分支辅助监督。
- `predict.py`：滑窗推理与阈值集成。
- `config.py`：默认超参数与推理策略。
- `README.md`：当前使用说明。

## 数据准备

> ⚠️ **请勿运行原仓库的 `download_dataset.py`，直接使用本项目的 `dataset/`。**

- 训练数据路径：`dataset/train.csv` + `dataset/train/*.wav`
- 测试数据路径：`dataset/test.csv` + `dataset/test/*.wav`

CSV 需包含：

- `audio_name`：WAV 文件名；
- `target`：0=真人，1=伪造（仅训练/验证需要）。

## 环境依赖

```text
python >= 3.9
pytorch >= 1.13
torchaudio >= 0.13  # 电话链路增广 & SSL 前端
soundfile
scikit-learn
numpy
tqdm
```

安装示例：

```bash
pip install torch torchaudio soundfile scikit-learn
```

> ℹ️ **torchaudio 为必选项**：电话链路模拟、RawBoost 与 SSL 特征全部依赖 torchaudio；缺失时脚本会抛出显式错误。

## 快速上手

### 训练示例

```bash
python -m Aasist.train \
  --data-root dataset \
  --epochs 5 \
  --batch-size 16 \
  --val-split 0.1 \
  --output best_aasist3.pth \
  --predict-output aasist3_prediction.csv
```

常用可选参数：

- `--config`: 传入 JSON 覆盖默认模型/推理配置。
- `--loss`: `cross_entropy`、`focal`、`class_balanced`、`margin`（AAM-Softmax）等。
- `--branch-loss-weight`: 分支辅助监督权重（默认 0.2，设为 0 关闭）。
- `--predict-output` / `--predict-batch-size`: 控制训练后自动推理。
- `--no-auto-predict`: 仅训练不推理。

训练阶段的关键变化：

- Loader 先做能量门限 VAD，确保切片中 ≥0.8 s 语音；
- 随机裁 2–6 s 片段训练，推理固定 3–5 s 滑窗 + 50% 重叠；
- 训练集默认启用电话链路/RawBoost 增广，可通过配置关闭；
- 训练完在验证集上搜索 F1 最优阈值并记录温度缩放，写入 checkpoint。

### 独立推理

```bash
python -m Aasist.predict \
  --checkpoint Aasist/best_aasist3.pth \
  --test-csv dataset/test.csv \
  --audio-dir dataset/test \
  --output aasist3_predictions.csv
```

推理脚本会：

1. 加载 checkpoint 中保存的配置、阈值与温度；
2. 对每条语音做 3–5 s 滑窗 + Top-K pooling 集成；
3. 首次运行时在 `Aasist/cache/` 下缓存各分支特征，加速后续推理。

## 进阶配置

可创建 JSON 配置覆盖默认值，例如：

```jsonc
{
  "sample_rate": 16000,
  "ssl_model": "wav2vec2_large",
  "feature_cache_root": "Aasist/cache",  // 特征缓存目录
  "train_min_chunk_seconds": 2.5,
  "train_max_chunk_seconds": 5.5,
  "eval_chunk_seconds": 4.0,
  "inference_chunk_seconds": 4.0,
  "inference_hop_ratio": 0.5,
  "fusion_hidden": [192, 96],
  "telephony_aug": false  // 关闭训练期电话增广
}
```

通过 `--config path/to/config.json` 在训练/推理脚本中引用。预测脚本会优先加载 checkpoint 内嵌配置，再应用外部覆盖。

## 常见问题

- **特征缓存在哪里？** 默认在 `Aasist/cache/`，可安全删除以重新生成。
- **未安装 torchaudio?** 请按上方命令安装，对应功能会自动解锁。
- **如何复现旧版单分支?** 在配置中把 `architecture` 设回 `AASIST` 并禁用 `telephony_aug` 即可（多分支默认更鲁棒，建议保留）。

## 模型复现说明

- AASIST 骨干沿用官方结构，新增分支在 `models/aasist3.py`；
- 数据加载、拆分与指标计算针对本地 CSV+WAV 做了适配；
- 训练/验证默认启用混合精度（仅 GPU 有效）。

## 参考

- 官方实现：<https://github.com/clovaai/aasist>
- 论文：Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, 2021.
