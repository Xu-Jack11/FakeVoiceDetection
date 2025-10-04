# AASIST 集成说明

本目录收录了 [clovaai/aasist](https://github.com/clovaai/aasist) 的核心模型代码，并针对本仓库的 Kaggle Fake Voice 数据集做了轻量适配。

## 目录结构

- `models/AASIST.py`：AASIST 模型主体，直接来源于官方实现，保持原有网络结构。
- `dataset.py`：加载本地 `dataset/` 目录中 WAV 文件的工具，默认采样率 16 kHz、最大长度 64 600 采样点。
- `train.py`：使用本地数据训练 / 验证 AASIST 的脚本，支持混合精度。
- `LICENSE`：保留原项目的 MIT 许可证。
- `README.md`：当前使用说明。

## 数据准备

> ⚠️ **请勿运行原仓库的 `download_dataset.py`，我们直接使用本项目的 `dataset/` 数据。**

- 训练数据路径：`dataset/train.csv` + `dataset/train/*.wav`
- 测试数据路径：`dataset/test.csv` + `dataset/test/*.wav`

CSV 的格式需包含列：

- `audio_name`：对应 WAV 文件名
- `target`：0 表示真人，1 表示伪造（仅训练/验证集需要）

## 环境依赖

```text
python >= 3.9
pytorch >= 1.13
soundfile
scikit-learn
numpy
tqdm
```

如需安装额外依赖，可执行：

```bash
pip install soundfile scikit-learn
```

## 训练示例

```bash
python -m Aasist.train --data-root dataset --epochs 5 --batch-size 16 --val-split 0.1 --output best_aasist.pth --predict-output aasist_prediction.csv
```

可选参数：

- `--freq-aug`：启用 AASIST 中的频率遮挡增强。
- `--config`：传入 JSON 文件覆盖模型默认超参数（字段与官方 `AASIST.conf` 中 `model_config` 对齐）。
- `--predict-output`：指定训练结束后自动预测的保存位置（默认 `Aasist/predictions_after_train.csv`）。
- `--predict-batch-size`：控制自动预测阶段的批大小。
- `--no-auto-predict`：若希望仅训练模型而不运行自动预测，可添加该参数。

训练完成后，脚本会自动加载最佳权重，对 `dataset/test.csv` + `dataset/test/*.wav` 执行一次推理，并生成 CSV 结果（`target` 中 0=伪造、1=真人）。

如需单独执行预测，可运行：

```bash
python -m Aasist.predict --checkpoint Aasist/best_model.pth --output aasist_predictions.csv
```

脚本输出最优模型权重，文件中包含 epoch、验证集 F1 等信息。

## 模型复现说明

- 结构保持与原仓库一致，超参数默认沿用 `AASIST.conf`。
- 数据加载、拆分与指标计算针对本地 CSV+WAV 数据做了定制。
- 训练/验证过程默认启用混合精度（仅 GPU 有效）。

## 参考

- 官方实现：<https://github.com/clovaai/aasist>
- 论文：Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, 2021.
