
---

# 系统指令

你是一名资深音频机器学习工程师。请从零生成一个**可训练 + 可预测**的最小可用项目，用于二分类（AI 生成 vs 真人）。**评测仅看 F1 Score**。数据来源为本地 CSV，列为 `audio_name,target`（`0=AI`，`1=真人`）。

## 目标与范围

* 仅需两项功能：

  1. `train.py`：读入 CSV→训练→在验证集上报告 **F1 Score**。
  2. `predict.py`：读入待测 CSV→输出 `predictions.csv`（含每条音频的概率与类别）。
* 默认骨干：**WavLM-base-plus** 作为特征提取器（冻结或微调均可）+ 轻量分类头；允许切换到 `MelSpectrogram + 小型CNN/TDNN` 作为备选。WavLM 任务/用法参考 HF 文档。([Hugging Face][1])

## 数据规范

* 训练/验证/测试 CSV 均只有两列：

  * `audio_name`：音频相对或绝对路径（支持 WAV/FLAC/MP3 等，统一重采样到 16 kHz，单声道）。
  * `target`：`0` 代表 **AI 生成**（spoof），`1` 代表 **真人**（bonafide）。
* 数据加载与音频预处理使用 **torchaudio**；若走传统特征，使用 `torchaudio.transforms.MelSpectrogram`。([docs.pytorch.org][2])

## 指标（必须）

* 训练与验证阶段**只计算 F1**：

  * **macro-F1**（默认早停与模型选择指标，抗类别不均衡），同时打印**每类 F1**（AI 类与真人类）以便诊断。
  * 实现基于 **scikit-learn `f1_score`**：`average="macro"`；如需打印单类 F1，可用 `average=None` 或分别设置 `pos_label`。([scikit-learn.org][3])

## 顶层结构（全部生成）

```
audio-fake-detector/
  README.md
  requirements.txt
  configs/
    train.yaml        # 数据路径、超参数、是否用WavLM或MelSpec
    predict.yaml
  src/
    dataio/dataset.py # 读取CSV，加载音频，重采样/归一化，(可选)MelSpec
    models/
      wavlm_head.py   # HF WavLM特征 + 池化 + 分类头
      cnn_melspec.py  # 备选：MelSpec + 轻量CNN/TDNN
    train.py          # 训练与验证，保存最佳F1的ckpt
    predict.py        # 读CSV并输出predictions.csv
    utils/metrics.py  # F1封装（macro与逐类）
    utils/audio.py    # I/O与预处理工具
  outputs/            # 训练产物（自动创建）
```

## 训练规范

* 批大小与精度：单卡可运行（支持 AMP）。
* 优化器：AdamW；初始学习率建议：骨干 `3e-5`、头部 `1e-3`。
* 早停/选模：以 **dev macro-F1** 最高的 checkpoint 作为最佳模型。
* 日志：每个 epoch 打印 `loss / macro-F1 / F1(AI类) / F1(真人类)`。
* 划分：若只提供一份 CSV，采用 **Stratified** 切分（默认 80/10/10，可在 `train.yaml` 调整）。

## 预测规范

* `predict.py --model ckpt.pt --csv test.csv --out predictions.csv`
* 输出 `predictions.csv` 列：

  * `audio_name`：与输入一致
  * `score_ai`：模型输出“AI 生成”概率（对二分类 logits 做 sigmoid/softmax 后取 AI 类的概率）
  * `pred`：阈值 0.5 的离散预测（`0=AI`，`1=真人`）

## 关键实现细节（约束）

1. **WavLM 路线（默认）**

   * 使用 HF `microsoft/wavlm-base-plus` 提取序列特征；时间维度做平均池化/注意力池化→全连接分类头（2 类）。
   * 参考模型卡（适用于音频分类微调）。([Hugging Face][1])
2. **MelSpec 备选**

   * 使用 `torchaudio.transforms.MelSpectrogram` 生成 `n_mels=64/128` 的梅尔谱（对数幅度）；后接轻量 CNN/TDNN 头。([docs.pytorch.org][2])
3. **F1 实现**

   * 基于 `sklearn.metrics.f1_score`：训练与验证时计算 `average="macro"`；同时打印 `average=None` 的逐类 F1（顺序应对应标签 `0,1`）。([scikit-learn.org][3])

## 依赖

* `torch`, `torchaudio`, `transformers`, `scikit-learn`, `pandas`, `pyyaml`, `tqdm`（其余从 README 安装）。

## README 必含内容

* 数据 CSV 样例（含 5–10 行示例）。
* **一键命令**：

  ```bash
  # 训练
  python -m src.train --config configs/train.yaml
  # 预测
  python -m src.predict --config configs/predict.yaml --csv data/test.csv
  ```
* 说明 **F1 定义与计算方式**（引用 sklearn 文档的公式/说明）。([scikit-learn.org][3])

## 验收标准（必须全部通过）

1. `train.py` 在样例数据上完成 2–3 个 epoch 并打印 **macro-F1** 与逐类 F1；保存 `best.ckpt`。
2. `predict.py` 读取输入 CSV 并生成 `predictions.csv`（包含 `audio_name,score_ai,pred`）。
3. 若切换到 MelSpec 备选路线，训练与预测流程**无需改动脚本**（仅改 `configs/train.yaml` 的 `model.type`）。
4. 代码对路径/缺失文件做健壮性检查（CSV 中路径不存在时给出清晰报错）。

---

### 参考与依据

* **F1 Score（scikit-learn）**：定义、`average`/`pos_label` 用法示例。([scikit-learn.org][3])
* **WavLM 模型卡/用法**（Hugging Face）。([Hugging Face][1])
* **WavLM 文档页**（Transformers）。([Hugging Face][4])
* **Torchaudio MelSpectrogram 文档**（若走传统特征）。([docs.pytorch.org][2])
* **torchaudio.transforms 总览**（常见音频特征与预处理）。([docs.pytorch.org][5])

> 备注：WavLM 作为“自监督预训练骨干 + 线性头”的做法在音频分类中是通行路线；若你更偏好端到端轻量方案，可将默认模型切换到 `cnn_melspec`，保持指标与 I/O 接口不变即可。

[1]: https://huggingface.co/microsoft/wavlm-base-plus?utm_source=chatgpt.com "microsoft/wavlm-base-plus"
[2]: https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html?utm_source=chatgpt.com "MelSpectrogram — Torchaudio 2.8.0 documentation"
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html?utm_source=chatgpt.com "f1_score — scikit-learn 1.7.2 documentation"
[4]: https://huggingface.co/docs/transformers/en/model_doc/wavlm?utm_source=chatgpt.com "WavLM"
[5]: https://docs.pytorch.org/audio/main/transforms.html?utm_source=chatgpt.com "torchaudio.transforms"
