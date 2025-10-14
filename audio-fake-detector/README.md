# Audio Fake Detector

Minimal yet extensible baseline for detecting AI-generated speech versus human speech. The project bundles training and inference utilities centred on a WavLM backbone with an optional Mel-spectrogram CNN fallback. Evaluation focuses exclusively on macro-F1.

## Project Layout

```
audio-fake-detector/
  configs/
    train.yaml       # data paths, hyperparameters, model choice
    predict.yaml     # inference defaults
  src/
    dataio/dataset.py
    models/wavlm_head.py
    models/cnn_melspec.py
    train.py
    predict.py
    utils/audio.py
    utils/metrics.py
  outputs/           # checkpoints written here (ignored if absent)
  requirements.txt
  README.md
```

## Dataset Format

All CSV files must expose the columns `audio_name` and (optionally for inference) `target`. Audio paths can be absolute or relative to the configured `audio_dir`. Audio is expected to resample to 16 kHz mono.

Example (`dataset/train.csv`, first 6 rows):

```text
audio_name,target
kaggle-audio-train-000001.wav,1
kaggle-audio-train-000002.wav,0
kaggle-audio-train-000003.wav,1
kaggle-audio-train-000004.wav,1
kaggle-audio-train-000005.wav,0
kaggle-audio-train-000006.wav,1
```

## Quickstart

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

Update `configs/train.yaml` so that `data.train_csv`, `data.audio_dir`, and related paths match your dataset layout (the defaults point to `../dataset/train.csv` and `../dataset/train` to fit the provided repository).

### Training

```bash
python -m src.train --config configs/train.yaml
```

Key behaviour:

- Default model: `model.type: wavlm` with the Hugging Face checkpoint `microsoft/wavlm-base-plus`.
- Stratified 80/10 split automatically triggered when no external validation CSV is given.
- Early stopping and checkpoint selection rely on validation macro-F1. The best model is stored as `outputs/best.ckpt`.
- Training audio is randomly cropped to 3–5 second snippets (configurable via `data.crop`) to keep WavLM sequence lengths manageable; evaluation/prediction uses deterministic centre crops.
- When `post_training.run_prediction` is enabled (default in `configs/train.yaml`), the training script reruns `src.predict` automatically using the freshly saved checkpoint.
- To fallback to the Mel-spectrogram CNN, set `model.type: cnn_melspec` in `configs/train.yaml`; no code changes required.

### Prediction

```bash
python -m src.predict --config configs/predict.yaml --csv ../dataset/test.csv
```

Outputs `predictions.csv` with the columns:

- `audio_name`: copied from the input CSV.
- `score_ai`: probability that the clip is AI-generated (class `0`).
- `pred`: discrete prediction (`0 = AI`, `1 = human`) using a 0.5 threshold on `score_ai`.

Override the destination via `--out path/to/predictions.csv`. The script reuses the checkpoint `outputs/best.ckpt` by default; supply another checkpoint with `--model`.

## F1 Metric

Macro-F1 is the sole optimisation target. It is computed with [`sklearn.metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) using `average="macro"` on validation outputs. For diagnostics, per-class F1 values (for labels `0` and `1`) are logged every epoch. The helper in `src/utils/metrics.py` wraps these calls.

## Configuration Highlights

- `data.sample_rate`: resampling target used during loading.
- `data.crop`: duration window (min/max seconds) and per-split sampling mode for audio truncation.
- `training.encoder_lr` / `training.head_lr`: split learning rates for the WavLM encoder and classifier head (ignored for the CNN model).
- `training.amp`: enable automatic mixed precision on CUDA devices.
- `training.resume_from`: optional checkpoint to resume training.
- `configs/predict.yaml`: set `data.audio_dir` to the directory containing inference audio files.
- `post_training`: configure automatic prediction after training (override CSV/output paths or disable if undesired).

## Troubleshooting

- Ensure audio files referenced in CSVs exist; the loader raises a clear `FileNotFoundError` when a path is missing.
- Large datasets benefit from increasing `data.num_workers` for faster loading.
- When switching models, keep `sample_rate` consistent with the pretrained backbone (16 kHz for WavLM).
