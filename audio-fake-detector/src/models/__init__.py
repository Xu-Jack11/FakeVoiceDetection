"""Model builders for audio fake detector."""

from .wavlm_head import WavLMClassifier
from .cnn_melspec import MelSpecCNN


def create_model(model_cfg: dict):
    """Factory method to build a model from config dictionary."""
    model_type = model_cfg.get("type", "wavlm")
    if model_type == "wavlm":
        params = model_cfg.get("wavlm", {})
        return WavLMClassifier(
            pretrained_name=params.get("pretrained_name", "microsoft/wavlm-base-plus"),
            freeze_encoder=params.get("freeze_encoder", False),
            dropout=params.get("dropout", 0.1),
        )
    if model_type == "cnn_melspec":
        params = model_cfg.get("cnn_melspec", {})
        return MelSpecCNN(
            sample_rate=params.get("sample_rate", 16000),
            n_mels=params.get("n_mels", 128),
            n_fft=params.get("n_fft", 1024),
            hop_length=params.get("hop_length", 320),
            dropout=params.get("dropout", 0.2),
        )
    raise ValueError(f"Unsupported model type '{model_type}'.")
