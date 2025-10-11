"""AASIST3 多分支幅度/相位/SSL 融合模型实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .AASIST import AASISTBackbone
from .fusion_head import LogitFusionHead
from ..features import CQCCFrontend, LFCCFrontend, PhaseFrontend, SSLFrontend


class SpectralEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = [in_channels, 64, 128, hidden_dim]
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.SiLU(),
                ]
            )
        self.network = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.network(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x


class SSLBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        feats = torch.cat([mean, max_pool], dim=-1)
        return self.mlp(feats)


class BranchClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        hidden = max(input_dim // 2, 32)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class Model(nn.Module):
    def __init__(self, d_args: Dict[str, object]) -> None:
        super().__init__()
        self.d_args = d_args
        self.sample_rate = int(d_args.get("sample_rate", 16000))
        cache_root = d_args.get("feature_cache_root", "Aasist/cache")
        cache_root = str(cache_root)

        cache_root_path = Path(cache_root)

        self.enable_lfcc = bool(d_args.get("enable_lfcc", True))
        self.enable_cqcc = bool(d_args.get("enable_cqcc", True))
        self.enable_phase = bool(d_args.get("enable_phase", True))
        self.enable_ssl = bool(d_args.get("enable_ssl", True))
        self.enable_aasist = bool(d_args.get("enable_aasist", True))

        # 幅度支路：LFCC + CQCC
        self.amplitude_extractors: list[tuple[str, nn.Module]] = []
        if self.enable_lfcc:
            self.lfcc = LFCCFrontend(cache_dir=cache_root_path / "lfcc")
            self.amplitude_extractors.append(("lfcc", self.lfcc))
        else:
            self.lfcc = None
        if self.enable_cqcc:
            self.cqcc = CQCCFrontend(cache_dir=cache_root_path / "cqcc")
            self.amplitude_extractors.append(("cqcc", self.cqcc))
        else:
            self.cqcc = None

        if self.amplitude_extractors:
            self.amplitude_encoder = SpectralEncoder(in_channels=len(self.amplitude_extractors))
            self.amplitude_head = BranchClassifier(self.amplitude_encoder.out_dim)
        else:
            self.amplitude_encoder = None
            self.amplitude_head = None

        # 相位支路
        if self.enable_phase:
            self.phase = PhaseFrontend(cache_dir=cache_root_path / "phase")
            self.phase_encoder = SpectralEncoder(in_channels=3)
            self.phase_head = BranchClassifier(self.phase_encoder.out_dim)
        else:
            self.phase = None
            self.phase_encoder = None
            self.phase_head = None

        # SSL 支路
        if self.enable_ssl:
            ssl_model_name = str(d_args.get("ssl_model", "wav2vec2_base"))
            self.ssl = SSLFrontend(
                model_name=ssl_model_name,
                cache_dir=cache_root_path / "ssl",
            )
            ssl_out_dim = self._infer_ssl_dim()
            self.ssl_encoder = SSLBranch(ssl_out_dim * 2)
            self.ssl_head = BranchClassifier(self.ssl_encoder.out_dim)
        else:
            self.ssl = None
            self.ssl_encoder = None
            self.ssl_head = None

        # AASIST 原始骨干用于增强幅度支路表示
        if self.enable_aasist:
            aasist_conf = dict(d_args)
            aasist_conf.setdefault("architecture", "AASIST3")
            self.aasist_backbone = AASISTBackbone(aasist_conf)
            gat_dims = aasist_conf.get("gat_dims", [64, 32])
            aasist_dim = 5 * gat_dims[-1] if isinstance(gat_dims, (list, tuple)) else 160
            self.aasist_head = BranchClassifier(aasist_dim)
        else:
            self.aasist_backbone = None
            self.aasist_head = None

        branch_names: list[str] = []
        if self.amplitude_encoder is not None:
            branch_names.append("amplitude")
        if self.phase_head is not None:
            branch_names.append("phase")
        if self.ssl_head is not None:
            branch_names.append("ssl")
        if self.aasist_head is not None:
            branch_names.append("aasist")
        if not branch_names:
            raise ValueError("必须至少启用一个 AASIST3 特征分支")

        self.branch_names = branch_names
        self.fusion = LogitFusionHead(
            branch_names,
            logit_dim=2,
            hidden_dims=d_args.get("fusion_hidden", (128, 64)),
            dropout=float(d_args.get("fusion_dropout", 0.1)),
        )

    def _infer_ssl_dim(self) -> int:
        dummy = torch.randn(1, self.sample_rate * 2)
        features = self.ssl(dummy, sample_rate=self.sample_rate, training=False)
        return features.size(-1)

    @staticmethod
    def _align_time_dim(feats: List[Tensor]) -> List[Tensor]:
        if not feats:
            return feats
        max_time = max(tensor.shape[-1] for tensor in feats)
        aligned: List[Tensor] = []
        for tensor in feats:
            current_time = tensor.shape[-1]
            if current_time == max_time:
                aligned.append(tensor)
                continue
            slice_len = min(current_time, max_time)
            new_shape = list(tensor.shape)
            new_shape[-1] = max_time
            padded = tensor.new_zeros(new_shape)
            padded[..., :slice_len] = tensor[..., :slice_len]
            aligned.append(padded)
        return aligned

    def _extract_batch(
        self,
        extractor: nn.Module,
        waveform: Tensor,
        branch: str,
        utt_ids: Optional[list[str]],
        training: bool,
    ) -> Tensor:
        # 训练阶段 or 无缓存需求时，直接整体并行提取，加快 GPU 利用率
        use_vectorized = training or utt_ids is None

        if use_vectorized:
            return extractor(
                waveform,
                sample_rate=self.sample_rate,
                cache_key=None,
                training=training,
                metadata={"branch": branch},
            )

        outputs = []
        batch_size = waveform.size(0)
        for i in range(batch_size):
            key = None
            if utt_ids is not None:
                key = f"{utt_ids[i]}_{branch}"
            feat = extractor(
                waveform[i : i + 1],
                sample_rate=self.sample_rate,
                cache_key=key,
                training=training,
                metadata={"branch": branch},
            )
            outputs.append(feat)
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        waveform: Tensor,
        utt_ids: Optional[list[str]] = None,
        training: bool = True,
        **kwargs: object,
    ) -> tuple[Dict[str, Tensor], Tensor]:
        branch_logits: dict[str, Tensor] = {}

        if self.amplitude_encoder is not None:
            amp_feats: list[Tensor] = []
            for feat_name, extractor in self.amplitude_extractors:
                feat = self._extract_batch(extractor, waveform, feat_name, utt_ids, training)
                amp_feats.append(feat.unsqueeze(1))
            amp_feats = self._align_time_dim(amp_feats)
            amp_input = torch.cat(amp_feats, dim=1)
            amp_repr = self.amplitude_encoder(amp_input)
            branch_logits["amplitude"] = self.amplitude_head(amp_repr)

        if self.phase is not None and self.phase_encoder is not None:
            phase = self._extract_batch(self.phase, waveform, "phase", utt_ids, training)
            phase_repr = self.phase_encoder(phase)
            branch_logits["phase"] = self.phase_head(phase_repr)

        if self.ssl is not None and self.ssl_encoder is not None:
            ssl_feats = self._extract_batch(self.ssl, waveform, "ssl", utt_ids, training)
            ssl_repr = self.ssl_encoder(ssl_feats)
            branch_logits["ssl"] = self.ssl_head(ssl_repr)

        if self.aasist_backbone is not None and self.aasist_head is not None:
            aasist_repr, _ = self.aasist_backbone(waveform, Freq_aug=training)
            branch_logits["aasist"] = self.aasist_head(aasist_repr)

        fused_logits = self.fusion(branch_logits)
        return branch_logits, fused_logits
