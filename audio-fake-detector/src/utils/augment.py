"""Data augmentation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch


def _lazy_import_signal():
    try:
        from scipy import signal  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError(
            "RawBoost augmentation requires SciPy. "
            "Install it with `pip install scipy` or disable RawBoost."
        ) from exc
    return signal


def _rand_range(low: float, high: float, integer: bool) -> float:
    value = float(np.random.uniform(low=low, high=high))
    if integer:
        return float(int(value))
    return value


def _norm_waveform(x: np.ndarray, always: bool) -> np.ndarray:
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak == 0:
        return x
    if always or peak > 1.0:
        return x / peak
    return x


def _gen_notch_coeffs(params: "RawBoostParameters", fs: int) -> np.ndarray:
    signal = _lazy_import_signal()
    b = np.array([1.0], dtype=np.float64)
    for _ in range(params.n_bands):
        fc = _rand_range(params.min_f, params.max_f, integer=False)
        bw = _rand_range(params.min_bw, params.max_bw, integer=False)
        coeff = int(_rand_range(params.min_coeff, params.max_coeff, integer=True))
        if coeff <= 0:
            coeff = 1
        if coeff % 2 == 0:
            coeff += 1
        f1 = max(fc - bw / 2.0, 1.0 / 1000.0)
        f2 = min(fc + bw / 2.0, fs / 2.0 - 1.0 / 1000.0)
        notch = signal.firwin(coeff, [float(f1), float(f2)], window="hamming", fs=fs)
        b = np.convolve(notch, b)
    gain = _rand_range(params.min_g, params.max_g, integer=False)
    _, h = signal.freqz(b, 1, fs=fs)
    scale = np.max(np.abs(h)) if h.size else 1.0
    if scale == 0:
        scale = 1.0
    return (10.0 ** (gain / 20.0)) * b / scale


def _filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    signal = _lazy_import_signal()
    n = b.shape[0] + 1
    xpad = np.pad(x, (0, n), mode="constant")
    y = signal.lfilter(b, 1, xpad)
    return y[int(n / 2) : int(y.shape[0] - n / 2)]


def _lnl_convolutive_noise(
    x: np.ndarray,
    params: "RawBoostParameters",
    fs: int,
) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    min_g = params.min_g
    max_g = params.max_g
    for i in range(params.n_f):
        if i == 1:
            min_g -= params.min_bias_lin_nonlinear
            max_g -= params.max_bias_lin_nonlinear
        updated_params = params.with_gain(min_g=min_g, max_g=max_g)
        b = _gen_notch_coeffs(updated_params, fs)
        y = y + _filter_fir(np.power(x, i + 1), b)
    y = y - np.mean(y)
    return _norm_waveform(y, always=False)


def _isd_additive_noise(x: np.ndarray, params: "RawBoostParameters") -> np.ndarray:
    beta = _rand_range(0, params.p, integer=False)
    y = x.copy()
    x_len = x.shape[0]
    n = int(x_len * (beta / 100.0))
    if n <= 0:
        return y
    perm = np.random.permutation(x_len)[:n]
    random_signs = ((2 * np.random.rand(perm.shape[0])) - 1) * (
        (2 * np.random.rand(perm.shape[0])) - 1
    )
    r = params.g_sd * x[perm] * random_signs
    y[perm] = x[perm] + r
    return _norm_waveform(y, always=False)


def _ssi_additive_noise(
    x: np.ndarray,
    params: "RawBoostParameters",
    fs: int,
) -> np.ndarray:
    noise = np.random.normal(0, 1, size=x.shape[0])
    b = _gen_notch_coeffs(params, fs)
    noise = _filter_fir(noise, b)
    noise = _norm_waveform(noise, always=True)
    snr = _rand_range(params.snr_min, params.snr_max, integer=False)
    noise = (
        noise / np.linalg.norm(noise, ord=2)
        * np.linalg.norm(x, ord=2)
        / 10.0 ** (0.05 * snr)
    )
    return x + noise


@dataclass
class RawBoostParameters:
    """Container for RawBoost hyper-parameters."""

    n_bands: int = 5
    min_f: float = 20.0
    max_f: float = 8000.0
    min_bw: float = 100.0
    max_bw: float = 1000.0
    min_coeff: float = 10.0
    max_coeff: float = 100.0
    min_g: float = 0.0
    max_g: float = 0.0
    min_bias_lin_nonlinear: float = 5.0
    max_bias_lin_nonlinear: float = 20.0
    n_f: int = 5
    p: float = 10.0
    g_sd: float = 2.0
    snr_min: float = 10.0
    snr_max: float = 40.0

    def with_gain(self, min_g: float, max_g: float) -> "RawBoostParameters":
        clone = RawBoostParameters(**self.to_dict())
        clone.min_g = min_g
        clone.max_g = max_g
        return clone

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_bands": self.n_bands,
            "min_f": self.min_f,
            "max_f": self.max_f,
            "min_bw": self.min_bw,
            "max_bw": self.max_bw,
            "min_coeff": self.min_coeff,
            "max_coeff": self.max_coeff,
            "min_g": self.min_g,
            "max_g": self.max_g,
            "min_bias_lin_nonlinear": self.min_bias_lin_nonlinear,
            "max_bias_lin_nonlinear": self.max_bias_lin_nonlinear,
            "n_f": self.n_f,
            "p": self.p,
            "g_sd": self.g_sd,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
        }


@dataclass
class RawBoostAugmentor:
    """Implements RawBoost augmentations for waveform tensors."""

    sample_rate: int
    algo: int = 4
    probability: float = 0.5
    parameters: RawBoostParameters = field(default_factory=RawBoostParameters)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("RawBoost probability must be in [0, 1].")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

    @classmethod
    def from_config(
        cls,
        sample_rate: int,
        algo: int,
        probability: float,
        overrides: Optional[Dict[str, float]] = None,
    ) -> "RawBoostAugmentor":
        params = RawBoostParameters()
        if overrides:
            for key, value in overrides.items():
                if not hasattr(params, key):
                    raise KeyError(f"Unsupported RawBoost parameter '{key}'.")
                setattr(params, key, value)
        return cls(sample_rate=sample_rate, algo=algo, probability=probability, parameters=params)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 1:
            raise ValueError("RawBoost expects a 1-D waveform tensor.")
        if self.probability < 1.0 and random.random() > self.probability:
            return waveform
        numpy_wave = waveform.detach().cpu().numpy().astype(np.float64)
        processed = self._process_numpy(numpy_wave)
        processed_tensor = torch.from_numpy(processed.astype(np.float32))
        return processed_tensor.to(waveform.device)

    def _process_numpy(self, waveform: np.ndarray) -> np.ndarray:
        algo = int(self.algo)
        params = self.parameters
        fs = self.sample_rate
        if algo == 1:
            return _lnl_convolutive_noise(waveform, params, fs)
        if algo == 2:
            return _isd_additive_noise(waveform, params)
        if algo == 3:
            return _ssi_additive_noise(waveform, params, fs)
        if algo == 4:
            y = _lnl_convolutive_noise(waveform, params, fs)
            y = _isd_additive_noise(y, params)
            return _ssi_additive_noise(y, params, fs)
        if algo == 5:
            y = _lnl_convolutive_noise(waveform, params, fs)
            return _isd_additive_noise(y, params)
        if algo == 6:
            y = _lnl_convolutive_noise(waveform, params, fs)
            return _ssi_additive_noise(y, params, fs)
        if algo == 7:
            y = _isd_additive_noise(waveform, params)
            return _ssi_additive_noise(y, params, fs)
        if algo == 8:
            y1 = _lnl_convolutive_noise(waveform, params, fs)
            y2 = _isd_additive_noise(waveform, params)
            combined = y1 + y2
            return _norm_waveform(combined, always=False)
        return waveform
