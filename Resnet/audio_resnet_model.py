"""
基于ResNet的音频真假检测模型
使用Mel频谱图作为输入特征，通过深度学习识别真人声音和AI生成声音
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 音频预处理类
class AudioPreprocessor:
    """音频预处理工具类，负责音频加载、特征提取和数据增强"""
    
    def __init__(
        self,
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        max_len=5,
        device=None,
        feature_types=("mel", "lfcc", "cqt"),
        n_lfcc=None,
        cqt_bins=None,
        cqt_filter_scale=1.0,
        cqt_bins_per_octave=12,
        cqt_fmin=None,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = max_len
        self.target_length = int(self.sample_rate * self.max_len / self.hop_length)
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_types = tuple(feature_types)
        self.n_lfcc = n_lfcc if n_lfcc is not None else self.n_mels
        self.cqt_bins_per_octave = cqt_bins_per_octave
        self.cqt_fmin = cqt_fmin if cqt_fmin is not None else 32.703195662574764
        self._target_device_index = self.device.index if self.device.index is not None else 0
        nyquist = self.sample_rate / 2.0
        max_octaves = np.log2(max(nyquist / max(self.cqt_fmin, 1e-6), 1.0))
        max_cqt_bins = int(np.floor(max_octaves * self.cqt_bins_per_octave))
        if max_cqt_bins <= 0:
            max_cqt_bins = self.cqt_bins_per_octave
        desired_cqt_bins = cqt_bins if cqt_bins is not None else self.n_mels
        if desired_cqt_bins > max_cqt_bins:
            warnings.warn(
                f"Requested CQT bins {desired_cqt_bins} exceed Nyquist limit; clipping to {max_cqt_bins}.",
                UserWarning,
                stacklevel=2
            )
            desired_cqt_bins = max_cqt_bins
        self.cqt_bins = max(desired_cqt_bins, 1)
        self.cqt_filter_scale = cqt_filter_scale
        self.feature_channels = len(self.feature_types)
        self._load_with_torchcodec = getattr(torchaudio, 'load_with_torchcodec', None)
        self._torchcodec_warned = False
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)
        self.db_transform = torchaudio.transforms.AmplitudeToDB().to(self.device)
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=self.sample_rate,
            n_filter=self.n_lfcc,
            f_min=0.0,
            f_max=nyquist,
            n_lfcc=self.n_lfcc,
            dct_type=2,
            norm='ortho',
            log_lf=True,
            speckwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'center': True
            }
        ).to(self.device)
        self._use_gpu_cqt = hasattr(torchaudio.transforms, 'CQT') and "cqt" in self.feature_types
        if self._use_gpu_cqt:
            self.cqt_transform = torchaudio.transforms.CQT(
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.cqt_fmin,
                n_bins=self.cqt_bins,
                bins_per_octave=self.cqt_bins_per_octave,
                filter_scale=self.cqt_filter_scale,
            ).to(self.device)
            self.cqt_db_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude').to(self.device)
        else:
            if "cqt" in self.feature_types:
                warnings.warn(
                    "torchaudio.transforms.CQT 不可用，将退回到 librosa CPU 实现。",
                    RuntimeWarning,
                    stacklevel=2
                )
            self.cqt_transform = None
            self.cqt_db_transform = None
        self._resamplers = {}

    def _ensure_device_context(self):
        if self.device.type != 'cuda':
            return
        if torch.cuda.device_count() <= self._target_device_index:
            raise RuntimeError(
                f"Requested CUDA device index {self._target_device_index} but only {torch.cuda.device_count()} devices available."
            )
        if not torch.cuda.is_initialized() or torch.cuda.current_device() != self._target_device_index:
            torch.cuda.set_device(self._target_device_index)
        
    def load_audio(self, file_path):
        '''Load audio and resample to the target rate if needed.'''
        try:
            self._ensure_device_context()
            if self._load_with_torchcodec is not None:
                try:
                    waveform, sr = self._load_with_torchcodec(file_path)
                except Exception as err:
                    if not self._torchcodec_warned:
                        warnings.warn(
                            "torchaudio.load_with_torchcodec failed; falling back to torchaudio.load."
                            " Install torchcodec for best performance.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        self._torchcodec_warned = True
                    waveform, sr = torchaudio.load(file_path)
            else:
                waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        try:
            waveform = waveform.to(self.device)
            if sr != self.sample_rate:
                resampler = self._resamplers.get(sr)
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate).to(self.device)
                    self._resamplers[sr] = resampler
                waveform = resampler(waveform)
            audio = waveform.mean(dim=0)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None


    def extract_mel_spectrogram(self, audio):
        '''Compute the log-mel spectrogram on the configured device.'''
        if audio is None:
            return None
        waveform = audio.unsqueeze(0) if audio.dim() == 1 else audio
        if waveform.size(1) < self.n_fft:
            pad_size = self.n_fft - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_size))
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.db_transform(mel_spec)
        return mel_spec_db.squeeze(0)


    def extract_lfcc(self, audio):
        """Compute Linear Frequency Cepstral Coefficients."""
        if audio is None:
            return None
        waveform = audio.unsqueeze(0) if audio.dim() == 1 else audio
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        try:
            lfcc = self.lfcc_transform(waveform)
            return lfcc.squeeze(0)
        except Exception as e:
            print(f"Error computing LFCC: {e}")
            return None


    def extract_cqt(self, audio):
        """Compute Constant-Q Transform magnitude in dB."""
        if audio is None:
            return None
        try:
            if self._use_gpu_cqt and self.cqt_transform is not None:
                waveform = audio.unsqueeze(0) if audio.dim() == 1 else audio
                if waveform.dim() == 3:
                    waveform = waveform.mean(dim=1)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                cqt = self.cqt_transform(waveform)
                cqt_mag = torch.abs(cqt)
                cqt_db = self.cqt_db_transform(cqt_mag)
                return cqt_db.squeeze(0)
            audio_np = audio.detach().cpu().numpy()
            cqt = librosa.cqt(
                audio_np,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.cqt_fmin,
                n_bins=self.cqt_bins,
                bins_per_octave=self.cqt_bins_per_octave,
                filter_scale=self.cqt_filter_scale,
            )
            if cqt.size == 0:
                return None
            cqt_mag = np.abs(cqt)
            ref = np.max(cqt_mag) if np.max(cqt_mag) > 0 else 1.0
            cqt_db = librosa.amplitude_to_db(cqt_mag, ref=ref)
            return torch.from_numpy(cqt_db).float().to(self.device)
        except Exception as e:
            print(f"Error computing CQT: {e}")
            return None


    def pad_or_trim(self, mel_spec):
        '''Pad or trim the spectrogram to the target length.'''
        if mel_spec.size(1) > self.target_length:
            mel_spec = mel_spec[:, :self.target_length]
        elif mel_spec.size(1) < self.target_length:
            pad_width = self.target_length - mel_spec.size(1)
            mel_spec = F.pad(mel_spec, (0, pad_width))
        return mel_spec


    def normalize(self, mel_spec):
        '''Standardize the spectrogram.'''
        mean = mel_spec.mean()
        std = mel_spec.std(unbiased=False)
        mel_spec = (mel_spec - mean) / (std + 1e-8)
        return mel_spec


    def process_audio(self, file_path):
        '''Full preprocessing pipeline executed on the selected device.'''
        audio = self.load_audio(file_path)
        if audio is None:
            return None
        self._ensure_device_context()
        feature_tensors = []
        for feature_name in self.feature_types:
            if feature_name == "mel":
                spec = self.extract_mel_spectrogram(audio)
                expected_bins = self.n_mels
            elif feature_name == "lfcc":
                spec = self.extract_lfcc(audio)
                expected_bins = self.n_lfcc
            elif feature_name == "cqt":
                spec = self.extract_cqt(audio)
                expected_bins = self.cqt_bins
            else:
                raise ValueError(f"Unsupported feature type: {feature_name}")

            if spec is None:
                spec = torch.zeros(expected_bins, self.target_length, device=self.device)
            else:
                spec = spec.to(self.device)
            if spec.size(0) != self.n_mels:
                spec = spec.unsqueeze(0).unsqueeze(0)
                spec = F.interpolate(spec, size=(self.n_mels, spec.size(-1)), mode='bilinear', align_corners=False)
                spec = spec.squeeze(0).squeeze(0)
            spec = self.pad_or_trim(spec)
            spec = self.normalize(spec)
            feature_tensors.append(spec)

        if not feature_tensors:
            return None

        features = torch.stack(feature_tensors, dim=0).float().contiguous()
        if features.device.type != 'cpu':
            features = features.cpu()
        return features


# 自定义数据集类
class AudioDataset(Dataset):
    """音频数据集类，支持特征缓存"""

    def __init__(
        self,
        csv_file,
        audio_dir,
        preprocessor,
        transform=None,
        feature_cache_dir=None,
        cache_miss_strategy="compute",
    ):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.preprocessor = preprocessor
        self.transform = transform
        self.feature_cache_dir = feature_cache_dir
        self.cache_miss_strategy = cache_miss_strategy
        self.feature_channels = getattr(preprocessor, 'feature_channels', None)
        self.n_mels = getattr(preprocessor, 'n_mels', None)
        self.target_length = getattr(preprocessor, 'target_length', None)
        if self.feature_cache_dir is not None:
            os.makedirs(self.feature_cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_name = self.data.iloc[idx]['audio_name']
        audio_path = os.path.join(self.audio_dir, audio_name)
        cache_path = None
        if self.feature_cache_dir is not None:
            base_name = os.path.splitext(os.path.basename(audio_name))[0]
            cache_path = os.path.join(self.feature_cache_dir, f"{base_name}.pt")

        features = None
        if cache_path is not None and os.path.isfile(cache_path):
            features = torch.load(cache_path, map_location='cpu')
            if isinstance(features, torch.Tensor):
                features = features.float()
                self.feature_channels = features.size(0)
                if features.dim() >= 2:
                    self.n_mels = features.size(1)
                if features.dim() >= 3:
                    self.target_length = features.size(2)
        else:
            if self.preprocessor is None:
                if self.cache_miss_strategy == 'error':
                    raise FileNotFoundError(f"特征缓存缺失: {cache_path}")
                raise RuntimeError("未提供预处理器，无法生成特征。")
            features = self.preprocessor.process_audio(audio_path)
            if isinstance(features, torch.Tensor):
                features = features.float()
            if features is not None and cache_path is not None and self.cache_miss_strategy != 'load':
                tmp_path = cache_path + '.tmp'
                torch.save(features, tmp_path)
                os.replace(tmp_path, cache_path)
            elif features is None and self.cache_miss_strategy == 'error':
                raise FileNotFoundError(f"无法为 {audio_name} 生成特征且未找到缓存。")
        
        if features is None:
            # 如果音频加载失败，返回零张量
            if self.feature_channels is None or self.n_mels is None or self.target_length is None:
                raise RuntimeError("无法确定特征形状，且音频加载失败。")
            features = torch.zeros(
                self.feature_channels,
                self.n_mels,
                self.target_length
            )
        
        # 获取标签（如果存在）
        if 'target' in self.data.columns:
            label = self.data.iloc[idx]['target']
            label = torch.tensor(label, dtype=torch.long)
            return features, label
        else:
            return features, audio_name

# ResNet基础块
class BasicBlock(nn.Module):
    """ResNet基础残差块"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 瓶颈块（用于更深的ResNet）
class Bottleneck(nn.Module):
    """ResNet瓶颈残差块"""
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 音频ResNet模型
class AudioResNet(nn.Module):
    """基于ResNet的音频分类模型"""
    
    def __init__(self, block, num_blocks, num_classes=2, input_channels=1):
        super(AudioResNet, self).__init__()
        self.in_planes = 64
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(0.5)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 输入: (batch_size, 1, n_mels, time_steps)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# 预定义的ResNet变体
def AudioResNet18(num_classes=2, input_channels=1):
    """ResNet-18 for audio classification"""
    return AudioResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)

def AudioResNet34(num_classes=2, input_channels=1):
    """ResNet-34 for audio classification"""
    return AudioResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels)

def AudioResNet50(num_classes=2, input_channels=1):
    """ResNet-50 for audio classification"""
    return AudioResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels)

def AudioResNet101(num_classes=2, input_channels=1):
    """ResNet-101 for audio classification"""
    return AudioResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_channels)

def AudioResNet152(num_classes=2, input_channels=1):
    """ResNet-152 for audio classification"""
    return AudioResNet(Bottleneck, [3, 8, 36, 3], num_classes, input_channels)