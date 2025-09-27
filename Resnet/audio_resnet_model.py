"""
基于ResNet的音频真假检测模型
使用Mel频谱图作为输入特征，通过深度学习识别真人声音和AI生成声音
"""

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
    
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512, max_len=5, device=None):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = max_len
        self.target_length = int(self.sample_rate * self.max_len / self.hop_length)
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)
        self.db_transform = torchaudio.transforms.AmplitudeToDB().to(self.device)
        self._resamplers = {}
        
    def load_audio(self, file_path):
        '''Load audio and resample to the target rate if needed.'''
        try:
            waveform, sr = torchaudio.load_with_torchcodec(file_path)
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
        mel_spec = self.extract_mel_spectrogram(audio)
        if mel_spec is None:
            return None
        mel_spec = self.pad_or_trim(mel_spec)
        mel_spec = self.normalize(mel_spec).unsqueeze(0).float()
        if self.device.type != 'cpu':
            mel_spec = mel_spec.cpu()
        return mel_spec


# 自定义数据集类
class AudioDataset(Dataset):
    """音频数据集类"""
    
    def __init__(self, csv_file, audio_dir, preprocessor, transform=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.preprocessor = preprocessor
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_name = self.data.iloc[idx]['audio_name']
        audio_path = os.path.join(self.audio_dir, audio_name)
        
        # 处理音频
        mel_spec = self.preprocessor.process_audio(audio_path)
        
        if mel_spec is None:
            # 如果音频加载失败，返回零张量
            mel_spec = torch.zeros(1, self.preprocessor.n_mels, self.preprocessor.target_length)
        
        # 获取标签（如果存在）
        if 'target' in self.data.columns:
            label = self.data.iloc[idx]['target']
            label = torch.tensor(label, dtype=torch.long)
            return mel_spec, label
        else:
            return mel_spec, audio_name

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
def AudioResNet18(num_classes=2):
    """ResNet-18 for audio classification"""
    return AudioResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def AudioResNet34(num_classes=2):
    """ResNet-34 for audio classification"""
    return AudioResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def AudioResNet50(num_classes=2):
    """ResNet-50 for audio classification"""
    return AudioResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def AudioResNet101(num_classes=2):
    """ResNet-101 for audio classification"""
    return AudioResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def AudioResNet152(num_classes=2):
    """ResNet-152 for audio classification"""
    return AudioResNet(Bottleneck, [3, 8, 36, 3], num_classes)