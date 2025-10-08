"""数据准备和验证脚本

用于:
1. 检查数据集完整性
2. 统计数据分布
3. 预提取和缓存特征
4. 验证音频文件
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

from Aasist.data.dataset_multipath import AASIST3Dataset, AASIST3DatasetConfig


def check_dataset_integrity(data_root: Path) -> Dict:
    """检查数据集完整性"""
    print("检查数据集完整性...")
    
    stats = {
        "train_csv_exists": False,
        "test_csv_exists": False,
        "train_dir_exists": False,
        "test_dir_exists": False,
        "num_train_samples": 0,
        "num_test_samples": 0,
        "missing_files": [],
        "corrupted_files": [],
    }
    
    # 检查文件存在性
    train_csv = data_root / "train.csv"
    test_csv = data_root / "test.csv"
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    
    stats["train_csv_exists"] = train_csv.exists()
    stats["test_csv_exists"] = test_csv.exists()
    stats["train_dir_exists"] = train_dir.exists()
    stats["test_dir_exists"] = test_dir.exists()
    
    # 检查训练集
    if stats["train_csv_exists"] and stats["train_dir_exists"]:
        df = pd.read_csv(train_csv)
        stats["num_train_samples"] = len(df)
        
        print(f"检查 {len(df)} 个训练音频...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            audio_path = train_dir / row["audio_name"]
            
            if not audio_path.exists():
                stats["missing_files"].append(str(audio_path))
            else:
                # 验证音频文件
                try:
                    info = sf.info(audio_path)
                    if info.frames == 0:
                        stats["corrupted_files"].append(str(audio_path))
                except Exception:
                    stats["corrupted_files"].append(str(audio_path))
    
    # 检查测试集
    if stats["test_dir_exists"]:
        test_files = list(test_dir.glob("*.wav"))
        stats["num_test_samples"] = len(test_files)
    
    return stats


def print_stats(stats: Dict):
    """打印统计信息"""
    print("\n" + "="*60)
    print("数据集统计".center(60))
    print("="*60)
    
    print(f"\n文件检查:")
    print(f"  train.csv: {'✓' if stats['train_csv_exists'] else '✗'}")
    print(f"  test.csv: {'✓' if stats['test_csv_exists'] else '✗'}")
    print(f"  train/: {'✓' if stats['train_dir_exists'] else '✗'}")
    print(f"  test/: {'✓' if stats['test_dir_exists'] else '✗'}")
    
    print(f"\n样本数量:")
    print(f"  训练集: {stats['num_train_samples']}")
    print(f"  测试集: {stats['num_test_samples']}")
    
    if stats['missing_files']:
        print(f"\n缺失文件 ({len(stats['missing_files'])}):")
        for f in stats['missing_files'][:5]:
            print(f"  - {f}")
        if len(stats['missing_files']) > 5:
            print(f"  ... 还有 {len(stats['missing_files']) - 5} 个")
    
    if stats['corrupted_files']:
        print(f"\n损坏文件 ({len(stats['corrupted_files'])}):")
        for f in stats['corrupted_files'][:5]:
            print(f"  - {f}")
        if len(stats['corrupted_files']) > 5:
            print(f"  ... 还有 {len(stats['corrupted_files']) - 5} 个")
    
    print("\n" + "="*60 + "\n")


def analyze_label_distribution(data_root: Path):
    """分析标签分布"""
    train_csv = data_root / "train.csv"
    
    if not train_csv.exists():
        print("train.csv 不存在,跳过标签分析")
        return
    
    df = pd.read_csv(train_csv)
    
    if "target" not in df.columns:
        print("train.csv 中没有 target 列")
        return
    
    print("\n标签分布:")
    print("-" * 40)
    
    value_counts = df["target"].value_counts()
    for label, count in value_counts.items():
        ratio = count / len(df) * 100
        print(f"  类别 {label}: {count:5d} ({ratio:5.2f}%)")
    
    # 计算不平衡比例
    imbalance_ratio = value_counts.max() / value_counts.min()
    print(f"\n不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("  ⚠️  数据不平衡较严重,建议使用 Class-Balanced Loss")


def analyze_audio_duration(data_root: Path, num_samples: int = 1000):
    """分析音频时长分布"""
    train_dir = data_root / "train"
    train_csv = data_root / "train.csv"
    
    if not train_dir.exists() or not train_csv.exists():
        print("跳过音频时长分析")
        return
    
    df = pd.read_csv(train_csv)
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    print(f"\n分析 {len(sample_df)} 个音频的时长...")
    durations = []
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        audio_path = train_dir / row["audio_name"]
        if audio_path.exists():
            try:
                info = sf.info(audio_path)
                duration = info.frames / info.samplerate
                durations.append(duration)
            except Exception:
                pass
    
    if durations:
        durations = np.array(durations)
        print("\n音频时长统计 (秒):")
        print("-" * 40)
        print(f"  最小: {durations.min():.2f}")
        print(f"  最大: {durations.max():.2f}")
        print(f"  平均: {durations.mean():.2f}")
        print(f"  中位数: {np.median(durations):.2f}")
        print(f"  25th percentile: {np.percentile(durations, 25):.2f}")
        print(f"  75th percentile: {np.percentile(durations, 75):.2f}")


def precache_features(
    data_root: Path,
    feature_types: list,
    cache_dir: Path,
    num_workers: int = 4,
):
    """预提取和缓存特征"""
    print(f"\n预提取特征: {', '.join(feature_types)}")
    print(f"缓存目录: {cache_dir}")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    config = AASIST3DatasetConfig(
        csv_path=data_root / "train.csv",
        audio_dir=data_root / "train",
        sample_rate=16000,
        training=False,
        fixed_chunk_duration=4.0,
        feature_types=feature_types,
        feature_cache_dir=cache_dir,
        use_vad=True,
        use_augmentation=False,
    )
    
    dataset = AASIST3Dataset(config)
    
    print(f"处理 {len(dataset)} 个样本...")
    for i in tqdm(range(len(dataset))):
        try:
            _ = dataset[i]
        except Exception as e:
            print(f"\n处理样本 {i} 时出错: {e}")
    
    print(f"\n特征缓存完成!")


def main():
    parser = argparse.ArgumentParser(description="数据准备和验证")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--check_integrity", action="store_true", help="检查完整性")
    parser.add_argument("--analyze_labels", action="store_true", help="分析标签分布")
    parser.add_argument("--analyze_duration", action="store_true", help="分析音频时长")
    parser.add_argument("--precache", action="store_true", help="预缓存特征")
    parser.add_argument("--feature_types", nargs="+", default=["lfcc", "cqcc", "phase"], help="特征类型")
    parser.add_argument("--cache_dir", type=str, default="./feature_cache", help="缓存目录")
    parser.add_argument("--all", action="store_true", help="执行所有分析")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"错误: 数据根目录不存在: {data_root}")
        return
    
    if args.all:
        args.check_integrity = True
        args.analyze_labels = True
        args.analyze_duration = True
    
    # 检查完整性
    if args.check_integrity:
        stats = check_dataset_integrity(data_root)
        print_stats(stats)
    
    # 分析标签分布
    if args.analyze_labels:
        analyze_label_distribution(data_root)
    
    # 分析音频时长
    if args.analyze_duration:
        analyze_audio_duration(data_root)
    
    # 预缓存特征
    if args.precache:
        cache_dir = Path(args.cache_dir)
        precache_features(data_root, args.feature_types, cache_dir)


if __name__ == "__main__":
    main()
