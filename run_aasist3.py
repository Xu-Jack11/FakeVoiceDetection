"""AASIST3 快速启动脚本

提供便捷的训练和推理接口
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train(args):
    """启动训练"""
    cmd = [
        sys.executable, "-m", "Aasist.train_aasist3",
        "--data_root", args.data_root,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--feature_types", *args.feature_types,
        "--loss_type", args.loss_type,
    ]
    
    if args.use_amp:
        cmd.append("--use_amp")
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def predict(args):
    """启动推理"""
    cmd = [
        sys.executable, "-m", "Aasist.predict_aasist3",
        "--checkpoint", args.checkpoint,
        "--test_dir", args.test_dir,
        "--output_csv", args.output_csv,
        "--batch_size", str(args.batch_size),
        "--window_size", str(args.window_size),
        "--hop_size", str(args.hop_size),
        "--pooling", args.pooling,
        "--threshold", str(args.threshold),
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def check_dependencies():
    """检查依赖"""
    print("Checking dependencies...")
    
    required = [
        "torch",
        "torchaudio",
        "numpy",
        "pandas",
        "sklearn",
        "soundfile",
        "librosa",
        "tqdm",
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements_aasist3.txt")
        return False
    
    print("\n所有依赖已安装!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AASIST3 快速启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查依赖
  python run_aasist3.py check
  
  # 训练模型
  python run_aasist3.py train --data_root ./dataset --output_dir ./output
  
  # 推理预测
  python run_aasist3.py predict --checkpoint ./output/best_aasist3.pth --test_dir ./dataset/test
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # Check 命令
    subparsers.add_parser("check", help="检查依赖")
    
    # Train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    train_parser.add_argument("--output_dir", type=str, default="./output_aasist3", help="输出目录")
    train_parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    train_parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    train_parser.add_argument(
        "--feature_types",
        nargs="+",
        default=["lfcc", "cqcc", "phase"],
        choices=["lfcc", "cqcc", "phase", "ssl"],
        help="特征类型",
    )
    train_parser.add_argument(
        "--loss_type",
        type=str,
        default="focal",
        choices=["focal", "ce", "class_balanced", "combined"],
        help="损失函数类型",
    )
    train_parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    
    # Predict 命令
    predict_parser = subparsers.add_parser("predict", help="推理预测")
    predict_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点")
    predict_parser.add_argument("--test_dir", type=str, required=True, help="测试音频目录")
    predict_parser.add_argument("--output_csv", type=str, default="predictions.csv", help="输出CSV")
    predict_parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    predict_parser.add_argument("--window_size", type=float, default=4.0, help="滑窗大小(秒)")
    predict_parser.add_argument("--hop_size", type=float, default=2.0, help="跳跃大小(秒)")
    predict_parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "top-k"],
        help="聚合方式",
    )
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "check":
        check_dependencies()
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
