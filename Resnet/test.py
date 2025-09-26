"""
音频真假检测模型测试脚本
快速测试模型是否能够正常工作
"""

import torch
import numpy as np
import pandas as pd
import os
from audio_resnet_model import AudioPreprocessor, AudioDataset, AudioResNet18

def test_audio_preprocessing():
    """测试音频预处理功能"""
    print("=== 测试音频预处理 ===")
    
    # 创建预处理器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        max_len=5,
        device=device
    )
    
    # 检查是否有音频文件
    test_audio_dir = 'dataset/test'
    if os.path.exists(test_audio_dir):
        audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.wav')]
        if audio_files:
            test_file = os.path.join(test_audio_dir, audio_files[0])
            print(f"测试文件: {test_file}")
            
            # 处理音频
            features = preprocessor.process_audio(test_file)
            if features is not None:
                print(f"多通道特征形状: {features.shape}")
                print("✓ 音频预处理测试通过")
                return True
            else:
                print("✗ 音频加载失败")
                return False
        else:
            print("✗ 没有找到音频文件")
            return False
    else:
        print("✗ 测试音频目录不存在")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 创建预处理器以获取输入尺寸
        preprocessor = AudioPreprocessor(max_len=5, device=device)
        channels = preprocessor.feature_channels
        time_steps = preprocessor.target_length

        # 创建模型
        model = AudioResNet18(num_classes=2, input_channels=channels)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        model = model.to(device)
        
        # 创建假输入
        batch_size = 4
        n_mels = preprocessor.n_mels
        dummy_input = torch.randn(batch_size, channels, n_mels, time_steps).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"使用设备: {device}")
        print("✓ 模型创建和前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def test_dataset():
    """测试数据集加载"""
    print("\n=== 测试数据集加载 ===")
    
    try:
        # 检查CSV文件
        train_csv = 'dataset/train.csv'
        if not os.path.exists(train_csv):
            print("✗ 训练CSV文件不存在")
            return False
        
        # 读取CSV
        df = pd.read_csv(train_csv)
        print(f"训练数据大小: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
        print(f"类别分布: {df['target'].value_counts().to_dict()}")
        
        # 创建预处理器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preprocessor = AudioPreprocessor(max_len=5, device=device)
        
        # 创建数据集（只测试前几个样本）
        small_df = df.head(5)
        small_df.to_csv('temp_test.csv', index=False)
        
        dataset = AudioDataset('temp_test.csv', 'dataset/train', preprocessor)
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试加载一个样本
        try:
            sample = dataset[0]
            if len(sample) == 2:  # 有标签
                features, label = sample
                print(f"样本0 - 特征形状: {features.shape}, 标签: {label}")
            else:  # 无标签（测试集）
                features = sample
                print(f"样本0 - 特征形状: {features.shape}")
            
            print("✓ 数据集加载测试通过")
            
            # 清理临时文件
            os.remove('temp_test.csv')
            return True
            
        except Exception as e:
            print(f"✗ 样本加载失败: {e}")
            if os.path.exists('temp_test.csv'):
                os.remove('temp_test.csv')
            return False
            
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        return False

def test_training_loop():
    """测试训练循环（不实际训练）"""
    print("\n=== 测试训练循环设置 ===")
    
    try:
        from train import AudioClassificationTrainer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preprocessor = AudioPreprocessor(max_len=5, device=device)

        # 创建模型
        model = AudioResNet18(
            num_classes=2,
            input_channels=preprocessor.feature_channels
        )
        
        # 创建训练器
        trainer = AudioClassificationTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            weight_decay=1e-4
        )
        
        print("✓ 训练器创建成功")
        print(f"设备: {device}")
        print(f"学习率: {trainer.learning_rate}")
        print(f"优化器: {type(trainer.optimizer).__name__}")
        print(f"损失函数: {type(trainer.criterion).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练循环测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试音频真假检测模型...\n")
    
    tests = [
        ("音频预处理", test_audio_preprocessing),
        ("模型创建", test_model_creation),
        ("数据集加载", test_dataset),
        ("训练循环设置", test_training_loop)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！模型可以开始训练。")
        print("\n要开始训练，请运行: conda run --name pytorch python train.py")
    else:
        print("❌ 部分测试失败，请检查错误信息并修复问题。")

if __name__ == "__main__":
    main()