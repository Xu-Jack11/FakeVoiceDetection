"""
音频真假检测训练和评估主程序
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from audio_resnet_model import (
    AudioPreprocessor, AudioDataset, AudioResNet18, AudioResNet34, AudioResNet50
)

class AudioClassificationTrainer:
    """音频分类训练器"""
    #学习率
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-4):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 记录训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for data, target in val_bar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, save_path='best_model.pth'):
        """完整训练流程"""
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        max_patience = 10
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 打印结果
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_preds': val_preds,
                    'val_targets': val_targets
                }, save_path)
                print(f'新的最佳模型已保存! 验证准确率: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= max_patience:
                print(f'早停触发! 已连续{max_patience}个epoch无改善')
                break
        
        print(f'\n训练完成!')
        print(f'最佳验证准确率: {best_val_acc:.2f}%')
        print(f'最佳验证损失: {best_val_loss:.4f}')
        
        return best_val_acc, best_val_loss
    
    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing')
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['AI Generated', 'Real Human'],
                yticklabels=['AI Generated', 'Real Human'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def predict_test_set(model, test_csv, test_audio_dir, preprocessor, device='cuda', batch_size=32):
    """对测试集进行预测"""
    # 创建测试数据集（无标签）
    test_dataset = AudioDataset(test_csv, test_audio_dir, preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    predictions = []
    audio_names = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Predicting')
        for data, names in test_bar:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            if isinstance(names, (list, tuple)):
                audio_names.extend(list(names))
            else:
                audio_names.extend(names.tolist())
    
    # 创建提交文件
    submission = pd.DataFrame({
        'audio_name': audio_names,
        'target': predictions
    })
    
    return submission

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 启用 cuDNN 自动调优以加速卷积操作（仅在使用 GPU 时有效）
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    
    # 数据路径
    train_csv = 'dataset/train.csv'
    test_csv = 'dataset/test.csv'
    train_audio_dir = 'dataset/train'
    test_audio_dir = 'dataset/test'
    
    # 音频预处理器
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        max_len=5,  # 5秒音频
        device=device
    )
    
    # 加载训练数据并划分训练/验证集
    train_df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['target']
    )
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"类别分布 - 训练集: {train_df['target'].value_counts().to_dict()}")
    print(f"类别分布 - 验证集: {val_df['target'].value_counts().to_dict()}")
    
    # 保存临时CSV文件
    train_df.to_csv('temp_train.csv', index=False)
    val_df.to_csv('temp_val.csv', index=False)
    
    # 创建数据集
    train_dataset = AudioDataset('temp_train.csv', train_audio_dir, preprocessor)
    val_dataset = AudioDataset('temp_val.csv', train_audio_dir, preprocessor)
    
    # 创建数据加载器
    batch_size = 64  # 根据GPU内存调整
    # DataLoader 优化：多进程加载、预取、固定内存
    num_workers = 12
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # 创建模型（可以选择不同的ResNet变体）
    model = AudioResNet50(num_classes=2)  # 也可以尝试AudioResNet34, AudioResNet50
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = AudioClassificationTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    # 开始训练
    best_val_acc, best_val_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        save_path='best_audio_model.pth'
    )
    
    # 绘制训练历史
    trainer.plot_training_history('training_history.png')
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load('best_audio_model.pth',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在验证集上进行详细评估
    print("\n=== 验证集详细评估 ===")
    val_preds, val_targets, val_probs = evaluate_model(model, val_loader, device)
    
    # 计算指标
    accuracy = accuracy_score(val_targets, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='weighted')
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(val_targets, val_preds, target_names=['AI Generated', 'Real Human']))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(val_targets, val_preds, 'confusion_matrix.png')
    
    # 对测试集进行预测
    print("\n=== 测试集预测 ===")
    submission = predict_test_set(
        model=model,
        test_csv=test_csv,
        test_audio_dir=test_audio_dir,
        preprocessor=preprocessor,
        device=device,
        batch_size=batch_size
    )
    
    # 保存预测结果
    submission.to_csv('submission.csv', index=False)
    print("预测结果已保存到 'submission.csv'")
    
    # 清理临时文件
    os.remove('temp_train.csv')
    os.remove('temp_val.csv')
    
    print("\n训练和评估完成!")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()