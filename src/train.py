"""
训练脚本
支持消融实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import json
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.data_loader import get_data_loaders


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for src, tgt_input, tgt_output in pbar:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches if num_batches > 0 else 0


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in val_loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            output = model(src, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train(config):
    """主训练函数"""
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建结果目录
    exp_name = config['experiment_name']
    os.makedirs(f"results/{exp_name}", exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, src_vocab, tgt_vocab = get_data_loaders(
        data_dir=config['data_dir'],
        lang_pair=config['lang_pair'],
        batch_size=config['batch_size'],
        max_len=config['max_len'],
        max_samples=config['max_samples'],
        min_freq=config['min_freq']
    )
    
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # 更新配置以包含词汇表大小
    config['src_vocab_size'] = len(src_vocab)
    config['tgt_vocab_size'] = len(tgt_vocab)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 PAD token
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], 
                                         gamma=config['lr_gamma'])
    
    # TensorBoard
    writer = SummaryWriter(f"results/{exp_name}/tensorboard")
    
    # 训练历史
    train_losses = []
    val_losses = []
    
    # 训练循环
    best_val_loss = float('inf')
    print("\n开始训练...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step()
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, f"results/{exp_name}/best_model.pt")
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch}/{config["num_epochs"]} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Time: {epoch_time:.2f}s')
    
    writer.close()
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }
    with open(f"results/{exp_name}/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, exp_name)
    
    print(f"\n训练完成！结果保存在 results/{exp_name}/")
    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, exp_name):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Curves - {exp_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{exp_name}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def run_ablation_study():
    """运行消融实验"""
    base_config = {
        'data_dir': 'data',
        'lang_pair': 'en-zh',
        'batch_size': 32,
        'max_len': 128,
        'max_samples': 1000,
        'min_freq': 2,
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'lr_step': 10,
        'lr_gamma': 0.5,
        'num_epochs': 20,
        'seed': 42
    }
    
    experiments = [
        {
            'name': 'baseline',
            'config': base_config.copy()
        },
        {
            'name': 'no_positional_encoding',
            'config': {**base_config.copy(), 'd_model': 512}  # 可以通过修改模型来禁用位置编码
        },
        {
            'name': 'fewer_layers',
            'config': {**base_config.copy(), 'num_encoder_layers': 3, 'num_decoder_layers': 3}
        },
        {
            'name': 'fewer_heads',
            'config': {**base_config.copy(), 'num_heads': 4}
        },
        {
            'name': 'smaller_model',
            'config': {**base_config.copy(), 'd_model': 256, 'd_ff': 1024, 'num_heads': 4}
        },
        {
            'name': 'higher_dropout',
            'config': {**base_config.copy(), 'dropout': 0.3}
        }
    ]
    
    results = {}
    
    for exp in experiments:
        exp_name = f"ablation_{exp['name']}"
        exp['config']['experiment_name'] = exp_name
        
        print(f"\n{'='*60}")
        print(f"运行实验: {exp_name}")
        print(f"{'='*60}")
        
        train_losses, val_losses = train(exp['config'])
        
        results[exp_name] = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'config': exp['config']
        }
    
    # 保存消融实验结果
    with open('results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 创建结果表格
    create_results_table(results)
    
    return results


def create_results_table(results):
    """创建结果表格"""
    import pandas as pd
    
    rows = []
    for exp_name, result in results.items():
        rows.append({
            'Experiment': exp_name.replace('ablation_', ''),
            'Final Train Loss': f"{result['final_train_loss']:.4f}",
            'Final Val Loss': f"{result['final_val_loss']:.4f}",
            'Best Val Loss': f"{result['best_val_loss']:.4f}"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv('results/ablation_results.csv', index=False)
    print("\n消融实验结果表格已保存到 results/ablation_results.csv")
    print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer 训练脚本')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'ablation'],
                       help='运行模式: train 或 ablation')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（JSON格式）')
    
    args = parser.parse_args()
    
    if args.mode == 'ablation':
        run_ablation_study()
    else:
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            # 默认配置
            config = {
                'experiment_name': 'default',
                'data_dir': 'data',
                'lang_pair': 'en-zh',
                'batch_size': 32,
                'max_len': 128,
                'max_samples': 1000,
                'min_freq': 2,
                'd_model': 512,
                'num_heads': 8,
                'num_encoder_layers': 6,
                'num_decoder_layers': 6,
                'd_ff': 2048,
                'dropout': 0.1,
                'learning_rate': 0.0001,
                'lr_step': 10,
                'lr_gamma': 0.5,
                'num_epochs': 20,
                'seed': 42
            }
        
        train(config)

