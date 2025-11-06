#!/bin/bash

# Transformer 训练和消融实验运行脚本

set -e

# 设置随机种子（确保可重现）
export PYTHONHASHSEED=42

# 创建必要的目录
mkdir -p data
mkdir -p results

echo "=========================================="
echo "Transformer 训练脚本"
echo "=========================================="

# 检查参数
MODE=${1:-train}

if [ "$MODE" == "train" ]; then
    echo "运行标准训练..."
    python src/train.py --mode train
    
elif [ "$MODE" == "ablation" ]; then
    echo "运行消融实验..."
    python src/train.py --mode ablation
    
else
    echo "用法: ./scripts/run.sh [train|ablation]"
    exit 1
fi

echo "=========================================="
echo "训练完成！"
echo "结果保存在 results/ 目录"
echo "=========================================="

