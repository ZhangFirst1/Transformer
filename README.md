## 快速开始

**注意：**项目中的数据集压缩包`data.zip`只有原数据集的1/15，训练的话请到原链接下载完整数据集。
消融实验结果在results下

https://huggingface.co/datasets/neulab/ted_multi

### 1. 环境要求

- Python >= 3.8
- CUDA >= 11.0 (可选，用于 GPU 加速)
- 至少 4GB RAM
- 至少 2GB 磁盘空间

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行训练

#### 标准训练

```bash
# Linux/Mac
bash scripts/run.sh train

# Windows PowerShell
python src/train.py --mode train

# Windows CMD
scripts\run.bat train
```

#### 消融实验

```bash
# Linux/Mac
bash scripts/run.sh ablation

# Windows PowerShell
python src/train.py --mode ablation

# Windows CMD
scripts\run.bat ablation
```

## 重现实验的精确命令

### 标准训练（随机种子：42）

```bash
python src/train.py --mode train --config configs/default.json
```

如果没有配置文件，使用默认配置：

```bash
python src/train.py --mode train
```

### 消融实验（随机种子：42）

```bash
python src/train.py --mode ablation
```

### 使用自定义配置

创建 JSON 配置文件（例如 `configs/my_config.json`）：

```json
{
    "experiment_name": "my_experiment",
    "data_dir": "data",
    "lang_pair": "en-zh",
    "batch_size": 32,
    "max_len": 128,
    "max_samples": 1000,
    "min_freq": 2,
    "d_model": 512,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1,
    "learning_rate": 0.0001,
    "lr_step": 10,
    "lr_gamma": 0.5,
    "num_epochs": 20,
    "seed": 42
}
```

然后运行：

```bash
python src/train.py --mode train --config configs/my_config.json
```

## 模型架构

### Transformer 组件

1. **MultiHeadAttention**: 多头自注意力机制
   - 支持缩放点积注意力
   - 可配置注意力头数

2. **PositionalEncoding**: 位置编码
   - 正弦/余弦位置编码
   - 支持最大序列长度配置

3. **EncoderBlock**: Encoder 块
   - 自注意力层
   - 前馈神经网络
   - 残差连接和层归一化

4. **DecoderBlock**: Decoder 块
   - 自注意力层（带因果掩码）
   - 交叉注意力层
   - 前馈神经网络
   - 残差连接和层归一化

5. **Transformer**: 完整模型
   - Encoder-Decoder 架构
   - 可配置层数、维度等超参数

### 关键实现片段

#### MultiHeadAttention

```python
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

#### EncoderBlock

```python
def forward(self, x, mask=None):
    # Self-attention with residual connection
    attn_output = self.self_attn(x, x, x, mask)
    x = self.norm1(x + self.dropout1(attn_output))
    
    # Feed-forward with residual connection
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout2(ff_output))
    return x
```

#### DecoderBlock

```python
def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
    # Self-attention
    attn_output = self.self_attn(x, x, x, tgt_mask)
    x = self.norm1(x + self.dropout1(attn_output))
    
    # Cross-attention
    cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
    x = self.norm2(x + self.dropout2(cross_attn_output))
    
    # Feed-forward
    ff_output = self.feed_forward(x)
    x = self.norm3(x + self.dropout3(ff_output))
    return x
```

## 实验设置

### 数据集

- **数据集**: TED Talks (Multi-lingual) 或自定义示例数据
- **语言对**: 英文-中文
- **数据量**: 默认 1000 条样本
- **序列长度**: 最大 128 tokens

### 默认超参数

- **模型维度 (d_model)**: 512
- **注意力头数 (num_heads)**: 8
- **Encoder 层数**: 6
- **Decoder 层数**: 6
- **前馈网络维度 (d_ff)**: 2048
- **Dropout**: 0.1
- **学习率**: 0.0001
- **批次大小**: 32
- **训练轮数**: 20
- **随机种子**: 42

### 消融实验设置

消融实验包括以下变体：

1. **baseline**: 标准配置
2. **fewer_layers**: 减少层数（3层 Encoder/Decoder）
3. **fewer_heads**: 减少注意力头数（4头）
4. **smaller_model**: 更小的模型（d_model=256, d_ff=1024）
5. **higher_dropout**: 更高的 Dropout（0.3）

## 结果

训练完成后，结果将保存在 `results/` 目录：

- `results/{experiment_name}/training_curves.png`: 训练曲线图
- `results/{experiment_name}/best_model.pt`: 最佳模型
- `results/{experiment_name}/history.json`: 训练历史
- `results/{experiment_name}/tensorboard/`: TensorBoard 日志
- `results/ablation_results.json`: 消融实验结果（JSON）
- `results/ablation_results.csv`: 消融实验结果（CSV）

### 查看训练曲线

训练曲线会自动保存为 PNG 图片。也可以使用 TensorBoard 查看：

```bash
tensorboard --logdir results/{experiment_name}/tensorboard
```

## 硬件要求


### 推荐配置
- CPU: 8 核或更多
- RAM: 8GB 或更多
- GPU: NVIDIA GPU with CUDA support
- 磁盘: 5GB+

### 训练时间估算

- **CPU**: 约 30-60 分钟/epoch
- **GPU**: 约 1-5 分钟/epoch


## 代码说明

### 核心文件

- **src/transformer.py**: 包含所有 Transformer 组件的实现
- **src/data_loader.py**: 数据加载、预处理和词汇表构建
- **src/train.py**: 训练循环、验证和消融实验
- **src/inference.py**: 模型推理（翻译）
