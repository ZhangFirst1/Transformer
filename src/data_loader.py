"""
数据加载和预处理模块
支持 TED Talks 多语言数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
from collections import Counter
import re


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, data, src_vocab, tgt_vocab, max_len=128):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # 转换为索引序列
        src_ids = self.text_to_ids(src_text, self.src_vocab, self.max_len)
        tgt_ids = self.text_to_ids(tgt_text, self.tgt_vocab, self.max_len)
        
        # 为 teacher forcing 准备输入和目标
        tgt_input = tgt_ids[:-1]
        tgt_output = tgt_ids[1:]
        
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_input, dtype=torch.long), \
               torch.tensor(tgt_output, dtype=torch.long)
    
    def text_to_ids(self, text, vocab, max_len):
        """将文本转换为 ID 序列"""
        # 简单的分词（按空格和标点）
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        ids = [vocab[token] for token in tokens]
        
        # 截断或填充
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [vocab['<PAD>']] * (max_len - len(ids))
        
        return ids


class Vocabulary:
    """词汇表"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = Counter()
    
    def build_vocab(self, texts, min_freq=2):
        """构建词汇表"""
        for text in texts:
            tokens = re.findall(r'\w+|[^\w\s]', text.lower())
            self.word_count.update(tokens)
        
        # 添加频率 >= min_freq 的词
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)
    
    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx['<UNK>'])


def load_tsv_data(tsv_file, src_lang='en', tgt_lang='zh-cn', max_samples=None):
    """
    加载 TSV 格式的 TED Talks 数据集
    
    Args:
        tsv_file: TSV文件路径
        src_lang: 源语言列名（默认'en'）
        tgt_lang: 目标语言列名（默认'zh-cn'）
        max_samples: 最大样本数，None表示加载所有数据
    
    Returns:
        list: (源文本, 目标文本) 元组列表
    """
    pairs = []
    
    if not os.path.exists(tsv_file):
        print(f"数据文件不存在: {tsv_file}")
        return pairs
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            src_text = row.get(src_lang, '').strip()
            tgt_text = row.get(tgt_lang, '').strip()
            
            # 只添加两个字段都不为空的行
            if src_text and tgt_text:
                pairs.append((src_text, tgt_text))
            
            # 如果设置了最大样本数，达到后停止
            if max_samples and len(pairs) >= max_samples:
                break
    
    return pairs


def load_ted_talks_data(data_dir='data', lang_pair='en-zh', max_samples=1000):
    """
    加载 TED Talks 数据集（兼容旧格式）
    如果没有数据文件，创建示例数据用于演示
    """
    data_file = os.path.join(data_dir, f'ted_talks_{lang_pair}.json')
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("创建示例数据用于演示...")
        return create_sample_data(max_samples)
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取源语言和目标语言文本
    pairs = []
    for item in data[:max_samples]:
        if 'source' in item and 'target' in item:
            pairs.append((item['source'], item['target']))
    
    return pairs


def create_sample_data(num_samples=1000):
    """创建示例数据用于演示"""
    # 简单的英文到中文翻译示例
    sample_pairs = [
        ("Hello world", "你好世界"),
        ("How are you", "你好吗"),
        ("Thank you very much", "非常感谢"),
        ("I love programming", "我喜欢编程"),
        ("Machine learning is interesting", "机器学习很有趣"),
        ("Natural language processing", "自然语言处理"),
        ("Deep learning models", "深度学习模型"),
        ("Artificial intelligence", "人工智能"),
        ("Neural networks", "神经网络"),
        ("Computer vision", "计算机视觉"),
    ]
    
    # 扩展数据
    pairs = []
    for i in range(num_samples):
        pairs.append(sample_pairs[i % len(sample_pairs)])
    
    return pairs


def prepare_data(train_pairs, val_pairs=None, min_freq=2):
    """
    准备训练和验证数据
    
    Args:
        train_pairs: 训练数据对列表
        val_pairs: 验证数据对列表，如果为None则从train_pairs中分割
        min_freq: 词汇最小频率
    
    Returns:
        train_pairs, val_pairs, src_vocab, tgt_vocab
    """
    # 如果验证集为空，从训练集中分割
    if val_pairs is None:
        split_idx = int(len(train_pairs) * 0.8)
        val_pairs = train_pairs[split_idx:]
        train_pairs = train_pairs[:split_idx]
    
    # 分离源语言和目标语言文本（使用训练集构建词汇表）
    src_texts = [pair[0] for pair in train_pairs]
    tgt_texts = [pair[1] for pair in train_pairs]
    
    # 构建词汇表
    src_vocab = Vocabulary()
    src_vocab.build_vocab(src_texts, min_freq)
    
    tgt_vocab = Vocabulary()
    tgt_vocab.build_vocab(tgt_texts, min_freq)
    
    return train_pairs, val_pairs, src_vocab, tgt_vocab


def get_data_loaders(data_dir='data', train_file='all_talks_train.tsv', 
                     test_file='all_talks_test.tsv', src_lang='en', tgt_lang='zh-cn',
                     batch_size=32, max_len=128, max_samples=None, min_freq=2,
                     lang_pair=None):
    """
    获取数据加载器
    
    Args:
        data_dir: 数据目录
        train_file: 训练集TSV文件名（默认'all_talks_train.tsv'）
        test_file: 测试集TSV文件名（默认'all_talks_test.tsv'）
        src_lang: 源语言列名（默认'en'）
        tgt_lang: 目标语言列名（默认'zh-cn'）
        batch_size: 批次大小
        max_len: 最大序列长度
        max_samples: 最大样本数（None表示加载所有数据）
        min_freq: 词汇最小频率
        lang_pair: 语言对（向后兼容参数，如果TSV文件不存在时使用）
    
    Returns:
        train_loader, val_loader, src_vocab, tgt_vocab
    """
    # 加载训练集和测试集
    train_tsv = os.path.join(data_dir, train_file)
    test_tsv = os.path.join(data_dir, test_file)
    
    print(f"加载训练集: {train_tsv}")
    train_pairs = load_tsv_data(train_tsv, src_lang, tgt_lang, max_samples)
    print(f"训练集样本数: {len(train_pairs)}")
    
    print(f"加载测试集: {test_tsv}")
    test_pairs = load_tsv_data(test_tsv, src_lang, tgt_lang, max_samples)
    print(f"测试集样本数: {len(test_pairs)}")
    
    # 如果TSV文件不存在，回退到旧的数据加载方式
    if not train_pairs and not test_pairs:
        print("TSV文件不存在，尝试使用旧的数据加载方式...")
        lang_pair = lang_pair or 'en-zh'
        data_pairs = load_ted_talks_data(data_dir, lang_pair, max_samples or 1000)
        train_pairs, test_pairs, src_vocab, tgt_vocab = prepare_data(data_pairs, min_freq=min_freq)
    else:
        # 准备数据（使用训练集构建词汇表）
        train_pairs, test_pairs, src_vocab, tgt_vocab = prepare_data(
            train_pairs, test_pairs, min_freq=min_freq
        )
    
    # 创建数据集
    train_dataset = TextDataset(train_pairs, src_vocab, tgt_vocab, max_len)
    val_dataset = TextDataset(test_pairs, src_vocab, tgt_vocab, max_len)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, src_vocab, tgt_vocab

