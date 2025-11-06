"""
推理脚本
用于测试训练好的模型
"""

import torch
import argparse
import json
import os
import sys
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.data_loader import Vocabulary


def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def text_to_ids(text, vocab, max_len):
    """将文本转换为 ID 序列"""
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab['<PAD>']] * (max_len - len(ids))
    
    return ids


def ids_to_text(ids, vocab):
    """将 ID 序列转换为文本"""
    tokens = []
    for id in ids:
        if id == vocab['<EOS>'] or id == vocab['<PAD>']:
            break
        if id in vocab.idx2word:
            tokens.append(vocab.idx2word[id])
    return ' '.join(tokens)


def translate(model, src_text, src_vocab, tgt_vocab, device, max_len=128):
    """翻译文本"""
    model.eval()
    
    # 编码源文本
    src_ids = text_to_ids(src_text, src_vocab, max_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # 生成目标文本
    tgt_ids = [tgt_vocab['<SOS>']]
    
    with torch.no_grad():
        encoder_output = model.encoder(src_tensor)
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            decoder_output = model.decoder(tgt_tensor, encoder_output)
            output = model.fc_out(decoder_output)
            
            next_token_id = output[0, -1, :].argmax().item()
            tgt_ids.append(next_token_id)
            
            if next_token_id == tgt_vocab['<EOS>']:
                break
    
    # 转换为文本
    translated_text = ids_to_text(tgt_ids, tgt_vocab)
    return translated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer 推理脚本')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--text', type=str, required=True,
                       help='要翻译的文本')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, config = load_model(args.model, device)
    
    # 这里需要加载词汇表（实际应用中应该保存词汇表）
    print("翻译功能需要词汇表，请使用训练脚本中保存的词汇表")
    print(f"模型配置: {config}")

