from enum import auto
from bertviz import head_view, model_view
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
import torch
import numpy as np
import json

def safe_to_numpy(tensor):
    """
    安全地将PyTorch张量转换为numpy数组
    
    Args:
        tensor: PyTorch张量
    
    Returns:
        numpy数组
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

# 计算每一层所有头的权值的和和平均
def calculate_layer_attention_sums(attention_weights):
    """
    计算每一层所有头的权值的和和平均
    
    Args:
        attention_weights: 注意力权重元组，包含每一层的注意力权重
                          shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    
    Returns:
        layer_sums: 字典，包含每一层的聚合结果
    """
    layer_sums = {}
    
    print(f"总层数: {len(attention_weights)}")
    print(f"每层注意力权重形状: {attention_weights[0].shape}")
    
    for layer_idx, layer_attention in enumerate(attention_weights):
        # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
        # 移除batch维度，得到 [num_heads, seq_len, seq_len]
        layer_attention = layer_attention[0]  # 取第一个batch
        
        print(f"\n=== 第 {layer_idx + 1} 层分析 ===")
        print(f"注意力权重形状: {layer_attention.shape}")
        print(f"头数: {layer_attention.shape[0]}")
        print(f"序列长度: {layer_attention.shape[1]}")
        
        # 方法1: 计算所有头的注意力权重求和
        sum_attention = layer_attention.sum(dim=0)  # [seq_len, seq_len]
        
        # 方法2: 计算平均权值 - 用求和权值除以头的个数
        num_heads = layer_attention.shape[0]
        mean_attention = sum_attention / num_heads  # [seq_len, seq_len]
        
        # 计算统计信息
        layer_stats = {
            'sum_attention': {
                'weights': safe_to_numpy(sum_attention),
                'mean_value': float(sum_attention.mean()),
                'max_value': float(sum_attention.max()),
                'std_value': float(sum_attention.std()),
                'description': f"所有{num_heads}个头的注意力权重求和"
            },
            'mean_attention': {
                'weights': safe_to_numpy(mean_attention),
                'mean_value': float(mean_attention.mean()),
                'max_value': float(mean_attention.max()),
                'std_value': float(mean_attention.std()),
                'description': f"求和权值除以头数({num_heads})"
            }
        }
        
        layer_sums[layer_idx] = layer_stats
        
        print(f"第 {layer_idx + 1} 层统计信息:")
        print(f"  求和聚合 - 均值: {layer_stats['sum_attention']['mean_value']:.4f}, 最大值: {layer_stats['sum_attention']['max_value']:.4f}")
        print(f"  平均权值 - 均值: {layer_stats['mean_attention']['mean_value']:.4f}, 最大值: {layer_stats['mean_attention']['max_value']:.4f}")
        #print(f"  直接平均 - 均值: {layer_stats['direct_mean_attention']['mean_value']:.4f}, 最大值: {layer_stats['direct_mean_attention']['max_value']:.4f}")
        print(f"  头数: {num_heads}")
    
    return layer_sums

def save_results_to_file(layer_sums, filename="attention_analysis_results.json"):
    """保存分析结果到JSON文件"""
    results = {
        'layer_attention_sums': {}
    }
    
    for layer_idx, layer_data in layer_sums.items():
        results['layer_attention_sums'][f'layer_{layer_idx}'] = {}
        for method, data in layer_data.items():
            results['layer_attention_sums'][f'layer_{layer_idx}'][method] = {
                'mean_value': data['mean_value'],
                'max_value': data['max_value'],
                'std_value': data['std_value'],
                'description': data.get('description', '')
            }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"分析结果已保存到: {filename}")

model_path: str = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'
model = BertModel.from_pretrained(model_path, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_path)

code = "def add(a, b): return a + b "
text = "A function that adds two numbers."
inputs = tokenizer.encode_plus(text, code, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']

# CodeBERT (based on RoBERTa) doesn't use token_type_ids
# 调用模型时只传入input_ids
outputs = model(input_ids, token_type_ids=token_type_ids)
attention = outputs[-1]  # attention 是一个包含每层注意力的元组

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
head_view(attention, tokens, sentence_b_start=token_type_ids[0].tolist().index(1))

html_obj = head_view(attention, tokens, sentence_b_start=token_type_ids[0].tolist().index(1), html_action='return')
with open("attention.html", "w", encoding="utf-8") as f:
    f.write(html_obj.data)

# 执行计算
print("开始计算每一层的注意力权重聚合...")
layer_attention_sums = calculate_layer_attention_sums(attention)

print(f"\n=== 计算完成 ===")
print(f"总共分析了 {len(layer_attention_sums)} 层")


# 保存结果
save_results_to_file(layer_attention_sums)
