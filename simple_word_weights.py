"""
简化版词对词权重计算
专门计算两个输入之间每个词与词的权重关系
"""

from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np

def calculate_word_to_word_weights(text1, text2, model_path):
    """
    计算两个输入之间每个词与词的权重
    
    Args:
        text1 (str): 第一个输入文本
        text2 (str): 第二个输入文本  
        model_path (str): BERT模型路径
        
    Returns:
        tuple: (text1_to_text2_weights, text2_to_text1_weights) 两个DataFrame
    """
    
    # 加载模型和分词器
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    print(f"输入1: {text1}")
    print(f"输入2: {text2}")
    print("-" * 50)
    
    # 编码输入
    inputs = tokenizer.encode_plus(
        text1, text2,
        return_tensors='pt',
        add_special_tokens=True,
        return_token_type_ids=True
    )
    
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    
    # 获取所有tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"所有tokens: {tokens}")
    
    # 找到第二个句子的开始位置
    sentence_b_start = token_type_ids[0].tolist().index(1) if 1 in token_type_ids[0] else len(tokens)
    
    # 分离两个句子的tokens
    text1_tokens = tokens[:sentence_b_start]
    text2_tokens = tokens[sentence_b_start:]
    
    print(f"文本1的tokens: {text1_tokens}")
    print(f"文本2的tokens: {text2_tokens}")
    print(f"文本2开始位置: {sentence_b_start}")
    print("-" * 50)
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids)
        attention_weights = outputs.attentions
    
    # 使用最后一层的注意力权重
    last_layer_attention = attention_weights[-1][0]  # [num_heads, seq_len, seq_len]
    
    # 平均所有注意力头
    avg_attention = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]
    
    # 提取交叉注意力
    # 文本1 -> 文本2 的注意力
    text1_to_text2 = avg_attention[:sentence_b_start, sentence_b_start:]
    
    # 文本2 -> 文本1 的注意力  
    text2_to_text1 = avg_attention[sentence_b_start:, :sentence_b_start]
    
    # 转换为DataFrame
    text1_to_text2_df = pd.DataFrame(
        text1_to_text2.detach().cpu().numpy(),
        index=text1_tokens,
        columns=text2_tokens
    )
    
    text2_to_text1_df = pd.DataFrame(
        text2_to_text1.detach().cpu().numpy(),
        index=text2_tokens,
        columns=text1_tokens
    )
    
    return text1_to_text2_df, text2_to_text1_df

def print_top_weights(df, title, top_n=5):
    """打印权重最高的词对"""
    print(f"\n{title} - 权重最高的{top_n}个词对:")
    
    # 将DataFrame转换为词对和权重的列表
    word_pairs = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            word_pairs.append((df.index[i], df.columns[j], df.iloc[i, j]))
    
    # 按权重排序
    word_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 打印前N个
    for k, (word1, word2, weight) in enumerate(word_pairs[:top_n], 1):
        print(f"  {k}. '{word1}' -> '{word2}': {weight:.4f}")

# 主程序
if __name__ == "__main__":
    # 模型路径
    model_path = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'
    
    # 两个输入
    text1 = "One Doctor is associated with multiple Requisitions."
    text2 = "A doctor may also indicate that the tests on a requisition are to be repeated for a specified number of times and interval."
    
    print("=== 计算两个输入之间的词对词权重 ===\n")
    
    # 计算权重
    text1_to_text2_weights, text2_to_text1_weights = calculate_word_to_word_weights(
        text1, text2, model_path
    )
    
    # 显示完整的权重矩阵
    print("\n=== 文本1 -> 文本2 权重矩阵 ===")
    print(text1_to_text2_weights.round(4))
    
    print("\n=== 文本2 -> 文本1 权重矩阵 ===")
    print(text2_to_text1_weights.round(4))
    
    # 显示权重最高的词对
    print_top_weights(text1_to_text2_weights, "文本1 -> 文本2")
    print_top_weights(text2_to_text1_weights, "文本2 -> 文本1")
    
    # 保存到CSV文件
    text1_to_text2_weights.to_csv("text1_to_text2_weights.csv")
    text2_to_text1_weights.to_csv("text2_to_text1_weights.csv")
    
    print(f"\n权重矩阵已保存到:")
    print(f"  - text1_to_text2_weights.csv")
    print(f"  - text2_to_text1_weights.csv")
    
    # 统计信息
    print(f"\n=== 统计信息 ===")
    print(f"文本1共有 {len(text1_to_text2_weights.index)} 个tokens")
    print(f"文本2共有 {len(text1_to_text2_weights.columns)} 个tokens")
    print(f"总共 {text1_to_text2_weights.size} 个权重值（文本1->文本2）")
    print(f"总共 {text2_to_text1_weights.size} 个权重值（文本2->文本1）")
    
    print(f"文本1->文本2 平均权重: {text1_to_text2_weights.mean().mean():.4f}")
    print(f"文本2->文本1 平均权重: {text2_to_text1_weights.mean().mean():.4f}")
    print(f"文本1->文本2 最高权重: {text1_to_text2_weights.max().max():.4f}")
    print(f"文本2->文本1 最高权重: {text2_to_text1_weights.max().max():.4f}")
