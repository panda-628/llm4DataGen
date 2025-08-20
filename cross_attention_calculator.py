from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

def safe_to_numpy(tensor):
    """安全地将PyTorch张量转换为numpy数组"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

class CrossAttentionCalculator:
    def __init__(self, model_path: str):
        """
        初始化交叉注意力计算器
        
        Args:
            model_path: BERT模型路径
        """
        self.model = BertModel.from_pretrained(model_path, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
    def process_inputs(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        处理两个输入文本，生成tokens和attention
        
        Args:
            text1: 第一个输入文本
            text2: 第二个输入文本
            
        Returns:
            包含处理结果的字典
        """
        # 编码输入
        inputs = self.tokenizer.encode_plus(
            text1, text2, 
            return_tensors='pt',
            add_special_tokens=True,
            return_token_type_ids=True
        )
        
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        
        # 获取tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 找到第二个句子的开始位置
        sentence_b_start = token_type_ids[0].tolist().index(1) if 1 in token_type_ids[0] else len(tokens)
        
        # 分离两个句子的tokens
        sentence_a_tokens = tokens[:sentence_b_start]
        sentence_b_tokens = tokens[sentence_b_start:]
        
        # 获取模型输出
        outputs = self.model(input_ids, token_type_ids=token_type_ids)
        attention_weights = outputs.attentions  # 每层的注意力权重
        
        return {
            'tokens': tokens,
            'sentence_a_tokens': sentence_a_tokens,
            'sentence_b_tokens': sentence_b_tokens,
            'sentence_b_start': sentence_b_start,
            'attention_weights': attention_weights,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids
        }
    
    def calculate_cross_attention_matrix(self, attention_weights: tuple, 
                                       sentence_b_start: int, 
                                       tokens: List[str],
                                       layer_idx: int = -1) -> Dict[str, np.ndarray]:
        """
        计算交叉注意力矩阵（两个输入之间的词对词权重）
        
        Args:
            attention_weights: 注意力权重元组
            sentence_b_start: 第二个句子开始的位置
            tokens: 所有tokens列表
            layer_idx: 要分析的层索引，-1表示最后一层
            
        Returns:
            包含不同聚合方式的交叉注意力矩阵的字典
        """
        if layer_idx == -1:
            layer_idx = len(attention_weights) - 1
            
        # 获取指定层的注意力权重 [batch_size, num_heads, seq_len, seq_len]
        layer_attention = attention_weights[layer_idx][0]  # 取第一个batch
        
        seq_len = layer_attention.shape[-1]
        sentence_a_len = sentence_b_start
        sentence_b_len = seq_len - sentence_b_start
        
        # 提取交叉注意力部分
        # A->B: sentence A的tokens对sentence B的tokens的注意力
        cross_attention_a_to_b = layer_attention[:, :sentence_a_len, sentence_b_start:]
        # B->A: sentence B的tokens对sentence A的tokens的注意力  
        cross_attention_b_to_a = layer_attention[:, sentence_b_start:, :sentence_a_len]
        
        # 不同的聚合方式
        results = {}
        
        # 1. 平均所有头 (最常用)
        results['mean_heads_a_to_b'] = safe_to_numpy(cross_attention_a_to_b.mean(dim=0))
        results['mean_heads_b_to_a'] = safe_to_numpy(cross_attention_b_to_a.mean(dim=0))
        
        # 2. 求和所有头
        results['sum_heads_a_to_b'] = safe_to_numpy(cross_attention_a_to_b.sum(dim=0))
        results['sum_heads_b_to_a'] = safe_to_numpy(cross_attention_b_to_a.sum(dim=0))
        
        # 3. 最大值聚合
        results['max_heads_a_to_b'] = safe_to_numpy(cross_attention_a_to_b.max(dim=0)[0])
        results['max_heads_b_to_a'] = safe_to_numpy(cross_attention_b_to_a.max(dim=0)[0])
        
        return results
    
    def analyze_all_layers_cross_attention(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        分析所有层的交叉注意力
        
        Args:
            text1: 第一个输入文本
            text2: 第二个输入文本
            
        Returns:
            完整的分析结果
        """
        # 处理输入
        processed = self.process_inputs(text1, text2)
        
        results = {
            'input_texts': {'text1': text1, 'text2': text2},
            'tokens': {
                'all_tokens': processed['tokens'],
                'sentence_a_tokens': processed['sentence_a_tokens'],
                'sentence_b_tokens': processed['sentence_b_tokens'],
                'sentence_b_start': processed['sentence_b_start']
            },
            'layers_analysis': {}
        }
        
        # 分析每一层
        for layer_idx in range(len(processed['attention_weights'])):
            cross_attention = self.calculate_cross_attention_matrix(
                processed['attention_weights'],
                processed['sentence_b_start'],
                processed['tokens'],
                layer_idx
            )
            
            results['layers_analysis'][f'layer_{layer_idx}'] = cross_attention
            
        return results
    
    def get_word_to_word_weights(self, text1: str, text2: str, 
                               layer_idx: int = -1, 
                               aggregation: str = 'mean_heads') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取词对词的权重，以DataFrame格式返回
        
        Args:
            text1: 第一个输入文本
            text2: 第二个输入文本
            layer_idx: 层索引，-1表示最后一层
            aggregation: 聚合方式 ('mean_heads', 'sum_heads', 'max_heads')
            
        Returns:
            (a_to_b_df, b_to_a_df): 两个DataFrame，分别表示A->B和B->A的权重
        """
        # 获取分析结果
        processed = self.process_inputs(text1, text2)
        
        if layer_idx == -1:
            layer_idx = len(processed['attention_weights']) - 1
            
        cross_attention = self.calculate_cross_attention_matrix(
            processed['attention_weights'],
            processed['sentence_b_start'],
            processed['tokens'],
            layer_idx
        )
        
        # 创建DataFrame
        sentence_a_tokens = processed['sentence_a_tokens']
        sentence_b_tokens = processed['sentence_b_tokens']
        
        # A->B权重矩阵
        a_to_b_matrix = cross_attention[f'{aggregation}_a_to_b']
        a_to_b_df = pd.DataFrame(
            a_to_b_matrix,
            index=sentence_a_tokens,
            columns=sentence_b_tokens
        )
        
        # B->A权重矩阵
        b_to_a_matrix = cross_attention[f'{aggregation}_b_to_a']
        b_to_a_df = pd.DataFrame(
            b_to_a_matrix,
            index=sentence_b_tokens,
            columns=sentence_a_tokens
        )
        
        return a_to_b_df, b_to_a_df
    
    def visualize_cross_attention(self, text1: str, text2: str, 
                                layer_idx: int = -1, 
                                aggregation: str = 'mean_heads',
                                save_path: str = None) -> None:
        """
        可视化交叉注意力权重
        
        Args:
            text1: 第一个输入文本
            text2: 第二个输入文本
            layer_idx: 层索引
            aggregation: 聚合方式
            save_path: 保存路径，如果为None则不保存
        """
        a_to_b_df, b_to_a_df = self.get_word_to_word_weights(
            text1, text2, layer_idx, aggregation
        )
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 绘制A->B注意力热图
        sns.heatmap(a_to_b_df, annot=True, fmt='.3f', cmap='Blues', ax=ax1)
        ax1.set_title(f'Text1 -> Text2 Attention (Layer {layer_idx}, {aggregation})')
        ax1.set_xlabel('Text2 Tokens')
        ax1.set_ylabel('Text1 Tokens')
        
        # 绘制B->A注意力热图
        sns.heatmap(b_to_a_df, annot=True, fmt='.3f', cmap='Oranges', ax=ax2)
        ax2.set_title(f'Text2 -> Text1 Attention (Layer {layer_idx}, {aggregation})')
        ax2.set_xlabel('Text1 Tokens')
        ax2.set_ylabel('Text2 Tokens')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def save_detailed_results(self, text1: str, text2: str, 
                            filename: str = "cross_attention_results.json") -> None:
        """
        保存详细的交叉注意力分析结果
        
        Args:
            text1: 第一个输入文本
            text2: 第二个输入文本
            filename: 保存文件名
        """
        results = self.analyze_all_layers_cross_attention(text1, text2)
        
        # 转换numpy数组为列表以便JSON序列化
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy_to_list(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"详细分析结果已保存到: {filename}")


# 示例使用
if __name__ == "__main__":
    # 初始化计算器
    model_path = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'
    calculator = CrossAttentionCalculator(model_path)
    
    # 示例文本
    text1 = "One Doctor is associated with multiple Requisitions."
    text2 = "A doctor may also indicate that the tests on a requisition are to be repeated for a specified number of times and interval."
    
    print("=== 交叉注意力分析 ===")
    print(f"Text1: {text1}")
    print(f"Text2: {text2}")
    print()
    
    # 获取词对词权重（最后一层，平均聚合）
    a_to_b_df, b_to_a_df = calculator.get_word_to_word_weights(text1, text2)
    
    print("Text1 -> Text2 注意力权重:")
    print(a_to_b_df)
    print("\nText2 -> Text1 注意力权重:")
    print(b_to_a_df)
    
    # 保存详细结果
    calculator.save_detailed_results(text1, text2)
    
    # 可视化
    calculator.visualize_cross_attention(text1, text2, save_path="cross_attention_visualization.png")
