"""
处理子词（subword）权重聚合的工具类
当词被tokenizer拆分成多个子词时，需要合理地聚合这些子词的权重
"""

import pandas as pd
import numpy as np
from transformers import BertTokenizer
from typing import Dict, List, Tuple, Any
import re

class SubwordWeightHandler:
    def __init__(self, tokenizer: BertTokenizer):
        """
        初始化子词权重处理器
        
        Args:
            tokenizer: BERT tokenizer
        """
        self.tokenizer = tokenizer
    
    def identify_subwords(self, tokens: List[str]) -> Dict[str, List[int]]:
        """
        识别哪些tokens属于同一个原始词
        
        Args:
            tokens: token列表
            
        Returns:
            {original_word: [token_indices]} 映射
        """
        word_to_indices = {}
        current_word = ""
        current_indices = []
        
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                # 这是一个子词片段
                current_indices.append(i)
            else:
                # 保存之前的词（如果存在）
                if current_word and current_indices:
                    word_to_indices[current_word] = current_indices.copy()
                
                # 开始新词
                current_word = token
                current_indices = [i]
        
        # 保存最后一个词
        if current_word and current_indices:
            word_to_indices[current_word] = current_indices.copy()
        
        return word_to_indices
    
    def reconstruct_words(self, tokens: List[str]) -> Dict[str, List[int]]:
        """
        重构原始词及其对应的token索引
        
        Args:
            tokens: token列表
            
        Returns:
            {reconstructed_word: [token_indices]} 映射
        """
        word_groups = {}
        current_word_tokens = []
        current_indices = []
        
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                # 子词片段，添加到当前词
                current_word_tokens.append(token[2:])  # 移除##前缀
                current_indices.append(i)
            else:
                # 完成之前的词
                if current_word_tokens:
                    reconstructed = ''.join(current_word_tokens)
                    word_groups[reconstructed] = current_indices.copy()
                
                # 开始新词
                current_word_tokens = [token]
                current_indices = [i]
        
        # 处理最后一个词
        if current_word_tokens:
            reconstructed = ''.join(current_word_tokens)
            word_groups[reconstructed] = current_indices.copy()
        
        return word_groups
    
    def aggregate_subword_weights(self, 
                                attention_matrix: pd.DataFrame,
                                method: str = "mean") -> pd.DataFrame:
        """
        聚合子词的注意力权重
        
        Args:
            attention_matrix: 原始注意力权重矩阵
            method: 聚合方法 ("mean", "max", "sum", "first", "last")
            
        Returns:
            聚合后的权重矩阵
        """
        # 获取行和列的tokens
        row_tokens = attention_matrix.index.tolist()
        col_tokens = attention_matrix.columns.tolist()
        
        # 重构原始词
        row_word_groups = self.reconstruct_words(row_tokens)
        col_word_groups = self.reconstruct_words(col_tokens)
        
        print(f"行token重构: {len(row_tokens)} -> {len(row_word_groups)} 词")
        print(f"列token重构: {len(col_tokens)} -> {len(col_word_groups)} 词")
        
        # 创建新的聚合矩阵
        new_matrix = []
        new_row_labels = []
        new_col_labels = list(col_word_groups.keys())
        
        for row_word, row_indices in row_word_groups.items():
            new_row_labels.append(row_word)
            row_weights = []
            
            for col_word, col_indices in col_word_groups.items():
                # 提取子矩阵 (row_indices x col_indices)
                submatrix = attention_matrix.iloc[row_indices, col_indices]
                
                # 根据方法聚合权重
                if method == "mean":
                    aggregated_weight = submatrix.values.mean()
                elif method == "max":
                    aggregated_weight = submatrix.values.max()
                elif method == "sum":
                    aggregated_weight = submatrix.values.sum()
                elif method == "first":
                    aggregated_weight = submatrix.iloc[0, 0]
                elif method == "last":
                    aggregated_weight = submatrix.iloc[-1, -1]
                else:
                    raise ValueError(f"不支持的聚合方法: {method}")
                
                row_weights.append(aggregated_weight)
            
            new_matrix.append(row_weights)
        
        # 创建新的DataFrame
        aggregated_df = pd.DataFrame(new_matrix,
                                   index=new_row_labels,
                                   columns=new_col_labels)
        
        return aggregated_df
    
    def extract_word_level_weights(self,
                                 csv_file: str,
                                 mappings: Dict[str, str],
                                 aggregation_method: str = "mean") -> Dict[str, float]:
        """
        从CSV文件中提取词级别的权重（处理子词聚合）
        
        Args:
            csv_file: CSV文件路径
            mappings: 映射字典 {text2_word: text1_word}
            aggregation_method: 聚合方法
            
        Returns:
            提取的权重字典
        """
        try:
            # 读取原始CSV
            df = pd.read_csv(csv_file, index_col=0)
            
            # 聚合子词
            aggregated_df = self.aggregate_subword_weights(df, aggregation_method)
            
            print(f"    子词聚合: {df.shape} -> {aggregated_df.shape}")
            
            # 提取映射权重
            weights = {}
            missing_mappings = []
            
            for text2_word, text1_word in mappings.items():
                mapping_key = f"{text2_word}->{text1_word}"
                
                try:
                    # 在聚合后的矩阵中查找
                    if text2_word in aggregated_df.index and text1_word in aggregated_df.columns:
                        weight = aggregated_df.loc[text2_word, text1_word]
                        
                        if isinstance(weight, pd.Series):
                            weight = weight.iloc[0]
                            print(f"    注意: {mapping_key} 仍存在重复，使用第一个值")
                        
                        if pd.isna(weight):
                            weight = 0.0
                        else:
                            weight = float(weight)
                        
                        weights[mapping_key] = weight
                        
                    else:
                        # 尝试模糊匹配
                        fuzzy_weight = self._fuzzy_match_weight(
                            text2_word, text1_word, aggregated_df)
                        
                        if fuzzy_weight is not None:
                            weights[mapping_key] = fuzzy_weight
                            print(f"    模糊匹配: {mapping_key} = {fuzzy_weight:.6f}")
                        else:
                            missing_mappings.append(mapping_key)
                            weights[mapping_key] = 0.0
                            
                except Exception as e:
                    print(f"    警告: 提取权重失败 {mapping_key}: {e}")
                    weights[mapping_key] = 0.0
            
            if missing_mappings:
                print(f"    缺失的映射: {missing_mappings}")
            
            return weights
            
        except Exception as e:
            print(f"    错误: 处理CSV文件失败 {csv_file}: {e}")
            return {f"{text2_word}->{text1_word}": 0.0 
                   for text2_word, text1_word in mappings.items()}
    
    def _fuzzy_match_weight(self, 
                          text2_word: str, 
                          text1_word: str, 
                          df: pd.DataFrame) -> float:
        """
        模糊匹配权重（处理词形变化等）
        
        Args:
            text2_word: 目标行词
            text1_word: 目标列词
            df: 权重矩阵
            
        Returns:
            匹配的权重值或None
        """
        # 尝试不同的匹配策略
        
        # 1. 小写匹配
        text2_lower = text2_word.lower()
        text1_lower = text1_word.lower()
        
        for row_word in df.index:
            if row_word.lower() == text2_lower:
                for col_word in df.columns:
                    if col_word.lower() == text1_lower:
                        return float(df.loc[row_word, col_word])
        
        # 2. 包含匹配
        for row_word in df.index:
            if text2_lower in row_word.lower() or row_word.lower() in text2_lower:
                for col_word in df.columns:
                    if text1_lower in col_word.lower() or col_word.lower() in text1_lower:
                        return float(df.loc[row_word, col_word])
        
        # 3. 前缀匹配
        for row_word in df.index:
            if row_word.lower().startswith(text2_lower[:3]) or text2_lower.startswith(row_word.lower()[:3]):
                for col_word in df.columns:
                    if col_word.lower().startswith(text1_lower[:3]) or text1_lower.startswith(col_word.lower()[:3]):
                        return float(df.loc[row_word, col_word])
        
        return None

def demo_subword_handling():
    """演示子词处理功能"""
    from transformers import BertTokenizer
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    handler = SubwordWeightHandler(tokenizer)
    
    # 测试文本
    text = "The requirements specification document"
    tokens = tokenizer.tokenize(text)
    
    print("=== 子词处理演示 ===")
    print(f"原始文本: {text}")
    print(f"Tokens: {tokens}")
    
    # 重构词
    word_groups = handler.reconstruct_words(tokens)
    print(f"重构的词: {word_groups}")
    
    # 创建模拟的注意力矩阵
    np.random.seed(42)
    mock_matrix = pd.DataFrame(
        np.random.rand(len(tokens), len(tokens)),
        index=tokens,
        columns=tokens
    )
    
    print(f"\n原始矩阵大小: {mock_matrix.shape}")
    
    # 聚合子词
    for method in ["mean", "max", "sum"]:
        aggregated = handler.aggregate_subword_weights(mock_matrix, method)
        print(f"{method.upper()}聚合后大小: {aggregated.shape}")

if __name__ == "__main__":
    demo_subword_handling()
