"""
详细的注意力头分析
计算每一层每一个头的词与词之间的权重，并分别保存到文件
"""

from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

class DetailedHeadAnalyzer:
    def __init__(self, model_path: str):
        """
        初始化详细头分析器
        
        Args:
            model_path: BERT模型路径
        """
        self.model = BertModel.from_pretrained(model_path, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model_config = self.model.config
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        
        print(f"模型配置:")
        print(f"  层数: {self.num_layers}")
        print(f"  每层注意力头数: {self.num_heads}")
        print(f"  总注意力头数: {self.num_layers * self.num_heads}")
    
    def process_inputs(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        处理两个输入文本
        
        Args:
            text1: 第一个输入文本
            text2: 第二个输入文本
            
        Returns:
            处理结果字典
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
        text1_tokens = tokens[:sentence_b_start]
        text2_tokens = tokens[sentence_b_start:]
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            attention_weights = outputs.attentions
        
        return {
            'tokens': tokens,
            'text1_tokens': text1_tokens,
            'text2_tokens': text2_tokens,
            'sentence_b_start': sentence_b_start,
            'attention_weights': attention_weights,
            'text1': text1,
            'text2': text2
        }
    
    def calculate_single_head_cross_attention(self, layer_attention: torch.Tensor, 
                                            head_idx: int, 
                                            sentence_b_start: int,
                                            text1_tokens: List[str],
                                            text2_tokens: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        计算单个注意力头的交叉注意力权重
        
        Args:
            layer_attention: 某一层的注意力权重 [num_heads, seq_len, seq_len]
            head_idx: 注意力头索引
            sentence_b_start: 第二个句子开始位置
            text1_tokens: 第一个文本的tokens
            text2_tokens: 第二个文本的tokens
            
        Returns:
            (text1_to_text2_df, text2_to_text1_df): 两个DataFrame
        """
        # 获取指定头的注意力权重 [seq_len, seq_len]
        head_attention = layer_attention[head_idx]
        
        # 提取交叉注意力
        text1_to_text2 = head_attention[:sentence_b_start, sentence_b_start:]
        text2_to_text1 = head_attention[sentence_b_start:, :sentence_b_start]
        
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
    
    def create_output_folder(self, base_name: str = "attention_heads_analysis") -> str:
        """
        创建输出文件夹
        
        Args:
            base_name: 基础文件夹名称
            
        Returns:
            创建的文件夹路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{base_name}_{timestamp}"
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"创建文件夹: {folder_name}")
        
        # 创建子文件夹
        subfolders = ["csv_files", "json_files", "summary"]
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_name, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
        
        return folder_name
    
    def save_head_results(self, text1_to_text2_df: pd.DataFrame, 
                         text2_to_text1_df: pd.DataFrame,
                         layer_idx: int, head_idx: int, 
                         output_folder: str) -> Dict[str, Any]:
        """
        保存单个头的结果到文件（修复版本）
        
        Args:
            text1_to_text2_df: 文本1到文本2的权重矩阵
            text2_to_text1_df: 文本2到文本1的权重矩阵
            layer_idx: 层索引
            head_idx: 头索引
            output_folder: 输出文件夹
            
        Returns:
            该头的统计信息
        """
        try:
            # 文件名前缀
            file_prefix = f"layer_{layer_idx:02d}_head_{head_idx:02d}"
            
            # 保存CSV文件
            csv_folder = os.path.join(output_folder, "csv_files")
            text1_to_text2_df.to_csv(os.path.join(csv_folder, f"{file_prefix}_text1_to_text2.csv"))
            text2_to_text1_df.to_csv(os.path.join(csv_folder, f"{file_prefix}_text2_to_text1.csv"))
            
            # 安全的统计信息计算函数
            def safe_float_conversion(value):
                """安全地将值转换为float"""
                try:
                    if pd.isna(value):
                        return 0.0
                    if isinstance(value, (pd.Series, np.ndarray)):
                        # 如果是Series或数组，取第一个非NaN值
                        if hasattr(value, 'dropna') and len(value.dropna()) > 0:
                            return float(value.dropna().iloc[0])
                        else:
                            return 0.0
                    return float(value)
                except (ValueError, TypeError, AttributeError):
                    return 0.0
            
            def safe_stats_calculation(df, stat_name):
                """安全地计算DataFrame统计信息"""
                try:
                    if df.empty:
                        return 0.0
                    
                    # 确保数据类型为数值型
                    numeric_df = df.select_dtypes(include=[np.number])
                    if numeric_df.empty:
                        # 尝试转换非数值列
                        numeric_df = df.apply(pd.to_numeric, errors='coerce')
                    
                    if stat_name == 'mean':
                        result = numeric_df.mean().mean()
                    elif stat_name == 'std':
                        result = numeric_df.std().mean()  # 使用mean而不是std().std()
                    elif stat_name == 'max':
                        result = numeric_df.max().max()
                    elif stat_name == 'min':
                        result = numeric_df.min().min()
                    else:
                        return 0.0
                    
                    return safe_float_conversion(result)
                except Exception as e:
                    print(f"  警告: 计算{stat_name}时出错: {e}")
                    return 0.0
            
            # 计算统计信息
            stats = {
                'layer': layer_idx,
                'head': head_idx,
                'text1_to_text2_stats': {
                    'shape': text1_to_text2_df.shape,
                    'mean': safe_stats_calculation(text1_to_text2_df, 'mean'),
                    'std': safe_stats_calculation(text1_to_text2_df, 'std'),
                    'max': safe_stats_calculation(text1_to_text2_df, 'max'),
                    'min': safe_stats_calculation(text1_to_text2_df, 'min')
                },
                'text2_to_text1_stats': {
                    'shape': text2_to_text1_df.shape,
                    'mean': safe_stats_calculation(text2_to_text1_df, 'mean'),
                    'std': safe_stats_calculation(text2_to_text1_df, 'std'),
                    'max': safe_stats_calculation(text2_to_text1_df, 'max'),
                    'min': safe_stats_calculation(text2_to_text1_df, 'min')
                }
            }
            
            # 安全地找出最高权重的词对
            try:
                if not text1_to_text2_df.empty and text1_to_text2_df.notna().any().any():
                    # 确保数据为数值型
                    numeric_df1 = text1_to_text2_df.apply(pd.to_numeric, errors='coerce')
                    text1_to_text2_max = numeric_df1.stack().idxmax()
                    text1_to_text2_max_weight = safe_float_conversion(text1_to_text2_df.loc[text1_to_text2_max])
                else:
                    text1_to_text2_max = ("N/A", "N/A")
                    text1_to_text2_max_weight = 0.0
                    
                if not text2_to_text1_df.empty and text2_to_text1_df.notna().any().any():
                    numeric_df2 = text2_to_text1_df.apply(pd.to_numeric, errors='coerce')
                    text2_to_text1_max = numeric_df2.stack().idxmax()
                    text2_to_text1_max_weight = safe_float_conversion(text2_to_text1_df.loc[text2_to_text1_max])
                else:
                    text2_to_text1_max = ("N/A", "N/A")
                    text2_to_text1_max_weight = 0.0
                    
            except Exception as e:
                print(f"  警告: 寻找最大权重词对时出错: {e}")
                text1_to_text2_max = ("N/A", "N/A")
                text2_to_text1_max = ("N/A", "N/A")
                text1_to_text2_max_weight = 0.0
                text2_to_text1_max_weight = 0.0
            
            stats['top_word_pairs'] = {
                'text1_to_text2_max': {
                    'word_pair': text1_to_text2_max,
                    'weight': text1_to_text2_max_weight
                },
                'text2_to_text1_max': {
                    'word_pair': text2_to_text1_max,
                    'weight': text2_to_text1_max_weight
                }
            }
            
            # 保存JSON统计信息
            json_folder = os.path.join(output_folder, "json_files")
            with open(os.path.join(json_folder, f"{file_prefix}_stats.json"), 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            return stats
            
        except Exception as e:
            print(f"  错误: 保存头 {layer_idx}-{head_idx} 结果时出错: {e}")
            # 返回默认统计信息
            return {
                'layer': layer_idx,
                'head': head_idx,
                'text1_to_text2_stats': {'shape': (0, 0), 'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0},
                'text2_to_text1_stats': {'shape': (0, 0), 'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0},
                'top_word_pairs': {
                    'text1_to_text2_max': {'word_pair': ("N/A", "N/A"), 'weight': 0.0},
                    'text2_to_text1_max': {'word_pair': ("N/A", "N/A"), 'weight': 0.0}
                }
            }
    
    def analyze_all_heads(self, text1: str, text2: str, 
                         output_folder: str = None) -> Dict[str, Any]:
        """
        分析所有层的所有头
        
        Args:
            text1: 第一个输入文本
            text2: 第二个输入文本
            output_folder: 输出文件夹（如果为None则自动创建）
            
        Returns:
            完整的分析结果
        """
        if output_folder is None:
            output_folder = self.create_output_folder()
        
        print(f"\n开始分析所有注意力头...")
        print(f"输出文件夹: {output_folder}")
        print(f"预计生成文件数: {self.num_layers * self.num_heads * 3} 个")  # 每个头3个文件
        
        # 处理输入
        processed = self.process_inputs(text1, text2)
        
        # 保存输入信息
        input_info = {
            'text1': text1,
            'text2': text2,
            'tokens': {
                'all_tokens': processed['tokens'],
                'text1_tokens': processed['text1_tokens'],
                'text2_tokens': processed['text2_tokens'],
                'sentence_b_start': processed['sentence_b_start']
            },
            'model_config': {
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'total_heads': self.num_layers * self.num_heads
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_folder, "input_info.json"), 'w', encoding='utf-8') as f:
            json.dump(input_info, f, indent=2, ensure_ascii=False)
        
        # 分析结果存储
        all_head_stats = []
        layer_summaries = []
        
        # 逐层逐头分析
        for layer_idx in range(self.num_layers):
            print(f"\n处理第 {layer_idx + 1}/{self.num_layers} 层...")
            
            layer_attention = processed['attention_weights'][layer_idx][0]  # [num_heads, seq_len, seq_len]
            layer_head_stats = []
            
            for head_idx in range(self.num_heads):
                try:
                    print(f"  处理头 {head_idx + 1}/{self.num_heads}...", end=' ')
                    
                    # 计算单个头的交叉注意力
                    text1_to_text2_df, text2_to_text1_df = self.calculate_single_head_cross_attention(
                        layer_attention, head_idx, 
                        processed['sentence_b_start'],
                        processed['text1_tokens'],
                        processed['text2_tokens']
                    )
                    
                    # 保存结果并获取统计信息
                    head_stats = self.save_head_results(
                        text1_to_text2_df, text2_to_text1_df,
                        layer_idx, head_idx, output_folder
                    )
                    
                    all_head_stats.append(head_stats)
                    layer_head_stats.append(head_stats)
                    print("完成")
                    
                except Exception as e:
                    print(f"分析过程中出现错误: {e}")
                    print(f"  跳过头 {head_idx + 1}，继续处理下一个...")
                    # 添加默认统计信息以保持一致性
                    default_stats = {
                        'layer': layer_idx,
                        'head': head_idx,
                        'text1_to_text2_stats': {'shape': (0, 0), 'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0},
                        'text2_to_text1_stats': {'shape': (0, 0), 'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0},
                        'top_word_pairs': {
                            'text1_to_text2_max': {'word_pair': ("N/A", "N/A"), 'weight': 0.0},
                            'text2_to_text1_max': {'word_pair': ("N/A", "N/A"), 'weight': 0.0}
                        }
                    }
                    all_head_stats.append(default_stats)
                    layer_head_stats.append(default_stats)
                    continue
            
            # 计算层级汇总
            layer_summary = self.calculate_layer_summary(layer_head_stats, layer_idx)
            layer_summaries.append(layer_summary)
        
        # 生成总体汇总报告
        overall_summary = self.generate_overall_summary(all_head_stats, layer_summaries, input_info)
        
        # 保存汇总报告
        summary_folder = os.path.join(output_folder, "summary")
        with open(os.path.join(summary_folder, "overall_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2, ensure_ascii=False)
        
        # 生成可读的汇总报告
        self.generate_readable_summary(overall_summary, summary_folder)
        
        print(f"\n=== 分析完成! ===")
        print(f"总共分析了 {len(all_head_stats)} 个注意力头")
        print(f"生成的文件保存在: {output_folder}")
        print(f"查看汇总报告: {os.path.join(summary_folder, 'readable_summary.txt')}")
        
        return {
            'output_folder': output_folder,
            'all_head_stats': all_head_stats,
            'layer_summaries': layer_summaries,
            'overall_summary': overall_summary,
            'input_info': input_info
        }
    
    def calculate_layer_summary(self, layer_head_stats: List[Dict], layer_idx: int) -> Dict[str, Any]:
        """计算层级汇总统计（增强版，包含最大权重词对）"""
        text1_to_text2_means = [stat['text1_to_text2_stats']['mean'] for stat in layer_head_stats]
        text2_to_text1_means = [stat['text2_to_text1_stats']['mean'] for stat in layer_head_stats]
        
        text1_to_text2_maxs = [stat['text1_to_text2_stats']['max'] for stat in layer_head_stats]
        text2_to_text1_maxs = [stat['text2_to_text1_stats']['max'] for stat in layer_head_stats]
        
        # 找出该层中文本1->文本2的最大权重词对
        text1_to_text2_max_idx = np.argmax(text1_to_text2_maxs)
        text1_to_text2_max_head = layer_head_stats[text1_to_text2_max_idx]
        
        # 找出该层中文本2->文本1的最大权重词对  
        text2_to_text1_max_idx = np.argmax(text2_to_text1_maxs)
        text2_to_text1_max_head = layer_head_stats[text2_to_text1_max_idx]
        
        return {
            'layer': layer_idx,
            'num_heads': len(layer_head_stats),
            'text1_to_text2_summary': {
                'mean_of_means': np.mean(text1_to_text2_means),
                'std_of_means': np.std(text1_to_text2_means),
                'max_of_maxs': np.max(text1_to_text2_maxs),
                'min_of_means': np.min(text1_to_text2_means)
            },
            'text2_to_text1_summary': {
                'mean_of_means': np.mean(text2_to_text1_means),
                'std_of_means': np.std(text2_to_text1_means),
                'max_of_maxs': np.max(text2_to_text1_maxs),
                'min_of_means': np.min(text2_to_text1_means)
            },
            # 新增：每层的最大权重词对信息
            'layer_max_word_pairs': {
                'text1_to_text2_max': {
                    'head': text1_to_text2_max_head['head'],
                    'weight': text1_to_text2_max_head['text1_to_text2_stats']['max'],
                    'word_pair': text1_to_text2_max_head['top_word_pairs']['text1_to_text2_max']['word_pair'],
                    'weight_from_pair': text1_to_text2_max_head['top_word_pairs']['text1_to_text2_max']['weight']
                },
                'text2_to_text1_max': {
                    'head': text2_to_text1_max_head['head'],
                    'weight': text2_to_text1_max_head['text2_to_text1_stats']['max'],
                    'word_pair': text2_to_text1_max_head['top_word_pairs']['text2_to_text1_max']['word_pair'],
                    'weight_from_pair': text2_to_text1_max_head['top_word_pairs']['text2_to_text1_max']['weight']
                }
            }
        }
    
    def generate_overall_summary(self, all_head_stats: List[Dict], 
                               layer_summaries: List[Dict], 
                               input_info: Dict) -> Dict[str, Any]:
        """生成总体汇总"""
        # 找出全局最高权重
        all_text1_to_text2_maxs = [stat['text1_to_text2_stats']['max'] for stat in all_head_stats]
        all_text2_to_text1_maxs = [stat['text2_to_text1_stats']['max'] for stat in all_head_stats]
        
        global_max_text1_to_text2_idx = np.argmax(all_text1_to_text2_maxs)
        global_max_text2_to_text1_idx = np.argmax(all_text2_to_text1_maxs)
        
        return {
            'input_summary': input_info,
            'analysis_summary': {
                'total_heads_analyzed': len(all_head_stats),
                'total_layers': len(layer_summaries),
                'heads_per_layer': self.num_heads
            },
            'global_statistics': {
                'text1_to_text2': {
                    'global_max_weight': float(np.max(all_text1_to_text2_maxs)),
                    'global_max_head': {
                        'layer': all_head_stats[global_max_text1_to_text2_idx]['layer'],
                        'head': all_head_stats[global_max_text1_to_text2_idx]['head'],
                        'word_pair': all_head_stats[global_max_text1_to_text2_idx]['top_word_pairs']['text1_to_text2_max']['word_pair']
                    },
                    'mean_across_all_heads': float(np.mean([stat['text1_to_text2_stats']['mean'] for stat in all_head_stats]))
                },
                'text2_to_text1': {
                    'global_max_weight': float(np.max(all_text2_to_text1_maxs)),
                    'global_max_head': {
                        'layer': all_head_stats[global_max_text2_to_text1_idx]['layer'],
                        'head': all_head_stats[global_max_text2_to_text1_idx]['head'],
                        'word_pair': all_head_stats[global_max_text2_to_text1_idx]['top_word_pairs']['text2_to_text1_max']['word_pair']
                    },
                    'mean_across_all_heads': float(np.mean([stat['text2_to_text1_stats']['mean'] for stat in all_head_stats]))
                }
            },
            'layer_summaries': layer_summaries
        }
    
    def generate_readable_summary(self, overall_summary: Dict, summary_folder: str):
        """生成可读的汇总报告（增强版，包含每层最大权重词对）"""
        summary_text = []
        
        summary_text.append("=" * 80)
        summary_text.append("注意力头详细分析汇总报告")
        summary_text.append("=" * 80)
        summary_text.append("")
        
        # 输入信息
        input_info = overall_summary['input_summary']
        summary_text.append("输入信息:")
        summary_text.append(f"  文本1: {input_info['text1']}")
        summary_text.append(f"  文本2: {input_info['text2']}")
        summary_text.append(f"  文本1 tokens数: {len(input_info['tokens']['text1_tokens'])}")
        summary_text.append(f"  文本2 tokens数: {len(input_info['tokens']['text2_tokens'])}")
        summary_text.append("")
        
        # Token详情
        summary_text.append("Token详情:")
        summary_text.append(f"  文本1 tokens: {input_info['tokens']['text1_tokens']}")
        summary_text.append(f"  文本2 tokens: {input_info['tokens']['text2_tokens']}")
        summary_text.append("")
        
        # 模型配置
        model_config = input_info['model_config']
        summary_text.append("模型配置:")
        summary_text.append(f"  总层数: {model_config['num_layers']}")
        summary_text.append(f"  每层头数: {model_config['num_heads']}")
        summary_text.append(f"  总头数: {model_config['total_heads']}")
        summary_text.append("")
        
        # 全局统计
        global_stats = overall_summary['global_statistics']
        summary_text.append("全局统计:")
        summary_text.append("  文本1 -> 文本2:")
        t1_to_t2 = global_stats['text1_to_text2']
        summary_text.append(f"    全局最高权重: {t1_to_t2['global_max_weight']:.6f}")
        summary_text.append(f"    来自: 第{t1_to_t2['global_max_head']['layer']+1}层第{t1_to_t2['global_max_head']['head']+1}头")
        summary_text.append(f"    词对: '{t1_to_t2['global_max_head']['word_pair'][0]}' → '{t1_to_t2['global_max_head']['word_pair'][1]}'")
        summary_text.append(f"    所有头平均权重: {t1_to_t2['mean_across_all_heads']:.6f}")
        summary_text.append("")
        
        summary_text.append("  文本2 -> 文本1:")
        t2_to_t1 = global_stats['text2_to_text1']
        summary_text.append(f"    全局最高权重: {t2_to_t1['global_max_weight']:.6f}")
        summary_text.append(f"    来自: 第{t2_to_t1['global_max_head']['layer']+1}层第{t2_to_t1['global_max_head']['head']+1}头")
        summary_text.append(f"    词对: '{t2_to_t1['global_max_head']['word_pair'][0]}' → '{t2_to_t1['global_max_head']['word_pair'][1]}'")
        summary_text.append(f"    所有头平均权重: {t2_to_t1['mean_across_all_heads']:.6f}")
        summary_text.append("")
        
        # 每层详细汇总（包含最大权重词对）
        summary_text.append("每层详细汇总:")
        summary_text.append("-" * 60)
        
        for layer_summary in overall_summary['layer_summaries']:
            layer_idx = layer_summary['layer']
            summary_text.append(f"\n第 {layer_idx+1} 层 (共 {layer_summary['num_heads']} 个头):")
            
            # 统计信息
            t1_to_t2_summary = layer_summary['text1_to_text2_summary']
            t2_to_t1_summary = layer_summary['text2_to_text1_summary']
            summary_text.append(f"  统计信息:")
            summary_text.append(f"    文本1→文本2: 平均={t1_to_t2_summary['mean_of_means']:.6f}, 最大={t1_to_t2_summary['max_of_maxs']:.6f}")
            summary_text.append(f"    文本2→文本1: 平均={t2_to_t1_summary['mean_of_means']:.6f}, 最大={t2_to_t1_summary['max_of_maxs']:.6f}")
            
            # 该层最大权重词对
            if 'layer_max_word_pairs' in layer_summary:
                max_pairs = layer_summary['layer_max_word_pairs']
                summary_text.append(f"  该层最大权重词对:")
                
                # 文本1->文本2最大权重词对
                t1_to_t2_max = max_pairs['text1_to_text2_max']
                summary_text.append(f"    文本1→文本2: '{t1_to_t2_max['word_pair'][0]}' → '{t1_to_t2_max['word_pair'][1]}'")
                summary_text.append(f"      权重: {t1_to_t2_max['weight_from_pair']:.6f} (来自第{t1_to_t2_max['head']+1}头)")
                
                # 文本2->文本1最大权重词对
                t2_to_t1_max = max_pairs['text2_to_text1_max']
                summary_text.append(f"    文本2→文本1: '{t2_to_t1_max['word_pair'][0]}' → '{t2_to_t1_max['word_pair'][1]}'")
                summary_text.append(f"      权重: {t2_to_t1_max['weight_from_pair']:.6f} (来自第{t2_to_t1_max['head']+1}头)")
            
            summary_text.append("")
        
        # 层级趋势分析
        summary_text.append("层级趋势分析:")
        summary_text.append("-" * 60)
        
        # 收集各层的平均权重用于趋势分析
        layer_means_t1_t2 = [ls['text1_to_text2_summary']['mean_of_means'] for ls in overall_summary['layer_summaries']]
        layer_means_t2_t1 = [ls['text2_to_text1_summary']['mean_of_means'] for ls in overall_summary['layer_summaries']]
        layer_maxs_t1_t2 = [ls['text1_to_text2_summary']['max_of_maxs'] for ls in overall_summary['layer_summaries']]
        layer_maxs_t2_t1 = [ls['text2_to_text1_summary']['max_of_maxs'] for ls in overall_summary['layer_summaries']]
        
        # 找出最活跃的层（平均权重最高）
        most_active_layer_t1_t2 = np.argmax(layer_means_t1_t2)
        most_active_layer_t2_t1 = np.argmax(layer_means_t2_t1)
        
        # 找出权重峰值层（最大权重最高）
        peak_layer_t1_t2 = np.argmax(layer_maxs_t1_t2)
        peak_layer_t2_t1 = np.argmax(layer_maxs_t2_t1)
        
        summary_text.append(f"最活跃层 (平均权重最高):")
        summary_text.append(f"  文本1→文本2: 第{most_active_layer_t1_t2+1}层 (平均权重: {layer_means_t1_t2[most_active_layer_t1_t2]:.6f})")
        summary_text.append(f"  文本2→文本1: 第{most_active_layer_t2_t1+1}层 (平均权重: {layer_means_t2_t1[most_active_layer_t2_t1]:.6f})")
        summary_text.append("")
        
        summary_text.append(f"权重峰值层 (最大权重最高):")
        summary_text.append(f"  文本1→文本2: 第{peak_layer_t1_t2+1}层 (最大权重: {layer_maxs_t1_t2[peak_layer_t1_t2]:.6f})")
        summary_text.append(f"  文本2→文本1: 第{peak_layer_t2_t1+1}层 (最大权重: {layer_maxs_t2_t1[peak_layer_t2_t1]:.6f})")
        summary_text.append("")
        
        # 使用指南
        summary_text.append("使用指南:")
        summary_text.append("-" * 60)
        summary_text.append("1. 查看CSV文件: csv_files/ 文件夹包含每个头的详细权重矩阵")
        summary_text.append("2. 查看JSON文件: json_files/ 文件夹包含每个头的统计信息")
        summary_text.append("3. 重点关注: 权重峰值层和最活跃层通常包含最重要的语义关联")
        summary_text.append("4. 词对解释: 权重越高表示两个词之间的注意力关联越强")
        summary_text.append("")
        
        summary_text.append("=" * 80)
        summary_text.append(f"分析完成时间: {input_info['timestamp']}")
        summary_text.append(f"总处理文件数: {overall_summary['analysis_summary']['total_heads_analyzed'] * 2} 个CSV文件")
        summary_text.append(f"总处理头数: {overall_summary['analysis_summary']['total_heads_analyzed']} 个注意力头")
        summary_text.append("=" * 80)
        
        # 保存到文件
        with open(os.path.join(summary_folder, "readable_summary.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_text))
        
        print(f"可读汇总报告已保存到: {os.path.join(summary_folder, 'readable_summary.txt')}")


# 主程序示例
if __name__ == "__main__":
    # 初始化分析器
    model_path = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'
    analyzer = DetailedHeadAnalyzer(model_path)
    
    # 两个输入文本
    text1 = "One Doctor is associated with multiple Requisitions."
    text2 = "A doctor may also indicate that the tests on a requisition are to be repeated for a specified number of times and interval."
    
    # 执行完整分析
    results = analyzer.analyze_all_heads(text1, text2)
    
    print(f"\n分析结果已保存到: {results['output_folder']}")
