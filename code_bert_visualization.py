import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer
from bertviz import head_view, model_view
import webbrowser
import os
from typing import Dict, List, Tuple, Optional
import re

class CodeBERTVisualizer:
    def __init__(self, model_path: str = 'microsoft/codebert-base'):
        try:
            self.model = AutoModel.from_pretrained(
                model_path, 
                output_attentions=True,
                attn_implementation="eager"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()
            
            # 检查分词器类型
            print(f"分词器类型: {type(self.tokenizer)}")
            print(f"模型类型: {type(self.model)}")
            
            # 检查特殊token
            print(f"SEP token: {self.tokenizer.sep_token}")
            print(f"CLS token: {self.tokenizer.cls_token}")
            print(f"PAD token: {self.tokenizer.pad_token}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        # 设置matplotlib中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_code_text_attention(self, code: str, text: str, max_length: int = 512):
        """分析代码和文本之间的注意力关系"""
        try:
            encoding = self.tokenizer.encode_plus(
                code,
                text,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            )
            
            input_ids = encoding['input_ids']
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            special_tokens_mask = encoding.get('special_tokens_mask')
            
            print(f"\n--- 编码结果 (encode_plus) ---")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Tokens: {tokens}")
            print(f"Special tokens mask: {special_tokens_mask}")
            
            # 检查是否有SEP token
            sep_positions = [i for i, token in enumerate(tokens) if token in ['[SEP]', '</s>', '<sep>']]
            print(f"SEP positions: {sep_positions}")
            
            if len(sep_positions) < 1:
                print("⚠️  警告: 未找到足够的分隔符，尝试替代方法...")
                return self._fallback_encoding_method(code, text, max_length)
            
        except Exception as e:
            print(f"encode_plus 方法失败: {e}")
            return self._fallback_encoding_method(code, text, max_length)
        
        # 确定代码和文本的token位置
        if len(sep_positions) >= 2:
            # 标准BERT格式: [CLS] code [SEP] text [SEP]
            code_start = 1  # 跳过[CLS]
            code_end = sep_positions[0] - 1
            text_start = sep_positions[0] + 1
            text_end = sep_positions[1] - 1
        elif len(sep_positions) == 1:
            # 只有一个分隔符的情况
            code_start = 1
            code_end = sep_positions[0] - 1
            text_start = sep_positions[0] + 1
            text_end = len(tokens) - 2  # 假设最后是[SEP]或</s>
        else:
            raise ValueError("无法确定代码和文本的边界")
        
        print(f"\n--- Token位置分析 ---")
        print(f"代码范围: {code_start}-{code_end}")
        print(f"文本范围: {text_start}-{text_end}")
        print(f"代码tokens: {tokens[code_start:code_end+1]}")
        print(f"文本tokens: {tokens[text_start:text_end+1]}")
        
        # 前向传播获取注意力权重
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=encoding.get('attention_mask'))
            attentions = outputs.attentions
        
        print(f"注意力层数: {len(attentions)}")
        print(f"注意力shape: {attentions[0].shape}")
        
        return {
            'tokens': tokens,
            'code_tokens': tokens[code_start:code_end+1],
            'text_tokens': tokens[text_start:text_end+1],
            'code_range': (code_start, code_end),
            'text_range': (text_start, text_end),
            'attentions': attentions,
            'input_ids': input_ids,
            'processed_texts': {'code': code, 'text': text}
        }
    
    # 1. 交互式可视化 - BertViz
    def visualize_interactive_head_view(self, result: Dict, save_path: str = "attention_head_view.html"):
        """使用BertViz创建交互式头视图"""
        tokens = result['tokens']
        attentions = result['attentions']
        
        html_obj = head_view(
            attentions,
            tokens,
            html_action='return'
        )
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_obj.data)
        
        # 自动打开浏览器
        webbrowser.open('file://' + os.path.realpath(save_path))
        print(f"交互式头视图已保存到: {save_path}")
    
    # def visualize_interactive_model_view(self, result: Dict, save_path: str = "attention_model_view.html"):
    #     """使用BertViz创建交互式模型视图"""
    #     tokens = result['tokens']
    #     attentions = result['attentions']
        
    #     html_obj = model_view(
    #         attentions,
    #         tokens,
    #         html_action='return'
    #     )
        
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         f.write(html_obj.data)
        
    #     webbrowser.open('file://' + os.path.realpath(save_path))
    #     print(f"交互式模型视图已保存到: {save_path}")
    
    # # 2. 静态可视化 - 注意力热力图
    # def visualize_attention_heatmap(self, result: Dict, layer_idx: int = 6, head_idx: int = 0, 
    #                                save_path: str = "attention_heatmap.png"):
    #     """创建注意力权重热力图"""
    #     # 获取指定层和头的注意力权重
    #     attention = result['attentions'][layer_idx][0, head_idx].numpy()
    #     tokens = result['tokens']
        
    #     # 创建图形
    #     plt.figure(figsize=(12, 10))
        
    #     # 绘制热力图
    #     sns.heatmap(
    #         attention, 
    #         xticklabels=tokens, 
    #         yticklabels=tokens,
    #         cmap='Blues',
    #         cbar=True,
    #         square=True,
    #         linewidths=0.1
    #     )
        
    #     plt.title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}', 
    #              fontsize=14, fontweight='bold')
    #     plt.xlabel('Target Tokens', fontsize=12)
    #     plt.ylabel('Source Tokens', fontsize=12)
    #     plt.xticks(rotation=45, ha='right')
    #     plt.yticks(rotation=0)
        
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.show()
    #     print(f"注意力热力图已保存到: {save_path}")
    
    # # 3. 代码-文本交叉注意力可视化
    # def visualize_cross_attention(self, result: Dict, layer_idx: int = 6, 
    #                              save_path: str = "cross_attention.png"):
    #     """可视化代码和文本之间的交叉注意力"""
    #     code_start, code_end = result['code_range']
    #     text_start, text_end = result['text_range']
        
    #     # 获取交叉注意力（平均所有头）
    #     attention = result['attentions'][layer_idx][0].mean(dim=0).numpy()
        
    #     # 提取代码->文本和文本->代码的注意力
    #     code_to_text = attention[code_start:code_end+1, text_start:text_end+1]
    #     text_to_code = attention[text_start:text_end+1, code_start:code_end+1]
        
    #     code_tokens = result['code_tokens']
    #     text_tokens = result['text_tokens']
        
    #     # 创建子图
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
    #     # 代码 -> 文本注意力
    #     sns.heatmap(
    #         code_to_text,
    #         xticklabels=text_tokens,
    #         yticklabels=code_tokens,
    #         cmap='Reds',
    #         ax=ax1,
    #         cbar=True
    #     )
    #     ax1.set_title(f'Code → Text Attention (Layer {layer_idx})', fontsize=14, fontweight='bold')
    #     ax1.set_xlabel('Text Tokens', fontsize=12)
    #     ax1.set_ylabel('Code Tokens', fontsize=12)
        
    #     # 文本 -> 代码注意力
    #     sns.heatmap(
    #         text_to_code,
    #         xticklabels=code_tokens,
    #         yticklabels=text_tokens,
    #         cmap='Blues',
    #         ax=ax2,
    #         cbar=True
    #     )
    #     ax2.set_title(f'Text → Code Attention (Layer {layer_idx})', fontsize=14, fontweight='bold')
    #     ax2.set_xlabel('Code Tokens', fontsize=12)
    #     ax2.set_ylabel('Text Tokens', fontsize=12)
        
    #     plt.xticks(rotation=45, ha='right')
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.show()
    #     print(f"交叉注意力图已保存到: {save_path}")
    
    # # 4. 层级注意力分析
    # def visualize_layer_attention_analysis(self, result: Dict, save_path: str = "layer_analysis.png"):
    #     """分析不同层的注意力模式"""
    #     code_start, code_end = result['code_range']
    #     text_start, text_end = result['text_range']
        
    #     num_layers = len(result['attentions'])
        
    #     # 计算每层的平均注意力强度
    #     layer_stats = []
    #     for layer_idx in range(num_layers):
    #         attention = result['attentions'][layer_idx][0].mean(dim=0).numpy()
            
    #         # 代码到文本的平均注意力
    #         code_to_text_avg = attention[code_start:code_end+1, text_start:text_end+1].mean()
            
    #         # 文本到代码的平均注意力
    #         text_to_code_avg = attention[text_start:text_end+1, code_start:code_end+1].mean()
            
    #         # 自注意力（代码内部和文本内部）
    #         code_self_attn = attention[code_start:code_end+1, code_start:code_end+1].mean()
    #         text_self_attn = attention[text_start:text_end+1, text_start:text_end+1].mean()
            
    #         layer_stats.append({
    #             'layer': layer_idx,
    #             'code_to_text': code_to_text_avg,
    #             'text_to_code': text_to_code_avg,
    #             'code_self': code_self_attn,
    #             'text_self': text_self_attn
    #         })
        
    #     # 绘制层级分析图
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
    #     layers = list(range(num_layers))
        
    #     # 交叉注意力趋势
    #     code_to_text_vals = [s['code_to_text'] for s in layer_stats]
    #     text_to_code_vals = [s['text_to_code'] for s in layer_stats]
        
    #     ax1.plot(layers, code_to_text_vals, 'o-', label='Code → Text', color='red', linewidth=2)
    #     ax1.plot(layers, text_to_code_vals, 's-', label='Text → Code', color='blue', linewidth=2)
    #     ax1.set_title('Cross-Modal Attention by Layer', fontweight='bold')
    #     ax1.set_xlabel('Layer')
    #     ax1.set_ylabel('Average Attention')
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
        
    #     # 自注意力趋势
    #     code_self_vals = [s['code_self'] for s in layer_stats]
    #     text_self_vals = [s['text_self'] for s in layer_stats]
        
    #     ax2.plot(layers, code_self_vals, 'o-', label='Code Self-Attention', color='orange', linewidth=2)
    #     ax2.plot(layers, text_self_vals, 's-', label='Text Self-Attention', color='green', linewidth=2)
    #     ax2.set_title('Self-Attention by Layer', fontweight='bold')
    #     ax2.set_xlabel('Layer')
    #     ax2.set_ylabel('Average Attention')
    #     ax2.legend()
    #     ax2.grid(True, alpha=0.3)
        
    #     # 注意力强度分布
    #     all_cross_attn = code_to_text_vals + text_to_code_vals
    #     ax3.hist(all_cross_attn, bins=20, alpha=0.7, color='purple', edgecolor='black')
    #     ax3.set_title('Cross-Attention Distribution', fontweight='bold')
    #     ax3.set_xlabel('Attention Strength')
    #     ax3.set_ylabel('Frequency')
    #     ax3.grid(True, alpha=0.3)
        
    #     # 层级热力图
    #     data_matrix = np.array([[s['code_to_text'], s['text_to_code'], 
    #                            s['code_self'], s['text_self']] for s in layer_stats])
        
    #     sns.heatmap(
    #         data_matrix.T,
    #         xticklabels=layers,
    #         yticklabels=['Code→Text', 'Text→Code', 'Code Self', 'Text Self'],
    #         cmap='viridis',
    #         ax=ax4,
    #         cbar=True
    #     )
    #     ax4.set_title('Attention Patterns Heatmap', fontweight='bold')
    #     ax4.set_xlabel('Layer')
        
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.show()
    #     print(f"层级分析图已保存到: {save_path}")
    
    # # 5. 最强注意力对应分析
    # def analyze_top_attention_pairs(self, result: Dict, layer_idx: int = 6, top_k: int = 10):
    #     """分析最强的注意力对应关系"""
    #     code_start, code_end = result['code_range']
    #     text_start, text_end = result['text_range']
        
    #     # 获取交叉注意力（平均所有头）
    #     attention = result['attentions'][layer_idx][0].mean(dim=0).numpy()
    #     code_to_text = attention[code_start:code_end+1, text_start:text_end+1]
        
    #     code_tokens = result['code_tokens']
    #     text_tokens = result['text_tokens']
        
    #     # 找到top-k最强的对应关系
    #     flat_indices = np.argpartition(code_to_text.flatten(), -top_k)[-top_k:]
    #     top_pairs = []
        
    #     for idx in flat_indices:
    #         i, j = np.unravel_index(idx, code_to_text.shape)
    #         attention_score = code_to_text[i, j]
    #         top_pairs.append((code_tokens[i], text_tokens[j], attention_score))
        
    #     # 按注意力强度排序
    #     top_pairs.sort(key=lambda x: x[2], reverse=True)
        
    #     print(f"\n=== Layer {layer_idx} Top-{top_k} Attention Pairs ===")
    #     for i, (code_token, text_token, score) in enumerate(top_pairs, 1):
    #         print(f"{i:2d}. '{code_token}' ← → '{text_token}' (强度: {score:.4f})")
        
    #     return top_pairs
    
    # 6. 综合可视化
    def create_comprehensive_visualization(self, code: str, text: str, save_dir: str = "CodeBERT_visualization_results"):
        """创建所有类型的可视化"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print("开始分析CodeBERT注意力...")
        result = self.analyze_code_text_attention(code, text)
        
        print(f"代码tokens数量: {len(result['code_tokens'])}")
        print(f"文本tokens数量: {len(result['text_tokens'])}")
        print(f"代码tokens: {result['code_tokens']}")
        print(f"文本tokens: {result['text_tokens']}")
        
        # 1. 交互式可视化
        print("\n生成交互式可视化...")
        self.visualize_interactive_head_view(result, f"{save_dir}/head_view.html")
        # self.visualize_interactive_model_view(result, f"{save_dir}/model_view.html")
        
        # # 2. 静态可视化
        # print("\n生成静态可视化...")
        # self.visualize_attention_heatmap(result, layer_idx=6, save_path=f"{save_dir}/attention_heatmap.png")
        
        # # 3. 交叉注意力
        # print("\n生成交叉注意力可视化...")
        # self.visualize_cross_attention(result, layer_idx=6, save_path=f"{save_dir}/cross_attention.png")
        
        # # 4. 层级分析
        # print("\n生成层级分析...")
        # self.visualize_layer_attention_analysis(result, save_path=f"{save_dir}/layer_analysis.png")
        
        # # 5. 最强对应分析
        # print("\n分析最强注意力对应...")
        # for layer in [2, 6, 10]:  # 浅层、中层、深层
        #     self.analyze_top_attention_pairs(result, layer_idx=layer, top_k=5)
        
        # print(f"\n所有可视化已完成！结果保存在: {save_dir}/")

# 使用示例
def demo_visualization():
    """演示可视化功能"""
    
    # 初始化可视化器
    visualizer = CodeBERTVisualizer()
    
    # 示例代码和描述
    code = "def add(a,b):return a + b"
    
    text = "This function adds two numbers"
    
    # 创建综合可视化
    visualizer.create_comprehensive_visualization(code, text)

if __name__ == "__main__":
    demo_visualization() 