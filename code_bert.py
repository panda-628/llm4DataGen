from transformers import AutoTokenizer, AutoModel
import torch

class PlantUMLBertAttention:
    def __init__(self, model_path: str = 'microsoft/codebert-base'):
        """
        初始化CodeBERT模型和分词器
        
        Args:
            model_path: CodeBERT模型路径或标识符
        """
        try:
            # 正确的CodeBERT模型标识符
            self.model = AutoModel.from_pretrained(
                model_path, 
                output_attentions=True,  # 启用注意力权重输出
                attn_implementation="eager"  # 确保兼容性
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()  # 设置为评估模式
            
            print(f"成功加载CodeBERT模型: {model_path}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def analyze_code_text_attention(self, code: str, text: str, max_length: int = 512):
        """
        分析代码和文本之间的注意力关系
        
        Args:
            code: 代码字符串
            text: 自然语言描述
            max_length: 最大序列长度
            
        Returns:
            包含tokens和注意力权重的字典
        """
        try:
            # 编码输入 - CodeBERT支持代码和文本的联合编码
            inputs = self.tokenizer(
                code, 
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=max_length
            )
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取tokens用于分析
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            return {
                'last_hidden_state': outputs.last_hidden_state,
                'attentions': outputs.attentions,
                'tokens': tokens,
                'input_ids': inputs['input_ids'],
                'inputs': inputs
            }
            
        except Exception as e:
            print(f"分析过程出错: {e}")
            raise

    def debug_tokenization(self, code: str, text: str):
        """
        调试token化过程，找出分隔符和结构
        """
        # 联合编码
        joint_inputs = self.tokenizer(code, text, return_tensors="pt", truncation=True, padding=True)
        joint_tokens = self.tokenizer.convert_ids_to_tokens(joint_inputs['input_ids'][0])
        
        # 分别编码
        code_inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True)
        code_tokens = self.tokenizer.convert_ids_to_tokens(code_inputs['input_ids'][0])
        
        text_inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        text_tokens = self.tokenizer.convert_ids_to_tokens(text_inputs['input_ids'][0])
        
        print("=== Token化调试信息 ===")
        print(f"联合编码tokens数量: {len(joint_tokens)}")
        print(f"联合编码所有tokens: {joint_tokens}")
        print(f"代码单独编码: {code_tokens}")
        print(f"文本单独编码: {text_tokens}")
        
        # 查找特殊tokens
        special_tokens = self.tokenizer.all_special_tokens
        print(f"特殊tokens: {special_tokens}")
        
        return {
            'joint_tokens': joint_tokens,
            'code_tokens': code_tokens,
            'text_tokens': text_tokens,
            'special_tokens': special_tokens
        }

    def calculate_matching_score_safe(self, code: str, text: str):
        """
        安全版本的匹配度计算，包含错误处理和调试信息
        """
        try:
            # 先调试token化
            debug_info = self.debug_tokenization(code, text)
            
            # 获取分析结果
            result = self.analyze_code_text_attention(code, text)
            
            print(f"\n=== 注意力分析调试 ===")
            print(f"注意力层数: {len(result['attentions'])}")
            print(f"注意力矩阵形状 (最后一层): {result['attentions'][-1].shape}")
            
            # 使用更安全的方法计算相似度
            similarity_score = self._calculate_safe_similarity(result, debug_info)
            
            return {
                'similarity_score': similarity_score,
                'debug_info': debug_info,
                'attention_shape': result['attentions'][-1].shape
            }
            
        except Exception as e:
            print(f"匹配度计算失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _calculate_safe_similarity(self, result, debug_info):
        """
        安全的相似度计算方法
        """
        try:
            # 使用CLS token计算简单相似度
            hidden_states = result['last_hidden_state'][0]  # [seq_len, hidden_size]
            
            # CLS token在第一个位置
            cls_representation = hidden_states[0]  # [hidden_size]
            
            # 计算序列的平均表示（排除特殊tokens）
            # 假设CLS在开头，SEP/EOS在结尾
            if len(hidden_states) > 2:
                content_representation = hidden_states[1:-1].mean(dim=0)  # [hidden_size]
                
                # 计算CLS与内容的相似度作为基准
                similarity = torch.cosine_similarity(
                    cls_representation.unsqueeze(0), 
                    content_representation.unsqueeze(0)
                )
                
                return float(similarity)
            else:
                return 0.0
                
        except Exception as e:
            print(f"相似度计算出错: {e}")
            return 0.0

    def calculate_matching_score_corrected(self, code: str, text: str):
        """
        修正版的匹配度计算
        """
        try:
            # 获取分析结果
            result = self.analyze_code_text_attention(code, text)
            debug_info = self.debug_tokenization(code, text)
            
            tokens = result['tokens']
            attentions = result['attentions']
            
            # 正确分析token结构
            # 代码部分：索引1到第一个
            first_sep = tokens.index('</s>')  # 第一个</s>的位置
            second_sep = tokens.index('</s>', first_sep + 1)  # 第二个</s>的位置
            
            code_start = 1  # 跳过开始的
            code_end = first_sep  # 到第一个</s>
            text_start = second_sep + 1  # 第二个</s>之后
            text_end = len(tokens) - 1  # 到最后一个</s>之前
            
            print(f"代码部分索引: {code_start} 到 {code_end}")
            print(f"文本部分索引: {text_start} 到 {text_end}")
            print(f"代码tokens: {tokens[code_start:code_end]}")
            print(f"文本tokens: {tokens[text_start:text_end]}")
            
            # 计算跨模态注意力（使用最后一层的平均注意力头）
            last_layer_attention = attentions[-1][0]  # 最后一层，第一个样本
            avg_attention = last_layer_attention.mean(dim=0)  # 平均所有注意力头
            
            # 代码对文本的注意力
            code_to_text = avg_attention[code_start:code_end, text_start:text_end].mean()
            print(f"代码对文本的注意力: {code_to_text}")
            
            # 文本对代码的注意力  
            text_to_code = avg_attention[text_start:text_end, code_start:code_end].mean()
            print(f"文本对代码的注意力: {text_to_code}")

            # 综合得分
            cross_modal_score = (code_to_text + text_to_code) / 2
            
            # CLS相似度（使用之前的安全方法）
            cls_score = self._calculate_safe_similarity(result, debug_info)
            print(f"CLS相似度: {cls_score}")
            
            # 最终匹配度
            final_score = (cross_modal_score + cls_score) / 2
            print(f"最终匹配度: {final_score}")
            
            return {
                'code_to_text_attention': float(code_to_text),
                'text_to_code_attention': float(text_to_code),
                'cross_modal_score': float(cross_modal_score),
                'cls_similarity': cls_score,
                'final_matching_score': float(final_score),
                'token_structure': {
                    'code_tokens': tokens[code_start:code_end],
                    'text_tokens': tokens[text_start:text_end],
                    'code_range': (code_start, code_end),
                    'text_range': (text_start, text_end)
                }
            }
            
        except Exception as e:
            print(f"修正版匹配度计算失败: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    try:
        # 初始化分析器
        analyzer = PlantUMLBertAttention()
        
        # 示例代码和描述
        code = "def add(a, b): return a + b"
        text = "A function that adds two numbers"
        
        # 使用修正版本
        result = analyzer.calculate_matching_score_corrected(code, text)
        
        print("\n=== 修正版匹配度分析 ===")
        print(f"代码→文本注意力: {result['code_to_text_attention']:.4f}")
        print(f"文本→代码注意力: {result['text_to_code_attention']:.4f}")
        print(f"跨模态注意力得分: {result['cross_modal_score']:.4f}")
        print(f"CLS相似度: {result['cls_similarity']:.4f}")
        print(f"最终匹配度: {result['final_matching_score']:.4f}")
        
    except Exception as e:
        print(f"程序执行失败: {e}")