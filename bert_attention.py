import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re

class UMLBertAttentionAnalyzer:
    def __init__(self, model_path: str = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'):
        """初始化BERT模型和分词器，专门用于UML类图到自然语言描述的注意力分析"""
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(
            model_path, 
            output_attentions=True,
            attn_implementation="eager"
        )
        self.model.eval()
    
    def preprocess_plantuml(self, plantuml_text: str) -> str:
        """预处理PlantUML文本，提取关键信息"""
        # 移除PlantUML语法标记
        text = re.sub(r'@startuml.*?@enduml', '', plantuml_text, flags=re.DOTALL)
        text = re.sub(r'@startuml|@enduml', '', text)
        
        # 清理格式
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = re.sub(r'[{}]', '', text)  # 移除大括号
        text = re.sub(r'[-+#~]', '', text)  # 移除可见性修饰符
        text = text.strip()
        
        return text
    
    def preprocess_description(self, description_text: str) -> str:
        """预处理自然语言描述，移除元信息"""
        # 移除模型描述的元信息部分
        if 'model_description:' in description_text:
            # 查找实际的系统描述部分
            if '**System Description**' in description_text:
                start = description_text.find('**System Description**')
                end = description_text.find('---', start)
                if end == -1:
                    end = description_text.find('### Key Improvements:', start)
                if end != -1:
                    description_text = description_text[start:end]
            else:
                # 如果没有找到特定标记，尝试提取主要描述
                lines = description_text.split('\n')
                cleaned_lines = []
                skip_meta = False
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['verification', 'identified issues', 'corrected description', 'key improvements']):
                        skip_meta = True
                    if not skip_meta and line.strip() and not line.startswith('#'):
                        cleaned_lines.append(line.strip())
                description_text = ' '.join(cleaned_lines)
        
        # 移除markdown格式
        description_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', description_text)  # 移除粗体
        description_text = re.sub(r'`([^`]+)`', r'\1', description_text)  # 移除代码标记
        description_text = re.sub(r'#+\s*', '', description_text)  # 移除标题标记
        description_text = re.sub(r'\s+', ' ', description_text).strip()
        
        return description_text
    
    def _prepare_inputs(self, plantuml_text: str, description_text: str) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """准备模型输入并计算关键token位置"""
        # 预处理输入
        processed_plantuml = self.preprocess_plantuml(plantuml_text)
        processed_description = self.preprocess_description(description_text)
        
        # 分词
        uml_inputs = self.tokenizer(processed_plantuml, return_tensors="pt", truncation=True, max_length=256)
        desc_inputs = self.tokenizer(processed_description, return_tensors="pt", truncation=True, max_length=256)
        
        # 处理tokens
        uml_tokens = uml_inputs['input_ids'][0, 1:-1]  # 移除[CLS]和[SEP]
        desc_tokens = desc_inputs['input_ids'][0, 1:-1]  # 移除[CLS]和[SEP]
        
        if len(uml_tokens) == 0:
            uml_tokens = torch.tensor([self.tokenizer.unk_token_id])
        if len(desc_tokens) == 0:
            desc_tokens = torch.tensor([self.tokenizer.unk_token_id])
            
        # 构建组合输入 [CLS] UML [SEP] DESC [SEP]
        combined_input = torch.cat([
            torch.tensor([[self.tokenizer.cls_token_id]]),
            uml_tokens.unsqueeze(0),
            torch.tensor([[self.tokenizer.sep_token_id]]),
            desc_tokens.unsqueeze(0),
            torch.tensor([[self.tokenizer.sep_token_id]])
        ], dim=1)
        
        # 计算关键token位置
        sep_positions = (combined_input == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[1]
        if len(sep_positions) < 2:
            raise ValueError("组合输入中SEP tokens不足")
            
        uml_start, uml_end = 1, sep_positions[0].item() - 1
        desc_start, desc_end = sep_positions[0].item() + 1, sep_positions[1].item() - 1
        
        return combined_input, (uml_start, uml_end, desc_start, desc_end)
    
    def analyze_uml_description_attention(self, plantuml_text: str, description_text: str) -> Dict:
        """分析PlantUML类图和自然语言描述之间的注意力"""
        combined_input, (uml_start, uml_end, desc_start, desc_end) = self._prepare_inputs(plantuml_text, description_text)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=combined_input,
                attention_mask=torch.ones_like(combined_input)
            )
            
        # 处理注意力权重
        tokens = self.tokenizer.convert_ids_to_tokens(combined_input[0])
        uml_desc_attention = []
        
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            uml_desc_attention.append({
                'layer': layer_idx,
                'uml_to_desc': layer_attn[0, :, uml_start:uml_end+1, desc_start:desc_end+1],
                'desc_to_uml': layer_attn[0, :, desc_start:desc_end+1, uml_start:uml_end+1]
            })
            
        return {
            'tokens': {
                'uml': tokens[uml_start:uml_end+1],
                'description': tokens[desc_start:desc_end+1]
            },
            'attention': uml_desc_attention,
            'processed_texts': {
                'uml': self.preprocess_plantuml(plantuml_text),
                'description': self.preprocess_description(description_text)
            }
        }

# 为UML特化的可视化函数
def visualize_uml_attention(result: Dict, layer_idx: int = 6, head_idx: int = 0, direction: str = 'uml_to_desc') -> None:
    """可视化UML类图到描述的注意力权重
    
    Args:
        result: 注意力分析结果
        layer_idx: 要可视化的层索引
        head_idx: 要可视化的头索引  
        direction: 'uml_to_desc' 或 'desc_to_uml'
    """
    try:
        if direction == 'uml_to_desc':
            attn_weights = result['attention'][layer_idx]['uml_to_desc'][head_idx].numpy()
            source_tokens = result['tokens']['uml']
            target_tokens = result['tokens']['description']
            title_prefix = 'UML-to-Description'
            xlabel = 'Description Tokens'
            ylabel = 'UML Tokens'
        else:
            attn_weights = result['attention'][layer_idx]['desc_to_uml'][head_idx].numpy()
            source_tokens = result['tokens']['description'] 
            target_tokens = result['tokens']['uml']
            title_prefix = 'Description-to-UML'
            xlabel = 'UML Tokens'
            ylabel = 'Description Tokens'
        
        if not source_tokens or not target_tokens or attn_weights.size == 0:
            print(f"警告：无法生成热力图 - tokens为空或注意力矩阵为空")
            return
            
        plt.figure(figsize=(max(12, len(target_tokens)*0.8), max(10, len(source_tokens)*0.6)))
        sns.heatmap(
            attn_weights, 
            annot=True, 
            xticklabels=target_tokens,
            yticklabels=source_tokens,
            fmt=".3f",
            cmap='Blues'
        )
        plt.title(f'{title_prefix} Attention (Layer {layer_idx}, Head {head_idx})')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"可视化过程中出错: {str(e)}")

def load_uml_and_description(uml_file: str, desc_file: str) -> Tuple[str, str]:
    """从文件加载UML和描述文本"""
    with open(uml_file, 'r', encoding='utf-8') as f:
        uml_text = f.read()
    
    with open(desc_file, 'r', encoding='utf-8') as f:
        desc_text = f.read()
    
    return uml_text, desc_text

if __name__ == "__main__":
    analyzer = UMLBertAttentionAnalyzer()
    
    # 示例：分析example1的UML和描述
    try:
        uml_text, desc_text = load_uml_and_description(
            'labResult/deepseek-coder-0615-1334- example4/1/domain.puml',
            'labResult/deepseek-coder-0615-1334- example4/1/description.txt'
        )
        
        result = analyzer.analyze_uml_description_attention(uml_text, desc_text)
        
        print("分析完成！")
        print(f"UML tokens: {len(result['tokens']['uml'])}")
        print(f"Description tokens: {len(result['tokens']['description'])}")
        print(f"Processed UML: {result['processed_texts']['uml'][:100]}...")
        print(f"Processed Description: {result['processed_texts']['description'][:100]}...")
        
        # 可视化不同方向的注意力
        visualize_uml_attention(result, layer_idx=6, head_idx=0, direction='uml_to_desc')
        visualize_uml_attention(result, layer_idx=6, head_idx=0, direction='desc_to_uml')
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("使用示例文本进行演示...")
        
        # 使用示例文本
        example_uml = """
        @startuml
        class User {
            - id: String
            - name: String
            + login()
            + logout()
        }
        
        class Order {
            - orderId: String
            - amount: Double
            + createOrder()
            + cancelOrder()
        }
        
        User "1" --> "many" Order : creates
        @enduml
        """
        
        example_desc = "The system allows users to manage orders. Users can login and logout, while orders can be created and cancelled. Each user can create multiple orders."
        
        result = analyzer.analyze_uml_description_attention(example_uml, example_desc)
        visualize_uml_attention(result)
