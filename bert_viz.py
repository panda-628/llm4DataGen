import re
from transformers import BertTokenizer, BertModel
from bertviz import head_view
import webbrowser
import os
import torch
import numpy as np
from typing import Dict

class PlantUMLBertAttention:
    def __init__(self, model_path: str = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess_plantuml(self, plantuml_text: str) -> str:
        # 移除PlantUML语法标记
        text = re.sub(r'@startuml|@enduml', '', plantuml_text)
        
        # 清理格式
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = re.sub(r'[{}]', '', text)  # 移除大括号
        text = re.sub(r'[-+#~]', '', text)  # 移除可见性修饰符
        text = text.strip()
        
        print("处理后的plantuml:",text)
        return text

    def preprocess_description(self, description: str) -> str:
        # 移除markdown格式
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', description)  # 移除粗体
        text = re.sub(r'`([^`]+)`', r'\1', description)  # 移除代码标记
        text = re.sub(r'#+\s*', '', text)  # 移除标题标记
        text = re.sub(r'\s+', ' ', text).strip()  # 合并空格
        
        return text

    def visualize_attention_head_view(self, sentence_a: str, sentence_b: str):
        inputs = self.tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']

        outputs = self.model(input_ids, token_type_ids=token_type_ids)
        attention = outputs[-1]  # attention 是一个包含每层注意力的元组

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # head_view(attention, tokens, sentence_b_start=token_type_ids[0].tolist().index(1))
        html_obj = head_view(attention, tokens, sentence_b_start=token_type_ids[0].tolist().index(1),html_action = 'return')
        with open("attention_head_view.html", "w", encoding="utf-8") as f:
            f.write(html_obj.data)

        # 自动在浏览器中打开
        webbrowser.open('file://' + os.path.realpath("attention_head_view.html"))

    def get_attention_weights(self, plantuml_text: str, description: str, max_length: int = 512) -> Dict:
        """
        获取PlantUML和描述之间的注意力权重
        
        Args:
            plantuml_text: PlantUML类图文本
            description: 系统描述文本
            max_length: 最大序列长度
            
        Returns:
            包含tokens和注意力权重的字典
        """
        # 预处理文本
        clean_plantuml = self.preprocess_plantuml(plantuml_text)
        clean_description = self.preprocess_description(description)
        
        # 构建输入序列: [CLS] PlantUML [SEP] Description [SEP]
        encoding = self.tokenizer(
            clean_plantuml,
            clean_description,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=False
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # 获取tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 找到分隔符位置
        sep_positions = [i for i, token in enumerate(tokens) if token == '[SEP]']
        if len(sep_positions) < 2:
            raise ValueError("未找到足够的[SEP]分隔符")
        
        # 确定PlantUML和Description的token位置
        plantuml_start = 1  # 跳过[CLS]
        plantuml_end = sep_positions[0] - 1
        desc_start = sep_positions[0] + 1
        desc_end = sep_positions[1] - 1
        
        # 获取实际的tokens（去除padding）
        actual_length = attention_mask.sum().item()
        plantuml_tokens = tokens[plantuml_start:plantuml_end+1]
        desc_tokens = tokens[desc_start:min(desc_end+1, actual_length)]
        
        # 前向传播获取注意力权重
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            attentions = outputs.attentions
        
        # 提取PlantUML和Description之间的注意力权重
        attention_data = []
        for layer_idx, layer_attention in enumerate(attentions):
            # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
            layer_attention = layer_attention[0]  # 移除batch维度
            
            # 提取PlantUML -> Description的注意力
            plantuml_to_desc = layer_attention[:, plantuml_start:plantuml_end+1, desc_start:min(desc_end+1, actual_length)]
            
            # 提取Description -> PlantUML的注意力
            desc_to_plantuml = layer_attention[:, desc_start:min(desc_end+1, actual_length), plantuml_start:plantuml_end+1]
            
            attention_data.append({
                'layer': layer_idx,
                'plantuml_to_desc': plantuml_to_desc,
                'desc_to_plantuml': desc_to_plantuml
            })
        
        return {
            'plantuml_tokens': plantuml_tokens,
            'desc_tokens': desc_tokens,
            'attention_data': attention_data,
            'processed_texts': {
                'plantuml': clean_plantuml,
                'description': clean_description
            },
            'input_ids': input_ids,
            'tokens': tokens,
            'attentions': attentions
        }

    def analyze_attention_summary(self, result: Dict, layer_idx: int = 6) -> Dict:
        """
        分析注意力权重的统计摘要
        
        Args:
            result: get_attention_weights的返回结果
            layer_idx: 要分析的层索引
            
        Returns:
            统计摘要信息
        """
        attention_layer = result['attention_data'][layer_idx]
        
        # 计算平均注意力权重（跨所有头）
        plantuml_to_desc_avg = attention_layer['plantuml_to_desc'].mean(dim=0).numpy()
        desc_to_plantuml_avg = attention_layer['desc_to_plantuml'].mean(dim=0).numpy()
        
        # 找到最高注意力权重的对应关系
        plantuml_tokens = result['plantuml_tokens']
        desc_tokens = result['desc_tokens']
        
        # PlantUML -> Description 最强对应
        p2d_max_idx = np.unravel_index(np.argmax(plantuml_to_desc_avg), plantuml_to_desc_avg.shape)
        p2d_max_weight = plantuml_to_desc_avg[p2d_max_idx]
        p2d_correspondence = (plantuml_tokens[p2d_max_idx[0]], desc_tokens[p2d_max_idx[1]], p2d_max_weight)
        
        # Description -> PlantUML 最强对应
        d2p_max_idx = np.unravel_index(np.argmax(desc_to_plantuml_avg), desc_to_plantuml_avg.shape)
        d2p_max_weight = desc_to_plantuml_avg[d2p_max_idx]
        d2p_correspondence = (desc_tokens[d2p_max_idx[0]], plantuml_tokens[d2p_max_idx[1]], d2p_max_weight)
        
        return {
            'layer': layer_idx,
            'plantuml_token_count': len(plantuml_tokens),
            'desc_token_count': len(desc_tokens),
            'avg_attention_p2d': float(plantuml_to_desc_avg.mean()),
            'avg_attention_d2p': float(desc_to_plantuml_avg.mean()),
            'max_correspondence_p2d': p2d_correspondence,
            'max_correspondence_d2p': d2p_correspondence
        }

def demo_usage():
    """演示如何使用PlantUMLBertAttention类"""

    # 示例数据
    plantuml_example = """
    @startuml
    class Dog{
        + name: String
        + bark(): void
    }
    @enduml
    """

    description_example = """
    The Dog has it own name and it can bark.
    """

    # 初始化分析器（如果有本地BERT模型，请替换路径）
    analyzer = PlantUMLBertAttention('bert-base-uncased')

    print("正在分析PlantUML和描述之间的注意力关系...")

    # 获取注意力权重
    result = analyzer.get_attention_weights(plantuml_example, description_example)

    print(f"PlantUML tokens数量: {len(result['plantuml_tokens'])}")
    print(f"Description tokens数量: {len(result['desc_tokens'])}")
    print("\nPlantUML tokens:", result['plantuml_tokens'][:10], "...")
    print("Description tokens:", result['desc_tokens'][:10], "...")

    # 使用BertViz生成所有可视化
    sentence_a = "class Dog name String bark void"
    sentence_b = " The Dog has it own name and it can bark. "
    analyzer.visualize_attention_head_view(sentence_a, sentence_b)

    # 分析统计摘要
    layers_to_analyze = [2, 6, 10]  # 浅层、中层、深层

    for layer in layers_to_analyze:
        print(f"\n=== 分析第{layer}层注意力 ===")
        
        # 获取统计摘要
        summary = analyzer.analyze_attention_summary(result, layer)
        print(f"平均注意力权重 (PlantUML->Description): {summary['avg_attention_p2d']:.4f}")
        print(f"平均注意力权重 (Description->PlantUML): {summary['avg_attention_d2p']:.4f}")
        print(f"最强对应 (PlantUML->Description): {summary['max_correspondence_p2d'][0]} -> {summary['max_correspondence_p2d'][1]} ({summary['max_correspondence_p2d'][2]:.4f})")


if __name__ == "__main__":
    # sentence_a = """
    # ```plantuml
    #     @startuml
    #     interface Controllable {
    #         turnOn(): void
    #         turnOff(): void
    #         getStatus(): String
    #     }

    #     abstract class Device {
    #         + deviceID: String
    #     }

    #     class SecurityDevice extends Device implements Controllable {
    #         + batteryLevel: int
    #         + activateAlarm(): void
    #         + validateAccessCode(accessCode: String): boolean
    #         + turnOn(): void
    #         + turnOff(): void
    #         + getStatus(): String
    #     }

    #     class ClimateDevice extends Device implements Controllable {
    #         + currentTemperature: float
    #         + adjustSettings(newTemp: float): void
    #         + turnOn(): void
    #         + turnOff(): void
    #         + getStatus(): String
    #     }

    #     class Camera extends SecurityDevice {
    #         + resolution: String
    #         + motionDetected(): boolean
    #         + recordVideo(duration: int): String
    #     }

    #     class DoorLock extends SecurityDevice {
    #         + lockState: String
    #         + unlockCode: String
    #         + autoLockEnabled: boolean
    #         + lock(): void
    #         + unlock(): void
    #     }

    #     class IndoorCamera extends Camera {
    #         + roomLocation: String
    #     }
    #     @enduml
    #     """
    # sentence_b = """The Smart Home System enables centralized management and control of interconnected devices within a residential environment. It primarily features two categories of devices: security devices and climate control devices. All devices share the core functionality of being turned on/off and having their operational status monitored.  

    #             Security devices offer intrusion detection and system-triggered alarm activation capabilities. These include surveillance cameras that detect motion, record video footage at specified resolutions, and indoor-specific cameras installed in designated rooms. Door lock mechanisms provide keyless entry through code validation, support manual or automatic locking, and track lock/unlock states. All security devices include battery level monitoring.  

    #             Climate control devices regulate indoor temperature conditions. These devices continuously monitor ambient temperature levels and allow users to modify thermostat settings to achieve desired comfort levels.  

    #             The system supports integration of multiple unique devices under a unified interface, facilitating coordinated automation for enhanced home security and environmental comfort.
    #             """
    # analyzer = PlantUMLBertAttention('bert-base-uncased')
    # sentence_a = "the animal didn't cross the street because it was too tired"
    # sentence_b = "the animal didn't cross the street because it was too tired"
    # analyzer.visualize_attention_head_view(sentence_a, sentence_b)
    demo_usage()