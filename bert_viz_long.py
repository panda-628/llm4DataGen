import re
from transformers import BertTokenizer, BertModel
from bertviz import head_view
import webbrowser
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json

class PlantUMLBertAttention:
    def __init__(self, model_path: str = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'):
        self.model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 512

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

    def extract_key_plantuml_elements(self, plantuml_text: str, max_tokens: int = 200) -> str:
        """
        从PlantUML中提取关键元素（类名、方法名等）
        """
        # 预处理
        clean_text = self.preprocess_plantuml(plantuml_text)
        
        # 提取关键元素
        key_elements = []
        
        # 提取类名
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, clean_text, re.IGNORECASE)
        key_elements.extend([f"class {cls}" for cls in classes])
        
        # 提取接口名
        interface_pattern = r'interface\s+(\w+)'
        interfaces = re.findall(interface_pattern, clean_text, re.IGNORECASE)
        key_elements.extend([f"interface {iface}" for iface in interfaces])
        
        # 提取方法名
        method_pattern = r'(\w+)\s*\([^)]*\)\s*:\s*(\w+)'
        methods = re.findall(method_pattern, clean_text)
        key_elements.extend([f"{method} {return_type}" for method, return_type in methods])
        
        # 提取属性
        attribute_pattern = r'(\w+)\s*:\s*(\w+)'
        attributes = re.findall(attribute_pattern, clean_text)
        key_elements.extend([f"{attr} {attr_type}" for attr, attr_type in attributes])
        
        # 提取关系
        relation_patterns = [
            r'(\w+)\s+extends\s+(\w+)',
            r'(\w+)\s+implements\s+(\w+)',
            r'(\w+)\s*-->\s*(\w+)',
            r'(\w+)\s*--\s*(\w+)'
        ]
        
        for pattern in relation_patterns:
            relations = re.findall(pattern, clean_text, re.IGNORECASE)
            key_elements.extend([f"{rel[0]} relates {rel[1]}" for rel in relations])
        
        # 组合关键元素
        key_text = ' '.join(key_elements)
        
        # 检查长度并截断
        tokens = self.tokenizer.tokenize(key_text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            key_text = self.tokenizer.convert_tokens_to_string(tokens)
        
        return key_text

    def extract_key_description_elements(self, description: str, max_tokens: int = 200) -> str:
        """
        从系统描述中提取关键信息
        """
        clean_desc = self.preprocess_description(description)
        
        # 按句子分割
        sentences = re.split(r'[.!?]+', clean_desc)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 提取包含关键词的句子
        key_words = ['class', 'interface', 'system', 'device', 'method', 'function', 
                    'attribute', 'property', 'extends', 'implements', 'inherit']
        
        key_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in key_words):
                key_sentences.append(sentence)
        
        # 如果没有找到关键句子，使用前几句
        if not key_sentences:
            key_sentences = sentences[:3]
        
        key_text = ' '.join(key_sentences)
        
        # 检查长度并截断
        tokens = self.tokenizer.tokenize(key_text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            key_text = self.tokenizer.convert_tokens_to_string(tokens)
        
        return key_text

    def check_input_length(self, text1: str, text2: str) -> Tuple[bool, Dict]:
        """
        检查输入文本的长度是否适合BERT处理
        """
        tokens1 = self.tokenizer.tokenize(text1)
        tokens2 = self.tokenizer.tokenize(text2)
        
        # 考虑特殊token: [CLS], [SEP], [SEP]
        total_tokens = len(tokens1) + len(tokens2) + 3
        
        return total_tokens <= self.max_length, {
            'text1_tokens': len(tokens1),
            'text2_tokens': len(tokens2),
            'total_tokens': total_tokens,
            'max_length': self.max_length,
            'is_truncated': total_tokens > self.max_length
        }

    def create_lightweight_visualization(self, sentence_a: str, sentence_b: str, 
                                       max_tokens_total: int = 200,
                                       selected_layers: List[int] = [2, 5, 10],
                                       selected_heads: List[int] = [0, 5, 6, 10]):
        """
        创建轻量级可视化，避免HTML文件过大
        """
        print("创建轻量级可视化...")
        
        # 确保输入文本不会太长
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_b = self.tokenizer.tokenize(sentence_b)
        
        # 根据总token限制调整文本长度
        max_a = max_tokens_total // 2
        max_b = max_tokens_total // 2
        
        if len(tokens_a) > max_a:
            tokens_a = tokens_a[:max_a]
            sentence_a = self.tokenizer.convert_tokens_to_string(tokens_a)
        
        if len(tokens_b) > max_b:
            tokens_b = tokens_b[:max_b]
            sentence_b = self.tokenizer.convert_tokens_to_string(tokens_b)
        
        print(f"调整后的文本长度: A={len(tokens_a)}, B={len(tokens_b)}")
        
        # 编码输入
        inputs = self.tokenizer.encode_plus(
            sentence_a, 
            sentence_b, 
            return_tensors='pt',
            max_length=max_tokens_total + 3,  # 加上特殊token
            truncation=True,
            padding=True
        )
        
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            attention = outputs[-1]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 只选择部分层和头进行可视化
        filtered_attention = []
        for layer_idx in selected_layers:
            if layer_idx < len(attention):
                layer_attention = attention[layer_idx]
                # 只选择部分头
                selected_layer_attention = layer_attention[:, selected_heads, :, :]
                filtered_attention.append(selected_layer_attention)
        
        # 生成轻量级可视化
        try:
            html_obj = head_view(filtered_attention, tokens, 
                               sentence_b_start=token_type_ids[0].tolist().index(1),
                               html_action='return')
            
            # 写入较小的HTML文件
            filename = "lightweight_attention.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_obj.data)

            print(f"轻量级可视化已生成: {filename}")
            print(f"只显示层: {selected_layers}, 头: {selected_heads}")
            
            # 检查文件大小
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"文件大小: {file_size:.2f} MB")
            
            if file_size < 10:  # 小于10MB才自动打开
                webbrowser.open('file://' + os.path.realpath(filename))
            else:
                print("文件仍然较大，请手动打开")
                
        except Exception as e:
            print(f"轻量级可视化生成出错: {e}")
            self.create_static_visualization(sentence_a, sentence_b)

    def create_static_visualization(self, sentence_a: str, sentence_b: str,
                                  layer_idx: int = 5,
                                  head_idx: int = 5,
                                  max_tokens_total: int = 100):
        """
        创建静态matplotlib可视化，避免交互式HTML的问题
        """
        print("创建静态可视化...")
        
        # 限制文本长度
        tokens_a = self.tokenizer.tokenize(sentence_a)[:max_tokens_total//2]
        tokens_b = self.tokenizer.tokenize(sentence_b)[:max_tokens_total//2]
        
        sentence_a = self.tokenizer.convert_tokens_to_string(tokens_a)
        sentence_b = self.tokenizer.convert_tokens_to_string(tokens_b)
        
        # 编码输入
        inputs = self.tokenizer.encode_plus(
            sentence_a, 
            sentence_b, 
            return_tensors='pt',
            max_length=max_tokens_total + 3,
            truncation=True,
            padding=True
        )
        
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            attention = outputs[-1]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 获取特定层和头的注意力
        layer_attention = attention[layer_idx][0, head_idx].numpy()
        
        # 只显示实际的token（去掉padding）
        actual_length = len([t for t in tokens if t != '[PAD]'])
        layer_attention = layer_attention[:actual_length, :actual_length]
        tokens = tokens[:actual_length]
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(layer_attention, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues',
                   cbar=True)
        
        plt.title(f'BERT Attention Heatmap (Layer {layer_idx}, Head {head_idx})')
        plt.xlabel('Target Tokens')
        plt.ylabel('Source Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图片
        filename = f"attention_static_L{layer_idx}_H{head_idx}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"静态可视化已保存: {filename}")

    def create_summary_visualization(self, sentence_a: str, sentence_b: str,
                                   max_tokens_total: int = 150):
        """
        创建注意力摘要可视化，显示关键统计信息
        """
        print("创建注意力摘要...")
        
        # 限制文本长度
        tokens_a = self.tokenizer.tokenize(sentence_a)[:max_tokens_total//2]
        tokens_b = self.tokenizer.tokenize(sentence_b)[:max_tokens_total//2]
        
        sentence_a = self.tokenizer.convert_tokens_to_string(tokens_a)
        sentence_b = self.tokenizer.convert_tokens_to_string(tokens_b)
        
        # 编码输入
        inputs = self.tokenizer.encode_plus(
            sentence_a, 
            sentence_b, 
            return_tensors='pt',
            max_length=max_tokens_total + 3,
            truncation=True,
            padding=True
        )
        
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            attention = outputs[-1]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        actual_length = len([t for t in tokens if t != '[PAD]'])
        tokens = tokens[:actual_length]
        
        # 分析不同层的注意力模式
        layer_stats = []
        for layer_idx in range(len(attention)):
            layer_att = attention[layer_idx][0].numpy()[:actual_length, :actual_length]
            
            # 计算统计信息
            avg_attention = np.mean(layer_att)
            max_attention = np.max(layer_att)
            attention_entropy = -np.sum(layer_att * np.log(layer_att + 1e-10), axis=1).mean()
            
            layer_stats.append({
                'layer': layer_idx,
                'avg_attention': avg_attention,
                'max_attention': max_attention,
                'entropy': attention_entropy
            })
        
        # 创建统计图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 平均注意力权重
        layers = [s['layer'] for s in layer_stats]
        avg_att = [s['avg_attention'] for s in layer_stats]
        axes[0, 0].plot(layers, avg_att, 'b-o')
        axes[0, 0].set_title('Average Attention by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Average Attention')
        
        # 2. 最大注意力权重
        max_att = [s['max_attention'] for s in layer_stats]
        axes[0, 1].plot(layers, max_att, 'r-o')
        axes[0, 1].set_title('Maximum Attention by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Maximum Attention')
        
        # 3. 注意力熵
        entropy = [s['entropy'] for s in layer_stats]
        axes[1, 0].plot(layers, entropy, 'g-o')
        axes[1, 0].set_title('Attention Entropy by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Entropy')
        
        # 4. 最后一层的注意力热力图（降采样）
        last_layer_att = attention[-1][0].mean(dim=0).numpy()[:actual_length, :actual_length]
        
        # 如果token太多，进行降采样
        if actual_length > 20:
            step = actual_length // 20
            sampled_att = last_layer_att[::step, ::step]
            sampled_tokens = tokens[::step]
        else:
            sampled_att = last_layer_att
            sampled_tokens = tokens
        
        im = axes[1, 1].imshow(sampled_att, cmap='Blues')
        axes[1, 1].set_title('Last Layer Attention (Averaged)')
        axes[1, 1].set_xticks(range(len(sampled_tokens)))
        axes[1, 1].set_yticks(range(len(sampled_tokens)))
        axes[1, 1].set_xticklabels(sampled_tokens, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(sampled_tokens)
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # 保存图片
        filename = "attention_summary.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"注意力摘要已保存: {filename}")
        
        # 保存数据到JSON
        with open("attention_stats.json", "w") as f:
            json.dump(layer_stats, f, indent=2)
        print("注意力统计数据已保存: attention_stats.json")

    def visualize_attention_optimized(self, sentence_a: str, sentence_b: str,
                                    method: str = "lightweight"):
        """
        优化的注意力可视化方法
        
        Args:
            sentence_a: 第一个句子
            sentence_b: 第二个句子  
            method: 可视化方法 ("lightweight", "static", "summary")
        """
        print(f"使用{method}方法进行可视化...")
        
        # 检查输入长度
        is_suitable, length_info = self.check_input_length(sentence_a, sentence_b)
        
        if not is_suitable:
            print(f"输入文本过长 ({length_info['total_tokens']} tokens)")
            print("自动提取关键信息...")
            
            # 自动提取关键信息
            if '@startuml' in sentence_a.lower() or 'class ' in sentence_a.lower():
                sentence_a = self.extract_key_plantuml_elements(sentence_a, 80)
                sentence_b = self.extract_key_description_elements(sentence_b, 80)
            else:
                sentence_b = self.extract_key_plantuml_elements(sentence_b, 80)
                sentence_a = self.extract_key_description_elements(sentence_a, 80)
        
        # 根据方法选择可视化
        if method == "lightweight":
            self.create_lightweight_visualization(sentence_a, sentence_b)
        elif method == "static":
            self.create_static_visualization(sentence_a, sentence_b)
        elif method == "summary":
            self.create_summary_visualization(sentence_a, sentence_b)
        else:
            print(f"未知方法: {method}")
            print("使用轻量级方法...")
            self.create_lightweight_visualization(sentence_a, sentence_b)

    def visualize_attention_head_view(self, sentence_a: str, sentence_b: str, 
                                    auto_extract_key: bool = True,
                                    max_tokens_per_text: int = 200):
        """
        改进的注意力可视化方法，支持长文本处理
        """
        # 检查输入长度
        is_suitable, length_info = self.check_input_length(sentence_a, sentence_b)
        
        if not is_suitable:
            print(f"警告: 输入文本过长 ({length_info['total_tokens']} > {self.max_length} tokens)")
            print("建议使用优化的可视化方法...")
            
            # 提供选择
            print("可选方法:")
            print("1. lightweight - 轻量级交互式可视化")
            print("2. static - 静态图片可视化")
            print("3. summary - 注意力摘要可视化")
            
            # 默认使用轻量级方法
            self.visualize_attention_optimized(sentence_a, sentence_b, "lightweight")
            return
            
        # 原有的方法继续处理短文本
        # ... existing code ...

    def visualize_long_text_sliding_window(self, plantuml_text: str, description: str, 
                                         window_size: int = 400, overlap: int = 100):
        """
        使用滑动窗口方法处理长文本
        """
        # 预处理
        clean_plantuml = self.preprocess_plantuml(plantuml_text)
        clean_description = self.preprocess_description(description)
        
        # 分词
        plantuml_tokens = self.tokenizer.tokenize(clean_plantuml)
        desc_tokens = self.tokenizer.tokenize(clean_description)
        
        print(f"PlantUML tokens: {len(plantuml_tokens)}")
        print(f"Description tokens: {len(desc_tokens)}")
        
        # 如果PlantUML太长，使用滑动窗口
        if len(plantuml_tokens) > window_size:
            print("PlantUML过长，使用滑动窗口处理...")
            
            windows = []
            for i in range(0, len(plantuml_tokens), window_size - overlap):
                window = plantuml_tokens[i:i + window_size]
                windows.append(self.tokenizer.convert_tokens_to_string(window))
                
                if i + window_size >= len(plantuml_tokens):
                    break
            
            # 为每个窗口生成可视化
            for idx, window in enumerate(windows):
                print(f"\n生成第{idx+1}个窗口的可视化...")
                
                # 截断描述以适应窗口
                desc_for_window = self.extract_key_description_elements(description, 200)
                
                # 使用优化的可视化方法
                self.visualize_attention_optimized(window, desc_for_window, "static")
        else:
            # 文本不长，直接处理
            self.visualize_attention_optimized(clean_plantuml, clean_description, "lightweight")

# ... existing code ...

def demo_long_text_usage():
    """演示如何处理长PlantUML和描述文本"""
    
    # 长PlantUML示例
    long_plantuml = """
    The IndividualBooking class, which implements the IBooking interface, includes attributes such as bookingId (a String unique identifier), checkInDate (a Date for the guest's arrival), and checkOutDate (a Date for the guest's departure), along with methods: calculateTotalCost() that returns a double representing the total booking cost, confirmBooking() that returns a boolean indicating successful booking confirmation, and cancelBooking() that returns a boolean indicating successful booking cancellation.
    """

    long_description = """
    The system facilitates room reservations for individual travelers, groups, and conference events. 
    It captures guest details (e.g., names, contact information) during booking creation.  
    Individual bookings require check-in/check-out dates and accommodate one or more guests in a single reserved room. 
    The system generates a unique confirmation number and sends notifications via the guest’s preferred contact method. 
    
    """

    # 初始化分析器
    analyzer = PlantUMLBertAttention('bert-base-uncased')

    print("=== 优化的长文本处理演示 ===")
    print("解决HTML文件过大问题的三种方法:")
    
    # 方法1: 轻量级交互式可视化
    print("\n🔹 方法1: 轻量级交互式可视化 (推荐)")
    print("特点: 减少层数和头数，文件较小，仍可交互")
    analyzer.visualize_attention_optimized(long_plantuml, long_description, "lightweight")
    
    # 方法2: 静态图片可视化
    print("\n🔹 方法2: 静态图片可视化")
    print("特点: 生成PNG图片，文件小，但不可交互")
    analyzer.visualize_attention_optimized(long_plantuml, long_description, "static")
    
    # 方法3: 注意力摘要可视化
    print("\n🔹 方法3: 注意力摘要可视化")
    print("特点: 显示统计信息和趋势，最适合分析")
    analyzer.visualize_attention_optimized(long_plantuml, long_description, "summary")
    
    # 方法4: 手动控制文本长度
    print("\n🔹 方法4: 手动控制文本长度")
    print("特点: 用户可以精确控制要可视化的内容")
    
    # 手动提取关键信息
    key_plantuml = analyzer.extract_key_plantuml_elements(long_plantuml, max_tokens=80)
    key_description = analyzer.extract_key_description_elements(long_description, max_tokens=80)
    
    print(f"提取的关键PlantUML: {key_plantuml[:150]}...")
    print(f"提取的关键描述: {key_description[:150]}...")
    
    # 使用提取的关键信息进行可视化
    analyzer.create_lightweight_visualization(
        key_plantuml, 
        key_description, 
        max_tokens_total=180,
        selected_layers=[6, 10],  # 只选择2层
        selected_heads=[0, 1]     # 只选择2个头
    )
    
    print("\n=== 使用建议 ===")
    print("1. 对于长文本，优先使用 'lightweight' 方法")
    print("2. 如果轻量级仍然过大，使用 'static' 方法")
    print("3. 如果需要分析整体趋势，使用 'summary' 方法")
    print("4. 可以通过调整 max_tokens_total 参数控制文件大小")
    print("5. 减少 selected_layers 和 selected_heads 可以进一步减小文件")

def demo_optimized_methods():
    """演示各种优化方法的使用"""
    
    print("=== 优化方法对比演示 ===")
    
    # 中等长度的示例
    medium_plantuml = """
    @startuml
    class User {
        + userId: String
        + username: String
        + email: String
        + login(): boolean
        + logout(): void
        + updateProfile(): void
    }
    
    class Order {
        + orderId: String
        + userId: String
        + orderDate: Date
        + totalAmount: float
        + status: String
        + createOrder(): boolean
        + updateStatus(): void
        + calculateTotal(): float
    }
    
    class Product {
        + productId: String
        + name: String
        + price: float
        + category: String
        + stock: int
        + updateStock(): void
        + getPrice(): float
    }
    
    User ||--o{ Order : places
    Order ||--o{ Product : contains
    @enduml
    """
    
    medium_description = """
    This e-commerce system manages users, orders, and products. Users can place multiple orders, 
    and each order can contain multiple products. The system tracks user information, order details, 
    and product inventory. Users authenticate through login/logout functionality and can update their profiles.
    Orders have lifecycle management with status tracking and total calculation. Products maintain 
    inventory levels and pricing information.
    """
    
    analyzer = PlantUMLBertAttention('bert-base-uncased')
    
    print("文本长度适中的情况:")
    is_suitable, length_info = analyzer.check_input_length(medium_plantuml, medium_description)
    print(f"总token数: {length_info['total_tokens']}")
    print(f"是否适合直接处理: {is_suitable}")
    
    if is_suitable:
        print("\n✅ 可以使用标准方法")
        analyzer.visualize_attention_head_view(medium_plantuml, medium_description, auto_extract_key=False)
    else:
        print("\n⚠️ 需要使用优化方法")
        analyzer.visualize_attention_optimized(medium_plantuml, medium_description, "lightweight")

if __name__ == "__main__":
    print("=== PlantUML BERT 注意力可视化工具 ===")
    print("解决HTML文件过大问题的优化版本\n")
    
    # 让用户选择演示类型
    while True:
        print("请选择演示类型:")
        print("1. 长文本优化处理演示")
        print("2. 中等长度文本对比演示")
        print("3. 退出")
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == "1":
            demo_long_text_usage()
            break
        elif choice == "2":
            demo_optimized_methods()
            break
        elif choice == "3":
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")