from plantuml_bert_attention import PlantUMLBertAttention

def analyze_your_data():
    """
    分析您自己的PlantUML类图和系统描述
    """
    # 您的PlantUML类图 (I)
    your_plantuml = """
    @startuml
    class BankAccount {
        - accountNumber: String
        - balance: Double
        - accountHolder: String
        + deposit(amount: Double)
        + withdraw(amount: Double)
        + getBalance(): Double
    }
    
    class Transaction {
        - transactionId: String
        - amount: Double
        - timestamp: Date
        - type: String
        + execute()
    }
    
    class Customer {
        - customerId: String
        - name: String
        - phone: String
        + openAccount()
        + closeAccount()
    }
    
    Customer "1" --> "many" BankAccount : owns
    BankAccount "1" --> "many" Transaction : generates
    @enduml
    """
    
    # 您的系统描述 (O)  
    your_description = """
    This banking system allows customers to manage their bank accounts and transactions. 
    Each customer has a unique ID, name, and phone number. Customers can open and close accounts.
    Bank accounts have account numbers, balances, and account holders. Accounts support deposit, 
    withdrawal, and balance inquiry operations. Every transaction has an ID, amount, timestamp, 
    and type, and can be executed to modify account balances.
    """
    
    # 初始化分析器
    print("初始化BERT模型...")
    analyzer = PlantUMLBertAttention()  # 使用HuggingFace的模型
    # 如果您有本地模型，可以这样使用:
    # analyzer = PlantUMLBertAttention('C:/path/to/your/bert/model')
    
    print("分析注意力权重...")
    result = analyzer.get_attention_weights(your_plantuml, your_description)
    
    # 显示基本信息
    print(f"\n=== 分析结果 ===")
    print(f"PlantUML tokens数量: {len(result['plantuml_tokens'])}")
    print(f"Description tokens数量: {len(result['desc_tokens'])}")
    
    print(f"\n处理后的PlantUML文本: {result['processed_texts']['plantuml']}")
    print(f"\n处理后的Description文本: {result['processed_texts']['description']}")
    
    # 分析不同层的注意力
    interesting_layers = [2, 6, 10]  # 浅层、中层、深层
    
    for layer in interesting_layers:
        print(f"\n=== 第{layer}层注意力分析 ===")
        
        # 获取统计摘要
        summary = analyzer.analyze_attention_summary(result, layer)
        print(f"PlantUML tokens: {summary['plantuml_token_count']}")
        print(f"Description tokens: {summary['desc_token_count']}")
        print(f"平均注意力 (PlantUML→Description): {summary['avg_attention_p2d']:.4f}")
        print(f"平均注意力 (Description→PlantUML): {summary['avg_attention_d2p']:.4f}")
        
        # 显示最强对应关系
        p2d_corr = summary['max_correspondence_p2d']
        d2p_corr = summary['max_correspondence_d2p']
        print(f"最强对应 (PlantUML→Description): '{p2d_corr[0]}' → '{p2d_corr[1]}' (权重: {p2d_corr[2]:.4f})")
        print(f"最强对应 (Description→PlantUML): '{d2p_corr[0]}' → '{d2p_corr[1]}' (权重: {d2p_corr[2]:.4f})")
        
        # 生成热力图
        print(f"\n生成第{layer}层热力图...")
        
        # PlantUML → Description 注意力
        analyzer.visualize_attention(
            result, 
            layer_idx=layer, 
            head_idx=0,  # 使用第0个注意力头
            direction='plantuml_to_desc',
            figsize=(14, 10),
            save_path=f'attention_p2d_layer_{layer}.png'
        )
        
        # Description → PlantUML 注意力  
        analyzer.visualize_attention(
            result,
            layer_idx=layer,
            head_idx=0,
            direction='desc_to_plantuml', 
            figsize=(14, 10),
            save_path=f'attention_d2p_layer_{layer}.png'
        )


def analyze_from_files(plantuml_file: str, description_file: str):
    """
    从文件加载PlantUML和描述进行分析
    
    Args:
        plantuml_file: PlantUML文件路径
        description_file: 描述文件路径
    """
    # 读取文件
    with open(plantuml_file, 'r', encoding='utf-8') as f:
        plantuml_text = f.read()
    
    with open(description_file, 'r', encoding='utf-8') as f:
        description_text = f.read()
    
    # 初始化分析器
    analyzer = PlantUMLBertAttention('bert-base-uncased')
    
    # 进行分析
    result = analyzer.get_attention_weights(plantuml_text, description_text)
    
    # 可视化中层注意力（通常最有意义）
    analyzer.visualize_attention(result, layer_idx=6, direction='plantuml_to_desc')
    analyzer.visualize_attention(result, layer_idx=6, direction='desc_to_plantuml')
    
    return result


if __name__ == "__main__":
    # 方式1: 直接分析硬编码的数据
    analyze_your_data()
    
    # 方式2: 从文件加载数据进行分析（取消注释以使用）
    # result = analyze_from_files('your_diagram.puml', 'your_description.txt')
    
    print("\n分析完成！热力图已显示并保存为PNG文件。") 