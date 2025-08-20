"""
使用示例：演示如何处理子词权重聚合
"""

from mapping_weight_analyzer import MappingWeightAnalyzer

def main():
    # 示例：比较不同的子词处理策略
    result_folder = "attention_heads_analysis_20250818_163445"
    
    print("=== 子词权重处理示例 ===\n")
    
    # 定义包含可能被分词的映射
    positive_mappings = {
        "requisition": "patient",  # 可能被分成 requis##ition
        "patient": "requisitions", # 可能变成复数形式
        "show": "associated"       # 简单词，不会被分割
    }
    
    negative_mappings = {
        "information": "patient",
        "show": "is"
    }
    
    # 测试不同的聚合方法
    methods = ["mean", "max", "sum", "first"]
    
    for method in methods:
        print(f"\n--- 使用 {method.upper()} 聚合方法 ---")
        
        try:
            # 启用子词聚合
            analyzer = MappingWeightAnalyzer(
                result_folder, 
                use_subword_aggregation=True,
                aggregation_method=method
            )
            
            analyzer.define_mappings(positive_mappings, negative_mappings)
            
            # 测试单个文件
            test_file = f"{result_folder}/csv_files/layer_00_head_00_text2_to_text1.csv"
            pos_weights = analyzer.extract_weights_from_csv(test_file, positive_mappings)
            
            print(f"正例权重 ({method}):")
            for mapping, weight in pos_weights.items():
                print(f"  {mapping}: {weight:.6f}")
                
        except Exception as e:
            print(f"错误: {e}")
    
    print(f"\n--- 对比：禁用子词聚合 ---")
    try:
        # 禁用子词聚合
        analyzer_basic = MappingWeightAnalyzer(
            result_folder, 
            use_subword_aggregation=False
        )
        
        analyzer_basic.define_mappings(positive_mappings, negative_mappings)
        
        test_file = f"{result_folder}/csv_files/layer_00_head_00_text2_to_text1.csv"
        pos_weights_basic = analyzer_basic.extract_weights_from_csv(test_file, positive_mappings)
        
        print(f"正例权重 (基础方法):")
        for mapping, weight in pos_weights_basic.items():
            print(f"  {mapping}: {weight:.6f}")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
