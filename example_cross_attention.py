"""
交叉注意力计算示例
演示如何计算两个输入之间每个词与词的权重关系
"""

from cross_attention_calculator import CrossAttentionCalculator
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 初始化计算器
    model_path = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'
    calculator = CrossAttentionCalculator(model_path)
    
    # 两个输入文本
    text1 = "One Doctor is associated with multiple Requisitions."
    text2 = "A doctor may also indicate that the tests on a requisition are to be repeated for a specified number of times and interval."
    
    print("=== 两个输入的交叉注意力分析 ===")
    print(f"输入1: {text1}")
    print(f"输入2: {text2}")
    print("=" * 80)
    
    # 1. 获取词对词权重矩阵
    print("\n1. 获取词对词权重矩阵（最后一层，平均聚合）...")
    a_to_b_df, b_to_a_df = calculator.get_word_to_word_weights(
        text1, text2, 
        layer_idx=-1, 
        aggregation='mean_heads'
    )
    
    print("\n输入1 -> 输入2 的注意力权重:")
    print(a_to_b_df.round(4))
    
    print("\n输入2 -> 输入1 的注意力权重:")
    print(b_to_a_df.round(4))
    
    # 2. 找出最高权重的词对
    print("\n2. 最高权重的词对关系:")
    
    # 输入1->输入2的最高权重
    max_val_a_to_b = a_to_b_df.max().max()
    max_pos_a_to_b = a_to_b_df.stack().idxmax()
    print(f"   输入1->输入2: '{max_pos_a_to_b[0]}' -> '{max_pos_a_to_b[1]}' (权重: {max_val_a_to_b:.4f})")
    
    # 输入2->输入1的最高权重
    max_val_b_to_a = b_to_a_df.max().max()
    max_pos_b_to_a = b_to_a_df.stack().idxmax()
    print(f"   输入2->输入1: '{max_pos_b_to_a[0]}' -> '{max_pos_b_to_a[1]}' (权重: {max_val_b_to_a:.4f})")
    
    # 3. 分析不同层的权重
    print("\n3. 不同层的权重比较:")
    layers_to_compare = [0, 3, 6, 9, 11]  # 选择几个层进行比较
    
    for layer_idx in layers_to_compare:
        a_to_b_layer, b_to_a_layer = calculator.get_word_to_word_weights(
            text1, text2, 
            layer_idx=layer_idx, 
            aggregation='mean_heads'
        )
        
        max_weight = max(a_to_b_layer.max().max(), b_to_a_layer.max().max())
        mean_weight = (a_to_b_layer.mean().mean() + b_to_a_layer.mean().mean()) / 2
        
        print(f"   第 {layer_idx+1} 层: 最大权重={max_weight:.4f}, 平均权重={mean_weight:.4f}")
    
    # 4. 比较不同聚合方式
    print("\n4. 不同聚合方式的比较（最后一层）:")
    aggregations = ['mean_heads', 'sum_heads', 'max_heads']
    
    for agg in aggregations:
        a_to_b_agg, b_to_a_agg = calculator.get_word_to_word_weights(
            text1, text2, 
            layer_idx=-1, 
            aggregation=agg
        )
        
        max_weight = max(a_to_b_agg.max().max(), b_to_a_agg.max().max())
        mean_weight = (a_to_b_agg.mean().mean() + b_to_a_agg.mean().mean()) / 2
        
        print(f"   {agg}: 最大权重={max_weight:.4f}, 平均权重={mean_weight:.4f}")
    
    # 5. 保存详细结果
    print("\n5. 保存详细分析结果...")
    calculator.save_detailed_results(text1, text2, "detailed_cross_attention_results.json")
    
    # 6. 生成可视化
    print("\n6. 生成可视化图表...")
    try:
        calculator.visualize_cross_attention(
            text1, text2, 
            layer_idx=-1, 
            aggregation='mean_heads',
            save_path="cross_attention_heatmap.png"
        )
        print("可视化图表已生成!")
    except Exception as e:
        print(f"可视化生成失败: {e}")
        print("您可能需要安装 matplotlib 和 seaborn: pip install matplotlib seaborn")
    
    # 7. 输出CSV格式的结果
    print("\n7. 保存CSV格式的权重矩阵...")
    a_to_b_df.to_csv("text1_to_text2_weights.csv", encoding='utf-8')
    b_to_a_df.to_csv("text2_to_text1_weights.csv", encoding='utf-8')
    print("CSV文件已保存:")
    print("   - text1_to_text2_weights.csv")
    print("   - text2_to_text1_weights.csv")
    
    print("\n=== 分析完成! ===")

if __name__ == "__main__":
    main()
