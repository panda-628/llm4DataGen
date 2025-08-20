"""
运行所有注意力头分析的简化版本
每一层每一个头的词与词权重都会被单独保存到文件中
"""

from detailed_head_analysis import DetailedHeadAnalyzer

def main():
    print("=" * 80)
    print("所有注意力头详细分析")
    print("=" * 80)
    
    # 模型路径
    model_path = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased'
    
    # 两个输入文本
    text1 = "The item class includes attributes: description、dimension、weight"
    text2 = "The route takes into account the available storage space of a vehicle and the dimensions and weights of scheduled items."
    
    print(f"输入文本1: {text1}")
    print(f"输入文本2: {text2}")
    print()
    
    # 初始化分析器
    print("正在加载BERT模型...")
    analyzer = DetailedHeadAnalyzer(model_path)
    print("模型加载完成!")
    print()
    
    # 执行完整分析
    print("开始分析所有注意力头...")
    print("这将创建一个包含以下内容的文件夹:")
    print("├── csv_files/          # 每个头的权重矩阵CSV文件")
    print("├── json_files/         # 每个头的统计信息JSON文件")
    print("├── summary/            # 汇总报告")
    print("└── input_info.json     # 输入信息")
    print()
    
    try:
        results = analyzer.analyze_all_heads(text1, text2)
        
        print("\n" + "=" * 80)
        print("分析完成!")
        print("=" * 80)
        print(f"结果保存位置: {results['output_folder']}")
        print()
        print("生成的文件结构:")
        print(f"├── {results['output_folder']}/")
        print("│   ├── csv_files/")
        print("│   │   ├── layer_00_head_00_text1_to_text2.csv")
        print("│   │   ├── layer_00_head_00_text2_to_text1.csv")
        print("│   │   ├── layer_00_head_01_text1_to_text2.csv")
        print("│   │   ├── layer_00_head_01_text2_to_text1.csv")
        print("│   │   └── ... (每个头2个CSV文件)")
        print("│   ├── json_files/")
        print("│   │   ├── layer_00_head_00_stats.json")
        print("│   │   ├── layer_00_head_01_stats.json")
        print("│   │   └── ... (每个头1个统计文件)")
        print("│   ├── summary/")
        print("│   │   ├── overall_summary.json")
        print("│   │   └── readable_summary.txt")
        print("│   └── input_info.json")
        print()
        
        # 显示关键统计信息
        overall_summary = results['overall_summary']
        global_stats = overall_summary['global_statistics']
        
        print("关键发现:")
        print("----------")
        
        # 文本1->文本2的最高权重
        t1_to_t2 = global_stats['text1_to_text2']
        print(f"文本1→文本2 全局最高权重: {t1_to_t2['global_max_weight']:.6f}")
        print(f"  来源: 第{t1_to_t2['global_max_head']['layer']+1}层 第{t1_to_t2['global_max_head']['head']+1}头")
        print(f"  词对: '{t1_to_t2['global_max_head']['word_pair'][0]}' → '{t1_to_t2['global_max_head']['word_pair'][1]}'")
        print()
        
        # 文本2->文本1的最高权重
        t2_to_t1 = global_stats['text2_to_text1']
        print(f"文本2→文本1 全局最高权重: {t2_to_t1['global_max_weight']:.6f}")
        print(f"  来源: 第{t2_to_t1['global_max_head']['layer']+1}层 第{t2_to_t1['global_max_head']['head']+1}头")
        print(f"  词对: '{t2_to_t1['global_max_head']['word_pair'][0]}' → '{t2_to_t1['global_max_head']['word_pair'][1]}'")
        print()
        
        print(f"总共分析了 {overall_summary['analysis_summary']['total_heads_analyzed']} 个注意力头")
        print(f"查看详细汇总报告: {results['output_folder']}/summary/readable_summary.txt")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查模型路径是否正确，以及是否有足够的内存。")

if __name__ == "__main__":
    main()
