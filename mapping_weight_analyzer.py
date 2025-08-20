import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import glob
from transformers import BertTokenizer
from subword_weight_handler import SubwordWeightHandler

class MappingWeightAnalyzer:
    def __init__(self, result_folder_path: str, 
                 model_path: str = 'C:\\AppAndData\\codeAndproject\\bertBaseUncased',
                 use_subword_aggregation: bool = True,
                 aggregation_method: str = "sum"):
        """
        初始化映射权重分析器
        
        Args:
            result_folder_path: 注意力分析结果文件夹路径
            model_path: BERT模型路径（用于tokenizer）
            use_subword_aggregation: 是否使用子词聚合
            aggregation_method: 子词聚合方法 ("mean", "max", "sum", "first", "last")
        """
        self.result_folder = result_folder_path
        self.csv_folder = os.path.join(result_folder_path, "csv_files")
        self.use_subword_aggregation = use_subword_aggregation
        self.aggregation_method = aggregation_method
        
        # 验证文件夹存在
        if not os.path.exists(self.csv_folder):
            raise FileNotFoundError(f"CSV文件夹不存在: {self.csv_folder}")
        
        # 初始化子词处理器（如果需要）
        if use_subword_aggregation:
            try:
                tokenizer = BertTokenizer.from_pretrained(model_path)
                self.subword_handler = SubwordWeightHandler(tokenizer)
                print(f"子词聚合已启用，方法: {aggregation_method}")
            except Exception as e:
                print(f"警告: 无法加载tokenizer，禁用子词聚合: {e}")
                self.use_subword_aggregation = False
                self.subword_handler = None
        else:
            self.subword_handler = None
        
        print(f"初始化映射权重分析器")
        print(f"结果文件夹: {result_folder_path}")
        print(f"CSV文件夹: {self.csv_folder}")
        print(f"子词聚合: {'启用' if self.use_subword_aggregation else '禁用'}")
    
    def define_mappings(self, positive_mappings: Dict[str, str], 
                       negative_mappings: Dict[str, str]) -> None:
        """
        定义正例和反例映射
        
        Args:
            positive_mappings: 正例映射字典 {text2_word: text1_word}
            negative_mappings: 反例映射字典 {text2_word: text1_word}
        """
        self.positive_mappings = positive_mappings
        self.negative_mappings = negative_mappings
        
        print(f"\n定义映射关系:")
        print(f"正例映射 ({len(positive_mappings)} 对):")
        for text2_word, text1_word in positive_mappings.items():
            print(f"  '{text2_word}' → '{text1_word}'")
        
        print(f"反例映射 ({len(negative_mappings)} 对):")
        for text2_word, text1_word in negative_mappings.items():
            print(f"  '{text2_word}' → '{text1_word}'")
    
    def find_csv_files(self) -> List[str]:
        """查找所有text2_to_text1的CSV文件"""
        pattern = os.path.join(self.csv_folder, "*_text2_to_text1.csv")
        csv_files = glob.glob(pattern)
        csv_files.sort()  # 按文件名排序
        
        print(f"\n找到 {len(csv_files)} 个text2_to_text1文件")
        return csv_files
    
    def parse_filename(self, filename: str) -> Tuple[int, int]:
        """
        从文件名解析层号和头号
        
        Args:
            filename: 文件名，如 "layer_00_head_01_text2_to_text1.csv"
            
        Returns:
            (layer_idx, head_idx)
        """
        basename = os.path.basename(filename)
        parts = basename.split('_')
        
        try:
            layer_idx = int(parts[1])  # layer_XX
            head_idx = int(parts[3])   # head_XX
            return layer_idx, head_idx
        except (IndexError, ValueError) as e:
            raise ValueError(f"无法解析文件名: {filename}, 错误: {e}")

        # 同时添加一个新的方法来生成CSV格式的权重表
    def export_weights_to_csv(self, results: Dict[str, Any], output_filename: str = "weights_matrix"):
        """
        导出权重到CSV文件，便于Excel等工具分析
        
        Args:
            results: 分析结果
            output_filename: 输出文件名前缀
        """
        import pandas as pd
        
        # 收集所有头的数据
        rows_positive = []
        rows_negative = []
        rows_ratio = []
        
        for head_key, head_data in results["layer_head_details"].items():
            layer_idx = head_data["layer"]
            head_idx = head_data["head"]
            
            # 基本信息
            base_info = {
                'Layer': layer_idx + 1,
                'Head': head_idx + 1,
                'Layer_Head': f"L{layer_idx+1}H{head_idx+1}"
            }
            
            # 正例权重
            pos_row = base_info.copy()
            pos_row.update(head_data["positive_weights"])
            pos_row['Mean'] = head_data["positive_stats"]["mean"]
            pos_row['Max'] = head_data["positive_stats"]["max"]
            pos_row['Min'] = head_data["positive_stats"]["min"]
            pos_row['Std'] = head_data["positive_stats"]["std"]
            rows_positive.append(pos_row)
            
            # 反例权重
            neg_row = base_info.copy()
            neg_row.update(head_data["negative_weights"])
            neg_row['Mean'] = head_data["negative_stats"]["mean"]
            neg_row['Max'] = head_data["negative_stats"]["max"]
            neg_row['Min'] = head_data["negative_stats"]["min"]
            neg_row['Std'] = head_data["negative_stats"]["std"]
            rows_negative.append(neg_row)
            
            # 判别比
            ratio_row = base_info.copy()
            pos_mean = head_data["positive_stats"]["mean"]
            neg_mean = head_data["negative_stats"]["mean"]
            ratio_row['Discrimination_Ratio'] = pos_mean / neg_mean if neg_mean > 0 else float('inf')
            ratio_row['Positive_Mean'] = pos_mean
            ratio_row['Negative_Mean'] = neg_mean
            rows_ratio.append(ratio_row)
        
        # 创建DataFrame并保存
        df_positive = pd.DataFrame(rows_positive)
        df_negative = pd.DataFrame(rows_negative)
        df_ratio = pd.DataFrame(rows_ratio)
        
        # 保存为CSV文件
        pos_file = os.path.join(self.result_folder, f"{output_filename}_positive.csv")
        neg_file = os.path.join(self.result_folder, f"{output_filename}_negative.csv")
        ratio_file = os.path.join(self.result_folder, f"{output_filename}_ratios.csv")
        
        df_positive.to_csv(pos_file, index=False, encoding='utf-8-sig')
        df_negative.to_csv(neg_file, index=False, encoding='utf-8-sig')
        df_ratio.to_csv(ratio_file, index=False, encoding='utf-8-sig')
        
        print(f"权重矩阵已导出为CSV:")
        print(f"  正例权重: {pos_file}")
        print(f"  反例权重: {neg_file}")
        print(f"  判别比率: {ratio_file}")
        
        return pos_file, neg_file, ratio_file
 
    def extract_weights_from_csv(self, csv_file: str, 
                                mappings: Dict[str, str]) -> Dict[str, float]:
        """
        从CSV文件中提取指定映射的权重（支持子词聚合）
        
        Args:
            csv_file: CSV文件路径
            mappings: 映射字典 {text2_word: text1_word}
            
        Returns:
            提取的权重字典 {mapping_key: weight}
        """
        try:
            # 如果启用了子词聚合，使用专门的处理器
            if self.use_subword_aggregation and self.subword_handler:
                return self.subword_handler.extract_word_level_weights(
                    csv_file, mappings, self.aggregation_method)
            
            # 否则使用原始方法
            return self._extract_weights_basic(csv_file, mappings)
            
        except Exception as e:
            print(f"    错误: 提取权重失败 {csv_file}: {e}")
            # 返回默认权重
            return {f"{text2_word}->{text1_word}": 0.0 
                   for text2_word, text1_word in mappings.items()}
    
    def _extract_weights_basic(self, csv_file: str, 
                              mappings: Dict[str, str]) -> Dict[str, float]:
        """
        基础权重提取方法（不进行子词聚合）
        
        Args:
            csv_file: CSV文件路径
            mappings: 映射字典 {text2_word: text1_word}
            
        Returns:
            提取的权重字典 {mapping_key: weight}
        """
        # 读取CSV文件
        df = pd.read_csv(csv_file, index_col=0)
        
        weights = {}
        missing_mappings = []
        
        for text2_word, text1_word in mappings.items():
            try:
                # 提取权重 (text2_word -> text1_word)
                if text2_word in df.index and text1_word in df.columns:
                    weight = df.loc[text2_word, text1_word]
                    
                    # 处理可能返回Series的情况（当有重复索引时）
                    if isinstance(weight, pd.Series):
                        # 如果返回Series，取第一个值
                        weight = weight.iloc[0]
                        print(f"    注意: {text2_word}->{text1_word} 存在重复索引，使用第一个匹配值")
                    
                    # 处理可能的NaN值
                    if pd.isna(weight):
                        weight = 0.0
                    else:
                        weight = float(weight)
                    weights[f"{text2_word}->{text1_word}"] = weight
                else:
                    missing_mappings.append(f"{text2_word}->{text1_word}")
                    weights[f"{text2_word}->{text1_word}"] = 0.0
                    
            except Exception as e:
                print(f"    警告: 提取权重失败 {text2_word}->{text1_word}: {e}")
                weights[f"{text2_word}->{text1_word}"] = 0.0
        
        if missing_mappings:
            print(f"    缺失的映射: {missing_mappings}")
        
        return weights
    
    def analyze_all_files(self) -> Dict[str, Any]:
        """
        分析所有CSV文件
        
        Returns:
            完整的分析结果
        """
        csv_files = self.find_csv_files()
        
        if not csv_files:
            raise FileNotFoundError("未找到任何text2_to_text1 CSV文件")
        
        # 初始化结果结构
        results = {
            "metadata": {
                "analysis_time": datetime.now().isoformat(),
                "total_files": len(csv_files),
                "positive_mappings": self.positive_mappings,
                "negative_mappings": self.negative_mappings
            },
            "layer_head_details": {},
            "layer_summaries": {},
            "global_summary": {}
        }
        
        print(f"\n开始分析 {len(csv_files)} 个文件...")
        
        # 用于全局统计
        all_positive_weights = []
        all_negative_weights = []
        layer_data = {}  # {layer_idx: [head_data...]}
        
        for i, csv_file in enumerate(csv_files, 1):
            try:
                layer_idx, head_idx = self.parse_filename(csv_file)
                print(f"  [{i:3d}/{len(csv_files)}] 处理 第{layer_idx+1}层第{head_idx+1}头...", end=" ")
                
                # 提取正例和反例权重
                positive_weights = self.extract_weights_from_csv(csv_file, self.positive_mappings)
                negative_weights = self.extract_weights_from_csv(csv_file, self.negative_mappings)
                
                # 计算统计信息
                pos_values = list(positive_weights.values())
                neg_values = list(negative_weights.values())
                
                head_result = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "positive_weights": positive_weights,
                    "negative_weights": negative_weights,
                    "positive_stats": {
                        "values": pos_values,
                        "mean": np.mean(pos_values),
                        "max": np.max(pos_values),
                        "min": np.min(pos_values),
                        "std": np.std(pos_values)
                    },
                    "negative_stats": {
                        "values": neg_values,
                        "mean": np.mean(neg_values),
                        "max": np.max(neg_values),
                        "min": np.min(neg_values),
                        "std": np.std(neg_values)
                    }
                }
                
                # 存储结果
                layer_head_key = f"layer_{layer_idx:02d}_head_{head_idx:02d}"
                results["layer_head_details"][layer_head_key] = head_result
                
                # 按层分组
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = []
                layer_data[layer_idx].append(head_result)
                
                # 收集全局数据
                all_positive_weights.extend(pos_values)
                all_negative_weights.extend(neg_values)
                
                print("完成")
                
            except Exception as e:
                print(f"错误: {e}")
                continue
        
        # 计算层级汇总
        print(f"\n计算层级汇总...")
        for layer_idx in sorted(layer_data.keys()):
            layer_heads = layer_data[layer_idx]
            
            # 收集该层所有头的权重
            layer_pos_values = []
            layer_neg_values = []
            layer_pos_means = []
            layer_neg_means = []
            layer_pos_maxs = []
            layer_neg_maxs = []
            
            for head_data in layer_heads:
                layer_pos_values.extend(head_data["positive_stats"]["values"])
                layer_neg_values.extend(head_data["negative_stats"]["values"])
                layer_pos_means.append(head_data["positive_stats"]["mean"])
                layer_neg_means.append(head_data["negative_stats"]["mean"])
                layer_pos_maxs.append(head_data["positive_stats"]["max"])
                layer_neg_maxs.append(head_data["negative_stats"]["max"])
            
            layer_summary = {
                "layer": layer_idx,
                "num_heads": len(layer_heads),
                "positive_layer_stats": {
                    "mean_of_all_values": np.mean(layer_pos_values),
                    "max_of_all_values": np.max(layer_pos_values),
                    "mean_of_head_means": np.mean(layer_pos_means),
                    "max_of_head_maxs": np.max(layer_pos_maxs),
                    "std_of_head_means": np.std(layer_pos_means)
                },
                "negative_layer_stats": {
                    "mean_of_all_values": np.mean(layer_neg_values),
                    "max_of_all_values": np.max(layer_neg_values),
                    "mean_of_head_means": np.mean(layer_neg_means),
                    "max_of_head_maxs": np.max(layer_neg_maxs),
                    "std_of_head_means": np.std(layer_neg_means)
                }
            }
            
            results["layer_summaries"][f"layer_{layer_idx:02d}"] = layer_summary
            print(f"  第{layer_idx+1}层: 正例均值={layer_summary['positive_layer_stats']['mean_of_all_values']:.6f}, "
                  f"反例均值={layer_summary['negative_layer_stats']['mean_of_all_values']:.6f}")
        
        # 计算全局汇总
        print(f"\n计算全局汇总...")
        global_summary = {
            "total_layers": len(layer_data),
            "total_mappings": len(self.positive_mappings) + len(self.negative_mappings),
            "positive_global_stats": {
                "total_values": len(all_positive_weights),
                "mean": np.mean(all_positive_weights),
                "max": np.max(all_positive_weights),
                "min": np.min(all_positive_weights),
                "std": np.std(all_positive_weights)
            },
            "negative_global_stats": {
                "total_values": len(all_negative_weights),
                "mean": np.mean(all_negative_weights),
                "max": np.max(all_negative_weights),
                "min": np.min(all_negative_weights),
                "std": np.std(all_negative_weights)
            },
            "discrimination_metrics": {
                "positive_negative_ratio": np.mean(all_positive_weights) / np.mean(all_negative_weights) if np.mean(all_negative_weights) > 0 else float('inf'),
                "effect_size": (np.mean(all_positive_weights) - np.mean(all_negative_weights)) / np.sqrt((np.std(all_positive_weights)**2 + np.std(all_negative_weights)**2) / 2)
            }
        }
        
        results["global_summary"] = global_summary
        
        print(f"全局正例权重: 均值={global_summary['positive_global_stats']['mean']:.6f}, "
              f"最大值={global_summary['positive_global_stats']['max']:.6f}")
        print(f"全局反例权重: 均值={global_summary['negative_global_stats']['mean']:.6f}, "
              f"最大值={global_summary['negative_global_stats']['max']:.6f}")
        print(f"正反例比率: {global_summary['discrimination_metrics']['positive_negative_ratio']:.3f}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_filename: str = "final_result") -> str:
        """
        保存分析结果到文件（增强版）
        
        Args:
            results: 分析结果
            output_filename: 输出文件名前缀
            
        Returns:
            保存的文件路径
        """
        # 保存详细JSON结果
        json_file = os.path.join(self.result_folder, f"{output_filename}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成可读的汇总报告
        txt_file = os.path.join(self.result_folder, f"{output_filename}.txt")
        self.generate_readable_report(results, txt_file)
        
        # 导出权重矩阵为CSV文件
        self.export_weights_to_csv(results, f"{output_filename}_weights")
        
        print(f"\n结果已保存:")
        print(f"  详细结果: {json_file}")
        print(f"  可读报告: {txt_file}")
        
        return json_file
    
    def generate_readable_report(self, results: Dict[str, Any], txt_file: str):
        """生成可读的分析报告（增强版，包含每个头的详细权重）"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("映射权重分析报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 元数据
        metadata = results["metadata"]
        report_lines.append("分析信息:")
        report_lines.append(f"  分析时间: {metadata['analysis_time']}")
        report_lines.append(f"  处理文件数: {metadata['total_files']}")
        report_lines.append("")
        
        # 映射定义
        report_lines.append("映射定义:")
        report_lines.append("  正例映射:")
        for text2, text1 in metadata["positive_mappings"].items():
            report_lines.append(f"    {text2} → {text1}")
        report_lines.append("  反例映射:")
        for text2, text1 in metadata["negative_mappings"].items():
            report_lines.append(f"    {text2} → {text1}")
        report_lines.append("")
        
        # 全局统计
        global_summary = results["global_summary"]
        report_lines.append("全局统计结果:")
        pos_stats = global_summary["positive_global_stats"]
        neg_stats = global_summary["negative_global_stats"]
        
        report_lines.append(f"  正例权重 (总计 {pos_stats['total_values']} 个值):")
        report_lines.append(f"    平均值: {pos_stats['mean']:.6f}")
        report_lines.append(f"    最大值: {pos_stats['max']:.6f}")
        report_lines.append(f"    最小值: {pos_stats['min']:.6f}")
        report_lines.append(f"    标准差: {pos_stats['std']:.6f}")
        report_lines.append("")
        
        report_lines.append(f"  反例权重 (总计 {neg_stats['total_values']} 个值):")
        report_lines.append(f"    平均值: {neg_stats['mean']:.6f}")
        report_lines.append(f"    最大值: {neg_stats['max']:.6f}")
        report_lines.append(f"    最小值: {neg_stats['min']:.6f}")
        report_lines.append(f"    标准差: {neg_stats['std']:.6f}")
        report_lines.append("")
        
        # 判别性指标
        disc_metrics = global_summary["discrimination_metrics"]
        report_lines.append("判别性分析:")
        report_lines.append(f"  正反例权重比率: {disc_metrics['positive_negative_ratio']:.3f}")
        report_lines.append(f"  效应大小: {disc_metrics['effect_size']:.3f}")
        report_lines.append("")
        
        # 层级汇总
        report_lines.append("各层汇总:")
        report_lines.append("-" * 60)
        for layer_key in sorted(results["layer_summaries"].keys()):
            layer_summary = results["layer_summaries"][layer_key]
            layer_idx = layer_summary["layer"]
            
            report_lines.append(f"第 {layer_idx+1} 层 (共 {layer_summary['num_heads']} 个头):")
            
            pos_layer = layer_summary["positive_layer_stats"]
            neg_layer = layer_summary["negative_layer_stats"]
            
            report_lines.append(f"  正例: 均值={pos_layer['mean_of_all_values']:.6f}, 最大={pos_layer['max_of_all_values']:.6f}")
            report_lines.append(f"  反例: 均值={neg_layer['mean_of_all_values']:.6f}, 最大={neg_layer['max_of_all_values']:.6f}")
            report_lines.append(f"  层内判别比: {pos_layer['mean_of_all_values']/neg_layer['mean_of_all_values']:.3f}")
            report_lines.append("")
        
        # === 新增：详细的每层每头权重列表 ===
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("详细权重列表 - 每层每头")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 按层组织数据
        layer_head_data = {}
        for head_key, head_data in results["layer_head_details"].items():
            layer_idx = head_data["layer"]
            if layer_idx not in layer_head_data:
                layer_head_data[layer_idx] = []
            layer_head_data[layer_idx].append((head_data["head"], head_data))
        
        # 遍历每一层
        for layer_idx in sorted(layer_head_data.keys()):
            heads_data = sorted(layer_head_data[layer_idx], key=lambda x: x[0])  # 按头编号排序
            
            report_lines.append(f"第 {layer_idx+1} 层详细权重:")
            report_lines.append("─" * 70)
            
            for head_idx, head_data in heads_data:
                report_lines.append(f"\n  第 {head_idx+1} 头:")
                
                # 正例权重
                report_lines.append("    正例权重:")
                pos_weights = head_data["positive_weights"]
                pos_stats = head_data["positive_stats"]
                
                for mapping, weight in pos_weights.items():
                    report_lines.append(f"      {mapping}: {weight:.6f}")
                
                report_lines.append(f"    正例统计: 均值={pos_stats['mean']:.6f}, 最大={pos_stats['max']:.6f}, 最小={pos_stats['min']:.6f}")
                
                # 反例权重
                report_lines.append("    反例权重:")
                neg_weights = head_data["negative_weights"]
                neg_stats = head_data["negative_stats"]
                
                for mapping, weight in neg_weights.items():
                    report_lines.append(f"      {mapping}: {weight:.6f}")
                
                report_lines.append(f"    反例统计: 均值={neg_stats['mean']:.6f}, 最大={neg_stats['max']:.6f}, 最小={neg_stats['min']:.6f}")
                
                # 该头的判别性能
                head_ratio = pos_stats['mean'] / neg_stats['mean'] if neg_stats['mean'] > 0 else float('inf')
                report_lines.append(f"    头内判别比: {head_ratio:.3f}")
            
            report_lines.append("")
        
        # === 新增：权重矩阵格式的汇总表 ===
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("权重矩阵汇总表")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 创建正例权重矩阵表
        report_lines.append("正例权重矩阵 (行=层, 列=头):")
        report_lines.append("-" * 50)
        
        # 表头
        max_heads = max(len(heads_data) for heads_data in layer_head_data.values())
        header = "层\\头  " + "".join(f"{i+1:>8}" for i in range(max_heads))
        report_lines.append(header)
        
        # 正例权重表
        for layer_idx in sorted(layer_head_data.keys()):
            heads_data = sorted(layer_head_data[layer_idx], key=lambda x: x[0])
            
            row = f"{layer_idx+1:>3}   "
            for head_idx, head_data in heads_data:
                pos_mean = head_data["positive_stats"]["mean"]
                row += f"{pos_mean:>8.4f}"
            
            report_lines.append(row)
        
        report_lines.append("")
        
        # 反例权重矩阵表
        report_lines.append("反例权重矩阵 (行=层, 列=头):")
        report_lines.append("-" * 50)
        report_lines.append(header)  # 复用表头
        
        for layer_idx in sorted(layer_head_data.keys()):
            heads_data = sorted(layer_head_data[layer_idx], key=lambda x: x[0])
            
            row = f"{layer_idx+1:>3}   "
            for head_idx, head_data in heads_data:
                neg_mean = head_data["negative_stats"]["mean"]
                row += f"{neg_mean:>8.4f}"
            
            report_lines.append(row)
        
        report_lines.append("")
        
        # 判别比矩阵表
        report_lines.append("判别比矩阵 (正例/反例, 行=层, 列=头):")
        report_lines.append("-" * 50)
        report_lines.append(header)  # 复用表头
        
        for layer_idx in sorted(layer_head_data.keys()):
            heads_data = sorted(layer_head_data[layer_idx], key=lambda x: x[0])
            
            row = f"{layer_idx+1:>3}   "
            for head_idx, head_data in heads_data:
                pos_mean = head_data["positive_stats"]["mean"]
                neg_mean = head_data["negative_stats"]["mean"]
                ratio = pos_mean / neg_mean if neg_mean > 0 else 999.9
                # 限制显示范围，避免过大的数字
                ratio = min(ratio, 999.9)
                row += f"{ratio:>8.2f}"
            
            report_lines.append(row)
        
        # === 新增：最佳和最差头分析 ===
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("最佳/最差注意力头分析")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 找出判别性能最好和最差的头
        all_heads_performance = []
        for head_key, head_data in results["layer_head_details"].items():
            pos_mean = head_data["positive_stats"]["mean"]
            neg_mean = head_data["negative_stats"]["mean"]
            ratio = pos_mean / neg_mean if neg_mean > 0 else float('inf')
            
            all_heads_performance.append({
                'layer': head_data["layer"],
                'head': head_data["head"],
                'pos_mean': pos_mean,
                'neg_mean': neg_mean,
                'ratio': ratio,
                'key': head_key
            })
        
        # 按判别比排序
        all_heads_performance.sort(key=lambda x: x['ratio'], reverse=True)
        
        # 最佳头（前5个）
        report_lines.append("判别性能最佳的注意力头 (正例/反例比率最高):")
        report_lines.append("-" * 60)
        for i, head_perf in enumerate(all_heads_performance[:5], 1):
            layer_idx = head_perf['layer']
            head_idx = head_perf['head']
            ratio = head_perf['ratio']
            pos_mean = head_perf['pos_mean']
            neg_mean = head_perf['neg_mean']
            
            report_lines.append(f"{i}. 第{layer_idx+1}层第{head_idx+1}头:")
            report_lines.append(f"   正例均值: {pos_mean:.6f}")
            report_lines.append(f"   反例均值: {neg_mean:.6f}")
            report_lines.append(f"   判别比率: {ratio:.3f}")
            
            # 显示该头的具体权重
            head_data = results["layer_head_details"][head_perf['key']]
            report_lines.append("   正例权重详情:")
            for mapping, weight in head_data["positive_weights"].items():
                report_lines.append(f"     {mapping}: {weight:.6f}")
            report_lines.append("")
        
        # 最差头（后5个）
        report_lines.append("判别性能最差的注意力头 (正例/反例比率最低):")
        report_lines.append("-" * 60)
        for i, head_perf in enumerate(all_heads_performance[-5:], 1):
            layer_idx = head_perf['layer']
            head_idx = head_perf['head']
            ratio = head_perf['ratio']
            pos_mean = head_perf['pos_mean']
            neg_mean = head_perf['neg_mean']
            
            report_lines.append(f"{i}. 第{layer_idx+1}层第{head_idx+1}头:")
            report_lines.append(f"   正例均值: {pos_mean:.6f}")
            report_lines.append(f"   反例均值: {neg_mean:.6f}")
            report_lines.append(f"   判别比率: {ratio:.3f}")
            report_lines.append("")
        
        # 结尾
        report_lines.append("=" * 80)
        report_lines.append("报告结束")
        report_lines.append("=" * 80)
        
        # 保存报告
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


def main():
    """主程序示例"""
    
    # 1. 设置结果文件夹路径
    result_folder = "attention_heads_analysis_20250819_145658"  # 替换为你的实际路径
    
    # 2. 初始化分析器
    try:
        analyzer = MappingWeightAnalyzer(result_folder)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 3. 定义映射关系
    positive_mappings = {
    "items": "item",
    "weights": "weight",
    "dimensions": "dimension"
    }
    negative_mappings = {
    "space": "item",
    "vehicle": "dimension",
    "route": "description"
    }
    
    analyzer.define_mappings(positive_mappings, negative_mappings)
    
    # 4. 执行分析
    try:
        results = analyzer.analyze_all_files()
        
        # 5. 保存结果
        output_file = analyzer.save_results(results, "final_result")
        
        print(f"\n=== 分析完成 ===")
        print(f"处理了 {results['metadata']['total_files']} 个文件")
        print(f"分析了 {results['global_summary']['total_layers']} 层")
        print(f"正例权重平均值: {results['global_summary']['positive_global_stats']['mean']:.6f}")
        print(f"反例权重平均值: {results['global_summary']['negative_global_stats']['mean']:.6f}")
        print(f"判别比率: {results['global_summary']['discrimination_metrics']['positive_negative_ratio']:.3f}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
