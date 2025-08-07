import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from code_bert import PlantUMLBertAttention

@dataclass
class ClassInfo:
    name: str
    attributes: List[str]
    methods: List[str]
    stereotypes: List[str]

@dataclass
class RelationshipInfo:
    source: str
    target: str
    relationship_type: str
    multiplicity: str
    label: str

class PlantUMLClassParser:
    def __init__(self):
        self.classes: Dict[str, ClassInfo] = {}
        self.relationships: List[RelationshipInfo] = []
        
    def parse_plantuml(self, puml_code: str) -> Dict:
        """
        解析PlantUML类图代码，提取结构信息
        """
        # 清理代码
        cleaned_code = self._clean_plantuml_code(puml_code)
        
        # 解析类
        self._parse_classes(cleaned_code)
        
        # 解析关系
        self._parse_relationships(cleaned_code)
        
        return {
            'classes': self.classes,
            'relationships': self.relationships,
            'summary': self._generate_summary()
        }
    
    def _clean_plantuml_code(self, code: str) -> str:
        """
        清理PlantUML代码，移除注释和格式化
        """
        # 移除注释
        code = re.sub(r"'.*$", "", code, flags=re.MULTILINE)
        # 移除@startuml和@enduml
        code = re.sub(r"@startuml|@enduml", "", code)
        # 移除多余空行
        code = re.sub(r"\n\s*\n", "\n", code)
        return code.strip()
    
    def _parse_classes(self, code: str):
        """
        解析类定义
        """
        # 修复后的正则表达式，支持implements和extends语法
        class_pattern = r'(class|interface|abstract\s+class|enum)\s+(\w+)(?:\s+(?:implements|extends)\s+\w+(?:\s*,\s*\w+)*)?(?:\s*<<(\w+)>>)?\s*\{'
        method_pattern = r'^\s*([+\-#~]?)\s*(\w+)\s*\([^)]*\)\s*:\s*(\w+)'
        attribute_pattern = r'^\s*([+\-#~]?)\s*(\w+)\s*:\s*(\w+)'
        
        lines = code.split('\n')
        current_class = None
        in_class_definition = False
        
        for line in lines:
            line = line.strip()
            
            # 检查类定义开始
            class_match = re.match(class_pattern, line)
            if class_match:
                class_type = class_match.group(1)
                class_name = class_match.group(2)
                stereotype = class_match.group(3) if class_match.group(3) else ""
                
                current_class = ClassInfo(
                    name=class_name,
                    attributes=[],
                    methods=[],
                    stereotypes=[stereotype] if stereotype else []
                )
                in_class_definition = True
                continue
            
            # 检查类定义结束
            if line == '}' and in_class_definition:
                if current_class:
                    self.classes[current_class.name] = current_class
                in_class_definition = False
                current_class = None
                continue
            
            # 解析类成员
            if in_class_definition and current_class:
                # 尝试匹配方法
                method_match = re.match(method_pattern, line)
                if method_match:
                    visibility = method_match.group(1) or ''
                    method_name = method_match.group(2)
                    return_type = method_match.group(3)
                    current_class.methods.append(f"{visibility}{method_name}(): {return_type}")
                    continue
                
                # 尝试匹配属性
                attr_match = re.match(attribute_pattern, line)
                if attr_match:
                    visibility = attr_match.group(1) or ''
                    attr_name = attr_match.group(2)
                    attr_type = attr_match.group(3)
                    current_class.attributes.append(f"{visibility}{attr_name}: {attr_type}")
                    continue
    
    def _parse_relationships(self, code: str):
        """
        解析类之间的关系
        """
        # 各种关系的正则表达式 - 修复后的版本
        relationship_patterns = [
            # 继承关系
            (r'(\w+)\s*<\|--\s*(\w+)', 'inheritance'),
            (r'(\w+)\s*<\|..\s*(\w+)', 'realization'),
            # 组合关系
            (r'(\w+)\s*"([^"]*?)"\s*\|\|--\|\|\s*"([^"]*?)"\s*(\w+)', 'composition'),
            (r'(\w+)\s*"([^"]*?)"\s*\*--\s*"([^"]*?)"\s*(\w+)', 'composition'),
            # 聚合关系
            (r'(\w+)\s*"([^"]*?)"\s*o--\s*"([^"]*?)"\s*(\w+)', 'aggregation'),
            # 关联关系
            (r'(\w+)\s*"([^"]*?)"\s*-->\s*"([^"]*?)"\s*(\w+)', 'association'),
            (r'(\w+)\s*"([^"]*?)"\s*--\s*"([^"]*?)"\s*(\w+)', 'association'),
            # 简单关系（没有引号）
            (r'(\w+)\s*\|\|--\|\|\s*(\w+)', 'composition'),
            (r'(\w+)\s*\*--\s*(\w+)', 'composition'),
            (r'(\w+)\s*o--\s*(\w+)', 'aggregation'),
            (r'(\w+)\s*-->\s*(\w+)', 'association'),
            (r'(\w+)\s*--\s*(\w+)', 'association'),
        ]
        
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern, rel_type in relationship_patterns:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    
                    if rel_type == 'inheritance' or rel_type == 'realization':
                        # 继承和实现关系：source <|-- target
                        source = groups[1]  # 子类
                        target = groups[0]  # 父类
                        multiplicity = ""
                        label = ""
                    else:
                        # 其他关系
                        if len(groups) == 4:
                            # 有引号的关系：source "mult1" rel "mult2" target
                            source = groups[0]
                            target = groups[3]
                            multiplicity = f"{groups[1]}-{groups[2]}"
                            label = ""
                        elif len(groups) == 2:
                            # 简单关系：source rel target
                            source = groups[0]
                            target = groups[1]
                            multiplicity = ""
                            label = ""
                        else:
                            # 其他情况，使用默认处理
                            source = groups[0]
                            target = groups[-1]
                            multiplicity = ""
                            label = ""
                    
                    relationship = RelationshipInfo(
                        source=source,
                        target=target,
                        relationship_type=rel_type,
                        multiplicity=multiplicity,
                        label=label
                    )
                    self.relationships.append(relationship)
                    break
    
    def _generate_summary(self) -> Dict:
        """
        生成类图摘要信息
        """
        return {
            'total_classes': len(self.classes),
            'total_relationships': len(self.relationships),
            'class_names': list(self.classes.keys()),
            'relationship_types': list(set(rel.relationship_type for rel in self.relationships)),
            'average_methods_per_class': sum(len(cls.methods) for cls in self.classes.values()) / len(self.classes) if self.classes else 0,
            'average_attributes_per_class': sum(len(cls.attributes) for cls in self.classes.values()) / len(self.classes) if self.classes else 0
        }
    
    def extract_textual_representation(self) -> str:
        """
        将解析后的类图转换为文本表示，用于BERT分析
        """
        text_parts = []
        
        # 添加类信息
        for class_name, class_info in self.classes.items():
            text_parts.append(f"class {class_name}")
            
            if class_info.attributes:
                text_parts.append(f"attributes: {', '.join(class_info.attributes)}")
            
            if class_info.methods:
                text_parts.append(f"methods: {', '.join(class_info.methods)}")
        
        # 添加关系信息
        for rel in self.relationships:
            text_parts.append(f"{rel.source} {rel.relationship_type} {rel.target}")
        
        return ". ".join(text_parts)

class PlantUMLSystemMatcher:
    def __init__(self):
        self.parser = PlantUMLClassParser()
        self.bert_analyzer = PlantUMLBertAttention()
        
    def analyze_plantuml_system_match(self, puml_code: str, system_description: str) -> Dict:
        """
        分析PlantUML类图与系统描述的匹配程度
        """
        # 解析PlantUML类图
        parsed_structure = self.parser.parse_plantuml(puml_code)
        
        # 转换为文本表示
        class_diagram_text = self.parser.extract_textual_representation()
        
        # 使用BERT分析匹配度
        bert_result = self.bert_analyzer.calculate_matching_score_corrected(
            class_diagram_text, system_description
        )
        
        # 进行结构化分析
        structural_analysis = self._analyze_structural_match(parsed_structure, system_description)
        
        # 生成详细报告
        detailed_analysis = self._generate_detailed_analysis(
            parsed_structure, system_description, bert_result, structural_analysis
        )
        
        return {
            'bert_scores': bert_result,
            'structural_analysis': structural_analysis,
            'parsed_structure': parsed_structure,
            'class_diagram_text': class_diagram_text,
            'detailed_analysis': detailed_analysis
        }
    
    def _analyze_structural_match(self, parsed_structure: Dict, system_description: str) -> Dict:
        """
        分析结构化匹配度
        """
        classes = parsed_structure['classes']
        relationships = parsed_structure['relationships']
        
        # 分析类名在描述中的出现频率
        class_name_matches = {}
        for class_name in classes.keys():
            # 改进的匹配逻辑
            match_result = self._improved_class_name_matching(class_name, system_description)
            class_name_matches[class_name] = match_result
        
        # 分析属性和方法的匹配
        attribute_method_matches = {}
        for class_name, class_info in classes.items():
            matches = {
                'attributes': [],
                'methods': []
            }
            
            # 检查属性
            for attr in class_info.attributes:
                attr_name = attr.split(':')[0].strip('+-#~ ')
                if self._flexible_word_match(attr_name, system_description):
                    matches['attributes'].append(attr_name)
            
            # 检查方法
            for method in class_info.methods:
                method_name = method.split('(')[0].strip('+-#~ ')
                if self._flexible_word_match(method_name, system_description):
                    matches['methods'].append(method_name)
            
            attribute_method_matches[class_name] = matches
        
        # 计算覆盖率
        total_classes = len(classes)
        matched_classes = sum(1 for match in class_name_matches.values() if match['direct_match'])
        class_coverage = matched_classes / total_classes if total_classes > 0 else 0
        
        return {
            'class_name_matches': class_name_matches,
            'attribute_method_matches': attribute_method_matches,
            'class_coverage': class_coverage,
            'total_classes': total_classes,
            'matched_classes': matched_classes
        }
    
    def _improved_class_name_matching(self, class_name: str, description: str) -> Dict:
        """
        改进的类名匹配逻辑，支持驼峰命名法分割匹配
        """
        desc_lower = description.lower()
        class_name_lower = class_name.lower()
        
        # 1. 直接匹配
        direct_match = class_name_lower in desc_lower
        occurrence_count = desc_lower.count(class_name_lower)
        
        # 2. 驼峰命名法分割匹配
        camel_words = self._split_camel_case(class_name)
        camel_match = False
        camel_score = 0
        matched_words = []
        
        for word in camel_words:
            word_lower = word.lower()
            if word_lower in desc_lower:
                camel_match = True
                camel_score += 1
                matched_words.append(word)
        
        # 3. 计算匹配度
        if len(camel_words) > 0:
            camel_match_ratio = camel_score / len(camel_words)
        else:
            camel_match_ratio = 0
        
        # 4. 综合判断是否匹配
        # 如果直接匹配或驼峰匹配度超过50%，认为匹配成功
        final_match = direct_match or camel_match_ratio >= 0.5
        
        return {
            'direct_match': final_match,
            'occurrence_count': occurrence_count
        }
    
    def _split_camel_case(self, word: str) -> List[str]:
        """
        分割驼峰命名法的单词
        """
        import re
        # 使用正则表达式分割驼峰命名
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', word)
        return [w for w in words if len(w) > 1]  # 过滤掉单字符
    
    def _flexible_word_match(self, word: str, description: str) -> bool:
        """
        灵活的词语匹配，支持驼峰分割
        """
        desc_lower = description.lower()
        
        # 直接匹配
        if word.lower() in desc_lower:
            return True
        
        # 驼峰分割匹配
        words = self._split_camel_case(word)
        for w in words:
            if w.lower() in desc_lower:
                return True
        
        return False
    
    def _generate_detailed_analysis(self, parsed_structure: Dict, system_description: str, 
                                   bert_result: Dict, structural_analysis: Dict) -> Dict:
        """
        生成详细的分析报告
        """
        return {
            'overall_score': (bert_result['final_matching_score'] + structural_analysis['class_coverage']) / 2,
            'bert_score': bert_result['final_matching_score'],
            'structural_score': structural_analysis['class_coverage'],
            'class_diagram_stats': parsed_structure['summary'],
            'matching_details': {
                'matched_classes': [name for name, match in structural_analysis['class_name_matches'].items() if match['direct_match']],
                'unmatched_classes': [name for name, match in structural_analysis['class_name_matches'].items() if not match['direct_match']],
                'high_frequency_classes': [name for name, match in structural_analysis['class_name_matches'].items() if match['occurrence_count'] > 2]
            },
            'recommendations': self._generate_recommendations(structural_analysis)
        }
    
    def _generate_recommendations(self, structural_analysis: Dict) -> List[str]:
        """
        生成改进建议
        """
        recommendations = []
        
        unmatched_classes = [name for name, match in structural_analysis['class_name_matches'].items() if not match['direct_match']]
        
        if unmatched_classes:
            recommendations.append(f"以下类在系统描述中未提及，建议在描述中添加相关内容: {', '.join(unmatched_classes)}")
        
        if structural_analysis['class_coverage'] < 0.7:
            recommendations.append("类图与系统描述的匹配度较低，建议检查类图设计或完善系统描述")
        
        # 检查属性和方法的匹配情况
        for class_name, matches in structural_analysis['attribute_method_matches'].items():
            if not matches['attributes'] and not matches['methods']:
                recommendations.append(f"类 {class_name} 的属性和方法在系统描述中未体现，建议添加相关功能描述")
        
        return recommendations

# 使用示例
if __name__ == "__main__":
    try:
        # 创建分析器
        matcher = PlantUMLSystemMatcher()
        
        # 测试用例 - 从现有文件读取
        with open('test.puml', 'r', encoding='utf-8') as f:
            puml_code = f.read()
        
        with open('testDescription.txt', 'r', encoding='utf-8') as f:
            system_description = f.read()
        
        # 进行匹配分析
        result = matcher.analyze_plantuml_system_match(puml_code, system_description)
        
        # 输出结果
        print("=== PlantUML类图与系统描述匹配分析 ===")
        print(f"总体匹配度: {result['detailed_analysis']['overall_score']:.3f}")
        print(f"BERT语义匹配度: {result['detailed_analysis']['bert_score']:.3f}")
        print(f"结构匹配度: {result['detailed_analysis']['structural_score']:.3f}")
        
        print(f"\n=== 类图统计 ===")
        stats = result['detailed_analysis']['class_diagram_stats']
        print(f"类的数量: {stats['total_classes']}")
        print(f"关系数量: {stats['total_relationships']}")
        print(f"类名列表: {', '.join(stats['class_names'])}")
        
        print(f"\n=== 匹配详情 ===")
        details = result['detailed_analysis']['matching_details']
        print(f"匹配的类: {', '.join(details['matched_classes'])}")
        print(f"未匹配的类: {', '.join(details['unmatched_classes'])}")
        
        # print(f"\n=== 改进建议 ===")
        # for rec in result['detailed_analysis']['recommendations']:
            # print(f"- {rec}")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc() 