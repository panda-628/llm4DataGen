import os
from openai import OpenAI
from config import *
from prompt import *
from prompt.utils import *
import time
from coderBERT_analyzer import *

def main():
    path = file['path']
    os.chdir(path)

    #读取puml格式文件
    model_framework = read_puml_file('C:/AppAndData/codeAndproject/llm4Data/example3.puml')
    #创建已经生成的领域名称列表
    domains_have_generated = []
    time_stamp = time.strftime("%m%d-%H%M", time.localtime())
    new_folder = running_params['llm'] + '-' + time_stamp + '- example3'
    os.makedirs(new_folder)

    for i in range(1,running_params['cycles'] + 1):

        prompt_gen_domain_name_by_skeleton = generate_domain_name_prompt(model_framework,domains_have_generated)
        #将生成的domain name提取出来
        domain_name = run_llm(prompt_gen_domain_name_by_skeleton)
        print("AI生成的内容：", domain_name)

        #对生成的domain_name与列表中已经存在的领域名称进行embedding相似度比较
        if len(domains_have_generated) == 0:
            #如果列表为空，直接添加生成的领域名称
            domains_have_generated.append(domain_name)
        else:
            if is_semantically_similar(domain_name, domains_have_generated):
                continue  # 如果与已有领域名称语义相似，则跳过当前循环
            else:
                domains_have_generated.append(domain_name)  # 否则添加到列表中

        print(f"Generated domain name: {domain_name}")

        #创建当前领域的结果文件夹
        domain_result_folder = f'{path}/{new_folder}/{i}'
        os.makedirs(domain_result_folder)

        result_file = f'{path}/{new_folder}/{i}/result.csv'
        model_framework_file = f'{path}/{new_folder}/{i}/domain.puml'
        description_file = f'{path}/{new_folder}/{i}/description.txt'

        f_result_file = open(result_file, 'w', encoding='utf-8')
        f_model_framework_file = open(model_framework_file, 'w', encoding='utf-8')
        f_description_file = open(description_file, 'w', encoding='utf-8')

        #生成模型映射
        prompt_gen_model_mapping = generate_gen_model_prompt(domain_name, model_framework)
        gen_model_mapping_answer = run_llm(prompt_gen_model_mapping)
        f_result_file.write(f'gen_model_mapping_answer: {gen_model_mapping_answer}\n')

        #替换模型映射
        prompt_replace_model_mapping = generate_replace_model_prompt(model_framework, gen_model_mapping_answer)
        replace_model_mapping_answer = run_llm(prompt_replace_model_mapping)
        f_result_file.write(f'replace_model_mapping_answer: {replace_model_mapping_answer}\n')

        #验证模型映射
        prompt_verify_model_mapping = generate_verify_model_prompt(replace_model_mapping_answer, domain_name)
        verify_model_mapping_answer = run_llm(prompt_verify_model_mapping)
        print("验证模型映射结果："+ verify_model_mapping_answer)
        f_result_file.write(f'verify_model_mapping_answer: {verify_model_mapping_answer}\n')

        #将验证模型映射中的模型映射提取出来
        start_index = verify_model_mapping_answer.find('corrected model')
        end_index = verify_model_mapping_answer.find('@enduml')
        extracted_model_mapping = verify_model_mapping_answer[start_index + len('corrected model'):end_index].strip()
        print("提取出的模型映射是："+ extracted_model_mapping)
        f_result_file.write(f'extracted_model_mapping: {extracted_model_mapping}\n')
        f_model_framework_file.write(f'{extracted_model_mapping}\n@enduml\n')

        #生成模型描述
        prompt_gen_model_description = generate_gen_model_description_prompt(extracted_model_mapping, domain_name)
        gen_model_description_answer = run_llm(prompt_gen_model_description)
        f_result_file.write(f'gen_model_description_answer: {gen_model_description_answer}\n')
        f_description_file.write(f'original_model_description: {gen_model_description_answer}\n')

        #验证模型描述
        prompt_verify_model_description = generate_verify_model_description_prompt(gen_model_description_answer, extracted_model_mapping)
        verify_model_description_answer = run_llm(prompt_verify_model_description)
        f_result_file.write(f'verify_model_description_answer: {verify_model_description_answer}\n')
        f_description_file.write(f'model_description: {verify_model_description_answer}\n')

        f_result_file.close()
        f_model_framework_file.close()
        f_description_file.close()
        print(f"Results saved to {result_file}")

        #开始评估
        analyzer = PlantUMLSystemMatcher()
        
        with open(model_framework_file, 'r', encoding='utf-8') as f:
            puml_code = f.read()

        with open(description_file, 'r', encoding='utf-8') as f:
            #从description_file中提取以### Final Modified Description开头，后面的内容
            description = f.read().split('Final Modified Description')[1]

        result = analyzer.analyze_plantuml_system_match(puml_code, description)
        print(result)

        #创建存储匹配结果的文件
        bert_result_file = f'{path}/{new_folder}/{i}/match_result.txt'
        f_bert_result_file = open(bert_result_file, 'w', encoding='utf-8')
        # 输出结果
        f_bert_result_file.write("=== PlantUML类图与系统描述匹配分析 ===\n")
        f_bert_result_file.write(f"总体匹配度: {result['detailed_analysis']['overall_score']:.3f}\n")
        f_bert_result_file.write(f"BERT语义匹配度: {result['detailed_analysis']['bert_score']:.3f}\n")
        f_bert_result_file.write(f"结构匹配度: {result['detailed_analysis']['structural_score']:.3f}\n")
        
        f_bert_result_file.write(f"\n=== 类图统计 ===")
        stats = result['detailed_analysis']['class_diagram_stats']
        f_bert_result_file.write(f"类的数量: {stats['total_classes']}")
        f_bert_result_file.write(f"关系数量: {stats['total_relationships']}")
        f_bert_result_file.write(f"类名列表: {', '.join(stats['class_names'])}")
        
        f_bert_result_file.write(f"\n=== 匹配详情 ===")
        details = result['detailed_analysis']['matching_details']
        f_bert_result_file.write(f"匹配的类: {', '.join(details['matched_classes'])}")
        f_bert_result_file.write(f"未匹配的类: {', '.join(details['unmatched_classes'])}")

        f_bert_result_file.close()

    
    #print("生成的全部领域模型名称：", domains_have_generated)

if __name__ == '__main__':
    main()