from config import *
from openai import OpenAI
import json
from prompt.prompt import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import *

#generate domain name prompt
def generate_domain_name_prompt(model_skeleton,domains_have_generated):
    return PROMPT_GEN_DOMAIN.format(model_skeleton,domains_have_generated)

#generate gen_model prompt
def generate_gen_model_prompt(domain, model_framework):
    return PROMPT_GEN_MODEL.format(domain, model_framework)

#generate replace_model prompt
def generate_replace_model_prompt(original_model, generated_mapping):
    return PROMPT_REPLACE_MODEL.format(original_model, generated_mapping)

#generate verify_model prompt
def generate_verify_model_prompt(generated_model, domain):
    return PROMPT_VERIFY_MODEL.format(generated_model, domain)

#generate gen_model_description prompt
def generate_gen_model_description_prompt(domain_model, domain):
    return PROMPT_GEN_MODEL_DESCRIPTION.format(domain_model, domain)

#generate verify_model_description prompt
def generate_verify_model_description_prompt(generated_description, domain_model):
    return PROMPT_VERIFY_MODEL_DESCRIPTION.format(generated_description, domain_model)

def read_puml_file(file_path: str) -> str | None:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the file: {e}")
        return None

#run LLM
def run_llm(prompt):
    client = OpenAI(
        api_key = running_params['API_KEY'],
        base_url = running_params['BASE_URL'],
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model = running_params['llm'],
        temperature = running_params['temperature'],
        max_tokens = running_params['max_tokens'],
    )
    return chat_completion.choices[0].message.content

def is_semantically_similar(name: str, name_list: list) -> bool:
    """
    检查给定名称与列表中任何名称是否语义相似
    
    参数:
    name (str): 要检查的名称
    name_list (list): 名称列表
    
    返回:
    bool: 如果存在相似度超过阈值的名称返回True, 否则返回False
    """
    # 加载预训练的embedding模型
    model = SentenceTransformer('all-MiniLM-L6-v2')
    threshold = running_params['embedding_threshold']  # 相似度阈值，可以根据需要调整
    
    # 计算单个名称的embedding
    name_embedding = model.encode([name])
    
    # 计算名称列表的embeddings
    name_list_embeddings = model.encode(name_list)
    
    # 计算余弦相似度
    similarities = cosine_similarity(name_embedding, name_list_embeddings)[0]
    
    # 检查是否有相似度超过阈值的项
    return np.max(similarities) >= threshold