"""
用大模型对相似度计算数据集进行二次清洗，过滤掉负样本中与query语义相同的样本
"""
import json

import os
import random
import time
import pandas as pd

os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_KEY'
from langchain.chat_models import ChatOpenAI
from openai import OpenAI


class ChatGPT():
    # llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    client = OpenAI()
    def predict(self,text):

        response = self.client.chat.completions.create(
            model="gpt-4o",  # gpt-4-1106-preview   gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content":text},
            ],
            response_format={"type": "json_object"},  # 👈 关键：强制 JSON 输出
            temperature=0.0  # 降低随机性，提高一致性
        )
        rsp = response.choices[0].message.content
        print(response.usage)
        return rsp


llm = ChatGPT()

# 这个 Prompt 的目的是把“长得像但语义不同”的真正硬负样本挑出来，剔除那些“长得像且语义也一致”的潜在正样本。
prompt_neg = """你是一个严谨的语义匹配专家。请判断以下 Query（用户查询）与 Candidate（候选文档）之间的语义关系。
判断标准：
完全匹配 (Match)：两者表达的意思完全一致，或者 Candidate 完美回答了 Query。
部分相关 (Partial)：话题相关，但具体意图不同（例如：修改支付密码 vs 修改登录密码）。
不相关 (Irrelevant)：话题完全不同。

输入：
Query: "{query}"
Candidate: "{neg_text}"
输出要求： 请仅输出 JSON 格式，包含 label（取值为 "完全匹配", "部分相关", "不相关"）和 reason（简短理由）。
注意： 只有当 label 为 "部分相关" 或 "不相关" 时，该候选文档才能作为负样本。如果 label 为 "完全匹配"，请务必标记。"""
def verify_data(item):
    query = item['query']
    pos_list = item['pos']
    neg_list = item['neg']

    clean_negs = []
    for neg in neg_list:
        # 调用 LLM 进行负样本校验
        # print(prompt_neg.format(query=query, neg_text = neg))

        response = llm.predict(prompt_neg.format(query=query, neg_text=neg))
        # print(response)
        response_json = json.loads(response)
        # 只有真正不相关的才保留
        if response_json['label'] in ["部分相关", "不相关"]:
            clean_negs.append(neg)

    if len(clean_negs) > 0:
        return {
            "query": query,
            "pos": pos_list,
            "neg": clean_negs
        }
    return None

# item = {"query": "询问装修程度。", "pos": ["询问装修程度。"], "neg": ["装修标准是多少钱？", "带装修吗？标准是什么？", "装修交付什么标准？", "{可选装修情况}{用品味装修}{房型特点}", "装修标准简单介绍"]}
# verify_data(item)
# 处理并保存
with open('./finetune_data_cleaned.jsonl', 'r', encoding='utf-8') as f:
    raw_mined_data = f.readlines()
    # print(raw_mined_data)


with open('./final_bge_train_data.jsonl', 'w', encoding='utf-8') as f:
    i = 0
    length = len(raw_mined_data)
    for item in raw_mined_data:
        i += 1
        print("进度:{}/{}".format(i, length))
        verified_item = verify_data(json.loads(item))
        if verified_item:
            f.write(json.dumps(verified_item, ensure_ascii=False) + '\n')
