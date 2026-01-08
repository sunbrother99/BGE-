
"""
在短文本语料中挖掘正样本和硬负样本
正样本挖掘：
1、用bge-small-zh-v1.5等开源的embedding模型计算每个短文本的embedding表示
2、用faiss等向量库召回Top10余弦相似度得分大于0.9的样本，作为当前query的正样本
3、用bge-reranker-v2-m3等重排模型（精度更高的模型）计算当前query与召回的每个正样本之间的相似度得分，得分大于0.9为正样本，否则抛弃。

硬负样本（难负样本）挖掘：
1、用bge-small-zh-v1.5等开源的embedding模型计算每个短文本的embedding表示
2、用faiss等向量库召回Top30，作为当前query的负样本的候选样本
3、用bge-reranker-v2-m3等重排模型（精细化过滤）计算当前query与召回的每个负样本之间的相似度得分，得分小于0.2的视为硬负样本
"""
import json
import faiss
import numpy as np
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer

import pandas as pd
# 1. 配置与模型加载
model_name = '/code/wangyue/embeddingModel/bge-small-zh-v1.5'
reranker_name = '/code/wangyue/embeddingModel/bge-reranker-v2-m3'

print("正在加载模型...")
embed_model = SentenceTransformer(model_name)
reranker = FlagReranker(reranker_name, use_fp16=True)

# ---------------------------------------------------------
# 模拟数据（请替换为你的实际变量）
# all_corpus: 全量语料库列表
# generated_pairs: LLM生成的 [{"query": "...", "pos_idx": 123}]
# ---------------------------------------------------------
# all_corpus = ["如何重置支付密码", "怎么修改支付密码", "我想换个登录密码", "账号被封了", "今天天气怎么样"] * 100
# generated_pairs = [{"query": "我想改支付密码", "pos_idx": 0}]

data = pd.read_csv("./rag问题_可用.csv",sep=",")
print(data)
all_corpus = data["gen_questions"].unique().tolist()
generated_pairs = [{"query": all_corpus[i], "pos_idx": i} for i in range(len(all_corpus))]

# 2. 第一步：构建语义等价组 (Equivalence Mapping)
print("步骤1: 正在扫描语料库构建语义等价组...")
corpus_embeddings = embed_model.encode(all_corpus, normalize_embeddings=True, show_progress_bar=True)
dimension = corpus_embeddings.shape[1]

# 使用 Faiss 进行余弦相似度检索
index = faiss.IndexFlatIP(dimension)
index.add(corpus_embeddings.astype('float32'))

# 阈值 0.95 以上判定为语义等价
sim_threshold = 0.9
# D表示查询向量与每个检索到的文档结果之间的“相似度距离”，形状为 (nq, k)，n是这批数据的个数，q表示查询文本， k为检索 Top-K 的数量
# I表示top10的在原始 corpus（语料库）中，最相似向量的下标（index）
D, I = index.search(corpus_embeddings.astype('float32'), 10)  # 查 Top 10
# print(D,I)
equivalent_map = {}
for i in range(len(all_corpus)):
    # 找到相似度大于阈值的索引集合
    sim_indices = I[i][D[i] > sim_threshold]
    equivalent_map[i] = set(sim_indices.tolist())

# 3. 第二步：挖掘硬负样本并过滤
print("步骤2: 开始挖掘并使用 Reranker 精滤硬负样本...")
final_dataset = []

# 为了提高效率，这里可以分批处理
for item in generated_pairs:
    query = item['query']
    pos_idx = item['pos_idx']

    # 获取该正样本所属的整个等价组索引
    pos_group_indices = equivalent_map.get(pos_idx, {pos_idx})
    pos_texts = [all_corpus[idx] for idx in pos_group_indices]

    # a. 初筛候选负样本 (Top 30)
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_emb.astype('float32'), 30)

    candidate_negs = []
    for idx in indices[0]:
        # 【关键过滤 1】：如果索引在等价组里，绝对不能当负样本
        if idx in pos_group_indices:
            continue
        candidate_negs.append(all_corpus[idx])

    # b. 使用 Reranker 进行精细化清洗
    if not candidate_negs:
        continue

    rerank_pairs = [[query, neg] for neg in candidate_negs]
    rerank_scores = reranker.compute_score(rerank_pairs)

    # 【关键过滤 2】：双保险逻辑
    actual_hard_negs = []
    additional_positives = []

    for neg_text, score in zip(candidate_negs, rerank_scores):
        if score > 0.9: #如果 Reranker 分数很高（如 > 0.9），说明它是潜在的未发现正样本，作为正样本的补充
            additional_positives.append(neg_text)
        # 如果 Reranker 分数很高（如 > 0.8），说明它是潜在的未发现正样本，剔除
        if score > 0.8:
            continue
       
        # 只有 Reranker 觉得不相关的才是真正的硬负样本
        # BGE Reranker v2 m3 的阈值建议在 0 左右，甚至更低（-1, -2）代表极度不相关
        if score < 0.2:
            actual_hard_negs.append(neg_text)

        if len(actual_hard_negs) >= 5:  # 每个 Query 存 5 个硬负样本足够了
            break

    # 4. 组装数据
    if actual_hard_negs:
        final_dataset.append({
            "query": query,
            "pos": pos_texts+additional_positives,  # 存入所有等价的正样本，增强模型鲁棒性
            "neg": actual_hard_negs
        })

# 5. 保存结果
output_file = './finetune_data_cleaned.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for data in final_dataset:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"全部完成！高质量训练数据已保存至 {output_file}，共 {len(final_dataset)} 条。")
