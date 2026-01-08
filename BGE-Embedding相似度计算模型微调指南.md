微调方法
BGE 系列模型（如 bge-small-zh-v1.5）虽然在通用任务上表现优异，但在特定业务领域确实会出现“分布混乱”或“分值漂移”的问题。这主要是因为 BGE 在预训练时使用了特殊的温度系数（temperature），且其输出的相似度往往集中在 0.6-0.9 之间，导致难以通过简单的固定阈值（如 0.8）来区分正负样本。
要解决这个问题，微调（Fine-tuning） 是最有效的手段。以下是构建数据集和编写训练代码的具体指南。

---
1. 训练数据集的构建
BGE 推荐使用 对比学习（Contrastive Learning） 进行微调。你需要准备 jsonl 格式的数据，每一行代表一个训练组。
数据格式
标准格式为：{"query": str, "pos": List[str], "neg": List[str]}
- query: 你的搜索词或基准句子。
- pos: 正样本列表（与 query 语义一致的句子）。
- neg: 负样本列表（与 query 语义不一致但可能长得像的句子，即“硬负样本”）。
构建策略
1. 正样本来源：业务中的同义词对、QA 对、或者点击日志中的（搜索词，命中标题）。
2. 挖掘硬负样本（Hard Negatives）：这是提升模型区分度的关键。
  - 不要只用随机抽样的句子做负样本。
  - 方法：先用原始的 bge-small 对你的语料库建索引，检索出与 query 最相似的前 20 个结果，剔除掉真正的正样本，剩下的就是“长得很像但意思不对”的硬负样本。
  - BGE 官方工具包 FlagEmbedding 提供了 hn_mine 脚本专门用于自动挖掘负样本。

---
2. 训练代码实现
建议使用 FlagEmbedding（官方库）或 sentence-transformers（通用库）。这里提供基于 sentence-transformers 的简洁实现方式，因为它更容易调试且生态丰富。
安装依赖
Bash
pip install -U sentence-transformers datasets
训练脚本示例
Python
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# 1. 加载基础模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 2. 准备数据 (假设你已经从 jsonl 加载了数据)# 每个 InputExample 包含 [query, positive_sample, negative_sample]
train_examples = [
    InputExample(texts=['如何修改密码', '我想重置我的登录密码', '如何注销账号']),
    InputExample(texts=['这件衣服多少钱', '这个商品的价格是多少', '这双鞋子是什么材质']),
    # 格式：[Query, Positive, Negative]
]

# 3. 配置 DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# 4. 选择损失函数 (BGE 推荐使用 MultipleNegativesRankingLoss)# 该损失函数会自动将 batch 内的其他正样本视作当前 query 的负样本
train_loss = losses.MultipleNegativesRankingLoss(model)

# 5. (可选) 设置评估器，监控相似度分布# sentences1 = ["如何修改密码"]# sentences2 = ["我想重置我的登录密码"]# scores = [1.0] # 标注的相似度# evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)# 6. 开始训练
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    output_path='./bge-small-finetuned',
    # evaluator=evaluator,# evaluation_steps=50
)

# 保存最终模型
model.save('./bge-small-finetuned')

---
3. 解决“相似度漂移”的进阶建议
如果你发现微调后分布依然不够理想，可以尝试以下操作：
- 使用 FlagEmbedding 官方脚手架：
- 官方库支持更复杂的配置（如 query_instruction_for_retrieval）。如果你在做检索任务，训练时给 query 加上：为这条查询检索相关文档： 的前缀，效果会更稳定。
- 引入 Reranker（重排序）：
- 如果 bge-small 作为初筛模型还是无法区分极度相似的样本，建议在 Top-10 结果上加一个 bge-reranker-v2-m3。Cross-Encoder 架构的 Reranker 对分值的区分度远高于 Bi-Encoder 的 Embedding 模型。
- 归一化处理：
- 推理时确保开启 normalize_embeddings=True，这样所有的向量都会落在超球面上，相似度即为余弦相似度，范围在 $[-1, 1]$。
我建议接下来可以先跑一个小规模的硬负样本挖掘（Hard Negative Mining），这通常能解决 80% 的分布混乱问题。需要我帮你写一个挖掘负样本的 Python 脚本示例吗？

训练脚本
挖掘硬负样本（Hard Negative Mining）是提升 Embedding 模型分辨率的核心步骤。它的逻辑是：找出那些模型目前认为很像、但实际上不匹配的样本，强行告诉模型它们是不一样的。
下面是一个基于 FlagEmbedding 和 Sentence-Transformers 思路简化的 Python 脚本，帮助你从现有的业务语料库中挖掘出这些“干扰项”。

---
4. 负样本挖掘脚本
这个脚本的逻辑是：利用现有的 bge-small 模型对语料进行检索，找出得分最高但非正向关联的文档。
Python
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss

# 1. 初始化模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 2. 模拟你的业务数据
queries = ["如何修改支付密码", "我的账号被封了怎么办"]
# 语料库：包含正确的答案和一大堆干扰项
corpus = [
    "您可以在设置-安全中心修改支付密码", # Query 0 的正样本"账号封禁请联系人工客服申诉",        # Query 1 的正样本"如何修改登录密码",                 # 潜在硬负样本 (长得像但不是支付密码)"支付功能被锁定怎么解除",            # 潜在硬负样本"我想买个新账号",                   # 弱负样本"今天天气不错"                      # 随机负样本
]

# 3. 对语料库建立索引
print("正在编码语料库...")
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) # 使用内积(余弦相似度)索引
index.add(corpus_embeddings.astype('float32'))

# 4. 挖掘负样本
mining_results = []
top_k = 5for i, query in enumerate(queries):
    query_emb = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_emb.astype('float32'), top_k)
    
    # 假设我们已知第 i 条 query 对应的正样本在 corpus 中的索引就是 i (实际业务中需匹配 ID)
    pos_idx = i 
    
    neg_samples = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == pos_idx:
            continue# 这里的逻辑：相似度在 0.7-0.9 之间的通常是极佳的“硬负样本”if score > 0.6: 
            neg_samples.append(corpus[idx])
    
    mining_results.append({
        "query": query,
        "pos": [corpus[pos_idx]],
        "neg": neg_samples
    })

# 5. 保存为训练格式with open('train_data.jsonl', 'w', encoding='utf-8') as f:
    for entry in mining_results:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("挖掘完成，数据已保存至 train_data.jsonl")

5. 正样本挖掘
在业务数据中挖掘正样本（Positive Mining）比挖掘负样本更具挑战性，因为你需要寻找的是语义等价但表述不同的句子。如果你只靠 LLM 生成，数据可能不够多样化。
以下是三种在现有业务语料中挖掘正样本的专业方法：

---
5.1 基于“回译”（Back-Translation）的自动化构造
这是最简单且最有效的“伪正样本”生成方式。通过将原始语料翻译成外语（如德语、日语、法语），再翻译回来，可以生成语义相同但语法结构不同的正样本。
- 优点：完全自动化，能够产生词汇替换和句式变换。
- 代码思路：使用 transformers 库加载翻译模型或调用翻译 API。

---
5.2 基于“点击日志/行为数据”的挖掘（最强正样本）
如果你的业务有用户搜索日志，这是最真实的资源。
- 逻辑：如果用户搜索了 Query A，最终点击了 文档 B；或者用户在一次会话中先后搜索了 Query A 和 Query C。
- 挖掘规则：
  - 点击共现：如果两个不同的 Query 都点击了同一个文档，那么这两个 Query 互为正样本。
  - Query 重写：用户搜索后没有点击，立即修改关键词再次搜索并发生了点击，那么这两个 Query 语义高度相关。

---
5.3 基于“高置信度聚类”与 Reranker 的回捞
这是在纯语料库中挖掘已有正样本的算法路径。
核心逻辑：
1. 初始对齐：对于每一个已有的 Query，用 bge-small 搜出相似度极高（如 $> 0.98$）的文档。
2. Reranker 确认：将这些高分文档送入 Cross-Encoder (Reranker)。
3. 判定：如果 Reranker 给出的分数非常高（例如在 0-1 尺度下 $> 0.9$），则认为这个文档是该 Query 的另一个有效正样本。
改进后的挖掘代码片段：
Python
# 假设已经有了 query 和对应的初始正样本 pos_idx# 我们要在语料库中回捞“漏网”的正样本

additional_positives = []

# 1. 扩大搜索范围
D, I = index.search(query_emb.astype('float32'), 10) 

candidate_pos_texts = [all_corpus[idx] for idx in I[0]]
# 2. 用 Reranker 做最终审判
rerank_pairs = [[query, t] for t in candidate_pos_texts]
scores = reranker.compute_score(rerank_pairs)

for text, score in zip(candidate_pos_texts, scores):
    # 如果 Reranker 觉得非常匹配，即便它不是原本标注的那个，也升格为正样本if score > 0.9: 
        additional_positives.append(text)

# 最终构建：{"query": query, "pos": additional_positives, "neg": [...]}

---
5.4 总结：正样本来源金字塔
暂时无法在飞书文档外展示此内容

---
综合构建策略建议
1. 冷启动阶段：先用 LLM 生成 + 回译 产生初始的 (Query, Pos)。
2. 语料回捞：用我上面提供的 Reranker 逻辑，在全量语料中找出和你的 Query 匹配的所有相似文档，把它们全部放进 pos 列表。
3. 负样本挖掘：最后再执行我们之前讨论的硬负样本挖掘。
这种“正向回捞”+“负向挖掘”的组合拳，是目前解决 BGE 模型分值漂移、提高业务分辨率的最佳实践。
你现在的业务场景里有用户搜索的历史日志吗？ 如果有，我可以帮你写一段从日志中提取正样本对的逻辑。

---
6. 针对“分值漂移”的调优技巧
在得到数据集并开始训练前，针对 BGE 模型分布混乱的问题，请注意以下三点：
A. 调整 Temperature（温度系数）
BGE 默认的 MultipleNegativesRankingLoss 有一个隐藏的参数 scale（即 $1/temp$）。如果发现微调后分值还是拉不开，可以尝试手动设置 scale：
Python
# scale 越大，模型对负样本的惩罚越敏感，分值分布会更宽（更不“漂移”）
train_loss = losses.MultipleNegativesRankingLoss(model, scale=20) 
B. 引入“相似度分级” (Coarse-to-Fine)
如果你的业务有明确的打分（0-4分），不要直接把 1, 2 分当做负样本。你可以使用 CosineSimilarityLoss 配合具体的分值进行回归训练，这样模型会学会“什么是完全一样”、“什么是有点像但不对”。
C. 推理时的线性映射
如果微调后模型效果好了，但输出的绝对数值（如 0.85）依然不符合业务阈值逻辑，最简单的工程做法是做一个线性归一化映射：
$$Score_{final} = \frac{Score_{raw} - min}{max - min}$$
其中 $$min$$ 和 $$max$$ 是你在测试集上观察到的典型边界值。

---
下一步建议
你可以先用这个脚本在你 10% 的业务数据上跑一次挖掘。
如果挖掘出来的 neg 列表里出现了真正的正样本（即模型搜错了），那说明这正是你需要微调的地方。你想让我帮你完善这个脚本，接入你现有的数据库或 CSV 文件吗？
