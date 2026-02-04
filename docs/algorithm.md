# MDKG 扩展模块算法文档

## 📚 概述

本文档详细介绍了 MDKG (Mental Disorder Knowledge Graph) 项目中两个核心扩展模块的算法原理和实现细节：

1. **Graph RAG (图增强检索增强生成)** - 基于 KGARevion 论文
2. **Link Prediction (药物重定位预测)** - 基于 TxGNN 论文

**参考论文**:
- KGARevion: https://arxiv.org/abs/2410.04660
- GraphRAG (From Local to Global): https://arxiv.org/abs/2404.16130

---

## 目录

- [第一部分：Graph RAG 模块](#第一部分graph-rag-模块)
  - [1.1 论文背景](#11-论文背景)
  - [1.2 核心算法](#12-核心算法)
  - [1.3 实现架构](#13-实现架构)
  - [1.4 工作流程](#14-工作流程)
  - [1.5 常见问题解答 (FAQ)](#15-常见问题解答-faq)
  - [1.6 评估方法](#16-评估方法)
  - [1.7 优化技术：Community Detection](#17-优化技术community-detection)
- [第二部分：Link Prediction 模块](#第二部分link-prediction-模块)
  - [2.1 论文背景](#21-论文背景)
  - [2.2 核心算法](#22-核心算法)
  - [2.3 实现架构](#23-实现架构)
  - [2.4 训练流程](#24-训练流程)
- [第三部分：模块对比与集成](#第三部分模块对比与集成)

---

## 第一部分：Graph RAG 模块

### 1.1 论文背景

**参考论文**: [KGARevion: Knowledge Graph Based Agent for Complex, Knowledge-Intensive QA in Medicine](https://arxiv.org/abs/2410.04660)

#### 1.1.1 问题定义

传统的大型语言模型 (LLM) 在生物医学问答中面临以下挑战：

- **幻觉问题 (Hallucination)**: LLM 可能生成看似合理但实际错误的医学知识
- **知识时效性**: 模型训练数据的截止日期限制了其对最新医学知识的掌握
- **推理透明性**: 缺乏对答案生成过程的可解释性

#### 1.1.2 解决方案

KGARevion 提出了一种基于知识图谱的智能体框架，通过以下机制解决上述问题：

1. **知识图谱锚定 (Knowledge Grounding)**: 将 LLM 生成的知识与结构化知识图谱进行验证
2. **迭代修正 (Iterative Revision)**: 对不一致的知识进行循环修正
3. **证据追溯 (Evidence Tracing)**: 为生成的答案提供知识图谱证据支持

### 1.2 核心算法

#### 1.2.1 四动作框架 (Four-Action Framework)

**重要说明**: 论文采用 **True/False 二元分类**，而非置信度分数。

Graph RAG 采用四个核心动作构成的推理循环：

```
┌─────────────────────────────────────────────────────────────┐
│                      Query (用户问题)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ACTION 1: Generate                        │
│  • 提取医学概念 (疾病、症状、药物、基因等)                      │
│  • 生成候选三元组 (head, relation, tail)                      │
│  • Choice-Aware: 每个答案选项生成不同三元组                    │
│  • Non-Choice-Aware: 仅从问题生成三元组                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ACTION 2: Review                         │
│  • 步骤1: 检查实体是否可映射到KG（UMLS Code映射）               │
│  • 步骤2: 使用Fine-tuned LLM判断True/False                    │
│                                                              │
│  分类结果:                                                    │
│  • TRUE:       两实体可映射且LLM判定为True                     │
│  • FALSE:      两实体可映射但LLM判定为False                    │
│  • INCOMPLETE: 实体无法映射（保留三元组）                       │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌───────────┐              ┌────────────────┐
        │ TRUE      │              │  FALSE         │
        │ (集合V)   │              │  (需修正)      │
        └───────────┘              └────────────────┘
                │                           │
                │                           ▼
                │          ┌─────────────────────────────────────┐
                │          │          ACTION 3: Revise           │
                │          │  • LLM修正head/tail实体或关系         │
                │          │  • 重新提交Review验证                 │
                │          │  • 迭代直到True或达最大轮数k          │
                │          └─────────────────────────────────────┘
                │                           │
                ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     ACTION 4: Answer                         │
│  • 使用True三元组 (V) + Incomplete三元组                      │
│  • 结合知识图谱上下文                                         │
│  • 生成有证据支持的答案                                       │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2.2 三元组验证算法 (Review Action)

**论文实现方法** (Section 3.2):

1. **实体映射**: 使用 UMLS Code 将实体映射到 KG
2. **LLM验证**: Fine-tuned LLM 输出 True/False
3. **Soft Constraint Rule**: 处理KG不完整情况

```python
def review_triplet(triplet, knowledge_graph, llm):
    """
    三元组验证算法 - 严格遵循KGARevion论文
    
    返回: TripletStatus (TRUE, FALSE, INCOMPLETE)
    """
    # Step 1: 实体映射检查
    head_mapped = kg.can_map_entity(triplet.head)  # UMLS Code
    tail_mapped = kg.can_map_entity(triplet.tail)
    
    if head_mapped and tail_mapped:
        # 两实体都可映射 → 使用LLM判定
        is_true = llm.verify_triplet_completion(triplet)
        return TripletStatus.TRUE if is_true else TripletStatus.FALSE
    else:
        # Soft Constraint: 无法完全映射 → 保留三元组
        return TripletStatus.INCOMPLETE
```

#### 1.2.3 Embedding对齐机制

论文使用以下方法对齐 KG embedding 与 LLM token embedding:

$$
e_{aligned} = \text{FFN}(\text{Attention}(e_{KG}, E_{LLM}))
$$

其中:
- $e_{KG}$: TransE 训练的 KG embedding
- $E_{LLM}$: LLM token embeddings
- Attention + FFN: 学习对齐映射

### 1.3 实现架构

#### 1.3.1 模块组成

```
GraphRAG/
├── kgarevion_agent.py    # 核心KGARevion实现
└── graph_rag_demo.py     # 演示脚本

核心类:
├── KGARevionAgent            # 主 RAG Agent (KGARevion)
├── KnowledgeGraphManager     # 知识图谱管理
├── CommunityManager          # 社区检测优化
├── LLMBackend (Abstract)     # LLM 后端接口
│   ├── OpenAIBackend         # OpenAI API
│   └── OllamaBackend         # 本地 Ollama
└── Triplet / QueryResult     # 数据结构
```

#### 1.3.2 知识图谱管理器

`KnowledgeGraphManager` 负责与 MDKG 知识图谱的交互：

| 功能 | 方法 | 说明 |
|------|------|------|
| 数据加载 | `_load_data()` | 加载实体链接和三元组 |
| 索引构建 | `_build_entity_index()` | 构建实体到三元组的快速索引 |
| 实体查询 | `get_entity_info()` | 获取链接的本体信息 |
| 关系检索 | `find_related_triplets()` | 查找相关三元组 |
| 三元组验证 | `verify_triplet()` | 验证三元组一致性 |

#### 1.3.3 支持的关系类型

```python
RELATION_TYPES = [
    "causes",           # 导致
    "treats",           # 治疗
    "associated_with",  # 关联
    "symptom_of",       # ...的症状
    "risk_factor_for",  # ...的风险因素
    "comorbid_with",    # 共病
    "contraindicated",  # 禁忌
    "side_effect_of",   # 副作用
    "biomarker_for",    # 生物标志物
    "affects",          # 影响
    "located_in",       # 位于
    "interacts_with",   # 相互作用
    "inhibits",         # 抑制
    "activates",        # 激活
    "diagnoses",        # 诊断
    "prevents",         # 预防
    "worsens",          # 恶化
    "improves",         # 改善
    "phenotype_of"      # 表型
]
```

### 1.4 工作流程

#### 1.4.1 完整查询流程示例

```python
# 1. 初始化组件
kg_manager = KnowledgeGraphManager({
    'entity_linking_path': 'output/entity_linking_results.json',
    'predictions_path': 'output/sampling_json_run_v1_sampled.json'
})

llm = OpenAIBackend(api_key="...", model="gpt-4")
rag = MentalDisorderGraphRAG(kg_manager, llm)

# 2. 执行查询
result = rag.query(
    "What are the treatment options for schizophrenia?",
    verbose=True
)

# 3. 结果结构
{
    "query": "What are the treatment options...",
    "answer": "Schizophrenia treatment typically includes...",
    "verified_triplets": [
        {"head": "risperidone", "relation": "treats", "tail": "schizophrenia"},
        {"head": "olanzapine", "relation": "treats", "tail": "schizophrenia"}
    ],
    "medical_concepts": ["schizophrenia", "antipsychotic", "treatment"],
    "confidence": 0.85,
    "reasoning_trace": [
        "Generated 5 triplets from 3 concepts",
        "Verified 3, rejected 2 triplets",
        "Round 1: Revised 2, verified 1"
    ]
}
```

---

### 1.5 常见问题解答 (FAQ)

基于 KGARevion 论文 (https://arxiv.org/abs/2410.04660) 和 GraphRAG 论文 (https://arxiv.org/abs/2404.16130)。

#### Q1: 如何评估 Graph RAG 的性能？使用什么评估指标？

**评估指标**: **Accuracy (准确率) + Standard Deviation (标准差)**

根据 KGARevion 论文 Table 2 和 Section 4.3:
- 在多个基准数据集上评估: MedQA, MedMCQA, MMLU-Med, PubMedQA
- 运行 **3次**，报告 **平均准确率 ± 标准差**
- 数据划分: 100/400/2000 作为 dev/test/train set

$$
\text{Accuracy} = \frac{\text{正确预测数}}{\text{总样本数}}
$$

**示例结果格式**: `78.65 ± 0.4%`

#### Q2: Confidence Score 是什么含义？表示什么？

**重要澄清**: **KGARevion 论文不使用 Confidence Score！**

论文采用 **True/False 二元分类**:
- 使用 fine-tuned LLM 对三元组输出 True 或 False
- 不是概率或连续置信度
- LLM 在 KG completion 任务上微调后直接判断

如果需要连续值，可以使用 LLM 输出的 logits 概率，但论文本身只用二元判定。

#### Q3: 初始的 Confidence 是如何确定的？

**论文中不存在"初始 Confidence"概念**。

工作流程:
1. **Generate**: 生成三元组 (无置信度)
2. **Review**: 直接通过 LLM 判定 True/False
3. 不需要初始化置信度

#### Q4: Triplets 是如何与 KG 进行 Matching 的？

**两步匹配策略** (Section 3.2.1-3.2.2):

**Step 1 - 实体映射 (Entity Mapping)**:
```
三元组实体 → UMLS Code → KG 实体
```
- 使用 UMLS 标准医学术语作为桥接
- 若实体无法映射 → `INCOMPLETE` 状态 (保留三元组)

**Step 2 - LLM 验证 (Triplet Verification)**:
- 获取关系描述 D(r)
- 使用 fine-tuned LLM (LoRA + TransE embeddings)
- 输出 True 或 False

**Embedding 对齐**:
$$
e_{aligned} = \text{FFN}(\text{Attention}(e_{TransE}, E_{LLM}))
$$

#### Q5: Match 有哪些类型？

**只有两种概念类型**:

| 类型 | 条件 | 结果 |
|------|------|------|
| **Entity Mapping** | 实体是否可映射到 KG (via UMLS) | Mappable / Not Mappable |
| **Triplet Classification** | LLM 判定三元组正确性 | True / False |

**三元组最终状态**:
- `TRUE`: 实体可映射 + LLM 判定 True
- `FALSE`: 实体可映射 + LLM 判定 False (需要 Revise)
- `INCOMPLETE`: 实体无法映射 (保留使用)

#### Q6: Revise Action 是如何实现的？

**论文 Section 3.3 和 Appendix E.3**:

**核心思路**: 让 LLM 修正被判定为 False 的三元组

```
Prompt 模板 (Appendix E.3):
### Instruction:
Given the following triplets consisting of a head entity, relation, and tail entity, 
please review and revise the triplets to ensure they are correct and helpful for 
answering the given question...

### Input:
Triplets: [(head1, rel1, tail1), ...]
Questions: {query}

### Response:
```

**迭代过程**:
1. 收集 False 三元组 (F 集合)
2. 提交给 LLM 修正
3. 对修正后的三元组重新 Review
4. 重复直到 True 或达到最大轮数 k (默认 k=2)

#### Q7: KGARevion 论文还有什么创新点？

1. **Structural-Semantic Embedding Alignment**
   - TransE 学习 KG 结构 embeddings
   - Attention + FFN 对齐到 LLM token embeddings
   - 支持 LoRA fine-tuning

2. **Question-Type Adaptive Strategy**
   - Choice-Aware: 对每个答案选项生成不同三元组
   - Non-Choice-Aware: 仅从问题生成 (Yes/No 类型)

3. **KG as Verifier (not Retriever)**
   - 不是从 KG 检索答案
   - 而是用 KG 验证 LLM 生成的知识
   - 解决幻觉问题

4. **Soft Constraint Rule**
   - 处理 KG 不完整情况
   - 无法映射的实体 → 保留三元组

#### Q8: 如何用 Community Detection 优化 Graph RAG？

**参考 GraphRAG 论文** (https://arxiv.org/abs/2404.16130):

**Leiden Algorithm** 用于层次化社区检测:

```
优化前复杂度: O(|Q| × |KG|)   # 全 KG 搜索
优化后复杂度: O(|Q| × |C|)    # 仅搜索相关社区
```

**实现步骤**:

1. **构建图结构**:
```python
G = nx.Graph()
for triplet in triplets:
    G.add_edge(triplet.head, triplet.tail, relation=triplet.relation)
```

2. **Leiden 社区检测**:
```python
from graspologic.partition import leiden
partition = leiden(G, resolution=1.0)
# 或使用 Louvain 作为替代
```

3. **社区范围搜索**:
```python
def find_triplets_optimized(query_entities):
    # 找到 query 实体所属的社区
    relevant_communities = get_communities_for_entities(query_entities)
    # 仅搜索这些社区内的三元组
    return search_within_communities(relevant_communities)
```

4. **层次化总结** (可选):
   - Community Level 0: 最细粒度
   - Community Level 1: 合并相似社区
   - Map-Reduce: 自底向上汇总

---

### 1.6 评估方法

#### 1.6.1 评估模块设计

参见 `GraphRAG/kgarevion_agent.py`

```python
@dataclass
class EvaluationResult:
    question: str
    predicted_answer: str
    ground_truth: str
    is_correct: bool
    true_triplets_count: int
    false_triplets_count: int
    incomplete_triplets_count: int
    
@dataclass
class EvaluationMetrics:
    accuracy: float
    std_deviation: float
    total_samples: int
    runs: int  # 通常为3
```

#### 1.6.2 多次运行评估

根据 KGARevion 论文 Table 2：运行 3 次，报告平均准确率 ± 标准差

```python
def evaluate_with_std(agent, dataset, runs=3):
    accuracies = []
    for _ in range(runs):
        acc = evaluate_single_run(agent, dataset)
        accuracies.append(acc)
    
    return {
        "accuracy": np.mean(accuracies),
        "std": np.std(accuracies),
        "runs": runs
    }
```

---

### 1.7 优化技术：Community Detection

#### 1.7.1 Leiden 算法原理

```
Input:  Graph G = (V, E)
Output: Community partition

1. Local Moving Phase:
   - 将每个节点移动到最大化模块度的社区
   
2. Refinement Phase:
   - 对社区进行细化调整
   
3. Aggregation Phase:
   - 将社区聚合成超节点
   - 递归重复直到收敛
```

#### 1.7.2 实现位置

- 核心实现: `GraphRAG/kgarevion_agent.py` → `CommunityManager` 类
- 功能:
  - `build_graph_from_triplets()`: 构建 NetworkX 图
  - `detect_communities()`: Leiden/Louvain 社区检测
  - `find_triplets_in_communities()`: 社区范围三元组搜索

---

## 第二部分：Link Prediction 模块

### 2.1 论文背景

**参考论文**: [TxGNN: Zero-shot prediction of therapeutic use of drugs with geometric deep learning](https://www.nature.com/articles/s41591-024-03233-x) (Nature Medicine, 2024)

#### 2.1.1 问题定义

药物重定位 (Drug Repurposing) 面临的核心挑战：

- **罕见病预测**: 对于训练数据稀少的疾病如何做出准确预测
- **零样本学习**: 如何预测从未在训练集中出现的药物-疾病关联
- **异构图建模**: 如何有效整合多类型实体和关系

#### 2.1.2 TxGNN 创新点

| 创新 | 描述 |
|------|------|
| **疾病原型学习** | 利用相似疾病的知识增强罕见病预测 |
| **稀有度加权** | 对低频疾病给予更高的原型聚合权重 |
| **度量学习** | 通过相似性计算实现知识迁移 |
| **两阶段训练** | 预训练 + 微调策略 |

### 2.2 核心算法

#### 2.2.1 异构关系图卷积网络 (HeteroRGCN)

HeteroRGCN 对异构知识图谱进行消息传递：

$$
h_v^{(l+1)} = \sigma \left( W_0^{(l)} h_v^{(l)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_v^r} \frac{1}{|\mathcal{N}_v^r|} W_r^{(l)} h_u^{(l)} \right)
$$

其中：
- $h_v^{(l)}$: 节点 $v$ 在第 $l$ 层的嵌入
- $\mathcal{R}$: 关系类型集合
- $\mathcal{N}_v^r$: 节点 $v$ 在关系 $r$ 下的邻居
- $W_r^{(l)}$: 关系特定的变换矩阵
- $W_0^{(l)}$: 自环（自身信息保留）的变换矩阵
- $\sigma$: 激活函数（如 ReLU）

**实现代码**:

```python
class HeteroRGCNLayer(nn.Module):
    def forward(self, G, feat_dict):
        # 为每种边类型构建消息传递函数
        funcs = {}
        for src_type, etype, dst_type in G.canonical_etypes:
            if G.num_edges((src_type, etype, dst_type)) > 0:
                funcs[(src_type, etype, dst_type)] = (
                    fn.copy_u('h', 'm'),   # 复制源节点特征作为消息
                    fn.mean('m', 'h_agg')  # 聚合消息取均值
                )
        
        # 批量更新所有节点类型
        G.multi_update_all(funcs, 'mean')
        
        # 应用层归一化和激活函数
        return {ntype: self.layer_norm(F.relu(G.nodes[ntype].data['h_agg']))
                for ntype in G.ntypes}
```

#### 2.2.2 DistMult 链接预测

DistMult 是一种简洁高效的知识图谱嵌入方法：

$$
\text{score}(h, r, t) = \langle e_h, W_r, e_t \rangle = \sum_i e_h^{(i)} \cdot W_r^{(i)} \cdot e_t^{(i)}
$$

其中 $e_h, e_t$ 是头尾实体嵌入，$W_r$ 是关系嵌入（对角矩阵形式）。

**训练目标** - 二元交叉熵损失：

$$
\mathcal{L} = -\frac{1}{|E|} \sum_{(h,r,t) \in E} \left[ y \log(\sigma(s)) + (1-y) \log(1-\sigma(s)) \right]
$$

其中 $y \in \{0, 1\}$ 表示正/负样本，$\sigma$ 是 sigmoid 函数。

#### 2.2.3 疾病原型学习 (Disease Prototype Learning)

这是 TxGNN 的核心创新，使模型能够预测罕见病：

```
┌─────────────────────────────────────────────────────────────┐
│                   Disease Prototype Learning                │
└─────────────────────────────────────────────────────────────┘

Step 1: 计算疾病相似度
┌─────────────────────────────────────────────────────────────┐
│  Embedding-based:  sim(d_i, d_j) = cos(e_{d_i}, e_{d_j})    │
│  Profile-based:    sim(d_i, d_j) = cos(p_{d_i}, p_{d_j})    │
│                    其中 p = [#genes, #symptoms, #drugs, ...]│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: 选择 Top-K 相似疾病作为原型
┌─────────────────────────────────────────────────────────────┐
│  对于疾病 d_q, 找到 K 个最相似的疾病:                         │
│  Prototype = {d_1, d_2, ..., d_K} 其中 sim(d_q, d_i) 最高   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: 聚合原型嵌入
┌─────────────────────────────────────────────────────────────┐
│  e_proto = Σ_i softmax(sim_i) × e_{d_i}                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 4: 稀有度加权融合
┌─────────────────────────────────────────────────────────────┐
│  α = exp(-λ × degree(d_q))     # 低度节点权重高              │
│  e_final = (1-α) × e_{d_q} + α × e_proto                    │
└─────────────────────────────────────────────────────────────┘
```

**数学表达**:

$$
e_{d}^{aug} = (1 - \alpha_d) \cdot e_d + \alpha_d \cdot \sum_{k=1}^{K} \frac{\exp(s_{d,k})}{\sum_{j=1}^{K} \exp(s_{d,j})} \cdot e_{p_k}
$$

其中稀有度权重：

$$
\alpha_d = \exp(-\lambda \cdot \text{degree}(d))
$$

其中：
- $s_{d,k} = \text{sim}(d, p_k)$ 是疾病 $d$ 与原型疾病 $p_k$ 的相似度
- $e_{p_k}$ 是第 $k$ 个原型疾病的嵌入向量
- $\lambda = 0.7$ 控制衰减速度，低度数疾病（罕见病）获得更高的 $\alpha$
- 分母 $\sum_{j=1}^{K} \exp(s_{d,j})$ 的求和范围为所有 $K$ 个原型，确保 softmax 正确归一化

#### 2.2.4 负采样策略

```python
class NegativeSampler:
    """
    负边采样策略
    
    对于正边 (h, r, t):
    - Tail corruption: 保持 (h, r)，随机采样 t' ≠ t
    - Head corruption: 保持 (r, t)，随机采样 h' ≠ h
    """
    def sample(self, pos_graph):
        neg_edges = {}
        for etype in pos_graph.canonical_etypes:
            src, dst = pos_graph.edges(etype=etype)
            n_pos = len(src)
            
            # 随机生成负样本尾实体
            neg_dst = torch.randint(
                0, self.num_nodes[dst_type], 
                (n_pos * self.neg_ratio,)
            )
            
            neg_edges[etype] = (src.repeat(self.neg_ratio), neg_dst)
        
        return dgl.heterograph(neg_edges)
```

### 2.3 实现架构

#### 2.3.1 模块组成

```
prediction/
├── __init__.py           # 包导出
├── data_loader.py        # 数据加载与图构建
├── models.py             # GNN 模型定义
├── predictor.py          # 预测器主类
├── evaluator.py          # 评估指标
└── demo.py               # 演示脚本

核心类:
├── MDKGDataLoader            # 数据加载
├── HeteroRGCN                # 图神经网络
│   ├── HeteroRGCNLayer       # 基础 RGCN 层
│   ├── AttentionHeteroRGCNLayer  # 注意力 RGCN 层
│   └── DistMultPredictor     # 链接预测器
├── DrugRepurposingPredictor  # 主预测类
└── LinkPredictor             # 推理封装
```

#### 2.3.2 支持的实体和关系类型

**实体类型** (来自 DPKG_types_Cor4.json):

| 类型 | 英文 | 描述 |
|------|------|------|
| drug | 药物 | 治疗药物 |
| disease | 疾病 | 精神疾病 |
| gene | 基因 | 相关基因 |
| signs | 体征 | 临床体征 |
| symptom | 症状 | 疾病症状 |
| Health_factors | 健康因素 | 风险/保护因素 |
| method | 方法 | 诊断/治疗方法 |
| physiology | 生理 | 生理过程 |
| region | 区域 | 脑区等解剖结构 |

**关系类型**:

| 关系 | 描述 | 示例 |
|------|------|------|
| treatment_for | 治疗 | (quetiapine, treatment_for, schizophrenia) |
| occurs_in | 发生于 | (symptom, occurs_in, disease) |
| located_in | 位于 | (receptor, located_in, brain_region) |
| help_diagnose | 辅助诊断 | (biomarker, help_diagnose, disease) |
| risk_factor_of | 风险因素 | (gene, risk_factor_of, disease) |
| associated_with | 关联 | (symptom, associated_with, disease) |
| characteristic_of | 特征 | (phenotype, characteristic_of, disease) |
| abbreviation_for | 缩写 | (ADHD, abbreviation_for, Attention...) |
| hyponym_of | 下位词 | (bipolar I, hyponym_of, bipolar disorder) |

### 2.4 训练流程

#### 2.4.1 两阶段训练策略

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Pre-training                    │
│  目标: 学习通用的知识图谱嵌入                                 │
│  数据: 所有边类型                                            │
│  Epochs: 50                                                 │
│  Learning Rate: 1e-3                                        │
│  Proto Learning: OFF                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 2: Fine-tuning                     │
│  目标: 优化药物-疾病预测能力                                  │
│  数据: 药物-疾病边 (treatment_for)                           │
│  Epochs: 200                                                │
│  Learning Rate: 5e-4                                        │
│  Proto Learning: ON                                         │
│  Similarity: embedding-based                                │
│  Aggregation: rarity-weighted                               │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.2 训练配置

```python
@dataclass
class TrainingConfig:
    # 模型架构
    n_inp: int = 128          # 输入维度
    n_hid: int = 128          # 隐藏层维度
    n_out: int = 128          # 输出维度
    attention: bool = False   # 是否使用注意力
    proto: bool = True        # 是否使用原型学习
    proto_num: int = 3        # 原型数量
    sim_measure: str = 'embedding'  # 相似度计算方式
    agg_measure: str = 'rarity'     # 聚合方式
    exp_lambda: float = 0.7   # 指数衰减参数
    dropout: float = 0.1      # Dropout 率
    
    # 训练超参数
    pretrain_epochs: int = 50
    finetune_epochs: int = 200
    pretrain_lr: float = 1e-3
    finetune_lr: float = 5e-4
    batch_size: int = 1024
    patience: int = 20        # 早停耐心值
    neg_ratio: int = 1        # 负采样比例
```

#### 2.4.3 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MRR | $\text{MRR} = \frac{1}{\|Q\|}\sum_{q \in Q} \frac{1}{\text{rank}_q}$ | 平均倒数排名 |
| Hits@K | $\text{Hits@K} = \frac{\|\{q: \text{rank}_q \leq K\}\|}{\|Q\|}$ | Top-K 命中率 |
| AUROC | Area Under ROC Curve | ROC 曲线下面积 |
| AUPRC | Area Under PR Curve | Precision-Recall 曲线下面积 |

### 2.5 预测示例

```python
# 初始化预测器
predictor = DrugRepurposingPredictor(
    data_folder="./models/InputsAndOutputs",
    config=TrainingConfig(proto=True, proto_num=3)
)

# 加载数据并训练
predictor.load_data()
predictor.train()

# 预测药物的潜在适应症
results = predictor.predict_repurposing("quetiapine")
# 输出:
# [('mania', 0.892), ('schizophrenia', 0.875), ('bipolar', 0.823), ...]

# 预测疾病的潜在药物
drugs = predictor.predict_drugs_for_disease("depression")
# 输出:
# [('fluoxetine', 0.901), ('sertraline', 0.887), ('escitalopram', 0.865), ...]
```

---

## 第三部分：模块对比与集成

### 3.1 功能对比

| 特性 | Graph RAG | Link Prediction |
|------|-----------|-----------------|
| **主要任务** | 问答生成 | 链接预测 |
| **输入** | 自然语言问题 | 实体 (药物/疾病) |
| **输出** | 结构化答案 | 排序的候选列表 |
| **核心技术** | LLM + KG 验证 | GNN + 度量学习 |
| **可解释性** | 高 (三元组证据) | 中 (嵌入相似度) |
| **罕见病支持** | 依赖 KG 覆盖 | 原型学习增强 |
| **实时性** | 需要 LLM 调用 | 离线嵌入即可 |

### 3.2 协同工作流

两个模块可以协同工作，提供更全面的分析：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户查询入口                              │
│  "What new treatments are being studied for depression?"    │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │                                      │
           ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│     Graph RAG       │              │   Link Prediction   │
│  • 提取医学概念      │              │  • 预测候选药物      │
│  • 验证现有知识      │              │  • 评估治疗可能性    │
│  • 生成解释性答案    │              │  • 排序推荐列表      │
└─────────────────────┘              └─────────────────────┘
           │                                      │
           └──────────────────┬──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       综合响应                               │
│  基于验证知识的答案 + 预测的潜在治疗选项                       │
│  例: "Current treatments include SSRIs (confidence: 0.92)   │
│       Predicted new candidates: Drug X (score: 0.78)..."    │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 技术栈总结

| 组件 | 版本/技术 |
|------|----------|
| Python | 3.10 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| DGL | 2.4.0+cu121 |
| Transformers | 最新版 |
| LLM 后端 | OpenAI / Ollama / HuggingFace |
| 图数据格式 | DGL HeteroGraph |
| 配置管理 | Conda (environment.yaml) |

---

## 📖 参考文献

1. **KGARevion**: Jin, H., et al. (2024). "Knowledge Graph Based Agent for Complex, Knowledge-Intensive QA in Medicine." *arXiv:2410.04660*

2. **GraphRAG**: Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130*

3. **TxGNN**: Huang, K., et al. (2024). "Zero-shot prediction of therapeutic use of drugs with geometric deep learning and clinician centered design." *Nature Medicine*.

4. **DistMult**: Yang, B., et al. (2015). "Embedding Entities and Relations for Learning and Inference in Knowledge Bases." *ICLR*.

5. **RGCN**: Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." *ESWC*.

6. **Leiden Algorithm**: Traag, V., et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports*.

---

## 🔗 相关链接

- KGARevion 论文: https://arxiv.org/abs/2410.04660
- GraphRAG 论文: https://arxiv.org/abs/2404.16130
- TxGNN 论文: https://www.nature.com/articles/s41591-024-03233-x
- DGL 文档: https://docs.dgl.ai/
- PyTorch 文档: https://pytorch.org/docs/

---

*文档版本: 2.0 (Updated with KGARevion paper compliance)*  
*最后更新: 2025-02*  
*MDKG Project*
