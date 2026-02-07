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

**参考论文**: 
- [TxGNN: Zero-shot prediction of therapeutic use of drugs with geometric deep learning](https://www.nature.com/articles/s41591-023-02233-x) (Nature Medicine, 2024)
- [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) (NeurIPS 2019)

#### 2.1.1 问题定义

药物重定位 (Drug Repurposing) 面临的核心挑战：

- **罕见病预测**: 对于训练数据稀少的疾病如何做出准确预测
- **零样本学习**: 如何预测从未在训练集中出现的药物-疾病关联
- **异构图建模**: 如何有效整合多类型实体和关系
- **预测可解释性**: 如何解释模型的预测结果，提供临床可信的证据

#### 2.1.2 TxGNN 创新点

| 创新 | 描述 |
|------|------|
| **疾病原型学习** | 利用相似疾病的知识增强罕见病预测 |
| **稀有度加权** | 对低频疾病给予更高的原型聚合权重 (λ=0.7) |
| **度量学习** | 通过相似性计算实现知识迁移 |
| **两阶段训练** | 预训练 + 微调策略 |
| **多种相似度度量** | embedding, profile, BERT, profile+embedding |

#### 2.1.3 本模块改进

基于 TxGNN 论文，我们在 MDKG 项目中进行了以下改进：

| 改进 | 描述 |
|------|------|
| **GNNExplainer 集成** | 基于 arxiv:1903.03894 实现预测可解释性 |
| **Mini-batch 训练** | 支持大规模知识图谱的可扩展训练 |
| **邻居采样** | 使用 DGL NeighborSampler 降低内存消耗 |
| **稀疏消息传递** | 优化大图上的 GNN 计算效率 |
| **硬负采样 (Hard Negative Sampling)** | 针对 "Risk Factor" vs "Treatment" 的混淆问题，训练时强制将风险/关联关系作为负样本，迫使模型区分不同语义 |
| **推理过滤 (Inference Filtering)** | 预测阶段实时检查 KG，自动过滤已知为禁忌或副作用的药物 |
| **可学习温度参数 (Learnable Temperature)** | DistMult 评分引入可学习标量 $\tau$，替代硬编码 $1/\sqrt{d}$ 缩放，保证训练-推理评分一致性 |
| **非药物实体过滤 (Pharmaceutical Filter)** | 推理阶段排除被 NER 错误标注为 "drug" 的代谢物、神经递质、信号分子等非治疗性实体 |
| **关系权重暖启动 (Warm-Start)** | 微调时缩放预训练权重（×0.1 + noise）而非全量重初始化，适应 MDKG 小数据量场景 |

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

**3层残差架构**（对齐TxGNN论文）:

```python
class HeteroRGCN(nn.Module):
    def encode(self, G):
        h = self.get_node_embeddings(G)
        
        # Layer 1
        h1 = self.layer1(G, h)
        h1 = {k: self.dropout(v) for k, v in h1.items()}
        
        # Layer 2 with residual connection
        h2 = self.layer2(G, h1)
        h2 = {k: self.dropout(h2[k]) + h1[k] for k in h2}  # Residual
        
        # Layer 3 with residual connection
        h3 = self.layer3(G, h2)
        h3 = {k: h3[k] + self.residual_proj[k](h2[k]) for k in h3}
        
        return h3
```

#### 2.2.2 DistMult 链接预测

DistMult 是一种简洁高效的知识图谱嵌入方法：

$$
\text{score}(h, r, t) = \frac{\langle e_h, W_r, e_t \rangle}{\tau} = \frac{\sum_i e_h^{(i)} \cdot W_r^{(i)} \cdot e_t^{(i)}}{\tau}
$$

其中 $\tau$ 是**可学习温度参数**，初始化为 $\sqrt{d}$（$d$ 为嵌入维度），在训练和推理中同时使用以保证评分一致性。

**训练目标** - 简单 BCE Loss（对齐 TxGNN）:

$$
\mathcal{L} = -\frac{1}{|E|} \sum_{(h,r,t)} \left[ y \log(\sigma(s/\tau)) + (1-y) \log(1-\sigma(s/\tau)) \right]
$$

> **设计说明**: 
> - TxGNN 原始论文仅使用 BCE loss，不含 Margin Ranking Loss
> - 可学习温度 $\tau$ 替代了之前硬编码的 $1/\sqrt{d}$ 缩放，避免训练-推理不一致
> - $\tau$ 初始值 $\sqrt{d}$ 等效于 $1/\sqrt{d}$ 缩放，但作为可训练参数自适应调整

**微调阶段关系权重暖启动**:

$$
W_r^{finetune} = 0.1 \cdot W_r^{pretrain} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.01^2)
$$

> 预训练将所有边视为正样本，导致关系权重偏大。暖启动保留结构信息但缩小量级。
> MDKG 仅有 ~40 条 `treatment_for` 边，不足以全量重初始化后重新学习（TxGNN 有 2.04M 条边）。

**非药物实体过滤**:

推理阶段对 "drug" 类型实体进行药物学过滤，排除被 NER 模型错误标注为药物的代谢物（glucose）、神经递质（serotonin）、信号分子（cAMP）、违禁品（MDMA）等非治疗性实体。

#### 2.2.3 疾病原型学习 (Disease Prototype Learning)

这是 TxGNN 的核心创新，使模型能够预测罕见病：

```
┌─────────────────────────────────────────────────────────────┐
│                   Disease Prototype Learning                │
└─────────────────────────────────────────────────────────────┘

Step 1: 计算疾病相似度 (sim_measure 选项)
┌─────────────────────────────────────────────────────────────┐
│  embedding:    sim(d_i, d_j) = cos(e_{d_i}, e_{d_j})        │
│  profile:      sim(d_i, d_j) = cos(p_{d_i}, p_{d_j})        │
│  bert:         sim(d_i, d_j) = cos(BERT(d_i), BERT(d_j))    │
│  profile+embedding: 结合两种方式                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: 选择 Top-K 相似疾病作为原型
┌─────────────────────────────────────────────────────────────┐
│  对于疾病 d_q, 找到 K 个最相似的疾病:                         │
│  Prototype = {d_1, d_2, ..., d_K}                           │
│  其中 sim(d_q, d_i) 最高且 d_i ≠ d_q                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: 聚合原型嵌入 (agg_measure 选项)
┌─────────────────────────────────────────────────────────────┐
│  rarity:  α = exp(-λ × degree(d_q)), λ=0.7 (TxGNN default)  │
│  avg:     α = 0.5                                           │
│  learn:   α = sigmoid(W_gate × [e_q; e_proto])              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 4: 最终嵌入
┌─────────────────────────────────────────────────────────────┐
│  e_final = (1-α) × e_{d_q} + α × Σ softmax(sim_i) × e_{d_i} │
└─────────────────────────────────────────────────────────────┘
```

**数学表达** (论文 Eq. 3):

$$
e_{d}^{aug} = (1 - \alpha_d) \cdot e_d + \alpha_d \cdot \sum_{k=1}^{K} \frac{\exp(s_{d,k})}{\sum_{j=1}^{K} \exp(s_{d,j})} \cdot e_{p_k}
$$

其中稀有度权重：

$$
\alpha_d = \exp(-\lambda \cdot \text{degree}(d)), \quad \lambda = 0.7
$$

### 2.3 GNNExplainer 可解释性模块

#### 2.3.1 算法原理

基于论文 [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) (NeurIPS 2019)。

GNNExplainer 通过优化边掩码来识别对预测最重要的子图：

$$
\max_{M} MI(Y, (G_S, X_S)) = H(Y) - H(Y | G = G_S, X = X_S)
$$

其中：
- $M \in [0,1]^{|E|}$: 边掩码（每条边的重要性权重）
- $G_S$: 重要子图
- $X_S$: 重要节点特征

**损失函数**:

$$
\mathcal{L} = -\log P_{\Phi}(Y | G \odot \sigma(M)) + \lambda_1 \|M\|_1 + \lambda_2 H(M)
$$

其中：
- 第一项：预测损失（最大化掩码后的预测分数）
- $\lambda_1 \|M\|_1$: 稀疏性正则化（鼓励小子图）
- $\lambda_2 H(M)$: 熵正则化（鼓励二值化掩码）

#### 2.3.2 实现架构

```
prediction/explainer.py
├── ExplanationResult      # 解释结果数据类
│   ├── edge_mask          # 边重要性掩码
│   ├── edge_importance    # 边重要性排序
│   ├── prediction_score   # 原始预测分数
│   ├── fidelity           # 解释保真度
│   ├── pathways           # 重要路径列表
│   └── metadata           # 元数据
│
└── GNNExplainer           # 解释器主类
    ├── _initialize_masks()    # 初始化可学习掩码
    ├── _extract_subgraph()    # 提取计算子图
    ├── _masked_forward()      # 带掩码的前向传播
    ├── _loss()                # 计算解释损失
    ├── explain()              # 生成解释
    └── extract_important_pathways()  # 提取重要路径
```

#### 2.3.3 使用示例

```python
from prediction import DrugRepurposingPredictor

# 初始化并训练模型
predictor = DrugRepurposingPredictor()
predictor.load_data()
predictor.train()

# 生成预测解释
explanation = predictor.explain_prediction(
    drug_name="metformin",
    disease_name="type 2 diabetes",
    num_hops=2,
    epochs=100
)

# 查看结果
print(f"预测分数: {explanation.prediction_score:.4f}")
print(f"解释保真度: {explanation.fidelity:.4f}")
print(f"重要边数量: {len([e for e in explanation.edge_importance if e[2] > 0.1])}")

# 重要路径
for pathway in explanation.pathways[:3]:
    print(f"路径: {pathway}")
```

#### 2.3.4 解释输出格式

```python
ExplanationResult:
  edge_mask: Tensor[num_edges]           # 每条边的重要性 [0,1]
  edge_importance: List[Tuple[           # 排序的边重要性
    src_type, dst_type, importance, relation, src_idx, dst_idx
  ]]
  prediction_score: float                # 原始预测分数
  masked_score: float                    # 掩码后预测分数
  fidelity: float                        # |orig - masked| / orig
  pathways: List[str]                    # 重要路径描述
  metadata: Dict                         # 包含drug_name, disease_name等
```

### 2.4 可扩展性优化

#### 2.4.1 Mini-batch 训练

对于大规模知识图谱（>100万边），使用 DGL 的邻居采样实现 mini-batch 训练：

```
时间复杂度比较：
┌─────────────────────────────────────────────────────────────┐
│  Full-batch:    O(|V| × |E| × L)     每epoch处理整个图       │
│  Mini-batch:    O(B × F^L × L)       每batch只处理采样子图   │
└─────────────────────────────────────────────────────────────┘

其中:
  |V|: 节点数
  |E|: 边数  
  L: GNN层数
  B: batch size
  F: fanout (每层采样的邻居数)
```

**实现**:

```python
class MiniBatchEdgeSampler:
    def __init__(self, G, fanouts=[15, 10, 5], batch_size=1024):
        self.sampler = NeighborSampler(fanouts)
        
    def create_dataloader(self, edge_dict, negative_sampler=None):
        sampler = as_edge_prediction_sampler(
            self.sampler,
            negative_sampler=negative_sampler
        )
        return DGLDataLoader(
            self.G, edge_dict, sampler,
            batch_size=self.batch_size,
            shuffle=True
        )
```

#### 2.4.2 稀疏消息传递

使用 DGL 的稀疏操作优化内存使用：

```python
class SparseMessagePassing(nn.Module):
    def forward(self, block, feat):
        with block.local_scope():
            block.srcdata['h'] = self.linear(feat)
            block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            return block.dstdata['h']
```

#### 2.4.3 疾病Profile缓存

预计算并缓存疾病profile以加速相似度计算：

```python
def compute_disease_profiles_batch(G, disease_indices, cache=None):
    """批量计算疾病profile，支持缓存"""
    cache = cache or {}
    for idx in disease_indices:
        if idx not in cache:
            cache[idx] = compute_disease_profile(G, idx)
    return cache
```

### 2.5 实现架构

#### 2.5.1 模块组成

```
prediction/
├── __init__.py           # 包导出
├── data_loader.py        # 数据加载与图构建
├── models.py             # GNN 模型定义
│   ├── HeteroRGCN            # 3层残差RGCN
│   ├── HeteroRGCNLayer       # 基础RGCN层
│   ├── AttentionHeteroRGCNLayer  # 注意力RGCN层
│   ├── DistMultPredictor     # 链接预测+原型学习
│   ├── MiniBatchEdgeSampler  # Mini-batch采样器
│   └── SparseMessagePassing  # 稀疏消息传递
├── predictor.py          # 预测器主类
│   ├── DrugRepurposingPredictor  # 主预测类
│   ├── TrainingConfig           # 训练配置
│   └── NegativeSampler          # 负采样器
├── explainer.py          # GNNExplainer模块 [NEW]
│   ├── ExplanationResult        # 解释结果
│   └── GNNExplainer             # 解释器
├── evaluator.py          # 评估指标
└── demo.py               # 演示脚本
```

#### 2.5.2 训练配置

```python
@dataclass
class TrainingConfig:
    # 模型架构
    n_inp: int = 256          # 输入维度
    n_hid: int = 256          # 隐藏层维度
    n_out: int = 256          # 输出维度
    attention: bool = True    # 是否使用注意力
    proto: bool = True        # 是否使用原型学习
    proto_num: int = 5        # 原型数量 (K)
    sim_measure: str = 'embedding'  # 相似度: embedding/profile/bert
    agg_measure: str = 'rarity'     # 聚合: rarity/avg/learn
    exp_lambda: float = 0.7   # 指数衰减参数 (TxGNN默认)
    dropout: float = 0.2      # Dropout 率
    
    # 训练超参数
    pretrain_epochs: int = 100
    finetune_epochs: int = 300
    pretrain_lr: float = 5e-4
    finetune_lr: float = 1e-4
    batch_size: int = 1024
    weight_decay: float = 1e-5
    patience: int = 30
    neg_ratio: int = 5        # 负采样比例
    
    # Mini-batch (可扩展性)
    use_mini_batch: bool = False
    fanouts: List[int] = [15, 10, 5]
    
    # 可解释性
    enable_explainer: bool = True
    explainer_epochs: int = 100
    explainer_lr: float = 0.01
```

### 2.6 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MRR | $\text{MRR} = \frac{1}{\|Q\|}\sum_{q \in Q} \frac{1}{\text{rank}_q}$ | 平均倒数排名 |
| Hits@K | $\text{Hits@K} = \frac{\|\{q: \text{rank}_q \leq K\}\|}{\|Q\|}$ | Top-K 命中率 |
| AUROC | Area Under ROC Curve | ROC 曲线下面积 |
| AUPRC | Area Under PR Curve | Precision-Recall 曲线下面积 |
| Fidelity | $\frac{|s_{orig} - s_{masked}|}{s_{orig}}$ | 解释保真度 (用于GNNExplainer) |

### 2.7 使用示例

#### 2.7.1 基础预测

```python
from prediction import DrugRepurposingPredictor, TrainingConfig

# 自定义配置
config = TrainingConfig(
    proto=True,
    proto_num=5,
    sim_measure='embedding',
    agg_measure='rarity',
    exp_lambda=0.7
)

# 初始化预测器
predictor = DrugRepurposingPredictor(
    data_folder="./models/InputsAndOutputs",
    config=config
)

# 加载数据并训练
predictor.load_data()
predictor.train()

# 预测药物的潜在适应症
results = predictor.predict_repurposing("quetiapine", top_k=10)
for disease, score, onto in results:
    print(f"{disease}: {score:.4f}")

# 预测疾病的潜在药物
drugs = predictor.predict_treatments("depression", top_k=10)
```

#### 2.7.2 带解释的预测

```python
# 生成预测并解释
explanation = predictor.explain_prediction(
    drug_name="quetiapine",
    disease_name="bipolar disorder",
    num_hops=2
)

print(f"预测分数: {explanation.prediction_score:.4f}")
print(f"重要路径:")
for path in explanation.pathways[:5]:
    print(f"  {path}")
```

#### 2.7.3 批量解释

```python
# 对多个药物-疾病对生成解释
pairs = [
    ("metformin", "type 2 diabetes"),
    ("quetiapine", "schizophrenia"),
    ("fluoxetine", "depression")
]

explanations = predictor.explain_batch(pairs)
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

3. **TxGNN**: Huang, K., et al. (2024). "Zero-shot prediction of therapeutic use of drugs with geometric deep learning and clinician centered design." *Nature Medicine*. https://www.nature.com/articles/s41591-023-02233-x

4. **GNNExplainer**: Ying, R., et al. (2019). "GNNExplainer: Generating Explanations for Graph Neural Networks." *NeurIPS*. https://arxiv.org/abs/1903.03894

5. **DistMult**: Yang, B., et al. (2015). "Embedding Entities and Relations for Learning and Inference in Knowledge Bases." *ICLR*.

6. **RGCN**: Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." *ESWC*.

7. **Leiden Algorithm**: Traag, V., et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports*.

---

## 🔗 相关链接

- KGARevion 论文: https://arxiv.org/abs/2410.04660
- GraphRAG 论文: https://arxiv.org/abs/2404.16130
- TxGNN 论文: https://www.nature.com/articles/s41591-023-02233-x
- GNNExplainer 论文: https://arxiv.org/abs/1903.03894
- DGL 文档: https://docs.dgl.ai/
- PyTorch 文档: https://pytorch.org/docs/

---

*文档版本: 3.0 (Updated with GNNExplainer and Scalability Optimizations)*  
*最后更新: 2025-02*  
*MDKG Project*
