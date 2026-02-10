# Mental Disorders Knowledge Graph (MDKG) Construction Pipeline

![MDKG Pipeline](https://github.com/user-attachments/assets/b4b94e2b-cf76-4751-a474-a5f9d9f32529)

## Project Overview

The **MDKG** project is designed to construct a comprehensive Knowledge Graph for Mental Disorders. It employs an end-to-end pipeline that includes:

1.  **Named Entity Recognition (NER) & Relation Extraction (RE)**: Utilizing a customized **SynSpERT** model (based on Spert.PL) to extract clinical entities and relationships from biomedical text.
2.  **Active Learning**: An entropy-based strategy to select the most informative samples for iterative model improvement.
3.  **Entity Linking**: Mapping extracted entities to standard biomedical ontologies (HPO, GO, MONDO, UBERON, UMLS).
4.  **Triplets Refinement**: Leveraging Large Language Models (LLMs) to validate and refine extracted knowledge triplets.

---

## Directory Structure

```plaintext
MDKG/
├── models/
│   ├── SynSpERT/              # Core NER/RE model (training & evaluation)
│   └── InputsAndOutputs/      # Data management directory
│       ├── configs/           # Model configurations (incl. prediction_config.json)
│       ├── input/             # Raw and processed datasets
│       ├── output/            # Training logs, saved models, and results
│       └── pretrained/        # Pretrained BERT/CODER models
├── predictor/                 # GraIL-based link prediction pipeline
│   ├── __main__.py            # (reserved) Package entry point
│   ├── pipeline_task0.py      # Task 0: Data protocol & dataset export
│   ├── pipeline_task1.py      # Task 1: GraIL training & evaluation runner
│   ├── kg_io.py               # KG JSON I/O, triple iteration, entity/relation collection
│   ├── grail_dataset.py       # Export MDKG JSON → GraIL tab-separated format
│   ├── grail_runner.py        # Subprocess wrapper for GraIL train/test scripts
│   ├── candidates.py          # Global candidate pair generator (disease × drug)
│   ├── split.py               # Edge removal validator (train/val split)
│   ├── metrics.py             # Evaluation metrics (AUC, MRR, Hits@K, F1, etc.)
│   └── normalize.py           # Text span normalization utility
├── grail/                     # GraIL reference implementation (modified)
│   ├── train.py               # Model training entry point
│   ├── test_auc.py            # AUC evaluation on test set
│   ├── test_ranking.py        # Ranking evaluation (MRR, Hits@K)
│   ├── model/dgl/             # R-GCN model architecture
│   ├── managers/              # Trainer & evaluator
│   ├── subgraph_extraction/   # Subgraph dataset & graph sampler
│   ├── utils/                 # Graph utilities, data processing
│   ├── data/                  # Datasets (MDKG_v1, FB15K237, WN18RR, etc.)
│   └── experiments/           # Saved experiment checkpoints & logs
├── shell/                     # Automation scripts for all pipelines
├── active_learning.py         # Main script for Active Learning sample selection
├── entity_linking.py          # Main script for Entity Linking
├── extract_entities.py        # Entity extraction utility
├── requirements.txt           # Required libraries
└── triplets_refine_prompt.py  # Triplet extraction and prompt generation
```

---

## Usage Guide

There are two ways to run this project:
1.  **Shell Scripts (Recommended)**: One-click automation for each stage.
2.  **Manual Python Scripts**: Step-by-step execution for debugging or customization.

### Stage 0: Environment Preparation

Before running any pipelines, setup the python environment and download necessary language models (Spacy/NLTK).

**Input**: `requirements.txt`  
**Output**: Installed libraries and models

#### Method 1: Shell Script
```bash
bash shell/setup_env.sh
```

#### Method 2: Manual Execution
```bash
# 1. Install Conda packages (Recommended for FAISS-GPU)
conda install -c pytorch -c nvidia faiss-gpu

# 2. Install Pip libraries
pip install -r requirements.txt

# 3. Download Spacy Models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# 4. Download NLTK Data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

### Stage 1: Data Preparation

This stage downloads the dataset, preprocesses it, and generates augmented data for training.

**Input**: Zenodo Dataset URL (configured in downloader)  
**Output**: `models/InputsAndOutputs/input/*.json`

#### Method 1: Shell Script
```bash
bash shell/data_pipeline.sh
```

#### Method 2: Manual Execution
```bash
# 1. Download Data
python models/SynSpERT/data_downloader.py

# 2. Preprocess & Split Data
cd models/SynSpERT
python generate_input.py

# 3. Generate Augmented Data (Optional)
python generate_augmented_input.py
cd ../..
```

---

### Stage 2: Model Training

Trains the joint NER & RE model using the prepared data.

**Input**: Processed data from Stage 1, Pretrained BERT/CODER model  
**Output**: Trained model weights in `models/InputsAndOutputs/output/save/run_v1/`

#### Method 1: Shell Script
```bash
bash shell/training.sh
```

#### Method 2: Manual Execution
```bash
# 1. Download Pretrained/Base Model
python models/SynSpERT/model_downloader.py

# 2. Run Training
cd models/SynSpERT
python main.py
cd ../..
```

---

### Stage 3: Active Learning

Generates features on the full dataset and selects the most informative samples for potential annotation.

**Input**: Trained Model (`run_v1`), Full Dataset (`md_KG_all_0217_agu.json`)  
**Output**: 
- Feature tensors in `models/InputsAndOutputs/output/log/run_v1/`
- Selected samples in `models/InputsAndOutputs/output/sampling_json_run_v1_sampled.json`

#### Method 1: Shell Script
```bash
bash shell/active_learning.sh
```

#### Method 2: Manual Execution
```bash
# 1. Generate Features (Entropy, Embeddings) on Full Dataset
cd models/SynSpERT
python generate_al_features.py
cd ../..

# 2. Calculate Entropy & Select Samples
python Active_learning.py
```

---

### Stage 4: Triplets Extraction & Refinement

Extracts high-confidence Knowledge Graph triplets from the model predictions and generates prompts for LLM-based verification.

**Input**: `sampling_json_run_v1_sampled.json` (Model predictions)  
**Output**: `models/InputsAndOutputs/output/triplet_refinement/triplets_for_refinement.json`

#### Method 1: Shell Script
```bash
bash shell/triplets_extract.sh
```

#### Method 2: Manual Execution
```bash
python triplets_refine_prompt.py
```

---

### Stage 5: Entity Linking & Ontology Matching

Links extracted entities to standard biomedical ontologies via embedding similarity (SapBERT/CODER).

**Input**: Training Entities, Online OBO Files (HPO, GO, etc.)  
**Output**: `models/InputsAndOutputs/output/entity_linking_results.json`

#### Method 1: Shell Script
```bash
bash shell/entity_linking.sh
```

#### Method 2: Manual Execution
```bash
# 1. Extract unique entities from training corpus
python extract_entities.py

# 2. Download official Ontology files (OBO format)
python models/SynSpERT/ontologies_downloader.py

# 3. Generate embeddings for Ontology terms
python models/SynSpERT/generate_ontology_embeddings.py

# 4. Perform Vector-based Entity Linking
python entity_linking.py
```

---

## GraIL Link Prediction Pipeline (`predictor/`)

### Overview

The `predictor` module implements a GraIL-based inductive link prediction pipeline for drug repositioning. It wraps the [GraIL reference implementation](https://github.com/kkteru/grail/) (ICML 2020) with a structured data processing and evaluation framework.

The pipeline currently implements **Task 0** (data protocol & evaluation pipeline setup) and **Task 1** (GraIL model training & evaluation). Future tasks (candidate ranking, bio-prior integration, etc.) are planned but not yet implemented.

### Architecture & Component Design

```
┌─────────────────────────────────────────────────────────────────┐
│                   predictor/ Module Structure                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │ pipeline_task0.py │      │ pipeline_task1.py │                │
│  │  (Data Protocol)  │      │  (Train & Test)   │                │
│  └────────┬─────────┘      └────────┬──────────┘                │
│           │                         │                           │
│     ┌─────▼─────┐            ┌──────▼───────┐                   │
│     │  kg_io.py  │            │grail_runner.py│                  │
│     │ (JSON I/O) │            │ (subprocess)  │                  │
│     └──┬────┬────┘            └──────┬───────┘                   │
│        │    │                        │                           │
│   ┌────▼┐ ┌▼──────────┐       ┌─────▼──────┐                    │
│   │split│ │grail_     │       │  grail/     │                    │
│   │ .py │ │dataset.py │       │ train.py    │                    │
│   └─────┘ └───────────┘       │ test_auc.py │                    │
│                               │ test_ranking│                    │
│  ┌──────────────┐             └─────────────┘                    │
│  │candidates.py │   ┌────────────┐   ┌─────────────┐            │
│  │(pair gen)    │   │ metrics.py │   │normalize.py │            │
│  └──────────────┘   └────────────┘   └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Component Descriptions

| Component | File | Responsibility |
|-----------|------|----------------|
| **KG I/O** | `kg_io.py` | Reads MDKG JSON files (records with `entities` and `relations` arrays); iterates triples, collects entity IDs, relation types, and statistics; validates required relations exist |
| **Normalize** | `normalize.py` | Lowercases, collapses whitespace, and replaces spaces with underscores in entity spans for consistent node IDs |
| **Split** | `split.py` | `EdgeRemovalValidator` — shuffles target-relation edges and splits them into train/validation sets by a configurable ratio; `remove_edges_from_records` — rewrites the JSON with held-out edges removed |
| **GraIL Dataset** | `grail_dataset.py` | Converts MDKG JSON into GraIL's tab-separated triple format (`head\trelation\ttail`); produces `train.txt`, `valid.txt`, `test.txt` |
| **Candidates** | `candidates.py` | `GlobalCandidateGenerator` — generates all (disease, drug) Cartesian product pairs; labels each pair as positive/negative based on existing `treatment_for` edges |
| **Metrics** | `metrics.py` | Computes AUC-ROC, AUC-PR, Precision, Recall, F1, MRR, and Hits@K (K=1,5,10,20) with optional per-group evaluation |
| **GraIL Runner** | `grail_runner.py` | Thin subprocess wrapper that invokes `train.py`, `test_auc.py`, and `test_ranking.py` from the GraIL repo |
| **Task 0** | `pipeline_task0.py` | Orchestrator for data validation, entity overlap checking, edge splitting, and GraIL dataset export |
| **Task 1** | `pipeline_task1.py` | Orchestrator for GraIL training and optional evaluation (AUC + ranking tests) |

#### Workflow

```
                     MDKG JSON Files
                    (train + test KG)
                          │
                          ▼
               ┌─────────────────────┐
               │  Task 0: Data Setup │
               │                     │
               │  1. validate_kg     │◄── kg_io.validate_kg_format()
               │  2. collect stats   │◄── kg_io.collect_stats()
               │  3. check overlap   │◄── kg_io.collect_entity_ids()
               │  4. split edges     │◄── split.EdgeRemovalValidator
               │  5. export dataset  │◄── grail_dataset.export_grail_dataset()
               └──────────┬──────────┘
                          │
                          ▼
              GraIL Dataset (train/valid/test.txt)
                          │
                          ▼
               ┌─────────────────────┐
               │ Task 1: Train/Test  │
               │                     │
               │  1. train model     │◄── grail_runner.run_grail_train()
               │  2. test AUC        │◄── grail_runner.run_grail_test_auc()
               │  3. test ranking    │◄── grail_runner.run_grail_test_ranking()
               └─────────────────────┘
                          │
                          ▼
               Evaluation Results (AUC, MRR, Hits@K)
```

### Stage 6: Link Prediction — Task 0 (Data Protocol)

Validates MDKG JSON format, checks train/test entity separation, splits target-relation edges for validation, and exports GraIL-compatible triple files.

**Input**: MDKG JSON files (with `entities` and `relations` per record)  
**Output**: GraIL dataset files + summary statistics

#### Usage

```bash
python -m predictor.pipeline_task0 \
  --train models/InputsAndOutputs/input/md_train_KG_0217_agu.json \
  --test  models/InputsAndOutputs/input/md_test_KG_0217_agu.json \
  --out   models/InputsAndOutputs/output/predictor_output \
  --relation treatment_for \
  --head-types drug \
  --tail-types disease \
  --val-ratio 0.2 \
  --seed 42
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--train` | Yes | — | Path to training KG JSON file |
| `--test` | Yes | — | Path to test KG JSON file |
| `--out` | Yes | — | Output directory for all generated files |
| `--relation` | No | `treatment_for` | Target relation type to focus on for prediction |
| `--head-types` | No | `drug` | Comma-separated head entity types (e.g., `drug` or `drug,compound`) |
| `--tail-types` | No | `disease` | Comma-separated tail entity types (e.g., `disease` or `disease,disorder`) |
| `--val-ratio` | No | `0.2` | Fraction of target-relation edges held out for validation (0.0–1.0) |
| `--seed` | No | `42` | Random seed for reproducible edge splitting |

#### Output Files

```
<out>/
├── summary.json                    # Statistics (entity counts, relation types, overlap)
└── grail/
    ├── train.txt                   # Training triples (tab-separated: head\trel\ttail)
    ├── valid.txt                   # Validation triples (held-out target edges)
    ├── test.txt                    # Test triples
    └── train_subgraph.json         # Full training JSON with held-out edges removed
```

---

### Stage 7: Link Prediction — Task 1 (GraIL Training & Evaluation)

Trains a GraIL inductive link prediction model on the exported dataset and optionally runs AUC and ranking evaluations.

**Input**: GraIL-format dataset under `grail/data/<dataset_name>/`  
**Output**: Trained model checkpoint, training log, evaluation metrics

#### Usage

```bash
python -m predictor.pipeline_task1 \
  --grail-repo ./grail \
  --dataset-name MDKG_v1 \
  --experiment-name exp_v3_hop2_stable \
  --run-tests \
  -- --hop 2 --num_epochs 20 --lr 0.0005 --margin 3.0 \
     --loss_reduction mean --edge_dropout 0.2 --batch_size 16 \
     --num_workers 0
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--grail-repo` | Yes | — | Path to the GraIL repository root directory |
| `--dataset-name` | Yes | — | Dataset folder name under `grail/data/` (e.g., `MDKG_v1`) |
| `--experiment-name` | Yes | — | Experiment name; creates folder under `grail/experiments/` for checkpoints and logs |
| `--python` | No | `python` | Python executable path (e.g., `python3`, `conda run -n MDKG python`) |
| `--run-tests` | No | `false` | If set, runs `test_auc.py` and `test_ranking.py` after training |
| `--` (remainder) | No | — | All arguments after `--` are forwarded directly to GraIL's `train.py` |

#### GraIL `train.py` Arguments (passed after `--`)

These是 `grail/train.py` 支持的全部 CLI 参数，通过 `--` 传递给 predictor wrapper。下表列出了源码中的真实默认值与我们推荐的 MDKG 配置：

**Experiment Setup**

| Argument | Short | Default | Recommended | Description |
|----------|-------|---------|-------------|-------------|
| `--experiment_name` | `-e` | `default` | — | 实验名称；在 `experiments/` 下创建同名文件夹保存模型和日志 |
| `--dataset` | `-d` | — | `MDKG_v1` | 数据集目录名（位于 `grail/data/` 下） |
| `--gpu` | | `0` | `0` | 使用哪个 GPU（设备编号） |
| `--disable_cuda` | | `False` | — | 禁用 CUDA，强制 CPU 运算 |
| `--load_model` | | `False` | — | 是否加载已有模型继续训练 |
| `--train_file` | `-tf` | `train` | `train` | 训练三元组文件名（不含 `.txt` 后缀） |
| `--valid_file` | `-vf` | `valid` | `valid` | 验证三元组文件名（不含 `.txt` 后缀） |

**Training Regime**

| Argument | Short | Default | Recommended | Description |
|----------|-------|---------|-------------|-------------|
| `--num_epochs` | `-ne` | `100` | `20` | 训练轮数 |
| `--eval_every` | | `3` | `3` | 每隔多少 epoch 做一次验证 |
| `--eval_every_iter` | | `455` | — | 每隔多少 iteration 做一次验证 |
| `--save_every` | | `10` | `10` | 每隔多少 epoch 保存一次 checkpoint |
| `--early_stop` | | `100` | `100` | 早停耐心值（验证 AUC 连续不提升的轮数） |
| `--optimizer` | | `Adam` | `Adam` | 优化器类型 |
| `--lr` | | `0.01` | `0.0005` | 学习率 |
| `--clip` | | `1000` | `1000` | 梯度裁剪最大范数 |
| `--l2` | | `5e-4` | `0.0` | L2 权重衰减正则化系数 |
| `--margin` | | `10` | `3.0` | MarginRankingLoss 的 margin 值 |
| `--loss_reduction` | | `sum` | `mean` | 损失缩减方式：`mean`（推荐）或 `sum` |
| `--loss_type` | | `margin` | `margin` | 损失函数：`margin`（MarginRankingLoss）或 `bce`（BCEWithLogitsLoss） |
| `--use_scheduler` | | `False` | — | 启用 CosineAnnealing 学习率调度器 |

**Data Processing**

| Argument | Short | Default | Recommended | Description |
|----------|-------|---------|-------------|-------------|
| `--max_links` | | `1000000` | — | 最大训练链接数（内存不足时可降低） |
| `--hop` | | `3` | `2` | 封闭子图跳数。越大上下文越丰富但越慢，推荐 `2` |
| `--max_nodes_per_hop` | `-max_h` | `None` | — | 每跳最大采样节点数（`None` = 不限制） |
| `--use_kge_embeddings` | `-kge` | `False` | — | 是否使用预训练 KGE 嵌入 |
| `--kge_model` | | `TransE` | — | 预训练 KGE 模型类型（仅当 `-kge True` 时生效） |
| `--model_type` | `-m` | `dgl` | `dgl` | 子图存储格式：`ssp` 或 `dgl` |
| `--constrained_neg_prob` | `-cn` | `0.0` | `0.0` | 约束负采样概率（采样同类型头/尾的概率） |
| `--batch_size` | | `16` | `16` | 训练批大小 |
| `--num_neg_samples_per_link` | `-neg` | `1` | `1` | 每条正样本对应的负样本数 |
| `--num_workers` | | `8` | **`0`** | DataLoader 工作进程数 |
| `--add_traspose_rels` | `-tr` | `False` | `False` | 是否添加逆关系（关系数翻倍） |
| `--enclosing_sub_graph` | `-en` | `True` | `True` | 是否仅使用封闭子图 |

> **⚠️ 重要：`--num_workers` 必须设置为 `0`！**  
> 默认值为 `8`，但在 CUDA 环境下使用多进程 DataLoader 会导致 **CUDA 初始化错误**（`RuntimeError: Cannot re-initialize CUDA in forked subprocess`）。这是因为 PyTorch 的 fork-based 多进程与 CUDA 上下文不兼容。**务必在所有训练和测试命令中传入 `--num_workers 0`**。

**Model Architecture**

| Argument | Short | Default | Recommended | Description |
|----------|-------|---------|-------------|-------------|
| `--rel_emb_dim` | `-r_dim` | `32` | `32` | 关系嵌入维度 |
| `--attn_rel_emb_dim` | `-ar_dim` | `32` | `32` | 注意力关系嵌入维度 |
| `--emb_dim` | `-dim` | `32` | `32` | 实体嵌入维度 |
| `--num_gcn_layers` | `-l` | `3` | `3` | R-GCN 消息传播层数 |
| `--num_bases` | `-b` | `4` | `4` | R-GCN 权重分解的基函数数量 |
| `--dropout` | | `0` | `0.2` | GNN 层节点特征 Dropout 率 |
| `--edge_dropout` | | `0.5` | `0.2` | 子图边 Dropout 率（训练时启用，评估时关闭） |
| `--gnn_agg_type` | `-a` | `sum` | `sum` | GNN 聚合类型：`sum`、`mlp` 或 `gru` |
| `--add_ht_emb` | `-ht` | `True` | `True` | 是否拼接头/尾节点嵌入到图池化表示 |
| `--has_attn` | `-attn` | `True` | `True` | 是否在模型中使用注意力机制 |

#### Output Files

```
grail/experiments/<experiment_name>/
├── best_graph_classifier.pth     # Best model checkpoint (by validation AUC)
├── graph_classifier_chk.pth      # Latest model checkpoint
├── log_train.txt                 # Training log (loss, AUC, AUC-PR per epoch)
└── params.json                   # Hyperparameters used
```

#### GraIL `test_auc.py` Arguments

用于评估模型在测试集上的 AUC-ROC 和 AUC-PR：

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--experiment_name` | `-e` | `default` | 实验名称（从对应目录加载 checkpoint） |
| `--dataset` | `-d` | `Toy` | 数据集目录名 |
| `--train_file` | `-tf` | `train` | 训练三元组文件名 |
| `--test_file` | `-t` | `test` | 测试三元组文件名 |
| `--runs` | | `1` | 重复运行次数（用于计算均值和标准差） |
| `--gpu` | | `0` | GPU 设备编号 |
| `--disable_cuda` | | `False` | 禁用 CUDA |
| `--max_links` | | `100000` | 最大测试链接数 |
| `--hop` | | `3` | 封闭子图跳数（需与训练时一致） |
| `--max_nodes_per_hop` | `-max_h` | `None` | 每跳最大节点数 |
| `--use_kge_embeddings` | `-kge` | `False` | 是否使用预训练 KGE 嵌入 |
| `--kge_model` | | `TransE` | KGE 模型类型 |
| `--model_type` | `-m` | `dgl` | 子图存储格式 |
| `--constrained_neg_prob` | `-cn` | `0` | 约束负采样概率 |
| `--num_neg_samples_per_link` | `-neg` | `1` | 每条正样本的负样本数 |
| `--batch_size` | | `16` | 批大小 |
| `--num_workers` | | `8` | DataLoader 工作进程数（**必须设为 `0`**） |
| `--add_traspose_rels` | `-tr` | `False` | 是否添加逆关系 |
| `--enclosing_sub_graph` | `-en` | `True` | 是否仅使用封闭子图 |

#### GraIL `test_ranking.py` Arguments

用于评估 MRR 和 Hits@K 排名指标：

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--experiment_name` | `-e` | `fb_v2_margin_loss` | 实验名称 |
| `--dataset` | `-d` | `FB237_v2` | 数据集目录名 |
| `--mode` | `-m` | `sample` | 负采样模式：`sample`（随机采样）、`all`（全实体枚举）、`ruleN`（规则预测） |
| `--use_kge_embeddings` | `-kge` | `False` | 是否使用预训练 KGE 嵌入 |
| `--kge_model` | | `TransE` | KGE 模型类型 |
| `--enclosing_sub_graph` | `-en` | `True` | 是否仅使用封闭子图 |
| `--hop` | | `3` | 封闭子图跳数（需与训练时一致） |
| `--add_traspose_rels` | `-tr` | `False` | 是否添加逆关系 |
| `--num_samples` | | `50` | `sample` 模式下每条测试链接的负样本数 |
| `--sequential` | | `False` | 使用顺序处理代替多进程（WSL/低内存环境更安全） |
| `--num_workers` | | `None` | 多进程 worker 数（`None` = 使用全部 CPU 核心） |

> **⚠️ `--mode all` 注意事项**：`all` 模式会对每条测试三元组枚举全部实体（MDKG_v1: 80×5104×2 = 816K 子图），内存消耗巨大。在 WSL 环境下**必须加 `--sequential`** 或限制 `--num_workers` 以避免 OOM 导致系统崩溃。

#### Running GraIL Directly (without predictor wrapper)

For finer control, you can invoke GraIL scripts directly:

```bash
# Activate environment
conda activate MDKG

# Train (⚠️ --num_workers 0 是必须的)
cd grail
GRAIL_LMDB_MAP_SIZE_MB=512 python train.py \
  -d MDKG_v1 -e exp_v3_hop2_stable \
  --hop 2 --num_epochs 20 --lr 0.0005 --margin 3.0 \
  --loss_reduction mean --edge_dropout 0.2 --batch_size 16 \
  --num_workers 0

# Evaluate AUC (⚠️ --num_workers 0 是必须的)
GRAIL_LMDB_MAP_SIZE_MB=512 python test_auc.py \
  -d MDKG_v1 -e exp_v3_hop2_stable --hop 2 \
  --num_workers 0

# Evaluate Ranking — sample mode（快速，WSL 安全）
GRAIL_LMDB_MAP_SIZE_MB=512 python test_ranking.py \
  -d MDKG_v1 -e exp_v3_hop2_stable --hop 2 --mode sample

# Evaluate Ranking — all mode（完整，需加 --sequential）
GRAIL_LMDB_MAP_SIZE_MB=512 python test_ranking.py \
  -d MDKG_v1 -e exp_v3_hop2_stable --hop 2 --mode all --sequential
```

> **⚠️ 关键注意事项**：
> - **`--num_workers 0` 是必须的**：`train.py` 和 `test_auc.py` 的默认值为 `8`，但在 CUDA 环境下多进程 DataLoader 会触发 `RuntimeError: Cannot re-initialize CUDA in forked subprocess`。务必显式传入 `--num_workers 0`。
> - 设置 `GRAIL_LMDB_MAP_SIZE_MB=512`（或更高）以避免 LMDB map-full 错误。
> - Ranking 测试中，`--mode sample`（50 negatives/triple）速度快且 WSL 安全；`--mode all` 枚举所有实体，需要大量内存，在 WSL 下**必须加 `--sequential`**。

---