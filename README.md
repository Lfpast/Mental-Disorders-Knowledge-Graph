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
│   ├── SynSpERT/              # Core model training and evaluation code
│   └── InputsAndOutputs/      # Data management directory
│       ├── configs/           # Model configurations (incl. prediction_config.json)
│       ├── input/             # Raw and processed datasets
│       ├── output/            # Training logs, saved models, and results
│       │   └── prediction/    # Drug repurposing model outputs (NEW)
│       └── pretrained/        # Pretrained BERT/CODER models
├── prediction/                # Drug Repurposing Prediction Module (NEW)
│   ├── __init__.py            # Module exports
│   ├── data_loader.py         # MDKG data loading and graph construction
│   ├── models.py              # HeteroRGCN and link prediction models
│   ├── predictor.py           # Main DrugRepurposingPredictor class
│   ├── evaluator.py           # Evaluation metrics (MRR, Hits@K, etc.)
│   └── demo.py                # Interactive demonstration script
├── GraphRAG/                  # Graph RAG Module (KGARevion-based)
│   ├── kgarevion_agent.py     # Core KGARevion implementation
│   └── graph_rag_demo.py      # Interactive demo script
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

### Stage 6: Graph RAG for Mental Health Q&A (NEW)

This stage enables **Knowledge Graph-enhanced Retrieval-Augmented Generation (Graph RAG)** for answering mental health questions. The implementation strictly follows the [KGARevion paper](https://arxiv.org/abs/2410.04660).

**Input**: Entity Linking Results, Extracted Triplets, LLM API  
**Output**: Accurate, knowledge-grounded answers to mental health questions

#### How Graph RAG Works

The system operates through four key actions (following KGARevion design):

1. **Generate**: Extract medical concepts from the query and generate relevant knowledge triplets using the LLM
2. **Review**: Verify generated triplets against the MDKG knowledge graph (True/False/Incomplete classification)
3. **Revise**: Iteratively correct rejected triplets (up to k rounds, default k=2)
4. **Answer**: Generate the final answer based on verified, grounded knowledge

**Key Features**:
- **True/False Binary Classification**: Following paper specification, not confidence scores
- **Soft Constraint Rule**: Handles incomplete KG by keeping unmappable triplets
- **Community Detection** (Optional): Leiden/Louvain algorithm for efficient triplet search

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Generate  │ -> │    Review    │ -> │   Revise    │ -> │    Answer    │
│  (Triplets) │    │(TRUE/FALSE/  │    │  (Correct)  │    │  (Response)  │
│             │    │ INCOMPLETE)  │    │             │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

#### Method 1: Shell Script
```bash
# Interactive Q&A mode
bash shell/graph_rag.sh -i

# Demo mode with sample questions
bash shell/graph_rag.sh -d -v

# Single query with community detection
bash shell/graph_rag.sh -q "What are the symptoms of depression?" -c

# Use local Ollama instead of OpenAI
bash shell/graph_rag.sh --llm ollama --model llama3 -i
```

#### Method 2: Python Script
```bash
# Interactive mode
python GraphRAG/graph_rag_demo.py --interactive

# Demo mode with verbose output
python GraphRAG/graph_rag_demo.py --demo --verbose

# Single query with OpenAI
export OPENAI_API_KEY="your-api-key"
python GraphRAG/graph_rag_demo.py -q "How does lithium treat bipolar disorder?"

# With community detection optimization
python GraphRAG/graph_rag_demo.py --community -i
```

#### Method 3: Python API
```python
from GraphRAG.kgarevion_agent import KGARevionAgent, KnowledgeGraphManager
from GraphRAG.kgarevion_agent import OpenAIBackend, OllamaBackend

# Initialize LLM backend
llm = OpenAIBackend(api_key="your-api-key", model="gpt-4")
# Or use local Ollama:
# llm = OllamaBackend(model="llama3")

# Initialize Knowledge Graph Manager
config = {
    'entity_linking_path': 'models/InputsAndOutputs/output/entity_linking_results.json',
    'predictions_path': 'models/InputsAndOutputs/output/sampling_json_run_v1_sampled.json'
}
kg_manager = KnowledgeGraphManager(config)

# Create KGARevion Agent
agent = KGARevionAgent(
    kg_manager=kg_manager,
    llm_backend=llm,
    use_community_detection=True  # Enable Leiden/Louvain optimization
)

# Query
result = agent.query("What are the risk factors for schizophrenia?")
print(result.answer)
print(f"True triplets: {len([t for t in result.verified_triplets if t.status.value == 'true'])}")
print(f"Incomplete triplets: {len([t for t in result.verified_triplets if t.status.value == 'incomplete'])}")
```

#### Supported LLM Backends

| Backend | Description | Setup |
|---------|-------------|-------|
| OpenAI | GPT-4, GPT-3.5-turbo | Set `OPENAI_API_KEY` env variable |
| Ollama | Local LLMs (Llama3, Mistral) | Install [Ollama](https://ollama.ai) and run `ollama pull llama3` |

---

### Stage 7: Drug Repurposing Prediction (NEW)

This stage enables **Drug-Disease Link Prediction** for drug repurposing in mental health disorders. The implementation is inspired by [TxGNN](https://www.nature.com/articles/s41591-024-03233-x) (Nature Medicine 2024).

**Input**: MDKG Knowledge Graph (triplets + entity linking)  
**Output**: Ranked drug-disease predictions for drug repurposing

#### How Drug Repurposing Prediction Works

The system implements a Graph Neural Network (GNN) approach with metric learning:

1. **Knowledge Graph Construction**: Build heterogeneous graph from MDKG triplets
2. **Pre-training**: Train GNN on all edge types for link prediction
3. **Fine-tuning**: Specialize on drug-disease relations with prototype learning
4. **Zero-shot Prediction**: Leverage disease similarity for novel predictions

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Load MDKG      │ -> │  Pre-train GNN  │ -> │  Fine-tune on   │ -> │  Predict New    │
│  Knowledge Graph│    │  (All edges)    │    │  Drug-Disease   │    │  Indications    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
        v                      v                      v                      v
   Build DGL graph        Learn node           Metric learning        Rank drugs for
   from triplets          embeddings           + prototypes           diseases
```

#### Key Features

- **Zero-shot Prediction**: Predict treatments for diseases with no training data
- **Disease Similarity**: Similar diseases share treatment patterns
- **Prototype Learning**: Aggregate knowledge from similar entities
- **Multi-relation**: Supports indication, contraindication, and off-label use

#### Method 1: Shell Script
```bash
# Train the model
bash shell/prediction.sh train

# Quick training (for testing)
bash shell/prediction.sh quick

# Predict new indications for a drug
bash shell/prediction.sh predict quetiapine

# Predict treatments for a disease
bash shell/prediction.sh treatments depression

# Evaluate model performance
bash shell/prediction.sh evaluate

# Interactive mode
bash shell/prediction.sh interactive
```

#### Method 2: Python Script
```bash
# Interactive demo
python -m prediction.demo

# Train and demo
python -m prediction.demo --train --demo

# Quick mode for testing
python -m prediction.demo --quick --demo

# Predict for specific drug
python -m prediction.demo --predict aripiprazole
```

#### Method 3: Python API
```python
from prediction import DrugRepurposingPredictor, DrugRepurposingEvaluator

# Initialize predictor
predictor = DrugRepurposingPredictor(
    data_folder="./models/InputsAndOutputs"
)

# Load data and train
predictor.load_data(split='random', seed=42)
predictor.train()

# Drug repurposing: Find new indications for a drug
results = predictor.predict_repurposing("quetiapine", top_k=10)
for disease, score, onto_info in results:
    print(f"{disease}: {score:.4f}")

# Treatment prediction: Find drugs for a disease  
treatments = predictor.predict_treatments("depression", top_k=10)
for drug, score, onto_info in treatments:
    print(f"{drug}: {score:.4f}")

# Get specific drug-disease score
score = predictor.predict_drug_disease("lithium", "bipolar")
print(f"Lithium for Bipolar: {score:.4f}")

# Evaluate model
evaluator = DrugRepurposingEvaluator(predictor)
results = evaluator.full_evaluation()
print(results['disease_centric'])
```

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MRR | Mean Reciprocal Rank of true predictions |
| Hits@K | Proportion of true predictions in top K |
| AUROC | Area Under ROC Curve (classification) |
| AUPRC | Area Under Precision-Recall Curve |

#### Model Architecture

The prediction module uses **HeteroRGCN** (Heterogeneous Relational Graph Convolutional Network):

- **Node Embeddings**: Learnable embeddings for drugs, diseases, genes, symptoms
- **Message Passing**: Relation-specific weight matrices for each edge type
- **Link Prediction**: DistMult scoring with relation embeddings
- **Metric Learning**: Prototype-based augmentation for rare diseases

---

## Requirements

- **OS**: Linux (Recommended)
- **Python**: 3.8+
- **GPU**: CUDA-capable GPU recommended for Training and BERT embedding generation.

### Key Python Packages
- `torch`, `transformers` (HuggingFace)
- `dgl` (Deep Graph Library for GNN)
- `numpy`, `scikit-learn`
- `faiss-cpu` or `faiss-gpu` (for fast vector search)
- `tqdm` (for progress bars)
- `openai` (for Graph RAG with OpenAI)
- `fastapi`, `uvicorn` (optional, for REST API server)

---

## References

- **TxGNN Paper**: [Zero-shot prediction of therapeutic use with geometric deep learning](https://www.nature.com/articles/s41591-024-03233-x)
  - The drug repurposing prediction is inspired by TxGNN's approach using GNNs with metric learning
- **KGARevion Paper**: [An AI Agent for Knowledge-Intensive Biomedical QA](https://arxiv.org/abs/2410.04660)
  - The Graph RAG implementation is inspired by the KGARevion approach which uses Generate-Review-Revise-Answer actions
- **SynSpERT**: Span-based Joint Entity and Relation Extraction with Transformers
- **SapBERT**: Self-Alignment Pre-training for Biomedical Entity Representations
