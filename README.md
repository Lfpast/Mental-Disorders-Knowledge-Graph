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
│       ├── configs/           # Model configurations
│       ├── input/             # Raw and processed datasets
│       ├── output/            # Training logs, saved models, and results
│       └── pretrained/        # Pretrained BERT/CODER models
├── shell/                     # Automation scripts for all pipelines
├── Active_learning.py         # Main script for Active Learning sample selection
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

## Requirements

- **OS**: Linux (Recommended)
- **Python**: 3.8+
- **GPU**: CUDA-capable GPU recommended for Training and BERT embedding generation.

### Key Python Packages
- `torch`, `transformers` (HuggingFace)
- `numpy`, `scikit-learn`
- `faiss-cpu` or `faiss-gpu` (for fast vector search)
- `tqdm` (for progress bars)
