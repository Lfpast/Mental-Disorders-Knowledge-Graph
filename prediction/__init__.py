"""
MDKG Drug Repurposing Prediction Module
========================================

A TxGNN-based knowledge graph link prediction module for drug repurposing
in mental health disorders. This module enables:

1. Drug-Disease Link Prediction: Predict potential therapeutic relationships
2. Zero-shot Prediction: Predict for diseases with limited treatment data
3. Disease Similarity Learning: Leverage similar disease profiles
4. Drug Repurposing: Identify new therapeutic uses for existing drugs
5. Prediction Explainability: GNNExplainer-based interpretation of predictions

Core Components:
- DrugRepurposingPredictor: Main prediction class with training and inference
- MDKGDataLoader: Knowledge graph data loading and processing
- HeteroRGCN: Graph neural network for node embeddings
- LinkPredictor: Link prediction with metric learning
- GNNExplainer: Explainability module (arxiv:1903.03894)

Scalability Features:
- Mini-batch training with neighbor sampling for large graphs
- Sparse operations for memory efficiency
- Cached disease similarity computation

Reference:
- TxGNN: Zero-shot prediction of therapeutic use with geometric deep learning
  (Nature Medicine, 2024)
- GNNExplainer: Generating Explanations for Graph Neural Networks
  (NeurIPS 2019, arxiv:1903.03894)
- MDKG: Mental Disorders Knowledge Graph

Author: JIA Yusheng, Jackson
"""

from .data_loader import MDKGDataLoader
from .models import HeteroRGCN, LinkPredictor, MiniBatchEdgeSampler
from .predictor import DrugRepurposingPredictor, TrainingConfig
from .explainer import GNNExplainer, ExplanationResult
from .evaluator import DrugRepurposingEvaluator

__version__ = "2.0.0"
__all__ = [
    # Data loading
    "MDKGDataLoader",
    # Models
    "HeteroRGCN", 
    "LinkPredictor",
    "MiniBatchEdgeSampler",
    # Prediction
    "DrugRepurposingPredictor",
    "TrainingConfig",
    # Explainability
    "GNNExplainer",
    "ExplanationResult",
    # Evaluation
    "DrugRepurposingEvaluator"
]
