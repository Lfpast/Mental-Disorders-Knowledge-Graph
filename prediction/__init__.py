"""
MDKG Drug Repurposing Prediction Module
========================================

A TxGNN-inspired knowledge graph link prediction module for drug repurposing
in mental health disorders. This module enables:

1. Drug-Disease Link Prediction: Predict potential therapeutic relationships
2. Zero-shot Prediction: Predict for diseases with limited treatment data
3. Disease Similarity Learning: Leverage similar disease profiles
4. Drug Repurposing: Identify new therapeutic uses for existing drugs

Core Components:
- DrugRepurposingPredictor: Main prediction class
- MDKGDataLoader: Knowledge graph data loading and processing
- HeteroRGCN: Graph neural network for node embeddings
- LinkPredictor: Link prediction with metric learning

Reference:
- TxGNN: Zero-shot prediction of therapeutic use with geometric deep learning
  (Nature Medicine, 2024)
- MDKG: Mental Disorders Knowledge Graph

Author: JIA Yusheng, Jackson
"""

from .data_loader import MDKGDataLoader
from .models import HeteroRGCN, LinkPredictor
from .predictor import DrugRepurposingPredictor
from .evaluator import DrugRepurposingEvaluator

__version__ = "1.0.0"
__all__ = [
    "MDKGDataLoader",
    "HeteroRGCN", 
    "LinkPredictor",
    "DrugRepurposingPredictor",
    "DrugRepurposingEvaluator"
]
