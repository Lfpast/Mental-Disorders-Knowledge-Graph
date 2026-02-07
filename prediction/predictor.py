"""
Drug Repurposing Predictor
==========================

Main prediction class that orchestrates data loading, model training,
and drug repurposing prediction for the MDKG project.

Based on TxGNN's approach to therapeutic prediction with:
- Pre-training on all knowledge graph edges
- Fine-tuning on drug-disease relations
- Metric learning for zero-shot prediction
- Disease-centric evaluation
- GNNExplainer integration for prediction interpretability

Scalability Optimizations:
- Mini-batch training with neighbor sampling
- Sparse operations for large graphs
- Memory-efficient prototype learning

Example:
    >>> from prediction import DrugRepurposingPredictor
    >>> predictor = DrugRepurposingPredictor(data_folder="./models/InputsAndOutputs")
    >>> predictor.load_data()
    >>> predictor.train(n_epochs=100)
    >>> results = predictor.predict_repurposing("quetiapine")
    >>> explanation = predictor.explain_prediction("quetiapine", "depression")
"""

import os
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

from .data_loader import MDKGDataLoader
from .models import HeteroRGCN, LinkPredictor, MiniBatchEdgeSampler
from .explainer import GNNExplainer, ExplanationResult


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model architecture
    n_inp: int = 256          # Increased from 128 for richer representations
    n_hid: int = 256          # Increased hidden dimension
    n_out: int = 256          # Increased output dimension
    attention: bool = True    # Enable attention for better aggregation
    proto: bool = True
    proto_num: int = 5        # Increased from 3 for more prototype neighbors
    sim_measure: str = 'embedding'  # Options: 'embedding', 'profile', 'bert', 'profile+embedding'
    agg_measure: str = 'rarity'     # Options: 'rarity', 'avg', 'learn', 'heuristics-0.8'
    exp_lambda: float = 0.7   # TxGNN paper default
    dropout: float = 0.2      # Increased dropout for regularization
    
    # Training hyperparameters
    pretrain_epochs: int = 100   # Increased for better pretraining
    finetune_epochs: int = 300   # Increased for better convergence
    pretrain_lr: float = 5e-4    # Reduced for more stable training
    finetune_lr: float = 1e-4    # Reduced for finer tuning
    batch_size: int = 1024
    weight_decay: float = 1e-5   # Added L2 regularization
    patience: int = 30           # Increased patience
    
    # Negative sampling
    neg_ratio: int = 5           # Increased from 1 for harder negative samples
    
    # Mini-batch training (scalability)
    use_mini_batch: bool = False  # Enable for large graphs
    fanouts: List[int] = field(default_factory=lambda: [15, 10, 5])
    num_workers: int = 0
    
    # Explainability
    enable_explainer: bool = True
    explainer_epochs: int = 100
    explainer_lr: float = 0.01
    
    # Logging
    print_every: int = 10
    eval_every: int = 10         # More frequent evaluation
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class NegativeSampler:
    """
    Negative edge sampler for link prediction training.
    
    Samples negative edges by corrupting the tail (or head) of positive edges.
    """
    
    def __init__(
        self,
        G: Any,
        neg_ratio: int = 1,
        mode: str = 'tail',
        device: str = 'cpu'
    ):
        """
        Initialize negative sampler.
        
        Args:
            G: DGL heterogeneous graph
            neg_ratio: Number of negatives per positive
            mode: Corruption mode ('tail', 'head', 'both')
            device: Computing device
        """
        self.G = G
        self.neg_ratio = neg_ratio
        self.mode = mode
        self.device = device
        
        # Cache number of nodes per type
        self.num_nodes = {ntype: G.num_nodes(ntype) for ntype in G.ntypes}
    
    def sample(self, pos_graph: Any) -> Any:
        """
        Sample negative edges.
        
        Args:
            pos_graph: Graph with positive edges
        
        Returns:
            Graph with negative edges
        """
        neg_edges = {}
        
        for etype in pos_graph.canonical_etypes:
            src_type, rel_type, dst_type = etype
            
            if pos_graph.num_edges(etype) == 0:
                continue
            
            src, dst = pos_graph.edges(etype=etype)
            n_pos = len(src)
            n_neg = n_pos * self.neg_ratio
            
            if self.mode in ['tail', 'both']:
                # Corrupt tail
                neg_dst = torch.randint(
                    0, self.num_nodes[dst_type], (n_neg,),
                    device=self.device
                )
                neg_src = src.repeat(self.neg_ratio)
            else:
                # Corrupt head
                neg_src = torch.randint(
                    0, self.num_nodes[src_type], (n_neg,),
                    device=self.device
                )
                neg_dst = dst.repeat(self.neg_ratio)
            
            neg_edges[etype] = (neg_src, neg_dst)
        
        # Create negative graph
        neg_graph = dgl.heterograph(
            neg_edges,
            num_nodes_dict={ntype: self.G.num_nodes(ntype) for ntype in self.G.ntypes}
        )
        
        return neg_graph.to(self.device)


class DrugRepurposingPredictor:
    """
    Main class for drug repurposing prediction.
    
    Provides a complete pipeline for:
    1. Loading MDKG data
    2. Building knowledge graph
    3. Training GNN model with pre-training and fine-tuning
    4. Making drug repurposing predictions
    5. Evaluating predictions
    
    Example:
        >>> predictor = DrugRepurposingPredictor()
        >>> predictor.load_data()
        >>> predictor.train()
        >>> results = predictor.predict_repurposing("quetiapine")
        >>> for disease, score in results[:10]:
        ...     print(f"{disease}: {score:.4f}")
    """
    
    def __init__(
        self,
        data_folder: str = "./models/InputsAndOutputs",
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            data_folder: Path to data folder
            config: Training configuration
            device: Computing device (auto-detect if None)
            output_dir: Directory for saving outputs
        """
        self.data_folder = data_folder
        self.config = config or TrainingConfig()
        self.output_dir = output_dir or os.path.join(data_folder, "output", "prediction")
        
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Components
        self.data_loader: Optional[MDKGDataLoader] = None
        self.G: Optional[Any] = None
        self.model: Optional[HeteroRGCN] = None
        self.predictor: Optional[LinkPredictor] = None
        
        # Training state
        self.is_trained = False
        self.best_model_state: Optional[Dict] = None
        self.training_history: List[Dict] = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"DrugRepurposingPredictor initialized")
        print(f"  Device: {self.device}")
        print(f"  Data folder: {self.data_folder}")
        print(f"  Output directory: {self.output_dir}")
    
    def load_data(
        self,
        split: str = 'random',
        seed: int = 42,
        use_cache: bool = True,
        data_source: str = 'sampled'
    ) -> None:
        """
        Load MDKG data and prepare for training.
        
        Args:
            split: Data split strategy ('random', 'disease_centric')
            seed: Random seed
            use_cache: Whether to use cached data
            data_source: Which data to load. Options:
                - 'sampled': Active learning sampled data (default, smaller)
                - 'full': Full dataset (md_KG_all_0217.json)
                - 'full_aug': Full dataset with augmentation
                - 'train': Training set only
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required. Install with: pip install dgl")
        
        print("\n" + "="*60)
        print("Loading MDKG Data")
        print("="*60)
        print(f"  Data source: {data_source}")
        
        self.data_loader = MDKGDataLoader(data_folder=self.data_folder, data_source=data_source)
        self.data_loader.load_data(use_cache=use_cache)
        self.G = self.data_loader.build_graph()
        self.data_loader.prepare_split(split=split, seed=seed)
        
        print(f"\nGraph ready with {self.G.num_nodes()} nodes and {self.G.num_edges()} edges")
    
    def _initialize_model(self) -> None:
        """Initialize the GNN model."""
        print("\n" + "="*60)
        print("Initializing Model")
        print("="*60)
        
        self.model = HeteroRGCN(
            G=self.G,
            in_size=self.config.n_inp,
            hidden_size=self.config.n_hid,
            out_size=self.config.n_out,
            attention=self.config.attention,
            proto=self.config.proto,
            proto_num=self.config.proto_num,
            sim_measure=self.config.sim_measure,
            agg_measure=self.config.agg_measure,
            exp_lambda=self.config.exp_lambda,
            dropout=self.config.dropout,
            device=self.device
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Proto learning: {self.config.proto}")
        print(f"  Attention: {self.config.attention}")
        print(f"  Hidden dimension: {self.config.n_hid}")
    
    def _create_training_graph(self, df: List[Dict]) -> Any:
        """Create DGL graph from triplets for training."""
        edges_by_type = defaultdict(lambda: ([], []))
        
        for triplet in df:
            src_type = triplet['x_type']
            dst_type = triplet['y_type']
            rel_type = triplet['relation']
            
            etype = (src_type, rel_type, dst_type)
            edges_by_type[etype][0].append(triplet['x_idx'])
            edges_by_type[etype][1].append(triplet['y_idx'])
        
        graph_data = {}
        for etype, (src, dst) in edges_by_type.items():
            graph_data[etype] = (torch.tensor(src), torch.tensor(dst))
        
        g = dgl.heterograph(
            graph_data,
            num_nodes_dict={ntype: self.G.num_nodes(ntype) for ntype in self.G.ntypes}
        )
        
        return g.to(self.device)
    
    def pretrain(
        self,
        n_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """
        Pre-train on all edge types for general link prediction.
        
        Args:
            n_epochs: Number of epochs (default from config)
            learning_rate: Learning rate (default from config)
        
        Returns:
            Training history
        """
        n_epochs = n_epochs or self.config.pretrain_epochs
        learning_rate = learning_rate or self.config.pretrain_lr
        
        print("\n" + "="*60)
        print(f"Pre-training for {n_epochs} epochs")
        print("="*60)
        
        # Initialize model if needed
        if self.model is None:
            self._initialize_model()
        
        self.model.train()
        self.G = self.G.to(self.device)
        
        # Optimizer
        optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Negative sampler
        neg_sampler = NegativeSampler(
            self.G,
            neg_ratio=self.config.neg_ratio,
            device=self.device
        )
        
        # Training loop
        history = {'loss': [], 'auc': []}
        best_loss = float('inf')
        
        pbar = tqdm(range(n_epochs), desc="Pre-training", unit="epoch")
        for epoch in pbar:
            self.model.train()
            
            # Sample positive edges from full graph
            pos_graph = self.G
            neg_graph = neg_sampler.sample(pos_graph)
            
            # Forward pass
            optimizer.zero_grad()
            pos_scores, neg_scores, pos_combined, neg_combined = self.model(
                self.G, pos_graph, neg_graph, pretrain_mode=True
            )
            
            # Combined loss: BCE + Margin Ranking Loss for better separation
            pos_labels = torch.ones_like(pos_combined)
            neg_labels = torch.zeros_like(neg_combined)
            
            all_scores = torch.cat([pos_combined, neg_combined])
            all_labels = torch.cat([pos_labels, neg_labels])
            
            bce_loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
            
            # Margin ranking loss for pairwise comparison
            if len(pos_combined) > 0 and len(neg_combined) > 0:
                n_pairs = min(len(pos_combined), len(neg_combined))
                margin_loss = F.margin_ranking_loss(
                    pos_combined[:n_pairs],
                    neg_combined[:n_pairs],
                    torch.ones(n_pairs, device=self.device),
                    margin=1.0
                )
                loss = bce_loss + 0.5 * margin_loss
            else:
                loss = bce_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            history['loss'].append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'best': f'{best_loss:.4f}'})
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        print(f"\nPre-training complete. Best loss: {best_loss:.4f}")
        return history
    
    def finetune(
        self,
        n_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """
        Fine-tune on drug-disease relations with metric learning.
        
        Args:
            n_epochs: Number of epochs (default from config)
            learning_rate: Learning rate (default from config)
        
        Returns:
            Training history
        """
        n_epochs = n_epochs or self.config.finetune_epochs
        learning_rate = learning_rate or self.config.finetune_lr
        
        print("\n" + "="*60)
        print(f"Fine-tuning for {n_epochs} epochs")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model not initialized. Run pretrain() first.")
        
        self.model.train()
        
        # Ensure graph is on device
        self.G = self.G.to(self.device)
        
        # Optimizer
        optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # Negative sampler
        neg_sampler = NegativeSampler(
            self.G,
            neg_ratio=self.config.neg_ratio,
            device=self.device
        )
        
        # Create training and validation graphs
        train_graph = self._create_training_graph(self.data_loader.df_train)
        valid_graph = self._create_training_graph(self.data_loader.df_valid)
        
        # Training loop
        history = {'train_loss': [], 'valid_loss': []}
        best_valid_loss = float('inf')
        patience_counter = 0
        last_valid_loss = None
        
        pbar = tqdm(range(n_epochs), desc="Fine-tuning", unit="epoch")
        for epoch in pbar:
            # Training
            self.model.train()
            
            neg_graph = neg_sampler.sample(train_graph)
            
            optimizer.zero_grad()
            pos_scores, neg_scores, pos_combined, neg_combined = self.model(
                self.G, train_graph, neg_graph, pretrain_mode=False
            )
            
            # Combined loss: BCE + Margin Ranking Loss
            pos_labels = torch.ones_like(pos_combined)
            neg_labels = torch.zeros_like(neg_combined)
            
            all_scores = torch.cat([pos_combined, neg_combined])
            all_labels = torch.cat([pos_labels, neg_labels])
            
            bce_loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
            
            # Margin ranking loss
            if len(pos_combined) > 0 and len(neg_combined) > 0:
                n_pairs = min(len(pos_combined), len(neg_combined))
                margin_loss = F.margin_ranking_loss(
                    pos_combined[:n_pairs],
                    neg_combined[:n_pairs],
                    torch.ones(n_pairs, device=self.device),
                    margin=1.0
                )
                train_loss = bce_loss + 0.5 * margin_loss
            else:
                train_loss = bce_loss
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            history['train_loss'].append(train_loss.item())
            
            # Validation
            if (epoch + 1) % self.config.eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    neg_valid = neg_sampler.sample(valid_graph)
                    _, _, pos_v, neg_v = self.model(
                        self.G, valid_graph, neg_valid, pretrain_mode=False
                    )
                    
                    all_v = torch.cat([pos_v, neg_v])
                    labels_v = torch.cat([
                        torch.ones_like(pos_v),
                        torch.zeros_like(neg_v)
                    ])
                    
                    valid_loss = F.binary_cross_entropy_with_logits(all_v, labels_v)
                    history['valid_loss'].append(valid_loss.item())
                    last_valid_loss = valid_loss.item()
                
                scheduler.step(valid_loss)
                
                # Early stopping
                if valid_loss.item() < best_valid_loss:
                    best_valid_loss = valid_loss.item()
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        pbar.close()
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            
            # Update progress bar
            postfix = {'train': f'{train_loss.item():.4f}', 'best_val': f'{best_valid_loss:.4f}'}
            if last_valid_loss is not None:
                postfix['val'] = f'{last_valid_loss:.4f}'
            pbar.set_postfix(postfix)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.is_trained = True
        print(f"\nFine-tuning complete. Best validation loss: {best_valid_loss:.4f}")
        
        return history
    
    def train(
        self,
        pretrain_epochs: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
        skip_pretrain: bool = False
    ) -> Dict[str, Any]:
        """
        Full training pipeline: pre-training + fine-tuning.
        
        Args:
            pretrain_epochs: Number of pre-training epochs
            finetune_epochs: Number of fine-tuning epochs
            skip_pretrain: Skip pre-training phase
        
        Returns:
            Combined training history
        """
        if self.data_loader is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        history = {}
        
        if not skip_pretrain:
            history['pretrain'] = self.pretrain(n_epochs=pretrain_epochs)
        else:
            if self.model is None:
                self._initialize_model()
        
        history['finetune'] = self.finetune(n_epochs=finetune_epochs)
        
        # Initialize predictor
        self.predictor = LinkPredictor(self.model, self.G, self.device)
        
        return history
    
    def predict_drug_disease(
        self,
        drug_name: str,
        disease_name: str,
        relation: str = 'treatment_for'
    ) -> float:
        """
        Predict the therapeutic relationship score between a drug and disease.
        
        Args:
            drug_name: Drug name (e.g., "quetiapine")
            disease_name: Disease name (e.g., "depression")
            relation: Relation type
        
        Returns:
            Prediction score (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find drug index
        drug_id = f"drug:{drug_name.lower()}"
        drug_idx = self.data_loader.entity2idx['drug'].get(drug_id)
        
        if drug_idx is None:
            # Try fuzzy match
            for eid, idx in self.data_loader.entity2idx['drug'].items():
                if drug_name.lower() in eid.lower():
                    drug_idx = idx
                    break
        
        # Find disease index
        disease_id = f"disease:{disease_name.lower()}"
        disease_idx = self.data_loader.entity2idx['disease'].get(disease_id)
        
        if disease_idx is None:
            # Try fuzzy match
            for eid, idx in self.data_loader.entity2idx['disease'].items():
                if disease_name.lower() in eid.lower():
                    disease_idx = idx
                    break
        
        if drug_idx is None:
            raise ValueError(f"Drug not found: {drug_name}")
        if disease_idx is None:
            raise ValueError(f"Disease not found: {disease_name}")
        
        return self.predictor.predict_drug_disease(drug_idx, disease_idx, relation)
    
    def predict_repurposing(
        self,
        drug_name: str,
        relation: str = 'treatment_for',
        top_k: int = 20,
        exclude_known: bool = True
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Predict potential new therapeutic uses for a drug.
        
        This is the main drug repurposing function that ranks all diseases
        by predicted therapeutic relationship score.
        
        Args:
            drug_name: Drug name
            relation: Relation type to predict
            top_k: Number of top predictions to return
            exclude_known: Exclude known indications from predictions
        
        Returns:
            List of (disease_name, score, ontology_info) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find drug index
        drug_id = f"drug:{drug_name.lower()}"
        drug_idx = self.data_loader.entity2idx['drug'].get(drug_id)
        
        if drug_idx is None:
            # Fuzzy match
            for eid, idx in self.data_loader.entity2idx['drug'].items():
                if drug_name.lower() in eid.lower():
                    drug_idx = idx
                    drug_id = eid
                    break
        
        if drug_idx is None:
            raise ValueError(f"Drug not found: {drug_name}")
        
        # Get all disease predictions
        rankings = self.predictor.predict_all_diseases_for_drug(drug_idx, relation)
        
        # Get known indications if excluding
        known_diseases = set()
        if exclude_known:
            for triplet in self.data_loader.df_train + self.data_loader.df_valid:
                if triplet['x_idx'] == drug_idx and triplet['x_type'] == 'drug':
                    known_diseases.add(triplet['y_idx'])
        
        # Format results
        results = []
        for disease_idx, score in rankings:
            if exclude_known and disease_idx in known_diseases:
                continue
            
            disease_name = self.data_loader.get_entity_name('disease', disease_idx)
            ontology_info = self.data_loader.get_entity_ontology('disease', disease_idx)
            
            results.append((disease_name, score, ontology_info))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def predict_treatments(
        self,
        disease_name: str,
        relation: str = 'treatment_for',
        top_k: int = 20,
        exclude_known: bool = True
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Predict potential drugs for treating a disease.
        
        Args:
            disease_name: Disease name
            relation: Relation type
            top_k: Number of top predictions
            exclude_known: Exclude known treatments
        
        Returns:
            List of (drug_name, score, ontology_info) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find disease index
        disease_id = f"disease:{disease_name.lower()}"
        disease_idx = self.data_loader.entity2idx['disease'].get(disease_id)
        
        if disease_idx is None:
            for eid, idx in self.data_loader.entity2idx['disease'].items():
                if disease_name.lower() in eid.lower():
                    disease_idx = idx
                    disease_id = eid
                    break
        
        if disease_idx is None:
            raise ValueError(f"Disease not found: {disease_name}")
        
        # Get all drug predictions
        rankings = self.predictor.predict_all_drugs_for_disease(disease_idx, relation)
        
        # Get known treatments
        known_drugs = set()
        if exclude_known:
            for triplet in self.data_loader.df_train + self.data_loader.df_valid:
                if triplet['y_idx'] == disease_idx and triplet['y_type'] == 'disease':
                    known_drugs.add(triplet['x_idx'])
        
        # Format results
        results = []
        for drug_idx, score in rankings:
            if exclude_known and drug_idx in known_drugs:
                continue
            
            drug_name = self.data_loader.get_entity_name('drug', drug_idx)
            ontology_info = self.data_loader.get_entity_ontology('drug', drug_idx)
            
            results.append((drug_name, score, ontology_info))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def explain_prediction(
        self,
        drug_name: str,
        disease_name: str,
        relation: str = 'treatment_for',
        num_hops: int = 2,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        return_pathways: bool = True,
        visualize: bool = False
    ) -> ExplanationResult:
        """
        Generate explanation for a drug-disease prediction using GNNExplainer.
        
        This method applies the GNNExplainer algorithm (arxiv:1903.03894) to
        identify the most important edges and nodes in the knowledge graph
        that contribute to the prediction.
        
        Args:
            drug_name: Drug name
            disease_name: Disease name
            relation: Relation type
            num_hops: Number of hops for subgraph extraction
            epochs: GNNExplainer optimization epochs (default from config)
            lr: Learning rate (default from config)
            return_pathways: Extract important pathways
            visualize: Generate visualization
        
        Returns:
            ExplanationResult containing edge importance scores and pathways
        
        Example:
            >>> explanation = predictor.explain_prediction("metformin", "type 2 diabetes")
            >>> print(f"Top pathway: {explanation.pathways[0]}")
            >>> print(f"Edge importance: {explanation.edge_importance[:5]}")
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        epochs = epochs or self.config.explainer_epochs
        lr = lr or self.config.explainer_lr
        
        # Find drug and disease indices
        drug_idx = self._find_entity_idx('drug', drug_name)
        disease_idx = self._find_entity_idx('disease', disease_name)
        
        if drug_idx is None:
            raise ValueError(f"Drug not found: {drug_name}")
        if disease_idx is None:
            raise ValueError(f"Disease not found: {disease_name}")
        
        # Initialize GNNExplainer with entity name mappings for readable output
        explainer = GNNExplainer(
            model=self.model,
            G=self.G,
            epochs=epochs,
            lr=lr,
            edge_size=0.005,
            edge_ent=1.0,
            node_feat_size=0.001,
            node_feat_ent=0.1,
            idx2entity=self.data_loader.idx2entity
        )
        
        # Generate explanation
        explanation = explainer.explain(
            drug_idx=drug_idx,
            disease_idx=disease_idx,
            drug_name=drug_name,
            disease_name=disease_name,
            relation=relation
        )
        
        # The explanation text is printed by the shell script
        # Just return the result here
        return explanation
    
    def explain_batch(
        self,
        drug_disease_pairs: List[Tuple[str, str]],
        relation: str = 'treatment_for',
        **kwargs
    ) -> List[ExplanationResult]:
        """
        Generate explanations for multiple drug-disease pairs.
        
        Args:
            drug_disease_pairs: List of (drug_name, disease_name) tuples
            relation: Relation type
            **kwargs: Additional arguments for explain_prediction
        
        Returns:
            List of ExplanationResult objects
        """
        explanations = []
        
        for drug_name, disease_name in tqdm(drug_disease_pairs, desc="Generating explanations"):
            try:
                explanation = self.explain_prediction(
                    drug_name, disease_name, relation, **kwargs
                )
                explanations.append(explanation)
            except Exception as e:
                print(f"Failed to explain {drug_name} -> {disease_name}: {e}")
                explanations.append(None)
        
        return explanations
    
    def _find_entity_idx(self, entity_type: str, entity_name: str) -> Optional[int]:
        """Find entity index by name with fuzzy matching.
        
        Tries multiple matching strategies:
        1. Exact match with constructed entity_id
        2. Substring match within entity_id
        3. Substring match ignoring entity_type prefix
        """
        entity_name_lower = entity_name.lower().strip()
        
        # Try exact match first
        entity_id = f"{entity_type}:{entity_name_lower}"
        entity_idx = self.data_loader.entity2idx.get(entity_type, {}).get(entity_id)
        
        if entity_idx is not None:
            return entity_idx
        
        # Try fuzzy match - look for best match
        candidates = []
        prefix = f"{entity_type}:"
        
        for eid, idx in self.data_loader.entity2idx.get(entity_type, {}).items():
            # Remove type prefix for matching
            eid_name = eid[len(prefix):] if eid.startswith(prefix) else eid
            
            # Exact match without prefix
            if eid_name == entity_name_lower:
                return idx
            
            # Substring match
            if entity_name_lower in eid_name:
                # Prefer shorter matches (more specific)
                candidates.append((len(eid_name), idx, eid_name))
            elif eid_name in entity_name_lower:
                candidates.append((len(eid_name) + 100, idx, eid_name))  # Lower priority
        
        if candidates:
            # Return the shortest (most specific) match
            candidates.sort()
            print(f"  Note: '{entity_name}' matched to '{candidates[0][2]}'")
            return candidates[0][1]
        
        # Print available entities for debugging
        all_entities = list(self.data_loader.entity2idx.get(entity_type, {}).keys())
        similar = [e for e in all_entities if any(word in e.lower() for word in entity_name_lower.split())]
        if similar:
            print(f"  Available similar {entity_type} entities: {similar[:5]}")
        
        return None
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            path: Save path (default: output_dir/model.pt)
        
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        path = path or os.path.join(self.output_dir, "model.pt")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'is_trained': self.is_trained,
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
        
        return path
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Path to saved model
        """
        if self.data_loader is None or self.G is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.config = TrainingConfig.from_dict(checkpoint['config'])
        self._initialize_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        
        self.predictor = LinkPredictor(self.model, self.G, self.device)
        
        print(f"Model loaded from {path}")
    
    def get_drug_names(self) -> List[str]:
        """Get list of all drug names in the knowledge graph."""
        if self.data_loader is None:
            return []
        return [
            self.data_loader.entities[eid].span
            for eid in self.data_loader.entity2idx.get('drug', {}).keys()
        ]
    
    def get_disease_names(self) -> List[str]:
        """Get list of all disease names in the knowledge graph."""
        if self.data_loader is None:
            return []
        return [
            self.data_loader.entities[eid].span
            for eid in self.data_loader.entity2idx.get('disease', {}).keys()
        ]


def test_predictor():
    """Test the DrugRepurposingPredictor."""
    print("Testing DrugRepurposingPredictor...")
    
    # Quick config for testing
    config = TrainingConfig(
        n_inp=32,
        n_hid=32,
        n_out=32,
        pretrain_epochs=2,
        finetune_epochs=5,
        print_every=1,
        eval_every=2
    )
    
    predictor = DrugRepurposingPredictor(
        data_folder="./models/InputsAndOutputs",
        config=config
    )
    
    try:
        predictor.load_data(split='random', seed=42)
        predictor.train(skip_pretrain=True)
        
        # Test predictions
        drugs = predictor.get_drug_names()[:3]
        for drug in drugs:
            print(f"\nRepurposing predictions for {drug}:")
            results = predictor.predict_repurposing(drug, top_k=5)
            for disease, score, onto in results:
                print(f"  {disease}: {score:.4f}")
        
        print("\nPredictor test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_predictor()
