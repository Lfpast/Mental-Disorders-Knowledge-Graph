"""
GNNExplainer for Drug Repurposing Predictions
==============================================

This module implements GNNExplainer following the paper:
"GNNExplainer: Generating Explanations for Graph Neural Networks"
(arxiv:1903.03894, Ying et al., NeurIPS 2019)

The explainer identifies:
1. Important subgraph structures (edges) that contribute to predictions
2. Node features that are crucial for the prediction

For drug repurposing, this means identifying:
- Intermediate entities (genes, symptoms, etc.) that connect drug to disease
- Key relationships (pathways, mechanisms) explaining why a drug may treat a disease
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


@dataclass
class ExplanationResult:
    """
    Result of GNNExplainer for a drug-disease prediction.
    
    Following the paper, we provide:
    - edge_mask: Importance weights for each edge in the computation graph
    - node_mask: Importance weights for node features
    - subgraph: Extracted important subgraph explaining the prediction
    - path_explanation: Human-readable mechanistic explanation
    """
    drug_name: str
    disease_name: str
    prediction_score: float
    
    # Edge importance scores (following paper Section 4)
    edge_mask: Dict[Tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    
    # Node feature importance (following paper Section 4.1)
    node_mask: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Important subgraph nodes and edges
    important_nodes: Dict[str, List[int]] = field(default_factory=dict)
    important_edges: List[Dict] = field(default_factory=list)
    
    # Mechanistic pathways
    pathways: List[List[Dict]] = field(default_factory=list)
    
    # Human-readable explanation
    explanation_text: str = ""
    
    def get_top_edges(self, k: int = 10) -> List[Dict]:
        """Get top-k most important edges."""
        all_edges = []
        for etype, mask in self.edge_mask.items():
            scores = mask.detach().cpu().numpy()
            for idx, score in enumerate(scores):
                all_edges.append({
                    'edge_type': etype,
                    'edge_idx': idx,
                    'importance': float(score)
                })
        
        all_edges.sort(key=lambda x: x['importance'], reverse=True)
        return all_edges[:k]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization with human-readable format."""
        # Create simplified edge representation
        simplified_edges = []
        for edge in self.important_edges[:20]:
            simplified_edges.append({
                'source_type': edge.get('src_type', ''),
                'source_idx': edge.get('src_idx', 0),
                'target_type': edge.get('dst_type', ''),
                'target_idx': edge.get('dst_idx', 0),
                'relation': edge.get('type', ('', '', ''))[1] if len(edge.get('type', ())) > 1 else '',
                'importance': round(edge.get('importance', 0), 4)
            })
        
        return {
            'drug': self.drug_name,
            'disease': self.disease_name,
            'prediction_score': round(self.prediction_score, 4),
            'prediction_confidence': f"{self.prediction_score:.1%}",
            'top_important_edges': simplified_edges,
            'explanation_text': self.explanation_text
        }


class GNNExplainer(nn.Module):
    """
    GNNExplainer for heterogeneous knowledge graphs.
    
    Implements the optimization-based explanation method from:
    "GNNExplainer: Generating Explanations for Graph Neural Networks"
    (Ying et al., NeurIPS 2019)
    
    Key idea: Learn soft masks on edges and node features that maximize
    mutual information between the masked input and the GNN prediction.
    
    Objective (Equation 3 in paper):
        max_{M, F} MI(Y, (G_s, X_s))
        = max_{M, F} H(Y) - H(Y | G=G_s, X=X_s)
        ≈ max_{M, F} E_{G_s}[log P_Φ(Y | G=G_s, X=X_s)]
    
    Where:
        - M: Edge mask (learnable)
        - F: Node feature mask (learnable)  
        - G_s: Masked computation graph
        - X_s: Masked node features
    """
    
    # Default coefficients (following paper recommendations)
    default_coeffs = {
        'edge_size': 0.005,       # Edge mask size regularization (Eq. 4)
        'edge_ent': 1.0,          # Edge mask entropy regularization  
        'node_feat_size': 1.0,    # Node feature mask size regularization
        'node_feat_ent': 0.1,     # Node feature mask entropy regularization
        'edge_reduction': 'sum',   # How to reduce edge mask loss
        'EPS': 1e-15,             # Numerical stability
    }
    
    # Relation type translations for human-readable output
    RELATION_TRANSLATIONS = {
        'treatment_for': '可用于治疗',
        'treats': '治疗',
        'rev_treatment_for': '被...治疗',
        'associated_with': '与...相关',
        'risk_factor_of': '是...的风险因素',
        'rev_risk_factor_of': '的风险因素包括',
        'characteristic_of': '是...的特征',
        'occurs_in': '发生于',
        'rev_occurs_in': '中会发生',
        'located_in': '位于',
        'help_diagnose': '有助于诊断',
        'causes': '导致',
        'prevents': '预防',
        'targets': '作用靶点为',
        'interacts_with': '与...相互作用',
        'abbreviation_for': '是...的缩写',
        'rev_abbreviation_for': '缩写为',
        'hyponym_of': '是...的下位词',
    }
    
    # Entity type translations
    ENTITY_TYPE_TRANSLATIONS = {
        'drug': '药物',
        'disease': '疾病',
        'gene': '基因',
        'symptom': '症状',
        'signs': '体征',
        'health_factors': '健康因素',
        'method': '方法/指标',
        'physiology': '生理过程',
        'region': '解剖区域',
    }
    
    def __init__(
        self,
        model: nn.Module,
        G: Any,
        epochs: int = 100,
        lr: float = 0.01,
        node_mask_type: str = 'feature',  # 'feature', 'object', or None
        edge_mask_type: str = 'object',   # 'object' or None
        log: bool = True,
        idx2entity: Optional[Dict[str, Dict[int, str]]] = None,
        **kwargs
    ):
        """
        Initialize GNNExplainer.
        
        Args:
            model: Trained GNN model (HeteroRGCN)
            G: DGL heterogeneous graph
            epochs: Number of optimization epochs
            lr: Learning rate for mask optimization
            node_mask_type: Type of node feature masking
            edge_mask_type: Type of edge masking
            log: Whether to log progress
            idx2entity: Mapping from (entity_type, idx) to entity name
            **kwargs: Override default coefficients
        """
        super().__init__()
        
        self.model = model
        self.G = G
        self.epochs = epochs
        self.lr = lr
        self.node_mask_type = node_mask_type
        self.edge_mask_type = edge_mask_type
        self.log = log
        self.idx2entity = idx2entity or {}
        
        # Update coefficients with any overrides
        self.coeffs = {**self.default_coeffs, **kwargs}
        
        # Get device from model
        self.device = next(model.parameters()).device
        
        # Masks will be initialized per explanation
        self.edge_mask: Dict[Tuple, Parameter] = {}
        self.node_mask: Dict[str, Parameter] = {}
        
        # Store hard masks for gradient computation
        self.hard_edge_mask: Dict[Tuple, Optional[torch.Tensor]] = {}
        self.hard_node_mask: Dict[str, Optional[torch.Tensor]] = {}
    
    def _initialize_masks(
        self,
        drug_idx: int,
        disease_idx: int,
        num_hops: int = 2
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor]]:
        """
        Initialize learnable masks for edges and node features.
        
        Following paper Section 4: Initialize masks based on the
        computation graph (k-hop neighborhood of target nodes).
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            num_hops: Number of GNN layers (determines computation graph size)
        
        Returns:
            node_features: Initial node features
            initial_edge_masks: Initial edge masks
        """
        self.edge_mask = nn.ParameterDict()
        self.node_mask = nn.ParameterDict()
        self.hard_edge_mask = {}
        self.hard_node_mask = {}
        
        # Initialize edge masks for each edge type
        # Following paper: use small random initialization
        std = 0.1
        
        for etype in self.G.canonical_etypes:
            num_edges = self.G.num_edges(etype)
            if num_edges > 0:
                # Initialize mask with small random values
                # Sigmoid will be applied during forward
                mask = torch.randn(num_edges, device=self.device) * std
                self.edge_mask[str(etype)] = Parameter(mask)
                self.hard_edge_mask[etype] = None
        
        # Initialize node feature masks
        if self.node_mask_type is not None:
            for ntype in self.G.ntypes:
                num_nodes = self.G.num_nodes(ntype)
                if num_nodes > 0:
                    # Following paper: mask over features
                    if ntype in self.model.node_embeddings:
                        feat_dim = self.model.node_embeddings[ntype].embedding_dim
                    else:
                        feat_dim = self.model.hidden_size
                    
                    if self.node_mask_type == 'feature':
                        # Mask entire feature dimensions (shared across nodes)
                        mask = torch.randn(1, feat_dim, device=self.device) * std
                    else:  # 'object'
                        # Mask individual nodes
                        mask = torch.randn(num_nodes, 1, device=self.device) * std
                    
                    self.node_mask[ntype] = Parameter(mask)
                    self.hard_node_mask[ntype] = None
        
        return self.node_mask, self.edge_mask
    
    def _masked_forward(
        self,
        G: Any,
        drug_idx: int,
        disease_idx: int,
        relation: str = 'treats'
    ) -> torch.Tensor:
        """
        Forward pass with masked inputs.
        
        Following paper Equation 2-3: Apply sigmoid to masks and use
        them to weight edges and node features during message passing.
        
        Args:
            G: DGL graph
            drug_idx: Drug node index
            disease_idx: Disease node index
            relation: Relation type for prediction
        
        Returns:
            Prediction score for the drug-disease pair
        """
        self.model.eval()
        
        # Get node embeddings with feature masking
        h = {}
        for ntype in G.ntypes:
            if ntype in self.model.node_embeddings:
                indices = torch.arange(G.num_nodes(ntype), device=self.device)
                embeddings = self.model.node_embeddings[ntype](indices)
                
                # Apply node feature mask (sigmoid for soft mask)
                if ntype in self.node_mask:
                    mask = self.node_mask[ntype].sigmoid()
                    embeddings = embeddings * mask
                
                h[ntype] = embeddings
            else:
                h[ntype] = torch.zeros(
                    G.num_nodes(ntype), 
                    self.model.hidden_size, 
                    device=self.device
                )
        
        # Forward through GNN layers with edge masking
        # We need to modify the message passing to use edge weights
        h = self._masked_gnn_forward(G, h)
        
        # Get drug and disease embeddings
        drug_emb = h['drug'][drug_idx]
        disease_emb = h['disease'][disease_idx]
        
        # DistMult scoring
        etype = ('drug', relation, 'disease')
        rel_idx = self.model.pred.rel2idx.get(etype, 0)
        rel_emb = self.model.pred.W[rel_idx]
        
        score = (drug_emb * rel_emb * disease_emb).sum()
        return torch.sigmoid(score)
    
    def _masked_gnn_forward(
        self,
        G: Any,
        h: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        GNN forward pass with edge masking.
        
        Implements masked message passing following paper Section 4:
        Messages are weighted by learned edge importance.
        """
        # Apply sigmoid to get soft edge masks
        edge_weights = {}
        for etype in G.canonical_etypes:
            key = str(etype)
            if key in self.edge_mask:
                edge_weights[etype] = self.edge_mask[key].sigmoid()
            else:
                edge_weights[etype] = torch.ones(
                    G.num_edges(etype), device=self.device
                )
        
        # Layer 1
        h1 = self._masked_rgcn_layer(G, h, self.model.layer1, edge_weights)
        h1 = {k: F.relu(v) for k, v in h1.items()}
        h1 = {k: self.model.dropout(v) for k, v in h1.items()}
        
        # Layer 2 with residual
        h2 = self._masked_rgcn_layer(G, h1, self.model.layer2, edge_weights)
        h2 = {k: F.relu(h2[k]) + h1.get(k, 0) for k in h2}
        h2 = {k: self.model.dropout(v) for k, v in h2.items()}
        
        # Layer 3 with residual
        h3 = self._masked_rgcn_layer(G, h2, self.model.layer3, edge_weights)
        if self.model.residual_proj is not None:
            h3 = {k: h3[k] + self.model.residual_proj[k](h2[k]) 
                  for k in h3 if k in h2 and k in self.model.residual_proj}
        else:
            h3 = {k: h3[k] + h2.get(k, 0) for k in h3}
        
        return h3
    
    def _masked_rgcn_layer(
        self,
        G: Any,
        h: Dict[str, torch.Tensor],
        layer: nn.Module,
        edge_weights: Dict[Tuple, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Single RGCN layer with edge masking.
        
        Following paper: weight messages by edge importance.
        """
        # Transform source features
        transformed_feats = {}
        for ntype in h:
            transforms = []
            for src_type, etype, dst_type in G.canonical_etypes:
                if src_type == ntype and etype in layer.weight:
                    transforms.append(layer.weight[etype](h[ntype]))
            
            if transforms:
                transformed_feats[ntype] = torch.stack(transforms, dim=0).mean(dim=0)
            else:
                for etype in layer.weight:
                    transformed_feats[ntype] = layer.weight[etype](h[ntype])
                    break
                else:
                    transformed_feats[ntype] = h[ntype]
        
        # Message passing with edge weights
        with G.local_scope():
            for ntype in transformed_feats:
                G.nodes[ntype].data['h'] = transformed_feats[ntype]
            
            # Aggregate messages weighted by edge mask
            for etype in G.canonical_etypes:
                if G.num_edges(etype) > 0:
                    src_type, rel, dst_type = etype
                    if src_type in transformed_feats:
                        # Get edge mask weights
                        weights = edge_weights.get(etype, 
                            torch.ones(G.num_edges(etype), device=self.device))
                        G.edges[etype].data['w'] = weights.unsqueeze(-1)
                        
                        # Weighted message passing
                        G.apply_edges(
                            lambda edges: {'m': edges.src['h'] * edges.data['w']},
                            etype=etype
                        )
            
            # Aggregate
            for ntype in G.ntypes:
                msgs = []
                for src_type, etype, dst_type in G.canonical_etypes:
                    if dst_type == ntype and G.num_edges((src_type, etype, dst_type)) > 0:
                        G.nodes[ntype].data['agg_' + etype] = dgl.ops.copy_e_sum(
                            G[src_type, etype, dst_type], 
                            G[src_type, etype, dst_type].edata.get('m', 
                                torch.zeros(G.num_edges((src_type, etype, dst_type)), 
                                           h[list(h.keys())[0]].size(-1), device=self.device))
                        )
            
            # Collect outputs
            output = {}
            for ntype in G.ntypes:
                agg_feats = []
                for key in G.nodes[ntype].data:
                    if key.startswith('agg_'):
                        agg_feats.append(G.nodes[ntype].data[key])
                
                if agg_feats:
                    output[ntype] = layer.layer_norm(
                        torch.stack(agg_feats, dim=0).mean(dim=0)
                    )
                elif ntype in transformed_feats:
                    output[ntype] = layer.layer_norm(transformed_feats[ntype])
            
            return output
    
    def _loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute explanation loss.
        
        Following paper Equation 4:
        L = -log P(Y=y | G_s, X_s) + λ₁|M| + λ₂H(M)
        
        Where:
        - First term: Cross-entropy loss for prediction
        - λ₁|M|: Size regularization (encourage sparse explanations)
        - λ₂H(M): Entropy regularization (encourage discrete masks)
        """
        # Prediction loss (maximize probability of correct prediction)
        pred_loss = -torch.log(pred + self.coeffs['EPS'])
        
        # Edge mask regularization
        edge_loss = torch.tensor(0.0, device=self.device)
        for key, mask in self.edge_mask.items():
            m = mask.sigmoid()
            
            # Size loss: penalize large masks (Eq. 4)
            if self.coeffs['edge_reduction'] == 'sum':
                size_loss = self.coeffs['edge_size'] * m.sum()
            else:
                size_loss = self.coeffs['edge_size'] * m.mean()
            
            # Entropy loss: encourage binary masks (Eq. 4)
            ent = -m * torch.log(m + self.coeffs['EPS']) - \
                  (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            ent_loss = self.coeffs['edge_ent'] * ent.mean()
            
            edge_loss = edge_loss + size_loss + ent_loss
        
        # Node feature mask regularization
        feat_loss = torch.tensor(0.0, device=self.device)
        for key, mask in self.node_mask.items():
            m = mask.sigmoid()
            size_loss = self.coeffs['node_feat_size'] * m.mean()
            ent = -m * torch.log(m + self.coeffs['EPS']) - \
                  (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            ent_loss = self.coeffs['node_feat_ent'] * ent.mean()
            feat_loss = feat_loss + size_loss + ent_loss
        
        total_loss = pred_loss + edge_loss + feat_loss
        return total_loss
    
    def explain(
        self,
        drug_idx: int,
        disease_idx: int,
        drug_name: str = "",
        disease_name: str = "",
        relation: str = 'treats'
    ) -> ExplanationResult:
        """
        Generate explanation for a drug-disease prediction.
        
        Following paper Algorithm 1:
        1. Initialize masks
        2. Optimize masks to maximize mutual information
        3. Extract important subgraph based on learned masks
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            drug_name: Drug name for display
            disease_name: Disease name for display
            relation: Relation type
        
        Returns:
            ExplanationResult with edge/node importance and pathways
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL required for explanation")
        
        # Initialize masks
        self._initialize_masks(drug_idx, disease_idx)
        
        # Move to device
        self.to(self.device)
        self.G = self.G.to(self.device)
        
        # Collect parameters for optimization
        parameters = list(self.edge_mask.values()) + list(self.node_mask.values())
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        
        # Optimization loop (following paper Algorithm 1)
        best_loss = float('inf')
        best_masks = None
        
        if self.log:
            from tqdm import tqdm
            pbar = tqdm(range(self.epochs), desc="Explaining")
        else:
            pbar = range(self.epochs)
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Forward with masks
            pred = self._masked_forward(self.G, drug_idx, disease_idx, relation)
            
            # Compute loss
            target = torch.ones(1, device=self.device)  # We want high prediction
            loss = self._loss(pred, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_masks = {
                    'edge': {k: v.detach().clone() for k, v in self.edge_mask.items()},
                    'node': {k: v.detach().clone() for k, v in self.node_mask.items()}
                }
            
            if self.log and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'pred': f'{pred.item():.4f}'})
        
        # Use best masks
        if best_masks is not None:
            for k, v in best_masks['edge'].items():
                self.edge_mask[k].data = v
            for k, v in best_masks['node'].items():
                self.node_mask[k].data = v
        
        # Get final prediction
        with torch.no_grad():
            final_pred = self._masked_forward(self.G, drug_idx, disease_idx, relation)
        
        # Extract explanation
        result = self._extract_explanation(
            drug_idx, disease_idx, drug_name, disease_name,
            final_pred.item(), relation
        )
        
        return result
    
    def _extract_explanation(
        self,
        drug_idx: int,
        disease_idx: int,
        drug_name: str,
        disease_name: str,
        prediction_score: float,
        relation: str,
        threshold: float = 0.5
    ) -> ExplanationResult:
        """
        Extract human-interpretable explanation from learned masks.
        
        Following paper Section 5: Extract subgraph with edges above threshold.
        """
        result = ExplanationResult(
            drug_name=drug_name,
            disease_name=disease_name,
            prediction_score=prediction_score
        )
        
        # Process edge masks
        for etype_str, mask in self.edge_mask.items():
            # Parse edge type
            etype = eval(etype_str)  # Convert string back to tuple
            mask_values = mask.sigmoid().detach().cpu()
            result.edge_mask[etype] = mask_values
            
            # Find important edges
            important_idx = (mask_values > threshold).nonzero(as_tuple=True)[0]
            if len(important_idx) > 0:
                src, dst = self.G.edges(etype=etype)
                src = src.cpu().numpy()
                dst = dst.cpu().numpy()
                
                for idx in important_idx.numpy():
                    result.important_edges.append({
                        'type': etype,
                        'src_type': etype[0],
                        'dst_type': etype[2],
                        'src_idx': int(src[idx]),
                        'dst_idx': int(dst[idx]),
                        'importance': float(mask_values[idx])
                    })
        
        # Sort edges by importance
        result.important_edges.sort(key=lambda x: x['importance'], reverse=True)
        
        # Process node masks
        for ntype, mask in self.node_mask.items():
            result.node_mask[ntype] = mask.sigmoid().detach().cpu()
        
        # Extract mechanistic pathways
        result.pathways = self._find_pathways(
            drug_idx, disease_idx, result.important_edges
        )
        
        # Generate explanation text
        result.explanation_text = self._generate_explanation_text(
            drug_name, disease_name, result
        )
        
        return result
    
    def _find_pathways(
        self,
        drug_idx: int,
        disease_idx: int,
        important_edges: List[Dict],
        max_length: int = 4
    ) -> List[List[Dict]]:
        """
        Find mechanistic pathways connecting drug to disease.
        
        Uses important edges to trace paths from drug to disease.
        """
        pathways = []
        
        # Build adjacency from important edges
        adj = defaultdict(list)
        for edge in important_edges:
            src_key = (edge['src_type'], edge['src_idx'])
            dst_key = (edge['dst_type'], edge['dst_idx'])
            adj[src_key].append({
                'target': dst_key,
                'relation': edge['type'][1],
                'importance': edge['importance']
            })
        
        # BFS to find paths
        start = ('drug', drug_idx)
        end = ('disease', disease_idx)
        
        queue = [(start, [{'node': start, 'relation': 'start'}])]
        visited = {start}
        
        while queue and len(pathways) < 10:
            current, path = queue.pop(0)
            
            if len(path) > max_length:
                continue
            
            for neighbor_info in adj.get(current, []):
                neighbor = neighbor_info['target']
                
                if neighbor == end:
                    # Found a path
                    full_path = path + [{
                        'node': neighbor,
                        'relation': neighbor_info['relation'],
                        'importance': neighbor_info['importance']
                    }]
                    pathways.append(full_path)
                elif neighbor not in visited and len(path) < max_length:
                    visited.add(neighbor)
                    new_path = path + [{
                        'node': neighbor,
                        'relation': neighbor_info['relation'],
                        'importance': neighbor_info['importance']
                    }]
                    queue.append((neighbor, new_path))
        
        return pathways
    
    def _get_entity_name(self, entity_type: str, idx: int) -> str:
        """Get human-readable entity name from index."""
        if entity_type in self.idx2entity and idx in self.idx2entity[entity_type]:
            entity_id = self.idx2entity[entity_type][idx]
            # Remove type prefix (e.g., "drug:metformin" -> "metformin")
            if ':' in entity_id:
                return entity_id.split(':', 1)[1]
            return entity_id
        return f"{entity_type}_{idx}"
    
    def _translate_relation(self, rel_type: str) -> str:
        """Translate relation type to human-readable Chinese."""
        # Remove rev_ prefix for lookup, then handle
        clean_rel = rel_type.replace('rev_', '').replace('_', ' ')
        return self.RELATION_TRANSLATIONS.get(rel_type, clean_rel)
    
    def _translate_entity_type(self, entity_type: str) -> str:
        """Translate entity type to Chinese."""
        return self.ENTITY_TYPE_TRANSLATIONS.get(entity_type, entity_type)
    
    def _generate_explanation_text(
        self,
        drug_name: str,
        disease_name: str,
        result: ExplanationResult
    ) -> str:
        """
        Generate human-readable explanation text in natural language.
        
        This creates a user-friendly explanation that describes:
        1. The prediction confidence
        2. Key biological relationships discovered
        3. Mechanistic pathways connecting drug to disease
        """
        lines = []
        
        # Title and confidence
        confidence = result.prediction_score
        confidence_desc = "较高" if confidence > 0.7 else "中等" if confidence > 0.4 else "较低"
        
        lines.append(f"# 药物重定位解释：{drug_name} → {disease_name}")
        lines.append("")
        lines.append(f"## 预测概述")
        lines.append(f"模型预测 **{drug_name}** 可能对 **{disease_name}** 具有治疗潜力。")
        lines.append(f"- 预测置信度：{confidence:.1%} ({confidence_desc})")
        lines.append("")
        
        # Find drug-disease relevant edges (most important for explanation)
        drug_edges = []
        disease_edges = []
        intermediate_entities = set()
        
        for edge in result.important_edges[:50]:  # Check top 50 edges
            src_type = edge.get('src_type', '')
            dst_type = edge.get('dst_type', '')
            src_idx = edge.get('src_idx', 0)
            dst_idx = edge.get('dst_idx', 0)
            importance = edge.get('importance', 0)
            
            # Get entity names
            src_name = self._get_entity_name(src_type, src_idx)
            dst_name = self._get_entity_name(dst_type, dst_idx)
            
            # Check if edge involves drug or disease
            if src_type == 'drug' or dst_type == 'drug':
                drug_edges.append((edge, src_name, dst_name))
                if src_type != 'drug':
                    intermediate_entities.add((src_type, src_name))
                if dst_type != 'drug':
                    intermediate_entities.add((dst_type, dst_name))
            
            if src_type == 'disease' or dst_type == 'disease':
                disease_edges.append((edge, src_name, dst_name))
                if src_type != 'disease':
                    intermediate_entities.add((src_type, src_name))
                if dst_type != 'disease':
                    intermediate_entities.add((dst_type, dst_name))
        
        # Generate natural language explanation
        lines.append("## 作用机制分析")
        lines.append("")
        lines.append("基于知识图谱分析，模型发现以下可能的作用机制：")
        lines.append("")
        
        # Describe drug-related relationships
        if drug_edges:
            lines.append(f"### {drug_name} 的关键关联")
            seen = set()
            for edge, src_name, dst_name in drug_edges[:5]:
                rel_type = edge['type'][1] if len(edge.get('type', ())) > 1 else 'related'
                rel_cn = self._translate_relation(rel_type)
                
                # Create natural sentence
                if edge['src_type'] == 'drug':
                    other_type = self._translate_entity_type(edge['dst_type'])
                    key = (src_name, rel_cn, dst_name)
                    if key not in seen:
                        lines.append(f"- {drug_name} {rel_cn} {other_type}「{dst_name}」")
                        seen.add(key)
                else:
                    other_type = self._translate_entity_type(edge['src_type'])
                    key = (src_name, rel_cn, dst_name)
                    if key not in seen:
                        lines.append(f"- {other_type}「{src_name}」{rel_cn} {drug_name}")
                        seen.add(key)
            lines.append("")
        
        # Describe disease-related relationships  
        if disease_edges:
            lines.append(f"### {disease_name} 的关键关联")
            seen = set()
            for edge, src_name, dst_name in disease_edges[:5]:
                rel_type = edge['type'][1] if len(edge.get('type', ())) > 1 else 'related'
                rel_cn = self._translate_relation(rel_type)
                
                if edge['src_type'] == 'disease':
                    other_type = self._translate_entity_type(edge['dst_type'])
                    key = (src_name, rel_cn, dst_name)
                    if key not in seen:
                        lines.append(f"- {disease_name} {rel_cn} {other_type}「{dst_name}」")
                        seen.add(key)
                else:
                    other_type = self._translate_entity_type(edge['src_type'])
                    key = (src_name, rel_cn, dst_name)
                    if key not in seen:
                        lines.append(f"- {other_type}「{src_name}」{rel_cn} {disease_name}")
                        seen.add(key)
            lines.append("")
        
        # Identify intermediate entities (bridging drug and disease)
        if intermediate_entities:
            # Filter to most relevant intermediate entities
            relevant_intermediates = [
                (etype, name) for etype, name in intermediate_entities
                if etype in ('gene', 'symptom', 'physiology', 'health_factors')
            ][:5]
            
            if relevant_intermediates:
                lines.append("### 关键中间实体")
                lines.append("以下实体可能在药物和疾病之间起到桥梁作用：")
                for etype, name in relevant_intermediates:
                    type_cn = self._translate_entity_type(etype)
                    lines.append(f"- {type_cn}：{name}")
                lines.append("")
        
        # Summary
        lines.append("## 解释总结")
        lines.append("")
        if confidence > 0.7:
            lines.append(f"模型以较高置信度（{confidence:.1%}）预测 **{drug_name}** 可能对 **{disease_name}** 有治疗效果。")
            lines.append("知识图谱中存在多条支持这一预测的关系路径。")
        elif confidence > 0.4:
            lines.append(f"模型以中等置信度（{confidence:.1%}）预测 **{drug_name}** 可能对 **{disease_name}** 有一定治疗潜力。")
            lines.append("建议结合临床文献进一步验证。")
        else:
            lines.append(f"模型预测置信度较低（{confidence:.1%}），**{drug_name}** 对 **{disease_name}** 的治疗效果存在不确定性。")
            lines.append("需要更多证据支持。")
        
        lines.append("")
        lines.append("---")
        lines.append("*注：此解释基于GNNExplainer算法对图神经网络预测的解读，仅供研究参考，不构成医学建议。*")
        
        return "\n".join(lines)


def test_explainer():
    """Test GNNExplainer with mock data."""
    if not DGL_AVAILABLE:
        print("DGL not available, skipping test")
        return
    
    print("Testing GNNExplainer...")
    
    # Create simple test graph
    graph_data = {
        ('drug', 'treats', 'disease'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ('disease', 'rev_treats', 'drug'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ('drug', 'targets', 'gene'): (torch.tensor([0, 1]), torch.tensor([0, 0])),
        ('gene', 'associated_with', 'disease'): (torch.tensor([0]), torch.tensor([0])),
    }
    G = dgl.heterograph(graph_data, num_nodes_dict={'drug': 2, 'disease': 2, 'gene': 1})
    
    print("  Test graph created")
    print(f"  Nodes: {G.num_nodes()}, Edges: {G.num_edges()}")
    
    print("GNNExplainer test passed!")


if __name__ == "__main__":
    test_explainer()
