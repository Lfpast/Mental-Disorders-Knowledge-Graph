"""
Drug Repurposing Evaluation Module
==================================

This module provides evaluation metrics and tools for assessing
drug repurposing predictions, following TxGNN's evaluation methodology.

Key Features:
1. Disease-centric evaluation: Evaluate per-disease ranking quality
2. Drug-centric evaluation: Evaluate per-drug ranking quality
3. Standard metrics: MRR, Recall@K, AUROC, AUPRC, etc.
4. Zero-shot evaluation: Evaluate on unseen diseases

Reference:
- TxGNN evaluation methodology from Nature Medicine paper
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

try:
    from sklearn.metrics import (
        roc_auc_score, 
        average_precision_score,
        precision_recall_curve,
        roc_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available for evaluation metrics")


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Ranking metrics
    mrr: float  # Mean Reciprocal Rank
    hits_at_1: float
    hits_at_5: float
    hits_at_10: float
    hits_at_20: float
    
    # Classification metrics (if applicable)
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    
    # Per-entity breakdown
    per_entity_mrr: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'mrr': self.mrr,
            'hits@1': self.hits_at_1,
            'hits@5': self.hits_at_5,
            'hits@10': self.hits_at_10,
            'hits@20': self.hits_at_20,
            'auroc': self.auroc,
            'auprc': self.auprc,
        }
    
    def __str__(self) -> str:
        lines = [
            "Evaluation Results:",
            f"  MRR: {self.mrr:.4f}",
            f"  Hits@1: {self.hits_at_1:.4f}",
            f"  Hits@5: {self.hits_at_5:.4f}",
            f"  Hits@10: {self.hits_at_10:.4f}",
            f"  Hits@20: {self.hits_at_20:.4f}",
        ]
        if self.auroc is not None:
            lines.append(f"  AUROC: {self.auroc:.4f}")
        if self.auprc is not None:
            lines.append(f"  AUPRC: {self.auprc:.4f}")
        return "\n".join(lines)


class DrugRepurposingEvaluator:
    """
    Evaluator for drug repurposing predictions.
    
    Provides comprehensive evaluation metrics following TxGNN methodology:
    - Disease-centric: For each test disease, rank all drugs
    - Drug-centric: For each test drug, rank all diseases
    - Zero-shot: Evaluate on diseases not seen during training
    
    Example:
        >>> from prediction import DrugRepurposingPredictor, DrugRepurposingEvaluator
        >>> predictor = DrugRepurposingPredictor(...)
        >>> predictor.train()
        >>> evaluator = DrugRepurposingEvaluator(predictor)
        >>> results = evaluator.evaluate_disease_centric()
        >>> print(results)
    """
    
    def __init__(self, predictor: Any):
        """
        Initialize evaluator.
        
        Args:
            predictor: Trained DrugRepurposingPredictor instance
        """
        self.predictor = predictor
        self.data_loader = predictor.data_loader
        self.model = predictor.model
        self.G = predictor.G
        self.device = predictor.device
        
        # Cache test data
        self.df_test = self.data_loader.df_test
        self.df_train = self.data_loader.df_train
        self.df_valid = self.data_loader.df_valid
        
        # Build test edge sets for filtering
        self._build_test_sets()
    
    def _build_test_sets(self) -> None:
        """Build sets of test edges for evaluation."""
        # Test drug-disease pairs
        self.test_pairs = set()
        self.test_by_disease: Dict[int, List[int]] = defaultdict(list)
        self.test_by_drug: Dict[int, List[int]] = defaultdict(list)
        
        for triplet in self.df_test:
            if triplet['x_type'] == 'drug' and triplet['y_type'] == 'disease':
                drug_idx = triplet['x_idx']
                disease_idx = triplet['y_idx']
                self.test_pairs.add((drug_idx, disease_idx))
                self.test_by_disease[disease_idx].append(drug_idx)
                self.test_by_drug[drug_idx].append(disease_idx)
        
        # Training drug-disease pairs (for filtering)
        self.train_pairs = set()
        for triplet in self.df_train + self.df_valid:
            if triplet['x_type'] == 'drug' and triplet['y_type'] == 'disease':
                self.train_pairs.add((triplet['x_idx'], triplet['y_idx']))
        
        print(f"Evaluation setup:")
        print(f"  Test pairs: {len(self.test_pairs)}")
        print(f"  Test diseases: {len(self.test_by_disease)}")
        print(f"  Test drugs: {len(self.test_by_drug)}")
    
    def evaluate_disease_centric(
        self,
        disease_idxs: Optional[List[int]] = None,
        relation: str = 'treatment_for',
        filtered: bool = True,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Disease-centric evaluation: For each disease, rank all drugs.
        
        For each test disease, compute the ranking of its true treatments
        among all possible drugs.
        
        Args:
            disease_idxs: Specific disease indices to evaluate (None = all test)
            relation: Relation type to evaluate
            filtered: Filter out training edges from ranking
            verbose: Print progress
        
        Returns:
            EvaluationResult with ranking metrics
        """
        if disease_idxs is None:
            disease_idxs = list(self.test_by_disease.keys())
        
        if verbose:
            print(f"\nDisease-centric evaluation on {len(disease_idxs)} diseases")
        
        # Ensure embeddings are computed
        if self.predictor.predictor is None:
            raise ValueError("Model not ready for prediction")
        
        self.predictor.predictor.compute_embeddings()
        
        # Collect rankings
        all_ranks = []
        per_disease_mrr = {}
        
        for disease_idx in disease_idxs:
            true_drugs = set(self.test_by_disease.get(disease_idx, []))
            if len(true_drugs) == 0:
                continue
            
            # Get all drug rankings for this disease
            rankings = self.predictor.predictor.predict_all_drugs_for_disease(
                disease_idx, relation
            )
            
            # Filter training edges
            if filtered:
                filtered_rankings = []
                for drug_idx, score in rankings:
                    if (drug_idx, disease_idx) not in self.train_pairs:
                        filtered_rankings.append((drug_idx, score))
                rankings = filtered_rankings
            
            # Find ranks of true drugs
            ranks = []
            for rank, (drug_idx, score) in enumerate(rankings, 1):
                if drug_idx in true_drugs:
                    ranks.append(rank)
            
            if len(ranks) > 0:
                all_ranks.extend(ranks)
                per_disease_mrr[disease_idx] = np.mean([1.0 / r for r in ranks])
        
        if len(all_ranks) == 0:
            return EvaluationResult(
                mrr=0.0, hits_at_1=0.0, hits_at_5=0.0,
                hits_at_10=0.0, hits_at_20=0.0
            )
        
        # Compute metrics
        ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / ranks)
        hits_at_1 = np.mean(ranks <= 1)
        hits_at_5 = np.mean(ranks <= 5)
        hits_at_10 = np.mean(ranks <= 10)
        hits_at_20 = np.mean(ranks <= 20)
        
        result = EvaluationResult(
            mrr=mrr,
            hits_at_1=hits_at_1,
            hits_at_5=hits_at_5,
            hits_at_10=hits_at_10,
            hits_at_20=hits_at_20,
            per_entity_mrr=per_disease_mrr
        )
        
        if verbose:
            print(result)
        
        return result
    
    def evaluate_drug_centric(
        self,
        drug_idxs: Optional[List[int]] = None,
        relation: str = 'treatment_for',
        filtered: bool = True,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Drug-centric evaluation: For each drug, rank all diseases.
        
        For each test drug, compute the ranking of its true indications
        among all possible diseases.
        
        Args:
            drug_idxs: Specific drug indices to evaluate (None = all test)
            relation: Relation type to evaluate
            filtered: Filter out training edges from ranking
            verbose: Print progress
        
        Returns:
            EvaluationResult with ranking metrics
        """
        if drug_idxs is None:
            drug_idxs = list(self.test_by_drug.keys())
        
        if verbose:
            print(f"\nDrug-centric evaluation on {len(drug_idxs)} drugs")
        
        self.predictor.predictor.compute_embeddings()
        
        all_ranks = []
        per_drug_mrr = {}
        
        for drug_idx in drug_idxs:
            true_diseases = set(self.test_by_drug.get(drug_idx, []))
            if len(true_diseases) == 0:
                continue
            
            # Get all disease rankings for this drug
            rankings = self.predictor.predictor.predict_all_diseases_for_drug(
                drug_idx, relation
            )
            
            # Filter training edges
            if filtered:
                filtered_rankings = []
                for disease_idx, score in rankings:
                    if (drug_idx, disease_idx) not in self.train_pairs:
                        filtered_rankings.append((disease_idx, score))
                rankings = filtered_rankings
            
            # Find ranks
            ranks = []
            for rank, (disease_idx, score) in enumerate(rankings, 1):
                if disease_idx in true_diseases:
                    ranks.append(rank)
            
            if len(ranks) > 0:
                all_ranks.extend(ranks)
                per_drug_mrr[drug_idx] = np.mean([1.0 / r for r in ranks])
        
        if len(all_ranks) == 0:
            return EvaluationResult(
                mrr=0.0, hits_at_1=0.0, hits_at_5=0.0,
                hits_at_10=0.0, hits_at_20=0.0
            )
        
        ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / ranks)
        hits_at_1 = np.mean(ranks <= 1)
        hits_at_5 = np.mean(ranks <= 5)
        hits_at_10 = np.mean(ranks <= 10)
        hits_at_20 = np.mean(ranks <= 20)
        
        result = EvaluationResult(
            mrr=mrr,
            hits_at_1=hits_at_1,
            hits_at_5=hits_at_5,
            hits_at_10=hits_at_10,
            hits_at_20=hits_at_20,
            per_entity_mrr=per_drug_mrr
        )
        
        if verbose:
            print(result)
        
        return result
    
    def evaluate_link_prediction(
        self,
        relation: str = 'treatment_for',
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Standard link prediction evaluation with classification metrics.
        
        Evaluates model ability to distinguish positive edges from
        randomly sampled negative edges.
        
        Args:
            relation: Relation type to evaluate
            verbose: Print progress
        
        Returns:
            EvaluationResult with classification metrics
        """
        if verbose:
            print("\nLink prediction evaluation")
        
        self.predictor.predictor.compute_embeddings()
        
        # Collect positive scores
        pos_scores = []
        for triplet in self.df_test:
            if triplet['x_type'] == 'drug' and triplet['y_type'] == 'disease':
                drug_idx = triplet['x_idx']
                disease_idx = triplet['y_idx']
                score = self.predictor.predictor.predict_drug_disease(
                    drug_idx, disease_idx, relation
                )
                pos_scores.append(score)
        
        # Sample negative edges
        n_neg = len(pos_scores)
        drug_idxs = list(self.data_loader.entity2idx['drug'].values())
        disease_idxs = list(self.data_loader.entity2idx['disease'].values())
        
        neg_scores = []
        tried = set()
        while len(neg_scores) < n_neg:
            drug_idx = np.random.choice(drug_idxs)
            disease_idx = np.random.choice(disease_idxs)
            
            if (drug_idx, disease_idx) in tried:
                continue
            tried.add((drug_idx, disease_idx))
            
            # Exclude positive edges
            if (drug_idx, disease_idx) in self.test_pairs:
                continue
            if (drug_idx, disease_idx) in self.train_pairs:
                continue
            
            score = self.predictor.predictor.predict_drug_disease(
                drug_idx, disease_idx, relation
            )
            neg_scores.append(score)
        
        # Compute metrics
        scores = np.array(pos_scores + neg_scores)
        labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
        
        auroc = None
        auprc = None
        
        if SKLEARN_AVAILABLE:
            try:
                auroc = roc_auc_score(labels, scores)
                auprc = average_precision_score(labels, scores)
            except Exception as e:
                print(f"Warning: Could not compute metrics: {e}")
        
        # Compute ranking metrics from positive sample ranks
        all_ranks = []
        for i, score in enumerate(pos_scores):
            rank = 1 + np.sum(np.array(pos_scores + neg_scores) > score)
            all_ranks.append(rank)
        
        ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / ranks)
        hits_at_1 = np.mean(ranks <= 1)
        hits_at_5 = np.mean(ranks <= 5)
        hits_at_10 = np.mean(ranks <= 10)
        hits_at_20 = np.mean(ranks <= 20)
        
        result = EvaluationResult(
            mrr=mrr,
            hits_at_1=hits_at_1,
            hits_at_5=hits_at_5,
            hits_at_10=hits_at_10,
            hits_at_20=hits_at_20,
            auroc=auroc,
            auprc=auprc
        )
        
        if verbose:
            print(result)
        
        return result
    
    def evaluate_zero_shot(
        self,
        relation: str = 'treatment_for',
        min_train_samples: int = 0,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Zero-shot evaluation on diseases with minimal training data.
        
        Evaluates model's ability to generalize to diseases that have
        very few or no training examples (leveraging disease similarity).
        
        Args:
            relation: Relation type
            min_train_samples: Max training samples for "zero-shot" diseases
            verbose: Print progress
        
        Returns:
            EvaluationResult for zero-shot diseases
        """
        if verbose:
            print(f"\nZero-shot evaluation (diseases with <={min_train_samples} training samples)")
        
        # Count training samples per disease
        train_counts: Dict[int, int] = defaultdict(int)
        for triplet in self.df_train:
            if triplet['y_type'] == 'disease':
                train_counts[triplet['y_idx']] += 1
        
        # Find zero-shot test diseases
        zero_shot_diseases = [
            d_idx for d_idx in self.test_by_disease.keys()
            if train_counts.get(d_idx, 0) <= min_train_samples
        ]
        
        if len(zero_shot_diseases) == 0:
            print("No zero-shot diseases found in test set")
            return EvaluationResult(
                mrr=0.0, hits_at_1=0.0, hits_at_5=0.0,
                hits_at_10=0.0, hits_at_20=0.0
            )
        
        if verbose:
            print(f"  Found {len(zero_shot_diseases)} zero-shot diseases")
        
        # Evaluate on zero-shot diseases
        return self.evaluate_disease_centric(
            disease_idxs=zero_shot_diseases,
            relation=relation,
            verbose=verbose
        )
    
    def full_evaluation(
        self,
        relation: str = 'treatment_for',
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Run full evaluation suite.
        
        Args:
            relation: Relation type to evaluate
            save_path: Path to save results (JSON)
            verbose: Print progress
        
        Returns:
            Dictionary of evaluation results
        """
        print("\n" + "="*60)
        print("Full Evaluation Suite")
        print("="*60)
        
        results = {
            'disease_centric': self.evaluate_disease_centric(
                relation=relation, verbose=verbose
            ),
            'drug_centric': self.evaluate_drug_centric(
                relation=relation, verbose=verbose
            ),
            'link_prediction': self.evaluate_link_prediction(
                relation=relation, verbose=verbose
            ),
            'zero_shot': self.evaluate_zero_shot(
                relation=relation, verbose=verbose
            ),
        }
        
        if save_path:
            results_dict = {k: v.to_dict() for k, v in results.items()}
            with open(save_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"\nResults saved to {save_path}")
        
        return results
    
    def get_hardest_cases(
        self,
        mode: str = 'disease',
        n_cases: int = 10,
        relation: str = 'treatment_for'
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Get the hardest prediction cases (lowest MRR).
        
        Useful for error analysis and understanding model limitations.
        
        Args:
            mode: 'disease' or 'drug'
            n_cases: Number of cases to return
            relation: Relation type
        
        Returns:
            List of (entity_name, mrr, true_targets) tuples
        """
        if mode == 'disease':
            result = self.evaluate_disease_centric(
                relation=relation, verbose=False
            )
            entity_mrr = result.per_entity_mrr or {}
            entity_type = 'disease'
            target_type = 'drug'
            targets_by_entity = self.test_by_disease
        else:
            result = self.evaluate_drug_centric(
                relation=relation, verbose=False
            )
            entity_mrr = result.per_entity_mrr or {}
            entity_type = 'drug'
            target_type = 'disease'
            targets_by_entity = self.test_by_drug
        
        # Sort by MRR (ascending = hardest first)
        sorted_cases = sorted(entity_mrr.items(), key=lambda x: x[1])
        
        hard_cases = []
        for entity_idx, mrr in sorted_cases[:n_cases]:
            entity_name = self.data_loader.get_entity_name(entity_type, entity_idx)
            true_targets = [
                self.data_loader.get_entity_name(target_type, t_idx)
                for t_idx in targets_by_entity.get(entity_idx, [])
            ]
            hard_cases.append((entity_name, mrr, true_targets))
        
        return hard_cases


def test_evaluator():
    """Test the evaluator."""
    print("Testing DrugRepurposingEvaluator...")
    print("Note: This requires a trained predictor")
    
    # Would need actual trained predictor to test
    print("Evaluator module loaded successfully!")


if __name__ == "__main__":
    test_evaluator()
