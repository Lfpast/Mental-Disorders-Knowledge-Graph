"""
Graph RAG Evaluation Module for MDKG

This module implements evaluation metrics based on the KGARevion paper 
(https://arxiv.org/abs/2410.04660).

Key Metrics (from paper):
1. Accuracy - Primary metric for both multi-choice and open-ended QA
2. Standard Deviation (std) - Variance across multiple runs

Evaluation Settings (from paper):
1. Multi-choice Reasoning: Select correct answer from candidates
2. Open-ended Reasoning: Generate answer without predefined choices

Additional Evaluation Scenarios:
1. Query Complexity Scenario (QSS): Performance vs number of medical concepts
2. Semantic Complexity Scenario (CSS): Performance with semantically similar answers

Author: MDKG Project
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    """A question-answering example for evaluation"""
    question: str
    answer_choices: List[str] = field(default_factory=list)  # For multi-choice
    correct_answer: str = ""
    correct_answer_idx: int = -1  # For multi-choice (0-indexed)
    question_type: str = "multi-choice"  # 'multi-choice' or 'open-ended'
    medical_concepts: List[str] = field(default_factory=list)
    difficulty: str = "basic"  # 'basic', 'intermediate', 'expert'
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer_choices": self.answer_choices,
            "correct_answer": self.correct_answer,
            "correct_answer_idx": self.correct_answer_idx,
            "question_type": self.question_type,
            "medical_concepts": self.medical_concepts,
            "difficulty": self.difficulty
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a single example"""
    example: QAExample
    predicted_answer: str
    is_correct: bool
    reasoning_trace: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "question": self.example.question,
            "correct_answer": self.example.correct_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "reasoning_trace": self.reasoning_trace
        }


@dataclass
class EvaluationMetrics:
    """
    Evaluation metrics following KGARevion paper.
    
    Primary Metric: Accuracy
    - Computed as: correct_predictions / total_predictions
    
    The paper reports:
    - Accuracy (Acc.) with standard deviation (std) across 3 runs
    - Separate metrics for multi-choice and open-ended settings
    """
    total_examples: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    
    # Per-run results for computing std
    run_accuracies: List[float] = field(default_factory=list)
    std: float = 0.0
    
    # Breakdown by difficulty (following MedDDx dataset structure)
    accuracy_by_difficulty: Dict[str, float] = field(default_factory=dict)
    
    # Breakdown by number of medical concepts (QSS scenario)
    accuracy_by_concept_count: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "total_examples": self.total_examples,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy,
            "std": self.std,
            "accuracy_by_difficulty": self.accuracy_by_difficulty,
            "accuracy_by_concept_count": self.accuracy_by_concept_count
        }


class GraphRAGEvaluator:
    """
    Evaluator for Graph RAG following KGARevion paper methodology.
    
    Evaluation Settings from paper:
    1. Multi-choice reasoning: Model selects from A/B/C/D options
    2. Open-ended reasoning: Model generates answer without options
    
    Metrics:
    - Accuracy: Primary metric
    - Standard deviation across multiple runs
    """
    
    def __init__(self, graph_rag_instance):
        """
        Initialize evaluator with a Graph RAG instance.
        
        Args:
            graph_rag_instance: An instance of MentalDisorderGraphRAG
        """
        self.rag = graph_rag_instance
    
    def _format_multi_choice_prompt(self, example: QAExample) -> str:
        """
        Format a multi-choice question for the model.
        Following KGARevion paper's approach.
        """
        prompt = f"{example.question}\n\n"
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for i, choice in enumerate(example.answer_choices):
            if i < len(labels):
                prompt += f"{labels[i]}. {choice}\n"
        prompt += "\nPlease select the correct answer (A, B, C, or D)."
        return prompt
    
    def _extract_answer_choice(self, response: str, num_choices: int) -> int:
        """
        Extract the selected answer choice from model response.
        
        Returns:
            Index of selected answer (0-based), or -1 if unable to parse
        """
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:num_choices]
        
        # Try to find explicit answer patterns
        patterns = [
            r'(?:answer|choice|select)\s*(?:is|:)?\s*([A-H])',
            r'^([A-H])[\.\)\s]',
            r'\b([A-H])\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in labels:
                    return labels.index(letter)
        
        return -1
    
    def _check_open_ended_answer(
        self, 
        predicted: str, 
        correct: str,
        use_llm_judge: bool = False
    ) -> bool:
        """
        Check if open-ended answer is correct.
        
        For open-ended reasoning (following paper's approach):
        - Extract key entities from both answers
        - Check if predicted answer contains the correct answer
        - Optionally use LLM as judge for more nuanced evaluation
        """
        # Simple matching: check if correct answer appears in prediction
        correct_lower = correct.lower().strip()
        predicted_lower = predicted.lower().strip()
        
        # Exact match
        if correct_lower == predicted_lower:
            return True
        
        # Substring match (correct answer mentioned in prediction)
        if correct_lower in predicted_lower:
            return True
        
        # Fuzzy match for medical terms
        correct_words = set(correct_lower.split())
        predicted_words = set(predicted_lower.split())
        
        # If most words from correct answer appear in prediction
        overlap = len(correct_words & predicted_words) / len(correct_words)
        if overlap >= 0.7:
            return True
        
        return False
    
    def evaluate_single(
        self, 
        example: QAExample, 
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a single QA example.
        
        Args:
            example: The QA example to evaluate
            verbose: Whether to print detailed progress
            
        Returns:
            EvaluationResult with prediction and correctness
        """
        if example.question_type == "multi-choice":
            # Format as multi-choice question
            prompt = self._format_multi_choice_prompt(example)
            result = self.rag.query(prompt, verbose=verbose)
            
            # Extract predicted choice
            predicted_idx = self._extract_answer_choice(
                result.answer, 
                len(example.answer_choices)
            )
            
            is_correct = predicted_idx == example.correct_answer_idx
            predicted_answer = (
                example.answer_choices[predicted_idx] 
                if 0 <= predicted_idx < len(example.answer_choices)
                else result.answer
            )
        else:
            # Open-ended: just ask the question
            result = self.rag.query(example.question, verbose=verbose)
            is_correct = self._check_open_ended_answer(
                result.answer, 
                example.correct_answer
            )
            predicted_answer = result.answer
        
        return EvaluationResult(
            example=example,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            reasoning_trace=result.reasoning_trace
        )
    
    def evaluate_dataset(
        self, 
        examples: List[QAExample],
        num_runs: int = 3,
        verbose: bool = False
    ) -> EvaluationMetrics:
        """
        Evaluate on a dataset following KGARevion methodology.
        
        The paper reports accuracy with standard deviation across 3 runs.
        
        Args:
            examples: List of QA examples
            num_runs: Number of evaluation runs (default 3, as in paper)
            verbose: Whether to print progress
            
        Returns:
            EvaluationMetrics with accuracy and std
        """
        all_run_results = []
        
        for run in range(num_runs):
            if verbose:
                logger.info(f"Evaluation run {run + 1}/{num_runs}")
            
            run_results = []
            for i, example in enumerate(examples):
                if verbose and i % 10 == 0:
                    logger.info(f"  Processing example {i + 1}/{len(examples)}")
                
                result = self.evaluate_single(example, verbose=False)
                run_results.append(result)
            
            all_run_results.append(run_results)
        
        # Compute metrics
        metrics = self._compute_metrics(all_run_results, examples)
        
        return metrics
    
    def _compute_metrics(
        self, 
        all_run_results: List[List[EvaluationResult]],
        examples: List[QAExample]
    ) -> EvaluationMetrics:
        """
        Compute evaluation metrics following paper methodology.
        """
        metrics = EvaluationMetrics()
        metrics.total_examples = len(examples)
        
        # Compute accuracy per run
        for run_results in all_run_results:
            correct = sum(1 for r in run_results if r.is_correct)
            accuracy = correct / len(run_results) if run_results else 0.0
            metrics.run_accuracies.append(accuracy)
        
        # Mean accuracy and std
        metrics.accuracy = np.mean(metrics.run_accuracies)
        metrics.std = np.std(metrics.run_accuracies)
        metrics.correct_predictions = int(metrics.accuracy * metrics.total_examples)
        
        # Accuracy by difficulty level
        difficulty_correct = defaultdict(list)
        for result in all_run_results[0]:  # Use first run for breakdown
            difficulty = result.example.difficulty
            difficulty_correct[difficulty].append(result.is_correct)
        
        for difficulty, correct_list in difficulty_correct.items():
            metrics.accuracy_by_difficulty[difficulty] = (
                sum(correct_list) / len(correct_list) if correct_list else 0.0
            )
        
        # Accuracy by concept count (QSS scenario from paper)
        concept_correct = defaultdict(list)
        for result in all_run_results[0]:
            n_concepts = len(result.example.medical_concepts)
            concept_correct[n_concepts].append(result.is_correct)
        
        for n_concepts, correct_list in concept_correct.items():
            metrics.accuracy_by_concept_count[n_concepts] = (
                sum(correct_list) / len(correct_list) if correct_list else 0.0
            )
        
        return metrics
    
    def generate_report(
        self, 
        metrics: EvaluationMetrics,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report following paper's reporting format.
        
        The paper reports: Acc. (std) for each dataset
        """
        report_lines = [
            "=" * 60,
            "Graph RAG Evaluation Report",
            "Following KGARevion paper methodology",
            "=" * 60,
            "",
            f"Total Examples: {metrics.total_examples}",
            f"Accuracy: {metrics.accuracy:.3f} (std: {metrics.std:.3f})",
            "",
            "Accuracy by Difficulty:",
        ]
        
        for difficulty, acc in sorted(metrics.accuracy_by_difficulty.items()):
            report_lines.append(f"  {difficulty}: {acc:.3f}")
        
        report_lines.extend([
            "",
            "Accuracy by Concept Count (QSS):"
        ])
        
        for n_concepts, acc in sorted(metrics.accuracy_by_concept_count.items()):
            report_lines.append(f"  {n_concepts} concepts: {acc:.3f}")
        
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            
            # Also save JSON metrics
            json_path = output_path.replace('.txt', '.json')
            with open(json_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
        
        return report


def load_medqa_dataset(path: str) -> List[QAExample]:
    """
    Load a medical QA dataset (MedQA, PubMedQA, etc.)
    
    Expected format:
    [
        {
            "question": "...",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "answer": "A",
            "medical_concepts": ["concept1", "concept2"]
        },
        ...
    ]
    """
    examples = []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        question = item.get("question", "")
        options = item.get("options", {})
        answer = item.get("answer", "")
        concepts = item.get("medical_concepts", [])
        difficulty = item.get("difficulty", "basic")
        
        # Convert options dict to list
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        answer_choices = [options.get(l, "") for l in labels if l in options]
        
        # Find correct answer index
        correct_idx = -1
        if answer in labels:
            correct_idx = labels.index(answer)
        
        example = QAExample(
            question=question,
            answer_choices=answer_choices,
            correct_answer=options.get(answer, answer),
            correct_answer_idx=correct_idx,
            question_type="multi-choice",
            medical_concepts=concepts,
            difficulty=difficulty
        )
        examples.append(example)
    
    return examples


def create_sample_evaluation_dataset() -> List[QAExample]:
    """
    Create a sample evaluation dataset for testing.
    Based on mental disorder questions.
    """
    examples = [
        QAExample(
            question="What is the primary neurotransmitter implicated in major depressive disorder?",
            answer_choices=["Dopamine", "Serotonin", "GABA", "Acetylcholine"],
            correct_answer="Serotonin",
            correct_answer_idx=1,
            question_type="multi-choice",
            medical_concepts=["depression", "neurotransmitter", "serotonin"],
            difficulty="basic"
        ),
        QAExample(
            question="Which of the following is a first-line treatment for schizophrenia?",
            answer_choices=["SSRIs", "Benzodiazepines", "Atypical antipsychotics", "Opioids"],
            correct_answer="Atypical antipsychotics",
            correct_answer_idx=2,
            question_type="multi-choice",
            medical_concepts=["schizophrenia", "antipsychotic", "treatment"],
            difficulty="basic"
        ),
        QAExample(
            question="What brain region is most associated with emotional regulation in anxiety disorders?",
            answer_choices=["Cerebellum", "Amygdala", "Motor cortex", "Occipital lobe"],
            correct_answer="Amygdala",
            correct_answer_idx=1,
            question_type="multi-choice",
            medical_concepts=["anxiety", "amygdala", "brain region", "emotional regulation"],
            difficulty="intermediate"
        ),
        QAExample(
            question="Which genetic variant has been most strongly associated with increased risk of bipolar disorder?",
            answer_choices=["APOE ε4", "CACNA1C", "HTT", "CFTR"],
            correct_answer="CACNA1C",
            correct_answer_idx=1,
            question_type="multi-choice",
            medical_concepts=["bipolar disorder", "genetics", "CACNA1C", "risk factor", "calcium channel"],
            difficulty="expert"
        ),
    ]
    
    return examples


if __name__ == "__main__":
    # Demo usage
    print("Graph RAG Evaluation Module")
    print("Based on KGARevion paper methodology")
    print()
    
    # Create sample dataset
    examples = create_sample_evaluation_dataset()
    print(f"Sample dataset contains {len(examples)} examples")
    
    for i, ex in enumerate(examples):
        print(f"\n{i+1}. {ex.question[:60]}...")
        print(f"   Difficulty: {ex.difficulty}")
        print(f"   Concepts: {', '.join(ex.medical_concepts[:3])}...")
