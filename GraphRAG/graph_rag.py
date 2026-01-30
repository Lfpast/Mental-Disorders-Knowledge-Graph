"""
Graph RAG for Mental Disorder Knowledge Graph (MDKG)

This module implements a Knowledge Graph-based Retrieval-Augmented Generation (Graph RAG)
system for mental health question answering. The implementation is inspired by the 
KGARevion paper (https://arxiv.org/abs/2410.04660), which introduces a knowledge graph-based 
agent for biomedical QA.

The system operates through four key actions:
1. Generate: Extract medical concepts and generate relevant triplets from the query
2. Review: Verify triplets against the grounded knowledge graph
3. Revise: Correct erroneous triplets iteratively
4. Answer: Generate the final answer based on verified triplets

Author: MDKG Project
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """Represents a knowledge graph triplet (head, relation, tail)"""
    head: str
    relation: str
    tail: str
    head_type: Optional[str] = None
    tail_type: Optional[str] = None
    confidence: float = 1.0
    source: str = "generated"  # 'generated', 'kg', 'revised'
    
    def to_dict(self) -> Dict:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "confidence": self.confidence,
            "source": self.source
        }
    
    def __hash__(self):
        return hash((self.head.lower(), self.relation.lower(), self.tail.lower()))
    
    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return (self.head.lower() == other.head.lower() and 
                self.relation.lower() == other.relation.lower() and 
                self.tail.lower() == other.tail.lower())


@dataclass
class ReviewResult:
    """Result of reviewing a triplet"""
    triplet: Triplet
    is_valid: bool
    kg_evidence: Optional[List[Dict]] = None
    reasoning: str = ""


@dataclass
class QueryResult:
    """Complete result of a Graph RAG query"""
    query: str
    answer: str
    verified_triplets: List[Triplet] = field(default_factory=list)
    rejected_triplets: List[Triplet] = field(default_factory=list)
    revised_triplets: List[Triplet] = field(default_factory=list)
    medical_concepts: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    confidence: float = 0.0


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """Generate text from prompt"""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.model = model
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a biomedical knowledge expert specializing in mental disorders, psychiatry, and neuroscience."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend"""
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"You are a biomedical knowledge expert specializing in mental disorders.\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        )
        return response.json().get("response", "")


class HuggingFaceBackend(LLMBackend):
    """HuggingFace Transformers backend for local models"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class KnowledgeGraphManager:
    """
    Manages the Mental Disorder Knowledge Graph
    Loads triplets and entity linking results from MDKG outputs
    """
    
    # Mental disorder specific relation types
    RELATION_TYPES = [
        "causes", "treats", "associated_with", "symptom_of", "risk_factor_for",
        "comorbid_with", "contraindicated_with", "side_effect_of", "biomarker_for",
        "affects", "located_in", "interacts_with", "inhibits", "activates",
        "diagnoses", "prevents", "worsens", "improves", "phenotype_of"
    ]
    
    # Mental disorder entity types
    ENTITY_TYPES = [
        "Disease", "Symptom", "Drug", "Gene", "Protein", "Brain_Region",
        "Neurotransmitter", "Treatment", "Risk_Factor", "Biomarker",
        "Phenotype", "Biological_Process", "Anatomical_Structure"
    ]
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize the Knowledge Graph Manager
        
        Args:
            config: Dictionary containing paths:
                - entity_linking_path: Path to entity linking results
                - triplets_path: Path to extracted triplets
                - predictions_path: Path to model predictions (sampling output)
        """
        self.config = config
        self.entities: Dict[str, Dict] = {}
        self.triplets: List[Triplet] = []
        self.entity_index: Dict[str, Set[int]] = {}  # entity -> triplet indices
        
        self._load_data()
    
    def _load_data(self):
        """Load knowledge graph data from files"""
        # Load entity linking results
        entity_linking_path = self.config.get('entity_linking_path')
        if entity_linking_path and os.path.exists(entity_linking_path):
            logger.info(f"Loading entity linking results from {entity_linking_path}")
            with open(entity_linking_path, 'r', encoding='utf-8') as f:
                self.entities = json.load(f)
            logger.info(f"Loaded {len(self.entities)} linked entities")
        
        # Load triplets from predictions
        predictions_path = self.config.get('predictions_path')
        if predictions_path and os.path.exists(predictions_path):
            logger.info(f"Loading predictions from {predictions_path}")
            self._extract_triplets_from_predictions(predictions_path)
            logger.info(f"Extracted {len(self.triplets)} triplets")
        
        # Build entity index for fast lookup
        self._build_entity_index()
    
    def _extract_triplets_from_predictions(self, predictions_path: str):
        """Extract triplets from NER&RE model predictions"""
        with open(predictions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for sample in data:
            tokens = sample.get('tokens', [])
            entities = sample.get('entities', [])
            relations = sample.get('relations', [])
            
            # Build entity lookup
            entity_lookup = {}
            for e in entities:
                entity_id = e.get('entity_id', e.get('id'))
                if entity_id is not None:
                    start, end = e.get('start', 0), e.get('end', 0)
                    span = ' '.join(tokens[start:end]) if tokens else e.get('span', '')
                    entity_lookup[entity_id] = {
                        'span': span,
                        'type': e.get('type', 'Unknown')
                    }
            
            # Extract triplets from relations
            for rel in relations:
                head_id = rel.get('head_id', rel.get('head'))
                tail_id = rel.get('tail_id', rel.get('tail'))
                rel_type = rel.get('type', 'associated_with')
                
                if head_id in entity_lookup and tail_id in entity_lookup:
                    head_info = entity_lookup[head_id]
                    tail_info = entity_lookup[tail_id]
                    
                    triplet = Triplet(
                        head=head_info['span'],
                        relation=rel_type,
                        tail=tail_info['span'],
                        head_type=head_info['type'],
                        tail_type=tail_info['type'],
                        source='kg'
                    )
                    self.triplets.append(triplet)
    
    def _build_entity_index(self):
        """Build index for fast entity lookup"""
        for idx, triplet in enumerate(self.triplets):
            head_lower = triplet.head.lower()
            tail_lower = triplet.tail.lower()
            
            if head_lower not in self.entity_index:
                self.entity_index[head_lower] = set()
            self.entity_index[head_lower].add(idx)
            
            if tail_lower not in self.entity_index:
                self.entity_index[tail_lower] = set()
            self.entity_index[tail_lower].add(idx)
    
    def get_entity_info(self, entity: str) -> Optional[Dict]:
        """Get linked entity information"""
        # Try exact match first
        if entity in self.entities:
            return self.entities[entity]
        
        # Try case-insensitive match
        entity_lower = entity.lower()
        for e, info in self.entities.items():
            if e.lower() == entity_lower:
                return info
        
        return None
    
    def find_related_triplets(self, entity: str, max_results: int = 20) -> List[Triplet]:
        """Find triplets related to an entity"""
        entity_lower = entity.lower()
        related_triplets = []
        
        # Direct match
        if entity_lower in self.entity_index:
            indices = list(self.entity_index[entity_lower])[:max_results]
            related_triplets.extend([self.triplets[i] for i in indices])
        
        # Partial match (for compound terms)
        for key, indices in self.entity_index.items():
            if entity_lower in key or key in entity_lower:
                for idx in list(indices)[:5]:
                    if self.triplets[idx] not in related_triplets:
                        related_triplets.append(self.triplets[idx])
                        if len(related_triplets) >= max_results:
                            break
            if len(related_triplets) >= max_results:
                break
        
        return related_triplets[:max_results]
    
    def verify_triplet(self, triplet: Triplet) -> Tuple[bool, List[Dict]]:
        """
        Verify if a triplet exists in or is consistent with the KG
        
        Returns:
            Tuple of (is_valid, supporting_evidence)
        """
        head_lower = triplet.head.lower()
        tail_lower = triplet.tail.lower()
        
        supporting_evidence = []
        
        # Check if both entities exist
        head_info = self.get_entity_info(triplet.head)
        tail_info = self.get_entity_info(triplet.tail)
        
        # Find related triplets for verification
        head_triplets = self.find_related_triplets(triplet.head, max_results=10)
        tail_triplets = self.find_related_triplets(triplet.tail, max_results=10)
        
        # Check for exact or similar triplet matches
        for kg_triplet in head_triplets + tail_triplets:
            if (kg_triplet.head.lower() == head_lower and 
                kg_triplet.tail.lower() == tail_lower):
                supporting_evidence.append({
                    "type": "exact_match",
                    "triplet": kg_triplet.to_dict()
                })
            elif (kg_triplet.tail.lower() == head_lower and 
                  kg_triplet.head.lower() == tail_lower):
                supporting_evidence.append({
                    "type": "reverse_match",
                    "triplet": kg_triplet.to_dict()
                })
            elif (head_lower in kg_triplet.head.lower() or 
                  head_lower in kg_triplet.tail.lower() or
                  tail_lower in kg_triplet.head.lower() or 
                  tail_lower in kg_triplet.tail.lower()):
                supporting_evidence.append({
                    "type": "partial_match",
                    "triplet": kg_triplet.to_dict()
                })
        
        # Entity linking evidence
        if head_info:
            supporting_evidence.append({
                "type": "entity_linked",
                "entity": triplet.head,
                "linked_info": head_info
            })
        if tail_info:
            supporting_evidence.append({
                "type": "entity_linked",
                "entity": triplet.tail,
                "linked_info": tail_info
            })
        
        # Determine validity based on evidence
        is_valid = len(supporting_evidence) > 0
        
        return is_valid, supporting_evidence
    
    def get_context_triplets(self, concepts: List[str], max_per_concept: int = 5) -> List[Triplet]:
        """Get relevant triplets for a list of concepts"""
        all_triplets = []
        seen = set()
        
        for concept in concepts:
            related = self.find_related_triplets(concept, max_results=max_per_concept)
            for t in related:
                if t not in seen:
                    seen.add(t)
                    all_triplets.append(t)
        
        return all_triplets


class MentalDisorderGraphRAG:
    """
    Graph RAG Agent for Mental Disorder Knowledge Graph
    
    Implements the KGARevion-style approach with four key actions:
    - Generate: Extract medical concepts and generate relevant triplets
    - Review: Verify triplets against the grounded knowledge graph
    - Revise: Correct erroneous triplets
    - Answer: Generate the final answer based on verified triplets
    """
    
    def __init__(
        self, 
        kg_manager: KnowledgeGraphManager,
        llm_backend: LLMBackend,
        max_revise_rounds: int = 2,
        triplet_confidence_threshold: float = 0.6
    ):
        """
        Initialize the Graph RAG Agent
        
        Args:
            kg_manager: Knowledge Graph Manager instance
            llm_backend: LLM backend for text generation
            max_revise_rounds: Maximum rounds of triplet revision
            triplet_confidence_threshold: Minimum confidence for accepting triplets
        """
        self.kg = kg_manager
        self.llm = llm_backend
        self.max_revise_rounds = max_revise_rounds
        self.confidence_threshold = triplet_confidence_threshold
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        # Try to find JSON in the response
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\[[^\[\]]*\]',  # JSON array
            r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
            r'```\s*([\s\S]*?)\s*```',  # Generic code block
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _generate_action(self, query: str) -> Tuple[List[str], List[Triplet]]:
        """
        Generate Action: Extract medical concepts and generate relevant triplets
        
        Args:
            query: User's question about mental disorders
            
        Returns:
            Tuple of (medical_concepts, generated_triplets)
        """
        prompt = f"""You are a biomedical knowledge extraction expert specializing in mental disorders.

## Task:
Analyze the following question and:
1. Extract all medical concepts (diseases, symptoms, drugs, genes, brain regions, etc.)
2. Generate relevant knowledge triplets that could help answer the question

## Question:
"{query}"

## Instructions:
- Identify specific medical entities mentioned or implied
- Generate triplets in format (head_entity, relation, tail_entity)
- Relations should be one of: causes, treats, associated_with, symptom_of, risk_factor_for, 
  comorbid_with, contraindicated_with, side_effect_of, biomarker_for, affects, 
  located_in, interacts_with, inhibits, activates, diagnoses, prevents, worsens, improves

## Response Format (JSON):
{{
    "medical_concepts": ["concept1", "concept2", ...],
    "triplets": [
        {{"head": "entity1", "relation": "relation_type", "tail": "entity2", "confidence": 0.0-1.0}},
        ...
    ]
}}

Provide your response:"""

        response = self.llm.generate(prompt)
        result = self._extract_json_from_response(response)
        
        medical_concepts = []
        triplets = []
        
        if result:
            medical_concepts = result.get("medical_concepts", [])
            for t in result.get("triplets", []):
                triplet = Triplet(
                    head=t.get("head", ""),
                    relation=t.get("relation", "associated_with"),
                    tail=t.get("tail", ""),
                    confidence=t.get("confidence", 0.8),
                    source="generated"
                )
                if triplet.head and triplet.tail:
                    triplets.append(triplet)
        
        return medical_concepts, triplets
    
    def _review_action(self, triplets: List[Triplet]) -> Tuple[List[Triplet], List[Triplet]]:
        """
        Review Action: Verify triplets against the knowledge graph
        
        Args:
            triplets: List of triplets to review
            
        Returns:
            Tuple of (verified_triplets, rejected_triplets)
        """
        verified = []
        rejected = []
        
        for triplet in triplets:
            is_valid, evidence = self.kg.verify_triplet(triplet)
            
            if is_valid:
                # Boost confidence based on KG evidence
                if any(e['type'] == 'exact_match' for e in evidence):
                    triplet.confidence = min(1.0, triplet.confidence + 0.3)
                elif any(e['type'] == 'reverse_match' for e in evidence):
                    triplet.confidence = min(1.0, triplet.confidence + 0.2)
                elif any(e['type'] == 'entity_linked' for e in evidence):
                    triplet.confidence = min(1.0, triplet.confidence + 0.1)
                
                if triplet.confidence >= self.confidence_threshold:
                    verified.append(triplet)
                else:
                    rejected.append(triplet)
            else:
                # Mark for potential revision
                triplet.confidence *= 0.5
                rejected.append(triplet)
        
        return verified, rejected
    
    def _revise_action(self, rejected_triplets: List[Triplet], query: str) -> List[Triplet]:
        """
        Revise Action: Attempt to correct rejected triplets
        
        Args:
            rejected_triplets: Triplets that failed review
            query: Original query for context
            
        Returns:
            List of revised triplets
        """
        if not rejected_triplets:
            return []
        
        # Format rejected triplets for the prompt
        triplet_strs = []
        for t in rejected_triplets[:5]:  # Limit to 5 triplets
            triplet_strs.append(f"({t.head}, {t.relation}, {t.tail})")
        
        prompt = f"""You are a biomedical knowledge correction expert.

## Original Question:
"{query}"

## Triplets to Revise:
The following triplets were identified as potentially incorrect or unverifiable:
{chr(10).join(triplet_strs)}

## Instructions:
Revise these triplets to make them more accurate and specific to mental disorders.
- Correct entity names to standard medical terminology
- Ensure relations are appropriate and specific
- Only include triplets you are confident about

## Response Format (JSON):
{{
    "revised_triplets": [
        {{"head": "corrected_entity1", "relation": "relation", "tail": "corrected_entity2", "confidence": 0.0-1.0}},
        ...
    ]
}}

Provide your revised triplets:"""

        response = self.llm.generate(prompt)
        result = self._extract_json_from_response(response)
        
        revised_triplets = []
        if result:
            for t in result.get("revised_triplets", []):
                triplet = Triplet(
                    head=t.get("head", ""),
                    relation=t.get("relation", "associated_with"),
                    tail=t.get("tail", ""),
                    confidence=t.get("confidence", 0.7),
                    source="revised"
                )
                if triplet.head and triplet.tail:
                    revised_triplets.append(triplet)
        
        return revised_triplets
    
    def _answer_action(
        self, 
        query: str, 
        verified_triplets: List[Triplet],
        medical_concepts: List[str],
        context_triplets: List[Triplet]
    ) -> str:
        """
        Answer Action: Generate the final answer based on verified knowledge
        
        Args:
            query: User's question
            verified_triplets: Triplets verified by the Review action
            medical_concepts: Extracted medical concepts
            context_triplets: Additional context from KG
            
        Returns:
            Generated answer string
        """
        # Format triplets for context
        verified_strs = []
        for t in verified_triplets:
            verified_strs.append(f"- {t.head} {t.relation} {t.tail} (confidence: {t.confidence:.2f})")
        
        context_strs = []
        for t in context_triplets[:10]:  # Limit context triplets
            context_strs.append(f"- {t.head} {t.relation} {t.tail}")
        
        prompt = f"""You are a mental health expert assistant. Answer the following question using the provided knowledge.

## Question:
"{query}"

## Verified Medical Knowledge:
{chr(10).join(verified_strs) if verified_strs else "No specific verified triplets available."}

## Additional Context from Knowledge Graph:
{chr(10).join(context_strs) if context_strs else "No additional context available."}

## Related Medical Concepts:
{', '.join(medical_concepts) if medical_concepts else "None identified."}

## Instructions:
1. Provide a comprehensive, accurate answer based on the available knowledge
2. If the knowledge is insufficient, acknowledge limitations
3. Always emphasize consulting healthcare professionals for medical advice
4. Use clear, professional language appropriate for medical information
5. Structure your answer with relevant details and explanations

## Answer:"""

        answer = self.llm.generate(prompt, temperature=0.3, max_tokens=1024)
        
        # Add disclaimer if not present
        if "consult" not in answer.lower() and "healthcare" not in answer.lower():
            answer += "\n\n*Note: This information is for educational purposes. Please consult a healthcare professional for medical advice.*"
        
        return answer
    
    def query(self, question: str, verbose: bool = False) -> QueryResult:
        """
        Process a mental health query using Graph RAG
        
        Args:
            question: User's question about mental disorders
            verbose: Whether to print progress information
            
        Returns:
            QueryResult containing the answer and reasoning trace
        """
        result = QueryResult(query=question)
        
        if verbose:
            logger.info(f"Processing query: {question[:100]}...")
        
        # Step 1: Generate Action
        if verbose:
            logger.info("Step 1: Generating medical concepts and triplets...")
        
        result.medical_concepts, generated_triplets = self._generate_action(question)
        result.reasoning_trace.append(
            f"Generated {len(generated_triplets)} triplets from {len(result.medical_concepts)} concepts"
        )
        
        # Step 2: Review Action
        if verbose:
            logger.info(f"Step 2: Reviewing {len(generated_triplets)} triplets...")
        
        verified_triplets, rejected_triplets = self._review_action(generated_triplets)
        result.verified_triplets = verified_triplets
        result.rejected_triplets = rejected_triplets
        result.reasoning_trace.append(
            f"Verified {len(verified_triplets)}, rejected {len(rejected_triplets)} triplets"
        )
        
        # Step 3: Revise Action (if needed)
        if rejected_triplets and self.max_revise_rounds > 0:
            if verbose:
                logger.info(f"Step 3: Revising {len(rejected_triplets)} rejected triplets...")
            
            for round_num in range(self.max_revise_rounds):
                revised = self._revise_action(rejected_triplets, question)
                if revised:
                    # Review revised triplets
                    newly_verified, still_rejected = self._review_action(revised)
                    verified_triplets.extend(newly_verified)
                    result.revised_triplets.extend(revised)
                    result.reasoning_trace.append(
                        f"Round {round_num + 1}: Revised {len(revised)}, verified {len(newly_verified)}"
                    )
                    rejected_triplets = still_rejected
                    
                    if not still_rejected:
                        break
        
        # Get additional context from KG
        context_triplets = self.kg.get_context_triplets(result.medical_concepts, max_per_concept=5)
        
        # Step 4: Answer Action
        if verbose:
            logger.info("Step 4: Generating answer...")
        
        result.answer = self._answer_action(
            question, 
            verified_triplets, 
            result.medical_concepts,
            context_triplets
        )
        
        # Calculate overall confidence
        if verified_triplets:
            result.confidence = sum(t.confidence for t in verified_triplets) / len(verified_triplets)
        else:
            result.confidence = 0.3  # Base confidence when no triplets verified
        
        result.reasoning_trace.append(f"Final confidence: {result.confidence:.2f}")
        
        return result


def create_graph_rag(
    config_path: Optional[str] = None,
    llm_type: str = "openai",
    llm_model: str = "gpt-4",
    api_key: Optional[str] = None,
    **kwargs
) -> MentalDisorderGraphRAG:
    """
    Factory function to create a Graph RAG instance
    
    Args:
        config_path: Path to configuration JSON file
        llm_type: Type of LLM backend ('openai', 'ollama', 'huggingface')
        llm_model: Model name for the LLM backend
        api_key: API key for OpenAI (or set OPENAI_API_KEY env var)
        **kwargs: Additional arguments for components
        
    Returns:
        Configured MentalDisorderGraphRAG instance
    """
    # Determine base directory (go up one level from GraphRAG folder)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Default configuration
    default_config = {
        "entity_linking_path": os.path.join(base_dir, "models", "InputsAndOutputs", "output", "entity_linking_results.json"),
        "predictions_path": os.path.join(base_dir, "models", "InputsAndOutputs", "output", "sampling_json_run_v1_sampled.json"),
        "triplets_path": os.path.join(base_dir, "models", "InputsAndOutputs", "output", "triplet_refinement", "triplets_for_refinement.json")
    }
    
    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            default_config.update(custom_config)
    
    # Initialize Knowledge Graph Manager
    kg_manager = KnowledgeGraphManager(default_config)
    
    # Initialize LLM Backend
    if llm_type == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter.")
        llm_backend = OpenAIBackend(api_key=api_key, model=llm_model, base_url=kwargs.get("base_url"))
    elif llm_type == "ollama":
        llm_backend = OllamaBackend(model=llm_model, base_url=kwargs.get("ollama_url", "http://localhost:11434"))
    elif llm_type == "huggingface":
        llm_backend = HuggingFaceBackend(model_name=llm_model)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    # Create and return Graph RAG instance
    return MentalDisorderGraphRAG(
        kg_manager=kg_manager,
        llm_backend=llm_backend,
        max_revise_rounds=kwargs.get("max_revise_rounds", 2),
        triplet_confidence_threshold=kwargs.get("confidence_threshold", 0.6)
    )


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mental Disorder Graph RAG")
    parser.add_argument("--query", "-q", type=str, help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--llm-type", type=str, default="openai", choices=["openai", "ollama", "huggingface"])
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        rag = create_graph_rag(
            llm_type=args.llm_type,
            llm_model=args.model,
            api_key=args.api_key
        )
        
        if args.interactive:
            print("\n" + "="*60)
            print("Mental Disorder Knowledge Graph RAG")
            print("="*60)
            print("Ask questions about mental health (type 'quit' to exit)\n")
            
            while True:
                question = input("Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                result = rag.query(question, verbose=args.verbose)
                print(f"\n{'='*40}")
                print("Answer:")
                print(result.answer)
                print(f"\nConfidence: {result.confidence:.2f}")
                if args.verbose:
                    print(f"Verified triplets: {len(result.verified_triplets)}")
                    print(f"Reasoning: {result.reasoning_trace}")
                print(f"{'='*40}\n")
        
        elif args.query:
            result = rag.query(args.query, verbose=args.verbose)
            print(f"\nAnswer:\n{result.answer}")
            print(f"\nConfidence: {result.confidence:.2f}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
