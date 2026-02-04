"""
Graph RAG for Mental Disorder Knowledge Graph (MDKG) - KGARevion Implementation

This module implements a Knowledge Graph-based Retrieval-Augmented Generation (Graph RAG)
system strictly following the KGARevion paper (https://arxiv.org/abs/2410.04660).

Key Algorithm Components (from paper):
1. Generate Action: Extract medical concepts and generate triplets from query
2. Review Action: Verify triplets using fine-tuned LLM with KG embeddings (TransE)
3. Revise Action: Iteratively correct rejected triplets
4. Answer Action: Generate final answer based on verified triplets

Optimization (from GraphRAG paper https://arxiv.org/abs/2404.16130):
- Community Detection using Leiden algorithm for efficient triplet search

Author: MDKG Project
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from collections import defaultdict
import logging

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from graspologic.partition import leiden
    HAS_LEIDEN = True
except ImportError:
    try:
        import community as community_louvain
        HAS_LEIDEN = False
        HAS_LOUVAIN = True
    except ImportError:
        HAS_LEIDEN = False
        HAS_LOUVAIN = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class TripletStatus(Enum):
    """Triplet validation status following KGARevion paper"""
    TRUE = "true"  # Verified as correct
    FALSE = "false"  # Factually wrong (entities mappable but LLM says False)
    INCOMPLETE = "incomplete"  # Entities cannot be mapped to KG (kept)


@dataclass
class Triplet:
    """
    Represents a knowledge graph triplet (head, relation, tail).
    
    Compatible with triplets_refine_prompt.py format:
    - head_entity / tail_entity as aliases for head / tail
    - head_type / tail_type for entity types
    - sentence for source context
    - confidence for backward compatibility (论文使用 True/False 二元分类)
    
    Following KGARevion paper:
    - status indicates validation result (TRUE/FALSE/INCOMPLETE)
    """
    head_entity: str
    head_type: str
    relation: str
    tail_entity: str
    tail_type: str
    sentence: str = ""
    confidence: float = 1.0
    status: TripletStatus = TripletStatus.INCOMPLETE
    source: str = "generated"  # 'generated', 'kg', 'revised'
    head_mapped: bool = False  # Can head be mapped to KG?
    tail_mapped: bool = False  # Can tail be mapped to KG?
    community_id: Optional[int] = None  # Community for optimization
    
    # Aliases for backward compatibility
    @property
    def head(self) -> str:
        return self.head_entity
    
    @property
    def tail(self) -> str:
        return self.tail_entity
    
    def to_dict(self) -> Dict:
        return {
            "head_entity": self.head_entity,
            "head_type": self.head_type,
            "relation": self.relation,
            "tail_entity": self.tail_entity,
            "tail_type": self.tail_type,
            "sentence": self.sentence,
            "confidence": self.confidence,
            "status": self.status.value,
            "source": self.source,
            "head_mapped": self.head_mapped,
            "tail_mapped": self.tail_mapped,
            "community_id": self.community_id
        }
    
    def __hash__(self):
        return hash((self.head_entity.lower(), self.relation.lower(), self.tail_entity.lower()))
    
    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return (self.head_entity.lower() == other.head_entity.lower() and 
                self.relation.lower() == other.relation.lower() and 
                self.tail_entity.lower() == other.tail_entity.lower())


@dataclass
class QueryResult:
    """Complete result of a Graph RAG query"""
    query: str
    answer: str
    true_triplets: List[Triplet] = field(default_factory=list)  # V set in paper
    false_triplets: List[Triplet] = field(default_factory=list)  # F set in paper
    incomplete_triplets: List[Triplet] = field(default_factory=list)  # Kept triplets
    revised_triplets: List[Triplet] = field(default_factory=list)
    medical_concepts: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)


# =============================================================================
# LLM Backends
# =============================================================================

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


# =============================================================================
# Community Detection (GraphRAG optimization)
# =============================================================================

class CommunityManager:
    """
    Community detection for KG optimization.
    
    Based on GraphRAG paper (From Local to Global):
    - Uses Leiden algorithm for hierarchical community detection
    - Communities provide efficient search scope
    - Reduces search complexity from O(|KG|) to O(|Community|)
    """
    
    def __init__(self):
        self.graph = None
        self.communities: Dict[int, Set[str]] = {}  # community_id -> entity names
        self.entity_to_community: Dict[str, int] = {}  # entity -> community_id
        self.hierarchy_level: int = 0
    
    def build_graph_from_triplets(self, triplets: List[Triplet]):
        """
        Build a NetworkX graph from triplets for community detection.
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX not installed. Community detection disabled.")
            return
        
        self.graph = nx.Graph()
        
        for triplet in triplets:
            head = triplet.head.lower()
            tail = triplet.tail.lower()
            
            # Add nodes with type information
            self.graph.add_node(head, node_type=triplet.head_type or "unknown")
            self.graph.add_node(tail, node_type=triplet.tail_type or "unknown")
            
            # Add edge with relation as attribute
            if self.graph.has_edge(head, tail):
                # Multiple relations between same entities
                self.graph[head][tail]['relations'].append(triplet.relation)
            else:
                self.graph.add_edge(head, tail, relations=[triplet.relation])
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def detect_communities(self, resolution: float = 1.0):
        """
        Detect communities using Leiden algorithm.
        
        Following GraphRAG paper:
        - Hierarchical community detection
        - Each level provides different granularity
        
        Args:
            resolution: Leiden resolution parameter (higher = more communities)
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            logger.warning("No graph to detect communities in.")
            return
        
        if HAS_LEIDEN:
            # Use graspologic's Leiden implementation
            try:
                partition = leiden(self.graph, resolution=resolution)
                self._process_partition(partition)
            except Exception as e:
                logger.warning(f"Leiden failed: {e}. Falling back to Louvain.")
                self._fallback_to_louvain()
        elif HAS_LOUVAIN:
            self._fallback_to_louvain()
        else:
            logger.warning("No community detection library available. Using single community.")
            self._single_community_fallback()
    
    def _fallback_to_louvain(self):
        """Fallback to Louvain algorithm if Leiden not available."""
        try:
            partition = community_louvain.best_partition(self.graph)
            self._process_partition_dict(partition)
        except Exception as e:
            logger.warning(f"Louvain failed: {e}. Using single community.")
            self._single_community_fallback()
    
    def _process_partition(self, partition):
        """Process Leiden partition result."""
        self.communities = defaultdict(set)
        for node, community_id in enumerate(partition):
            node_name = list(self.graph.nodes())[node]
            self.communities[community_id].add(node_name)
            self.entity_to_community[node_name] = community_id
    
    def _process_partition_dict(self, partition: Dict):
        """Process partition dictionary (from Louvain)."""
        self.communities = defaultdict(set)
        for node, community_id in partition.items():
            self.communities[community_id].add(node)
            self.entity_to_community[node] = community_id
        
        logger.info(f"Detected {len(self.communities)} communities")
    
    def _single_community_fallback(self):
        """Put all entities in a single community."""
        self.communities = {0: set(self.graph.nodes()) if self.graph else set()}
        for node in (self.graph.nodes() if self.graph else []):
            self.entity_to_community[node] = 0
    
    def get_community_for_entity(self, entity: str) -> Optional[int]:
        """Get the community ID for an entity."""
        return self.entity_to_community.get(entity.lower())
    
    def get_entities_in_community(self, community_id: int) -> Set[str]:
        """Get all entities in a community."""
        return self.communities.get(community_id, set())
    
    def get_related_communities(self, entities: List[str]) -> Set[int]:
        """
        Get all communities related to a set of entities.
        Used for scoping triplet search.
        """
        communities = set()
        for entity in entities:
            comm_id = self.get_community_for_entity(entity)
            if comm_id is not None:
                communities.add(comm_id)
        return communities


# =============================================================================
# Knowledge Graph Manager
# =============================================================================

class KnowledgeGraphManager:
    """
    Manages the Mental Disorder Knowledge Graph.
    
    Key features (following KGARevion paper):
    1. Entity mapping using UMLS-like codes
    2. TransE-style embeddings for entities (simplified version)
    3. Community-based search optimization (from GraphRAG paper)
    """
    
    # Mental disorder specific relation types
    RELATION_TYPES = [
        "causes", "treats", "associated_with", "symptom_of", "risk_factor_for",
        "comorbid_with", "contraindicated_with", "side_effect_of", "biomarker_for",
        "affects", "located_in", "interacts_with", "inhibits", "activates",
        "diagnoses", "prevents", "worsens", "improves", "phenotype_of"
    ]
    
    # Relation description templates (following paper's D(r) dictionary)
    RELATION_DESCRIPTIONS = {
        "causes": "{head} causes {tail}",
        "treats": "{head} is used to treat {tail}",
        "associated_with": "{head} is associated with {tail}",
        "symptom_of": "{head} is a symptom of {tail}",
        "risk_factor_for": "{head} is a risk factor for {tail}",
        "comorbid_with": "{head} is comorbid with {tail}",
        "contraindicated_with": "{head} is contraindicated with {tail}",
        "side_effect_of": "{head} is a side effect of {tail}",
        "biomarker_for": "{head} is a biomarker for {tail}",
        "affects": "{head} affects {tail}",
        "located_in": "{head} is located in {tail}",
        "interacts_with": "{head} interacts with {tail}",
        "inhibits": "{head} inhibits {tail}",
        "activates": "{head} activates {tail}",
        "diagnoses": "{head} diagnoses {tail}",
        "prevents": "{head} prevents {tail}",
        "worsens": "{head} worsens {tail}",
        "improves": "{head} improves {tail}",
        "phenotype_of": "{head} is a phenotype of {tail}"
    }
    
    def __init__(self, config: Dict[str, str], use_community_detection: bool = True):
        """
        Initialize the Knowledge Graph Manager.
        
        Args:
            config: Dictionary containing paths
            use_community_detection: Whether to use community-based optimization
        """
        self.config = config
        self.entities: Dict[str, Dict] = {}  # Entity to linked info
        self.triplets: List[Triplet] = []
        self.entity_index: Dict[str, Set[int]] = {}  # entity -> triplet indices
        self.entity_embeddings: Dict[str, np.ndarray] = {}  # Simplified TransE embeddings
        
        # Community detection for optimization
        self.use_community_detection = use_community_detection
        self.community_manager = CommunityManager() if use_community_detection else None
        
        self._load_data()
        
        if use_community_detection and self.triplets:
            self._build_communities()
    
    def _load_data(self):
        """Load knowledge graph data from files."""
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
        
        # Initialize embeddings (simplified TransE-style)
        self._initialize_embeddings()
    
    def _extract_triplets_from_predictions(self, predictions_path: str):
        """Extract triplets from NER&RE model predictions."""
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
                sentence = sample.get('sents', ' '.join(tokens))
                
                if head_id in entity_lookup and tail_id in entity_lookup:
                    head_info = entity_lookup[head_id]
                    tail_info = entity_lookup[tail_id]
                    
                    triplet = Triplet(
                        head_entity=head_info['span'],
                        head_type=head_info['type'],
                        relation=rel_type,
                        tail_entity=tail_info['span'],
                        tail_type=tail_info['type'],
                        sentence=sentence,
                        source='kg'
                    )
                    self.triplets.append(triplet)
    
    def _build_entity_index(self):
        """Build index for fast entity lookup."""
        for idx, triplet in enumerate(self.triplets):
            head_lower = triplet.head.lower()
            tail_lower = triplet.tail.lower()
            
            if head_lower not in self.entity_index:
                self.entity_index[head_lower] = set()
            self.entity_index[head_lower].add(idx)
            
            if tail_lower not in self.entity_index:
                self.entity_index[tail_lower] = set()
            self.entity_index[tail_lower].add(idx)
    
    def _initialize_embeddings(self, dim: int = 128):
        """
        Initialize entity embeddings (simplified TransE-style).
        
        In the full paper implementation, these would be:
        1. Trained using TransE on the full KG
        2. Aligned with LLM token embeddings using attention + FFN
        
        This is a simplified version using random embeddings for demonstration.
        """
        all_entities = set()
        for triplet in self.triplets:
            all_entities.add(triplet.head.lower())
            all_entities.add(triplet.tail.lower())
        
        for entity in all_entities:
            self.entity_embeddings[entity] = np.random.randn(dim).astype(np.float32)
            self.entity_embeddings[entity] /= np.linalg.norm(self.entity_embeddings[entity])
    
    def _build_communities(self):
        """Build community structure for optimized search."""
        if self.community_manager and self.triplets:
            self.community_manager.build_graph_from_triplets(self.triplets)
            self.community_manager.detect_communities()
            
            # Assign community IDs to triplets
            for triplet in self.triplets:
                head_comm = self.community_manager.get_community_for_entity(triplet.head)
                tail_comm = self.community_manager.get_community_for_entity(triplet.tail)
                # Use head's community as primary
                triplet.community_id = head_comm if head_comm is not None else tail_comm
    
    def can_map_entity(self, entity: str) -> bool:
        """
        Check if an entity can be mapped to the KG.
        
        Following paper: uses UMLS codes for mapping.
        Here we use entity linking results as a proxy.
        """
        entity_lower = entity.lower()
        
        # Direct match in linked entities
        if entity_lower in self.entities or entity in self.entities:
            return True
        
        # Check if entity appears in KG triplets
        if entity_lower in self.entity_index:
            return True
        
        # Fuzzy match
        for e in self.entities.keys():
            if entity_lower in e.lower() or e.lower() in entity_lower:
                return True
        
        return False
    
    def get_entity_embedding(self, entity: str) -> Optional[np.ndarray]:
        """Get the embedding for an entity."""
        return self.entity_embeddings.get(entity.lower())
    
    def find_triplets_in_communities(
        self, 
        entities: List[str], 
        max_results: int = 50
    ) -> List[Triplet]:
        """
        Find triplets efficiently using community-based search.
        
        This is the key optimization from GraphRAG paper:
        - First find communities containing query entities
        - Then only search within those communities
        """
        if not self.use_community_detection or not self.community_manager:
            return self.find_related_triplets(entities, max_results)
        
        # Get relevant communities
        communities = self.community_manager.get_related_communities(entities)
        
        if not communities:
            # Fallback to direct search
            return self.find_related_triplets(entities, max_results)
        
        # Search only in relevant communities
        results = []
        for triplet in self.triplets:
            if triplet.community_id in communities:
                for entity in entities:
                    entity_lower = entity.lower()
                    if (entity_lower in triplet.head.lower() or 
                        entity_lower in triplet.tail.lower()):
                        results.append(triplet)
                        break
            
            if len(results) >= max_results:
                break
        
        return results
    
    def find_related_triplets(
        self, 
        entities: List[str], 
        max_results: int = 20
    ) -> List[Triplet]:
        """Find triplets related to any of the given entities."""
        all_triplets = []
        seen = set()
        
        for entity in entities:
            entity_lower = entity.lower()
            
            # Direct match
            if entity_lower in self.entity_index:
                indices = list(self.entity_index[entity_lower])[:max_results]
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        all_triplets.append(self.triplets[idx])
            
            # Partial match
            for key, indices in self.entity_index.items():
                if entity_lower in key or key in entity_lower:
                    for idx in list(indices)[:5]:
                        if idx not in seen:
                            seen.add(idx)
                            all_triplets.append(self.triplets[idx])
                
                if len(all_triplets) >= max_results:
                    break
            
            if len(all_triplets) >= max_results:
                break
        
        return all_triplets[:max_results]
    
    def get_relation_description(self, triplet: Triplet) -> str:
        """
        Get the natural language description for a relation.
        Following paper's D(r) dictionary.
        """
        template = self.RELATION_DESCRIPTIONS.get(
            triplet.relation.lower(),
            "{head} is related to {tail}"
        )
        return template.format(head=triplet.head, tail=triplet.tail)


# =============================================================================
# KGARevion Agent
# =============================================================================

class KGARevionAgent:
    """
    Graph RAG Agent following KGARevion paper.
    
    Four Key Actions:
    1. Generate: Extract concepts and generate triplets
    2. Review: Verify triplets (True/False) using KG
    3. Revise: Correct False triplets
    4. Answer: Generate final answer from True triplets
    """
    
    def __init__(
        self,
        kg_manager: KnowledgeGraphManager,
        llm_backend: LLMBackend,
        max_revise_rounds: int = 2
    ):
        """
        Initialize the KGARevion Agent.
        
        Args:
            kg_manager: Knowledge Graph Manager instance
            llm_backend: LLM backend for text generation
            max_revise_rounds: Maximum rounds of revision (k in paper)
        """
        self.kg = kg_manager
        self.llm = llm_backend
        self.max_revise_rounds = max_revise_rounds
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[^{}]*\}',
            r'\[[^\[\]]*\]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    # =========================================================================
    # ACTION 1: Generate
    # =========================================================================
    
    def _generate_action(
        self, 
        query: str,
        answer_choices: Optional[List[str]] = None
    ) -> Tuple[List[str], List[Triplet]]:
        """
        Generate Action: Extract medical concepts and generate triplets.
        
        Following paper Section 3.1:
        - For choice-aware: generate triplets per answer candidate
        - For non-choice-aware (Yes/No): generate from question only
        """
        # Determine question type
        is_choice_aware = answer_choices and not all(
            c.lower() in ['yes', 'no', 'maybe'] for c in answer_choices
        )
        
        if is_choice_aware:
            return self._generate_choice_aware(query, answer_choices)
        else:
            return self._generate_non_choice_aware(query)
    
    def _generate_choice_aware(
        self, 
        query: str, 
        answer_choices: List[str]
    ) -> Tuple[List[str], List[Triplet]]:
        """
        Generate triplets for choice-aware questions.
        Following paper: generate triplets based on each answer candidate.
        """
        all_concepts = set()
        all_triplets = []
        
        # First extract medical concepts from question
        concepts_prompt = f"""Extract all medical terminologies from this question.
Question: "{query}"

Return JSON: {{"medical_terminologies": ["term1", "term2", ...]}}"""
        
        concepts_response = self.llm.generate(concepts_prompt)
        concepts_result = self._extract_json_from_response(concepts_response)
        if concepts_result:
            all_concepts.update(concepts_result.get("medical_terminologies", []))
        
        # Generate triplets for each answer candidate
        for choice in answer_choices:
            triplet_prompt = f"""Generate knowledge triplets for this question and answer.

Question: "{query}"
Answer candidate: "{choice}"
Medical concepts: {list(all_concepts)}

Generate triplets that could help verify if this answer is correct.
Use relations: {self.kg.RELATION_TYPES[:10]}

Return JSON: {{
    "triplets": [
        {{"head": "entity1", "relation": "relation_type", "tail": "entity2"}},
        ...
    ]
}}"""
            
            triplet_response = self.llm.generate(triplet_prompt)
            triplet_result = self._extract_json_from_response(triplet_response)
            
            if triplet_result:
                for t in triplet_result.get("triplets", []):
                    triplet = Triplet(
                        head_entity=t.get("head", ""),
                        head_type=t.get("head_type", "unknown"),
                        relation=t.get("relation", "associated_with"),
                        tail_entity=t.get("tail", ""),
                        tail_type=t.get("tail_type", "unknown"),
                        source="generated"
                    )
                    if triplet.head_entity and triplet.tail_entity:
                        all_triplets.append(triplet)
        
        return list(all_concepts), all_triplets
    
    def _generate_non_choice_aware(self, query: str) -> Tuple[List[str], List[Triplet]]:
        """
        Generate triplets for non-choice-aware questions (Yes/No/Maybe).
        Following paper: generate from question stem only.
        """
        prompt = f"""Analyze this medical question and generate relevant knowledge triplets.

Question: "{query}"

Instructions:
1. Extract all medical entities (diseases, symptoms, drugs, genes, brain regions, etc.)
2. Generate triplets that represent relevant medical knowledge
3. Use relations from: {self.kg.RELATION_TYPES[:10]}

Return JSON: {{
    "medical_concepts": ["concept1", "concept2", ...],
    "triplets": [
        {{"head": "entity1", "relation": "relation_type", "tail": "entity2"}},
        ...
    ]
}}"""
        
        response = self.llm.generate(prompt)
        result = self._extract_json_from_response(response)
        
        concepts = []
        triplets = []
        
        if result:
            concepts = result.get("medical_concepts", [])
            for t in result.get("triplets", []):
                triplet = Triplet(
                    head_entity=t.get("head", ""),
                    head_type=t.get("head_type", "unknown"),
                    relation=t.get("relation", "associated_with"),
                    tail_entity=t.get("tail", ""),
                    tail_type=t.get("tail_type", "unknown"),
                    source="generated"
                )
                if triplet.head_entity and triplet.tail_entity:
                    triplets.append(triplet)
        
        return concepts, triplets
    
    # =========================================================================
    # ACTION 2: Review
    # =========================================================================
    
    def _review_action(self, triplets: List[Triplet]) -> Tuple[List[Triplet], List[Triplet], List[Triplet]]:
        """
        Review Action: Verify triplets against the knowledge graph.
        
        Following paper Section 3.2:
        - First check if entities can be mapped to KG
        - Then use LLM to judge True/False
        
        Returns:
            Tuple of (true_triplets, false_triplets, incomplete_triplets)
        """
        true_triplets = []  # V set
        false_triplets = []  # F set
        incomplete_triplets = []  # Kept due to incomplete KG
        
        for triplet in triplets:
            # Step 1: Check if entities can be mapped (paper's UMLS mapping)
            triplet.head_mapped = self.kg.can_map_entity(triplet.head)
            triplet.tail_mapped = self.kg.can_map_entity(triplet.tail)
            
            if triplet.head_mapped and triplet.tail_mapped:
                # Both entities mappable -> verify with LLM
                is_true = self._verify_triplet_with_llm(triplet)
                
                if is_true:
                    triplet.status = TripletStatus.TRUE
                    true_triplets.append(triplet)
                else:
                    triplet.status = TripletStatus.FALSE
                    false_triplets.append(triplet)
            else:
                # Cannot fully map -> incomplete knowledge, keep it
                triplet.status = TripletStatus.INCOMPLETE
                incomplete_triplets.append(triplet)
        
        return true_triplets, false_triplets, incomplete_triplets
    
    def _verify_triplet_with_llm(self, triplet: Triplet) -> bool:
        """
        Verify a triplet using LLM.
        
        Following paper Section 3.2.2:
        - Get relation description D(r)
        - Use fine-tuned LLM to output True/False
        
        Note: This is a simplified version. The full paper uses:
        1. TransE embeddings aligned with LLM embeddings
        2. LoRA fine-tuned LLM on KG completion task
        """
        # Get relation description (paper's D(r))
        relation_desc = self.kg.get_relation_description(triplet)
        
        # Find supporting evidence from KG
        related_triplets = self.kg.find_related_triplets([triplet.head, triplet.tail], max_results=5)
        
        evidence_strs = []
        for t in related_triplets:
            evidence_strs.append(f"({t.head}, {t.relation}, {t.tail})")
        
        prompt = f"""Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity.

Triplet to verify: ({triplet.head}, {triplet.relation}, {triplet.tail})
Description: {relation_desc}

Supporting evidence from knowledge graph:
{chr(10).join(evidence_strs) if evidence_strs else "No direct evidence found."}

Please determine the correctness of the triple and respond with True or False.
Output only 'True' or 'False'."""
        
        response = self.llm.generate(prompt, temperature=0.0, max_tokens=10)
        
        # Parse True/False response
        response_lower = response.lower().strip()
        return 'true' in response_lower and 'false' not in response_lower
    
    # =========================================================================
    # ACTION 3: Revise
    # =========================================================================
    
    def _revise_action(self, false_triplets: List[Triplet], query: str) -> List[Triplet]:
        """
        Revise Action: Correct false triplets.
        
        Following paper Section 3.3 and Appendix D.1:
        - Provide instruction to LLM to revise incorrect triplets
        - LLM modifies head or tail entity
        """
        if not false_triplets:
            return []
        
        # Format triplets for revision
        triplet_strs = []
        for t in false_triplets[:5]:  # Limit to 5
            triplet_strs.append(f"({t.head}, {t.relation}, {t.tail})")
        
        prompt = f"""### Instruction:
Given the following triplets consisting of a head entity, relation, and tail entity, please review and revise the triplets to ensure they are correct and helpful for answering the given question. The revision should focus on correcting the head entity, relation, or tail entity as needed to make the triplet accurate and relevant.

Only return the revised triplets in JSON format with the key 'Revised_Triplets' and the value as the corrected triplets.

### Input:
Triplets: {chr(10).join(triplet_strs)}

Questions: {query}

### Response:"""
        
        response = self.llm.generate(prompt)
        result = self._extract_json_from_response(response)
        
        revised_triplets = []
        if result:
            revised_list = result.get("Revised_Triplets", result.get("revised_triplets", []))
            for t in revised_list:
                if isinstance(t, dict):
                    triplet = Triplet(
                        head_entity=t.get("head", ""),
                        head_type=t.get("head_type", "unknown"),
                        relation=t.get("relation", "associated_with"),
                        tail_entity=t.get("tail", ""),
                        tail_type=t.get("tail_type", "unknown"),
                        source="revised"
                    )
                elif isinstance(t, (list, tuple)) and len(t) >= 3:
                    triplet = Triplet(
                        head_entity=str(t[0]),
                        head_type="unknown",
                        relation=str(t[1]),
                        tail_entity=str(t[2]),
                        tail_type="unknown",
                        source="revised"
                    )
                else:
                    continue
                
                if triplet.head_entity and triplet.tail_entity:
                    revised_triplets.append(triplet)
        
        return revised_triplets
    
    # =========================================================================
    # ACTION 4: Answer
    # =========================================================================
    
    def _answer_action(
        self,
        query: str,
        true_triplets: List[Triplet],
        incomplete_triplets: List[Triplet],
        medical_concepts: List[str]
    ) -> str:
        """
        Answer Action: Generate final answer based on verified triplets.
        
        Following paper Section 3.3:
        - Use True triplets and incomplete triplets (kept due to KG limitations)
        - Generate comprehensive answer
        """
        # Combine true and incomplete triplets for context
        all_valid = true_triplets + incomplete_triplets
        
        triplet_strs = []
        for t in all_valid[:15]:
            triplet_strs.append(f"- {t.head} {t.relation} {t.tail}")
        
        # Get additional context from KG
        context_triplets = self.kg.find_triplets_in_communities(
            medical_concepts, max_results=10
        )
        context_strs = []
        for t in context_triplets:
            context_strs.append(f"- {t.head} {t.relation} {t.tail}")
        
        prompt = f"""You are a mental health expert assistant.

Question: "{query}"

Verified Knowledge Triplets:
{chr(10).join(triplet_strs) if triplet_strs else "No verified triplets available."}

Additional Context from Knowledge Graph:
{chr(10).join(context_strs) if context_strs else "No additional context."}

Based on the above knowledge, provide a comprehensive and accurate answer.
If knowledge is insufficient, acknowledge limitations.
Always recommend consulting healthcare professionals for medical advice.

Answer:"""
        
        answer = self.llm.generate(prompt, temperature=0.3, max_tokens=1024)
        
        return answer
    
    # =========================================================================
    # Main Query Method
    # =========================================================================
    
    def query(
        self,
        question: str,
        answer_choices: Optional[List[str]] = None,
        verbose: bool = False
    ) -> QueryResult:
        """
        Process a query using the KGARevion pipeline.
        
        Args:
            question: User's question
            answer_choices: Optional answer candidates for multi-choice
            verbose: Whether to print progress
            
        Returns:
            QueryResult with answer and triplet details
        """
        result = QueryResult(query=question)
        
        # ACTION 1: Generate
        if verbose:
            logger.info("Action 1: Generating triplets...")
        
        result.medical_concepts, generated_triplets = self._generate_action(
            question, answer_choices
        )
        result.reasoning_trace.append(
            f"Generated {len(generated_triplets)} triplets from {len(result.medical_concepts)} concepts"
        )
        
        # ACTION 2: Review
        if verbose:
            logger.info(f"Action 2: Reviewing {len(generated_triplets)} triplets...")
        
        true_triplets, false_triplets, incomplete_triplets = self._review_action(
            generated_triplets
        )
        result.true_triplets = true_triplets
        result.false_triplets = false_triplets
        result.incomplete_triplets = incomplete_triplets
        
        result.reasoning_trace.append(
            f"Review: {len(true_triplets)} true, {len(false_triplets)} false, {len(incomplete_triplets)} incomplete"
        )
        
        # ACTION 3: Revise (if there are false triplets)
        for round_num in range(self.max_revise_rounds):
            if not false_triplets:
                break
            
            if verbose:
                logger.info(f"Action 3: Revise round {round_num + 1}...")
            
            revised = self._revise_action(false_triplets, question)
            result.revised_triplets.extend(revised)
            
            if revised:
                # Review revised triplets
                new_true, new_false, new_incomplete = self._review_action(revised)
                true_triplets.extend(new_true)
                incomplete_triplets.extend(new_incomplete)
                false_triplets = new_false
                
                result.reasoning_trace.append(
                    f"Revise round {round_num + 1}: {len(new_true)} newly verified"
                )
        
        result.true_triplets = true_triplets
        result.false_triplets = false_triplets
        result.incomplete_triplets = incomplete_triplets
        
        # ACTION 4: Answer
        if verbose:
            logger.info("Action 4: Generating answer...")
        
        result.answer = self._answer_action(
            question,
            true_triplets,
            incomplete_triplets,
            result.medical_concepts
        )
        
        result.reasoning_trace.append(
            f"Final: {len(result.true_triplets)} true triplets used"
        )
        
        return result


# =============================================================================
# Factory Function
# =============================================================================

def create_kgarevion_agent(
    config_path: Optional[str] = None,
    llm_type: str = "openai",
    llm_model: str = "gpt-4",
    api_key: Optional[str] = None,
    use_community_detection: bool = True,
    **kwargs
) -> KGARevionAgent:
    """
    Factory function to create a KGARevion Agent.
    
    Args:
        config_path: Path to configuration JSON file
        llm_type: Type of LLM backend
        llm_model: Model name
        api_key: API key for OpenAI
        use_community_detection: Enable community-based optimization
        
    Returns:
        Configured KGARevionAgent instance
    """
    # Determine base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Default configuration
    default_config = {
        "entity_linking_path": os.path.join(
            base_dir, "models", "InputsAndOutputs", "output", "entity_linking_results.json"
        ),
        "predictions_path": os.path.join(
            base_dir, "models", "InputsAndOutputs", "output", "sampling_json_run_v1_sampled.json"
        ),
    }
    
    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            default_config.update(custom_config)
    
    # Initialize Knowledge Graph Manager
    kg_manager = KnowledgeGraphManager(
        default_config, 
        use_community_detection=use_community_detection
    )
    
    # Initialize LLM Backend
    if llm_type == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required.")
        llm_backend = OpenAIBackend(
            api_key=api_key, 
            model=llm_model, 
            base_url=kwargs.get("base_url")
        )
    elif llm_type == "ollama":
        llm_backend = OllamaBackend(
            model=llm_model, 
            base_url=kwargs.get("ollama_url", "http://localhost:11434")
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    return KGARevionAgent(
        kg_manager=kg_manager,
        llm_backend=llm_backend,
        max_revise_rounds=kwargs.get("max_revise_rounds", 2)
    )


# Alias for backward compatibility
MentalDisorderGraphRAG = KGARevionAgent
create_graph_rag = create_kgarevion_agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KGARevion Agent for Mental Disorder QA")
    parser.add_argument("--query", "-q", type=str, help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--llm-type", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--no-community", action="store_true", help="Disable community detection")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    try:
        agent = create_kgarevion_agent(
            llm_type=args.llm_type,
            llm_model=args.model,
            use_community_detection=not args.no_community
        )
        
        if args.interactive:
            print("\n" + "=" * 60)
            print("KGARevion Agent for Mental Disorder Knowledge Graph")
            print("=" * 60)
            print("Type 'quit' to exit\n")
            
            while True:
                question = input("Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                result = agent.query(question, verbose=args.verbose)
                print(f"\nAnswer:\n{result.answer}")
                print(f"\nTrue triplets: {len(result.true_triplets)}")
                print(f"Reasoning: {result.reasoning_trace}")
                print()
        
        elif args.query:
            result = agent.query(args.query, verbose=args.verbose)
            print(f"\nAnswer:\n{result.answer}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
