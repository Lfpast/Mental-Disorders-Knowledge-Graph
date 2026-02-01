"""
MDKG Data Loader for Drug Repurposing
======================================

This module handles loading and processing of MDKG knowledge graph data
for link prediction and drug repurposing tasks.

Features:
- Load triplets from MDKG extraction pipeline
- Integrate entity linking results (UMLS, HPO, GO, MONDO, UBERON)
- Build heterogeneous graph structure
- Create train/valid/test splits with disease-centric evaluation
"""

import os
import json
import pickle
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("Warning: DGL not available. Install with: pip install dgl")


@dataclass
class EntityInfo:
    """Entity information from knowledge graph."""
    entity_id: str
    entity_type: str
    span: str
    ontology_id: Optional[str] = None
    ontology_source: Optional[str] = None
    ontology_term: Optional[str] = None


@dataclass
class RelationInfo:
    """Relation information from knowledge graph."""
    relation_type: str
    head_entity: EntityInfo
    tail_entity: EntityInfo
    confidence: float = 1.0


class MDKGDataLoader:
    """
    Data loader for MDKG knowledge graph.
    
    Loads triplets extracted from medical literature and builds
    a heterogeneous graph for link prediction tasks.
    
    Example:
        >>> loader = MDKGDataLoader(data_folder="./models/InputsAndOutputs")
        >>> loader.load_data()
        >>> loader.prepare_split(split='mental_health', seed=42)
        >>> G = loader.get_graph()
    """
    
    # Entity type mapping for graph construction (from DPKG_types_Cor4.json)
    ENTITY_TYPE_MAP = {
        # Core types from DPKG_types_Cor4.json
        'drug': 'drug',
        'disease': 'disease',
        'gene': 'gene',
        'signs': 'signs',
        'symptom': 'symptom',
        'Health_factors': 'health_factors',
        'method': 'method',
        'physiology': 'physiology',
        'region': 'region',
        # Additional types that may appear in data
        'anatomy': 'anatomy',
        'pathway': 'pathway',
        'protein': 'protein',
    }
    
    # Relation type mapping - based on DPKG_types_Cor4.json
    # All 9 relation types from the schema
    RELATION_TYPE_MAP = {
        # Drug-disease relations (primary for drug repurposing)
        'treatment_for': ('drug', 'treats', 'disease'),
        # Location/anatomical relations
        'occurs_in': ('entity', 'occurs_in', 'entity'),
        'located_in': ('entity', 'located_in', 'entity'),
        # Diagnostic/clinical relations
        'help_diagnose': ('entity', 'diagnoses', 'entity'),
        'risk_factor_of': ('entity', 'risk_of', 'entity'),
        # General associations
        'associated_with': ('entity', 'associated_with', 'entity'),
        'characteristic_of': ('entity', 'characteristic_of', 'entity'),
        # Semantic relations
        'abbreviation_for': ('entity', 'abbreviation_for', 'entity'),
        'hyponym_of': ('entity', 'hyponym_of', 'entity'),
        # Extended relations (may appear in data)
        'causes': ('entity', 'causes', 'entity'),
        'prevents': ('drug', 'prevents', 'disease'),
        'interacts_with': ('drug', 'interacts_with', 'drug'),
        'contraindicated_for': ('drug', 'contraindicated_for', 'disease'),
    }
    
    # Drug repurposing relevant relation types (for prediction focus)
    # Primary: treatment_for
    # Secondary: risk_factor_of, help_diagnose, associated_with (when drug-disease)
    DD_RELATION_TYPES = ['treats', 'prevents', 'contraindicated_for']
    
    # All relation types that could be relevant for drug repurposing analysis
    REPURPOSING_RELEVANT_RELATIONS = [
        'treatment_for',  # Direct treatment
        'risk_factor_of',  # Risk factors may suggest preventive drugs
        'associated_with',  # General association could indicate drug effects
        'characteristic_of',  # Disease characteristics drug may target
    ]
    
    def __init__(
        self,
        data_folder: str = "./models/InputsAndOutputs",
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize MDKG data loader.
        
        Args:
            data_folder: Path to InputsAndOutputs folder
            config_path: Optional path to configuration file
            cache_dir: Directory for caching processed data
        """
        self.data_folder = data_folder
        self.config_path = config_path
        self.cache_dir = cache_dir or os.path.join(data_folder, "cache", "prediction")
        
        # Paths
        self.triplets_path = os.path.join(
            data_folder, "output", "sampling_json_run_v1_sampled.json"
        )
        self.entity_linking_path = os.path.join(
            data_folder, "output", "entity_linking_results.json"
        )
        
        # Data storage
        self.entities: Dict[str, EntityInfo] = {}
        self.relations: List[RelationInfo] = []
        self.entity_linking: Dict[str, Dict] = {}
        
        # Entity indices
        self.entity2idx: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.idx2entity: Dict[str, Dict[int, str]] = defaultdict(dict)
        
        # Relation indices
        self.relation2idx: Dict[str, int] = {}
        self.idx2relation: Dict[int, str] = {}
        
        # Graph
        self.G: Optional[Any] = None
        
        # Data splits
        self.df_train: List[Dict] = []
        self.df_valid: List[Dict] = []
        self.df_test: List[Dict] = []
        
        # Split info
        self.split: Optional[str] = None
        self.disease_eval_idx: List[int] = []
        
        # Statistics
        self.stats: Dict[str, Any] = {}
        
    def load_data(
        self,
        triplets_path: Optional[str] = None,
        entity_linking_path: Optional[str] = None,
        use_cache: bool = True
    ) -> None:
        """
        Load MDKG data from files.
        
        Args:
            triplets_path: Override path to triplets file
            entity_linking_path: Override path to entity linking file
            use_cache: Whether to use cached processed data
        """
        triplets_path = triplets_path or self.triplets_path
        entity_linking_path = entity_linking_path or self.entity_linking_path
        
        # Check for cache
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        if use_cache and os.path.exists(cache_file):
            print("Loading from cache...")
            self._load_cache(cache_file)
            return
        
        print("Loading MDKG data...")
        
        # Load entity linking results
        if os.path.exists(entity_linking_path):
            print(f"  Loading entity linking from {entity_linking_path}")
            with open(entity_linking_path, 'r', encoding='utf-8') as f:
                self.entity_linking = json.load(f)
            print(f"  Loaded {len(self.entity_linking)} entity mappings")
        
        # Load triplets
        if os.path.exists(triplets_path):
            print(f"  Loading triplets from {triplets_path}")
            with open(triplets_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            self._process_samples(samples)
        else:
            raise FileNotFoundError(f"Triplets file not found: {triplets_path}")
        
        # Build entity indices
        self._build_entity_indices()
        
        # Print statistics
        self._compute_statistics()
        
        # Save cache
        if use_cache:
            self._save_cache(cache_file)
    
    def _process_samples(self, samples: List[Dict]) -> None:
        """Process extraction samples to extract entities and relations."""
        entity_spans_seen: Set[Tuple[str, str]] = set()
        
        for sample in samples:
            entities_in_sample: Dict[str, EntityInfo] = {}
            
            # Process entities
            for ent in sample.get('entities', []):
                span = ent['span'].lower()
                ent_type = self.ENTITY_TYPE_MAP.get(ent['type'], ent['type'])
                
                # Get ontology mapping
                onto_info = self.entity_linking.get(ent['span'], {})
                
                entity_info = EntityInfo(
                    entity_id=f"{ent_type}:{span}",
                    entity_type=ent_type,
                    span=span,
                    ontology_id=onto_info.get('entity_id'),
                    ontology_source=onto_info.get('entity_ontology'),
                    ontology_term=onto_info.get('entity_onto_term')
                )
                
                entities_in_sample[str(ent['entity_id_in_sentence'])] = entity_info
                
                # Add to global entities
                key = (ent_type, span)
                if key not in entity_spans_seen:
                    entity_spans_seen.add(key)
                    self.entities[entity_info.entity_id] = entity_info
            
            # Process relations
            for rel in sample.get('relations', []):
                head_idx = str(rel['head'])
                tail_idx = str(rel['tail'])
                
                if head_idx in entities_in_sample and tail_idx in entities_in_sample:
                    head_entity = entities_in_sample[head_idx]
                    tail_entity = entities_in_sample[tail_idx]
                    
                    relation_info = RelationInfo(
                        relation_type=rel['type'],
                        head_entity=head_entity,
                        tail_entity=tail_entity
                    )
                    self.relations.append(relation_info)
        
        print(f"  Processed {len(self.entities)} unique entities")
        print(f"  Processed {len(self.relations)} relations")
    
    def _build_entity_indices(self) -> None:
        """Build entity to index mappings for each entity type."""
        # Group entities by type
        entities_by_type: Dict[str, List[str]] = defaultdict(list)
        for entity_id, entity in self.entities.items():
            entities_by_type[entity.entity_type].append(entity_id)
        
        # Build indices
        for entity_type, entity_ids in entities_by_type.items():
            for idx, entity_id in enumerate(sorted(entity_ids)):
                self.entity2idx[entity_type][entity_id] = idx
                self.idx2entity[entity_type][idx] = entity_id
        
        # Build relation indices
        relation_types = set()
        for rel in self.relations:
            # Determine canonical relation type based on entity types
            head_type = rel.head_entity.entity_type
            tail_type = rel.tail_entity.entity_type
            rel_type = rel.relation_type
            
            # Create canonical edge type
            canonical_rel = f"{head_type}_{rel_type}_{tail_type}"
            relation_types.add(canonical_rel)
        
        for idx, rel_type in enumerate(sorted(relation_types)):
            self.relation2idx[rel_type] = idx
            self.idx2relation[idx] = rel_type
    
    def _compute_statistics(self) -> None:
        """Compute and print dataset statistics."""
        self.stats = {
            'num_entities': len(self.entities),
            'num_relations': len(self.relations),
            'entity_types': {},
            'relation_types': {},
        }
        
        # Count entities by type
        for entity in self.entities.values():
            etype = entity.entity_type
            self.stats['entity_types'][etype] = self.stats['entity_types'].get(etype, 0) + 1
        
        # Count relations by type
        for rel in self.relations:
            rtype = rel.relation_type
            self.stats['relation_types'][rtype] = self.stats['relation_types'].get(rtype, 0) + 1
        
        print("\nDataset Statistics:")
        print(f"  Total entities: {self.stats['num_entities']}")
        print(f"  Total relations: {self.stats['num_relations']}")
        print("\n  Entities by type:")
        for etype, count in sorted(self.stats['entity_types'].items(), key=lambda x: -x[1]):
            print(f"    {etype}: {count}")
        print("\n  Relations by type:")
        for rtype, count in sorted(self.stats['relation_types'].items(), key=lambda x: -x[1]):
            print(f"    {rtype}: {count}")
    
    def build_graph(self) -> Any:
        """
        Build DGL heterogeneous graph from loaded data.
        
        Returns:
            DGL heterogeneous graph
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for graph construction. Install with: pip install dgl")
        
        print("\nBuilding heterogeneous graph...")
        
        # Collect edges by canonical edge type
        edges_by_type: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = defaultdict(
            lambda: ([], [])
        )
        
        for rel in self.relations:
            head_type = rel.head_entity.entity_type
            tail_type = rel.tail_entity.entity_type
            rel_type = rel.relation_type
            
            head_id = rel.head_entity.entity_id
            tail_id = rel.tail_entity.entity_id
            
            # Get indices
            if head_id in self.entity2idx[head_type] and tail_id in self.entity2idx[tail_type]:
                head_idx = self.entity2idx[head_type][head_id]
                tail_idx = self.entity2idx[tail_type][tail_id]
                
                etype = (head_type, rel_type, tail_type)
                edges_by_type[etype][0].append(head_idx)
                edges_by_type[etype][1].append(tail_idx)
                
                # Add reverse edge
                rev_etype = (tail_type, f"rev_{rel_type}", head_type)
                edges_by_type[rev_etype][0].append(tail_idx)
                edges_by_type[rev_etype][1].append(head_idx)
        
        # Create graph data
        graph_data = {}
        for etype, (src, dst) in edges_by_type.items():
            if len(src) > 0:
                graph_data[etype] = (torch.tensor(src), torch.tensor(dst))
        
        # Number of nodes per type
        num_nodes_dict = {
            ntype: len(self.entity2idx[ntype])
            for ntype in self.entity2idx.keys()
        }
        
        # Create DGL graph
        self.G = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        
        print(f"  Created graph with {self.G.num_nodes()} nodes and {self.G.num_edges()} edges")
        print(f"  Node types: {len(self.G.ntypes)}, Edge types: {len(self.G.canonical_etypes)}")
        
        return self.G
    
    def prepare_split(
        self,
        split: str = 'random',
        seed: int = 42,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> None:
        """
        Prepare train/valid/test splits.
        
        Args:
            split: Split strategy ('random', 'mental_health', 'drug_centric', 'disease_centric')
            seed: Random seed
            train_ratio: Training set ratio
            valid_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        print(f"\nPreparing {split} split (seed={seed})...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.split = split
        
        # Filter drug-disease relations for prediction
        dd_relations = [
            rel for rel in self.relations
            if rel.relation_type in ['treatment_for', 'contraindicated_for', 'prevents']
            and rel.head_entity.entity_type == 'drug'
            and rel.tail_entity.entity_type == 'disease'
        ]
        
        print(f"  Found {len(dd_relations)} drug-disease relations for prediction")
        
        if split == 'mental_health':
            # Split by mental health related diseases
            mental_health_keywords = [
                'depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd',
                'ocd', 'adhd', 'autism', 'dementia', 'alzheimer', 'parkinson',
                'psychosis', 'mania', 'panic', 'phobia', 'stress'
            ]
            
            def is_mental_health(disease_span: str) -> bool:
                return any(kw in disease_span.lower() for kw in mental_health_keywords)
            
            mh_relations = [r for r in dd_relations if is_mental_health(r.tail_entity.span)]
            other_relations = [r for r in dd_relations if not is_mental_health(r.tail_entity.span)]
            
            print(f"  Mental health relations: {len(mh_relations)}")
            print(f"  Other relations: {len(other_relations)}")
            
            # Mental health goes to test, others to train/valid
            random.shuffle(mh_relations)
            random.shuffle(other_relations)
            
            n_train = int(len(other_relations) * 0.9)
            self.df_train = self._relations_to_df(other_relations[:n_train])
            self.df_valid = self._relations_to_df(other_relations[n_train:])
            self.df_test = self._relations_to_df(mh_relations)
            
        elif split == 'disease_centric':
            # Group relations by disease for disease-centric evaluation
            disease_relations: Dict[str, List[RelationInfo]] = defaultdict(list)
            for rel in dd_relations:
                disease_relations[rel.tail_entity.entity_id].append(rel)
            
            diseases = list(disease_relations.keys())
            random.shuffle(diseases)
            
            n_train = int(len(diseases) * train_ratio)
            n_valid = int(len(diseases) * valid_ratio)
            
            train_diseases = set(diseases[:n_train])
            valid_diseases = set(diseases[n_train:n_train + n_valid])
            test_diseases = set(diseases[n_train + n_valid:])
            
            self.disease_eval_idx = [
                self.entity2idx['disease'].get(d)
                for d in test_diseases
                if d in self.entity2idx.get('disease', {})
            ]
            
            train_rels = [r for r in dd_relations if r.tail_entity.entity_id in train_diseases]
            valid_rels = [r for r in dd_relations if r.tail_entity.entity_id in valid_diseases]
            test_rels = [r for r in dd_relations if r.tail_entity.entity_id in test_diseases]
            
            self.df_train = self._relations_to_df(train_rels)
            self.df_valid = self._relations_to_df(valid_rels)
            self.df_test = self._relations_to_df(test_rels)
            
        else:  # random split
            random.shuffle(dd_relations)
            
            n_train = int(len(dd_relations) * train_ratio)
            n_valid = int(len(dd_relations) * valid_ratio)
            
            self.df_train = self._relations_to_df(dd_relations[:n_train])
            self.df_valid = self._relations_to_df(dd_relations[n_train:n_train + n_valid])
            self.df_test = self._relations_to_df(dd_relations[n_train + n_valid:])
        
        print(f"  Train: {len(self.df_train)} triplets")
        print(f"  Valid: {len(self.df_valid)} triplets")
        print(f"  Test: {len(self.df_test)} triplets")
    
    def _relations_to_df(self, relations: List[RelationInfo]) -> List[Dict]:
        """Convert relations to dataframe-like structure."""
        df = []
        for rel in relations:
            head_type = rel.head_entity.entity_type
            tail_type = rel.tail_entity.entity_type
            
            head_idx = self.entity2idx[head_type].get(rel.head_entity.entity_id)
            tail_idx = self.entity2idx[tail_type].get(rel.tail_entity.entity_id)
            
            if head_idx is not None and tail_idx is not None:
                df.append({
                    'x_idx': head_idx,
                    'x_type': head_type,
                    'x_id': rel.head_entity.entity_id,
                    'x_name': rel.head_entity.span,
                    'relation': rel.relation_type,
                    'y_idx': tail_idx,
                    'y_type': tail_type,
                    'y_id': rel.tail_entity.entity_id,
                    'y_name': rel.tail_entity.span,
                })
        return df
    
    def get_graph(self) -> Any:
        """Get the constructed DGL graph."""
        if self.G is None:
            self.build_graph()
        return self.G
    
    def get_drug_disease_etypes(self) -> List[Tuple[str, str, str]]:
        """Get drug-disease edge types for prediction."""
        dd_etypes = []
        if self.G is not None:
            for etype in self.G.canonical_etypes:
                src, rel, dst = etype
                if src == 'drug' and dst == 'disease':
                    dd_etypes.append(etype)
                elif src == 'disease' and dst == 'drug':
                    dd_etypes.append(etype)
        return dd_etypes
    
    def get_entity_name(self, entity_type: str, idx: int) -> str:
        """Get entity name from type and index."""
        entity_id = self.idx2entity[entity_type].get(idx)
        if entity_id and entity_id in self.entities:
            return self.entities[entity_id].span
        return f"Unknown-{entity_type}-{idx}"
    
    def get_entity_ontology(self, entity_type: str, idx: int) -> Optional[Dict]:
        """Get ontology information for an entity."""
        entity_id = self.idx2entity[entity_type].get(idx)
        if entity_id and entity_id in self.entities:
            entity = self.entities[entity_id]
            return {
                'ontology_id': entity.ontology_id,
                'ontology_source': entity.ontology_source,
                'ontology_term': entity.ontology_term
            }
        return None
    
    def _save_cache(self, cache_file: str) -> None:
        """Save processed data to cache."""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        cache_data = {
            'entities': self.entities,
            'relations': self.relations,
            'entity_linking': self.entity_linking,
            'entity2idx': dict(self.entity2idx),
            'idx2entity': dict(self.idx2entity),
            'relation2idx': self.relation2idx,
            'idx2relation': self.idx2relation,
            'stats': self.stats,
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"\nSaved cache to {cache_file}")
    
    def _load_cache(self, cache_file: str) -> None:
        """Load processed data from cache."""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.entities = cache_data['entities']
        self.relations = cache_data['relations']
        self.entity_linking = cache_data['entity_linking']
        self.entity2idx = defaultdict(dict, cache_data['entity2idx'])
        self.idx2entity = defaultdict(dict, cache_data['idx2entity'])
        self.relation2idx = cache_data['relation2idx']
        self.idx2relation = cache_data['idx2relation']
        self.stats = cache_data['stats']
        
        print(f"Loaded {len(self.entities)} entities and {len(self.relations)} relations from cache")


def test_data_loader():
    """Test the data loader."""
    import sys
    
    # Get data folder from command line or use default
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    else:
        data_folder = "./models/InputsAndOutputs"
    
    print(f"Testing MDKGDataLoader with data folder: {data_folder}")
    
    loader = MDKGDataLoader(data_folder=data_folder)
    loader.load_data(use_cache=False)
    
    if DGL_AVAILABLE:
        G = loader.build_graph()
        loader.prepare_split(split='random', seed=42)
        
        print("\nSample train triplets:")
        for triplet in loader.df_train[:3]:
            print(f"  {triplet['x_name']} --[{triplet['relation']}]--> {triplet['y_name']}")


if __name__ == "__main__":
    test_data_loader()
