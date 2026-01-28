"""
Optimized Entity Linking Script - Skip UMLS processing for faster results

This module implements an `EntityLinker` class which links unique entity
mentions to ontology terms (GO, HPO, MONDO, UBERON) and optionally to
UMLS concepts. The implementation uses SapBERT for embedding generation
and FAISS for fast nearest-neighbor search.

Inputs:
- `config` (dict) describing model and data paths. Keys:
  - embedding_model_path: str (path to pretrained SapBERT)
  - database_embedding_path: str (path to directory containing *_terms.json files)
  - unique_entity_list_path: str (path to JSON array of entity strings)
  - enable_umls: bool (optional, default False)

Outputs:
- `link_entities()` -> dict mapping entity surface strings to link info:
  {
    "entity": {
      "entity_id": str,
      "entity_ontology": str,
      "entity_onto_term": str,
      "similarity": float
    },
    ...
  }

Notes:
- This file augments methods with type hints and structured docstrings to
  make inputs/outputs explicit for maintainability and automated analysis.
"""
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import faiss
import gc
from tqdm import tqdm
import json
import os
import spacy
from spacy.tokens import Span
import torch
import heapq
import scispacy
from scispacy.linking import EntityLinker as ScispacyEntityLinker


class EntityLinker:
    def __init__(self, config):
        """
        Initialize the entity linker
        Args:
            config (dict): Configuration dictionary containing:
                - embedding_model_path: Path to SapBERT model
                - database_embedding_path: Base path for database embedding vectors
                - unique_entity_list_path: Path to unique entity list
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config['embedding_model_path'])
        self.model = BertModel.from_pretrained(
            config['embedding_model_path'],
            return_dict=False,
            output_hidden_states=True
        ).to(self.device)

        # Load entity list and terminology data
        self._load_data()

        # Initialize FAISS indices
        self._initialize_indices()

        # Initialize UMLS linker (optional, can be slow)
        self.enable_umls = config.get('enable_umls', False)
        if self.enable_umls:
            self._initialize_umls_linker()

    def _load_data(self):
        """Load required data files"""
        base_path = self.config['database_embedding_path']

        # Load unique entity list
        with open(self.config['unique_entity_list_path'], "r", encoding='utf-8') as f:
            self.unique_entity_texts = json.load(f)

        # Load terms for each ontology
        self.ontology_terms = {}
        for ontology in ['go', 'hpo', 'mondo', 'uberon']:
            path = os.path.join(base_path, f'{ontology}_terms.json')
            with open(path, 'r') as f:
                self.ontology_terms[ontology.upper()] = json.load(f)

    def _initialize_indices(self):
        """Initialize FAISS indices"""
        print("\n" + "-"*60)
        print("Building FAISS indices...")
        print("-"*60)
        
        self.indices = {}
        for ontology, terms in tqdm(self.ontology_terms.items(), desc="  Building indices"):
            self.indices[ontology] = self.create_faiss_index(terms, use_gpu=torch.cuda.is_available())
        
        print(f"  ✓ Built {len(self.indices)} FAISS indices")

    def _initialize_umls_linker(self):
        """Initialize UMLS linker"""
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = self.nlp.get_pipe("scispacy_linker")

    def get_bert_embedding(self, text: str) -> "np.ndarray":
        """
        Compute a SapBERT embedding for the given text.

        Args:
            text (str): Input text to encode.

        Returns:
            np.ndarray: 1-D numpy array representing the pooled embedding
                (shape: [d]).

        Structure:
            {
                "type": "ndarray",
                "shape": ["d"],
                "dtype": "float32"
            }
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        return outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()

    def get_best_match(self, entity_embeddings: "np.ndarray", index, terms_with_embeddings: list) -> dict:
        """
        Query FAISS index to obtain the best matching term for an embedding.

        Args:
            entity_embeddings (np.ndarray): Query embedding(s), shape [1, d] or [d].
            index: FAISS index object supporting `search`.
            terms_with_embeddings (list[dict]): List of term dicts aligned with
                the index where each dict contains at least a `sapbert_embedding`.

        Returns:
            dict: The term dict corresponding to the nearest neighbor.

        Structure:
            {
                "type": "dict",
                "keys": ["id", "name", "sapbert_embedding", ...]
            }
        """
        if entity_embeddings.ndim == 1:
            entity_embeddings = entity_embeddings[np.newaxis, :]
        D, I = index.search(entity_embeddings, 1)
        best_idx = I[0][0]
        best_term = terms_with_embeddings[best_idx]
        return best_term

    def cosine_similarity(self, vec1: "np.ndarray", vec2: "np.ndarray") -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): Vector 1.
            vec2 (np.ndarray): Vector 2.

        Returns:
            float: Cosine similarity in [-1, 1].
        """
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def create_faiss_index(self, terms_with_embeddings: list, use_gpu: bool):
        """
        Create a FAISS index from a list of term dicts containing embeddings.

        Args:
            terms_with_embeddings (list[dict]): Each dict must have 'sapbert_embedding'.
            use_gpu (bool): Whether to create a GPU-backed index (if available).

        Returns:
            FAISS index object ready for nearest neighbor queries.
        """
        embedding_size = len(terms_with_embeddings[0]['sapbert_embedding'])
        embedding_matrix = np.array([term['sapbert_embedding'] for term in terms_with_embeddings]).astype('float32')
        faiss.normalize_L2(embedding_matrix)

        if use_gpu and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(embedding_size)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(embedding_matrix)
            return gpu_index
        else:
            index = faiss.IndexFlatL2(embedding_size)
            index.add(embedding_matrix)
            return index

    def process_entities(self, entities, processed_entity, index, terms_with_embeddings, ontology_name, id_key,
                         term_name_key):
        """
        Process entities and update the `processed_entity` dictionary.
        """
        for entity in tqdm(entities, desc=f'Processing entities for {ontology_name}'):
            entity_embeddings = self.get_bert_embedding(entity).reshape(1, -1)
            best_term = self.get_best_match(entity_embeddings, index, terms_with_embeddings)
            best_term_embedding = np.array(best_term['sapbert_embedding']).flatten()
            similarity = self.cosine_similarity(entity_embeddings, best_term_embedding)

            if similarity > 0.88:
                entity_info = {
                    'entity_id': best_term[id_key],
                    'entity_ontology': ontology_name,
                    'entity_onto_term': best_term[term_name_key],
                    'similarity': float(similarity)
                }
                if entity in processed_entity:
                    if processed_entity[entity]['similarity'] < similarity:
                        processed_entity[entity] = entity_info
                else:
                    processed_entity[entity] = entity_info

    def link_entities(self):
        """
        Execute entity linking process
        Returns:
            dict: Processed entity mapping results
        """
        print("\n" + "="*60)
        print("ENTITY LINKING PROCESS")
        print("="*60)
        print(f"Total entities to process: {len(self.unique_entity_texts)}")
        print(f"UMLS linking: {'Enabled' if self.enable_umls else 'Disabled'}")
        print()
        
        processed_entity = {}

        # Process ontology matching
        ontologies = [
            {
                "name": name,
                "index": self.indices[name],
                "terms": self.ontology_terms[name],
                "id_key": "id",
                "term_name_key": "name"
            }
            for name in ["HPO", "GO", "UBERON", "MONDO"]
        ]

        print("-"*60)
        print(f"Processing {len(ontologies)} ontologies...")
        print("-"*60)
        
        for i, ontology in enumerate(ontologies, 1):
            print(f"\n[{i}/{len(ontologies)}] {ontology['name']} ({len(ontology['terms'])} terms)")
            self.process_entities(
                self.unique_entity_texts,
                processed_entity,
                ontology["index"],
                ontology["terms"],
                ontology["name"],
                ontology["id_key"],
                ontology["term_name_key"]
            )
            print(f"  Current linked entities: {len(processed_entity)}")

        # Process UMLS matching (if enabled)
        if self.enable_umls:
            print("\n" + "-"*60)
            print("Processing UMLS linking (this may take a while)...")
            print("-"*60)
            self._process_umls_entities(processed_entity)
        
        print("\n" + "="*60)
        print(f"LINKING COMPLETE: {len(processed_entity)} entities linked")
        print("="*60)

        return processed_entity

    def _process_umls_entities(self, processed_entity):
        """Process UMLS entity linking"""
        for text in tqdm(self.unique_entity_texts, desc='Processing unique entities'):
            doc = self.nlp(text)
            self.linker(doc)
            entity_embeddings = self.get_bert_embedding(text)
            entity_matches = []

            if doc.ents:
                for ent in doc.ents:
                    if ent._.kb_ents:
                        for umls_ent in ent._.kb_ents:
                            linked_entity = self.nlp.get_pipe("scispacy_linker").kb.cui_to_entity[umls_ent[0]]
                            for name in [linked_entity.canonical_name] + linked_entity.aliases:
                                umls_embedding = self.get_bert_embedding(name)
                                similarity = self.cosine_similarity(entity_embeddings, umls_embedding)
                                if similarity > 0.85:
                                    entity_matches.append(
                                        {'umls_name': name, 'umls_cui': umls_ent[0], 'similarity': float(similarity)})

            doc = self.nlp(text)
            span = Span(doc, start=0, end=len(doc), label="CHEMICAL")
            doc.ents = [span]
            self.linker(doc)
            if doc.ents:
                for ent in doc.ents:
                    if ent._.kb_ents:
                        for umls_ent in ent._.kb_ents:
                            linked_entity = self.nlp.get_pipe("scispacy_linker").kb.cui_to_entity[umls_ent[0]]
                            for name in [linked_entity.canonical_name] + linked_entity.aliases:
                                umls_embedding = self.get_bert_embedding(name)
                                similarity = self.cosine_similarity(entity_embeddings, umls_embedding)
                                if similarity > 0.85:
                                    entity_matches.append(
                                        {'umls_name': name, 'umls_cui': umls_ent[0], 'similarity': float(similarity)})

            top_matches = heapq.nlargest(1, entity_matches, key=lambda x: x['similarity'])
            if top_matches:
                top_match = top_matches[0]
                if text in processed_entity:
                    if top_match['similarity'] > processed_entity[text]['similarity']:
                        entity_info = {
                            'entity_id': top_match['umls_cui'],
                            'entity_ontology': 'UMLS',
                            'entity_onto_term': top_match['umls_name'],
                            'similarity': float(top_match['similarity'])
                        }
                        processed_entity[text] = entity_info
                else:
                    entity_info = {
                        'entity_id': top_match['umls_cui'],
                        'entity_ontology': 'UMLS',
                        'entity_onto_term': top_match['umls_name'],
                        'similarity': float(top_match['similarity'])
                    }
                    processed_entity[text] = entity_info


# Usage example
if __name__ == "__main__":
    # Base directory resolution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    config = {
        "embedding_model_path": os.path.join(BASE_DIR, "models", "InputsAndOutputs", "pretrained"),
        "database_embedding_path": os.path.join(BASE_DIR, "models", "InputsAndOutputs", "output", "linked_ontology"),
        "unique_entity_list_path": os.path.join(BASE_DIR, "models", "InputsAndOutputs", "output", "linked_ontology", "entity_sample.json"),
        "enable_umls": True  # Set to True to enable UMLS linking (slow, requires ~1 hour for 10680 entities)
    }

    linker = EntityLinker(config)
    results = linker.link_entities()
    
    # Save results to file
    import json
    output_file = os.path.join(BASE_DIR, "models", "InputsAndOutputs", "output", "entity_linking_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Entity linking completed!")
    print(f"Results saved to {output_file}")
    print(f"Total entities linked: {len(results)}")
    
    # Print sample results
    print("\nSample results:")
    for i, (entity, info) in enumerate(list(results.items())[:5]):
        print(f"  Entity: {entity}")
        print(f"    -> {info}")
