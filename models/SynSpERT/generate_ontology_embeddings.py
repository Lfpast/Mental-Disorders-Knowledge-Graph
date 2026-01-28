"""
Generate embeddings for biomedical ontologies using official OBO files

This script:
1. Parses official .obo files from InputsAndOutputs/input/ontologies/
2. Generates embeddings using CODER++ model
3. Saves results to InputsAndOutputs/output/linked_ontology/

Inputs:
- OBO files located in `models/InputsAndOutputs/input/ontologies/` (hpo.obo, go.obo, etc.)
- CODER++ model in `models/InputsAndOutputs/pretrained/`

Outputs:
- JSON files with embedded terms under `models/InputsAndOutputs/output/linked_ontology/`:
  - `go_terms.json`, `hpo_terms.json`, etc., each containing term dicts with `sapbert_embedding` vectors.

Prerequisites:
- Run ontologies_downloader.py first to download .obo files
- CODER++ model in InputsAndOutputs/pretrained/
"""
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json
import os
import re
from tqdm import tqdm
from typing import List, Dict, Optional


class OBOParser:
    """Parser for OBO format ontology files"""
    
    @staticmethod
    def parse_obo(file_path: str) -> List[Dict]:
        """
        Parse an OBO file and extract terms with id and name
        
        Args:
            file_path: Path to .obo file
            
        Returns:
            List of term dictionaries with 'id' and 'name' keys
        """
        terms = []
        current_term = {}
        in_term = False
        
        print(f"  Parsing {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line == '[Term]':
                    if current_term and 'id' in current_term and 'name' in current_term:
                        terms.append(current_term)
                    current_term = {}
                    in_term = True
                elif line.startswith('[') and line.endswith(']'):
                    # Other stanza type (e.g., [Typedef])
                    if current_term and 'id' in current_term and 'name' in current_term:
                        terms.append(current_term)
                    current_term = {}
                    in_term = False
                elif in_term:
                    if line.startswith('id:'):
                        current_term['id'] = line[3:].strip()
                    elif line.startswith('name:'):
                        current_term['name'] = line[5:].strip()
                    elif line.startswith('synonym:'):
                        # Extract synonyms for better matching
                        match = re.search(r'"([^"]+)"', line)
                        if match:
                            if 'synonyms' not in current_term:
                                current_term['synonyms'] = []
                            current_term['synonyms'].append(match.group(1))
                    elif line.startswith('is_obsolete: true'):
                        current_term['obsolete'] = True
        
        # Don't forget the last term
        if current_term and 'id' in current_term and 'name' in current_term:
            terms.append(current_term)
        
        # Filter out obsolete terms
        terms = [t for t in terms if not t.get('obsolete', False)]
        
        return terms


class OntologyEmbeddingGenerator:
    """Generate embeddings for ontology terms using CODER++ model"""
    
    def __init__(self, model_path: str):
        """
        Initialize embedding generator with CODER++ model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Model path: {model_path}")
            raise
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 32, max_length: int = 128) -> List[List[float]]:
        """
        Get embeddings for a batch of texts (more efficient than single processing)
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            max_length: Maximum token length
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, return_dict=True)
                
                # Mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend([emb.tolist() for emb in batch_embeddings])
                
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Fallback: add None for failed items
                embeddings.extend([None] * len(batch_texts))
        
        return embeddings
    
    def generate_embeddings_for_terms(
        self, 
        terms: List[Dict], 
        ontology_name: str = "",
        batch_size: int = 64
    ) -> List[Dict]:
        """
        Generate embeddings for a list of terms with progress bar
        
        Args:
            terms: List of term dictionaries with 'name' key
            ontology_name: Name of ontology for display
            batch_size: Batch size for processing
            
        Returns:
            List of terms with 'sapbert_embedding' added
        """
        texts = [term['name'] for term in terms]
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"  Processing {len(texts)} terms in {total_batches} batches (batch_size={batch_size})...")
        
        all_embeddings = []
        
        # Progress bar for batches
        with tqdm(total=len(texts), desc=f"  Generating {ontology_name}", unit="terms") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embs = self.get_batch_embeddings(batch_texts, batch_size=len(batch_texts))
                all_embeddings.extend(batch_embs)
                pbar.update(len(batch_texts))
        
        # Add embeddings to terms
        success_count = 0
        for term, embedding in zip(terms, all_embeddings):
            if embedding is not None:
                term['sapbert_embedding'] = embedding
                success_count += 1
        
        print(f"  ✓ Successfully embedded {success_count}/{len(terms)} terms")
        
        return terms


def main():
    """Main function to generate ontology embeddings"""
    
    # Configuration
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Paths
    obo_dir = os.path.join(repo_root, "models", "InputsAndOutputs", "input", "ontologies")
    output_dir = os.path.join(repo_root, "models", "InputsAndOutputs", "output", "linked_ontology")
    model_path = os.path.join(repo_root, "models", "InputsAndOutputs", "pretrained")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("GENERATING ONTOLOGY EMBEDDINGS")
    print("="*60)
    print(f"OBO files directory: {obo_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")
    print()
    
    # Check for .obo files
    ontologies = ['hpo', 'go', 'mondo', 'uberon']
    obo_files = {}
    
    print("Checking for OBO files...")
    for onto in ontologies:
        obo_file = os.path.join(obo_dir, f"{onto}.obo")
        if os.path.exists(obo_file):
            file_size = os.path.getsize(obo_file) / (1024 * 1024)
            print(f"  ✓ {onto.upper()}: {obo_file} ({file_size:.1f} MB)")
            obo_files[onto] = obo_file
        else:
            print(f"  ✗ {onto.upper()}: NOT FOUND at {obo_file}")
    
    if not obo_files:
        print("\n✗ No OBO files found!")
        print("Please run ontologies_downloader.py first to download the ontology files.")
        return
    
    # Initialize embedding generator
    print("\n" + "-"*60)
    print("Loading CODER++ model...")
    try:
        generator = OntologyEmbeddingGenerator(model_path)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nPlease ensure CODER++ model is available at:")
        print(f"  {model_path}")
        return
    
    # Process each ontology
    print("\n" + "-"*60)
    print("Processing ontologies...")
    
    results = {}
    
    for onto, obo_file in obo_files.items():
        print(f"\n[{onto.upper()}]")
        
        # Parse OBO file
        terms = OBOParser.parse_obo(obo_file)
        print(f"  Found {len(terms)} terms")
        
        if not terms:
            print(f"  ✗ No terms found, skipping...")
            continue
        
        # Generate embeddings
        terms_with_embeddings = generator.generate_embeddings_for_terms(
            terms,
            ontology_name=onto.upper(),
            batch_size=64
        )
        
        # Save to output file
        output_file = os.path.join(output_dir, f"{onto}_terms.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(terms_with_embeddings, f, ensure_ascii=False)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ Saved to {output_file} ({file_size:.1f} MB)")
        
        results[onto] = {
            'terms': len(terms),
            'output_file': output_file
        }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_terms = sum(r['terms'] for r in results.values())
    print(f"Total terms processed: {total_terms}")
    for onto, info in results.items():
        print(f"  - {onto.upper()}: {info['terms']} terms")
    
    print(f"\nOutput files saved to: {output_dir}")
    print("\n✓ Embedding generation complete!")


if __name__ == "__main__":
    main()
