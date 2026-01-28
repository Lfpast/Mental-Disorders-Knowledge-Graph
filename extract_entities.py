"""
Extract unique entities from NER training data

This module provides utilities to extract and persist unique entity mentions
from the NER training dataset used in this repository.

Inputs:
- data_file: str (path to JSON file with training samples)

Outputs:
- Returns a set[str] of unique entity texts extracted from samples.
- When run as a script, writes `entity_sample.json` under
  `models/InputsAndOutputs/output/linked_ontology/` containing a JSON array
  of extracted entity strings.
"""
import json
import os
from collections import defaultdict
from typing import Set


def extract_entities_from_data(data_file: str) -> Set[str]:
    """
    Extract all unique entities from the training data JSON file.

    Args:
        data_file (str): Path to the JSON file containing training data. The
            file is expected to be a list of samples where each sample may
            contain 'tokens' (list[str]) and 'entities' (list[dict]) with
            integer 'start' and 'end' indexes into the tokens list.

    Returns:
        Set[str]: A set containing unique entity surface forms reconstructed
            from token spans. Example element: "major depressive disorder".

    Structure of the returned data:
        {
            "type": "set",
            "items": "strings",
            "example": ["entity one", "entity two"]
        }
    """
    unique_entities: Set[str] = set()

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        if 'entities' in sample:
            for entity in sample['entities']:
                # Entity format: {"start": int, "end": int, "type": str}
                # Reconstruct entity text from tokens
                if 'start' in entity and 'end' in entity:
                    tokens = sample.get('tokens', [])
                    entity_tokens = tokens[entity['start']:entity['end']]
                    entity_text = ' '.join(entity_tokens)
                    if entity_text.strip():
                        unique_entities.add(entity_text)

    return unique_entities

def main():
    # Paths based on repository root
    repo_root = os.path.abspath(os.path.dirname(__file__))
    data_root = os.path.join(repo_root, "models", "InputsAndOutputs")
    input_dir = os.path.join(data_root, "input")
    output_dir_base = os.path.join(data_root, "output")

    train_file = os.path.join(input_dir, "md_train_KG_0217.json")

    # Extract entities
    print("Extracting entities from training data...")
    unique_entities = extract_entities_from_data(train_file)
    
    print(f"Found {len(unique_entities)} unique entities")
    
    # Save to file
    output_dir = os.path.join(output_dir_base, "linked_ontology")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "entity_sample.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(list(unique_entities), f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(unique_entities)} entities to {output_file}")
    
    # Print sample entities
    print("\nSample entities:")
    for entity in list(unique_entities)[:20]:
        print(f"  - {entity}")

if __name__ == "__main__":
    main()
