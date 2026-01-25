"""
Extract unique entities from NER training data
"""
import json
import os
from collections import defaultdict

def extract_entities_from_data(data_file):
    """
    Extract all unique entities from the training data JSON file
    
    Args:
        data_file: Path to the JSON file containing training data
        
    Returns:
        Set of unique entity texts
    """
    unique_entities = set()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for sample in data:
        if 'entities' in sample:
            for entity in sample['entities']:
                # Entity format: {"start": int, "end": int, "type": str}
                # We need to reconstruct the entity text from tokens
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
