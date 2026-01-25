"""
Triplets Refinement Script
This script refines triplets extracted from NER&RE model using a Large Language Model (LLM).
The refinement process validates, normalizes, and improves the quality of extracted triplets.
"""
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Triplet:
    """Represents a knowledge graph triplet (head, relation, tail)"""
    head_entity: str
    head_type: str
    relation: str
    tail_entity: str
    tail_type: str
    sentence: str
    confidence: float = 1.0


def extract_triplets_from_predictions(prediction_file: str) -> List[Triplet]:
    """
    Extract triplets from NER&RE model predictions.
    
    Args:
        prediction_file: Path to the JSON file containing model predictions
        
    Returns:
        List of Triplet objects
    """
    triplets = []
    
    with open(prediction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for sample in data:
        tokens = sample.get('tokens', [])
        entities = sample.get('entities', [])
        relations = sample.get('relations', [])
        sentence = sample.get('sents', ' '.join(tokens))
        
        # Build entity lookup by entity_id
        entity_lookup = {e['entity_id']: e for e in entities}
        
        for rel in relations:
            head_id = rel.get('head_id')
            tail_id = rel.get('tail_id')
            rel_type = rel.get('type')
            
            if head_id in entity_lookup and tail_id in entity_lookup:
                head_entity = entity_lookup[head_id]
                tail_entity = entity_lookup[tail_id]
                
                triplet = Triplet(
                    head_entity=head_entity.get('span', ''),
                    head_type=head_entity.get('type', ''),
                    relation=rel_type,
                    tail_entity=tail_entity.get('span', ''),
                    tail_type=tail_entity.get('type', ''),
                    sentence=sentence
                )
                triplets.append(triplet)
    
    return triplets


def generate_refinement_prompt(triplet: Triplet) -> str:
    """
    Generate a prompt for LLM to refine a triplet.
    
    Args:
        triplet: The triplet to be refined
        
    Returns:
        A prompt string for the LLM
    """
    prompt = f"""You are a biomedical knowledge graph expert. Your task is to validate and refine the following triplet extracted from a scientific text.

## Original Sentence:
"{triplet.sentence}"

## Extracted Triplet:
- Head Entity: "{triplet.head_entity}" (Type: {triplet.head_type})
- Relation: "{triplet.relation}"
- Tail Entity: "{triplet.tail_entity}" (Type: {triplet.tail_type})

## Instructions:
1. Validate if the triplet accurately represents the relationship described in the sentence.
2. Check if the entity types are correct (disease, gene, symptom, drug, etc.).
3. Normalize entity names to standard biomedical terminology if needed.
4. Verify the relation type is appropriate and specific enough.
5. Assign a confidence score (0.0-1.0) based on how clearly the relationship is stated.

## Response Format (JSON):
{{
    "is_valid": true/false,
    "refined_triplet": {{
        "head_entity": "normalized head entity name",
        "head_type": "entity type",
        "relation": "refined relation type",
        "tail_entity": "normalized tail entity name", 
        "tail_type": "entity type"
    }},
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of any changes made or why the triplet was marked invalid"
}}

Please provide your response:"""
    
    return prompt


def generate_batch_refinement_prompt(triplets: List[Triplet], batch_size: int = 5) -> str:
    """
    Generate a batch prompt for refining multiple triplets at once.
    
    Args:
        triplets: List of triplets to refine
        batch_size: Number of triplets per batch
        
    Returns:
        A prompt string for batch processing
    """
    prompt = """You are a biomedical knowledge graph expert. Your task is to validate and refine the following triplets extracted from scientific texts about mental disorders.

## Triplets to Refine:
"""
    
    for i, triplet in enumerate(triplets[:batch_size]):
        prompt += f"""
### Triplet {i+1}:
- Sentence: "{triplet.sentence[:200]}..."
- Head: "{triplet.head_entity}" ({triplet.head_type})
- Relation: "{triplet.relation}"
- Tail: "{triplet.tail_entity}" ({triplet.tail_type})
"""
    
    prompt += """
## Instructions:
For each triplet, provide:
1. Whether it's valid (true/false)
2. Refined entity names (normalized to standard terminology)
3. Refined relation type if needed
4. Confidence score (0.0-1.0)

## Response Format (JSON array):
[
    {
        "triplet_id": 1,
        "is_valid": true/false,
        "head_entity": "normalized name",
        "head_type": "type",
        "relation": "relation",
        "tail_entity": "normalized name",
        "tail_type": "type",
        "confidence": 0.0-1.0
    },
    ...
]

Please provide your response:"""
    
    return prompt


def save_triplets_for_refinement(triplets: List[Triplet], output_file: str):
    """
    Save triplets to a JSON file for manual or LLM-based refinement.
    
    Args:
        triplets: List of triplets to save
        output_file: Output file path
    """
    data = []
    for i, triplet in enumerate(triplets):
        data.append({
            "id": i,
            "head_entity": triplet.head_entity,
            "head_type": triplet.head_type,
            "relation": triplet.relation,
            "tail_entity": triplet.tail_entity,
            "tail_type": triplet.tail_type,
            "sentence": triplet.sentence,
            "prompt": generate_refinement_prompt(triplet)
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(data)} triplets with prompts to {output_file}")


def refine_with_openai(triplets: List[Triplet], api_key: str, model: str = "gpt-4") -> List[Dict]:
    """
    Refine triplets using OpenAI API.
    
    Args:
        triplets: List of triplets to refine
        api_key: OpenAI API key
        model: Model to use (default: gpt-4)
        
    Returns:
        List of refined triplet dictionaries
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Please install openai: pip install openai")
        return []
    
    client = OpenAI(api_key=api_key)
    refined_triplets = []
    
    for triplet in tqdm(triplets, desc="Refining triplets with LLM"):
        prompt = generate_refinement_prompt(triplet)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a biomedical knowledge graph expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                refined_triplets.append(result)
        except Exception as e:
            print(f"Error refining triplet: {e}")
            refined_triplets.append({"error": str(e), "original": triplet.__dict__})
    
    return refined_triplets


def main():
    """Main function to run triplet refinement"""
    
    # Configuration - paths relative to repository root
    repo_root = os.path.abspath(os.path.dirname(__file__))
    prediction_file = os.path.join(repo_root, "models", "InputsAndOutputs", "output", "sampling_json_run_v1_sampled.json")
    output_dir = os.path.join(repo_root, "models", "InputsAndOutputs", "output", "triplet_refinement")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract triplets from predictions
    print("Extracting triplets from model predictions...")
    triplets = extract_triplets_from_predictions(prediction_file)
    print(f"Extracted {len(triplets)} triplets")
    
    # Step 2: Save triplets with prompts for refinement
    triplets_file = os.path.join(output_dir, "triplets_for_refinement.json")
    save_triplets_for_refinement(triplets, triplets_file)
    
    # Step 3: Generate sample prompts
    prompts_file = os.path.join(output_dir, "sample_prompts.txt")
    with open(prompts_file, 'w', encoding='utf-8') as f:
        for i, triplet in enumerate(triplets[:5]):  # Sample first 5
            f.write(f"{'='*80}\n")
            f.write(f"TRIPLET {i+1}\n")
            f.write(f"{'='*80}\n")
            f.write(generate_refinement_prompt(triplet))
            f.write("\n\n")
    
    print(f"Generated sample prompts in {prompts_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRIPLET REFINEMENT SUMMARY")
    print("="*60)
    print(f"Total triplets extracted: {len(triplets)}")
    print(f"\nRelation type distribution:")
    
    relation_counts = {}
    for t in triplets:
        rel = t.relation
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"  - {rel}: {count}")
    
    print(f"\nEntity type distribution:")
    entity_types = {}
    for t in triplets:
        entity_types[t.head_type] = entity_types.get(t.head_type, 0) + 1
        entity_types[t.tail_type] = entity_types.get(t.tail_type, 0) + 1
    
    for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"  - {etype}: {count}")
    
    print(f"\nOutput files:")
    print(f"  - Triplets JSON: {triplets_file}")
    print(f"  - Sample prompts: {prompts_file}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
To refine triplets with an LLM:

Option 1: Use OpenAI API
    Set your API key: export OPENAI_API_KEY='your-key'
    Then run: python Triplets_refine_prompt.py --use-openai

Option 2: Manual Refinement
    Review the prompts in 'sample_prompts.txt'
    Use ChatGPT or other LLM interfaces manually
    
Option 3: Local LLM (e.g., Ollama)
    Install Ollama and run a local model
    Modify this script to use local API endpoint
""")


if __name__ == "__main__":
    import sys
    
    if "--use-openai" in sys.argv:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Please set OPENAI_API_KEY environment variable")
            sys.exit(1)
        
        repo_root = os.path.abspath(os.path.dirname(__file__))
        prediction_file = os.path.join(repo_root, "models", "InputsAndOutputs", "output", "sampling_json_run_v1_sampled.json")
        triplets = extract_triplets_from_predictions(prediction_file)
        
        # Refine first 10 triplets as demo
        refined = refine_with_openai(triplets[:10], api_key)
        
        output_dir = os.path.join(repo_root, "models", "InputsAndOutputs", "output", "triplet_refinement")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "refined_triplets.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(refined, f, ensure_ascii=False, indent=2)
        
        print(f"Refined triplets saved to {output_file}")
    else:
        main()
