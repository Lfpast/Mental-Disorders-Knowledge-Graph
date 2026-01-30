#!/usr/bin/env python3
"""
Drug Repurposing Prediction Demo
================================

Interactive demonstration of the drug repurposing prediction module
for the MDKG project.

Features:
1. Interactive mode: Query drugs and diseases in real-time
2. Demo mode: Showcase predictions with sample queries
3. Training mode: Train the model from scratch
4. Evaluation mode: Evaluate model performance

Usage:
    python demo.py                    # Interactive mode
    python demo.py --demo             # Demo mode with sample queries
    python demo.py --train            # Train model
    python demo.py --evaluate         # Evaluate trained model
    python demo.py --predict drug     # Predict repurposing for a drug
"""

import os
import sys
import argparse
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prediction import DrugRepurposingPredictor, DrugRepurposingEvaluator
    from prediction.predictor import TrainingConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the MDKG project root directory")
    sys.exit(1)


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║       MDKG Drug Repurposing Prediction System                 ║
║   Based on TxGNN: Zero-shot Therapeutic Prediction            ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_prediction_results(results, title="Prediction Results"):
    """Pretty print prediction results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    for i, (name, score, onto_info) in enumerate(results, 1):
        onto_str = ""
        if onto_info and onto_info.get('ontology_id'):
            onto_str = f" [{onto_info.get('ontology_source', 'UMLS')}: {onto_info.get('ontology_id')}]"
        
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"  {i:2d}. {name[:40]:<40} {bar} {score:.4f}{onto_str}")


def demo_mode(predictor: DrugRepurposingPredictor):
    """Run demo mode with sample queries."""
    print("\n" + "="*60)
    print("  Demo Mode: Sample Drug Repurposing Predictions")
    print("="*60)
    
    # Sample mental health related drugs
    sample_drugs = [
        "quetiapine",
        "aripiprazole", 
        "lithium",
        "valproate",
        "ketamine",
        "escitalopram",
    ]
    
    # Sample diseases
    sample_diseases = [
        "depression",
        "anxiety",
        "bipolar",
        "schizophrenia",
    ]
    
    available_drugs = predictor.get_drug_names()
    
    print("\n📊 Drug Repurposing Predictions")
    print("-" * 40)
    
    for drug in sample_drugs:
        # Find matching drug
        matching = [d for d in available_drugs if drug.lower() in d.lower()]
        if matching:
            drug_name = matching[0]
            try:
                results = predictor.predict_repurposing(drug_name, top_k=5)
                print_prediction_results(results, f"New indications for: {drug_name}")
            except Exception as e:
                print(f"  Could not predict for {drug}: {e}")
    
    print("\n📊 Treatment Predictions for Diseases")
    print("-" * 40)
    
    for disease in sample_diseases:
        try:
            results = predictor.predict_treatments(disease, top_k=5)
            print_prediction_results(results, f"Predicted treatments for: {disease}")
        except Exception as e:
            print(f"  Could not predict for {disease}: {e}")


def interactive_mode(predictor: DrugRepurposingPredictor):
    """Run interactive query mode."""
    print("\n" + "="*60)
    print("  Interactive Mode")
    print("="*60)
    print("""
Commands:
  drug <name>      - Predict new indications for a drug
  disease <name>   - Predict treatments for a disease
  score <drug> <disease> - Get specific prediction score
  list drugs       - List available drugs
  list diseases    - List available diseases
  help             - Show this help
  quit/exit        - Exit the program
""")
    
    while True:
        try:
            user_input = input("\n🔍 Enter command: ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif cmd == 'help':
                print(__doc__)
            
            elif cmd == 'drug':
                if not args:
                    print("Usage: drug <drug_name>")
                    continue
                try:
                    results = predictor.predict_repurposing(args, top_k=15)
                    print_prediction_results(results, f"Predicted new indications for: {args}")
                except ValueError as e:
                    print(f"Error: {e}")
                    # Suggest similar drugs
                    similar = [d for d in predictor.get_drug_names() if args.lower() in d.lower()]
                    if similar:
                        print(f"Did you mean: {', '.join(similar[:5])}")
            
            elif cmd == 'disease':
                if not args:
                    print("Usage: disease <disease_name>")
                    continue
                try:
                    results = predictor.predict_treatments(args, top_k=15)
                    print_prediction_results(results, f"Predicted treatments for: {args}")
                except ValueError as e:
                    print(f"Error: {e}")
                    similar = [d for d in predictor.get_disease_names() if args.lower() in d.lower()]
                    if similar:
                        print(f"Did you mean: {', '.join(similar[:5])}")
            
            elif cmd == 'score':
                score_parts = args.split()
                if len(score_parts) < 2:
                    print("Usage: score <drug_name> <disease_name>")
                    continue
                drug = score_parts[0]
                disease = ' '.join(score_parts[1:])
                try:
                    score = predictor.predict_drug_disease(drug, disease)
                    print(f"\n  Prediction score for {drug} → {disease}: {score:.4f}")
                except ValueError as e:
                    print(f"Error: {e}")
            
            elif cmd == 'list':
                if args.lower().startswith('drug'):
                    drugs = predictor.get_drug_names()
                    print(f"\nAvailable drugs ({len(drugs)}):")
                    for i, d in enumerate(sorted(drugs)[:50], 1):
                        print(f"  {i}. {d}")
                    if len(drugs) > 50:
                        print(f"  ... and {len(drugs) - 50} more")
                elif args.lower().startswith('disease'):
                    diseases = predictor.get_disease_names()
                    print(f"\nAvailable diseases ({len(diseases)}):")
                    for i, d in enumerate(sorted(diseases)[:50], 1):
                        print(f"  {i}. {d}")
                    if len(diseases) > 50:
                        print(f"  ... and {len(diseases) - 50} more")
                else:
                    print("Usage: list drugs|diseases")
            
            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


def train_model(
    data_folder: str,
    output_dir: Optional[str] = None,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 200,
    skip_pretrain: bool = False
) -> DrugRepurposingPredictor:
    """Train the drug repurposing model."""
    print("\n" + "="*60)
    print("  Training Drug Repurposing Model")
    print("="*60)
    
    config = TrainingConfig(
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        n_hid=128,
        n_inp=128,
        n_out=128,
        proto=True,
        proto_num=3,
    )
    
    predictor = DrugRepurposingPredictor(
        data_folder=data_folder,
        config=config,
        output_dir=output_dir
    )
    
    predictor.load_data(split='random', seed=42)
    predictor.train(skip_pretrain=skip_pretrain)
    
    # Save model
    predictor.save_model()
    
    return predictor


def evaluate_model(predictor: DrugRepurposingPredictor, save_results: bool = True):
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("  Evaluating Model Performance")
    print("="*60)
    
    evaluator = DrugRepurposingEvaluator(predictor)
    
    save_path = None
    if save_results:
        save_path = os.path.join(predictor.output_dir, "evaluation_results.json")
    
    results = evaluator.full_evaluation(save_path=save_path)
    
    # Print hardest cases for analysis
    print("\n" + "="*60)
    print("  Error Analysis: Hardest Prediction Cases")
    print("="*60)
    
    hard_diseases = evaluator.get_hardest_cases(mode='disease', n_cases=5)
    print("\nHardest diseases to predict treatments for:")
    for disease, mrr, true_drugs in hard_diseases:
        print(f"  - {disease} (MRR: {mrr:.4f})")
        print(f"    True drugs: {', '.join(true_drugs[:3])}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MDKG Drug Repurposing Prediction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Interactive mode
  python demo.py --demo             # Demo with sample queries  
  python demo.py --train            # Train the model
  python demo.py --evaluate         # Evaluate model
  python demo.py --predict lithium  # Predict for specific drug
        """
    )
    
    parser.add_argument(
        '--data-folder', '-d',
        default='./models/InputsAndOutputs',
        help='Path to data folder'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for models and results'
    )
    parser.add_argument(
        '--model-path', '-m',
        default=None,
        help='Path to pre-trained model'
    )
    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='Evaluate the model'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode with sample queries'
    )
    parser.add_argument(
        '--predict', '-p',
        type=str,
        default=None,
        help='Predict repurposing for a specific drug'
    )
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=50,
        help='Number of pretraining epochs'
    )
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=200,
        help='Number of finetuning epochs'
    )
    parser.add_argument(
        '--skip-pretrain',
        action='store_true',
        help='Skip pretraining phase'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode (fewer epochs for testing)'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Quick mode overrides
    if args.quick:
        args.pretrain_epochs = 5
        args.finetune_epochs = 10
        args.skip_pretrain = True
    
    # Determine what to do
    predictor = None
    
    if args.train:
        predictor = train_model(
            data_folder=args.data_folder,
            output_dir=args.output_dir,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            skip_pretrain=args.skip_pretrain
        )
    elif args.model_path or os.path.exists(
        os.path.join(args.data_folder, "output", "prediction", "model.pt")
    ):
        # Load existing model
        model_path = args.model_path or os.path.join(
            args.data_folder, "output", "prediction", "model.pt"
        )
        
        print(f"Loading model from {model_path}")
        predictor = DrugRepurposingPredictor(data_folder=args.data_folder)
        predictor.load_data(use_cache=True)
        predictor.load_model(model_path)
    else:
        # Train a quick model for demo
        print("No trained model found. Training a quick model...")
        predictor = train_model(
            data_folder=args.data_folder,
            output_dir=args.output_dir,
            pretrain_epochs=5,
            finetune_epochs=20,
            skip_pretrain=True
        )
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_model(predictor)
    
    # Predict for specific drug
    if args.predict:
        try:
            results = predictor.predict_repurposing(args.predict, top_k=20)
            print_prediction_results(results, f"Drug Repurposing for: {args.predict}")
        except ValueError as e:
            print(f"Error: {e}")
            similar = [d for d in predictor.get_drug_names() if args.predict.lower() in d.lower()]
            if similar:
                print(f"Similar drugs found: {', '.join(similar[:10])}")
    
    # Demo mode
    if args.demo:
        demo_mode(predictor)
    
    # Interactive mode (default if no other action)
    if not (args.train or args.evaluate or args.predict or args.demo):
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
