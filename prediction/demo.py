#!/usr/bin/env python3
"""
Drug Repurposing Prediction Demo
================================

Interactive demonstration of the drug repurposing prediction module
for the MDKG project.

All settings (model path, data source, hyperparameters, etc.) are
configured through prediction_config.json. Edit the config file
to change behavior — no CLI flags needed for model/data selection.

Features:
1. Interactive mode: Query drugs and diseases in real-time
2. Demo mode: Showcase predictions with sample queries
3. Training mode: Train the model from scratch
4. Evaluation mode: Evaluate model performance

Usage:
    python -m prediction.demo                          # Interactive mode
    python -m prediction.demo --demo                   # Demo with samples
    python -m prediction.demo --train                  # Train model
    python -m prediction.demo --treatments depression  # Predict
    python -m prediction.demo --config path/to/config  # Custom config
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prediction import DrugRepurposingPredictor, DrugRepurposingEvaluator
    from prediction.predictor import TrainingConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the MDKG project root directory")
    sys.exit(1)


# Default config path
DEFAULT_CONFIG_PATH = "./models/InputsAndOutputs/configs/prediction_config.json"


def load_prediction_config(config_path: str) -> Dict[str, Any]:
    """Load prediction configuration from JSON file.
    
    Args:
        config_path: Path to prediction_config.json
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        print("  Using built-in defaults.")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Loaded config from: {config_path}")
    return config


def build_training_config(cfg: Dict[str, Any]) -> TrainingConfig:
    """Build TrainingConfig from prediction_config.json sections.
    
    Args:
        cfg: Full config dict loaded from prediction_config.json
        
    Returns:
        TrainingConfig instance
    """
    model_cfg = cfg.get('model', {})
    training_cfg = cfg.get('training', {})
    
    # Merge model + training sections into TrainingConfig fields
    merged = {}
    for key in ['n_inp', 'n_hid', 'n_out', 'attention', 'proto', 'proto_num',
                'sim_measure', 'agg_measure', 'exp_lambda', 'dropout']:
        if key in model_cfg:
            merged[key] = model_cfg[key]
    
    for key in ['pretrain_epochs', 'finetune_epochs', 'pretrain_lr', 'finetune_lr',
                'batch_size', 'weight_decay', 'patience', 'neg_ratio',
                'print_every', 'eval_every']:
        if key in training_cfg:
            merged[key] = training_cfg[key]
    
    return TrainingConfig(**{k: v for k, v in merged.items()
                            if k in TrainingConfig.__dataclass_fields__})


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
    cfg: Dict[str, Any],
    skip_pretrain: bool = False,
    pretrain_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
) -> DrugRepurposingPredictor:
    """Train the drug repurposing model using prediction_config.json settings.
    
    Args:
        cfg: Full config dict from prediction_config.json
        skip_pretrain: Whether to skip pretraining phase
        pretrain_epochs: Override pretrain epochs (None = use config)
        finetune_epochs: Override finetune epochs (None = use config)
    """
    paths_cfg = cfg.get('paths', {})
    data_cfg = cfg.get('data', {})
    
    data_folder = paths_cfg.get('data_folder', './models/InputsAndOutputs')
    output_dir = os.path.join(data_folder, paths_cfg.get('output_dir', 'output/prediction'))
    data_source = data_cfg.get('data_source', 'sampled')
    split = data_cfg.get('split', 'random')
    seed = data_cfg.get('seed', 42)
    
    print("\n" + "="*60)
    print("  Training Drug Repurposing Model")
    print("="*60)
    print(f"  Data source: {data_source}")
    
    config = build_training_config(cfg)
    
    # Apply CLI overrides if provided
    if pretrain_epochs is not None:
        config.pretrain_epochs = pretrain_epochs
    if finetune_epochs is not None:
        config.finetune_epochs = finetune_epochs
    
    predictor = DrugRepurposingPredictor(
        data_folder=data_folder,
        config=config,
        output_dir=output_dir
    )
    
    predictor.load_data(split=split, seed=seed, data_source=data_source)
    predictor.train(skip_pretrain=skip_pretrain)
    
    # Save model to config-specified path
    model_file = paths_cfg.get('model_file', 'model.pt')
    model_path = os.path.join(output_dir, model_file)
    predictor.save_model(model_path)
    
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
All model, data, and training settings are configured via prediction_config.json.
Edit the config file to change data source, model path, hyperparameters, etc.

Examples:
  python -m prediction.demo                          # Interactive mode (uses config)
  python -m prediction.demo --train                  # Train model
  python -m prediction.demo --treatments depression  # Predict treatments
  python -m prediction.demo --config ./my_config.json --train  # Custom config
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default=DEFAULT_CONFIG_PATH,
        help='Path to prediction_config.json (default: %(default)s)'
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
        '--treatments',
        type=str,
        default=None,
        help='Predict treatments for a specific disease'
    )
    parser.add_argument(
        '--explain',
        nargs=2,
        metavar=('DRUG', 'DISEASE'),
        help='Explain a specific drug-disease prediction'
    )
    parser.add_argument(
        '--explain-batch',
        nargs=2,
        metavar=('INPUT_FILE', 'OUTPUT_FILE'),
        help='Batch explanation generation (input: drug,disease csv)'
    )
    parser.add_argument(
        '--batch-predict',
        nargs=2,
        metavar=('INPUT_FILE', 'OUTPUT_FILE'),
        help='Batch prediction for drugs (input: one drug per line)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive prediction mode'
    )
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=None,
        help='Override pretrain epochs from config'
    )
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=None,
        help='Override finetune epochs from config'
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
    
    # Load configuration from prediction_config.json
    cfg = load_prediction_config(args.config)
    paths_cfg = cfg.get('paths', {})
    data_cfg = cfg.get('data', {})
    
    data_folder = paths_cfg.get('data_folder', './models/InputsAndOutputs')
    output_dir = os.path.join(data_folder, paths_cfg.get('output_dir', 'output/prediction'))
    model_file = paths_cfg.get('model_file', 'model.pt')
    model_path = os.path.join(output_dir, model_file)
    data_source = data_cfg.get('data_source', 'sampled')
    split = data_cfg.get('split', 'random')
    seed = data_cfg.get('seed', 42)
    
    # Quick mode overrides
    if args.quick:
        args.pretrain_epochs = 5
        args.finetune_epochs = 10
        args.skip_pretrain = True
    
    # Determine what to do
    predictor = None
    
    if args.train:
        predictor = train_model(
            cfg=cfg,
            skip_pretrain=args.skip_pretrain,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
        )
    elif os.path.exists(model_path):
        # Load existing model from config-specified path
        print(f"Loading model from {model_path}")
        print(f"  Data source: {data_source}")
        
        predictor = DrugRepurposingPredictor(
            data_folder=data_folder,
            output_dir=output_dir
        )
        predictor.load_data(split=split, seed=seed, use_cache=True, data_source=data_source)
        predictor.load_model(model_path)
    else:
        # Train a quick model for demo
        print(f"No trained model found at {model_path}. Training a quick model...")
        quick_cfg = dict(cfg)
        quick_cfg['training'] = dict(cfg.get('training', {}))
        quick_cfg['training']['pretrain_epochs'] = 5
        quick_cfg['training']['finetune_epochs'] = 20
        predictor = train_model(cfg=quick_cfg, skip_pretrain=True)
    
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

    # Predict treatments for specific disease
    if args.treatments:
        try:
            results = predictor.predict_treatments(args.treatments, top_k=20)
            print_prediction_results(results, f"Predicted Treatments for: {args.treatments}")
        except ValueError as e:
            print(f"Error: {e}")
            similar = [d for d in predictor.get_disease_names() if args.treatments.lower() in d.lower()]
            if similar:
                print(f"Similar diseases found: {', '.join(similar[:10])}")

    # Explain prediction
    if args.explain:
        drug_name, disease_name = args.explain
        print(f"\nGenerating explanation for {drug_name} -> {disease_name}...")
        try:
            explanation = predictor.explain_prediction(
                drug_name=drug_name,
                disease_name=disease_name,
                epochs=100
            )
            print("\n" + "="*70)
            if explanation.explanation_text:
                print(explanation.explanation_text)
            else:
                print(f"Prediction Score: {explanation.prediction_score:.4f}")
            
            # Save explanation
            output_file = os.path.join(
                output_dir,
                f"explanation_{drug_name}_{disease_name}.json".replace(" ", "_")
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(explanation.to_dict(), f, indent=2, ensure_ascii=False)
            print("\n" + "="*70)
            print(f"Explanation saved to: {output_file}")
            
        except Exception as e:
            print(f"Error generating explanation: {e}")

    # Batch explanation
    if args.explain_batch:
        input_file, output_file = args.explain_batch
        print(f"\nBatch explaining from {input_file} to {output_file}...")
        try:
            pairs = []
            with open(input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ',' in line:
                         pairs.append(line.split(',', 1))
            
            print(f"Processing {len(pairs)} pairs...")
            results = []
            for drug, disease in pairs:
                drug = drug.strip()
                disease = disease.strip()
                print(f"  Explaining: {drug} -> {disease}")
                try:
                    expl = predictor.explain_prediction(drug, disease, epochs=50)
                    results.append(expl.to_dict())
                except Exception as e:
                    print(f"  Error: {e}")
                    results.append({
                        'drug': drug, 'disease': disease, 'error': str(e)
                    })
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Batch explanation error: {e}")

    # Batch prediction
    if args.batch_predict:
        input_path, output_path = args.batch_predict
        print(f"\nBatch predicting from {input_path} to {output_path}...")
        try:
            results = {}
            with open(input_path, 'r') as f:
                drugs = [line.strip() for line in f if line.strip()]
            
            for drug in drugs:
                print(f"  Processing: {drug}")
                try:
                    preds = predictor.predict_repurposing(drug, top_k=20)
                    results[drug] = [{'disease': d, 'score': s} for d, s, _ in preds]
                except Exception as e:
                    print(f"  Error: {e}")
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
            
        except Exception as e:
            print(f"Batch prediction error: {e}")

    # Demo mode
    if args.demo:
        demo_mode(predictor)
    
    # Interactive mode (default if no other action or explicitly requested)
    if args.interactive or not (args.train or args.evaluate or args.predict or args.treatments or args.explain or args.explain_batch or args.batch_predict or args.demo):
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
