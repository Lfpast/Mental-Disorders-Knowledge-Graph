#!/bin/bash
# ==============================================================================
# MDKG Drug Repurposing Prediction Pipeline
# ==============================================================================
#
# This script provides automation for training and running the drug repurposing
# prediction module based on TxGNN methodology with GNNExplainer support.
#
# Features:
# - GNN model training with pre-training and fine-tuning
# - Drug repurposing predictions
# - Prediction explainability with GNNExplainer
# - Model evaluation with disease-centric metrics
# - Interactive prediction mode
#
# Usage:
#   ./shell/prediction.sh train              # Train the model
#   ./shell/prediction.sh predict <drug>     # Predict for a drug
#   ./shell/prediction.sh treatments <disease> # Predict treatments for disease
#   ./shell/prediction.sh explain <drug> <disease> # Explain prediction
#   ./shell/prediction.sh evaluate           # Evaluate model
#   ./shell/prediction.sh demo               # Run interactive demo
#   ./shell/prediction.sh quick              # Quick training for testing
#
# References:
#   - TxGNN: https://www.nature.com/articles/s41591-023-02233-x
#   - GNNExplainer: https://arxiv.org/abs/1903.03894
#
# Requirements:
#   - Python 3.8+
#   - PyTorch
#   - DGL (Deep Graph Library)
#   - MDKG triplet extraction completed
#
# ==============================================================================

set -e

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PREDICTION_DIR="$PROJECT_ROOT/prediction"
DATA_DIR="$PROJECT_ROOT/models/InputsAndOutputs"
OUTPUT_DIR="$DATA_DIR/output/prediction"

# Training configuration (aligned with TxGNN paper)
PRETRAIN_EPOCHS=100
FINETUNE_EPOCHS=300
HIDDEN_DIM=256
LEARNING_RATE=0.0001
BATCH_SIZE=1024
PROTO_NUM=5
EXP_LAMBDA=0.7

# Quick mode settings
QUICK_PRETRAIN_EPOCHS=5
QUICK_FINETUNE_EPOCHS=20

# GNNExplainer settings
EXPLAINER_EPOCHS=100
EXPLAINER_LR=0.01
NUM_HOPS=2

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
    echo ""
}

print_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is required but not found"
        exit 1
    fi
    
    # Check required packages
    python3 -c "import torch" 2>/dev/null || {
        echo "Error: PyTorch is required. Install with: pip install torch"
        exit 1
    }
    
    python3 -c "import dgl" 2>/dev/null || {
        echo "Error: DGL is required. Install with: pip install dgl"
        exit 1
    }
    
    print_step "All dependencies satisfied"
}

check_data() {
    print_step "Checking MDKG data..."
    
    TRIPLETS_FILE="$DATA_DIR/output/sampling_json_run_v1_sampled.json"
    
    if [ ! -f "$TRIPLETS_FILE" ]; then
        echo "Error: Triplets file not found: $TRIPLETS_FILE"
        echo "Please run the MDKG extraction pipeline first."
        exit 1
    fi
    
    # Count triplets
    TRIPLET_COUNT=$(python3 -c "import json; print(len(json.load(open('$TRIPLETS_FILE'))))")
    print_step "Found $TRIPLET_COUNT samples in triplets file"
    
    # Check entity linking
    ENTITY_FILE="$DATA_DIR/output/entity_linking_results.json"
    if [ -f "$ENTITY_FILE" ]; then
        ENTITY_COUNT=$(python3 -c "import json; print(len(json.load(open('$ENTITY_FILE'))))")
        print_step "Found $ENTITY_COUNT entity mappings"
    fi
}

# ==============================================================================
# Training Functions
# ==============================================================================

train_model() {
    print_header "Training Drug Repurposing Model"
    
    local PRETRAIN=$PRETRAIN_EPOCHS
    local FINETUNE=$FINETUNE_EPOCHS
    local SKIP_PRETRAIN=""
    local DATA_SOURCE="sampled"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                PRETRAIN=$QUICK_PRETRAIN_EPOCHS
                FINETUNE=$QUICK_FINETUNE_EPOCHS
                SKIP_PRETRAIN="--skip-pretrain"
                shift
                ;;
            --skip-pretrain)
                SKIP_PRETRAIN="--skip-pretrain"
                shift
                ;;
            --pretrain-epochs)
                PRETRAIN=$2
                shift 2
                ;;
            --finetune-epochs)
                FINETUNE=$2
                shift 2
                ;;
            --data-source)
                DATA_SOURCE=$2
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    print_step "Data source: $DATA_SOURCE"
    print_step "Pre-training epochs: $PRETRAIN"
    print_step "Fine-tuning epochs: $FINETUNE"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    cd "$PROJECT_ROOT"
    
    python3 -m prediction.demo \
        --train \
        --data-folder "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --pretrain-epochs $PRETRAIN \
        --finetune-epochs $FINETUNE \
        --data-source "$DATA_SOURCE" \
        $SKIP_PRETRAIN
    
    print_step "Model saved to: $OUTPUT_DIR/model.pt"
}

# ==============================================================================
# Prediction Functions
# ==============================================================================

predict_drug() {
    local DRUG_NAME=$1
    
    if [ -z "$DRUG_NAME" ]; then
        echo "Usage: $0 predict <drug_name>"
        echo "Example: $0 predict quetiapine"
        exit 1
    fi
    
    print_header "Drug Repurposing Prediction: $DRUG_NAME"
    
    cd "$PROJECT_ROOT"
    
    python3 -m prediction.demo \
        --predict "$DRUG_NAME" \
        --data-folder "$DATA_DIR"
}

predict_disease() {
    local DISEASE_NAME=$1
    
    if [ -z "$DISEASE_NAME" ]; then
        echo "Usage: $0 treatments <disease_name>"
        echo "Example: $0 treatments depression"
        exit 1
    fi
    
    print_header "Treatment Prediction: $DISEASE_NAME"
    
    cd "$PROJECT_ROOT"
    
    python3 -c "
import sys
sys.path.insert(0, '.')
from prediction import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor(data_folder='$DATA_DIR')
predictor.load_data(use_cache=True)

model_path = '$OUTPUT_DIR/model.pt'
import os
if os.path.exists(model_path):
    predictor.load_model(model_path)
else:
    print('Training quick model...')
    predictor.train(pretrain_epochs=5, finetune_epochs=20, skip_pretrain=True)

results = predictor.predict_treatments('$DISEASE_NAME', top_k=20)

print()
print('=' * 60)
print(f'  Predicted Treatments for: $DISEASE_NAME')
print('=' * 60)

for i, (drug, score, onto) in enumerate(results, 1):
    bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
    print(f'  {i:2d}. {drug[:40]:<40} {bar} {score:.4f}')
"
}

# ==============================================================================
# Explainability Functions (GNNExplainer)
# ==============================================================================

explain_prediction() {
    local DRUG_NAME=$1
    local DISEASE_NAME=$2
    
    if [ -z "$DRUG_NAME" ] || [ -z "$DISEASE_NAME" ]; then
        echo "Usage: $0 explain <drug_name> <disease_name>"
        echo "Example: $0 explain metformin 'type 2 diabetes'"
        exit 1
    fi
    
    print_header "Explaining Prediction: $DRUG_NAME -> $DISEASE_NAME"
    
    cd "$PROJECT_ROOT"
    
    python3 -c "
import sys
sys.path.insert(0, '.')
from prediction import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor(data_folder='$DATA_DIR')
predictor.load_data(use_cache=True)

model_path = '$OUTPUT_DIR/model.pt'
import os
if os.path.exists(model_path):
    predictor.load_model(model_path)
else:
    print('Training quick model...')
    predictor.train(pretrain_epochs=5, finetune_epochs=20, skip_pretrain=True)

# Generate explanation
print()
print('Generating explanation using GNNExplainer...')
print('(This may take a moment)')
print()

try:
    explanation = predictor.explain_prediction(
        drug_name='$DRUG_NAME',
        disease_name='$DISEASE_NAME',
        num_hops=$NUM_HOPS,
        epochs=$EXPLAINER_EPOCHS
    )
    
    print()
    print('=' * 70)
    print()
    
    # Print the human-readable explanation
    if explanation.explanation_text:
        print(explanation.explanation_text)
    else:
        print(f'Prediction Score: {explanation.prediction_score:.4f}')
        print('No detailed explanation available.')
    
    # Save explanation
    import json
    output_file = '$OUTPUT_DIR/explanation_${DRUG_NAME}_${DISEASE_NAME}.json'
    with open(output_file.replace(' ', '_'), 'w', encoding='utf-8') as f:
        json.dump(explanation.to_dict(), f, indent=2, ensure_ascii=False)
    print()
    print('=' * 70)
    print(f'Explanation saved to: {output_file.replace(\" \", \"_\")}')
    
except Exception as e:
    print(f'Error generating explanation: {e}')
    import traceback
    traceback.print_exc()
"
}

explain_batch() {
    local INPUT_FILE=$1
    local OUTPUT_FILE=$2
    
    if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
        echo "Usage: $0 explain-batch <input_file> <output_file>"
        echo "Input file should contain drug,disease pairs, one per line"
        exit 1
    fi
    
    print_header "Batch Explanation Generation"
    
    cd "$PROJECT_ROOT"
    
    python3 -c "
import sys
import json
sys.path.insert(0, '.')
from prediction import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor(data_folder='$DATA_DIR')
predictor.load_data(use_cache=True)

model_path = '$OUTPUT_DIR/model.pt'
import os
if os.path.exists(model_path):
    predictor.load_model(model_path)
else:
    print('Training quick model...')
    predictor.train(pretrain_epochs=5, finetune_epochs=20, skip_pretrain=True)

# Read pairs from file
pairs = []
with open('$INPUT_FILE', 'r') as f:
    for line in f:
        line = line.strip()
        if line and ',' in line:
            drug, disease = line.split(',', 1)
            pairs.append((drug.strip(), disease.strip()))

print(f'Processing {len(pairs)} drug-disease pairs...')

results = []
for drug, disease in pairs:
    print(f'  Explaining: {drug} -> {disease}')
    try:
        explanation = predictor.explain_prediction(drug, disease, epochs=50)
        results.append({
            'drug': drug,
            'disease': disease,
            'prediction_score': explanation.prediction_score,
            'fidelity': explanation.fidelity,
            'top_edges': [(e[0], e[1], float(e[2]), e[3]) for e in explanation.edge_importance[:10]],
            'pathways': explanation.pathways[:3]
        })
    except Exception as e:
        results.append({
            'drug': drug,
            'disease': disease,
            'error': str(e)
        })

with open('$OUTPUT_FILE', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to: $OUTPUT_FILE')
"
}

# ==============================================================================
# Evaluation Functions
# ==============================================================================

evaluate_model() {
    print_header "Evaluating Model Performance"
    
    cd "$PROJECT_ROOT"
    
    python3 -m prediction.demo \
        --evaluate \
        --data-folder "$DATA_DIR"
    
    print_step "Evaluation results saved to: $OUTPUT_DIR/evaluation_results.json"
}

# ==============================================================================
# Demo Functions
# ==============================================================================

run_demo() {
    print_header "Running Interactive Demo"
    
    cd "$PROJECT_ROOT"
    
    python3 -m prediction.demo \
        --demo \
        --data-folder "$DATA_DIR"
}

run_interactive() {
    print_header "Interactive Prediction Mode"
    
    cd "$PROJECT_ROOT"
    
    python3 -m prediction.demo \
        --data-folder "$DATA_DIR"
}

# ==============================================================================
# Quick Mode
# ==============================================================================

quick_demo() {
    print_header "Quick Demo Mode"
    print_step "Training a quick model and running demo..."
    
    cd "$PROJECT_ROOT"
    
    python3 -m prediction.demo \
        --quick \
        --demo \
        --data-folder "$DATA_DIR"
}

# ==============================================================================
# Batch Prediction
# ==============================================================================

batch_predict() {
    local INPUT_FILE=$1
    local OUTPUT_FILE=$2
    
    if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
        echo "Usage: $0 batch <input_file> <output_file>"
        echo "Input file should contain one drug name per line"
        exit 1
    fi
    
    print_header "Batch Prediction"
    
    cd "$PROJECT_ROOT"
    
    python3 -c "
import sys
import json
sys.path.insert(0, '.')
from prediction import DrugRepurposingPredictor

predictor = DrugRepurposingPredictor(data_folder='$DATA_DIR')
predictor.load_data(use_cache=True)

model_path = '$OUTPUT_DIR/model.pt'
import os
if os.path.exists(model_path):
    predictor.load_model(model_path)
else:
    print('Training quick model...')
    predictor.train(pretrain_epochs=5, finetune_epochs=20, skip_pretrain=True)

results = {}
with open('$INPUT_FILE', 'r') as f:
    drugs = [line.strip() for line in f if line.strip()]

for drug in drugs:
    print(f'Processing: {drug}')
    try:
        predictions = predictor.predict_repurposing(drug, top_k=20)
        results[drug] = [{'disease': d, 'score': s} for d, s, _ in predictions]
    except Exception as e:
        results[drug] = {'error': str(e)}

with open('$OUTPUT_FILE', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to: $OUTPUT_FILE')
"
}

# ==============================================================================
# Main
# ==============================================================================

show_help() {
    echo "MDKG Drug Repurposing Prediction Pipeline"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train [options]           Train the drug repurposing model"
    echo "  predict <drug>            Predict new indications for a drug"
    echo "  treatments <disease>      Predict treatments for a disease"
    echo "  explain <drug> <disease>  Explain a drug-disease prediction"
    echo "  explain-batch <in> <out>  Batch explanation generation"
    echo "  evaluate                  Evaluate model performance"
    echo "  demo                      Run demo with sample predictions"
    echo "  interactive               Start interactive prediction mode"
    echo "  quick                     Quick demo (train + predict)"
    echo "  batch <in> <out>          Batch prediction from file"
    echo "  check                     Check dependencies and data"
    echo ""
    echo "Training Options:"
    echo "  --quick              Quick training mode (fewer epochs)"
    echo "  --skip-pretrain      Skip pre-training phase"
    echo "  --pretrain-epochs N  Set pre-training epochs (default: $PRETRAIN_EPOCHS)"
    echo "  --finetune-epochs N  Set fine-tuning epochs (default: $FINETUNE_EPOCHS)"
    echo "  --data-source SOURCE Set data source for training:"
    echo "                         sampled  - Active Learning sampled data (default)"
    echo "                         full     - Full MDKG dataset"
    echo "                         full_aug - Full augmented dataset"
    echo "                         train    - Training split only"
    echo ""
    echo "Explainability Options:"
    echo "  The explain command uses GNNExplainer to identify important"
    echo "  subgraph structures that contribute to predictions."
    echo "  Reference: https://arxiv.org/abs/1903.03894"
    echo ""
    echo "Examples:"
    echo "  $0 train --quick"
    echo "  $0 train --data-source full"
    echo "  $0 predict quetiapine"
    echo "  $0 treatments depression"
    echo "  $0 explain metformin 'type 2 diabetes'"
    echo "  $0 batch drugs.txt results.json"
}

main() {
    local CMD=${1:-help}
    shift || true
    
    case $CMD in
        train)
            check_dependencies
            check_data
            train_model "$@"
            ;;
        predict)
            predict_drug "$@"
            ;;
        treatments)
            predict_disease "$@"
            ;;
        explain)
            explain_prediction "$@"
            ;;
        explain-batch)
            explain_batch "$@"
            ;;
        evaluate)
            evaluate_model
            ;;
        demo)
            run_demo
            ;;
        interactive)
            run_interactive
            ;;
        quick)
            check_dependencies
            check_data
            quick_demo
            ;;
        batch)
            batch_predict "$@"
            ;;
        check)
            check_dependencies
            check_data
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $CMD"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
