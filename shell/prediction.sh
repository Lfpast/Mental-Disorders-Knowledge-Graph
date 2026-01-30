#!/bin/bash
# ==============================================================================
# MDKG Drug Repurposing Prediction Pipeline
# ==============================================================================
#
# This script provides automation for training and running the drug repurposing
# prediction module based on TxGNN methodology.
#
# Features:
# - GNN model training with pre-training and fine-tuning
# - Drug repurposing predictions
# - Model evaluation with disease-centric metrics
# - Interactive prediction mode
#
# Usage:
#   ./shell/prediction.sh train          # Train the model
#   ./shell/prediction.sh predict <drug> # Predict for a drug
#   ./shell/prediction.sh evaluate       # Evaluate model
#   ./shell/prediction.sh demo           # Run interactive demo
#   ./shell/prediction.sh quick          # Quick training for testing
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

# Training configuration
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=200
HIDDEN_DIM=128
LEARNING_RATE=0.0005
BATCH_SIZE=1024

# Quick mode settings
QUICK_PRETRAIN_EPOCHS=5
QUICK_FINETUNE_EPOCHS=20

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
            *)
                shift
                ;;
        esac
    done
    
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
    echo "  train [options]      Train the drug repurposing model"
    echo "  predict <drug>       Predict new indications for a drug"
    echo "  treatments <disease> Predict treatments for a disease"
    echo "  evaluate             Evaluate model performance"
    echo "  demo                 Run demo with sample predictions"
    echo "  interactive          Start interactive prediction mode"
    echo "  quick                Quick demo (train + predict)"
    echo "  batch <in> <out>     Batch prediction from file"
    echo "  check                Check dependencies and data"
    echo ""
    echo "Training Options:"
    echo "  --quick              Quick training mode (fewer epochs)"
    echo "  --skip-pretrain      Skip pre-training phase"
    echo "  --pretrain-epochs N  Set pre-training epochs"
    echo "  --finetune-epochs N  Set fine-tuning epochs"
    echo ""
    echo "Examples:"
    echo "  $0 train --quick"
    echo "  $0 predict quetiapine"
    echo "  $0 treatments depression"
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
