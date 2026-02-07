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

# Data source (can be overridden with --data-source)
# Options: sampled, full, full_aug, train
DATA_SOURCE="sampled"
MODEL_FILE=""

# ==============================================================================
# Helper Functions
# ==============================================================================

##
# Print section header
# @param $1 Header text
##
print_header() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
    echo ""
}

##
# Print step with timestamp
# @param $1 Step description
##
print_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

##
# Check if dependencies are installed
# @brief Verifies python3, torch, and dgl are available
##
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

##
# Check if input data exists
# @brief Verifies availability of triplets and entity linking files
##
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

##
# Train the drug repurposing model
# @param ... Training arguments (pretrain-epochs, finetune-epochs, etc.)
##
train_model() {
    print_header "Training Drug Repurposing Model"
    
    local PRETRAIN=$PRETRAIN_EPOCHS
    local FINETUNE=$FINETUNE_EPOCHS
    local SKIP_PRETRAIN=""
    local LOCAL_DATA_SOURCE="$DATA_SOURCE"
    
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
                LOCAL_DATA_SOURCE=$2
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    print_step "Data source: $LOCAL_DATA_SOURCE"
    print_step "Pre-training epochs: $PRETRAIN"
    print_step "Fine-tuning epochs: $FINETUNE"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    cd "$PROJECT_ROOT"
    
    # Construct base command
    CMD="python3 -m prediction.demo \
        --train \
        --data-folder \"$DATA_DIR\" \
        --output-dir \"$OUTPUT_DIR\" \
        --pretrain-epochs $PRETRAIN \
        --finetune-epochs $FINETUNE \
        --data-source \"$LOCAL_DATA_SOURCE\" \
        $SKIP_PRETRAIN"
        
    # Add model file arg if provided (though mostly used for loading, good for consistency)
    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
    
    print_step "Model saved to: $OUTPUT_DIR/model.pt"
}

# ==============================================================================
# Prediction Functions
# ==============================================================================

##
# Predict repurposing candidates for a specific drug
# @param $1 Drug name
##
predict_drug() {
    local DRUG_NAME=$1
    
    if [ -z "$DRUG_NAME" ]; then
        echo "Usage: $0 predict <drug_name>"
        echo "Example: $0 predict quetiapine"
        exit 1
    fi
    
    print_header "Drug Repurposing Prediction: $DRUG_NAME"
    
    cd "$PROJECT_ROOT"
    
    CMD="python3 -m prediction.demo \
        --predict \"$DRUG_NAME\" \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""
        
    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
}

##
# Predict potential treatments for a specific disease
# @param $1 Disease name
##
predict_disease() {
    local DISEASE_NAME=$1
    
    if [ -z "$DISEASE_NAME" ]; then
        echo "Usage: $0 treatments <disease_name>"
        echo "Example: $0 treatments depression"
        exit 1
    fi
    
    print_header "Treatment Prediction: $DISEASE_NAME"
    
    cd "$PROJECT_ROOT"
    
    CMD="python3 -m prediction.demo \
        --treatments \"$DISEASE_NAME\" \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""
        
    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
}

# ==============================================================================
# Explainability Functions (GNNExplainer)
# ==============================================================================

##
# Explain a prediction for a drug-disease pair
# @param $1 Drug name
# @param $2 Disease name
##
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
    
    CMD="python3 -m prediction.demo \
        --explain \"$DRUG_NAME\" \"$DISEASE_NAME\" \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""
        
    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
}

##
# Batch generate explanations for drug-disease pairs
# @param $1 Input file (csv: drug,disease)
# @param $2 Output file (json)
##
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
    
    CMD="python3 -m prediction.demo \
        --explain-batch \"$INPUT_FILE\" \"$OUTPUT_FILE\" \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""

    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
}

# ==============================================================================
# Evaluation Functions
# ==============================================================================

##
# Evaluate model performance using standard metrics
##
evaluate_model() {
    print_header "Evaluating Model Performance"
    
    cd "$PROJECT_ROOT"
    
    CMD="python3 -m prediction.demo \
        --evaluate \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""

    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
    
    print_step "Evaluation results saved to: $OUTPUT_DIR/evaluation_results.json"
}

# ==============================================================================
# Demo Functions
# ==============================================================================

##
# Run demo with sample queries
##
run_demo() {
    print_header "Running Interactive Demo"
    
    cd "$PROJECT_ROOT"
    
    CMD="python3 -m prediction.demo \
        --demo \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""
        
    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
}

##
# Start interactive shell
##
run_interactive() {
    print_header "Interactive Prediction Mode"
    
    cd "$PROJECT_ROOT"
    
    CMD="python3 -m prediction.demo \
        --interactive \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""

    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
}

# ==============================================================================
# Quick Mode
# ==============================================================================

##
# Quick demo for testing (train + demo)
##
quick_demo() {
    print_header "Quick Demo Mode"
    print_step "Training a quick model and running demo..."
    
    cd "$PROJECT_ROOT"
    
    python3 -m prediction.demo \
        --quick \
        --demo \
        --data-folder "$DATA_DIR" \
        --data-source "$DATA_SOURCE"
}

# ==============================================================================
# Batch Prediction
# ==============================================================================

##
# Batch predict candidates for a list of drugs
# @param $1 Input file (drug per line)
# @param $2 Output file (json)
##
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
    
    CMD="python3 -m prediction.demo \
        --batch-predict \"$INPUT_FILE\" \"$OUTPUT_FILE\" \
        --data-folder \"$DATA_DIR\" \
        --data-source \"$DATA_SOURCE\""

    if [ ! -z "$MODEL_FILE" ]; then
        CMD="$CMD --model-path \"$MODEL_FILE\""
    fi
    
    eval $CMD
}

# ==============================================================================
# Main
# ==============================================================================

show_help() {
    echo "MDKG Drug Repurposing Prediction Pipeline"
    echo ""
    echo "Usage: $0 [--data-source SOURCE] [--model-file FILE] <command> [options]"
    echo ""
    echo "Global Options:"
    echo "  --data-source SOURCE  Set data source (sampled, full, full_aug, train)"
    echo "  --model-file FILE     Path to specific model file (optional)"
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
    echo ""
    echo "Example with explicit model:"
    echo "  $0 --data-source full --model-file my_model.pt predict quetiapine"
}

main() {
    # Parse global options first
    local args=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --data-source)
                DATA_SOURCE="$2"
                shift 2
                ;;
            --model-file)
                MODEL_FILE="$2"
                shift 2
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done
    
    # Restore positional parameters
    set -- "${args[@]}"
    
    local CMD=${1:-help}
    shift || true
    
    case $CMD in
        train)
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
            evaluate_model "$@"
            ;;
        demo)
            run_demo "$@"
            ;;
        interactive)
            run_interactive "$@"
            ;;
        quick)
            quick_demo "$@"
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
            echo "Error: Unknown command '$CMD'"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
