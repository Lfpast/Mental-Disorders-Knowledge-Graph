#!/bin/bash

# ==============================================================================
# MDKG Triplets Extraction & Refinement Script
# ==============================================================================
# This script automates the Triplets Extraction and Prompt Generation process:
# 1. extract_triplets_from_predictions: Extracts triplets from active learning samples
# 2. generate_refinement_prompt: Generates LLM prompts for refinement
# 3. Saves results to models/InputsAndOutputs/output/triplet_refinement/
#
# Usage: ./shell/triplets_extract.sh
# ==============================================================================

# Ensure we are in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================================================="
echo "Starting MDKG Triplets Extraction Pipeline"
echo "Project Root: $PROJECT_ROOT"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Step 1: Run Triplets Refinement Prompt Generator
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 1/1] Extracting Triplets & Generating Prompts..."
echo "Running: python triplets_refine_prompt.py"
echo "------------------------------------------------------------------------------"

cd "$PROJECT_ROOT"
python triplets_refine_prompt.py

if [ $? -ne 0 ]; then
    echo "[ERROR] Triplets extraction failed."
    exit 1
fi

echo "[OK] Triplets extraction completed successfully."

echo ""
echo "=============================================================================="
echo "Triplets Extraction Pipeline Completed Successfully!"
echo "Check 'models/InputsAndOutputs/output/triplet_refinement/' for results."
echo "=============================================================================="
