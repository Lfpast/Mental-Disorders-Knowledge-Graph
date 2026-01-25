#!/bin/bash

# ==============================================================================
# MDKG Active Learning Pipeline Script
# ==============================================================================
# This script automates the Active Learning process:
# 1. Feature Generation (using generate_al_features.py)
# 2. Sample Selection (using active_learning.py)
#
# Usage: ./shell/active_learning.sh
# ==============================================================================

# Ensure we are in the project root directory
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assuming the script is in /path/to/project/shell, go up one level to project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================================================="
echo "Starting MDKG Active Learning Pipeline"
echo "Project Root: $PROJECT_ROOT"
echo "=============================================================================="

# Activate environment if needed (Optional, uncomment if you use conda/venv)
# source /path/to/venv/bin/activate
# conda activate mdkg_env

# ------------------------------------------------------------------------------
# Step 1: Generate Active Learning Features
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 1/2] Generating Features..."
echo "Running: python models/SynSpERT/generate_al_features.py"
echo "------------------------------------------------------------------------------"

cd "$PROJECT_ROOT"
python models/SynSpERT/generate_al_features.py

if [ $? -ne 0 ]; then
    echo "Error: Feature generation failed."
    exit 1
fi

echo "Feature generation completed successfully."

# ------------------------------------------------------------------------------
# Step 2: Run Active Learning Sample Selection
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 2/2] Selecting Samples..."
echo "Running: python active_learning.py"
echo "------------------------------------------------------------------------------"

cd "$PROJECT_ROOT"
python Active_learning.py

if [ $? -ne 0 ]; then
    echo "Error: Sample selection failed."
    exit 1
fi

echo "Sample selection completed successfully."

echo ""
echo "=============================================================================="
echo "Active Learning Pipeline Completed Successfully!"
echo "Check the 'models/InputsAndOutputs/output/' directory for results."
echo "=============================================================================="
