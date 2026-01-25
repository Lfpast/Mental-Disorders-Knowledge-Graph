#!/bin/bash

# ==============================================================================
# MDKG Training Pipeline Script
# ==============================================================================
# This script automates the Training process:
# 1. Checks/Downloads Pretrained Model (using model_downloader.py)
# 2. Runs Main Training Loop (using main.py)
#
# Usage: ./shell/training.sh
# ==============================================================================

# Ensure we are in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SYNSPERT_DIR="$PROJECT_ROOT/models/SynSpERT"

echo "=============================================================================="
echo "Starting MDKG Training Pipeline"
echo "Project Root: $PROJECT_ROOT"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Step 1: Checking/Downloading Pretrained Model
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 1/2] Checking/Downloading Pretrained Model..."
echo "Running: python models/SynSpERT/model_downloader.py"
echo "------------------------------------------------------------------------------"

python "$SYNSPERT_DIR/model_downloader.py"

if [ $? -ne 0 ]; then
    echo "[ERROR] Model download failed."
    exit 1
fi

echo "[OK] Model setup complete."

# ------------------------------------------------------------------------------
# Step 2: Main Training Loop
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 2/2] Starting Training Process..."
echo "Running: python main.py (inside models/SynSpERT)"
echo "------------------------------------------------------------------------------"

# Change directory to SynSpERT so relative paths in python script work
cd "$SYNSPERT_DIR"
python main.py

if [ $? -ne 0 ]; then
    echo "[ERROR] Training process failed."
    exit 1
fi

echo "[OK] Training process finished."

echo ""
echo "=============================================================================="
echo "MDKG Training Pipeline Completed Successfully!"
echo "Check the 'models/InputsAndOutputs/output/' directory for results."
echo "=============================================================================="
