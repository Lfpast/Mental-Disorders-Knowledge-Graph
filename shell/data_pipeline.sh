#!/bin/bash

# ==============================================================================
# MDKG Data Pipeline Script
# ==============================================================================
# This script automates the Data Preparation process:
# 1. Downloads Dataset (using data_downloader.py)
# 2. Generates Input - Preprocessing & Splitting (using generate_input.py)
# 3. Generates Augmented Input (using generate_augmented_input.py)
#
# Usage: ./shell/data_pipeline.sh
# ==============================================================================

# Ensure we are in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SYNSPERT_DIR="$PROJECT_ROOT/models/SynSpERT"

echo "=============================================================================="
echo "Starting MDKG Data Pipeline"
echo "Project Root: $PROJECT_ROOT"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Step 1: Download Data
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 1/3] Downloading Dataset..."
echo "Running: python models/SynSpERT/data_downloader.py"
echo "------------------------------------------------------------------------------"

python "$SYNSPERT_DIR/data_downloader.py"

if [ $? -ne 0 ]; then
    echo "[ERROR] Dataset download failed."
    exit 1
fi

echo "[OK] Download complete."

# ------------------------------------------------------------------------------
# Step 2: Generate Input (Preprocess and Split)
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 2/3] Generating Input (Preprocessing & Splitting)..."
echo "Running: python generate_input.py (inside models/SynSpERT)"
echo "------------------------------------------------------------------------------"

# Change directory to SynSpERT so relative paths in python script work
cd "$SYNSPERT_DIR"
python generate_input.py

if [ $? -ne 0 ]; then
    echo "[ERROR] Input generation failed."
    exit 1
fi

echo "[OK] Input generation complete."

# ------------------------------------------------------------------------------
# Step 3: Generate Augmented Input
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 3/3] Generating Augmented Input..."
echo "Running: python generate_augmented_input.py (inside models/SynSpERT)"
echo "------------------------------------------------------------------------------"

# Note: We are already in SYNSPERT_DIR from previous step, but good practice to ensure
cd "$SYNSPERT_DIR"
python generate_augmented_input.py

if [ $? -ne 0 ]; then
    echo "[ERROR] Data Augmentation failed."
    exit 1
fi

echo "[OK] Data Augmentation complete."

echo ""
echo "=============================================================================="
echo "MDKG Data Pipeline Completed Successfully!"
echo "Check the 'models/InputsAndOutputs/input/' directory for results."
echo "=============================================================================="
