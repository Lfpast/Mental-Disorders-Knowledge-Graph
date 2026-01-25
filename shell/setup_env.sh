#!/bin/bash

# ==============================================================================
# MDKG Environment Setup Script
# ==============================================================================
# This script prepares the Python environment for MDKG:
# 1. Installs Python dependencies from requirements.txt
# 2. Downloads required Spacy models (en_core_web_sm, en_core_sci_sm)
# 3. Downloads NLTK data (punkt)
#
# Usage: ./shell/setup_env.sh
# ==============================================================================

# Ensure we are in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assuming the script is in /path/to/project/shell, go up one level to project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================================================="
echo "Starting MDKG Environment Setup"
echo "Project Root: $PROJECT_ROOT"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Step 1: Install Python Dependencies (Conda + Pip)
# ------------------------------------------------------------------------------
echo ""
echo "[Setup Step 1/3] Installing Python Dependencies..."

# 1.1 Install FAISS (GPU preferred) using Conda if available
# FAISS-GPU is best installed via Conda to ensure CUDA compatibility
if command -v conda &> /dev/null; then
    echo "Conda detected. Installing faiss-gpu via conda..."
    # Try install from pytorch channel or conda-forge
    conda install -y -c pytorch -c nvidia faiss-gpu
    if [ $? -ne 0 ]; then
         echo "[WARNING] Conda install failed. Trying alternatives..."
    fi
else
    echo "[WARNING] Conda not found. Skipping conda-specific installs."
    echo "Note: faiss-gpu installation via pip might be unstable or require system libraries."
fi

# 1.2 Install remaining dependencies via Pip
echo "Running: pip install -r requirements.txt"
echo "------------------------------------------------------------------------------"

cd "$PROJECT_ROOT"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found in root. Skipping pip install."
fi

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi

echo "[OK] Dependencies installed."

# ------------------------------------------------------------------------------
# Step 2: Download Spacy Models
# ------------------------------------------------------------------------------
echo ""
echo "[Setup Step 2/3] Downloading Spacy Models..."
echo "------------------------------------------------------------------------------"

# Install standard English model
echo "Downloading en_core_web_sm..."
python -m spacy download en_core_web_sm

# Install SciSpacy model (Biomedical)
# Note: Version must be compatible with spacy. 
# For scispacy>=0.5.0, we usually use en_core_sci_sm-0.5.x
SCISPACY_MODEL_URL="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"

echo "Downloading en_core_sci_sm from $SCISPACY_MODEL_URL..."
pip install "$SCISPACY_MODEL_URL"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to download Spacy models."
    exit 1
fi

echo "[OK] Spacy models installed."

# ------------------------------------------------------------------------------
# Step 3: Download NLTK Data
# ------------------------------------------------------------------------------
echo ""
echo "[Setup Step 3/3] Downloading NLTK Data..."
echo "------------------------------------------------------------------------------"

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to download NLTK data."
    exit 1
fi

echo "[OK] NLTK data downloaded."

echo ""
echo "=============================================================================="
echo "Environment Setup Completed Successfully!"
echo "You are ready to run the MDKG pipelines."
echo "=============================================================================="
