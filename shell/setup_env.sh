#!/bin/bash

# ==============================================================================
# MDKG Environment Setup Script
# ==============================================================================
# This script prepares the Python environment for MDKG using Conda.
#
# Usage:
#   1. Create environment:  conda env create -f environment.yaml
#   2. Activate:            conda activate MDKG
#   3. Post-install:        ./shell/setup_env.sh
#
# Or run full setup (creates conda env + post-install):
#   ./shell/setup_env.sh --full
#
# ==============================================================================

set -e  # Exit on error

# Ensure we are in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=============================================================================="
echo "Starting MDKG Environment Setup"
echo "Project Root: $PROJECT_ROOT"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Check for --full flag (create conda env from scratch)
# ------------------------------------------------------------------------------
FULL_SETUP=false
if [[ "$1" == "--full" ]]; then
    FULL_SETUP=true
fi

# ------------------------------------------------------------------------------
# Step 1: Create/Update Conda Environment (only with --full flag)
# ------------------------------------------------------------------------------
if [ "$FULL_SETUP" = true ]; then
    echo ""
    echo "[Setup Step 1/5] Setting up Conda Environment..."
    echo "------------------------------------------------------------------------------"

    if ! command -v conda &> /dev/null; then
        echo "[ERROR] Conda not found. Please install Miniconda or Anaconda first."
        echo "        https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    if [ ! -f "environment.yaml" ]; then
        echo "[ERROR] environment.yaml not found!"
        exit 1
    fi

    echo "Found environment.yaml - using Conda for dependency management"
    
    # Check if MDKG environment exists
    if conda env list | grep -q "^MDKG "; then
        echo "Environment 'MDKG' exists. Updating..."
        conda env update -f environment.yaml --prune
    else
        echo "Creating new 'MDKG' environment..."
        conda env create -f environment.yaml
    fi
    
    echo ""
    echo "[OK] Conda environment ready."
    echo "[INFO] Activate with: conda activate MDKG"
else
    echo "[INFO] Skipping conda env creation. Use --full to create/update conda environment."
fi

# ------------------------------------------------------------------------------
# Step 2: Install PyTorch and DGL with correct CUDA version
# ------------------------------------------------------------------------------
echo ""
echo "[Setup Step 2/5] Installing PyTorch and DGL with CUDA 12.1..."
echo "------------------------------------------------------------------------------"

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/cu121/repo.html

echo "[OK] PyTorch and DGL installed."

# ------------------------------------------------------------------------------
# Step 3: Verify DGL Installation
# ------------------------------------------------------------------------------
echo ""
echo "[Setup Step 3/5] Verifying Installation..."
echo "------------------------------------------------------------------------------"

python -c "
import sys
try:
    import dgl
    import torch
    print(f'✓ DGL version: {dgl.__version__}')
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA version: {torch.version.cuda}')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'✗ DGL import failed: {e}')
    print('  Please ensure conda environment is activated: conda activate MDKG')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "[WARNING] DGL verification failed. You may need to activate the conda environment."
fi

# ------------------------------------------------------------------------------
# Step 4: Download Spacy Models
# ------------------------------------------------------------------------------
echo ""
echo "[Setup Step 4/5] Downloading Spacy Models..."
echo "------------------------------------------------------------------------------"

# Install standard English model
echo "Downloading en_core_web_sm..."
python -m spacy download en_core_web_sm

# Check if SciSpacy model is already installed
if python -c "import spacy; spacy.load('en_core_sci_sm')" 2>/dev/null; then
    echo "en_core_sci_sm already installed."
else
    echo "Downloading en_core_sci_sm..."
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
fi

echo "[OK] Spacy models installed."

# ------------------------------------------------------------------------------
# Step 5: Download NLTK Data
# ------------------------------------------------------------------------------
echo ""
echo "[Setup Step 5/5] Downloading NLTK Data..."
echo "------------------------------------------------------------------------------"

python -c "
import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
print('✓ NLTK data downloaded')
"

echo "[OK] NLTK data downloaded."

# ------------------------------------------------------------------------------
# Final Summary
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "Environment Setup Completed Successfully!"
echo "=============================================================================="
echo ""
echo "Quick verification:"
python -c "
import torch
import transformers
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
try:
    import dgl
    print(f'  DGL: {dgl.__version__}')
except:
    print('  DGL: Not available')
try:
    import spacy
    print(f'  Spacy: {spacy.__version__}')
except:
    pass
"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate MDKG"
echo "  2. Run prediction demo:  python -m prediction.demo"
echo "==============================================================================" 
