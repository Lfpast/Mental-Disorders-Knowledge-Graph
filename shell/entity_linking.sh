#!/bin/bash

# ==============================================================================
# MDKG Entity Linking Pipeline Script
# ==============================================================================
# This script automates the Entity Linking pipeline:
# 1. extract_entities.py: Extracts unique entities from training data
# 2. ontologies_downloader.py: Downloads official OBO ontology files
# 3. generate_ontology_embeddings.py: Generates embeddings for ontology terms
# 4. entity_linking.py: Links entities to ontology terms using SapBERT
#
# Usage: ./shell/entity_linking.sh
# ==============================================================================

# Ensure we are in the project root directory
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assuming the script is in /path/to/project/shell, go up one level to project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================================================="
echo "Starting MDKG Entity Linking Pipeline"
echo "Project Root: $PROJECT_ROOT"
echo "=============================================================================="

# Activate environment if needed (Optional, uncomment if you use conda/venv)
# source /path/to/venv/bin/activate
# conda activate mdkg_env

cd "$PROJECT_ROOT"

# ------------------------------------------------------------------------------
# Step 1: Extract Entities
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 1/4] Extracting Entities..."
echo "Running: python extract_entities.py"
echo "------------------------------------------------------------------------------"

python extract_entities.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Entity extraction failed."
    exit 1
fi

echo "Entity extraction completed successfully."

# ------------------------------------------------------------------------------
# Step 2: Download Ontologies
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 2/4] Downloading Ontologies..."
echo "Running: python models/SynSpERT/ontologies_downloader.py"
echo "------------------------------------------------------------------------------"

python models/SynSpERT/ontologies_downloader.py

if [ $? -ne 0 ]; then
    echo "Error: Ontology download failed."
    exit 1
fi

echo "Ontology download completed successfully."

# ------------------------------------------------------------------------------
# Step 3: Generate Ontology Embeddings
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 3/4] Generating Ontology Embeddings..."
echo "Running: python models/SynSpERT/generate_ontology_embeddings.py"
echo "------------------------------------------------------------------------------"

python models/SynSpERT/generate_ontology_embeddings.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Ontology embedding generation failed."
    exit 1
fi

echo "Ontology embedding generation completed successfully."

# ------------------------------------------------------------------------------
# Step 4: Run Entity Linking
# ------------------------------------------------------------------------------
echo ""
echo "[Pipeline Step 4/4] Linking Entities..."
echo "Running: python entity_linking.py"
echo "------------------------------------------------------------------------------"

python entity_linking.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Entity linking failed."
    exit 1
fi

echo "Entity linking completed successfully."

echo ""
echo "=============================================================================="
echo "Entity Linking Pipeline Completed Successfully!"
echo "Check the 'models/InputsAndOutputs/output/' directory for results."
echo "=============================================================================="
