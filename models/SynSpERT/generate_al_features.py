#!/usr/bin/env python3
"""
Generate Active Learning Features
==================================
This script runs model evaluation on the full dataset to generate
entropy and embedding features needed for active learning.
"""

import os
import sys
import subprocess

# Determine the absolute path of the script directory (models/SynSpERT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Change working directory to SCRIPT_DIR to ensure relative imports (like import Runner) work
os.chdir(SCRIPT_DIR)

# Add SCRIPT_DIR to sys.path if not present
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print("=" * 70)
print("Active Learning Feature Generation")
print("=" * 70)
print("")
print("This script will run model evaluation to generate:")
print("  • entropy_relation_run_v1.pt")
print("  • entropy_entities_run_v1.pt")  
print("  • pooler_output_run_v1.pt")
print("  • labelprediction_run_v1.pt")
print("")

# Import Runner for execution
try:
    import Runner
    print("[✓] Imported Runner module")
except ImportError as e:
    print(f"[✗] Failed to import Runner: {e}")
    sys.exit(1)

# Configuration
BASE_DIR = "../InputsAndOutputs/"
BERT_PATH = BASE_DIR + "pretrained"
TOKENIZER_PATH = BERT_PATH
MODEL_TYPE = "syn_spert"
CONFIG_PATH = BASE_DIR + "configs/config-coder.json"

LOG_PATH = BASE_DIR + "output/log/"
SAVE_PATH = BASE_DIR + "output/save/"
CACHE_PATH = BASE_DIR + "output/cache/"

# Use the best model from training
names = 'run_v1'    # Make sure this name is consistent with the run name while training
MODEL_PATH = SAVE_PATH + names + "/final_model"

TYPES_PATH = BASE_DIR + "configs/DPKG_types_Cor4.json"
TEST_PATH = "../InputsAndOutputs/input/md_KG_all_0217_agu.json"

SEED = 11

USE_POS = '--use_pos'
USE_ENTITY_CLF = 'logits'

print(f"[*] Configuration:")
print(f"    Model: {MODEL_PATH}")
print(f"    Test data: {TEST_PATH}")
print(f"    Names: {names}")
print("")

# Build evaluation command
eval_args_list = f"eval --model_type {MODEL_TYPE} --label 'scierc_train' \
  --model_path {MODEL_PATH} --tokenizer_path {MODEL_PATH} \
  --dataset_path {TEST_PATH} \
  --types_path {TYPES_PATH} --cache_path {CACHE_PATH} \
  --size_embedding 25 \
  {USE_POS} --pos_embedding 25 \
  --use_entity_clf {USE_ENTITY_CLF} \
  --save_features \
  --eval_batch_size 1 --lr 5e-5 --lr_warmup 0.1 \
  --weight_decay 0.01 --max_grad_norm 1.0 --prop_drop 0.1 \
  --max_span_size 10 --rel_filter_threshold 0.4 --max_pairs 1000 \
  --sampling_processes 4 --sampling_limit 100 \
  --store_predictions True --store_examples True \
  --log_path {LOG_PATH} --save_path {SAVE_PATH} \
  --max_seq_length 512 --config_path {CONFIG_PATH} \
  --Names {names} \
  --gradient_accumulation_steps 1 --wordpiece_aligned_dep_graph --seed {SEED}"

input_args_list = eval_args_list.split()

print("[*] Running model evaluation to generate features...")
print("    This may take a few minutes...")
print("")

try:
    r = Runner.Runner()
    r.run(input_args_list)
    print("")
    print("=" * 70)
    print("[✓] Feature generation completed successfully!")
    print("=" * 70)
    print("")
    print("Generated files saved in:")
    print(f"  {LOG_PATH}{names}/")
    print("")
    
except Exception as e:
    print("")
    print("=" * 70)
    print(f"[✗] Error during feature generation: {e}")
    print("=" * 70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
