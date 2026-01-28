"""
MDKG Active Learning Script
===========================
Select the most informative samples using entropy-based methods.

This script performs the following high-level steps:
1. Loads precomputed feature tensors and entropy scores saved by
   `models/SynSpERT/generate_al_features.py`.
2. Runs clustering (K-Means or weighted K-Means) on the feature
   embeddings to partition the unlabeled pool.
3. Scores clusters using entropy and class distribution to select
   high-information samples.
4. Saves a JSON file containing the selected samples.

Inputs (paths & types):
- root data directory: str (constants defined in the module)
- required files (torch tensors saved with torch.save):
  - `entropy_relation_{Names}.pt` (torch.Tensor of shape [n_sample])
  - `entropy_entities_{Names}.pt` (torch.Tensor of shape [n_sample])
  - `labelprediction_{Names}.pt` (torch.Tensor with label predictions)
  - `pooler_output_{Names}.pt` (torch.Tensor of shape [n_sample, d])

Outputs:
- `sampling_json_{SaveNames}.json` (JSON array of selected sample objects)
- Saved PyTorch tensors for debugging: `weighted_embedding_{SaveNames}.pt`, `class_number_{SaveNames}.pt`

Return / Side-effects:
- This module, when executed, writes files to disk and prints a summary
  to stdout. There is no return value when run as a script.

Configuration:
- Names: str - Model run name (default 'run_v1')
- topK: int - Number of clusters to select (default 20)
- sample_per_group: int - Samples per cluster (default 10)
- ncentroids: int - Number of K-Means clusters (default 20)
"""

import json
import numpy as np
import faiss
import torch
from sklearn.cluster import MiniBatchKMeans
import copy
import os

# ==================== Configuration ====================
# Model Name
Names = 'run_v1'
SaveNames = 'run_v1_sampled'

# Path Configuration using relative paths for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Assuming this script is at the project root
# Log path where features are stored
root_path = os.path.join(BASE_DIR, "models", "InputsAndOutputs", "output", "log", Names) 
# Directory to save the sampled json file
output_dir = os.path.join(BASE_DIR, "models", "InputsAndOutputs", "output")

# All data path
all_file_json_path = os.path.join(BASE_DIR, "models", "InputsAndOutputs", "input", "md_KG_all_0217_agu.json")

# Sample Configuration
n_sample = 6864  # Total number of samples

# Active Learning Hyperparameters
topK = 20  # Number of clusters to select
sample_per_group = 10  # Samples per cluster (Total selected: topK * sample_per_group = 200)
beta = 0.1  # Class entropy weight
gamma = 0.1  # Entity entropy weight
weight = True  # Use weighted K-Means
ncentroids = 20  # Number of K-Means clusters

print("=" * 70)
print("MDKG Active Learning - Sample Selection")
print("=" * 70)
print(f"Config: topK={topK}, sample_per_group={sample_per_group}")
print(f"Total samples to select: {topK * sample_per_group}")
print("=" * 70)

# ==================== Step 1: Load Feature Files ====================
print("\n[Step 1/4] Loading feature files...")
try:
    entropy_relation = torch.load(
        os.path.join(root_path, f'entropy_relation_{Names}.pt'), 
        map_location=torch.device('cpu')
    ).reshape(n_sample, )
    
    entropy_entities = torch.load(
        os.path.join(root_path, f'entropy_entities_{Names}.pt'), 
        map_location=torch.device('cpu')
    ).reshape(n_sample, )
    
    entropy = entropy_relation + gamma * entropy_entities
    entropy = entropy.cpu().numpy() if isinstance(entropy, torch.Tensor) else entropy
    torch.save(torch.from_numpy(entropy), os.path.join(root_path, f'total_entropy_{Names}.pt'))
    
    unlabeled_pred = torch.load(
        os.path.join(root_path, f'labelprediction_{Names}.pt'), 
        map_location=torch.device('cpu')
    )
    
    unlabeled_feat = torch.load(
        os.path.join(root_path, f'pooler_output_{Names}.pt'), 
        map_location=torch.device('cpu')
    ).cpu().numpy()
    
    d = unlabeled_feat.shape[-1]
    print(f"✓ Successfully loaded features (Dim: {d})")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Please run first: python models/SynSpERT/generate_al_features.py")
    exit(1)

# ==================== Step 2: K-Means Clustering ====================
print("\n[Step 2/4] K-Means Clustering...")
weights = np.ones(n_sample,)

if weight:
    kmeans = MiniBatchKMeans(n_clusters=ncentroids, random_state=0, batch_size=256, n_init=3, max_iter=100)
    kmeans.fit(unlabeled_feat, sample_weight=copy.deepcopy(entropy))
    index = faiss.IndexFlatL2(d)
    index.add(kmeans.cluster_centers_.astype('float32'))
    D, I = index.search(unlabeled_feat.astype('float32'), 1)
else:
    kmeans = faiss.Clustering(int(d), ncentroids)
    index = faiss.IndexFlatL2(d)
    kmeans.train(unlabeled_feat.astype('float32'), index)
    centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
    index.add(centroid.astype('float32'))
    D, I = index.search(unlabeled_feat.astype('float32'), 1)

I = I.flatten()
print(f"✓ Clustering complete ({ncentroids} clusters)")

# ==================== Step 3: Calculate Cluster Scores ====================
print("\n[Step 3/4] Calculating cluster scores and sorting...")
scores = []
indexes = []
class_number = np.ones(n_sample,)

for i in range(ncentroids):
    idx = (I == i)
    class_number[idx] = i
    weights[idx] = entropy[idx] / (np.sum(entropy[idx]) + 1e-12)
    mean_entropy = np.mean(entropy[idx]) if np.any(idx) else 0
    
    class_sum = torch.sum(unlabeled_pred[idx], dim=0, keepdim=False).numpy()
    if np.sum(class_sum) == 0:
        class_frequency = np.zeros(class_sum.shape[0],)
    else:
        class_frequency = class_sum / (np.sum(class_sum) + 1e-12)
    
    class_entropy = np.sum(np.abs(class_frequency * np.log(class_frequency + 1e-12)))
    value = mean_entropy + beta * class_entropy
    scores.append(value)
    
    sorted_idx = np.argsort(entropy[idx])
    idxs = np.arange(len(I))[idx][sorted_idx]
    indexes.append(list(idxs))

print(f"✓ Score calculation complete")

# ==================== Step 4: Sample Selection ====================
print("\n[Step 4/4] Selecting high-information samples...")
sample_idx = []
weighted_embedding = np.ones((n_sample, d))

for i in range(n_sample):
    weighted_embedding[i, :] = weights[i] * unlabeled_feat[i, :]

torch.save(torch.from_numpy(weighted_embedding), 
           os.path.join(root_path, f'weighted_embedding_{SaveNames}.pt'))
torch.save(torch.from_numpy(class_number), 
           os.path.join(root_path, f'class_number_{SaveNames}.pt'))

for i in np.argsort(scores)[::-1][:topK]:
    samples_to_take = min(topK * sample_per_group - len(sample_idx), sample_per_group, len(indexes[i]))
    selected_samples = indexes[i][-samples_to_take:]
    sample_idx.extend(selected_samples)
    if len(sample_idx) >= topK * sample_per_group:
        break

print(f"✓ Sample selection complete: {len(sample_idx)} samples")

# ==================== Step 5: Save Results ====================
print("\n[Step 5/5] Saving results...")
with open(all_file_json_path, 'r', encoding='utf-8') as fr:
    all_json_text = json.load(fr)

sampling_json_text = [all_json_text[idx] for idx in sample_idx]

output_path = os.path.join(output_dir, f'sampling_json_{SaveNames}.json')
with open(output_path, 'w', encoding='utf-8') as fw:
    json.dump(sampling_json_text, fw, ensure_ascii=False, indent=2)

# ==================== Summary Output ====================
print("\n" + "=" * 70)
print("Done! Active Learning Sample Selection")
print("=" * 70)
print(f"Total Samples: {n_sample}")
print(f"Selected Samples: {len(sample_idx)}")
print(f"Selection Ratio: {len(sample_idx) / n_sample * 100:.2f}%")
print(f"\n✓ Output File:")
print(f"  {output_path}")
print("=" * 70)
