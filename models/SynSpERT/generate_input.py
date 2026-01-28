from ann2json import Annotation
from sklearn.model_selection import train_test_split
from glob import glob
import json
import collections
import os
import shutil

date = '0217'  # Date or version identifier for the dataset

# Paths to data files
annotated_file_path = os.path.join('..', 'InputsAndOutputs', 'input', 'dataset')  # Set your annotated data path here
train_file_path = os.path.join('..', 'InputsAndOutputs', 'input', 'md_train_KG_' + date + '.json')  # Set your train file path here
test_file_path = os.path.join('..', 'InputsAndOutputs', 'input', 'md_test_KG_' + date + '.json')  # Set your test file path here
all_file_path = os.path.join('..', 'InputsAndOutputs', 'input', 'md_KG_all_' + date + '.json')  # Set your all data file here


def replace_quotes_in_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r', encoding='utf8') as file:
                lines = file.readlines()


            lines = [line.replace('"', "'") for line in lines]

            with open(filepath, 'w', encoding='utf8') as file:
                file.writelines(lines)

replace_quotes_in_files(annotated_file_path)




print("="*50)
print(f"[INFO] Starting Preprocessing | Date: {date}")
print("="*50)

# Collect .ann files (one per annotated sample) and extract PMIDs robustly
files = glob(os.path.join(annotated_file_path, '*.ann'))
ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

print(f"[INFO] Found {len(ids)} annotation files. Processing...")

for pmid in ids:
    ann = Annotation(annotated_file_path, pmid)
    if ann.check_file_exists():
        try:
            ann.to_json()
        except Exception as e:
            print(f"[ERROR] Failed to process {pmid}: {e}")
    else:
        print(f"[WARN] Skipping {pmid}: missing .txt or .ann file")
files = sorted(glob(os.path.join(annotated_file_path, '*.json')))

print(f"[INFO] JSON conversion complete. Loading data...")

data = list()
for f in files:
    data_ = json.load(open(f, 'r', encoding='utf8'))
    for d in data_:
        data.append(d)

ys = [d['relations'][0]['type'] if len(d['relations'])>0 else 'no_relation' for d in data]
counter = collections.Counter(ys)

print("-" * 40)
print(f"{'Relation Type':<25} | {'Count':<10}")
print("-" * 40)
for rel_type, count in counter.most_common():
    print(f"{rel_type:<25} | {count:<10}")
print("-" * 40)

# Aggregate tokens across the dataset
tokens = []
for d in data:
    for t in d.get('tokens', []):
        tokens.append(t)

print(f"[INFO] Total tokens in dataset: {len(tokens)}")

x_train, x_test, y_train, y_test = train_test_split(data, ys,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=ys)
print(f"[INFO] Data splitting complete:")
print(f"       - Training set samples: {len(x_train)}")
print(f"       - Test set samples:     {len(x_test)}")

entities = []
for d in data:
    if len(d)!=0:
        entities = entities+d['entities']

with open(train_file_path, 'w') as f:
    json.dump(x_train, f)


with open(test_file_path, 'w') as f:
    json.dump(x_test, f)

with open(all_file_path, 'w') as f:
    json.dump(data, f)

# Delete the dataset directory after processing
print(f"Cleaning up: deleting {annotated_file_path}")
if os.path.exists(annotated_file_path):
    shutil.rmtree(annotated_file_path)



