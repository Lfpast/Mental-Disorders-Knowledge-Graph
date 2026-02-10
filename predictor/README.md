# MDKG GraIL Pipeline (Task 0/1)

This folder provides a thin pipeline around the GraIL reference implementation.
The model and subgraph extraction are **not** re-implemented here; they are run
via the official GraIL repo:

- Paper: https://arxiv.org/abs/1911.06962
- Repo: https://github.com/kkteru/grail/

## Task 0: data protocol + dataset export

This pipeline reads the MDKG JSON files (with `entities` / `relations` per record),
validates the presence of the `treatment_for` relation, checks entity overlap,
and exports a GraIL-style dataset (`train.txt`, `valid.txt`, `test.txt`).
The triple format follows GraIL data files: `head<TAB>relation<TAB>tail`.

Example:

```
python -m predictor.pipeline_task0 \
  --train models/InputsAndOutputs/input/md_train_KG_0217_agu.json \
  --test models/InputsAndOutputs/input/md_test_KG_0217_agu.json \
  --out models/InputsAndOutputs/output/predictor_output \
  --relation treatment_for \
  --head-types drug \
  --tail-types disease \
  --val-ratio 0.2
```

Outputs:
- `models/InputsAndOutputs/output/predictor_output/summary.json`
- `models/InputsAndOutputs/output/predictor_output/grail/train.txt`
- `models/InputsAndOutputs/output/predictor_output/grail/valid.txt`
- `models/InputsAndOutputs/output/predictor_output/grail/test.txt`
- `models/InputsAndOutputs/output/predictor_output/grail/train_subgraph.json` (train graph with held-out edges removed)

## Task 1: run GraIL

Clone the GraIL repo and place the exported dataset under `grail/data/<dataset_name>`.
For example:

```
# inside grail repo
mkdir -p data/MDKG_v1
cp /path/to/models/InputsAndOutputs/output/predictor_output/grail/*.txt data/MDKG_v1/
```

Then run:

```
python -m predictor.pipeline_task1 \
  --grail-repo /path/to/grail \
  --dataset-name MDKG_v1 \
  --experiment-name grail_mdkg_v1 \
  --run-tests
```

## Notes on weighted BCE

GraIL's training loop is defined in the official repo. If you need weighted BCE,
you should apply it inside GraIL's loss function (see the repo's training script).
This keeps the core model consistent with the reference implementation.
