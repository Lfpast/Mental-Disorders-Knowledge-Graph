import argparse
import json
import os
from typing import Set

from .grail_dataset import export_grail_dataset
from .kg_io import collect_entity_ids, collect_relation_types, collect_stats, validate_kg_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 0: data protocol and evaluation pipeline setup")
    parser.add_argument("--train", required=True, help="Path to training KG json")
    parser.add_argument("--test", required=True, help="Path to test KG json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--relation", default="treatment_for", help="Target relation type")
    parser.add_argument("--head-types", default="drug", help="Comma-separated head entity types")
    parser.add_argument("--tail-types", default="disease", help="Comma-separated tail entity types")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Edge removal ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    validate_kg_format(args.train, args.relation)
    validate_kg_format(args.test, args.relation)

    train_stats = collect_stats(args.train)
    test_stats = collect_stats(args.test)

    head_types: Set[str] = set(filter(None, args.head_types.split(",")))
    tail_types: Set[str] = set(filter(None, args.tail_types.split(",")))

    train_entities = collect_entity_ids(args.train, allowed_types=head_types | tail_types)
    test_entities = collect_entity_ids(args.test, allowed_types=head_types | tail_types)
    overlap = train_entities.intersection(test_entities)

    relation_types = {
        "train": sorted(list(collect_relation_types(args.train))),
        "test": sorted(list(collect_relation_types(args.test))),
    }

    grail_output = os.path.join(args.out, "grail")
    grail_stats = export_grail_dataset(
        train_path=args.train,
        test_path=args.test,
        output_dir=grail_output,
        relation_type=args.relation,
        head_types=head_types,
        tail_types=tail_types,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    summary = {
        "train_stats": train_stats,
        "test_stats": test_stats,
        "relation_types": relation_types,
        "entity_overlap_count": len(overlap),
        "grail": grail_stats,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
