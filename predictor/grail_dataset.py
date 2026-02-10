import os
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .kg_io import EdgeKey, iter_edge_keys, iter_triples
from .split import EdgeRemovalValidator, remove_edges_from_records

Triple = Tuple[str, str, str]
EdgeKey = Tuple[str, str, str, str, str]


def write_triples(path: str, triples: Iterable[Triple]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for head, rel, tail in triples:
            handle.write(f"{head}\t{rel}\t{tail}\n")


def edge_key_to_triple(edge_key: EdgeKey) -> Triple:
    head_type, head_span, rel_type, tail_type, tail_span = edge_key
    return (f"{head_type}::{head_span}", rel_type, f"{tail_type}::{tail_span}")


def export_grail_dataset(
    train_path: str,
    test_path: str,
    output_dir: str,
    relation_type: str,
    head_types: Set[str],
    tail_types: Set[str],
    val_ratio: float,
    seed: int,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    relation_filter = {relation_type}
    train_edges = list(
        iter_edge_keys(
            train_path,
            relation_filter=relation_filter,
            head_types=head_types,
            tail_types=tail_types,
        )
    )

    splitter = EdgeRemovalValidator(removal_ratio=val_ratio, seed=seed)
    train_edge_keys, val_edge_keys = splitter.split_edges(train_edges)

    train_subgraph_path = os.path.join(output_dir, "train_subgraph.json")
    remove_edges_from_records(
        train_path,
        train_subgraph_path,
        removed_edges=set(val_edge_keys),
        relation_filter=relation_filter,
    )

    train_triples = iter_triples(train_subgraph_path)
    valid_triples = (edge_key_to_triple(edge) for edge in val_edge_keys)
    test_triples = iter_triples(
        test_path,
        relation_filter=relation_filter,
        head_types=head_types,
        tail_types=tail_types,
    )

    write_triples(os.path.join(output_dir, "train.txt"), train_triples)
    write_triples(os.path.join(output_dir, "valid.txt"), valid_triples)
    write_triples(os.path.join(output_dir, "test.txt"), test_triples)

    return {
        "train_edges": len(train_edges),
        "val_edges": len(val_edge_keys),
        "train_subgraph": train_subgraph_path,
        "grail_dir": output_dir,
    }
