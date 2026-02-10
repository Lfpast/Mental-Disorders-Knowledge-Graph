import json
import random
from typing import Iterable, Iterator, List, Optional, Set

from .kg_io import EdgeKey, build_entity_map, iter_records, make_edge_key


def write_json_array(path: str, records: Iterator[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("[")
        first = True
        for record in records:
            if not first:
                handle.write(",\n")
            json.dump(record, handle, ensure_ascii=True)
            first = False
        handle.write("]")


class EdgeRemovalValidator:
    def __init__(self, removal_ratio: float = 0.2, seed: int = 42) -> None:
        self.removal_ratio = removal_ratio
        self.seed = seed

    def split_edges(self, edges: List[EdgeKey]) -> tuple[List[EdgeKey], List[EdgeKey]]:
        rng = random.Random(self.seed)
        edges_copy = list(edges)
        rng.shuffle(edges_copy)
        n_val = int(len(edges_copy) * self.removal_ratio)
        val_edges = edges_copy[:n_val]
        train_edges = edges_copy[n_val:]
        return train_edges, val_edges


def remove_edges_from_records(
    input_path: str,
    output_path: str,
    removed_edges: Set[EdgeKey],
    relation_filter: Set[str],
) -> None:
    def iter_filtered() -> Iterator[dict]:
        for record in iter_records(input_path):
            entity_map = build_entity_map(record)
            relations = []
            for relation in record.get("relations", []):
                rel_type = relation.get("type")
                if rel_type not in relation_filter:
                    relations.append(relation)
                    continue
                head = entity_map.get(relation.get("head_id"))
                tail = entity_map.get(relation.get("tail_id"))
                if head is None or tail is None:
                    relations.append(relation)
                    continue
                key = make_edge_key(head, str(rel_type), tail)
                if key in removed_edges:
                    continue
                relations.append(relation)
            record["relations"] = relations
            yield record

    write_json_array(output_path, iter_filtered())
