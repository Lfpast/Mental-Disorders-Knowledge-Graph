import json
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

try:
    import ijson  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ijson = None

from .normalize import normalize_span

Entity = Dict[str, object]
Relation = Dict[str, object]
Triple = Tuple[str, str, str]
EdgeKey = Tuple[str, str, str, str, str]


def iter_records(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        if ijson is not None:
            for record in ijson.items(handle, "item"):
                yield record
            return
        data = json.load(handle)
        for record in data:
            yield record


def build_entity_map(record: dict) -> Dict[str, Entity]:
    return {entity["entity_id"]: entity for entity in record.get("entities", [])}


def make_node_id(entity: Entity) -> str:
    return f"{entity['type']}::{normalize_span(str(entity['span']))}"


def make_edge_key(head: Entity, rel_type: str, tail: Entity) -> EdgeKey:
    return (
        str(head["type"]),
        normalize_span(str(head["span"])),
        rel_type,
        str(tail["type"]),
        normalize_span(str(tail["span"])),
    )


def iter_triples(
    path: str,
    relation_filter: Optional[Set[str]] = None,
    head_types: Optional[Set[str]] = None,
    tail_types: Optional[Set[str]] = None,
) -> Iterator[Triple]:
    for record in iter_records(path):
        entity_map = build_entity_map(record)
        for relation in record.get("relations", []):
            rel_type = relation.get("type")
            if relation_filter and rel_type not in relation_filter:
                continue
            head = entity_map.get(relation.get("head_id"))
            tail = entity_map.get(relation.get("tail_id"))
            if head is None or tail is None:
                continue
            if head_types and head["type"] not in head_types:
                continue
            if tail_types and tail["type"] not in tail_types:
                continue
            yield (make_node_id(head), str(rel_type), make_node_id(tail))


def iter_edge_keys(
    path: str,
    relation_filter: Optional[Set[str]] = None,
    head_types: Optional[Set[str]] = None,
    tail_types: Optional[Set[str]] = None,
) -> Iterator[EdgeKey]:
    for record in iter_records(path):
        entity_map = build_entity_map(record)
        for relation in record.get("relations", []):
            rel_type = relation.get("type")
            if relation_filter and rel_type not in relation_filter:
                continue
            head = entity_map.get(relation.get("head_id"))
            tail = entity_map.get(relation.get("tail_id"))
            if head is None or tail is None:
                continue
            if head_types and head["type"] not in head_types:
                continue
            if tail_types and tail["type"] not in tail_types:
                continue
            yield make_edge_key(head, str(rel_type), tail)


def collect_entity_ids(
    path: str, allowed_types: Optional[Set[str]] = None
) -> Set[str]:
    result: Set[str] = set()
    for record in iter_records(path):
        for entity in record.get("entities", []):
            if allowed_types and entity["type"] not in allowed_types:
                continue
            result.add(make_node_id(entity))
    return result


def collect_relation_types(path: str) -> Set[str]:
    rels: Set[str] = set()
    for record in iter_records(path):
        for relation in record.get("relations", []):
            rels.add(str(relation.get("type")))
    return rels


def collect_stats(path: str) -> dict:
    entity_counts: Dict[str, int] = {}
    relation_counts: Dict[str, int] = {}
    record_count = 0
    for record in iter_records(path):
        record_count += 1
        for entity in record.get("entities", []):
            entity_counts[entity["type"]] = entity_counts.get(entity["type"], 0) + 1
        for relation in record.get("relations", []):
            relation_counts[relation["type"]] = relation_counts.get(relation["type"], 0) + 1
    return {
        "records": record_count,
        "entity_counts": entity_counts,
        "relation_counts": relation_counts,
    }


def validate_kg_format(path: str, required_relation: str) -> None:
    stats = collect_stats(path)
    relation_counts = stats["relation_counts"]
    if required_relation not in relation_counts:
        raise ValueError(
            f"Missing required relation '{required_relation}' in {path}."
        )
