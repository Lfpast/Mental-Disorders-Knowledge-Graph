from typing import Dict, Iterable, List, Optional, Sequence

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("scikit-learn is required for metrics.") from exc


def _group_indices(group_ids: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(idx)
    return groups


def mrr(y_true: Sequence[int], y_score: Sequence[float], group_ids: Optional[Sequence[str]] = None) -> float:
    if group_ids is None:
        group_ids = ["all"] * len(y_true)
    groups = _group_indices(group_ids)
    total = 0.0
    for indices in groups.values():
        ranked = sorted(indices, key=lambda i: y_score[i], reverse=True)
        rank = 0
        for pos, idx in enumerate(ranked, start=1):
            if y_true[idx] == 1:
                rank = pos
                break
        if rank > 0:
            total += 1.0 / rank
    return total / max(len(groups), 1)


def hits_at_k(
    y_true: Sequence[int],
    y_score: Sequence[float],
    k: int,
    group_ids: Optional[Sequence[str]] = None,
) -> float:
    if group_ids is None:
        group_ids = ["all"] * len(y_true)
    groups = _group_indices(group_ids)
    hit_total = 0
    for indices in groups.values():
        ranked = sorted(indices, key=lambda i: y_score[i], reverse=True)
        topk = ranked[:k]
        if any(y_true[idx] == 1 for idx in topk):
            hit_total += 1
    return hit_total / max(len(groups), 1)


def compute_all_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    group_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mrr": mrr(y_true, y_score, group_ids=group_ids),
    }
    for k in [1, 5, 10, 20]:
        metrics[f"hits@{k}"] = hits_at_k(y_true, y_score, k, group_ids=group_ids)
    return metrics
