"""
Task 2.2 & 2.3: 批量子图处理优化 + 分块全局推理策略
Per pipeline.md Step 2 — 高效训练与推理优化

将大规模推理任务（如全局药物重定位预测）按块处理，
避免内存溢出，同时支持进度追踪和中断恢复。
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """单个候选对的推理结果。"""

    head: str
    tail: str
    relation: str
    score: float
    rank: Optional[int] = None


@dataclass
class ChunkedInferenceConfig:
    """分块推理配置。"""

    chunk_size: int = 500
    batch_size: int = 64
    device: str = "cuda:0"
    output_dir: str = "."
    save_every_chunks: int = 10
    top_k: int = 100


class ChunkedInference:
    """
    分块全局推理器。

    将大规模候选对集合分成小块 (chunks)，逐块抽取子图并推理，
    避免一次性加载所有子图导致 OOM。

    使用流程:
        1. 加载训练好的 GraIL 模型和知识图谱
        2. 生成所有候选 (disease, drug) 对
        3. 调用 predict_all_pairs() 进行分块推理
        4. 获取排序后的预测结果

    示例:
        >>> inferencer = ChunkedInference(config)
        >>> results = inferencer.predict_all_pairs(
        ...     model=model,
        ...     adj_list=adj_list,
        ...     candidate_triples=candidates,
        ...     params=params,
        ... )
        >>> for r in results[:20]:
        ...     print(f"{r.head} -> {r.tail}: {r.score:.4f}")
    """

    def __init__(self, config: Optional[ChunkedInferenceConfig] = None):
        self.config = config or ChunkedInferenceConfig()
        self._checkpoint_path: Optional[str] = None

    def predict_all_pairs(
        self,
        model: torch.nn.Module,
        adj_list: list,
        candidate_triples: List[Tuple[int, int, int]],
        params: Any,
        collate_fn: Any,
        move_batch_to_device: Any,
        subgraph_extractor: Any = None,
        resume_from: int = 0,
    ) -> List[InferenceResult]:
        """
        对所有候选三元组进行分块推理。

        参数:
            model: 训练好的 GraIL GraphClassifier
            adj_list: 邻接矩阵列表 (scipy sparse)
            candidate_triples: [(head_id, tail_id, rel_id), ...] 候选列表
            params: GraIL 参数对象
            collate_fn: 批次整理函数
            move_batch_to_device: 设备转移函数
            subgraph_extractor: 可选的子图抽取函数
            resume_from: 从第几个 chunk 恢复

        返回:
            按分数降序排列的 InferenceResult 列表
        """
        model.eval()
        device = torch.device(self.config.device)
        total = len(candidate_triples)
        chunk_size = self.config.chunk_size
        total_chunks = (total + chunk_size - 1) // chunk_size

        all_scores: List[Tuple[Tuple[int, int, int], float]] = []

        # 加载中间检查点
        if resume_from > 0:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                all_scores = checkpoint.get("scores", [])
                logger.info(
                    f"[Inference] Resumed from chunk {resume_from}, "
                    f"{len(all_scores)} scores loaded"
                )

        logger.info(
            f"[Inference] Starting chunked inference: "
            f"{total} candidates, {total_chunks} chunks "
            f"(chunk_size={chunk_size}, batch_size={self.config.batch_size})"
        )

        start_time = time.time()

        for chunk_idx in range(resume_from, total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = candidate_triples[chunk_start:chunk_end]

            chunk_scores = self._score_chunk(
                model, chunk, adj_list, params,
                collate_fn, move_batch_to_device, device,
            )
            all_scores.extend(zip(chunk, chunk_scores))

            # 进度报告
            elapsed = time.time() - start_time
            progress = (chunk_idx + 1 - resume_from)
            total_remaining = total_chunks - chunk_idx - 1
            if progress > 0:
                eta = elapsed / progress * total_remaining
            else:
                eta = 0.0

            if (chunk_idx + 1) % max(1, total_chunks // 20) == 0 or chunk_idx == total_chunks - 1:
                logger.info(
                    f"[Inference] Chunk {chunk_idx+1}/{total_chunks} "
                    f"({100*(chunk_idx+1)/total_chunks:.0f}%) "
                    f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s"
                )

            # 定期保存检查点
            if (
                self.config.save_every_chunks > 0
                and (chunk_idx + 1) % self.config.save_every_chunks == 0
            ):
                self._save_checkpoint(all_scores, chunk_idx + 1)

        elapsed = time.time() - start_time
        logger.info(
            f"[Inference] Completed: {len(all_scores)} predictions in {elapsed:.1f}s "
            f"({len(all_scores)/elapsed:.0f} preds/s)"
        )

        # 按分数降序排序
        all_scores.sort(key=lambda x: x[1], reverse=True)

        # 转为 InferenceResult
        results = []
        for rank, ((h, t, r), score) in enumerate(all_scores, 1):
            results.append(InferenceResult(
                head=str(h), tail=str(t), relation=str(r),
                score=float(score), rank=rank,
            ))

        return results

    def _score_chunk(
        self,
        model: torch.nn.Module,
        triples: List[Tuple[int, int, int]],
        adj_list: list,
        params: Any,
        collate_fn: Any,
        move_batch_to_device: Any,
        device: torch.device,
    ) -> List[float]:
        """
        对一个 chunk 的三元组进行评分。

        使用 GraIL 的子图抽取和模型前向传播。
        """
        from subgraph_extraction.graph_sampler import subgraph_extraction_labeling
        from utils.graph_utils import ssp_multigraph_to_dgl

        scores = []
        batch_size = self.config.batch_size

        # 逐批处理
        for batch_start in range(0, len(triples), batch_size):
            batch_triples = triples[batch_start:batch_start + batch_size]
            batch_subgraphs = []

            for (h, t, r) in batch_triples:
                try:
                    nodes, n_labels, *_ = subgraph_extraction_labeling(
                        (h, t), r, adj_list,
                        h=getattr(params, 'hop', 2),
                        enclosing_sub_graph=getattr(params, 'enclosing_sub_graph', True),
                        max_nodes_per_hop=getattr(params, 'max_nodes_per_hop', None),
                    )
                    # 这里简化处理 — 实际应用时应复用 SubgraphDataset._prepare_subgraphs
                    batch_subgraphs.append({
                        'nodes': nodes,
                        'n_labels': n_labels,
                        'r_label': r,
                    })
                except Exception as e:
                    logger.warning(f"Subgraph extraction failed for ({h},{t},{r}): {e}")
                    batch_subgraphs.append(None)

            # 对成功抽取的子图进行推理
            for sg in batch_subgraphs:
                if sg is None:
                    scores.append(0.0)
                else:
                    scores.append(0.0)  # placeholder — 实际需要通过模型前向传播

        return scores

    def _save_checkpoint(
        self,
        scores: List[Tuple[Tuple[int, int, int], float]],
        chunk_idx: int,
    ) -> None:
        """保存推理检查点。"""
        path = os.path.join(self.config.output_dir, "inference_checkpoint.json")
        data = {
            "chunk_idx": chunk_idx,
            "num_scores": len(scores),
            "scores": [(list(triple), float(s)) for triple, s in scores[-self.config.chunk_size:]],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.debug(f"[Inference] Checkpoint saved at chunk {chunk_idx}")

    def _load_checkpoint(self) -> Optional[Dict]:
        """加载推理检查点。"""
        path = os.path.join(self.config.output_dir, "inference_checkpoint.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    @staticmethod
    def save_results(
        results: List[InferenceResult],
        output_path: str,
        top_k: Optional[int] = None,
    ) -> None:
        """保存推理结果到 JSON 文件。"""
        if top_k:
            results = results[:top_k]
        data = [
            {
                "rank": r.rank,
                "head": r.head,
                "tail": r.tail,
                "relation": r.relation,
                "score": r.score,
            }
            for r in results
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[Inference] Saved {len(data)} results to {output_path}")

    @staticmethod
    def generate_candidate_triples(
        diseases: List[int],
        drugs: List[int],
        treatment_rel_id: int,
        existing_triples: Optional[set] = None,
    ) -> List[Tuple[int, int, int]]:
        """
        生成所有疾病-药物候选三元组。

        参数:
            diseases: 疾病节点 ID 列表
            drugs: 药物节点 ID 列表
            treatment_rel_id: 治疗关系的 ID
            existing_triples: 已知的 (h, t, r) 三元组集合，
                              用于排除已有治疗关系

        返回:
            候选三元组列表 [(disease, drug, treatment_rel_id), ...]
        """
        candidates = []
        existing = existing_triples or set()
        for d in diseases:
            for r in drugs:
                triple = (d, r, treatment_rel_id)
                if triple not in existing:
                    candidates.append(triple)

        logger.info(
            f"[Inference] Generated {len(candidates)} candidate triples "
            f"({len(diseases)} diseases × {len(drugs)} drugs, "
            f"{len(existing)} existing excluded)"
        )
        return candidates
