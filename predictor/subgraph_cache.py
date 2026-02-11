"""
Task 2.1: 智能子图缓存系统
Per pipeline.md Step 2 — 高效训练与推理优化

提供 LRU 缓存策略，避免重复计算子图。
GraIL 的 SubgraphDataset 已通过 GRAIL_CACHE_SUBGRAPHS 环境变量
原生集成了 in-memory 缓存。此模块为 predictor pipeline 提供
独立的缓存层，用于推理阶段的子图复用。
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SubgraphCache:
    """
    LRU 内存缓存，用于存储已构建的 DGL 子图。

    使用场景:
      - 训练阶段: 通过 GRAIL_CACHE_SUBGRAPHS=1 环境变量激活，
        在 SubgraphDataset 中自动缓存所有子图。
      - 推理阶段: 在 ChunkedInference 中缓存热点子图，
        同一节点参与多个候选对时避免重复抽取。

    参数:
        max_size: 最大缓存条目数。默认 50000 用于推理，
                  训练时建议设为数据集大小。
    """

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Optional[Any]:
        """查询缓存，命中时自动移至末尾 (LRU)。"""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: Any, value: Any) -> None:
        """插入或更新条目，满时淘汰最久未用条目。"""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """缓存命中率。"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息。"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.2%}",
        }

    def clear(self) -> None:
        """清空缓存并重置统计。"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("[SubgraphCache] Cache cleared")

    def log_stats(self) -> None:
        """打印缓存统计日志。"""
        s = self.stats()
        logger.info(
            f"[SubgraphCache] size={s['size']}/{s['max_size']}, "
            f"hits={s['hits']}, misses={s['misses']}, "
            f"hit_rate={s['hit_rate']}"
        )
