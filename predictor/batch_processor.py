"""
Task 2.2: 批量子图处理优化
Per pipeline.md Step 2 — 高效训练与推理优化

提供训练和推理时的性能监控、批量处理优化工具。
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class TrainingOptimizer:
    """
    GraIL 训练优化器 — 封装了第二步中的所有训练加速策略。

    支持的优化:
      1. 子图缓存 (GRAIL_CACHE_SUBGRAPHS=1): 在内存中预缓存所有
         LMDB 子图，消除重复 I/O 和 DGL 图构建开销。约 25% 加速。
      2. AMP 混合精度 (--use_amp): 可选的自动混合精度训练。
         对大模型有效，小模型 (<50K params) 可能引入额外开销。
      3. 直接图构建: ssp_multigraph_to_dgl 优化为跳过 NetworkX，
         直接从 scipy sparse 构建 DGL 图。

    使用方法:
        通过 GraIL train.py 的 CLI 参数启用:
            python train.py -d MDKG_v1 -e my_exp \\
                --cache_subgraphs \\
                --use_amp \\
                --hop 2 --batch_size 16 ...
    """

    # 推荐配置 (基于 MDKG_v1 实验结果)
    RECOMMENDED_CONFIG = {
        "hop": 2,
        "max_nodes_per_hop": 100,
        "batch_size": 16,
        "lr": 0.0005,
        "margin": 3,
        "loss_reduction": "mean",
        "dropout": 0.2,
        "edge_dropout": 0.2,
        "clip": 10,
        "num_workers": 0,
        "cache_subgraphs": True,
        "use_amp": False,  # 小模型不建议开启
    }

    @staticmethod
    def get_grail_train_cmd(
        dataset: str = "MDKG_v1",
        experiment: str = "exp_optimized",
        grail_dir: str = ".",
        num_epochs: int = 20,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成推荐的 GraIL 训练命令。

        参数:
            dataset: 数据集名称
            experiment: 实验名称
            grail_dir: GraIL 仓库路径
            num_epochs: 训练轮数
            extra_args: 额外参数覆盖

        返回:
            完整的训练命令字符串
        """
        config = dict(TrainingOptimizer.RECOMMENDED_CONFIG)
        if extra_args:
            config.update(extra_args)

        cmd_parts = [
            "GRAIL_LMDB_MAP_SIZE_MB=512",
            f"python train.py -d {dataset} -e {experiment}",
            f"--num_epochs {num_epochs}",
        ]

        for key, value in config.items():
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(flag)
            else:
                cmd_parts.append(f"{flag} {value}")

        return " ".join(cmd_parts)

    @staticmethod
    def estimate_training_time(
        num_epochs: int,
        num_train_triples: int,
        batch_size: int = 16,
        cache_enabled: bool = True,
    ) -> Dict[str, float]:
        """
        估算训练时间。

        基于 MDKG_v1 基准测试:
        - 无优化: ~52s/epoch (7398 triples, batch_size=16)
        - 缓存优化: ~39s/epoch
        - 缓存预热: ~12s (一次性)
        """
        # 基于基准测试的每三元组时间 (毫秒)
        MS_PER_TRIPLE_NOCACHE = 52000 / 7398  # ~7.0 ms/triple
        MS_PER_TRIPLE_CACHED = 39000 / 7398  # ~5.3 ms/triple
        CACHE_WARMUP_MS_PER_TRIPLE = 12000 / 7398  # ~1.6 ms/triple

        ms_per_triple = MS_PER_TRIPLE_CACHED if cache_enabled else MS_PER_TRIPLE_NOCACHE
        epoch_time = num_train_triples * ms_per_triple / 1000

        warmup = num_train_triples * CACHE_WARMUP_MS_PER_TRIPLE / 1000 if cache_enabled else 0

        total = warmup + num_epochs * epoch_time

        return {
            "epoch_time_sec": round(epoch_time, 1),
            "warmup_sec": round(warmup, 1),
            "total_sec": round(total, 1),
            "total_min": round(total / 60, 1),
        }


class EpochTimer:
    """训练 epoch 计时器。"""

    def __init__(self):
        self.epoch_times: list = []
        self._start: Optional[float] = None

    def start(self) -> None:
        self._start = time.time()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.time() - self._start
        self.epoch_times.append(elapsed)
        self._start = None
        return elapsed

    @property
    def avg_epoch_time(self) -> float:
        return sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0.0

    @property
    def total_time(self) -> float:
        return sum(self.epoch_times)

    def eta(self, remaining_epochs: int) -> float:
        """估算剩余训练时间 (秒)。"""
        return self.avg_epoch_time * remaining_epochs

    def summary(self) -> str:
        n = len(self.epoch_times)
        if n == 0:
            return "No epochs recorded"
        return (
            f"Epochs: {n}, "
            f"Avg: {self.avg_epoch_time:.1f}s, "
            f"Total: {self.total_time:.1f}s ({self.total_time/60:.1f}min)"
        )
