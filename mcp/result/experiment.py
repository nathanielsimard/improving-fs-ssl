import os
import sys
from typing import Callable, List, Optional

import numpy as np

from mcp.config.evaluation import BestWeightsMetric
from mcp.config.parser import ExperimentConfig
from mcp.result.logger import ResultRecord, load_records_from_file
from mcp.utils.logging import create_logger

logger = create_logger(__name__)


class EpochResult(object):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def load(self) -> List[List[ResultRecord]]:
        return load_records_from_file(self.file_name)

    @staticmethod
    def losses(records: List[List[ResultRecord]]) -> List[List[float]]:
        return [[r.loss for r in rec] for rec in records]

    @staticmethod
    def times(records: List[List[ResultRecord]]) -> List[List[float]]:
        return [[r.time for r in rec] for rec in records]

    @staticmethod
    def metric(records: List[List[ResultRecord]]) -> List[List[float]]:
        return [[r.metric for r in rec] for rec in records]

    @staticmethod
    def task_name(records: List[List[ResultRecord]]) -> List[str]:
        return [r.name for r in records[0]]

    @staticmethod
    def metric_name(records: List[List[ResultRecord]]) -> List[str]:
        return [r.metric_name for r in records[0]]

    @staticmethod
    def reduce(
        values: List[List[float]],
        reduce_task: Optional[Callable] = np.mean,
        reduce_iter: Optional[Callable] = np.mean,
    ) -> np.ndarray:
        if reduce_task is None and reduce_iter is None:
            raise ValueError("Must reduce on something")

        if reduce_task is not None:
            values = [reduce_task(np.asarray(vv), axis=-1) for vv in values]

        if reduce_iter is not None:
            values = reduce_iter(np.asarray(values), axis=0)

        return values


class ExperimentResult(object):
    def __init__(self, config: ExperimentConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._records_dir_train = os.path.join(self.output_dir, "train")
        self._records_dir_eval = os.path.join(self.output_dir, "evaluation")

    def best_epoch(self) -> int:
        metric, best_idx, name = self._best_weights_metric()
        metrics = self.metric("eval", metric)

        indexes = np.argsort(metrics)
        index = indexes[best_idx]

        best_metric = metrics[index]
        epoch = index + 1

        logger.info(f"Found the best epoch to be {epoch} with {best_metric} {name}")
        return epoch

    def records(self, tag: str, train: bool = True) -> List[EpochResult]:
        records_dir = self._records_dir_train if train else self._records_dir_eval

        results: List[EpochResult] = []
        for epoch in range(1, sys.maxsize):
            file_name = os.path.join(records_dir, f"{tag}-{epoch}")
            if not os.path.exists(file_name):
                break

            results.append(EpochResult(file_name))

        return results

    def task_names(self, tag: str, train: bool = True) -> List[str]:
        e_records = self.records(tag, train=train)[0]
        return EpochResult.task_name(e_records.load())

    def metric_names(self, tag: str, train: bool = True) -> List[str]:
        e_records = self.records(tag, train)[0]
        return EpochResult.metric_name(e_records.load())

    def metric(
        self, tag: str, metric, reduce_task=np.mean, reduce_iter=np.mean, train=True
    ) -> np.ndarray:
        e_records = self.records(tag, train=train)
        return np.asarray(
            [
                EpochResult.reduce(
                    metric(records.load()),
                    reduce_task=reduce_task,
                    reduce_iter=reduce_iter,
                )
                for records in e_records
            ]
        )

    def _best_weights_metric(self):
        if self.config.evaluation.metric == BestWeightsMetric.LOSS:
            metric = EpochResult.losses
            best_idx = 0  # Best loss is the smallest one.
            name = "valid loss"
        elif self.config.evaluation.metric == BestWeightsMetric.TIME:
            metric = EpochResult.times
            best_idx = -1  # Best time is the bigger one.
            name = "time"
        elif self.config.evaluation.metric == BestWeightsMetric.METRIC:
            metric = EpochResult.metric
            best_idx = -1  # We assume the bigger the metric is, the better.
            name = str(self.metric_names("eval"))
        else:
            raise ValueError(
                f"Evaluation Metric not yet supported: {self.config.evaluation.metric}"
            )
        return metric, best_idx, name
