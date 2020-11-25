import os
import sys
from typing import Callable, List, Optional

import numpy as np

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
            values = [reduce_task(np.asarray(vv), axis=1) for vv in values]

        if reduce_iter is not None:
            values = reduce_iter(np.asarray(values), axis=0)

        return values


class ExperimentResult(object):
    def __init__(self, config: ExperimentConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._records_dir = os.path.join(self.output_dir, "train")

    def best_epoch(self) -> int:
        losses = self.metric("train", EpochResult.losses)

        indexes = np.argsort(np.asarray(losses))
        index = indexes[0]
        epoch = index + 1
        valid_loss = losses[index]

        logger.info(f"Found the best epoch to be {epoch} with valid loss {valid_loss}")
        return epoch

    def records(self, tag: str) -> List[EpochResult]:
        results: List[EpochResult] = []
        for epoch in range(1, sys.maxsize):
            file_name = os.path.join(self._records_dir, f"{tag}-{epoch}")
            if not os.path.exists(file_name):
                break

            results.append(EpochResult(file_name))

        return results

    def task_names(self, tag: str) -> List[str]:
        e_records = self.records(tag)[0]
        return EpochResult.task_name(e_records.load())

    def metric_names(self, tag: str) -> List[str]:
        e_records = self.records(tag)[0]
        return EpochResult.metric_name(e_records.load())

    def metric(
        self, tag: str, metric, reduce_task=np.mean, reduce_iter=np.mean
    ) -> np.ndarray:
        e_records = self.records(tag)
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
