import os
import numpy as np
from typing import List

from mcp.result.logger import ResultRecord, load_records_from_file
from mcp.config.parser import ExperimentConfig


class ExperimentResult(object):
    def __init__(self, config: ExperimentConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._records_dir = os.path.join(self.output_dir, "train")

    def best_epoch(self) -> int:
        losses = []
        for epoch in range(1, self.config.trainer.epochs + 1):
            file_name = os.path.join(self._records_dir, f"valid-{epoch}")
            records_valid = load_records_from_file(file_name)
            loss = np.asarray([self._records_loss(rs) for rs in records_valid]).mean()
            losses.append(loss)

        return np.argmin(np.asarray(losses)) + 1

    def _records_loss(self, records: List[ResultRecord]) -> float:
        return np.asarray([r.loss for r in records]).mean()
