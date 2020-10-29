import os
import numpy as np
from typing import List

from mcp.result.logger import ResultRecord, load_records_from_file
from mcp.config.parser import ExperimentConfig
from mcp.utils.logging import create_logger

logger = create_logger(__name__)


class ExperimentResult(object):
    def __init__(self, config: ExperimentConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._records_dir = os.path.join(self.output_dir, "train")

    def best_epoch(self) -> int:
        losses = []
        for epoch in range(1, self.config.trainer.epochs + 1):
            try:
                file_name = os.path.join(self._records_dir, f"eval-{epoch}")
                records_valid = load_records_from_file(file_name)
                loss = np.asarray(
                    [self._records_loss(rs) for rs in records_valid]
                ).mean()
                losses.append(loss)
            except FileNotFoundError:
                logger.warning(
                    f"Training did not complete {epoch-1}/{self.config.trainer.epochs}"
                )
                break

        indexes = np.argsort(np.asarray(losses))
        index = indexes[0]
        epoch = index + 1
        valid_loss = losses[index]

        logger.info(f"Found the best epoch to be {epoch} with valid loss {valid_loss}")
        return epoch

    def _records_loss(self, records: List[ResultRecord]) -> float:
        return np.asarray([r.loss for r in records]).mean()
