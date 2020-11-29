import os
from typing import List, NamedTuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mcp.config.evaluation import BestWeightsMetric
from mcp.data.dataloader.dataloader import DataLoader, FewShotDataLoaderFactory
from mcp.data.dataset.dataset import FewShotDataset
from mcp.model.base import Model
from mcp.result.logger import ResultLogger
from mcp.task.base import Task
from mcp.training.loop import TrainingLoop
from mcp.utils.logging import create_logger

logger = create_logger(__name__)


class TrainerLoggers(NamedTuple):
    train: ResultLogger
    support: ResultLogger
    evaluation: ResultLogger


class Trainer(object):
    def __init__(
        self,
        model: Model,
        optimizer_train: Optimizer,
        optimizer_support: Optimizer,
        scheduler_train: _LRScheduler,
        scheduler_support: _LRScheduler,
        dataloader_train: DataLoader,
        dataset_valid: FewShotDataset,
        dataloader_valid_factory: FewShotDataLoaderFactory,
        tasks_train: List[Task],
        tasks_valid: List[Task],
        epochs: int,
        training_loop: TrainingLoop,
        trainer_loggers: TrainerLoggers,
        device: torch.device,
        save_path: str,
        checkpoint_metric: BestWeightsMetric,
        num_valid_iterations: int,
        num_checkpoints: int,
    ):
        self.model = model
        self.optimizer_train = optimizer_train
        self.optimizer_support = optimizer_support
        self.scheduler_train = scheduler_train
        self.scheduler_support = scheduler_support
        self.dataloader_train = dataloader_train
        self.dataloader_valid_factory = dataloader_valid_factory
        self.dataset_valid = dataset_valid
        self.tasks_train = tasks_train
        self.tasks_valid = tasks_valid
        self.epochs = epochs
        self.training_loop = training_loop
        self.logger = trainer_loggers
        self.device = device
        self.save_path = save_path
        self.checkpoint_metric = checkpoint_metric
        self.num_valid_iterations = num_valid_iterations
        self.num_checkpoints = num_checkpoints
        self.valid_metrics: List[float] = []
        self.checkpoints: List[int] = []

    def fit(self, starting_epoch=0):
        self.model.to(self.device)
        for task in self.tasks_train + self.tasks_valid:
            task.to(self.device)

        logger.info(
            f"Fitting the model | {self.model.num_trainable_parameters()} parameters | "
            + f"{len(self.dataloader_train)} train batches "
        )

        for epoch in range(starting_epoch + 1, self.epochs + 1):
            self._training_phase(epoch)

            metric = 0.0
            for i in range(self.num_valid_iterations):
                dataloader_valid = self.dataloader_valid_factory.create(
                    self.dataset_valid
                )
                self._training_support_phase(epoch, dataloader_valid)
                metric += self._evaluation_phase(epoch, dataloader_valid)

            self._save_checkpoint(epoch, metric / self.num_valid_iterations)

    def _training_phase(self, epoch):
        self.training_loop.fit_one(
            self.model,
            self.tasks_train,
            self.dataloader_train,
            self.optimizer_train,
            self.scheduler_train,
            self.logger.train.epoch(epoch, self.epochs),
            train_model=True,
        )

    def _training_support_phase(self, epoch, dataloader_valid):
        self.training_loop.fit_support(
            self.model,
            self.tasks_valid,
            dataloader_valid.support,
            self.optimizer_support,
            self.scheduler_support,
            self.logger.support.epoch(epoch, self.epochs),
        )

    def _evaluation_phase(self, epoch, dataloader_valid) -> float:
        return self.training_loop.evaluate(
            self.model,
            self.tasks_valid,
            dataloader_valid.query,
            self.logger.evaluation.epoch(epoch, self.epochs),
        )

    def _save_checkpoint(self, epoch: int, metric: float):
        logger.info(f"Saving checkpoint | epoch {epoch} - metric {metric}")
        self.valid_metrics.append(metric)
        self.checkpoints.append(epoch)
        self.save(epoch)

        idxs = np.argsort(np.asarray(self.valid_metrics))
        if len(idxs) > self.num_checkpoints:
            # Worse metric ids
            if self.checkpoint_metric == BestWeightsMetric.LOSS:
                worse_metric_id = -1
            elif self.checkpoint_metric == BestWeightsMetric.METRIC:
                worse_metric_id = 0
            elif self.checkpoint_metric == BestWeightsMetric.TIME:
                worse_metric_id = 0
            else:
                raise ValueError(f"Unsupported metric {self.checkpoint_metric}")

            idx = idxs[worse_metric_id]
            epoch = self.checkpoints[idx]
            metric = self.valid_metrics[idx]

            logger.info(f"Remove checkpoint | epoch {epoch} - metric {metric}")

            os.remove(self._trainer_path(epoch))
            os.remove(self._model_path(epoch))

            for task in self.tasks_train:
                os.remove(self._task_path(task.name, epoch))

            del self.valid_metrics[idx]
            del self.checkpoints[idx]

    def save(self, epoch: int):
        self.model.save(self._model_path(epoch))
        for task in self.tasks_train:
            task.save(self._task_path(task.name, epoch))

        torch.save(
            {
                "optimizer_state_dict": self.optimizer_train.state_dict(),
                "scheduler_state_dict": self.scheduler_train.state_dict(),
                "checkpoints": self.checkpoints,
                "valid_metrics": self.valid_metrics,
            },
            self._trainer_path(epoch),
        )

    def load(self, epoch: int):
        self.model.load(self._model_path(epoch), self.device)
        for task in self.tasks_train:
            task.load(self._task_path(task.name, epoch), self.device)

        trainer_checkpoint = torch.load(
            self._trainer_path(epoch), map_location=self.device
        )
        self.optimizer_train.load_state_dict(trainer_checkpoint["optimizer_state_dict"])
        self.scheduler_train.load_state_dict(trainer_checkpoint["scheduler_state_dict"])

        self.checkpoints = trainer_checkpoint["checkpoints"]
        self.valid_metrics = trainer_checkpoint["valid_metrics"]

    def _model_path(self, epoch: int) -> str:
        return os.path.join(self.save_path, f"model-{epoch}.pth")

    def _task_path(self, name: str, epoch: int) -> str:
        return os.path.join(self.save_path, f"task-{name}-{epoch}.pth")

    def _trainer_path(self, epoch: int) -> str:
        return os.path.join(self.save_path, f"trainer-{epoch}.pth")
