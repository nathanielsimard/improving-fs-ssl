from typing import List, NamedTuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mcp.data.dataloader.dataloader import DataLoader, FewShotDataLoader
from mcp.model.base import Model
from mcp.task.base import Task
from mcp.training.loop import TrainingLogger, TrainingLoop
from mcp.utils.logging import create_logger

logger = create_logger(__name__)


class TrainerLoggers(NamedTuple):
    train: TrainingLogger
    support: TrainingLogger
    evaluation: TrainingLogger


class Trainer(object):
    def __init__(
        self,
        model: Model,
        optimizer_train: Optimizer,
        optimizer_support: Optimizer,
        scheduler_train: _LRScheduler,
        scheduler_support: _LRScheduler,
        dataloader_train: DataLoader,
        dataloader_valid: FewShotDataLoader,
        tasks_train: List[Task],
        tasks_valid: List[Task],
        epochs: int,
        training_loop: TrainingLoop,
        trainer_loggers: TrainerLoggers,
        device: torch.device,
    ):
        self.model = model
        self.optimizer_train = optimizer_train
        self.optimizer_support = optimizer_support
        self.scheduler_train = scheduler_train
        self.scheduler_support = scheduler_support
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.tasks_train = tasks_train
        self.tasks_valid = tasks_valid
        self.epochs = epochs
        self.training_loop = training_loop
        self.logger = trainer_loggers
        self.device = device

    def fit(self):
        self.model.to(self.device)
        for task in self.tasks_train + self.tasks_valid:
            task.to(self.device)

        logger.info(
            f"Fitting the model | {self.model.num_trainable_parameters()} parameters | "
            + f"{len(self.dataloader_train)} train batches | "
            + f"{len(self.dataloader_valid)} valid batches"
        )

        for epoch in range(1, self.epochs + 1):
            self._training_phase(epoch)
            self._training_support_phase(epoch)
            self._evaluation_phase(epoch)

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

    def _training_support_phase(self, epoch):
        self.training_loop.fit_support(
            self.model,
            self.tasks_valid,
            self.dataloader_valid.support,
            self.optimizer_support,
            self.scheduler_support,
            self.logger.support.epoch(epoch, self.epochs),
        )

    def _evaluation_phase(self, epoch):
        self.training_loop.evaluate(
            self.model,
            self.tasks_valid,
            self.dataloader_valid.query,
            self.logger.evaluation.epoch(epoch, self.epochs),
        )
