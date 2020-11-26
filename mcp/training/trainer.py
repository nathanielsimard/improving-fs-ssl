import os
from typing import List, NamedTuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mcp.data.dataloader.dataloader import DataLoader, FewShotDataLoader
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
        dataloader_valid: FewShotDataLoader,
        tasks_train: List[Task],
        tasks_valid: List[Task],
        epochs: int,
        training_loop: TrainingLoop,
        trainer_loggers: TrainerLoggers,
        device: torch.device,
        save_path: str,
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
        self.save_path = save_path
        self.num_checkpoints = 5
        self.valid_losses: List[float] = []
        self.checkpoints: List[int] = []

    def fit(self, starting_epoch=0):
        self.model.to(self.device)
        for task in self.tasks_train + self.tasks_valid:
            task.to(self.device)

        logger.info(
            f"Fitting the model | {self.model.num_trainable_parameters()} parameters | "
            + f"{len(self.dataloader_train)} train batches | "
            + f"{len(self.dataloader_valid)} valid batches"
        )

        for epoch in range(starting_epoch + 1, self.epochs + 1):
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
        loss = self.training_loop.evaluate(
            self.model,
            self.tasks_valid,
            self.dataloader_valid.query,
            self.logger.evaluation.epoch(epoch, self.epochs),
        )
        self._save_checkpoint(epoch, loss)

    def _save_checkpoint(self, epoch: int, loss: float):
        logger.info(f"Saving checkpoint | epoch {epoch} - loss {loss}")
        self.valid_losses.append(loss)
        self.checkpoints.append(epoch)
        self.save(epoch)

        idxs = np.argsort(np.asarray(self.valid_losses))
        if len(idxs) > self.num_checkpoints:
            idx = idxs[-1]
            epoch = self.checkpoints[idx]
            loss = self.valid_losses[idx]

            logger.info(f"Remove checkpoint | epoch {epoch} - loss {loss}")

            os.remove(self._trainer_path(epoch))
            os.remove(self._model_path(epoch))

            for task in self.tasks_train:
                os.remove(self._task_path(task.name, epoch))

            del self.valid_losses[idx]
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
                "valid_losses": self.valid_losses,
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
        self.valid_losses = trainer_checkpoint["valid_losses"]

    def _model_path(self, epoch: int) -> str:
        return os.path.join(self.save_path, f"model-{epoch}.pth")

    def _task_path(self, name: str, epoch: int) -> str:
        return os.path.join(self.save_path, f"task-{name}-{epoch}.pth")

    def _trainer_path(self, epoch: int) -> str:
        return os.path.join(self.save_path, f"trainer-{epoch}.pth")
