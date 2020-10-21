from typing import List
import torch

from torch.optim import Optimizer
from mcp.data.dataset.dataset import DataLoader, FewShotDataLoader

from mcp.model.base import Model
from mcp.task.base import Task
from mcp.utils.logging import create_logger

logger = create_logger(__name__)


class Trainer(object):
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        dataloader_train: DataLoader,
        dataloader_valid: FewShotDataLoader,
        tasks_train: List[Task],
        tasks_valid: List[Task],
        epochs: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.tasks_train = tasks_train
        self.tasks_valid = tasks_valid
        self.epochs = epochs

    def fit(self):
        logger.info(
            f"Fitting the model | {self.model.num_trainable_parameters()} parameters | "
            + f"{len(self.dataloader_train)} train batches | "
            + f"{len(self.dataloader_valid)} valid batches"
        )

        for epoch in range(1, self.epochs + 1):
            self._train(self.model, self.tasks_train, epoch, self.dataloader_train)
            self._train(
                self.model, self.tasks_valid, epoch, self.dataloader_valid.support
            )

    def _train(
        self, model: Model, tasks: List[Task], epoch: int, dataloader: DataLoader
    ):
        for i, (x, y) in enumerate(dataloader):
            # Multi-task
            batch_id = i + 1
            self._step(model, tasks, batch_id, epoch, x, y)

    def _step(
        self,
        model: Model,
        tasks: List[Task],
        batch_id: int,
        epoch: int,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        self.optimizer.zero_grad()
        losses = [task.run(model, x, y) for task in tasks]
        loss: torch.Tensor = sum(losses)  # type: ignore
        loss.backward()
        self.optimizer.step()
        loss_info = [f"{t.name}: {l}" for t, l in zip(tasks, losses)]
        logger.info(
            f"Epoch {epoch}/{self.epochs}, Batch {batch_id}/{len(self.dataloader_train)}: {' | '.join(loss_info)}"
        )
