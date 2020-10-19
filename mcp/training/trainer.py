from typing import List

from torch.optim import Optimizer
from torch.utils.data import DataLoader

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
        dataloader_valid: DataLoader,
        tasks: List[Task],
        epochs: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.tasks = tasks
        self.epochs = epochs

    def fit(self):
        logger.info(
            f"Fitting the model | {self.model.num_trainable_parameters()} parameters | "
            + f"{len(self.dataloader_train)} train batches | "
            + f"{len(self.dataloader_valid)} valid batches"
        )

        for epoch in range(1, self.epochs + 1):
            for i, (x, y) in enumerate(self.dataloader_train):
                # Multi-task
                self.optimizer.zero_grad()
                losses = [task.run(self.model, x, y) for task in self.tasks]
                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                loss_info = [f"{t.name}: {l}" for t, l in zip(self.tasks, losses)]
                logger.info(
                    f"Epoch {epoch}/{self.epochs}, Batch {i+1}/{len(self.dataloader_train)}: {' | '.join(loss_info)}"
                )
