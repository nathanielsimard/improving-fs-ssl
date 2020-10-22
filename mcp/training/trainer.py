from typing import List

import torch
from torch.optim import Optimizer

from mcp.data.dataset.dataset import DataLoader, FewShotDataLoader
from mcp.model.base import Model
from mcp.task.base import Task, TaskOutput
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
            self._train(self.tasks_train, epoch, self.dataloader_train, "Training")
            self._train(
                self.tasks_valid,
                epoch,
                self.dataloader_valid.support,
                "Training - Support",
            )
            self._evaluate(
                self.tasks_valid, epoch, self.dataloader_valid.query, "Evaluation"
            )

    def _train(self, tasks: List[Task], epoch: int, dataloader: DataLoader, tag: str):
        self.model.train()
        for task in tasks:
            task.train()

        for i, (x, y) in enumerate(dataloader):
            # Multi-task
            log_template = self._log_template(i + 1, epoch, dataloader, tag)
            self._step(tasks, x, y, log_template)

    def _evaluate(self, tasks: List[Task], epoch: int, dataloader: DataLoader, tag):
        self.model.eval()
        for task in tasks:
            task.eval()

        for i, (x, y) in enumerate(dataloader):
            log_template = self._log_template(i + 1, epoch, dataloader, tag)
            self._compute(tasks, x, y, log_template)

    def _step(
        self, tasks: List[Task], x: torch.Tensor, y: torch.Tensor, log_template: str,
    ):
        self.optimizer.zero_grad()
        outputs = self._compute(tasks, x, y, log_template)
        loss: torch.Tensor = sum([o.loss for o in outputs])  # type: ignore
        loss.backward()
        self.optimizer.step()

    def _compute(
        self, tasks: List[Task], x: torch.Tensor, y: torch.Tensor, log_template: str,
    ) -> List[TaskOutput]:
        outputs = [task.run(self.model, x, y) for task in tasks]
        info = [
            f"{t.name}: loss={o.loss:.3f} {o.metric_name}={o.metric:.3f}"
            for t, o in zip(tasks, outputs)
        ]

        logger.info(f"{log_template} - {' | '.join(info)}")

        return outputs

    def _log_template(
        self, batch_id: int, epoch: int, dataloader: DataLoader, tag: str
    ):
        return f"{tag} - Epoch {epoch}/{self.epochs} Batch {batch_id}/{len(dataloader)}"
