from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mcp.data.dataloader.dataloader import DataLoader
from mcp.model.base import Model
from mcp.task.base import Task, TaskOutput
from mcp.utils.logging import create_logger
from copy import deepcopy

logger = create_logger(__name__)


class TrainingLogger(object):
    def __init__(self, str_format: str, output: str):
        self.str_format = str_format
        self.output = output

    def log(self, outputs: List[TaskOutput], names: List[str], extra=""):
        info = [
            f"{n}: loss={o.loss:.3f} {o.metric_name}={o.metric:.3f}"
            for n, o in zip(names, outputs)
        ]

        logger.info(f"{self.str_format} - {extra} {' | '.join(info)}")

    def with_tag(self, tag: str):
        self_copy = deepcopy(self)
        self_copy.str_format = self_copy.str_format + tag
        return self_copy


class TrainingLoop(object):
    def __init__(
        self, device: torch.device, support_min_loss: float, support_max_epochs: int,
    ):
        self.device = device
        self.support_min_loss = support_min_loss
        self.support_max_epochs = support_max_epochs

    def fit_one(
        self,
        model,
        tasks: List[Task],
        dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        training_logger: TrainingLogger,
        train_model=True,
    ) -> float:
        model.train(train_model)

        for task in tasks:
            task.train()

        task_names = [t.name for t in tasks]

        losses = 0.0
        total = 0.0
        for i, (x, y) in enumerate(dataloader):
            outputs = self._step(model, tasks, x, y, optimizer)

            training_logger.log(
                outputs, task_names, extra=f"Batch {i+1}/{len(dataloader)}"
            )

            for o in outputs:
                losses += o.loss.item()
                total += 1.0
        scheduler.step()
        return losses / total

    def fit_support(
        self,
        model,
        tasks: List[Task],
        dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        training_logger: TrainingLogger,
    ):
        support_loss = 1.0
        support_epoch = 0

        while (
            support_loss > self.support_min_loss
            and support_epoch < self.support_max_epochs
        ):
            support_epoch += 1
            support_loss = self.fit_one(
                model,
                tasks,
                dataloader,
                optimizer,
                scheduler,
                training_logger.with_tag(f" - Support Epoch {support_epoch}"),
                train_model=False,
            )

    def evaluate(
        self,
        model: Model,
        tasks: List[Task],
        dataloader: DataLoader,
        training_logger: TrainingLogger,
    ):
        task_names = [t.name for t in tasks]

        model.eval()

        for task in tasks:
            task.eval()

        for i, (x, y) in enumerate(dataloader):
            outputs = self._compute(model, tasks, x, y)
            training_logger.log(
                outputs, task_names, extra=f"Batch {i+1}/{len(dataloader)}"
            )

    def _step(
        self,
        model: Model,
        tasks: List[Task],
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: Optimizer,
    ) -> List[TaskOutput]:
        optimizer.zero_grad()
        outputs = self._compute(model, tasks, x, y)
        loss: torch.Tensor = sum([o.loss for o in outputs])  # type: ignore
        loss.backward()
        optimizer.step()

        return outputs

    def _compute(
        self, model: Model, tasks: List[Task], x: torch.Tensor, y: torch.Tensor,
    ) -> List[TaskOutput]:
        x = x.to(self.device)
        y = y.to(self.device)

        return [task.run(model, x, y) for task in tasks]
