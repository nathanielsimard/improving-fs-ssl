from copy import deepcopy
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mcp.config.evaluation import BestWeightsMetric
from mcp.data.dataloader.dataloader import DataLoader
from mcp.model.base import Model
from mcp.result.logger import ResultLogger
from mcp.task.base import Task, TaskOutput
from mcp.task.compute import TaskCompute
from mcp.utils.logging import create_logger

logger = create_logger(__name__)


class TrainingLoop(object):
    def __init__(
        self,
        device: torch.device,
        support_min_loss: float,
        support_max_epochs: int,
        compute: TaskCompute,
        checkpoint_metric: BestWeightsMetric,
    ):
        self.device = device
        self.support_min_loss = support_min_loss
        self.support_max_epochs = support_max_epochs
        self.compute = compute
        self.checkpoint_metric = checkpoint_metric

    def fit_one(
        self,
        model,
        tasks: List[Task],
        dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        training_logger: ResultLogger,
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

            training_logger.log(outputs, task_names, i + 1, len(dataloader))

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
        training_logger: ResultLogger,
    ):
        support_loss = 1.0
        support_epoch = 0

        # Don't change default optimizer and scheduler states
        optimizer_state_dict = deepcopy(optimizer.state_dict())
        scheduler_state_dict = deepcopy(scheduler.state_dict())

        # Reset tasks states
        for task in tasks:
            task.reset()

        model.freeze_weights()

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
                training_logger.epoch(support_epoch, self.support_max_epochs),
                train_model=False,
            )

        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)
        model.defreeze_weights()

    def evaluate(
        self,
        model: Model,
        tasks: List[Task],
        dataloader: DataLoader,
        training_logger: ResultLogger,
    ) -> float:
        task_names = [t.name for t in tasks]

        model.eval()
        model.freeze_weights()

        for task in tasks:
            task.eval()

        running_metric = 0.0
        total = 0

        for i, (x, y) in enumerate(dataloader):
            outputs = self._compute(model, tasks, x, y)
            training_logger.log(outputs, task_names, i + 1, len(dataloader))

            if self.checkpoint_metric == BestWeightsMetric.LOSS:
                running_metric += sum(o.loss.item() for o in outputs)
            elif self.checkpoint_metric == BestWeightsMetric.METRIC:
                running_metric += sum(o.metric for o in outputs)
            elif self.checkpoint_metric == BestWeightsMetric.TIME:
                running_metric += sum(o.time for o in outputs)
            else:
                raise ValueError(f"Unsupported metric {self.checkpoint_metric}")

            total += len(tasks)

        model.defreeze_weights()
        return running_metric / total

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

        computed = [task.run(model, x, y) for task in tasks]
        self.compute.cache_clear()
        return computed
