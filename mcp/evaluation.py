from typing import NamedTuple
import os

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mcp.data.dataloader.dataloader import FewShotDataLoaderFactory
from mcp.data.dataset.dataset import FewShotDataset
from mcp.model.base import Model
from mcp.result.logger import ResultLogger
from mcp.task.base import Task
from mcp.training.loop import TrainingLoop


class EvaluationLoggers(NamedTuple):
    support: ResultLogger
    evaluation: ResultLogger


class Evaluation(object):
    def __init__(
        self,
        few_shot_dataloader_factory: FewShotDataLoaderFactory,
        few_shot_dataset: FewShotDataset,
        model: Model,
        task: Task,
        training_loop: TrainingLoop,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loggers: EvaluationLoggers,
        num_iterations: int,
        save_path: str,
    ):
        self.few_shot_dataloader_factory = few_shot_dataloader_factory
        self.few_shot_dataset = few_shot_dataset
        self.model = model
        self.task = task
        self.training_loop = training_loop
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loggers = loggers
        self.num_iterations = num_iterations
        self.save_path = save_path

    def eval(self, epoch: int):
        self.model.load(self._model_path(epoch))
        for i in range(1, self.num_iterations + 1):
            dataloader = self.few_shot_dataloader_factory.create(self.few_shot_dataset)
            self.training_loop.fit_support(
                self.model,
                [self.task],
                dataloader.support,
                self.optimizer,
                self.scheduler,
                self.loggers.support.epoch(i, self.num_iterations),
            )
            self.training_loop.evaluate(
                self.model,
                [self.task],
                dataloader.query,
                self.loggers.evaluation.epoch(i, self.num_iterations),
            )

    def _model_path(self, epoch: int) -> str:
        return os.path.join(self.save_path, f"model-{epoch}.pth")
