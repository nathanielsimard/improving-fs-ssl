from typing import NamedTuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from mcp.data.dataloader.dataloader import FewShotDataLoaderFactory
from mcp.data.dataset.dataset import FewShotDataset
from mcp.model.base import Model
from mcp.task.base import Task
from mcp.training.loop import TrainingLogger, TrainingLoop


class EvaluationLoggers(NamedTuple):
    support: TrainingLogger
    evaluation: TrainingLogger


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
    ):
        self.few_shot_dataloader_factory = few_shot_dataloader_factory
        self.few_shot_dataset = few_shot_dataset
        self.model = model
        self.task = task
        self.training_loop = training_loop
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loggers = loggers

    def eval(self):
        dataloader = self.few_shot_dataloader_factory.create(self.few_shot_dataset)
        self.training_loop.fit_support(
            self.model,
            [self.task],
            dataloader.support,
            self.optimizer,
            self.scheduler,
            self.loggers.support.epoch(1, 1),
        )
        self.training_loop.evaluate(
            self.model,
            [self.task],
            dataloader.query,
            self.loggers.evaluation.epoch(1, 1),
        )
