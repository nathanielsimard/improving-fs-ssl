from typing import List, NewType

import torch
from injector import Module, inject, provider, singleton

from mcp.config.optimizer import OptimizerType, _OptimizerConfig
from mcp.config.parser import ExperimentConfig
from mcp.config.scheduler import SchedulerType, _SchedulerConfig
from mcp.context.task import TasksTrain, TasksValid, TaskTest
from mcp.model.base import Model

OptimizerTrain = NewType("OptimizerTrain", torch.optim.Optimizer)
OptimizerValid = NewType("OptimizerValid", torch.optim.Optimizer)
OptimizerTest = NewType("OptimizerTest", torch.optim.Optimizer)

SchedulerTrain = NewType("SchedulerTrain", torch.optim.lr_scheduler._LRScheduler)
SchedulerValid = NewType("SchedulerValid", torch.optim.lr_scheduler._LRScheduler)
SchedulerTest = NewType("SchedulerTest", torch.optim.lr_scheduler._LRScheduler)


class OptimizerModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @inject
    @singleton
    def provide_optimizer_train(
        self, model: Model, tasks_train: TasksTrain, tasks_valid: TasksValid
    ) -> OptimizerTrain:
        modules = [model] + tasks_train
        parameters = self._merge_param(modules)
        return self._create_optimizer(self.config.optimizer.train, parameters)

    @provider
    @inject
    @singleton
    def provide_optimizer_test(self, model: Model, task: TaskTest) -> OptimizerTest:
        return self._create_optimizer(self.config.optimizer.support, task.parameters())  # type: ignore

    @provider
    @inject
    @singleton
    def provide_optimizer_valid(
        self, model: Model, tasks_valid: TasksValid
    ) -> OptimizerValid:
        parameters = self._merge_param(tasks_valid)
        return self._create_optimizer(self.config.optimizer.support, parameters)

    def _create_optimizer(self, config: _OptimizerConfig, parameters):
        if config.type == OptimizerType.SGD:
            return torch.optim.SGD(  # type: ignore
                parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.sgd.momentum,
            )
        elif config.type == OptimizerType.ADAM:
            return torch.optim.Adam(  # type: ignore
                parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer not yet supported {config.type}")

    def _merge_param(self, modules: List[torch.nn.Module]):
        for module in modules:
            for parameter in module.parameters():
                yield parameter


class SchedulerModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @inject
    @singleton
    def provide_scheduler_train(self, optimizer: OptimizerTrain) -> SchedulerTrain:
        return self._create_scheduler(optimizer, self.config.scheduler.train)

    @provider
    @inject
    @singleton
    def provide_scheduler_valid(self, optimizer: OptimizerValid) -> SchedulerValid:
        return self._create_scheduler(optimizer, self.config.scheduler.support)

    @provider
    @inject
    @singleton
    def provide_scheduler_test(self, optimizer: OptimizerTest) -> SchedulerTest:
        return self._create_scheduler(optimizer, self.config.scheduler.support)

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer, config: _SchedulerConfig
    ):
        if config.type == SchedulerType.MULTI_STEP:
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.multistep.milestones,
                gamma=config.multistep.gamma,
            )
        elif config.type == SchedulerType.CONSTANT:
            return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 1.0)  # type: ignore
        else:
            raise ValueError(f"Scheduler not yet supported {config.type}")
