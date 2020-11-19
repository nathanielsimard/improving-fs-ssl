from copy import deepcopy
from typing import NewType

import torch
from injector import Injector, Module, inject, multiprovider, provider, singleton

from mcp.config.parser import ExperimentConfig
from mcp.config.task import TaskType
from mcp.data.dataset.dataset import DatasetMetadata
from mcp.data.dataset.transforms import KorniaTransforms
from mcp.task.base import Task
from mcp.task.byol import BYOLTask
from mcp.task.compute import TaskCompute
from mcp.task.rotation import BatchRotation, RotationTask
from mcp.task.supervised import SupervisedTask

TasksTrain = NewType("TasksTrain", list)
TasksValid = NewType("TasksValid", list)
TaskTest = NewType("TaskTest", Task)


SupervisedTaskTrain = NewType("SupervisedTaskTrain", SupervisedTask)
SupervisedTaskValid = NewType("SupervisedTaskValid", SupervisedTask)


class TaskModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self._default_task_classes = {
            TaskType.ROTATION: RotationTask,
            TaskType.SUPERVISED: SupervisedTask,
            TaskType.BYOL: BYOLTask,
        }

    @provider
    @inject
    @singleton
    def provide_compute(self, transforms: KorniaTransforms) -> TaskCompute:
        return TaskCompute(transforms)

    @provider
    @inject
    @singleton
    def provide_batch_rotation(self, transforms: KorniaTransforms) -> BatchRotation:
        return BatchRotation(transforms)

    @provider
    @inject
    def provide_rotation_task(
        self, compute: TaskCompute, batch_rotation: BatchRotation
    ) -> RotationTask:
        return RotationTask(self.config.model.embedding_size, compute, batch_rotation)

    @provider
    @inject
    def provide_byol_task(self, compute: TaskCompute) -> BYOLTask:
        return BYOLTask(
            self.config.model.embedding_size,
            compute,
            self.config.task.byol.head_size,
            self.config.task.byol.tau,
        )

    @provider
    @inject
    def provide_train_supervised_task(
        self, metadata: DatasetMetadata, compute: TaskCompute
    ) -> SupervisedTaskTrain:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size, metadata.train_num_class, compute
        )

    @provider
    @inject
    def provide_valid_supervised_task(
        self, metadata: DatasetMetadata, compute: TaskCompute
    ) -> SupervisedTaskValid:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size, metadata.valid_num_class, compute
        )

    @provider
    @inject
    @singleton
    def provide_test_task(
        self, metadata: DatasetMetadata, compute: TaskCompute,
    ) -> TaskTest:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size, metadata.test_num_class, compute
        )

    @multiprovider
    @inject
    @singleton
    def provide_train_tasks(self, injector: Injector) -> TasksTrain:
        return [  # type: ignore
            injector.get(self._get_train_class(t)) for t in self.config.task.types  # type: ignore
        ]

    @multiprovider
    @inject
    @singleton
    def provide_valid_tasks(self, injector: Injector) -> TasksValid:
        return [  # type: ignore
            injector.get(self._get_valid_class(t)) for t in self.config.task.types  # type: ignore
        ]

    def _get_train_class(self, task: TaskType):
        classes = deepcopy(self._default_task_classes)
        classes[TaskType.SUPERVISED] = SupervisedTaskTrain  # type: ignore
        return classes[task]

    def _get_valid_class(self, task: TaskType):
        classes = deepcopy(self._default_task_classes)
        classes[TaskType.SUPERVISED] = SupervisedTaskValid  # type: ignore
        return classes[task]
