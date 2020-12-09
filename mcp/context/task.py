from copy import deepcopy
from typing import NewType

import torch
from injector import Injector, Module, inject, multiprovider, provider, singleton

from mcp.config.parser import ExperimentConfig
from mcp.config.task import TaskType
from mcp.data.dataset.dataset import DatasetMetadata
from mcp.data.dataset.transforms import KorniaTransforms
from mcp.task.base import Task, WeightedTask
from mcp.task.byol import BYOLTask
from mcp.task.compute import TaskCompute
from mcp.task.rotation import BatchRotation, RotationTask
from mcp.task.solarization import BatchSolarization, SolarizationTask
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
            TaskType.SOLARIZATION: SolarizationTask,
            TaskType.SUPERVISED: SupervisedTask,
            TaskType.BYOL: BYOLTask,
        }

    @provider
    @inject
    @singleton
    def provide_compute(self, transforms: KorniaTransforms) -> TaskCompute:
        return TaskCompute(
            transforms,
            self.config.transform.difficulty,
            tuple(self.config.transform.scale),  # type: ignore
        )

    @provider
    @inject
    @singleton
    def provide_batch_rotation(self, transforms: KorniaTransforms) -> BatchRotation:
        return BatchRotation(transforms)

    @provider
    @inject
    @singleton
    def provide_batch_solarization(
        self, transforms: KorniaTransforms
    ) -> BatchSolarization:
        return BatchSolarization(transforms)

    @provider
    @inject
    def provide_rotation_task(
        self, compute: TaskCompute, batch_rotation: BatchRotation
    ) -> RotationTask:
        return RotationTask(
            self.config.model.embedding_size,
            compute,
            batch_rotation,
            compute_tfm=self.config.task.rotation.compute_tfm,
        )

    @provider
    @inject
    def provide_solarization_task(
        self, compute: TaskCompute, batch_solarization: BatchSolarization
    ) -> SolarizationTask:
        return SolarizationTask(
            self.config.model.embedding_size, compute, batch_solarization
        )

    @provider
    @inject
    def provide_byol_task(
        self, transforms: KorniaTransforms, compute: TaskCompute
    ) -> BYOLTask:
        def fix_key(key):
            if key is None:
                return None
            return tuple(key)

        return BYOLTask(
            self.config.model.embedding_size,
            transforms,
            self.config.task.byol.head_size,
            self.config.task.byol.hidden_size,
            self.config.task.byol.tau,
            tuple(self.config.transform.scale),  # type: ignore
            fix_key(self.config.task.byol.key_transforms),
            fix_key(self.config.task.byol.key_forwards),
            compute,
        )

    @provider
    @inject
    def provide_train_supervised_task(
        self, metadata: DatasetMetadata, compute: TaskCompute
    ) -> SupervisedTaskTrain:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size,
            metadata.train_num_class,
            compute,
            self.config.task.supervised.key_transform,
            self.config.task.supervised.key_forward,
        )

    @provider
    @inject
    def provide_valid_supervised_task(
        self, metadata: DatasetMetadata, compute: TaskCompute
    ) -> SupervisedTaskValid:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size,
            metadata.valid_num_class,
            compute,
            self.config.task.supervised.key_transform,
            self.config.task.supervised.key_forward,
        )

    @provider
    @inject
    @singleton
    def provide_test_task(
        self, metadata: DatasetMetadata, compute: TaskCompute,
    ) -> TaskTest:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size,
            metadata.test_num_class,
            compute,
            self.config.task.supervised.key_transform,
            self.config.task.supervised.key_forward,
        )

    @multiprovider
    @inject
    @singleton
    def provide_train_tasks(self, injector: Injector) -> TasksTrain:
        return [
            WeightedTask(injector.get(self._get_train_class(t)), w)  # type: ignore
            for t, w in zip(self.config.task.train, self.config.task.weights)
        ]

    @multiprovider
    @inject
    @singleton
    def provide_valid_tasks(self, injector: Injector) -> TasksValid:
        return [  # type: ignore
            injector.get(self._get_valid_class(t)) for t in self.config.task.valid  # type: ignore
        ]

    def _get_train_class(self, task: TaskType):
        classes = deepcopy(self._default_task_classes)
        classes[TaskType.SUPERVISED] = SupervisedTaskTrain  # type: ignore
        return classes[task]

    def _get_valid_class(self, task: TaskType):
        classes = deepcopy(self._default_task_classes)
        classes[TaskType.SUPERVISED] = SupervisedTaskValid  # type: ignore
        return classes[task]
