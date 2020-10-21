from typing import List, NewType

import torch
from injector import Injector, Module, inject, multiprovider, provider, singleton
from torch.utils.data import DataLoader

from mcp.config.dataset import Source
from mcp.config.optimizer import OptimizerType
from mcp.config.parser import ExperimentConfig
from mcp.config.trainer import TaskType
from mcp.data.dataset.cifar import CifarFsDatasetLoader
from mcp.data.dataset.dataset import (
    FewShotDataLoaderSplits,
    FewShotDatasetSplits,
    FewShotDataLoader,
    FewShotDataset,
    Dataset,
    DatasetLoader,
    DatasetMetadata,
    DatasetSplits,
    create_few_shot_datasets,
)
from mcp.model.resnet import ResNet18
from mcp.task.base import Task
from mcp.task.supervised import SupervisedTask
from mcp.training.trainer import Trainer

Model = NewType("Model", torch.nn.Module)


def create_injector(
    config: ExperimentConfig, output_dir: str, device: torch.device
) -> Injector:
    return Injector(
        [
            TrainerModule(config, output_dir, device),
            DataModule(config, output_dir, device),
            ModelModule(config, output_dir, device),
        ]
    )


class ModelModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @inject
    @singleton
    def provide_model(self, dataset_metadata: DatasetMetadata) -> Model:
        return ResNet18(dataset_metadata.train_num_class)


class TrainerModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @multiprovider
    @inject
    @singleton
    def provide_tasks(self, injector: Injector) -> List[Task]:
        return [injector.get(self._get_class(t)) for t in self.config.trainer.tasks]

    @provider
    @singleton
    def provide_supervised_task(self) -> SupervisedTask:
        return SupervisedTask()

    @provider
    @inject
    @singleton
    def provide_trainer(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        dataloader_splits: FewShotDataLoaderSplits,
        tasks: List[Task],
    ) -> Trainer:
        return Trainer(
            model,
            optimizer,
            dataloader_splits.train,
            dataloader_splits.valid,
            tasks,
            self.config.trainer.epochs,
        )

    @provider
    @inject
    @singleton
    def provide_optimizer(self, model: Model) -> torch.optim.Optimizer:
        if self.config.optimizer.type == OptimizerType.SGD:
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                momentum=self.config.optimizer.sgd.momentum,
            )
        else:
            raise ValueError(
                f"Optimizer not yet supported {self.config.optimizer.type}"
            )

    def _get_class(self, task: TaskType):
        if task == TaskType.SUPERVISED:
            return SupervisedTask
        else:
            raise ValueError(f"Task type not yet supported {task}")


class DataModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @inject
    @singleton
    def provide_dataset_metadata(
        self, dataset_loader: DatasetLoader
    ) -> DatasetMetadata:
        return dataset_loader.metadata

    @provider
    @inject
    @singleton
    def provide_dataset_splits(self, dataset_loader: DatasetLoader) -> DatasetSplits:
        return dataset_loader.load(self.output_dir)

    @provider
    @inject
    @singleton
    def provide_few_shot_dataset_splits(
        self, dataset_splits: DatasetSplits
    ) -> FewShotDatasetSplits:
        valid = create_few_shot_datasets(
            dataset_splits.valid, self.config.dataset.num_samples
        )
        test = create_few_shot_datasets(
            dataset_splits.test, self.config.dataset.num_samples
        )

        return FewShotDatasetSplits(dataset_splits.train, valid, test)

    @provider
    @inject
    @singleton
    def provide_dataset_loader(self) -> DatasetLoader:
        if self.config.dataset.source == Source.CIFAR_FS:
            return CifarFsDatasetLoader(self.config.dataset.cifar_fs.convert_labels)
        else:
            raise ValueError(
                f"Dataset source not yet supported {self.config.dataset.source}"
            )

    @provider
    @inject
    @singleton
    def provide_few_shot_dataloader_splits(
        self, dataset_splits: FewShotDatasetSplits
    ) -> FewShotDataLoaderSplits:
        pin_memory = True if self.device.type == "cuda" else False

        def create(dataset: Dataset) -> DataLoader:
            return DataLoader(
                dataset,
                batch_size=self.config.dataloader.batch_size,
                shuffle=self.config.dataloader.shuffle,
                pin_memory=pin_memory,
            )

        def create_few_shot(dataset: FewShotDataset) -> FewShotDataLoader:
            return FewShotDataLoader(
                support=create(dataset.support), query=create(dataset.query)
            )

        return FewShotDataLoaderSplits(
            create(dataset_splits.train),
            create_few_shot(dataset_splits.valid),
            create_few_shot(dataset_splits.test),
        )
