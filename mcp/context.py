from typing import NewType

import torch
from injector import Injector, Module, inject, provider, singleton
from torch.utils.data import DataLoader

from mcp.config.dataset import Source
from mcp.config.optimizer import OptimizerType
from mcp.config.parser import ExperimentConfig
from mcp.data.dataset.cifar import CifarFsDatasetLoader
from mcp.data.dataset.dataset import DataLoaderSplits, DatasetLoader, DatasetSplits
from mcp.model.resnet import MLP
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
    def provide_model(self) -> Model:
        return MLP()


class TrainerModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @inject
    @singleton
    def provide_trainer(
        self, optimizer: torch.optim.Optimizer, dataloader_splits: DataLoaderSplits
    ) -> Trainer:
        return Trainer(optimizer, dataloader_splits.train, dataloader_splits.valid)

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


class DataModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @inject
    @singleton
    def provide_dataset_splits(self, dataset_loader: DatasetLoader) -> DatasetSplits:
        return dataset_loader.load(self.output_dir)

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
    def provider_dataloader(self, dataset_splits: DatasetSplits) -> DataLoaderSplits:
        pin_memory = True if self.device.type == "cuda" else False

        def create(dataset):
            return DataLoader(
                dataset,
                batch_size=self.config.dataloader.batch_size,
                shuffle=self.config.dataloader.shuffle,
                pin_memory=pin_memory,
            )

        return DataLoaderSplits(
            create(dataset_splits.train),
            create(dataset_splits.valid),
            create(dataset_splits.test),
        )
