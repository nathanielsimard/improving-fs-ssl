import os
from typing import NewType

import torch
from injector import Injector, Module, inject, provider, singleton

from mcp.config.dataset import Source
from mcp.config.parser import ExperimentConfig
from mcp.context.optimizer import (
    OptimizerModule,
    OptimizerTest,
    OptimizerTrain,
    OptimizerValid,
    SchedulerModule,
    SchedulerTest,
    SchedulerTrain,
    SchedulerValid,
)
from mcp.context.task import TaskModule, TasksTrain, TasksValid, TaskTest
from mcp.data.dataloader.dataloader import (
    DataLoaderFactory,
    FewShotDataLoaderFactory,
    FewShotDataLoaderSplits,
)
from mcp.data.dataset import cifar
from mcp.data.dataset.dataset import (
    DatasetLoader,
    DatasetMetadata,
    DatasetSplits,
    FewShotDatasetSplits,
    create_few_shot_datasets,
)
from mcp.data.dataset.transforms import KorniaTransforms
from mcp.evaluation import Evaluation, EvaluationLoggers
from mcp.model.base import Model
from mcp.model.resnet import ResNet18
from mcp.result.experiment import ExperimentResult
from mcp.result.logger import ResultLogger
from mcp.task.supervised import SupervisedTask
from mcp.training.loop import TrainingLoop
from mcp.training.trainer import Trainer, TrainerLoggers

ValidFewShotDataLoaderFactory = NewType(
    "ValidFewShotDataLoaderFactory", FewShotDataLoaderFactory
)
TestFewShotDataLoaderFactory = NewType(
    "TestFewShotDataLoaderFactory", FewShotDataLoaderFactory
)


def create_injector(
    config: ExperimentConfig, output_dir: str, device: torch.device
) -> Injector:
    return Injector(
        [
            TrainerModule(config, output_dir, device),
            DataModule(config, output_dir, device),
            TaskModule(config, output_dir, device),
            ModelModule(config, output_dir, device),
            EvaluationModule(config, output_dir, device),
            OptimizerModule(config, output_dir, device),
            SchedulerModule(config, output_dir, device),
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
        return ResNet18(self.config.model.embedding_size)


SupervisedTaskTrain = NewType("SupervisedTaskTrain", SupervisedTask)
SupervisedTaskValid = NewType("SupervisedTaskValid", SupervisedTask)


class EvaluationModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @singleton
    def provide_evaluation_loggers(self) -> EvaluationLoggers:
        output_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(output_dir, exist_ok=True)

        return EvaluationLoggers(
            support=ResultLogger(
                "Evaluation Support", os.path.join(output_dir, "support")
            ),
            evaluation=ResultLogger(
                "Evaluation Support", os.path.join(output_dir, "eval")
            ),
        )

    @provider
    @singleton
    @inject
    def provide_evaluation(
        self,
        dataloader_factory: TestFewShotDataLoaderFactory,
        dataset_splits: FewShotDatasetSplits,
        task: TaskTest,
        model: Model,
        training_loop: TrainingLoop,
        optimizer: OptimizerTest,
        scheduler: SchedulerTest,
        loggers: EvaluationLoggers,
    ) -> Evaluation:
        return Evaluation(
            dataloader_factory,
            dataset_splits.test,
            model,
            task,
            training_loop,
            optimizer,
            scheduler,
            loggers,
            self.config.evaluation.num_iterations,
            checkpoint_dir(self.output_dir),
            self.device,
        )

    @provider
    @singleton
    def provide_experiment_result(self) -> ExperimentResult:
        return ExperimentResult(self.config, self.output_dir)


class TrainerModule(Module):
    def __init__(self, config: ExperimentConfig, output_dir: str, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

    @provider
    @singleton
    def provide_kornia_transformations(self) -> KorniaTransforms:
        if self.config.dataset.source == Source.CIFAR_FS:
            return KorniaTransforms(cifar.IMAGES_MEAN, cifar.IMAGES_STD, (32, 32), 4)
        else:
            raise ValueError(
                f"Dataset source not yet supported {self.config.dataset.source}"
            )

    @provider
    @singleton
    def provide_training_loop(self) -> TrainingLoop:
        return TrainingLoop(
            self.device,
            self.config.trainer.support_training.min_loss,
            self.config.trainer.support_training.max_epochs,
        )

    @provider
    @singleton
    def provide_training_loggers(self) -> TrainerLoggers:
        output_dir = os.path.join(self.output_dir, "train")
        os.makedirs(output_dir, exist_ok=True)

        return TrainerLoggers(
            train=ResultLogger("Training", os.path.join(output_dir, "train")),
            support=ResultLogger(
                "Training - Support", os.path.join(output_dir, "support")
            ),
            evaluation=ResultLogger(
                "Training - Evaluation", os.path.join(output_dir, "eval")
            ),
        )

    @provider
    @inject
    @singleton
    def provide_trainer(
        self,
        model: Model,
        optimizer_train: OptimizerTrain,
        optimizer_support: OptimizerValid,
        scheduler_train: SchedulerTrain,
        scheduler_support: SchedulerValid,
        dataloader_splits: FewShotDataLoaderSplits,
        trainer_loggers: TrainerLoggers,
        training_loop: TrainingLoop,
        tasks_train: TasksTrain,
        tasks_valid: TasksValid,
    ) -> Trainer:
        return Trainer(
            model,
            optimizer_train,
            optimizer_support,
            scheduler_train,
            scheduler_support,
            dataloader_splits.train,
            dataloader_splits.valid,
            tasks_train,
            tasks_valid,
            self.config.trainer.epochs,
            training_loop,
            trainer_loggers,
            self.device,
            checkpoint_dir(self.output_dir),
        )


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
            return cifar.CifarFsDatasetLoader(
                self.config.dataset.cifar_fs.convert_labels
            )
        else:
            raise ValueError(
                f"Dataset source not yet supported {self.config.dataset.source}"
            )

    @provider
    @singleton
    def provide_dataloader_factory(self) -> DataLoaderFactory:
        pin_memory = True if self.device.type == "cuda" else False

        return DataLoaderFactory(
            self.config.dataloader.batch_size,
            self.config.dataloader.shuffle,
            pin_memory,
        )

    @provider
    @inject
    @singleton
    def provide_valid_few_shot_dataloader_factory(
        self, dataloader_factory: DataLoaderFactory, dataset_metadata: DatasetMetadata
    ) -> ValidFewShotDataLoaderFactory:
        return FewShotDataLoaderFactory(  # type: ignore
            dataset_metadata.valid_num_class,
            self.config.dataset.n_way,
            dataloader_factory,
        )

    @provider
    @inject
    @singleton
    def provide_few_shot_dataloader_factory(
        self, dataloader_factory: DataLoaderFactory, dataset_metadata: DatasetMetadata
    ) -> TestFewShotDataLoaderFactory:
        return FewShotDataLoaderFactory(  # type: ignore
            dataset_metadata.test_num_class,
            self.config.dataset.n_way,
            dataloader_factory,
        )

    @provider
    @inject
    @singleton
    def provide_few_shot_dataloader_splits(
        self,
        dataset_splits: FewShotDatasetSplits,
        dataloader_factory: DataLoaderFactory,
        test_dt: TestFewShotDataLoaderFactory,
        valid_dt: ValidFewShotDataLoaderFactory,
    ) -> FewShotDataLoaderSplits:
        return FewShotDataLoaderSplits(
            dataloader_factory.create(dataset_splits.train),
            valid_dt.create(dataset_splits.valid),
            test_dt.create(dataset_splits.test),
        )


def checkpoint_dir(output_dir):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
