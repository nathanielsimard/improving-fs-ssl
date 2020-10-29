import os
from typing import List, NewType

import torch
from injector import Injector, Module, inject, multiprovider, provider, singleton

from mcp.config.dataset import Source
from mcp.config.optimizer import OptimizerType, _OptimizerConfig
from mcp.config.parser import ExperimentConfig
from mcp.config.scheduler import SchedulerType, _SchedulerConfig
from mcp.config.trainer import TaskType
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
from mcp.task.base import Task
from mcp.task.supervised import SupervisedTask
from mcp.training.loop import TrainingLoop
from mcp.training.trainer import Trainer, TrainerLoggers

TasksTrain = NewType("TasksTrain", list)
TasksValid = NewType("TasksValid", list)
TaskTest = NewType("TaskTest", Task)

OptimizerTrain = NewType("OptimizerTrain", torch.optim.Optimizer)
OptimizerSupport = NewType("OptimizerSupport", torch.optim.Optimizer)
SchedulerTrain = NewType("SchedulerTrain", torch.optim.lr_scheduler._LRScheduler)
SchedulerSupport = NewType("SchedulerSupport", torch.optim.lr_scheduler._LRScheduler)

OptimizerEvaluation = NewType("OptimizerEvaluation", torch.optim.Optimizer)
SchedulerEvaluation = NewType(
    "SchedulerEvaluation", torch.optim.lr_scheduler._LRScheduler
)

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
            ModelModule(config, output_dir, device),
            EvaluationModule(config, output_dir, device),
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
        optimizer: OptimizerEvaluation,
        scheduler: SchedulerEvaluation,
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

    @multiprovider
    @inject
    @singleton
    def provide_train_tasks(self, injector: Injector) -> TasksTrain:
        return [  # type: ignore
            injector.get(self._get_train_class(t)) for t in self.config.trainer.tasks  # type: ignore
        ]

    @multiprovider
    @inject
    @singleton
    def provide_valid_tasks(self, injector: Injector) -> TasksValid:
        return [  # type: ignore
            injector.get(SupervisedTaskValid)
        ]

    @multiprovider
    @inject
    @singleton
    def provide_test_task(
        self, metadata: DatasetMetadata, transforms: KorniaTransforms,
    ) -> TaskTest:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size, metadata.test_num_class, transforms
        )

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
    @inject
    @singleton
    def provide_train_supervised_task(
        self, metadata: DatasetMetadata, transforms: KorniaTransforms
    ) -> SupervisedTaskTrain:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size, metadata.train_num_class, transforms
        )

    @provider
    @inject
    @singleton
    def provide_valid_supervised_task(
        self, metadata: DatasetMetadata, transforms: KorniaTransforms
    ) -> SupervisedTaskValid:
        return SupervisedTask(  # type: ignore
            self.config.model.embedding_size, metadata.valid_num_class, transforms
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
        optimizer_support: OptimizerSupport,
        scheduler_train: SchedulerTrain,
        scheduler_support: SchedulerSupport,
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

    @provider
    @inject
    @singleton
    def provide_optimizer_scheduler_train(
        self, optimizer: OptimizerTrain
    ) -> SchedulerTrain:
        return self._create_scheduler(optimizer, self.config.scheduler.train)

    @provider
    @inject
    @singleton
    def provide_optimizer_scheduler_support(
        self, optimizer: OptimizerSupport
    ) -> SchedulerSupport:
        return self._create_scheduler(optimizer, self.config.scheduler.support)

    @provider
    @inject
    @singleton
    def provide_optimizer_scheduler_eval(
        self, optimizer: OptimizerEvaluation
    ) -> SchedulerEvaluation:
        return self._create_scheduler(optimizer, self.config.scheduler.support)

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
    def provide_optimizer_eval(
        self, model: Model, task: TaskTest
    ) -> OptimizerEvaluation:
        return self._create_optimizer(self.config.optimizer.support, task.parameters())  # type: ignore

    @provider
    @inject
    @singleton
    def provide_optimizer_support(
        self, model: Model, tasks_valid: TasksValid
    ) -> OptimizerSupport:
        parameters = self._merge_param(tasks_valid)
        return self._create_optimizer(self.config.optimizer.support, parameters)

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

    def _get_train_class(self, task: TaskType):
        if task == TaskType.SUPERVISED:
            return SupervisedTaskTrain
        else:
            raise ValueError(f"Training Task type not yet supported {task}")


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
