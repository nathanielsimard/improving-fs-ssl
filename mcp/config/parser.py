from copy import deepcopy
from multiprocessing import cpu_count
from typing import List, NamedTuple

from mcp.config.dataloader import DataLoaderConfig
from mcp.config.dataloader import parse as parse_dataloader
from mcp.config.dataset import DatasetConfig
from mcp.config.dataset import parse as parse_dataset
from mcp.config.evaluation import EvaluationConfig
from mcp.config.evaluation import parse as parse_evaluation
from mcp.config.loader import ConfigType, merge
from mcp.config.model import ModelConfig
from mcp.config.model import parse as parse_model
from mcp.config.optimizer import OptimizerConfig
from mcp.config.optimizer import parse as parse_optimizer
from mcp.config.scheduler import SchedulerConfig
from mcp.config.scheduler import parse as parse_scheduler
from mcp.config.task import TaskConfig
from mcp.config.task import parse as parse_task
from mcp.config.trainer import TrainerConfig
from mcp.config.trainer import parse as parse_trainer
from mcp.config.transform import TransformConfig
from mcp.config.transform import parse as parse_transform

_DEFAULT_OPTIMIZER_CONFIG = {
    "type": "sgd",
    "sgd": {"momentum": 0.9},
    "weight_decay": 5e-4,
    "learning_rate": 0.05,
}

_DEFAULT_SCHEDULER_CONFIG = {
    "type": "multistep",
    "multistep": {"milestones": [45, 60, 75], "gamma": 0.1},
}

_SUPPORT_SCHEDULER_CONFIG = deepcopy(_DEFAULT_SCHEDULER_CONFIG)
_SUPPORT_SCHEDULER_CONFIG["type"] = "constant"

DEFAULT_CONFIG: ConfigType = {
    "seed": 42,
    "dataset": {
        "num_samples": 5,
        "n_way": 5,
        "source": "cifar_fs",
        "cifar_fs": {"convert_labels": True},
    },
    "dataloader": {"batch_size": 64, "shuffle": True, "num_workers": cpu_count()},
    "optimizer": {
        "train": deepcopy(_DEFAULT_OPTIMIZER_CONFIG),
        "support": deepcopy(_DEFAULT_OPTIMIZER_CONFIG),
    },
    "scheduler": {
        "train": deepcopy(_DEFAULT_SCHEDULER_CONFIG),
        "support": deepcopy(_SUPPORT_SCHEDULER_CONFIG),
    },
    "trainer": {
        "epochs": 90,
        "support_training": {"max_epochs": 300, "min_loss": 0.01},
        "num_valid_iterations": 5,
        "num_checkpoints": 10,
    },
    "task": {
        "train": ["rotation", "byol"],
        "valid": ["supervised"],
        "byol": {
            "head_size": 128,
            "hidden_size": 2048,
            "tau": 0.99,
            "scale": [0.08, 1.0],
        },
        "rotation": {"compute_tfm": False},
        "weights": [1, 250],
    },
    "model": {"embedding_size": 512},
    "transform": {"crop_size": [96, 96], "crop_padding": 4, "image_size": [96, 96]},
    "evaluation": {"num_iterations": 25, "metric": "metric"},
}


class ExperimentConfig(NamedTuple):
    seed: int
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    task: TaskConfig
    transform: TransformConfig


def parse(
    configs: List[ConfigType], default: ConfigType = DEFAULT_CONFIG
) -> ExperimentConfig:
    """Parse the configurations.

    Multiple config are supported and will be merge in order.
    It implies that the last config in the list has the highest priority
    and will override any preceding config with the same key.
    """
    config = default

    for c in configs:
        config = merge(c, config)

    return ExperimentConfig(
        seed=config["seed"],
        dataset=parse_dataset(config),
        dataloader=parse_dataloader(config),
        optimizer=parse_optimizer(config),
        scheduler=parse_scheduler(config),
        trainer=parse_trainer(config),
        model=parse_model(config),
        evaluation=parse_evaluation(config),
        task=parse_task(config),
        transform=parse_transform(config),
    )
