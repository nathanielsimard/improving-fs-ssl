from multiprocessing import cpu_count
from typing import List, NamedTuple

from mcp.config.dataloader import DataLoaderConfig
from mcp.config.dataloader import parse as parse_dataloader
from mcp.config.dataset import DatasetConfig
from mcp.config.dataset import parse as parse_dataset
from mcp.config.loader import ConfigType, merge
from mcp.config.model import ModelConfig
from mcp.config.model import parse as parse_model
from mcp.config.optimizer import OptimizerConfig
from mcp.config.optimizer import parse as parse_optimizer
from mcp.config.trainer import TrainerConfig
from mcp.config.trainer import parse as parse_trainer

_DEFAULT_OPTIMIZER_CONFIG = {
    "type": "sgd",
    "sgd": {"momentum": 0.9},
    "weight_decay": 5e-4,
    "learning_rate": 0.05,
}
DEFAULT_CONFIG: ConfigType = {
    "dataset": {
        "num_samples": 5,
        "source": "cifar_fs",
        "cifar_fs": {"convert_labels": True},
    },
    "dataloader": {"batch_size": 32, "shuffle": True, "num_workers": cpu_count()},
    "optimizer": {
        "train": _DEFAULT_OPTIMIZER_CONFIG,
        "support": _DEFAULT_OPTIMIZER_CONFIG,
    },
    "trainer": {
        "epochs": 90,
        "tasks": ["supervised"],
        "support_training": {"max_epochs": 50, "min_loss": 0.001},
    },
    "model": {"embedding_size": 256},
}


class ExperimentConfig(NamedTuple):
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    model: ModelConfig


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
        dataset=parse_dataset(config),
        dataloader=parse_dataloader(config),
        optimizer=parse_optimizer(config),
        trainer=parse_trainer(config),
        model=parse_model(config),
    )
