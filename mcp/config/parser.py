from typing import List, NamedTuple

from mcp.config.dataset import DatasetConfig
from mcp.config.dataset import parse as parse_dataset
from mcp.config.loader import ConfigType, merge

DEFAULT_CONFIG: ConfigType = {
    "dataset": {"source": "cifar_fs", "cifar_fs": {"convert_labels": True}}
}


class ExperimentConfig(NamedTuple):
    dataset: DatasetConfig


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

    return ExperimentConfig(dataset=parse_dataset(config))
