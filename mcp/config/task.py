from enum import Enum
from typing import List, NamedTuple, Optional

from mcp.config.loader import ConfigType


class TaskType(Enum):
    SUPERVISED = "supervised"
    SOLARIZATION = "solarization"
    ROTATION = "rotation"
    BYOL = "byol"


class RotationConfig(NamedTuple):
    compute_tfm: bool


class BYOLConfig(NamedTuple):
    head_size: int
    hidden_size: int
    tau: float
    key_forwards: Optional[List[str]]
    key_transforms: Optional[List[str]]


class SupervisedConfig(NamedTuple):
    key_forwards: List[str]
    key_transforms: List[str]


class TaskConfig(NamedTuple):
    train: List[TaskType]
    weights: List[float]
    valid: List[TaskType]
    byol: BYOLConfig
    rotation: RotationConfig
    supervised: SupervisedConfig


def parse(config: ConfigType) -> TaskConfig:
    config = config["task"]
    task_config = TaskConfig(
        train=[TaskType(t) for t in config["train"]],
        weights=config["weights"],
        valid=[TaskType(t) for t in config["valid"]],
        byol=_parse_byol(config),
        rotation=_parse_rotation(config),
        supervised=_parse_supervised(config),
    )

    assert len(task_config.train) == len(
        task_config.weights
    ), "Number of tasks must match task weights."
    return task_config


def _parse_byol(config: ConfigType) -> BYOLConfig:
    config = config["byol"]

    key_transforms = config["key_transforms"]

    if key_transforms is not None:
        assert len(key_transforms) == 2, "Should have two keys"

    key_forwards = config["key_forwards"]

    if key_forwards is not None:
        assert len(key_forwards) == 2, "Should have two keys"

    return BYOLConfig(
        head_size=config["head_size"],
        hidden_size=config["hidden_size"],
        tau=config["tau"],
        key_forwards=key_forwards,
        key_transforms=key_transforms,
    )


def _parse_supervised(config: ConfigType) -> SupervisedConfig:
    config = config["supervised"]

    key_transforms = config["key_transforms"]
    assert len(key_transforms) > 0, "Should have at least 1 key"

    key_forwards = config["key_forwards"]
    assert len(key_forwards) > 0, "Should have at least 1 key"

    return SupervisedConfig(key_transforms=key_transforms, key_forwards=key_forwards)


def _parse_rotation(config: ConfigType) -> RotationConfig:
    config = config["rotation"]

    return RotationConfig(compute_tfm=config["compute_tfm"])
