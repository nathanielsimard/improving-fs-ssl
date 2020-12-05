from enum import Enum
from typing import List, NamedTuple

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
    scale: List[float]


class TaskConfig(NamedTuple):
    train: List[TaskType]
    weights: List[float]
    valid: List[TaskType]
    byol: BYOLConfig
    rotation: RotationConfig


def parse(config: ConfigType) -> TaskConfig:
    config = config["task"]
    task_config = TaskConfig(
        train=[TaskType(t) for t in config["train"]],
        weights=config["weights"],
        valid=[TaskType(t) for t in config["valid"]],
        byol=_parse_byol(config),
        rotation=_parse_rotation(config),
    )

    assert len(task_config.train) == len(
        task_config.weights
    ), "Number of tasks must match task weights."
    return task_config


def _parse_byol(config: ConfigType) -> BYOLConfig:
    config = config["byol"]
    scale = config["scale"]

    assert len(scale) == 2, "Scale must have two values"

    return BYOLConfig(
        head_size=config["head_size"],
        hidden_size=config["hidden_size"],
        tau=config["tau"],
        scale=scale,
    )


def _parse_rotation(config: ConfigType) -> RotationConfig:
    config = config["rotation"]

    return RotationConfig(compute_tfm=config["compute_tfm"],)
