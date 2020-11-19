from enum import Enum
from typing import List, NamedTuple

from mcp.config.loader import ConfigType


class TaskType(Enum):
    SUPERVISED = "supervised"
    ROTATION = "rotation"
    BYOL = "byol"


class BYOLConfig(NamedTuple):
    head_size: int
    head_n_hiddens: int
    tau: float


class TaskConfig(NamedTuple):
    types: List[TaskType]
    byol: BYOLConfig


def parse(config: ConfigType) -> TaskConfig:
    config = config["task"]

    return TaskConfig(
        types=[TaskType(t) for t in config["types"]], byol=_parse_byol(config)
    )


def _parse_byol(config: ConfigType) -> BYOLConfig:
    config = config["byol"]

    return BYOLConfig(
        head_size=config["head_size"],
        head_n_hiddens=config["head_n_hiddens"],
        tau=config["tau"],
    )
