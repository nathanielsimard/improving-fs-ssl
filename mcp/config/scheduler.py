from enum import Enum
from typing import List, NamedTuple

from mcp.config.loader import ConfigType


class SchedulerType(Enum):
    MULTI_STEP = "multistep"
    CONSTANT = "constant"


class MultistepConfig(NamedTuple):
    milestones: List[int]
    gamma: float


class _SchedulerConfig(NamedTuple):
    type: SchedulerType
    multistep: MultistepConfig


class SchedulerConfig(NamedTuple):
    train: _SchedulerConfig
    support: _SchedulerConfig


def _parse_multistep(config: ConfigType) -> MultistepConfig:
    config = config["multistep"]
    return MultistepConfig(milestones=config["milestones"], gamma=config["gamma"])


def parse(config: ConfigType) -> SchedulerConfig:
    config = config["scheduler"]
    return SchedulerConfig(
        train=_parse(config, "train"), support=_parse(config, "support"),
    )


def _parse(config: ConfigType, tag: str) -> _SchedulerConfig:
    config = config[tag]
    return _SchedulerConfig(
        type=SchedulerType(config["type"]), multistep=_parse_multistep(config),
    )
