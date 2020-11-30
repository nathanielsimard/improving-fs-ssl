from enum import Enum
from typing import NamedTuple

from mcp.config.loader import ConfigType


class BestWeightsMetric(Enum):
    TIME = "time"
    LOSS = "loss"
    METRIC = "metric"


class EvaluationConfig(NamedTuple):
    num_iterations: int
    metric: BestWeightsMetric


def parse(config: ConfigType) -> EvaluationConfig:
    config = config["evaluation"]

    return EvaluationConfig(
        num_iterations=config["num_iterations"],
        metric=BestWeightsMetric(config["metric"]),
    )
