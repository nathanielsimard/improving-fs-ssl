from typing import NamedTuple

from mcp.config.loader import ConfigType


class EvaluationConfig(NamedTuple):
    num_iterations: int


def parse(config: ConfigType) -> EvaluationConfig:
    config = config["evaluation"]

    return EvaluationConfig(num_iterations=config["num_iterations"],)
