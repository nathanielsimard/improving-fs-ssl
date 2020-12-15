from enum import Enum
from typing import List, NamedTuple

from mcp.config.loader import ConfigType


class Difficulty(Enum):
    DEFAULT = "default"
    NONE = "none"
    HARD = "hard"


class TransformConfig(NamedTuple):
    crop_size: List[int]
    crop_padding: int
    image_size: List[int]
    scale: List[int]
    difficulty: Difficulty


def parse(config: ConfigType) -> TransformConfig:
    config = config["transform"]
    crop_size = config["crop_size"]
    image_size = config["image_size"]
    scale = config["scale"]

    assert len(crop_size) == 2, "Crop size must have 2 values"
    assert len(image_size) == 2, "Image size must have 2 values"
    assert len(scale) == 2, "Scale must have two values"

    return TransformConfig(
        crop_size=crop_size,
        image_size=image_size,
        crop_padding=config["crop_padding"],
        scale=scale,
        difficulty=Difficulty(config["difficulty"]),
    )
