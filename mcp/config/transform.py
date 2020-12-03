from typing import List, NamedTuple

from mcp.config.loader import ConfigType


class TransformConfig(NamedTuple):
    crop_size: List[int]
    crop_padding: int
    image_size: List[int]


def parse(config: ConfigType) -> TransformConfig:
    config = config["transform"]
    crop_size = config["crop_size"]
    image_size = config["image_size"]

    assert len(crop_size) == 2, "Crop size must have 2 values"
    assert len(image_size) == 2, "Image size must have 2 values"

    return TransformConfig(
        crop_size=crop_size, image_size=image_size, crop_padding=config["crop_padding"],
    )
