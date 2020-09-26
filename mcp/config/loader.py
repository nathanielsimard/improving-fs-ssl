from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Union

import yaml

from mcp.utils import logging

logger = logging.create_logger(__name__)

ItemType = Any
ConfigType = Dict[str, Any]


def load(file_path: str) -> ConfigType:
    logger.info(f"Loading config '{file_path}'")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save(config: ConfigType, file_path: str):
    logger.info(f"Saving config '{file_path}'")
    with open(file_path, "w") as file:
        return yaml.dump(config, file)


def merge(config: ConfigType, default: ConfigType) -> ConfigType:
    """Merge two configurations.

    Support nested keys, but everyting in a list will be overrided
    by the new config if a new one is provided.
    """
    leaf_keys_default: List[str] = _find_leaf_keys(default)
    leaf_keys_overrided: List[str] = _find_leaf_keys(config)

    for key in leaf_keys_overrided:
        if key not in leaf_keys_default:
            logger.warning(f"Key : {key} is not in default config.")
            raise ValueError(f"Key : {key} is not in default config.")

    logger.debug(leaf_keys_overrided)
    new_config = deepcopy(default)

    for key in leaf_keys_overrided:
        keys = key.split(".")
        if len(keys) > 1:
            new_value = _read_keys(config, keys)
            sub_config = _read_keys(new_config, keys[:-1])
            sub_config[keys[-1]] = new_value
        else:
            new_value = config[key]
            new_config[key] = new_value

        logger.debug(f"Overrided key {key} with value {new_value}")

    return new_config


def to_dict(config) -> Dict:
    format_dict = {}
    attributes = config._asdict()
    for key, value in attributes.items():
        try:
            value._asdict()
            format_dict[key] = to_dict(value)
        except Exception:
            if isinstance(value, Enum):
                format_dict[key] = value.value
            elif isinstance(value, list):
                value_list = []
                for v in value:
                    if isinstance(v, Enum):
                        value_list.append(v.value)
                    else:
                        value_list.append(v)
                format_dict[key] = value_list  # type: ignore
            else:
                format_dict[key] = value

    return format_dict


def _find_leaf_keys(config: ConfigType) -> List[str]:
    keys = list(config.keys())
    leaf_keys: List[str] = []
    while len(keys) != 0:
        key = keys.pop()
        item = _read_keys(config, key.split("."))
        _merge(item, leaf_keys, keys, key)

    return leaf_keys


def _merge(
    item: Union[ConfigType, ItemType], leaf_keys: List[str], keys: List[str], key: str
):
    try:
        for s in item.keys():
            keys.append(f"{key}.{s}")
    except AttributeError:
        leaf_keys.append(key)


def _read_keys(config: ConfigType, keys: List[str]) -> Union[ConfigType, ItemType]:
    for k in keys:
        config = config[k]
    return config
