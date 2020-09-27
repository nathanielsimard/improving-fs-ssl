import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import List, Optional

FORMATTER = logging.Formatter("[%(levelname)s] %(message)s")

handlers: List[logging.Handler] = []
level = None


def initialize(file_name: Optional[str] = None, debug: bool = False, std: bool = False):
    if file_name is not None:
        _initialize_handler_file(file_name)

    if std:
        _initialize_handler_std()

    _initialize_level(debug)


def _initialize_handler_file(file_name):
    """Create a handler_file that rotate files each 512 mb."""
    handler_file = RotatingFileHandler(
        file_name, mode="a", maxBytes=536_870_912, backupCount=4, encoding=None
    )
    handler_file.setFormatter(FORMATTER)
    handlers.append(handler_file)


def _initialize_handler_std():
    handler_std = logging.StreamHandler(stream=sys.stdout)
    handler_std.setFormatter(FORMATTER)
    handlers.append(handler_std)


def _initialize_level(debug):
    global level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and formatter."""
    initialized = level is not None and len(handlers) > 0

    if not initialized:
        initialize()

    logger = logging.getLogger(name)
    logger.setLevel(level)  # type: ignore

    for handler in handlers:
        logger.addHandler(handler)

    return logger
