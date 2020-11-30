import random
from typing import Optional

import numpy as np
import torch

from mcp.config.parser import ExperimentConfig
from mcp.context.base import create_injector
from mcp.evaluation import Evaluation
from mcp.result.experiment import ExperimentResult
from mcp.training.trainer import Trainer
from mcp.viz.base import Vizualization


def run_train(
    config: ExperimentConfig,
    output_dir: str,
    device_str: str,
    checkpoint: Optional[int] = None,
):
    set_seed(config.seed)
    device = torch.device(device_str)
    injector = create_injector(config, output_dir, device)

    trainer = injector.get(Trainer)
    if checkpoint is not None:
        trainer.load(checkpoint)
        trainer.fit(starting_epoch=checkpoint)
    else:
        trainer.fit()


def run_eval(config: ExperimentConfig, result_dir: str, device_str: str):
    set_seed(config.seed)
    device = torch.device(device_str)
    injector = create_injector(config, result_dir, device)

    evaluation = injector.get(Evaluation)
    result = injector.get(ExperimentResult)
    evaluation.eval(result.best_epoch())


def run_viz(config: ExperimentConfig, result_dir: str, device_str: str):
    set_seed(config.seed)
    device = torch.device(device_str)
    injector = create_injector(config, result_dir, device)

    viz = injector.get(Vizualization)
    viz.plot()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
