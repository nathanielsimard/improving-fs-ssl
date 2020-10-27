from typing import Optional

import torch

from mcp.config.parser import ExperimentConfig
from mcp.context import create_injector
from mcp.evaluation import Evaluation
from mcp.training.trainer import Trainer


def run_train(
    config: ExperimentConfig,
    output_dir: str,
    device_str: str,
    checkpoint: Optional[int] = None,
):
    device = torch.device(device_str)
    injector = create_injector(config, output_dir, device)

    trainer = injector.get(Trainer)
    if checkpoint is not None:
        trainer.load(checkpoint)
    trainer.fit()


def run_eval(config: ExperimentConfig, result_dir: str, device_str: str):
    device = torch.device(device_str)
    injector = create_injector(config, result_dir, device)

    evaluation = injector.get(Evaluation)
    evaluation.eval()
