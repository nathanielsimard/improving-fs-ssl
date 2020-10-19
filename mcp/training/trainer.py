from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mcp.utils.logging import create_logger

logger = create_logger(__name__)


class Trainer(object):
    def __init__(
        self,
        optimizer: Optimizer,
        dataloader_train: DataLoader,
        dataloader_valid: DataLoader,
    ):
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid

    def fit(self, model: nn.Module):
        num_params = num_trainable_parameters(model)
        logger.info(
            f"Fitting the model | {num_params} parameters | "
            + f"{len(self.dataloader_train)} train batches | "
            + f"{len(self.dataloader_valid)} valid batches"
        )


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
