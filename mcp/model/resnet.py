import torch
from torchvision.models.resnet import BasicBlock, ResNet

from mcp.model.base import Model


class ResNet18(Model):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
