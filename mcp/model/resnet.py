import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from mcp.model.base import Model


class _ResNet(Model):
    def __init__(self, resnet: ResNet, embed_size: int):
        super().__init__()
        self.encoder = resnet
        self.output = nn.Linear(embed_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.relu(x)
        return self.output(x)


class ResNet18(_ResNet):
    def __init__(self, embed_size: int):
        super().__init__(ResNet(BasicBlock, [2, 2, 2, 2], embed_size), embed_size)


class ResNet50(_ResNet):
    def __init__(self, embed_size: int):
        super().__init__(ResNet(Bottleneck, [3, 4, 6, 3], embed_size), embed_size)
