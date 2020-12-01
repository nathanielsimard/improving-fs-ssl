import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from mcp.model.base import Model


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class _ResNet(Model):
    def __init__(self, resnet: ResNet, embed_size: int, expansion: int):
        super().__init__()
        assert embed_size == 512 * expansion, "Embedding size must match ResNet output."
        self.encoder = resnet
        # Overwrite FC layer of our encoder
        self.encoder.fc = Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ResNet18(_ResNet):
    def __init__(self, embed_size: int):
        block = BasicBlock
        super().__init__(
            ResNet(block, [2, 2, 2, 2], embed_size), embed_size, block.expansion
        )


class ResNet50(_ResNet):
    def __init__(self, embed_size: int):
        block = Bottleneck
        super().__init__(
            ResNet(block, [3, 4, 6, 3], embed_size), embed_size, block.expansion
        )
